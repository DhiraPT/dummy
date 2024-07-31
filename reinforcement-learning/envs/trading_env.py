import math
from typing import Any, SupportsFloat, Tuple, List
import pandas as pd
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from custom_types.asset_type import Asset


class TradingEnv(gymnasium.Env):
    
    def __init__(self, assets: List[Asset], start_timestamp, end_timestamp, initial_balance: float = 10000.0, fee: float = 0.0025,
                 window_size: int = 10):
        super().__init__()

        self.assets = assets
        self.assets_names = [asset['name'] for asset in assets]
        self.num_assets = len(assets)
        self.timeframes = ['1min', '5min', '15min', '30min', '60min']
        self.start_timestamp = pd.to_datetime(start_timestamp, unit='s')
        self.end_timestamp = pd.to_datetime(end_timestamp, unit='s')
        self.initial_balance = initial_balance
        self.fee = fee
        self.window_size = window_size

        self.current_timestamp = self.start_timestamp
        self.balance = self.initial_balance
        self.positions = [0] * self.num_assets
        self.net_worth = self.initial_balance

        # Actions: buy, hold, sell for each asset
        # 1 - buying the stock with maximum allowed proportion (with respect to the balance)
        # -1 - selling the stock with maximum allowed proportion (with respect to the position)
        # 0 - holding the stock
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets,), dtype=np.float32)

        # Observations: OHLCV and indicators for each asset
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_assets, len(self.assets[0]['data']), self.window_size, len(self.assets[0]['data']['1min'].columns),),
            dtype=np.float32
        )

        self.episode_info = {
            'balance': [],
            'positions': [],
            'net_worth': [],
        }

    def _get_obs(self) -> ObsType:
        observation = []
        for asset in self.assets:
            asset_observation = []
            for timeframe in self.timeframes:
                past_candle_timestamps = asset['data'][timeframe].index[asset['data'][timeframe].index < self.current_timestamp][-self.window_size:]
                asset_observation.append(
                    np.array([
                        asset['data'][timeframe].loc[past_candle_timestamp].values
                        for past_candle_timestamp in past_candle_timestamps
                    ], dtype=np.float32)
                )
            observation.append(np.array(asset_observation))
        return np.array(observation)
    
    def _get_info(self) -> dict[str, Any]:
        return {
            'balance': self.balance,
            'positions': {asset_name: position for asset_name, position in zip(self.assets_names, self.positions)},
            'net_worth': self.net_worth,
        }
    
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.current_timestamp = self.start_timestamp
        self.balance = self.initial_balance
        self.positions = [0] * self.num_assets
        self.net_worth = self.initial_balance

        self.episode_info = {
            'balance': [],
            'positions': [],
            'net_worth': [],
        }

        return self._get_obs(), self._get_info()
    
    def _normalize_actions(self, actions: ActType, prices: List[int]) -> ActType:
        buy_actions = np.clip(actions, 0, 1)
        total_buy_cost = np.sum(buy_actions * prices) * (1 + self.fee)
        
        if total_buy_cost == 0:
            return actions

        if total_buy_cost > self.balance:
            scaling_factor = self.balance / total_buy_cost

            for i in range(len(actions)):
                if actions[i] > 0:
                    actions[i] *= scaling_factor

        return actions

    def step(self, actions: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Take current 1min candle's opening price as current price
        current_prices = [asset['data']['1min']['open'].loc[self.current_timestamp] for asset in self.assets]

        # Normalize buy actions to be within the balance
        actions = self._normalize_actions(actions, current_prices)

        for asset_name, action, current_price in zip(self.assets_names, actions, current_prices):
            if action > 0:
                self._buy(asset_name, action, current_price)
            elif action < 0:
                self._sell(asset_name, action, current_price)
            else:
                self._hold(asset_name, current_price)

        # Calculate current net worth
        current_net_worth = self.balance + np.sum([position * current_price for position, current_price in zip(self.positions, current_prices)])

        observation = self._get_obs()
        reward = self._calculate_reward()
        # An episode is done iff the agent has reached the target
        terminated = self.current_timestamp >= self.end_timestamp
        # An episode is done iff the agent has gone out of bounds
        truncated = self.net_worth <= 0
        info = self._get_info()

        self.net_worth = current_net_worth
        self.current_timestamp += pd.Timedelta(minutes=1)

        # Append step info to episode info
        self.episode_info['balance'].append(self.balance)
        self.episode_info['positions'].append(self.positions.copy())
        self.episode_info['net_worth'].append(self.net_worth)

        return observation, reward, terminated, truncated, info

    def _buy(self, asset_name: str, action: float, price: float):
        index = self.assets_names.index(asset_name)
        maximum_price_precision = self.assets[index]['pair_decimals']
        maximum_order_precision = self.assets[index]['lot_decimals']
        minimum_order_size = self.assets[index]['ordermin']
        minimum_cost = self.assets[index]['costmin']

        buy_cost_excluding_fee = self.balance * action
        buy_quantity = math.floor(buy_cost_excluding_fee / price * 10**maximum_order_precision) / 10**maximum_order_precision
        buy_cost_excluding_fee = buy_quantity * price
        if buy_quantity < minimum_order_size or buy_cost_excluding_fee < minimum_cost:
            return
        self.balance -= buy_cost_excluding_fee * (1 + self.fee)
        self.positions[self.assets_names.index(asset_name)] += buy_quantity

    def _sell(self, asset_name: str, action: float, price: float):
        index = self.assets_names.index(asset_name)
        maximum_price_precision = self.assets[index]['pair_decimals']
        maximum_order_precision = self.assets[index]['lot_decimals']
        minimum_order_size = self.assets[index]['ordermin']
        minimum_cost = self.assets[index]['costmin']

        sell_quantity = math.floor(self.positions[index] * -action * 10**maximum_order_precision) / 10**maximum_order_precision
        sell_cost_excluding_fee = sell_quantity * price
        if sell_quantity < minimum_order_size or sell_cost_excluding_fee < minimum_cost:
            return
        self.balance += sell_cost_excluding_fee * (1 - self.fee)
        self.positions[index] -= sell_quantity

    def _hold(self, asset_name: str, price: float):
        pass

    def _calculate_reward(self):
        reward = 0
        for asset in self.assets:
            for timeframe in self.timeframes:
                current_candle_timestamp = asset['data'][timeframe].index[asset['data'][timeframe].index <= self.current_timestamp][-1]
                reward += self.positions[self.assets_names.index(asset['name'])] * ((1 - self.fee) * asset['data'][timeframe]['close'].loc[current_candle_timestamp] - asset['data']['1min']['open'].loc[self.current_timestamp])
        return reward

    def render(self):
        dates = pd.date_range(start=self.start_timestamp, periods=len(self.episode_info['net_worth']), freq='min')

        # Create a figure with subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=('Net Worth over Time', 'Positions over Time', 'Price over Time'))

        # Add net worth trace
        fig.add_trace(go.Scatter(x=dates, y=self.episode_info['net_worth'], mode='lines', name='Net Worth'), row=1, col=1)

        # Add positions traces
        for i, asset_name in enumerate(self.assets_names):
            positions = [position[i] for position in self.episode_info['positions']]
            fig.add_trace(go.Bar(x=dates, y=positions, name=asset_name), row=2, col=1)

        # Add price traces for each asset
        for asset in self.assets:
            fig.add_trace(go.Scatter(x=dates, y=asset['data']['1min'].loc[self.start_timestamp:self.end_timestamp]['close'],
                          mode='lines', name=asset['name']), row=3, col=1)

        # Update layout
        fig.update_layout(
            title='Trading RL Visualization',
            xaxis_title='Date',
            yaxis_title='Net Worth (USD)',
            xaxis2_title='Date',
            yaxis2_title='Positions',
            xaxis3_title='Date',
            yaxis3_title='Price (USD)',
            xaxis=dict(tickformat='%b %Y'),
            barmode='stack'
        )

        fig.show()

        print(f'Net Worth: {self.net_worth}')
        print(f'Positions: {self.positions}')
        print(f'Balance: {self.balance}')

    def close(self):
        plt.close('all')
