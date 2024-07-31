from typing import List

from algorithms import ppo_algorithm
from envs.trading_env import TradingEnv
from custom_types.asset_type import Asset
from utils.asset_utils import preprocess_data


if __name__ == "__main__":
    asset_names = ['SOL']
    assets: List[Asset] = []
    for asset_name in asset_names:
        data = {}
        for i in [1, 5, 15, 30, 60]:
            data[f'{i}min'] = preprocess_data(f'./data/{asset_name}USD_{i}.csv', i)
        assets.append({'name': asset_name, 'data': data, 'pair_decimals': 2, 'lot_decimals': 8, 'ordermin': 0.02, 'costmin': 0.5})

    trading_env_config = {'assets': assets, 'start_timestamp': 1648166340, 'end_timestamp': 1648771140, 'initial_balance': 10000.0, 'fee': 0.0025, 'window_size': 10}
    ppo_algorithm.train_model(trading_env_config, num_envs=8, total_timesteps=1000000)
    print('Testing the model')
    ppo_algorithm.eval(trading_env_config, num_envs=1)
