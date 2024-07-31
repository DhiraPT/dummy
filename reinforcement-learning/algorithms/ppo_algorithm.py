from typing import List
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import pandas as pd
import torch

from envs.trading_env import TradingEnv
from custom_types.asset_type import Asset


def make_env(env_config: dict[str, any], rank: int, seed: int = 42):
    def _init():
        env = TradingEnv(env_config['assets'],
                         pd.to_datetime(env_config['start_timestamp'], unit='s'),
                         pd.to_datetime(env_config['end_timestamp'], unit='s'),
                         initial_balance=env_config.get('initial_balance', 10000.0),
                         fee=env_config.get('fee', 0.0025),
                         window_size=env_config.get('window_size', 10))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train_model(env_config: dict[str, any], num_envs: int = 1, model: PPO = None, total_timesteps: int = 10000) -> None:
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_config, i) for i in range(num_envs)])

    if model:
        model.set_env(vec_env)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    else:
        # Define the model
        model = PPO('MlpPolicy', vec_env, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu')
        model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("ppo_crypto_trading_1648166340_1648771140")


def eval(env_config: dict[str, any], num_envs: int = 1) -> None:
    # Load the model
    model = PPO.load("ppo_crypto_trading_1648166340_1648771140", device='cuda' if torch.cuda.is_available() else 'cpu')

    start_timestamp = pd.to_datetime(env_config['start_timestamp'], unit='s')
    end_timestamp = pd.to_datetime(env_config['end_timestamp'], unit='s')

    env = TradingEnv(env_config['assets'],
                     start_timestamp=start_timestamp,
                     end_timestamp=end_timestamp,
                     initial_balance=env_config.get('initial_balance', 10000.0),
                     fee=env_config.get('fee', 0.0025),
                     window_size=env_config.get('window_size', 10))

    # Evaluate the model
    observation, _ = env.reset(seed=42)
    for _ in pd.date_range(start=start_timestamp, end=end_timestamp, freq='1min'):
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if truncated:
            break

    env.render()
    env.close()
