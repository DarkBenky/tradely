# import tensoflow as tf
import numpy as np
import pandas as pd
from portfolio_env import PortfolioEnv

DEBUG = True

if __name__ == "__main__":
    env = PortfolioEnv()
    observation = env.reset()
    done = False

    while not done:
        action = env.sample()
        observation, reward, done, info = env.step(action)
        if DEBUG:
            print("Obs shape:", observation.shape)
            actions = { key : value for key, value in zip(env.asset_names, action)}
        print(f"Step: {env.step_count}, Reward: {reward}, Done: {done}, Info: {info}")