# import tensoflow as tf
import numpy as np
import pandas as pd
from portfolio_env import PortfolioEnv

if __name__ == "__main__":
    env = PortfolioEnv()
    observation = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(f"Step: {env.step_count}, Reward: {reward}, Done: {done}, Info: {info}")