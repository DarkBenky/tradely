import numpy as np
import pandas as pd

class PortfolioEnv():
    def __init__(self,dfs: dict):
        self.candle_interval = '5m'
        self.high_timeframes = ['15m', '1h', '4h', '1d']
        self.assets = []
        self.lookback_window_size = 48
        self.cash_inflation_rate = 0.004
        self.initial_account_balance = 10000.0
        self.fee_rate = 0.001
        pass
       
    def _initialize_state(self):
        pass
        
    def _initialize_benchmark_portfolio(self):
        pass
    
    def _update_benchmark_value(self):
        pass
    
    def _calculate_benchmark_outperformance(self) -> float:
        pass
    
    def _get_current_price(self, symbol: str) -> float:
        pass
    
    def _calculate_future_profit(self) -> float:
        pass

        
    def normalize_features(self) -> np.ndarray:
        pass

    def get_observation(self) -> np.ndarray:
        pass
    
    def _calculate_portfolio_value(self) -> float:
        pass
    
    def _rebalance_portfolio(self):
        pass
    
    def _calculate_reward(self) -> float:
        pass
    
    def step(self):
        #    
       pass
    
    def render(self):
        pass

    def reset(self):
        pass