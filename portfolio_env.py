import numpy as np
import pandas as pd
import os
import random

def load_and_align_data(data_dir='data'):
    data_files = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('_combined_data.csv'):
            symbol = filename.replace('_combined_data.csv', '')
            if symbol == 'MATICUSDT':  # Skip empty dataframes
                continue
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            data_files[symbol] = df
    
    print(f"Loaded {len(data_files)} data files")
    
    # Find common timestamps across all symbols
    common_timestamps = None
    for symbol, df in data_files.items():
        if common_timestamps is None:
            common_timestamps = set(df.index)
        else:
            common_timestamps = common_timestamps.intersection(set(df.index))
    
    common_timestamps = sorted(list(common_timestamps))
    print(f"Found {len(common_timestamps)} common timestamps")
    
    # Align each dataframe to common timestamps and remove timestamp index
    aligned_data = {}
    for symbol, df in data_files.items():
        aligned_df = df.loc[common_timestamps].reset_index(drop=True)
        aligned_data[symbol] = aligned_df
    
    return aligned_data

class PortfolioEnv():
    def __init__(self, data_dir='data'):
        self.df = load_and_align_data(data_dir)
        self.step_count = random.randint(1000, len(self.df['BTCUSDT']) - 1000)
        self.candle_interval = '5m'
        self.high_timeframes = ['15m', '1h', '4h', '1d']

        self.lookback_window_size = 48
        self.lookforward_window_size = 30
        
        self.initial_account_balance = 10000.0
        self.portfolio = {'cash': self.initial_account_balance}
        self.portfolio_value = self.initial_account_balance
        for symbol in self.df.keys():
            self.portfolio[symbol] = 0.0
        self.passive_portfolio = {}
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            self.passive_portfolio[symbol] =  self.initial_account_balance / len(self.df) / price
        self.passive_portfolio_value = self.initial_account_balance
       
        self.cash_inflation_rate = 0.004
        self.fee_rate = 0.001
       
    def _portfolio_portfolio_value(self):
        total_value = self.portfolio['cash']
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            total_value += self.portfolio[symbol] * price
        self.portfolio_value = total_value
        return total_value

    def _update_benchmark_value(self):
        total_value = 0.0
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            total_value += self.passive_portfolio[symbol] * price
        self.passive_portfolio_value = total_value
        return total_value
    
    def _calculate_benchmark_outperformance(self) -> float:
        pass
    
    def _get_current_price(self, symbol: str) -> float:
        pass
    
    def _calculate_future_profit(self) -> float:
        future_profits = {}
        for symbol, data in self.df.items():
            current_price = data['close'].iloc[self.step_count]
            values, weights = [], []
            for i in range(1, self.lookforward_window_size + 1):
                future_price = data['close'].iloc[self.step_count + i]
                ret = (future_price - current_price) / current_price
                values.append(ret)
                # Farther future gets more weight
                weights.append(i / self.lookforward_window_size)
            weighted_return = np.dot(values, weights) / sum(weights)
            future_profits[symbol] = weighted_return
        return future_profits

        
        
    def normalize_features(self) -> np.ndarray:
        pass

    def get_observation(self) -> np.ndarray:
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