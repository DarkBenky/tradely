import numpy as np
import pandas as pd
import os
import random
from collections import deque
from enum import Enum

DEBUG = False

def load_and_align_data(data_dir='data', max_records=None):
    data_files = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('_combined_data.csv'):
            symbol = filename.replace('_combined_data.csv', '')
            if symbol == 'MATICUSDT':
                continue
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            data_files[symbol] = df
    
    print(f"Loaded {len(data_files)} data files")
    
    common_timestamps = None
    for symbol, df in data_files.items():
        if common_timestamps is None:
            common_timestamps = set(df.index)
        else:
            common_timestamps = common_timestamps.intersection(set(df.index))
    
    common_timestamps = sorted(list(common_timestamps))
    
    if max_records and len(common_timestamps) > max_records:
        common_timestamps = common_timestamps[:max_records]
        print(f"Limited to {max_records} records")
    
    print(f"Found {len(common_timestamps)} common timestamps")
    
    aligned_data = {}
    for symbol, df in data_files.items():
        aligned_df = df.loc[common_timestamps].reset_index(drop=True)
        aligned_data[symbol] = aligned_df
    
    return aligned_data

class ModelType(Enum):
    TRANSFORMER_SHAPE = "transformer_shape"
    FLAT_SHAPE = "flatte"
    THREE_D_SHAPE = "3d"

class PortfolioEnv():
    def __init__(self, data_dir='data', max_records=None, rebalancing_interval=12, previous_env_reward=12, previous_reward_to_current_ration=0.25, obs_shape=ModelType.TRANSFORMER_SHAPE):
        self.initial_account_balance = 10000.0
        self.rebalance_interval = rebalancing_interval
        self.reset_value = self.initial_account_balance // 2
        self.df = load_and_align_data(data_dir, max_records)
        self.asset_names = list(self.df.keys()) + ['cash']
        self.prev_reward = deque(maxlen=previous_env_reward)
        self.previous_reward_to_current_ration = previous_reward_to_current_ration
        self.obs_shape = obs_shape
        
        total_length = len(self.df['BTCUSDT'])
        self.lookback_window_size = 24
        
        min_step = max(1000, self.lookback_window_size + 288)
        max_step = total_length - 1000
        
        if min_step >= max_step:
            raise ValueError(f"Dataset too small: {total_length} records")
        
        self.step_count = random.randint(min_step, max_step)
        self.high_timeframes = ['1h', '4h']
        self.high_timeframes_count = [24, 12]
        
        self.portfolio = {'cash': self.initial_account_balance}
        self.portfolio_value = self.initial_account_balance
        for symbol in self.df.keys():
            self.portfolio[symbol] = 0.0
        
        self.passive_portfolio = {}
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            self.passive_portfolio[symbol] = self.initial_account_balance / len(self.df) / price
        self.passive_portfolio_value = self.initial_account_balance
       
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
    
    def sample(self):
        action = []
        for _ in self.asset_names:
            action.append(random.uniform(0, 1))
        total = sum(action)
        action = [a / total for a in action]
        return action
        

    def get_observation(self, type='transformer'):
        total_length = len(self.df['BTCUSDT'])
        if self.step_count < 0 or self.step_count >= total_length - 2:
            self.step_count = max(1000, self.lookback_window_size + 288)
        
        obs = [[] for _ in range(len(self.df))]
        
        aggregation_factors = {
            '1h': 12,
            '4h': 48,
        }
        
        exclude_cols = ['timestamp', 'close_time', 'ignore']
        sample_symbol = list(self.df.keys())[0]
        all_available_cols = [col for col in self.df[sample_symbol].columns 
                             if col not in exclude_cols]
        
        num_features = len(all_available_cols)
        
        for i, symbol in enumerate(self.df.keys()):
            symbol_data = self.df[symbol]
            start_idx = max(0, self.step_count - self.lookback_window_size)
            lookback_data = symbol_data.iloc[start_idx:self.step_count]
            
            available_cols = [col for col in all_available_cols if col in lookback_data.columns]
            candle_data = lookback_data[available_cols].values.astype(float)
            candle_data = np.nan_to_num(candle_data, nan=0.0, posinf=1.0, neginf=0.0)
            obs[i].append(candle_data)

        for tf, candles_count in zip(self.high_timeframes, self.high_timeframes_count):
            agg_factor = aggregation_factors[tf]
            _obs = [[] for _ in range(len(self.df))]
            
            for i, symbol in enumerate(self.df.keys()):
                symbol_data = self.df[symbol]
                base_candles_needed = candles_count * agg_factor
                start_idx = max(0, self.step_count - base_candles_needed)
                lookback_data = symbol_data.iloc[start_idx:self.step_count]
                
                available_cols = [col for col in all_available_cols if col in lookback_data.columns]
                
                aggregated = []
                for j in range(0, len(lookback_data), agg_factor):
                    chunk = lookback_data.iloc[j:j + agg_factor]
                    if len(chunk) > 0:
                        agg_candle = {}
                        for col in available_cols:
                            if col == 'open':
                                agg_candle[col] = chunk[col].iloc[0]
                            elif col == 'high':
                                agg_candle[col] = chunk[col].max()
                            elif col == 'low':
                                agg_candle[col] = chunk[col].min()
                            elif col == 'close':
                                agg_candle[col] = chunk[col].iloc[-1]
                            elif col in ['volume', 'quote_asset_volume', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'number_of_trades']:
                                agg_candle[col] = chunk[col].sum()
                            else:
                                agg_candle[col] = chunk[col].iloc[-1]
                        
                        agg_values = [agg_candle.get(col, 0.0) for col in available_cols]
                        aggregated.append(agg_values)
                
                if len(aggregated) > candles_count:
                    aggregated = aggregated[-candles_count:]
                
                if len(aggregated) > 0:
                    agg_array = np.array(aggregated, dtype=float)
                    agg_array = np.nan_to_num(agg_array, nan=0.0, posinf=1.0, neginf=0.0)
                    _obs[i] = agg_array
                else:
                    _obs[i] = np.array([])
            
            for i in range(len(self.df)):
                obs[i].append(_obs[i])
        
        portfolio_state = []
        total_portfolio_value = self._portfolio_portfolio_value()
        
        for symbol in self.df.keys():
            holdings = self.portfolio.get(symbol, 0.0)
            price = self.df[symbol]['close'].iloc[self.step_count]
            holdings_value = holdings * price
            holdings_pct = holdings_value / total_portfolio_value if total_portfolio_value > 0 else 0
            portfolio_state.append(holdings_pct)
        
        cash_pct = self.portfolio['cash'] / total_portfolio_value if total_portfolio_value > 0 else 0
        portfolio_state.append(cash_pct)
        
        portfolio_value_ratio = total_portfolio_value / self.initial_account_balance
        portfolio_state.append(portfolio_value_ratio)
        
        benchmark_value = self._update_benchmark_value()
        benchmark_ratio = benchmark_value / self.initial_account_balance
        portfolio_state.append(benchmark_ratio)
        
        portfolio_state_array = np.array(portfolio_state, dtype=float)
        
        if self.obs_shape == ModelType.FLAT_SHAPE:
            for i in range(len(self.df)):
                obs[i].append(portfolio_state_array)
            
            flattened_obs = []
            for i in range(len(self.df)):
                for obs_part in obs[i]:
                    if obs_part.size > 0:
                        flattened_obs.append(obs_part.flatten())
            
            if flattened_obs:
                final_obs = np.concatenate(flattened_obs)
            else:
                final_obs = np.array([])
            
            final_obs = np.nan_to_num(final_obs, nan=0.0, posinf=1.0, neginf=0.0)
            return final_obs
        
        elif self.obs_shape == ModelType.THREE_D_SHAPE:
            total_timesteps = self.lookback_window_size + sum(self.high_timeframes_count)
            num_assets = len(self.df)
            
            result = np.zeros((total_timesteps, num_features, num_assets), dtype=float)
            
            timestep_offset = 0
            for i, symbol in enumerate(self.df.keys()):
                base_candles = obs[i][0]
                if base_candles.size > 0:
                    for t in range(min(base_candles.shape[0], self.lookback_window_size)):
                        result[t, :, i] = base_candles[t, :]
            
            timestep_offset = self.lookback_window_size
            
            for tf_idx, candles_count in enumerate(self.high_timeframes_count):
                for i, symbol in enumerate(self.df.keys()):
                    tf_candles = obs[i][tf_idx + 1]
                    if tf_candles.size > 0:
                        for t in range(min(tf_candles.shape[0], candles_count)):
                            result[timestep_offset + t, :, i] = tf_candles[t, :]
                timestep_offset += candles_count
            
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
            return result
        
        elif self.obs_shape == ModelType.TRANSFORMER_SHAPE:
            total_timesteps = self.lookback_window_size + sum(self.high_timeframes_count)
            num_assets = len(self.df)
            features_per_timestep = num_features * num_assets + len(portfolio_state_array)
            
            result = np.zeros((total_timesteps, features_per_timestep), dtype=float)
            
            for t in range(self.lookback_window_size):
                feature_offset = 0
                for i, symbol in enumerate(self.df.keys()):
                    base_candles = obs[i][0]
                    if base_candles.size > 0 and t < base_candles.shape[0]:
                        result[t, feature_offset:feature_offset + num_features] = base_candles[t, :]
                    feature_offset += num_features
                
                result[t, feature_offset:] = portfolio_state_array
            
            timestep_offset = self.lookback_window_size
            for tf_idx, candles_count in enumerate(self.high_timeframes_count):
                for t in range(candles_count):
                    feature_offset = 0
                    for i, symbol in enumerate(self.df.keys()):
                        tf_candles = obs[i][tf_idx + 1]
                        if tf_candles.size > 0 and t < tf_candles.shape[0]:
                            result[timestep_offset + t, feature_offset:feature_offset + num_features] = tf_candles[t, :]
                        feature_offset += num_features
                    
                    result[timestep_offset + t, feature_offset:] = portfolio_state_array
                timestep_offset += candles_count
            
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
            return result
        
        else:
            raise ValueError(f"Unknown obs_shape: {self.obs_shape}")
        
    def _rebalance_portfolio(self, action):
        current_value = self._portfolio_portfolio_value()
        
        num_cryptos = len(self.df)
        action = np.array(action).flatten()
        
        crypto_actions = action[:num_cryptos]
        cash_action = action[num_cryptos] if len(action) > num_cryptos else 0.0
        
        crypto_actions = [float(np.clip(float(a), 0.0, 1.0)) for a in crypto_actions]
        cash_action = float(np.clip(float(cash_action), 0.0, 1.0))
        
        total_allocation = sum(crypto_actions) + cash_action
        if total_allocation > 0:
            crypto_actions = [a / total_allocation for a in crypto_actions]
            cash_action = cash_action / total_allocation
        else:
            crypto_actions = [1.0 / len(self.asset_names)] * num_cryptos
            cash_action = 1.0 / len(self.asset_names)
        
        target_values = {}
        
        for i, symbol in enumerate(self.df.keys()):
            target_values[symbol] = current_value * crypto_actions[i]
        
        target_cash = current_value * cash_action
        
        total_fees = 0.0
        
        for i, symbol in enumerate(self.df.keys()):
            current_price = self.df[symbol]['close'].iloc[self.step_count]
            current_holdings = self.portfolio[symbol]
            current_value_in_symbol = current_holdings * current_price
            
            target_value_in_symbol = target_values[symbol]
            value_diff = target_value_in_symbol - current_value_in_symbol
            
            if abs(value_diff) > 1.0:
                if value_diff < 0:
                    sell_value = abs(value_diff)
                    sell_quantity = min(sell_value / current_price, current_holdings)
                    actual_sell_value = sell_quantity * current_price
                    fee = actual_sell_value * self.fee_rate
                    net_proceeds = actual_sell_value - fee
                    
                    self.portfolio['cash'] += net_proceeds
                    self.portfolio[symbol] -= sell_quantity
                    total_fees += fee
        
        for i, symbol in enumerate(self.df.keys()):
            current_price = self.df[symbol]['close'].iloc[self.step_count]
            current_holdings = self.portfolio[symbol]
            current_value_in_symbol = current_holdings * current_price
            
            target_value_in_symbol = target_values[symbol]
            value_diff = target_value_in_symbol - current_value_in_symbol
            
            if value_diff > 1.0 and self.portfolio['cash'] > 1.0:
                buy_value = min(value_diff, self.portfolio['cash'] / (1 + self.fee_rate))
                fee = buy_value * self.fee_rate
                total_cost = buy_value + fee
                
                if self.portfolio['cash'] >= total_cost:
                    buy_quantity = buy_value / current_price
                    self.portfolio['cash'] -= total_cost
                    self.portfolio[symbol] += buy_quantity
                    total_fees += fee
        
        if self.portfolio['cash'] < 0:
            self.portfolio['cash'] = 0.0
        
        return total_fees
    
    def step_fast(self, action):
        prev_portfolio_value = self._portfolio_portfolio_value()
        self._rebalance_portfolio(action)
        
        for symbol in self.df.keys():
            if self.portfolio[symbol] < 0:
                self.portfolio[symbol] = 0.0
        
        if self.portfolio['cash'] < 0:
            self.portfolio['cash'] = 0.0
        
        self.step_count += self.rebalance_interval
        
        current_value = self._portfolio_portfolio_value()
        reward = current_value - prev_portfolio_value
        self.prev_reward.append(reward)
        
        reward = reward * (1 - self.previous_reward_to_current_ration) + (sum(self.prev_reward) / len(self.prev_reward)) * self.previous_reward_to_current_ration if self.prev_reward else reward
        
        return reward
    
    def step(self, action):
        if self.step_count >= len(self.df['BTCUSDT']) - 1000:
            return self.get_observation(), 0, True, {'reason': 'end_of_data'}
        
        prev_portfolio_value = self._portfolio_portfolio_value()
        
        fees_paid = self._rebalance_portfolio(action)
        
        for symbol in self.df.keys():
            if self.portfolio[symbol] < 0:
                self.portfolio[symbol] = 0.0
        
        if self.portfolio['cash'] < 0:
            self.portfolio['cash'] = 0.0
        
        self.step_count += self.rebalance_interval
        
        current_value = self._portfolio_portfolio_value()
        reward = current_value - prev_portfolio_value

        self.prev_reward.append(reward)
        
        obs = self.get_observation()
        
        done = False
        info = {}
        
        if current_value <= self.reset_value:
            done = True
            info['reason'] = 'reset_threshold_reached'
        
        if self.step_count >= len(self.df['BTCUSDT']) - 1000:
            done = True
            info['reason'] = 'end_of_data'
        
        info['portfolio_value'] = current_value
        info['benchmark_value'] = self.passive_portfolio_value
        info['fees_paid'] = fees_paid
        info['step'] = self.step_count
        info['cash_allocation'] = self.portfolio['cash'] / current_value if current_value > 0 else 1.0
        
        reward = reward * (1 - self.previous_reward_to_current_ration) + (sum(self.prev_reward) / len(self.prev_reward)) * self.previous_reward_to_current_ration if self.prev_reward else reward

        return obs, reward, done, info
    
    def render(self, mode='human'):
        portfolio_value = self._portfolio_portfolio_value()
        benchmark_value = self._update_benchmark_value()
        
        print(f"\nStep: {self.step_count}")
        print(f"Portfolio: ${portfolio_value:,.2f} | Benchmark: ${benchmark_value:,.2f}")
        print(f"Return: {(portfolio_value / self.initial_account_balance - 1) * 100:+.2f}%")
        print(f"Cash: ${self.portfolio['cash']:,.2f} ({self.portfolio['cash']/portfolio_value*100:.1f}%)\n")

    def reset(self):
        total_length = len(self.df['BTCUSDT'])
        min_step = max(1000, self.lookback_window_size + 288)
        max_step = total_length - 1000
        
        if min_step >= max_step:
            raise ValueError(f"Dataset too small: {total_length} records")
        
        self.step_count = random.randint(min_step, max_step)

        self.prev_reward.clear()
        
        self.portfolio = {'cash': 2000.0}
        self.portfolio_value = self.initial_account_balance
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            if np.isnan(price) or price <= 0:
                price = 1.0
            self.portfolio[symbol] = (self.initial_account_balance - 2000) / len(self.df) / price
        
        self.passive_portfolio = {}
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            if np.isnan(price) or price <= 0:
                price = 1.0
            self.passive_portfolio[symbol] = self.initial_account_balance / len(self.df) / price
        self.passive_portfolio_value = self.initial_account_balance
        
        obs = self.get_observation()
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs