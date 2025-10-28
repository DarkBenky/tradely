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
        self.asset_names = list(self.df.keys())  # Add asset_names attribute
        self.step_count = random.randint(2000, len(self.df['BTCUSDT']) - 1000)
        self.candle_interval = '5m'
        self.high_timeframes = ['15m', '1h', '4h', '1d']
        self.high_timeframes_count = [64, 48, 24, 3]

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
        benchmark_value = self._update_benchmark_value()
        outperformance = self.portfolio_value / benchmark_value if benchmark_value > 0 else 1.0
        return outperformance

    def _get_current_priced(self, symbol) -> float:
        price = self.df[symbol]['close'].iloc[self.step_count]
        return price
    
    def sample(self):
        action = []
        for symbol in self.df.keys():
            action.append(random.uniform(0, 1))
        # Normalize to sum to <= 1
        total = sum(action)
        if total > 1.0:
            action = [a / total for a in action]
        return action
        
    
    def get_observation(self):
        obs = [[] for _ in range(len(self.df))]
        
        # Define aggregation factors (how many 5m candles make up each timeframe)
        aggregation_factors = {
            '15m': 3,    # 3 * 5m = 15m
            '1h': 12,    # 12 * 5m = 1h
            '4h': 48,    # 48 * 5m = 4h
            '1d': 288    # 288 * 5m = 1d
        }
        
        # Columns to exclude from observation (metadata, not features)
        exclude_cols = ['timestamp', 'close_time', 'ignore']
        
        def normalize_data(data, col_names):
            """
            Advanced normalization with different strategies per column type:
            - Price-like (OHLC, MAs, EMAs, BBs, VWAP): Use percentage change from close
            - Volume: Log scale + min-max
            - RSI: Already 0-100, just divide by 100
            - MACD: Standardize (z-score)
            - ATR: Normalize by close price
            - Profile data: Min-max normalization
            """
            normalized = np.zeros_like(data, dtype=float)
            
            for col_idx, col_name in enumerate(col_names):
                col_data = data[:, col_idx].copy()
                
                # Handle NaN values
                if np.all(np.isnan(col_data)):
                    normalized[:, col_idx] = 0
                    continue
                
                # Get current close price for price-relative normalization
                close_idx = col_names.index('close') if 'close' in col_names else None
                current_close = data[:, close_idx] if close_idx is not None else None
                
                # Apply different normalization based on column type
                if col_name in ['open', 'high', 'low', 'close', 'ma_5', 'ma_10', 'ma_20', 
                               'ema_5', 'ema_10', 'ema_20', 'bb_upper', 'bb_middle', 'bb_lower', 
                               'vwap', 'hvn_level_1', 'hvn_level_2', 'hvn_level_3', 'hvn_level_4', 
                               'hvn_level_5', 'lvn_level_1', 'lvn_level_2', 'lvn_level_3']:
                    # Price-like: normalize as percentage of close price
                    if current_close is not None and col_name != 'close':
                        normalized[:, col_idx] = np.where(current_close != 0, 
                                                         col_data / current_close, 
                                                         1.0)
                    else:
                        # For close itself, use returns
                        valid_mask = ~np.isnan(col_data)
                        if np.any(valid_mask):
                            first_valid = col_data[valid_mask][0]
                            normalized[:, col_idx] = np.where(first_valid != 0,
                                                             col_data / first_valid,
                                                             1.0)
                
                elif col_name in ['volume', 'quote_asset_volume', 'taker_buy_base_asset_volume',
                                 'taker_buy_quote_asset_volume', 'number_of_trades']:
                    # Volume: log scale + min-max normalization
                    col_data_pos = np.where(col_data > 0, col_data, 1e-10)
                    log_data = np.log1p(col_data_pos)
                    valid_data = log_data[~np.isnan(log_data)]
                    if len(valid_data) > 0:
                        data_min, data_max = np.nanmin(log_data), np.nanmax(log_data)
                        if data_max > data_min:
                            normalized[:, col_idx] = (log_data - data_min) / (data_max - data_min)
                        else:
                            normalized[:, col_idx] = 0.5
                
                elif col_name == 'rsi':
                    # RSI is already 0-100, just normalize to 0-1
                    normalized[:, col_idx] = np.clip(col_data / 100.0, 0, 1)
                
                elif col_name in ['macd_line', 'macd_signal', 'macd_histogram']:
                    # MACD: standardize with z-score
                    valid_data = col_data[~np.isnan(col_data)]
                    if len(valid_data) > 0:
                        mean, std = np.nanmean(col_data), np.nanstd(col_data)
                        if std > 0:
                            z_score = (col_data - mean) / std
                            # Clip to [-3, 3] and scale to [0, 1]
                            normalized[:, col_idx] = (np.clip(z_score, -3, 3) + 3) / 6
                        else:
                            normalized[:, col_idx] = 0.5
                
                elif col_name == 'atr':
                    # ATR: normalize by close price (typical ATR as % of price)
                    if current_close is not None:
                        normalized[:, col_idx] = np.where(current_close != 0,
                                                         col_data / current_close,
                                                         0)
                
                elif 'profile' in col_name or 'strength' in col_name or 'htn_' in col_name or 'ltn_' in col_name:
                    # Profile data and strengths: min-max normalization
                    valid_data = col_data[~np.isnan(col_data)]
                    if len(valid_data) > 0:
                        data_min, data_max = np.nanmin(col_data), np.nanmax(col_data)
                        if data_max > data_min:
                            normalized[:, col_idx] = (col_data - data_min) / (data_max - data_min)
                        else:
                            normalized[:, col_idx] = 0.5
                
                else:
                    # Default: min-max normalization
                    valid_data = col_data[~np.isnan(col_data)]
                    if len(valid_data) > 0:
                        data_min, data_max = np.nanmin(col_data), np.nanmax(col_data)
                        if data_max > data_min:
                            normalized[:, col_idx] = (col_data - data_min) / (data_max - data_min)
                        else:
                            normalized[:, col_idx] = 0.5
            
            # Replace any remaining NaN with 0
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
            return normalized

        # Get all available columns (exclude metadata)
        sample_symbol = list(self.df.keys())[0]
        all_available_cols = [col for col in self.df[sample_symbol].columns 
                             if col not in exclude_cols]
        
        # Collect base candle data (5m lookback window)
        for i, symbol in enumerate(self.df.keys()):
            symbol_data = self.df[symbol]
            start_idx = max(0, self.step_count - self.lookback_window_size)
            lookback_data = symbol_data.iloc[start_idx:self.step_count]
            
            # Extract all available features
            available_cols = [col for col in all_available_cols if col in lookback_data.columns]
            candle_data = lookback_data[available_cols].values.astype(float)
            
            # Normalize the data
            normalized_data = normalize_data(candle_data, available_cols)
            obs[i].append(normalized_data)

        # Aggregate and add higher timeframe candles
        for tf, candles_count in zip(self.high_timeframes, self.high_timeframes_count):
            agg_factor = aggregation_factors[tf]
            _obs = [[] for _ in range(len(self.df))]
            
            for i, symbol in enumerate(self.df.keys()):
                symbol_data = self.df[symbol]
                
                # Calculate the number of base candles needed
                base_candles_needed = candles_count * agg_factor
                start_idx = max(0, self.step_count - base_candles_needed)
                lookback_data = symbol_data.iloc[start_idx:self.step_count]
                
                # Get column list
                available_cols = [col for col in all_available_cols if col in lookback_data.columns]
                
                # Aggregate candles
                aggregated = []
                for j in range(0, len(lookback_data), agg_factor):
                    chunk = lookback_data.iloc[j:j + agg_factor]
                    if len(chunk) > 0:
                        agg_candle = {}
                        
                        for col in available_cols:
                            if col in ['open']:
                                agg_candle[col] = chunk[col].iloc[0]
                            elif col in ['high']:
                                agg_candle[col] = chunk[col].max()
                            elif col in ['low']:
                                agg_candle[col] = chunk[col].min()
                            elif col in ['close']:
                                agg_candle[col] = chunk[col].iloc[-1]
                            elif col in ['volume', 'quote_asset_volume', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'number_of_trades']:
                                # Sum volume-like metrics
                                agg_candle[col] = chunk[col].sum()
                            else:
                                # For indicators and profile data, use last value
                                agg_candle[col] = chunk[col].iloc[-1]
                        
                        agg_values = [agg_candle.get(col, np.nan) for col in available_cols]
                        aggregated.append(agg_values)
                
                # Keep only the most recent candles_count candles
                if len(aggregated) > candles_count:
                    aggregated = aggregated[-candles_count:]
                
                # Normalize aggregated data
                if len(aggregated) > 0:
                    agg_array = np.array(aggregated, dtype=float)
                    normalized_agg = normalize_data(agg_array, available_cols)
                    _obs[i] = normalized_agg
                else:
                    _obs[i] = np.array([])
            
            # Add to observation
            for i in range(len(self.df)):
                obs[i].append(_obs[i])
        
        # Add current portfolio holdings to observation (normalized)
        portfolio_state = []
        total_portfolio_value = self._portfolio_portfolio_value()
        
        for symbol in self.df.keys():
            holdings = self.portfolio.get(symbol, 0.0)
            price = self.df[symbol]['close'].iloc[self.step_count]
            # Normalize holdings as percentage of total portfolio
            holdings_value = holdings * price
            holdings_pct = holdings_value / total_portfolio_value if total_portfolio_value > 0 else 0
            portfolio_state.append(holdings_pct)
        
        # Add cash as percentage of total portfolio
        cash_pct = self.portfolio['cash'] / total_portfolio_value if total_portfolio_value > 0 else 0
        portfolio_state.append(cash_pct)
        
        # Add portfolio value change rate (relative to initial balance)
        portfolio_value_ratio = total_portfolio_value / self.initial_account_balance
        portfolio_state.append(portfolio_value_ratio)
        
        # Add benchmark comparison
        benchmark_value = self._update_benchmark_value()
        benchmark_ratio = benchmark_value / self.initial_account_balance
        portfolio_state.append(benchmark_ratio)
        
        # Add outperformance ratio
        outperformance = total_portfolio_value / benchmark_value if benchmark_value > 0 else 1.0
        portfolio_state.append(outperformance)
        
        for i in range(len(self.df)):
            obs[i].append(np.array(portfolio_state))
        
        # Flatten and concatenate all observations into a single array
        flattened_obs = []
        for i in range(len(self.df)):
            for obs_part in obs[i]:
                if obs_part.size > 0:  # Only add non-empty arrays
                    flattened_obs.append(obs_part.flatten())
        
        # Concatenate all parts into one flat array
        if flattened_obs:
            final_obs = np.concatenate(flattened_obs)
        else:
            final_obs = np.array([])
        
        return final_obs
    
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

    
    def _rebalance_portfolio(self, action):
        """
        Rebalance portfolio based on action.
        Action is a list with target percentages for each symbol in order of self.asset_names.
        Example: [0.3, 0.2, 0.1, ...] for ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', ...]
        Remaining percentage goes to cash.
        """
        # Calculate current portfolio value
        current_value = self._portfolio_portfolio_value()
        
        # Calculate target holdings value for each symbol
        target_values = {}
        total_target_pct = 0
        
        for i, symbol in enumerate(self.df.keys()):
            target_pct = action[i] if i < len(action) else 0.0
            target_pct = np.clip(target_pct, 0.0, 1.0)  # Ensure 0-1 range
            total_target_pct += target_pct
            target_values[symbol] = current_value * target_pct
        
        # Normalize if total exceeds 100%
        if total_target_pct > 1.0:
            for symbol in target_values.keys():
                target_values[symbol] /= total_target_pct
            total_target_pct = 1.0
        
        # Calculate trades needed
        total_fees = 0.0
        for symbol in self.df.keys():
            current_price = self._get_current_priced(symbol)
            current_holdings = self.portfolio[symbol]
            current_value_in_symbol = current_holdings * current_price
            
            target_value_in_symbol = target_values[symbol]
            value_diff = target_value_in_symbol - current_value_in_symbol
            
            if abs(value_diff) > 0.01:  # Only trade if difference is significant
                trade_value = abs(value_diff)
                fee = trade_value * self.fee_rate
                total_fees += fee
                
                # Update holdings
                if value_diff > 0:  # Buy
                    # Deduct from cash including fee
                    self.portfolio['cash'] -= (trade_value + fee)
                    self.portfolio[symbol] += trade_value / current_price
                else:  # Sell
                    # Add to cash minus fee
                    self.portfolio['cash'] += (trade_value - fee)
                    self.portfolio[symbol] -= trade_value / current_price
        
        # Apply cash inflation (opportunity cost)
        self.portfolio['cash'] *= (1 - self.cash_inflation_rate)
        
        return total_fees
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on:
        1. Weighted future returns (farther future = more weight)
        2. Portfolio alignment with future profit potential
        3. Concentration penalty for risk management
        """
        # Get future profit predictions (weighted toward farther future)
        future_profits = self._calculate_future_profit()
        
        # Update current portfolio value
        current_value = self._portfolio_portfolio_value()
        
        # Calculate portfolio-weighted expected return
        portfolio_weighted_return = 0.0
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            holdings_value = self.portfolio[symbol] * price
            weight = holdings_value / current_value if current_value > 0 else 0
            
            # Weight by portfolio allocation and future profit
            portfolio_weighted_return += weight * future_profits[symbol]
        
        # Base reward: portfolio's alignment with future returns
        reward = portfolio_weighted_return * 1000  # Scale up for meaningful rewards
        
        # Calculate benchmark's weighted future return for comparison
        benchmark_weighted_return = 0.0
        benchmark_value = self._update_benchmark_value()
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            benchmark_holdings_value = self.passive_portfolio[symbol] * price
            benchmark_weight = benchmark_holdings_value / benchmark_value if benchmark_value > 0 else 0
            benchmark_weighted_return += benchmark_weight * future_profits[symbol]
        
        # Bonus for better positioning than benchmark
        outperformance = portfolio_weighted_return - benchmark_weighted_return
        reward += outperformance * 500
        
        # Calculate portfolio concentration (Herfindahl index) for risk penalty
        concentration = 0
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            holdings_value = self.portfolio[symbol] * price
            weight = holdings_value / current_value if current_value > 0 else 0
            concentration += weight ** 2
        
        # Penalize over-concentration (>0.5 Herfindahl index)
        if concentration > 0.5:
            reward -= (concentration - 0.5) * 100
        
        # Penalty for holding too much cash when there are good opportunities
        cash_weight = self.portfolio['cash'] / current_value if current_value > 0 else 1.0
        avg_future_profit = np.mean(list(future_profits.values()))
        if avg_future_profit > 0.01 and cash_weight > 0.3:  # Good opportunities but holding >30% cash
            reward -= cash_weight * 50
        
        return reward
    
    def step(self, action):
        """
        Execute one step in the environment.
        Action: list with target allocation percentages for each symbol in order of self.asset_names
        Returns: observation, reward, done, info
        """
        # Check if we've reached the end of data
        if self.step_count >= len(self.df['BTCUSDT']) - self.lookforward_window_size - 1:
            return self.get_observation(), 0, True, {'reason': 'end_of_data'}
        
        # Store previous values for reward calculation
        prev_portfolio_value = self._portfolio_portfolio_value()
        prev_benchmark_value = self.passive_portfolio_value
        
        # Execute rebalancing
        fees_paid = self._rebalance_portfolio(action)
        
        # Move to next step
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get new observation
        obs = self.get_observation()
        
        # Check if done (end of episode or bankruptcy)
        done = False
        info = {}
        
        current_value = self._portfolio_portfolio_value()
        if current_value <= self.initial_account_balance * 0.1:  # Lost 90%
            done = True
            info['reason'] = 'bankruptcy'
            reward -= 100  # Large penalty
        
        if self.step_count >= len(self.df['BTCUSDT']) - self.lookforward_window_size - 1:
            done = True
            info['reason'] = 'end_of_data'
        
        # Add info
        info['portfolio_value'] = current_value
        info['benchmark_value'] = self.passive_portfolio_value
        info['outperformance'] = self._calculate_benchmark_outperformance()
        info['fees_paid'] = fees_paid
        info['step'] = self.step_count
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment state.
        """
        portfolio_value = self._portfolio_portfolio_value()
        benchmark_value = self._update_benchmark_value()
        outperformance = self._calculate_benchmark_outperformance()
        
        print(f"\n{'='*60}")
        print(f"Step: {self.step_count}")
        print(f"{'='*60}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Benchmark Value: ${benchmark_value:,.2f}")
        print(f"Outperformance: {(outperformance - 1) * 100:+.2f}%")
        print(f"Total Return: {(portfolio_value / self.initial_account_balance - 1) * 100:+.2f}%")
        print(f"Benchmark Return: {(benchmark_value / self.initial_account_balance - 1) * 100:+.2f}%")
        print(f"\nCash: ${self.portfolio['cash']:,.2f} ({self.portfolio['cash']/portfolio_value*100:.1f}%)")
        print(f"\nHoldings:")
        print(f"{'Symbol':<12} {'Quantity':>12} {'Price':>12} {'Value':>14} {'% Portfolio':>12}")
        print(f"{'-'*60}")
        
        for symbol in sorted(self.df.keys()):
            quantity = self.portfolio[symbol]
            if quantity > 0.00001:  # Only show non-zero holdings
                price = self._get_current_priced(symbol)
                value = quantity * price
                pct = (value / portfolio_value * 100) if portfolio_value > 0 else 0
                print(f"{symbol:<12} {quantity:>12.6f} ${price:>11,.2f} ${value:>13,.2f} {pct:>11.2f}%")
        
        print(f"{'='*60}\n")

    def reset(self):
        """
        Reset the environment to initial state.
        Returns: initial observation
        """
        # Reset step count to random position
        self.step_count = random.randint(2000, len(self.df['BTCUSDT']) - 1000)
        
        # Reset portfolio
        self.portfolio = {'cash': self.initial_account_balance}
        self.portfolio_value = self.initial_account_balance
        for symbol in self.df.keys():
            self.portfolio[symbol] = 0.0
        
        # Reset benchmark portfolio (equal weight across all assets)
        self.passive_portfolio = {}
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            self.passive_portfolio[symbol] = self.initial_account_balance / len(self.df) / price
        self.passive_portfolio_value = self.initial_account_balance
        
        # Return initial observation
        return self.get_observation()