import numpy as np
import pandas as pd
import os
import random

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

class PortfolioEnv():
    def __init__(self, data_dir='data', max_records=None):
        self.initial_account_balance = 10000.0
        self.reset_value = self.initial_account_balance // 2
        self.df = load_and_align_data(data_dir, max_records)
        self.asset_names = list(self.df.keys()) + ['cash']
        self.step_count = random.randint(2000, len(self.df['BTCUSDT']) - 1000)
        self.max_steps = len(self.df['BTCUSDT']) - self.step_count - 100
        self.candle_interval = '5m'
        self.high_timeframes = ['15m', '1h', '4h', '1d']
        self.high_timeframes_count = [64, 48, 24, 3]

        self.lookback_window_size = 48
        self.lookforward_window_size = 30
        
        self.portfolio = {'cash': self.initial_account_balance}
        self.portfolio_value = self.initial_account_balance
        for symbol in self.df.keys():
            self.portfolio[symbol] = 0.0
        
        # Initialize average prices tracking
        self.average_prices = {}
        
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
    
    def _calculate_average_price(self, symbol, current_quantity):
        """
        Calculate average purchase price for a symbol.
        """
        if current_quantity <= 0.000001:
            return 0.0
        
        return self.average_prices.get(symbol, self._get_current_priced(symbol))
    
    def sample(self):
        action = []
        # FIXED: Generate action for all assets including cash
        for _ in self.asset_names:
            action.append(random.uniform(0, 1))
        # Normalize to sum to 1 (required for proper allocation)
        total = sum(action)
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
            
            # Add average price (normalized by current price)
            avg_price = self.average_prices.get(symbol, price)
            avg_price_normalized = avg_price / price if price > 0 else 1.0
            portfolio_state.append(avg_price_normalized)
            
            # Add profit/loss percentage for the asset
            if avg_price > 0 and holdings > 0.000001:
                pl_percent = (price - avg_price) / avg_price
            else:
                pl_percent = 0.0
            portfolio_state.append(pl_percent)
        
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
        Action is a list with target percentages for each asset including cash.
        Example: [0.3, 0.2, 0.1, ..., 0.15] for ['BTCUSDT', 'ETHUSDT', ..., 'cash']
        
        Only rebalances when the allocation difference exceeds a threshold.
        """
        # Calculate current portfolio value
        current_value = self._portfolio_portfolio_value()
        
        # FIXED: Separate crypto and cash allocations
        num_cryptos = len(self.df)
        crypto_actions = action[:num_cryptos]
        cash_action = action[num_cryptos] if len(action) > num_cryptos else 0.0
        
        # Ensure all values are in 0-1 range
        crypto_actions = [np.clip(a, 0.0, 1.0) for a in crypto_actions]
        cash_action = np.clip(cash_action, 0.0, 1.0)
        
        # Normalize to sum to 1.0
        total_allocation = sum(crypto_actions) + cash_action
        if total_allocation > 0:
            crypto_actions = [a / total_allocation for a in crypto_actions]
            cash_action = cash_action / total_allocation
        else:
            # If all zeros, default to equal allocation
            crypto_actions = [1.0 / len(self.asset_names)] * num_cryptos
            cash_action = 1.0 / len(self.asset_names)
        
        # Calculate target values and current allocations
        target_values = {}
        current_allocations = {}
        
        for i, symbol in enumerate(self.df.keys()):
            current_price = self._get_current_priced(symbol)
            current_holdings = self.portfolio[symbol]
            current_value_in_symbol = current_holdings * current_price
            current_allocations[symbol] = current_value_in_symbol / current_value if current_value > 0 else 0
            
            target_values[symbol] = current_value * crypto_actions[i]
        
        target_cash = current_value * cash_action
        current_cash_allocation = self.portfolio['cash'] / current_value if current_value > 0 else 0
        
        # Calculate trades needed with threshold check
        total_fees = 0.0
        rebalance_threshold = 0.025  # 2.5% threshold
        
        # FIXED: Track pending buy orders to ensure we don't exceed available cash
        pending_buys = []
        
        # First pass: identify all trades and validate cash availability
        for i, symbol in enumerate(self.df.keys()):
            current_price = self._get_current_priced(symbol)
            current_holdings = self.portfolio[symbol]
            current_value_in_symbol = current_holdings * current_price
            
            target_value_in_symbol = target_values[symbol]
            value_diff = target_value_in_symbol - current_value_in_symbol
            
            # Calculate allocation difference percentage
            allocation_diff = abs(crypto_actions[i] - current_allocations[symbol])
            
            # Only trade if the allocation difference exceeds the threshold
            if allocation_diff > rebalance_threshold and abs(value_diff) > 0.01:
                trade_value = abs(value_diff)
                fee = trade_value * self.fee_rate
                
                if value_diff > 0:  # Buy
                    total_cost = trade_value + fee
                    pending_buys.append((symbol, trade_value, fee, current_price, allocation_diff))
                else:  # Sell
                    # Selling adds to cash, so we can execute immediately
                    sell_quantity = trade_value / current_price
                    self.portfolio['cash'] += (trade_value - fee)
                    self.portfolio[symbol] -= sell_quantity
                    total_fees += fee
                    
                    # If we sold all, remove from average prices
                    if self.portfolio[symbol] <= 0.000001:
                        self.average_prices.pop(symbol, None)
                    
                    if DEBUG:
                        print(f"SELL {symbol}: ${trade_value:.2f} (allocation diff: {allocation_diff:.3f})")
        
        # FIXED: Process buy orders in order of allocation difference (most important first)
        # Sort by allocation difference (descending) to prioritize most important trades
        pending_buys.sort(key=lambda x: x[4], reverse=True)
        
        available_cash = self.portfolio['cash']
        
        for symbol, trade_value, fee, current_price, allocation_diff in pending_buys:
            total_cost = trade_value + fee
            
            # FIXED: Check if we have enough cash for this buy
            if available_cash >= total_cost:
                # Calculate quantity to buy
                buy_quantity = trade_value / current_price
                
                # Update average price using weighted average
                current_holdings = self.portfolio[symbol]
                current_avg_price = self.average_prices.get(symbol, 0)
                current_total_value = current_holdings * current_avg_price if current_holdings > 0 else 0
                new_total_value = current_total_value + trade_value
                new_quantity = current_holdings + buy_quantity
                
                if new_quantity > 0:
                    self.average_prices[symbol] = new_total_value / new_quantity
                else:
                    self.average_prices[symbol] = current_price
                
                # Deduct from cash including fee
                self.portfolio['cash'] -= total_cost
                available_cash -= total_cost
                self.portfolio[symbol] += buy_quantity
                total_fees += fee
                
                if DEBUG:
                    print(f"BUY {symbol}: ${trade_value:.2f} (allocation diff: {allocation_diff:.3f})")
            else:
                # FIXED: If not enough cash, buy what we can afford
                if available_cash > fee:  # Need at least enough for fee + minimal trade
                    affordable_trade_value = (available_cash - fee) / (1 + self.fee_rate)
                    if affordable_trade_value > 1.0:  # Only trade if meaningful amount
                        affordable_fee = affordable_trade_value * self.fee_rate
                        buy_quantity = affordable_trade_value / current_price
                        
                        # Update average price
                        current_holdings = self.portfolio[symbol]
                        current_avg_price = self.average_prices.get(symbol, 0)
                        current_total_value = current_holdings * current_avg_price if current_holdings > 0 else 0
                        new_total_value = current_total_value + affordable_trade_value
                        new_quantity = current_holdings + buy_quantity
                        
                        if new_quantity > 0:
                            self.average_prices[symbol] = new_total_value / new_quantity
                        
                        total_cost = affordable_trade_value + affordable_fee
                        self.portfolio['cash'] -= total_cost
                        available_cash -= total_cost
                        self.portfolio[symbol] += buy_quantity
                        total_fees += affordable_fee
                        
                        if DEBUG:
                            print(f"PARTIAL BUY {symbol}: ${affordable_trade_value:.2f} (could afford ${trade_value:.2f})")
        
        # Also check cash rebalancing threshold
        cash_allocation_diff = abs(cash_action - current_cash_allocation)
        if cash_allocation_diff > rebalance_threshold:
            if DEBUG:
                print(f"CASH allocation diff: {cash_allocation_diff:.3f}")
        
        # Apply cash inflation (opportunity cost)
        self.portfolio['cash'] *= (1 - self.cash_inflation_rate / (365 * (5/60)))  # 5-minute interval
        
        # FIXED: Ensure cash doesn't go negative due to rounding errors
        if self.portfolio['cash'] < -0.01:  # Small negative due to rounding
            if DEBUG:
                print(f"WARNING: Cash is negative: ${self.portfolio['cash']:.2f}, resetting to 0")
            self.portfolio['cash'] = 0.0
        
        return total_fees
    
    # def _calculate_reward(self) -> float:
    #     """
    #     Calculate reward based on:
    #     1. Weighted future returns (farther future = more weight)
    #     2. Portfolio alignment with future profit potential
    #     3. Concentration penalty for risk management
    #     4. FIXED: Reward for optimal cash usage
    #     """
    #     # Get future profit predictions (weighted toward farther future)
    #     future_profits = self._calculate_future_profit()
        
    #     # Update current portfolio value
    #     current_value = self._portfolio_portfolio_value()
        
    #     # Calculate portfolio-weighted expected return
    #     portfolio_weighted_return = 0.0
    #     for symbol in self.df.keys():
    #         price = self.df[symbol]['close'].iloc[self.step_count]
    #         holdings_value = self.portfolio[symbol] * price
    #         weight = holdings_value / current_value if current_value > 0 else 0
            
    #         # Weight by portfolio allocation and future profit
    #         portfolio_weighted_return += weight * future_profits[symbol]
        
    #     # Base reward: portfolio's alignment with future returns
    #     reward = portfolio_weighted_return * 1000  # Scale up for meaningful rewards
        
    #     # Calculate benchmark's weighted future return for comparison
    #     benchmark_weighted_return = 0.0
    #     benchmark_value = self._update_benchmark_value()
    #     for symbol in self.df.keys():
    #         price = self.df[symbol]['close'].iloc[self.step_count]
    #         benchmark_holdings_value = self.passive_portfolio[symbol] * price
    #         benchmark_weight = benchmark_holdings_value / benchmark_value if benchmark_value > 0 else 0
    #         benchmark_weighted_return += benchmark_weight * future_profits[symbol]
        
    #     # Bonus for better positioning than benchmark
    #     outperformance = portfolio_weighted_return - benchmark_weighted_return
    #     reward += outperformance * 500
        
    #     # Calculate portfolio concentration (Herfindahl index) for risk penalty
    #     concentration = 0
    #     for symbol in self.df.keys():
    #         price = self.df[symbol]['close'].iloc[self.step_count]
    #         holdings_value = self.portfolio[symbol] * price
    #         weight = holdings_value / current_value if current_value > 0 else 0
    #         concentration += weight ** 2
        
    #     # Penalize over-concentration (>0.5 Herfindahl index)
    #     if concentration > 0.5:
    #         reward -= (concentration - 0.5) * 100
        
    #     # FIXED: Smart cash penalty - only penalize when missing good opportunities
    #     cash_weight = self.portfolio['cash'] / current_value if current_value > 0 else 1.0
    #     avg_future_profit = np.mean(list(future_profits.values()))
    #     max_future_profit = max(future_profits.values())
        
    #     # If there are strong positive opportunities and holding too much cash, penalize
    #     if max_future_profit > 0.02 and cash_weight > 0.3:  
    #         reward -= cash_weight * 50
    #     # If market looks bad (negative returns expected), reward holding cash
    #     elif avg_future_profit < -0.01 and cash_weight > 0.3:
    #         reward += cash_weight * 30
        
    #     return reward

    def _calculate_reward(self) -> float:
        """
        Enhanced reward function based on:
        1. Sharpe-like ratio of future returns
        2. Risk-adjusted portfolio alignment
        3. Dynamic concentration management
        4. Smart cash positioning
        5. Trend momentum bonus
        """
        # Get future profit predictions
        future_profits = self._calculate_future_profit()
        
        # Update current portfolio value
        current_value = self._portfolio_portfolio_value()
        
        # ===== 1. Calculate portfolio-weighted expected return =====
        portfolio_weighted_return = 0.0
        portfolio_weights = {}
        
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            holdings_value = self.portfolio[symbol] * price
            weight = holdings_value / current_value if current_value > 0 else 0
            portfolio_weights[symbol] = weight
            portfolio_weighted_return += weight * future_profits[symbol]
        
        # ===== 2. Calculate risk (volatility) of future returns =====
        # Use standard deviation of future returns as risk measure
        future_returns_list = list(future_profits.values())
        returns_std = np.std(future_returns_list) if len(future_returns_list) > 1 else 0.001
        
        # Calculate portfolio risk (weighted by holdings)
        portfolio_risk = 0.0
        for symbol in self.df.keys():
            # Simple approximation: weight * individual asset volatility
            portfolio_risk += portfolio_weights[symbol] * abs(future_profits[symbol])
        portfolio_risk = max(portfolio_risk, 0.001)  # Avoid division by zero
        
        # ===== 3. Sharpe-like ratio reward (return/risk) =====
        sharpe_reward = (portfolio_weighted_return / returns_std) * 100
        
        # ===== 4. Benchmark comparison with dynamic weighting =====
        benchmark_weighted_return = 0.0
        benchmark_value = self._update_benchmark_value()
        
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            benchmark_holdings_value = self.passive_portfolio[symbol] * price
            benchmark_weight = benchmark_holdings_value / benchmark_value if benchmark_value > 0 else 0
            benchmark_weighted_return += benchmark_weight * future_profits[symbol]
        
        # Outperformance bonus (scaled by confidence)
        outperformance = portfolio_weighted_return - benchmark_weighted_return
        confidence_multiplier = 1.0 / (1.0 + returns_std * 10)  # Lower when returns are volatile
        outperformance_reward = outperformance * 800 * confidence_multiplier
        
        # ===== 5. Dynamic concentration management =====
        # Calculate Herfindahl index
        concentration = sum(w ** 2 for w in portfolio_weights.values())
        
        # Get market conditions
        avg_future_profit = np.mean(future_returns_list)
        max_future_profit = max(future_returns_list)
        profit_spread = max(future_returns_list) - min(future_returns_list)
        
        # Dynamic concentration threshold based on market conditions
        if profit_spread > 0.05:  # High dispersion = opportunity for concentration
            optimal_concentration = 0.35  # Allow more concentration
            concentration_tolerance = 0.25
        else:  # Low dispersion = diversification safer
            optimal_concentration = 0.20  # Encourage diversification
            concentration_tolerance = 0.15
        
        # Penalty for deviation from optimal concentration
        concentration_deviation = abs(concentration - optimal_concentration)
        if concentration_deviation > concentration_tolerance:
            concentration_penalty = (concentration_deviation - concentration_tolerance) * 200
        else:
            concentration_penalty = 0
        
        # ===== 6. Smart cash management =====
        cash_weight = self.portfolio['cash'] / current_value if current_value > 0 else 1.0
        
        # Identify strong opportunities and weak assets
        strong_opportunities = [p for p in future_returns_list if p > 0.03]
        weak_assets = [p for p in future_returns_list if p < -0.02]
        
        # Cash reward/penalty logic
        if len(strong_opportunities) >= 2 and cash_weight > 0.3:
            # Missing good opportunities - penalize excess cash
            cash_adjustment = -(cash_weight - 0.2) * 100
        elif avg_future_profit < -0.015 and cash_weight > 0.3:
            # Market looks bad, reward holding cash
            cash_adjustment = (cash_weight - 0.2) * 80
        elif cash_weight < 0.05 and avg_future_profit < -0.01:
            # No cash buffer in bad market - penalize
            cash_adjustment = -(0.05 - cash_weight) * 60
        elif cash_weight > 0.7:
            # Too much cash (probably always bad)
            cash_adjustment = -(cash_weight - 0.5) * 150
        else:
            # Reasonable cash position
            cash_adjustment = 0
        
        # ===== 7. Momentum alignment bonus =====
        # Reward for being positioned in assets with consistent positive trends
        momentum_score = 0.0
        for symbol in self.df.keys():
            if portfolio_weights[symbol] > 0.05:  # Only for meaningful positions
                # Check if future returns are consistently positive
                if future_profits[symbol] > 0.02:
                    momentum_score += portfolio_weights[symbol] * future_profits[symbol] * 200
                elif future_profits[symbol] < -0.02:
                    # Penalize being positioned in declining assets
                    momentum_score -= portfolio_weights[symbol] * abs(future_profits[symbol]) * 150
        
        # ===== 8. Diversification bonus in uncertain markets =====
        diversification_bonus = 0.0
        if returns_std > 0.03:  # High uncertainty
            # Count number of meaningful positions (>5%)
            num_positions = sum(1 for w in portfolio_weights.values() if w > 0.05)
            if num_positions >= 3:  # Well diversified
                diversification_bonus = num_positions * 15
        
        # ===== 9. Combine all components =====
        reward = (
            sharpe_reward +              # Risk-adjusted return
            outperformance_reward +       # Beat benchmark
            momentum_score +              # Momentum alignment
            diversification_bonus +       # Diversification in uncertain times
            cash_adjustment -             # Smart cash positioning
            concentration_penalty         # Concentration management
        )
        
        # ===== 10. Clip reward to reasonable range =====
        reward = np.clip(reward, -200, 200)
        
        return reward
    
    def step(self, action):
        """
        Execute one step in the environment.
        Action: list with target allocation percentages for each asset INCLUDING cash
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

        if self.portfolio_value <= self.reset_value:
            done = True
            info['reason'] = 'reset_threshold_reached'
            reward -= 50  # Penalty for hitting reset threshold
        
        # Add info
        info['portfolio_value'] = current_value
        info['benchmark_value'] = self.passive_portfolio_value
        info['outperformance'] = self._calculate_benchmark_outperformance()
        info['fees_paid'] = fees_paid
        info['step'] = self.step_count
        info['cash_allocation'] = self.portfolio['cash'] / current_value if current_value > 0 else 1.0
        info['portfolio_layout'] = {symbol: self.portfolio[symbol] for symbol in self.df.keys()}
        info['portfolio_layout']['cash'] = self.portfolio['cash']
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment state.
        """
        portfolio_value = self._portfolio_portfolio_value()
        benchmark_value = self._update_benchmark_value()
        outperformance = self._calculate_benchmark_outperformance()
        
        print(f"\n{'='*90}")
        print(f"Step: {self.step_count}")
        print(f"{'='*90}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Benchmark Value: ${benchmark_value:,.2f}")
        print(f"Outperformance: {(outperformance - 1) * 100:+.2f}%")
        print(f"Total Return: {(portfolio_value / self.initial_account_balance - 1) * 100:+.2f}%")
        print(f"Benchmark Return: {(benchmark_value / self.initial_account_balance - 1) * 100:+.2f}%")
        print(f"\nCash: ${self.portfolio['cash']:,.2f} ({self.portfolio['cash']/portfolio_value*100:.1f}%)")
        print(f"\nHoldings:")
        print(f"{'Symbol':<12} {'Quantity':>12} {'Avg Price':>12} {'Cur Price':>12} {'P/L %':>8} {'Value':>14} {'% Port':>10}")
        print(f"{'-'*90}")
        
        # Track total P&L
        total_pl = 0
        
        for symbol in sorted(self.df.keys()):
            quantity = self.portfolio[symbol]
            if quantity > 0.00001:  # Only show non-zero holdings
                current_price = self._get_current_priced(symbol)
                value = quantity * current_price
                
                # Calculate average price
                avg_price = self._calculate_average_price(symbol, quantity)
                
                if avg_price > 0:
                    pl_percent = ((current_price - avg_price) / avg_price) * 100
                    pl_value = (current_price - avg_price) * quantity
                    total_pl += pl_value
                else:
                    pl_percent = 0.0
                    pl_value = 0.0
                
                pct_portfolio = (value / portfolio_value * 100) if portfolio_value > 0 else 0
                
                # Color coding for P&L
                pl_color = ""
                pl_reset = ""
                if pl_percent > 0:
                    pl_color = "\033[92m"  # Green
                    pl_reset = "\033[0m"
                elif pl_percent < 0:
                    pl_color = "\033[91m"  # Red
                    pl_reset = "\033[0m"
                
                print(f"{symbol:<12} {quantity:>12.6f} ${avg_price:>11.2f} ${current_price:>11.2f} "
                      f"{pl_color}{pl_percent:>7.1f}%{pl_reset} ${value:>13,.2f} {pct_portfolio:>9.1f}%")
        
        # Print total P&L
        print(f"{'-'*90}")
        total_pl_color = "\033[92m" if total_pl >= 0 else "\033[91m"
        total_pl_reset = "\033[0m"
        print(f"{'Total P&L:':<12} {'':>12} {'':>12} {'':>12} {total_pl_color}${total_pl:>13,.2f}{total_pl_reset} {'':>10}")
        print(f"{'='*90}\n")

    def reset(self):
        """
        Reset the environment to initial state.
        Returns: initial observation
        """
        # Reset step count to random position
        self.step_count = random.randint(2000, len(self.df['BTCUSDT']) - 1000)
        
        # Reset portfolio
        self.portfolio = {'cash': 2000.0}
        self.portfolio_value = self.initial_account_balance
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            self.portfolio[symbol] = (self.initial_account_balance - 2000) / len(self.df) / price
        
        # Reset average prices
        self.average_prices = {}
        for symbol in self.df.keys():
            if self.portfolio[symbol] > 0:
                self.average_prices[symbol] = self.df[symbol]['close'].iloc[self.step_count]
        
        # Reset benchmark portfolio (equal weight across all assets)
        self.passive_portfolio = {}
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            self.passive_portfolio[symbol] = self.initial_account_balance / len(self.df) / price
        self.passive_portfolio_value = self.initial_account_balance
        
        # Return initial observation
        return self.get_observation()