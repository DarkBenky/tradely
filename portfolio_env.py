"""
Portfolio Trading Environment with Multi-Timeframe Reward Function

REWARD SYSTEM:
--------------
The agent is rewarded based on ACTUAL future portfolio returns across multiple timeframes.
This encourages the model to learn both short-term and long-term price prediction and optimal allocation.

Reward = weighted_average(returns_normalized_by_days) * 100 + outperformance_bonus

Multi-Timeframe Approach:
- Next Step (5 min): 40% weight - immediate price action
- Next Day (24h): 30% weight - intraday trends
- Next Week (7d): 20% weight - swing trading
- Next Month (30d): 10% weight - position trading

Each return is normalized by the number of days to make them comparable:
- A 1% gain in 5 minutes = much stronger signal than 1% gain in 30 days
- Returns are scaled: return / days

Safety Features:
1. All timeframe lookups check for out-of-bounds indices
2. Missing timeframes are gracefully skipped with weight rebalancing
3. Returns are clipped to ±200% per day to prevent extreme values
4. NaN and Inf values are sanitized to 0.0

Key Points:
1. Looks across multiple timeframes (not just 1 step)
2. Rewards positions that will increase across all timeframes
3. Naturally penalizes holding cash (0% return = opportunity cost)
4. Includes bonus for beating the buy-and-hold benchmark
5. Day-normalized returns ensure fairness across timeframes

This design ensures the agent learns both short-term tactical and long-term strategic positioning.
"""

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
        
        # Calculate safe bounds for initial step_count
        total_length = len(self.df['BTCUSDT'])
        self.lookback_window_size = 48
        
        # Need enough lookback for observation
        # Need enough lookforward for higher timeframes (max is 288 for 1d)
        # Need enough space for multi-timeframe reward lookahead (max is 288*30 for next_month)
        max_lookahead = 288 * 30  # 30 days
        min_step = max(2000, self.lookback_window_size + 288 * 3)
        max_step = total_length - max_lookahead - 100
        
        if min_step >= max_step:
            min_step = max(100, self.lookback_window_size + 288 * 3)
            max_step = total_length - max_lookahead - 50
            
        if min_step >= max_step:
            raise ValueError(f"Dataset too small: {total_length} records. Need at least {min_step + max_lookahead + 100}")
        
        self.step_count = random.randint(min_step, max_step)
        self.max_steps = total_length - self.step_count - max_lookahead - 10
        self.candle_interval = '5m'
        self.high_timeframes = ['15m', '1h', '4h', '1d']
        self.high_timeframes_count = [64, 48, 24, 3]

        # Multi-timeframe reward: looks at next step, next day, next week, next month
        
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
        for _ in self.asset_names:
            action.append(random.uniform(0, 1))
        # Normalize to sum to 1
        total = sum(action)
        action = [a / total for a in action]
        return action
        

    def get_observation(self):
        # Validate step_count is within bounds
        total_length = len(self.df['BTCUSDT'])
        if self.step_count < 0 or self.step_count >= total_length - 2:
            print(f"WARNING: step_count {self.step_count} out of bounds (max: {total_length}), resetting")
            self.step_count = max(2000, self.lookback_window_size + 288 * 3)
        
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
        
        # Final safety check - replace any NaN/Inf values
        if np.any(np.isnan(final_obs)) or np.any(np.isinf(final_obs)):
            print(f"WARNING: NaN/Inf detected in observation, cleaning...")
            final_obs = np.nan_to_num(final_obs, nan=0.0, posinf=1.0, neginf=0.0)
        
        return final_obs
    
    def _calculate_future_profit(self) -> dict:
        """
        Calculate the ACTUAL future profit for each asset across multiple timeframes.
        This looks ahead and is used ONLY for reward calculation (not available to agent).
        Returns a weighted average of returns over:
        - Next step (1 candle = 5 minutes)
        - Next day (288 candles = 24 hours)
        - Next week (288 * 7 = 2016 candles)
        - Next month (288 * 30 = 8640 candles)
        
        Each return is normalized by the number of days to make them comparable.
        Weights: 40% next step, 30% next day, 20% next week, 10% next month
        """
        future_profits = {}
        data_length = len(self.df['BTCUSDT'])
        
        # Define timeframes (in 5-minute candles)
        timeframes = {
            'next_step': 1,           # 5 minutes
            'next_day': 288,          # 24 hours
            'next_week': 288 * 7,     # 7 days
            'next_month': 288 * 30,   # 30 days
        }
        
        # Weights for each timeframe (must sum to 1.0)
        weights = {
            'next_step': 0.40,
            'next_day': 0.30,
            'next_week': 0.20,
            'next_month': 0.10,
        }
        
        for symbol, data in self.df.items():
            current_price = data['close'].iloc[self.step_count]
            
            # Protect against invalid current price
            if current_price <= 0 or np.isnan(current_price):
                future_profits[symbol] = 0.0
                continue
            
            weighted_return = 0.0
            total_weight_used = 0.0
            
            for tf_name, candles_ahead in timeframes.items():
                # Safety check: ensure we don't go out of bounds
                future_idx = self.step_count + candles_ahead
                
                if future_idx >= data_length:
                    # Out of bounds - skip this timeframe
                    continue
                
                try:
                    future_price = data['close'].iloc[future_idx]
                    
                    # Validate future price
                    if future_price <= 0 or np.isnan(future_price):
                        continue
                    
                    # Calculate raw return
                    raw_return = (future_price - current_price) / current_price
                    
                    # Normalize by number of days to make returns comparable
                    # (5-minute candles, so 288 candles = 1 day)
                    days = candles_ahead / 288.0
                    if days <= 0:
                        days = 1.0 / 288.0  # Minimum (for next_step)
                    
                    normalized_return = raw_return / days
                    
                    # Sanity check and clip extreme values
                    if np.isnan(normalized_return) or np.isinf(normalized_return):
                        normalized_return = 0.0
                    else:
                        # Clip to ±200% per day (very extreme movements)
                        normalized_return = np.clip(normalized_return, -2.0, 2.0)
                    
                    # Apply weight
                    weight = weights[tf_name]
                    weighted_return += normalized_return * weight
                    total_weight_used += weight
                    
                except (IndexError, KeyError):
                    # If we can't access the data, skip this timeframe
                    continue
            
            # Normalize by actual weights used (in case some timeframes were skipped)
            if total_weight_used > 0:
                final_return = weighted_return / total_weight_used
            else:
                final_return = 0.0
            
            # Final safety check
            if np.isnan(final_return) or np.isinf(final_return):
                final_return = 0.0
            
            future_profits[symbol] = final_return
            
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
        rebalance_threshold = 0.015  # Reduced from 0.025 to 1.5% for more active rebalancing
        
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
                    # FIXED: When selling, we want to REDUCE our holdings by trade_value worth
                    # But we need to account for the fee. If we sell X worth of assets:
                    # - We get: X - (X * fee_rate) = X * (1 - fee_rate) in cash
                    # - We need to sell: X / current_price quantity
                    # The fee is deducted from the proceeds, not added to the quantity
                    
                    # CRITICAL FIX: Validate we have enough to sell!
                    max_sellable_value = current_holdings * current_price
                    actual_sell_value = min(trade_value, max_sellable_value)
                    
                    if actual_sell_value < trade_value * 0.99:  # Sold significantly less than intended
                        if DEBUG:
                            print(f"WARNING: Wanted to sell ${trade_value:.2f} of {symbol} but only have ${max_sellable_value:.2f}")
                    
                    sell_quantity = actual_sell_value / current_price
                    fee = actual_sell_value * self.fee_rate
                    net_proceeds = actual_sell_value - fee
                    
                    # Safety check: don't sell more than we have (with tiny epsilon for rounding)
                    if sell_quantity > current_holdings:
                        sell_quantity = current_holdings
                        actual_sell_value = sell_quantity * current_price
                        fee = actual_sell_value * self.fee_rate
                        net_proceeds = actual_sell_value - fee
                    
                    self.portfolio['cash'] += net_proceeds
                    self.portfolio[symbol] -= sell_quantity
                    total_fees += fee
                    
                    # If we sold all (or nearly all), remove from average prices
                    if self.portfolio[symbol] <= 0.000001:
                        self.portfolio[symbol] = 0.0  # Clean up rounding errors
                        self.average_prices.pop(symbol, None)
                    
                    if DEBUG:
                        print(f"SELL {symbol}: ${actual_sell_value:.2f} -> ${net_proceeds:.2f} after fee (allocation diff: {allocation_diff:.3f})")
        
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
                # available_cash = trade_value + fee
                # We want to find max trade_value where: trade_value + (trade_value * fee_rate) <= available_cash
                # => trade_value * (1 + fee_rate) <= available_cash
                # => trade_value <= available_cash / (1 + fee_rate)
                if available_cash > 0.10:  # Need at least $0.10 to make a trade
                    affordable_trade_value = available_cash / (1 + self.fee_rate)
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
                            print(f"PARTIAL BUY {symbol}: ${affordable_trade_value:.2f} (wanted ${trade_value:.2f})")
        
        # Also check cash rebalancing threshold
        cash_allocation_diff = abs(cash_action - current_cash_allocation)
        if cash_allocation_diff > rebalance_threshold:
            if DEBUG:
                print(f"CASH allocation diff: {cash_allocation_diff:.3f}")
        
        # FIXED: Only apply cash inflation if we actually hold significant cash AND made trades
        # This prevents unfair penalty when not trading (benchmark never pays this)
        # Only apply if: (1) we have cash, (2) we made trades this step (total_fees > 0)
        if total_fees > 0 and self.portfolio['cash'] > 1.0:
            # Apply cash inflation only on the cash portion (opportunity cost)
            # Rate is very small per 5-minute interval: 0.004 / (365 * 288) ≈ 0.000000038 per step
            self.portfolio['cash'] *= (1 - self.cash_inflation_rate / (365 * 288))
        
        # FIXED: Ensure cash doesn't go negative due to rounding errors
        if self.portfolio['cash'] < -0.01:  # Small negative due to rounding
            if DEBUG:
                print(f"WARNING: Cash is negative: ${self.portfolio['cash']:.2f}, resetting to 0")
            self.portfolio['cash'] = 0.0
        
        return total_fees
    
    def _calculate_reward(self) -> float:
        """
        Simple, focused reward: MAXIMIZE ACTUAL FUTURE PROFIT
        
        The agent should learn to hold positions that will go up in the NEXT step
        and avoid/short positions that will go down.
        
        Reward = Portfolio-weighted actual return in next step
        """
        # Get ACTUAL next-step returns for each asset
        future_profits = self._calculate_future_profit()
        
        # Calculate current portfolio value and weights
        current_value = self._portfolio_portfolio_value()
        
        # Safety check
        if current_value <= 0 or np.isnan(current_value) or np.isinf(current_value):
            return 0.0
        
        # Calculate the actual profit the portfolio will make based on current holdings
        portfolio_future_return = 0.0
        
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            
            # Skip if price is invalid
            if price <= 0 or np.isnan(price):
                continue
            
            holdings_value = self.portfolio[symbol] * price
            weight = holdings_value / current_value
            
            # Weight the actual future return by portfolio allocation
            portfolio_future_return += weight * future_profits[symbol]
        
        # Safety check for portfolio return
        if np.isnan(portfolio_future_return) or np.isinf(portfolio_future_return):
            portfolio_future_return = 0.0
        
        # Cash earns nothing (actually loses to inflation, but simplified)
        # This naturally penalizes holding too much cash
        
        # FIXED: Much more conservative reward scaling to prevent instability
        # A 1% portfolio return gives reward of 1.0 (reduced from 10)
        reward = portfolio_future_return * 100
        
        # Add benchmark comparison bonus
        # Reward beating the passive buy-and-hold strategy
        benchmark_future_return = 0.0
        benchmark_value = self._update_benchmark_value()
        
        if benchmark_value > 0 and not np.isnan(benchmark_value):
            for symbol in self.df.keys():
                price = self.df[symbol]['close'].iloc[self.step_count]
                if price <= 0 or np.isnan(price):
                    continue
                    
                benchmark_holdings_value = self.passive_portfolio[symbol] * price
                benchmark_weight = benchmark_holdings_value / benchmark_value
                benchmark_future_return += benchmark_weight * future_profits[symbol]
            
            # Safety check
            if not np.isnan(benchmark_future_return) and not np.isinf(benchmark_future_return):
                # FIXED: Smaller bonus for outperforming benchmark (reduced from 500 to 50)
                outperformance = portfolio_future_return - benchmark_future_return
                reward += outperformance * 50  # Extra reward for beating benchmark
        
        # Final safety checks
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        # FIXED: Tighter clipping to prevent extreme values (±10 instead of ±100)
        reward = np.clip(reward, -10.0, 10.0)
        
        return float(reward)
    
    def step(self, action):
        """
        Execute one step in the environment.
        Action: list with target allocation percentages for each asset INCLUDING cash
        Returns: observation, reward, done, info
        """
        # Check if we've reached the end of data
        # Need enough space for multi-timeframe lookahead (max is 288*30 for next_month)
        max_lookahead = 288 * 30  # 30 days
        if self.step_count >= len(self.df['BTCUSDT']) - max_lookahead - 10:
            return self.get_observation(), 0, True, {'reason': 'end_of_data'}
        
        # Store previous values for reward calculation
        prev_portfolio_value = self._portfolio_portfolio_value()
        prev_benchmark_value = self.passive_portfolio_value
        
        # Execute rebalancing (uses current step prices)
        fees_paid = self._rebalance_portfolio(action)
        
        # CRITICAL: Validate portfolio state after rebalancing
        # Check for negative holdings (should never happen with our fixes)
        for symbol in self.df.keys():
            if self.portfolio[symbol] < -0.000001:  # Allow tiny rounding errors
                print(f"CRITICAL ERROR: Negative holdings for {symbol}: {self.portfolio[symbol]}")
                self.portfolio[symbol] = 0.0  # Fix it
        
        if self.portfolio['cash'] < -0.01:
            print(f"CRITICAL ERROR: Negative cash: ${self.portfolio['cash']:.2f}")
            self.portfolio['cash'] = 0.0  # Fix it
        
        # Move to next step
        self.step_count += 1
        
        # Calculate reward (this looks ahead across multiple timeframes)
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
        
        if self.step_count >= len(self.df['BTCUSDT']) - max_lookahead - 10:
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
        # Calculate safe bounds for step_count
        total_length = len(self.df['BTCUSDT'])
        
        # Need enough lookback for observation (lookback_window_size = 48)
        # Need enough lookforward for higher timeframes (max is 288 for 1d with count of 3)
        # Need enough space for multi-timeframe reward lookahead (max is 288*30 for next_month)
        max_lookahead = 288 * 30  # 30 days
        min_step = max(2000, self.lookback_window_size + 288 * 3)
        max_step = total_length - max_lookahead - 100  # Reserve space for lookahead + buffer
        
        # Validate bounds
        if min_step >= max_step:
            # Data is too small, use minimal safe bounds
            min_step = max(100, self.lookback_window_size + 288 * 3)
            max_step = total_length - max_lookahead - 50
            
        if min_step >= max_step:
            raise ValueError(f"Dataset too small: {total_length} records. Need at least {min_step + max_lookahead + 100}")
        
        # Reset step count to random position within safe bounds
        self.step_count = random.randint(min_step, max_step)
        
        # Recalculate max_steps based on new position
        self.max_steps = total_length - self.step_count - max_lookahead - 10
        
        # Validate the chosen step_count has valid data
        try:
            test_price = self.df['BTCUSDT']['close'].iloc[self.step_count]
            if np.isnan(test_price) or test_price <= 0:
                print(f"WARNING: Invalid price at step {self.step_count}, trying different position")
                # Try a few more times
                for attempt in range(5):
                    self.step_count = random.randint(min_step, max_step)
                    test_price = self.df['BTCUSDT']['close'].iloc[self.step_count]
                    if not np.isnan(test_price) and test_price > 0:
                        break
        except Exception as e:
            print(f"ERROR in reset: {e}")
            self.step_count = min_step  # Fallback to minimum safe position
        
        # Reset portfolio
        self.portfolio = {'cash': 2000.0}
        self.portfolio_value = self.initial_account_balance
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            # Validate price
            if np.isnan(price) or price <= 0:
                print(f"WARNING: Invalid price for {symbol} at step {self.step_count}, using 1.0")
                price = 1.0
            self.portfolio[symbol] = (self.initial_account_balance - 2000) / len(self.df) / price
        
        # Reset average prices
        self.average_prices = {}
        for symbol in self.df.keys():
            if self.portfolio[symbol] > 0:
                price = self.df[symbol]['close'].iloc[self.step_count]
                if not np.isnan(price) and price > 0:
                    self.average_prices[symbol] = price
                else:
                    self.average_prices[symbol] = 1.0
        
        # Reset benchmark portfolio (equal weight across all assets)
        self.passive_portfolio = {}
        for symbol in self.df.keys():
            price = self.df[symbol]['close'].iloc[self.step_count]
            if np.isnan(price) or price <= 0:
                price = 1.0
            self.passive_portfolio[symbol] = self.initial_account_balance / len(self.df) / price
        self.passive_portfolio_value = self.initial_account_balance
        
        # Return initial observation
        obs = self.get_observation()
        
        # Final validation - if observation has NaN, try one more reset
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"WARNING: Invalid observation after reset, trying fallback position")
            self.step_count = min_step
            self.max_steps = total_length - self.step_count - 100
            obs = self.get_observation()
        
        # Always clean any remaining NaN/Inf to prevent training issues
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs