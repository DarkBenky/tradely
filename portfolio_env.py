import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import warnings
from collections import deque
warnings.filterwarnings('ignore')


class AggressiveMultiCurrencyPortfolioEnv(gym.Env):
    """
    Aggressive Multi-Currency Portfolio Environment
    Designed specifically to outperform passive benchmarks through active strategies
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        dfs: Dict[str, pd.DataFrame],
        symbols: List[str],
        starting_balance: float = 10000.0,
        fee_rate: float = 0.001,
        lookback_window: int = 50,
        rebalance_frequency: int = 6,
        risk_free_rate: float = 0.02,
        max_drawdown_penalty: float = 1.0,
        normalize_obs: bool = True,
        reward_horizon: int = 1,
        benchmark_weight: float = 3.0,
        future_profit_weight: float = 2.0,
        future_window: int = 8,
        risk_adjustment: bool = False,
        dynamic_penalties: bool = False,
        use_attention_weights: bool = True,
        volatility_scaling: bool = False,
        periods_per_year: Optional[int] = None,
        clip_metrics: bool = False,
        momentum_lookback: int = 10,
        concentration_limit: float = 0.7,
        trend_following: bool = True
        # Removed leverage_limit parameter
    ):
        super(AggressiveMultiCurrencyPortfolioEnv, self).__init__()
        
        self.dfs = dfs
        self.symbols = symbols
        self.num_assets = len(symbols)
        self.starting_balance = starting_balance
        self.fee_rate = fee_rate
        self.lookback_window = int(lookback_window)
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_penalty = max_drawdown_penalty
        self.normalize_obs = normalize_obs
        self.reward_horizon = reward_horizon
        self.benchmark_weight = benchmark_weight
        self.future_profit_weight = future_profit_weight
        self.future_window = future_window
        self.risk_adjustment = risk_adjustment
        self.dynamic_penalties = dynamic_penalties
        self.use_attention_weights = use_attention_weights
        self.volatility_scaling = volatility_scaling
        self.periods_per_year = periods_per_year
        self.clip_metrics = clip_metrics
        
        # Aggressive strategy parameters
        self.momentum_lookback = momentum_lookback
        self.concentration_limit = concentration_limit
        self.trend_following = trend_following
        
        # Initialize RNG
        self.np_random = np.random.default_rng()
        
        self._align_dataframes()
        
        first_df = self.dfs[symbols[0]]
        self.feature_columns = [col for col in first_df.columns 
                               if pd.api.types.is_numeric_dtype(first_df[col])]
        self.num_features_per_asset = len(self.feature_columns)
        
        # Action space without short positions (no negative allocations)
        self.action_space = spaces.Box(
            low=0.0,  # No short positions
            high=1.0,  # Maximum 100% long
            shape=(self.num_assets,),
            dtype=np.float32
        )
        
        # FIXED: Calculate observation dimension correctly
        # Features: lookback_window * num_features_per_asset * num_assets
        feature_dim = self.lookback_window * self.num_features_per_asset * self.num_assets
        
        # Portfolio state: weights (num_assets) + 11 scalars + momentum_signals (num_assets) + trend_signals (num_assets)
        # Removed gross_exposure and net_exposure (2 scalars)
        portfolio_state_dim = self.num_assets + 11 + self.num_assets + self.num_assets
        
        obs_dim = feature_dim + portfolio_state_dim
        
        print(f"Aggressive Portfolio Environment")
        print(f"Observation dimension: {obs_dim}")
        print(f"  - Features: {feature_dim}")
        print(f"  - Portfolio state: {portfolio_state_dim}")
        print(f"Concentration limit: {self.concentration_limit}")
        print(f"No leverage or short positions allowed")

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(obs_dim),),
            dtype=np.float32
        )
        
        # Initialize state variables
        self._initialize_state()
        
        # Risk management - looser limits but no leverage
        self.volatility_lookback = 15
        self.correlation_lookback = 30
        self.risk_limits = {
            'max_single_asset': self.concentration_limit,
            'max_drawdown': 0.4,
            'target_volatility': 0.25
        }
        
        # Aggressive reward parameters
        self.reward_components = {
            'returns': 1.5,
            'sharpe': 0.05,
            'drawdown': 0.05,
            'benchmark': 3.0,
            'future': 2.0,
            'risk_adjusted': 0.0,
            'diversification': -0.1,
            'momentum': 0.5,
            'trend': 0.8
        }
        
        # Performance tracking
        self.rolling_sharpe = deque(maxlen=50)
        self.rolling_volatility = deque(maxlen=30)
        self.rolling_max_drawdown = deque(maxlen=30)
        
    def _initialize_state(self):
        """Initialize all state variables"""
        self.current_step = 0
        self.portfolio_value = self.starting_balance
        self.cash_balance = self.starting_balance
        self.holdings = {symbol: 0.0 for symbol in self.symbols}
        self.weights = {symbol: 0.0 for symbol in self.symbols}
        
        # Enhanced performance tracking
        self.portfolio_history = [self.starting_balance]
        self.returns_history = []
        self.weights_history = []
        self.peak_value = self.starting_balance
        self.max_drawdown = 0.0
        self.trades_executed = 0
        self.total_fees_paid = 0.0
        self.turnover_history = []
        
        self.action_history = []
        self.value_at_action = []
        self.steps_since_rebalance = 0
        
        # Benchmark portfolio
        self.benchmark_value = self.starting_balance
        self.benchmark_history = [self.starting_balance]
        self.benchmark_weights = {symbol: 1.0/self.num_assets for symbol in self.symbols}
        self.benchmark_holdings = {}
        
        # Risk metrics
        self.portfolio_volatility = 0.0
        self.portfolio_beta = 0.0
        self.diversification_ratio = 0.0
        
        # Aggressive strategy state
        self.momentum_signals = {symbol: 0.0 for symbol in self.symbols}
        self.trend_signals = {symbol: 0.0 for symbol in self.symbols}
        
    def _initialize_benchmark_portfolio(self):
        """Initialize equal-weight benchmark portfolio"""
        self.benchmark_holdings = {}
        
        for symbol in self.symbols:
            current_price = self._get_current_price(symbol)
            allocation = self.starting_balance * self.benchmark_weights[symbol]
            units = allocation / (current_price + 1e-8)
            self.benchmark_holdings[symbol] = units
    
    def _update_benchmark_value(self):
        """Update benchmark portfolio value"""
        if not self.benchmark_holdings:
            self.benchmark_value = self.starting_balance
            return
            
        total_value = 0.0
        for symbol in self.symbols:
            current_price = self._get_current_price(symbol)
            total_value += self.benchmark_holdings[symbol] * current_price
        self.benchmark_value = total_value
        self.benchmark_history.append(total_value)
    
    def _calculate_benchmark_outperformance(self) -> float:
        """Calculate risk-adjusted outperformance vs benchmark"""
        agent_return = (self.portfolio_value - self.starting_balance) / (self.starting_balance + 1e-8)
        benchmark_return = (self.benchmark_value - self.starting_balance) / (self.starting_balance + 1e-8)
        
        # Risk-adjusted outperformance
        if len(self.returns_history) > 10:
            agent_vol = np.std(self.returns_history[-20:], ddof=1) * np.sqrt(252) if len(self.returns_history) >= 20 else 0.1
            bench_vol = self._calculate_benchmark_volatility()
            
            if agent_vol > 0 and bench_vol > 0:
                risk_adjustment = (bench_vol / (agent_vol + 1e-8)) - 1.0
                return (agent_return - benchmark_return) * (1 + risk_adjustment * 0.5)
        
        return agent_return - benchmark_return
    
    def _calculate_benchmark_volatility(self) -> float:
        """Calculate benchmark portfolio volatility"""
        if len(self.benchmark_history) < 2:
            return 0.0
        
        bench_returns = []
        for i in range(1, min(20, len(self.benchmark_history))):
            ret = (self.benchmark_history[i] - self.benchmark_history[i-1]) / (self.benchmark_history[i-1] + 1e-8)
            bench_returns.append(ret)
        
        return np.std(bench_returns, ddof=1) * np.sqrt(252) if bench_returns else 0.0
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol with validation"""
        idx = self.current_step + self.lookback_window
        if idx < len(self.dfs[symbol]):
            return float(self.dfs[symbol].iloc[int(idx)]['close'])
        else:
            # Fallback to the last available price
            return float(self.dfs[symbol].iloc[-1]['close'])
    
    def _calculate_advanced_future_profit(self) -> float:
        """Advanced future profit calculation with momentum and volatility adjustment"""
        if self.future_window == 0:
            return 0.0
        
        total_profit_potential = 0.0
        max_future_steps = min(self.future_window, self.max_steps - self.current_step - 1)
        
        if max_future_steps <= 0:
            return 0.0
        
        # Adaptive weights based on recent volatility
        recent_volatilities = {}
        for symbol in self.symbols:
            prices = []
            for i in range(min(10, self.current_step)):
                idx = self.current_step + self.lookback_window - i
                # Ensure we're using integer indexing
                if 0 <= idx < len(self.dfs[symbol]):
                    price = self.dfs[symbol].iloc[int(idx)]['close']
                    prices.append(float(price))
            
            if len(prices) > 1:
                returns = np.diff(prices) / (np.array(prices[:-1]) + 1e-8)
                recent_volatilities[symbol] = np.std(returns, ddof=1) if len(returns) > 0 else 0.1
            else:
                recent_volatilities[symbol] = 0.1
        
        for symbol in self.symbols:
            if self.holdings[symbol] > 0:
                current_price = self._get_current_price(symbol)
                future_prices = []
                
                # Get future prices with confidence weights
                for i in range(1, max_future_steps + 1):
                    future_idx = self.current_step + self.lookback_window + i
                    if future_idx < len(self.dfs[symbol]):
                        future_price = float(self.dfs[symbol].iloc[int(future_idx)]['close'])
                        # Adjust for volatility - less confidence in volatile assets
                        vol_factor = 1.0 / (1.0 + recent_volatilities[symbol] * 10)
                        future_prices.append((future_price, vol_factor))
                    else:
                        future_prices.append((current_price, 0.1))
                
                if future_prices:
                    # Weight by volatility-adjusted confidence
                    weights = np.array([fp[1] for fp in future_prices])
                    weights = weights / (np.sum(weights) + 1e-8)
                    
                    weighted_future_price = sum(fp[0] * w for fp, w in zip(future_prices, weights))
                    profit_potential = (weighted_future_price - current_price) / (current_price + 1e-8)
                    
                    # Adjust for momentum
                    momentum = self._calculate_momentum(symbol)
                    momentum_adjusted_potential = profit_potential * (1 + momentum * 0.5)
                    
                    position_value = self.holdings[symbol] * current_price
                    total_profit_potential += momentum_adjusted_potential * (position_value / (self.portfolio_value + 1e-8))
        
        return total_profit_potential

    def _calculate_momentum(self, symbol: str) -> float:
        """Calculate price momentum for a symbol"""
        if self.current_step < 10:
            return 0.0
        
        try:
            current_price = self._get_current_price(symbol)
            past_idx = self.current_step + self.lookback_window - 10
            if 0 <= past_idx < len(self.dfs[symbol]):
                past_price = float(self.dfs[symbol].iloc[int(past_idx)]['close'])
                return (current_price - past_price) / (past_price + 1e-8)
        except (IndexError, KeyError, ValueError):
            pass
        return 0.0

    def _calculate_momentum_signals(self):
        """Calculate momentum signals for aggressive positioning"""
        for symbol in self.symbols:
            if self.current_step < self.momentum_lookback:
                self.momentum_signals[symbol] = 0.0
                continue
                
            try:
                current_price = self._get_current_price(symbol)
                past_idx = self.current_step + self.lookback_window - self.momentum_lookback
                if 0 <= past_idx < len(self.dfs[symbol]):
                    past_price = float(self.dfs[symbol].iloc[int(past_idx)]['close'])
                    # Simple price momentum
                    momentum = (current_price - past_price) / (past_price + 1e-8)
                    # Normalize momentum to [-1, 1] range
                    self.momentum_signals[symbol] = np.tanh(momentum * 10)
                else:
                    self.momentum_signals[symbol] = 0.0
            except (IndexError, KeyError, ValueError):
                self.momentum_signals[symbol] = 0.0
    
    def _calculate_trend_signals(self):
        """Calculate trend following signals"""
        for symbol in self.symbols:
            if self.current_step < 20:
                self.trend_signals[symbol] = 0.0
                continue
                
            try:
                # Use multiple timeframes for trend confirmation
                prices = []
                for offset in [5, 10, 20]:
                    idx = self.current_step + self.lookback_window - offset
                    if 0 <= idx < len(self.dfs[symbol]):
                        price = float(self.dfs[symbol].iloc[int(idx)]['close'])
                        prices.append(price)
                
                if len(prices) >= 3:
                    current_price = self._get_current_price(symbol)
                    # Weighted trend score
                    short_trend = (current_price - prices[0]) / (prices[0] + 1e-8)
                    medium_trend = (current_price - prices[1]) / (prices[1] + 1e-8)
                    long_trend = (current_price - prices[2]) / (prices[2] + 1e-8)
                    
                    # Combined trend signal with weights
                    trend_score = (short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2)
                    self.trend_signals[symbol] = np.clip(trend_score * 5, -1, 1)
                else:
                    self.trend_signals[symbol] = 0.0
            except (IndexError, KeyError, ValueError):
                self.trend_signals[symbol] = 0.0

    def _adaptive_normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Advanced adaptive normalization with outlier detection"""
        if not self.normalize_obs or len(features) == 0:
            return features
        
        # Robust scaling with adaptive thresholds
        median = np.median(features, axis=0, keepdims=True)
        mad = np.median(np.abs(features - median), axis=0, keepdims=True) + 1e-8
        
        # Adaptive outlier threshold based on feature distribution
        outlier_threshold = 3.0 * mad
        
        # Winsorize outliers
        features_clipped = np.clip(features, median - outlier_threshold, median + outlier_threshold)
        
        # Scale to [-1, 1] range
        feature_range = np.ptp(features_clipped, axis=0, keepdims=True) + 1e-8
        normalized = 2.0 * (features_clipped - np.min(features_clipped, axis=0, keepdims=True)) / feature_range - 1.0
        
        return normalized

    def _calculate_risk_metrics(self):
        """Calculate advanced risk metrics"""
        # Portfolio volatility
        if len(self.returns_history) >= 10:
            recent_returns = np.array(self.returns_history[-min(20, len(self.returns_history)):])
            self.portfolio_volatility = np.std(recent_returns, ddof=1) * np.sqrt(252) if recent_returns.size > 0 else 0.0
            self.rolling_volatility.append(self.portfolio_volatility)
        
        # Diversification ratio
        if len(self.weights_history) >= 5:
            current_weights = np.array([self.weights[s] for s in self.symbols])
            if np.sum(current_weights) > 0:
                # Simple diversification measure (1 - Herfindahl index)
                self.diversification_ratio = 1 - np.sum(current_weights ** 2)
        
        # Update rolling metrics
        self.rolling_sharpe.append(self._calculate_sharpe_ratio())
        self.rolling_max_drawdown.append(self.max_drawdown)

    def _calculate_dynamic_penalties(self) -> Dict[str, float]:
        """Calculate dynamic penalties based on market conditions"""
        penalties = {}
        
        # Volatility-adjusted penalties
        if len(self.rolling_volatility) > 0:
            current_vol = self.portfolio_volatility
            avg_vol = np.mean(self.rolling_volatility) if self.rolling_volatility else 0.1
            
            # Increase penalties in high volatility regimes
            vol_ratio = current_vol / (avg_vol + 1e-8)
            penalties['volatility_multiplier'] = min(3.0, max(0.5, vol_ratio))
        else:
            penalties['volatility_multiplier'] = 1.0
        
        # Drawdown penalty scaling
        if len(self.rolling_max_drawdown) > 0:
            avg_drawdown = np.mean(self.rolling_max_drawdown) if self.rolling_max_drawdown else 0.05
            if avg_drawdown > 0:
                drawdown_ratio = self.max_drawdown / (avg_drawdown + 1e-8)
                penalties['drawdown_multiplier'] = min(3.0, max(0.5, drawdown_ratio))
            else:
                penalties['drawdown_multiplier'] = 1.0
        else:
            penalties['drawdown_multiplier'] = 1.0
        
        return penalties

    def sample_action(self) -> np.ndarray:
        """Sample aggressive actions with momentum bias"""
        if self.use_attention_weights and any(abs(s) > 0.1 for s in self.momentum_signals.values()):
            # Momentum-biased sampling
            base_alpha = np.ones(self.num_assets)
            for i, symbol in enumerate(self.symbols):
                # Bias sampling towards assets with strong momentum
                momentum_strength = abs(self.momentum_signals[symbol])
                base_alpha[i] = 1.0 + momentum_strength * 2
            
            random_weights = self.np_random.dirichlet(base_alpha)
        else:
            # Standard sampling without negative weights
            random_weights = self.np_random.dirichlet(np.ones(self.num_assets))
        
        return random_weights.astype(np.float32)
        
    def _align_dataframes(self):
        """Align dataframes and calculate max steps"""
        min_length = min(len(df) for df in self.dfs.values())
        
        for symbol in self.symbols:
            self.dfs[symbol] = self.dfs[symbol].iloc[:min_length].reset_index(drop=True)
        
        self.max_steps = max(0, min_length - self.lookback_window - 1)
    
    def _get_aggressive_observation(self) -> np.ndarray:
        """Enhanced observation with momentum and trend signals"""
        obs_parts = []
        
        # Price and volume data
        for symbol in self.symbols:
            df = self.dfs[symbol]
            start_idx = self.current_step
            end_idx = self.current_step + self.lookback_window
            
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            
            window_data = df.iloc[start_idx:end_idx][self.feature_columns].values
            
            if window_data.shape[0] < self.lookback_window:
                rows_needed = self.lookback_window - window_data.shape[0]
                if window_data.size == 0:
                    window_data = np.zeros((self.lookback_window, len(self.feature_columns)), dtype=float)
                else:
                    pad = np.zeros((rows_needed, window_data.shape[1]), dtype=float)
                    window_data = np.vstack([pad, window_data])
            
            if self.normalize_obs and window_data.size > 0:
                window_data = self._adaptive_normalize_features(window_data)
            
            obs_parts.append(window_data.flatten())
        
        # Enhanced portfolio state without leverage metrics
        current_weights = np.array([self.weights[s] for s in self.symbols], dtype=np.float32)
        cash_ratio = self.cash_balance / (self.portfolio_value + 1e-8)
        portfolio_value_norm = self.portfolio_value / self.starting_balance
        current_drawdown = self.max_drawdown
        
        # Risk metrics
        if len(self.returns_history) > 1:
            recent_returns = np.array(self.returns_history[-min(15, len(self.returns_history)):])
            volatility = np.std(recent_returns, ddof=1) * np.sqrt(252) if recent_returns.size > 0 else 0.0
        else:
            volatility = 0.0
        
        sharpe_ratio = self._calculate_sharpe_ratio()
        total_return = (self.portfolio_value - self.starting_balance) / (self.starting_balance + 1e-8)
        benchmark_outperformance = self._calculate_benchmark_outperformance()
        future_profit_potential = self._calculate_advanced_future_profit()
        
        # Aggressive strategy signals
        momentum_signals = np.array([self.momentum_signals[s] for s in self.symbols])
        trend_signals = np.array([self.trend_signals[s] for s in self.symbols])
        
        # Concentration metrics
        concentration = np.sum(current_weights)
        top_holding_concentration = np.max(current_weights)
        
        # FIXED: Removed gross_exposure and net_exposure (2 scalars)
        portfolio_state = np.array([
            *current_weights,                    # num_assets elements
            cash_ratio,                          # 1
            portfolio_value_norm,                # 1  
            current_drawdown,                    # 1
            volatility,                          # 1
            sharpe_ratio,                        # 1
            total_return,                        # 1
            benchmark_outperformance,            # 1
            future_profit_potential,             # 1
            concentration,                       # 1
            top_holding_concentration,           # 1
            # Removed gross_exposure and net_exposure
            *momentum_signals,                   # num_assets elements
            *trend_signals                       # num_assets elements
        ], dtype=np.float32)
        
        obs_parts.append(portfolio_state)
        observation = np.concatenate(obs_parts)
        
        # Debug: Verify observation shape matches expected dimension
        expected_dim = (self.lookback_window * self.num_features_per_asset * self.num_assets + 
                       self.num_assets + 11 + self.num_assets + self.num_assets)  # Fixed: 11 instead of 13
        
        if observation.shape[0] != expected_dim:
            print(f"WARNING: Observation shape mismatch. Expected: {expected_dim}, Got: {observation.shape[0]}")
            # If there's a mismatch, pad or truncate to expected dimension
            if observation.shape[0] < expected_dim:
                # Pad with zeros
                padding = np.zeros(expected_dim - observation.shape[0], dtype=np.float32)
                observation = np.concatenate([observation, padding])
            else:
                # Truncate
                observation = observation[:expected_dim]
        
        return observation.astype(np.float32)
    
    def _calculate_value_at_risk(self, alpha: float = 0.05) -> float:
        """Calculate Value at Risk"""
        if len(self.returns_history) < 20:
            return 0.0
        
        returns = np.array(self.returns_history[-20:])
        return -np.percentile(returns, alpha * 100) if len(returns) > 0 else 0.0
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        total_value = self.cash_balance
        
        for symbol in self.symbols:
            if self.holdings[symbol] > 0:
                current_price = self._get_current_price(symbol)
                total_value += self.holdings[symbol] * current_price
        
        return total_value
    
    def _rebalance_portfolio_aggressive(self, target_weights: np.ndarray):
        """Aggressive rebalancing without short positions"""
        # Apply aggressive risk limits
        target_weights = self._apply_aggressive_risk_limits(target_weights)
        
        # Normalize weights to sum to 1 (no leverage)
        weight_sum = np.sum(target_weights)
        if weight_sum > 1.0 or weight_sum < 0.99:  # Normalize if not approximately 1.0
            target_weights = target_weights / (weight_sum + 1e-8)
        
        current_value = self._calculate_portfolio_value()
        fees_paid = 0.0
        
        # Calculate target values - only positive values allowed
        target_values = {symbol: current_value * target_weights[i] 
                        for i, symbol in enumerate(self.symbols)}
        
        current_prices = {symbol: self._get_current_price(symbol) for symbol in self.symbols}
        current_values = {symbol: self.holdings[symbol] * current_prices[symbol]
                         for symbol in self.symbols}
        
        # Calculate turnover
        turnover = 0.0
        for symbol in self.symbols:
            current_weight = current_values[symbol] / (current_value + 1e-8)
            target_weight = target_weights[self.symbols.index(symbol)]
            turnover += abs(current_weight - target_weight)
        
        self.turnover_history.append(turnover)
        
        # Execute trades - only long positions
        for symbol in self.symbols:
            current_val = current_values[symbol]
            target_val = target_values[symbol]
            
            # Selling
            if current_val > target_val:
                value_change = current_val - target_val
                if abs(value_change) > 1e-6:
                    units_to_sell = value_change / (current_prices[symbol] + 1e-8)
                    sell_value = units_to_sell * current_prices[symbol]
                    fee = sell_value * self.fee_rate
                    self.cash_balance += sell_value - fee
                    self.holdings[symbol] -= units_to_sell
                    fees_paid += fee
                    self.trades_executed += 1
        
        for symbol in self.symbols:
            current_val = self.holdings[symbol] * current_prices[symbol]
            target_val = target_values[symbol]
            
            # Buying
            if target_val > current_val:
                value_change = target_val - current_val
                if abs(value_change) > 1e-6:
                    units_to_buy = value_change / (current_prices[symbol] + 1e-8)
                    buy_value = units_to_buy * current_prices[symbol]
                    fee = buy_value * self.fee_rate
                    total_cost = buy_value + fee
                    
                    if self.cash_balance >= total_cost:
                        self.cash_balance -= total_cost
                        self.holdings[symbol] += units_to_buy
                        fees_paid += fee
                        self.trades_executed += 1
        
        # Update weights
        final_value = self._calculate_portfolio_value()
        for i, symbol in enumerate(self.symbols):
            asset_value = self.holdings[symbol] * current_prices[symbol]
            self.weights[symbol] = asset_value / (final_value + 1e-8)
        
        self.total_fees_paid += fees_paid
        self.weights_history.append([self.weights[s] for s in self.symbols])
        
        return fees_paid
    
    def _apply_aggressive_risk_limits(self, weights: np.ndarray) -> np.ndarray:
        """Apply aggressive risk management without leverage"""
        # Ensure no negative weights
        weights = np.maximum(weights, 0.0)
        
        # Apply concentration limit
        max_single_asset = self.concentration_limit
        weights = np.minimum(weights, max_single_asset)
        
        # Ensure some minimum diversification but allow high concentration
        if np.max(weights) > 0.8:  # If any position > 80%
            # Allow but ensure at least 2 assets have some exposure
            non_zero_positions = np.sum(weights > 0.01)
            if non_zero_positions < 2:
                # Find second best opportunity
                sorted_weights = np.argsort(weights)[::-1]
                second_asset = sorted_weights[1]
                weights[second_asset] = 0.05  # Minimum 5% allocation
        
        return weights
    
    def _calculate_aggressive_reward(self) -> float:
        """Highly aggressive reward function focused on outperformance"""
        current_value = self._calculate_portfolio_value()
        self.value_at_action.append(current_value)
        
        # Update risk metrics
        self._calculate_risk_metrics()
        
        # Update peak and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / (self.peak_value + 1e-8)
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Get the latest period return
        period_return = self.returns_history[-1] if self.returns_history else 0.0
        
        # Aggressive reward components
        return_reward = period_return * 50 * self.reward_components['returns']
        
        # Sharpe ratio component - minimal weight
        sharpe_reward = 0.0
        if len(self.returns_history) >= 5:
            sharpe_ratio = self._calculate_sharpe_ratio()
            sharpe_reward = sharpe_ratio * 0.01 * self.reward_components['sharpe']
        
        # Minimal drawdown penalty
        drawdown_penalty = (-self.max_drawdown_penalty * drawdown * 10 *
                           self.reward_components['drawdown'])
        
        # Massive benchmark outperformance reward
        benchmark_reward = 0.0
        benchmark_outperformance = self._calculate_benchmark_outperformance()
        benchmark_reward = (benchmark_outperformance * self.benchmark_weight * 100 *
                          self.reward_components['benchmark'])
        
        # High future profit potential reward
        future_reward = 0.0
        future_profit_potential = self._calculate_advanced_future_profit()
        future_reward = (future_profit_potential * self.future_profit_weight * 50 *
                        self.reward_components['future'])
        
        # Momentum-based reward
        momentum_reward = 0.0
        for symbol in self.symbols:
            if self.weights[symbol] > 0.01:
                # Reward aligning with momentum
                position_return = period_return * self.weights[symbol]
                momentum_alignment = position_return * self.momentum_signals[symbol]
                momentum_reward += momentum_alignment * self.reward_components['momentum']
        
        # Trend-following reward
        trend_reward = 0.0
        if self.trend_following:
            for symbol in self.symbols:
                if self.weights[symbol] > 0.01:
                    # Reward trend-following behavior
                    trend_alignment = self.weights[symbol] * self.trend_signals[symbol]
                    trend_reward += trend_alignment * self.reward_components['trend']
        
        # Concentration reward (positive for aggressive strategy)
        current_weights = np.array([self.weights[s] for s in self.symbols])
        concentration = np.sum(current_weights)
        concentration_reward = (concentration - 1.0) * 2 * self.reward_components['diversification']
        
        # Reduced turnover penalty
        turnover_penalty = 0.0
        if self.turnover_history:
            avg_turnover = np.mean(self.turnover_history[-3:]) if len(self.turnover_history) >= 3 else 0.0
            turnover_penalty = -avg_turnover * 0.5
        
        # Combine all reward components
        total_reward = (return_reward + sharpe_reward + drawdown_penalty + 
                       benchmark_reward + future_reward + momentum_reward + 
                       trend_reward + concentration_reward + turnover_penalty)
        
        # Additional aggressive scaling
        total_reward = total_reward * 2.0
        
        return total_reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute aggressive environment step"""
        action = np.asarray(action, dtype=float).flatten()
        if action.size != self.num_assets:
            raise ValueError(f"Action must have length {self.num_assets}, got {action.size}")
        
        # Calculate momentum and trend signals BEFORE rebalancing
        self._calculate_momentum_signals()
        self._calculate_trend_signals()
        
        # Store previous portfolio value
        prev_value = self._calculate_portfolio_value()
        
        if self.steps_since_rebalance >= self.rebalance_frequency:
            fees = self._rebalance_portfolio_aggressive(action)
            self.steps_since_rebalance = 0
        else:
            fees = 0.0
        
        self.current_step += 1
        self.steps_since_rebalance += 1
        
        # Calculate current value AFTER rebalancing
        current_value = self._calculate_portfolio_value()
        self.portfolio_value = current_value
        
        # Calculate period return
        period_return = (current_value - prev_value) / (prev_value + 1e-8)
        self.returns_history.append(period_return)
        
        # Update portfolio history
        self.portfolio_history.append(current_value)
        
        self._update_benchmark_value()
        
        reward = self._calculate_aggressive_reward()
        
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Tolerant termination conditions
        if current_value < self.starting_balance * 0.1:
            terminated = True
            reward -= 2
        elif self.max_drawdown > 0.7:
            terminated = True
            reward -= 5
        
        observation = self._get_aggressive_observation()
        
        info = {
            'portfolio_value': current_value,
            'cash_balance': self.cash_balance,
            'holdings': dict(self.holdings),
            'weights': dict(self.weights),
            'total_return': (current_value - self.starting_balance) / (self.starting_balance + 1e-8),
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'trades_executed': self.trades_executed,
            'total_fees_paid': self.total_fees_paid,
            'fees_this_step': fees,
            'benchmark_value': self.benchmark_value,
            'benchmark_outperformance': self._calculate_benchmark_outperformance(),
            'future_profit_potential': self._calculate_advanced_future_profit(),
            'portfolio_volatility': self.portfolio_volatility,
            'diversification_ratio': self.diversification_ratio,
            'value_at_risk': self._calculate_value_at_risk(0.05),
            'momentum_signals': dict(self.momentum_signals),
            'trend_signals': dict(self.trend_signals)
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(self.returns_history) < 2:
            return 0.0

        rets = np.array(self.returns_history)
        ppy = int(self.periods_per_year) if (self.periods_per_year and self.periods_per_year > 0) else 252
        mean_return = float(np.mean(rets))
        std_return = float(np.std(rets, ddof=1))
        if std_return < 1e-8:
            return 0.0
        period_rf = (self.risk_free_rate / ppy) if ppy > 0 else 0.0
        sharpe = (mean_return - period_rf) / std_return * np.sqrt(ppy)
        
        if self.clip_metrics:
            return float(np.clip(sharpe, -10.0, 10.0))
        else:
            return float(sharpe)
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment with enhanced initialization"""
        super().reset(seed=seed)
        
        # Re-initialize RNG with new seed
        self.np_random = np.random.default_rng(seed)
        
        self._initialize_state()
        
        if options and 'start_step' in options:
            self.current_step = options['start_step']
        else:
            max_start = max(0, self.max_steps - 100)
            self.current_step = self.np_random.integers(0, max(1, max_start))
        
        self._initialize_benchmark_portfolio()
        
        observation = self._get_aggressive_observation()
        info = {
            'portfolio_value': self.starting_balance,
            'start_step': self.current_step,
            'benchmark_value': self.benchmark_value
        }
        
        return observation, info
    
    def render(self, mode='human'):
        """Enhanced rendering with risk metrics"""
        if mode == 'human':
            current_value = self._calculate_portfolio_value()
            total_return = (current_value - self.starting_balance) / (self.starting_balance + 1e-8)
            benchmark_return = (self.benchmark_value - self.starting_balance) / (self.starting_balance + 1e-8)
            outperformance = self._calculate_benchmark_outperformance()
            
            print(f"\n{'='*80}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Portfolio Value: ${current_value:.2f} (Return: {total_return*100:.2f}%)")
            print(f"Benchmark Value: ${self.benchmark_value:.2f} (Return: {benchmark_return*100:.2f}%)")
            print(f"Outperformance: {outperformance*100:.2f}%")
            print(f"Future Profit Potential: {self._calculate_advanced_future_profit()*100:.2f}%")
            print(f"Cash Balance: ${self.cash_balance:.2f}")
            print(f"\nRisk Metrics:")
            print(f"  Volatility: {self.portfolio_volatility*100:.2f}%")
            print(f"  Max Drawdown: {self.max_drawdown*100:.2f}%")
            print(f"  Sharpe Ratio: {self._calculate_sharpe_ratio():.3f}")
            print(f"  VaR (95%): {self._calculate_value_at_risk(0.05)*100:.2f}%")
            print(f"  Diversification: {self.diversification_ratio*100:.1f}%")
            print(f"\nHoldings:")
            for symbol in self.symbols:
                amount = self.holdings[symbol]
                weight = self.weights[symbol]
                price = self._get_current_price(symbol)
                value = amount * price
                print(f"  {symbol}: {amount:.6f} units @ ${price:.2f} = ${value:.2f} ({weight*100:.1f}%)")
            print(f"\nTrading Stats:")
            print(f"  Trades Executed: {self.trades_executed}")
            print(f"  Total Fees: ${self.total_fees_paid:.2f}")
            print(f"  Turnover: {np.mean(self.turnover_history[-5:])*100:.1f}%" if self.turnover_history else "  Turnover: 0.0%")
            print(f"{'='*80}\n")
    
    def get_final_metrics(self) -> Dict:
        """Comprehensive final performance metrics"""
        current_value = self._calculate_portfolio_value()
        total_return = (current_value - self.starting_balance) / (self.starting_balance + 1e-8)
        benchmark_return = (self.benchmark_value - self.starting_balance) / (self.starting_balance + 1e-8)
        outperformance = self._calculate_benchmark_outperformance()
        
        if len(self.returns_history) > 0:
            returns_array = np.array(self.returns_history)
            win_rate = np.sum(returns_array > 0) / len(returns_array) if len(returns_array) > 0 else 0
            avg_win = np.mean(returns_array[returns_array > 0]) if np.any(returns_array > 0) else 0
            avg_loss = np.mean(returns_array[returns_array < 0]) if np.any(returns_array < 0) else 0
            profit_factor = abs(avg_win / (avg_loss + 1e-8)) if avg_loss != 0 else float('inf')
            
            # Risk-adjusted metrics
            sortino_ratio = self._calculate_sortino_ratio()
            calmar_ratio = total_return / (self.max_drawdown + 1e-8) if self.max_drawdown > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            sortino_ratio = 0
            calmar_ratio = 0
        
        return {
            'final_portfolio_value': current_value,
            'final_benchmark_value': self.benchmark_value,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'outperformance': outperformance,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades_executed': self.trades_executed,
            'total_fees_paid': self.total_fees_paid,
            'fee_percentage': self.total_fees_paid / (self.starting_balance + 1e-8),
            'portfolio_volatility': self.portfolio_volatility,
            'diversification_ratio': self.diversification_ratio,
            'value_at_risk': self._calculate_value_at_risk(0.05)
        }
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return)"""
        if len(self.returns_history) < 2:
            return 0.0

        rets = np.array(self.returns_history)
        ppy = int(self.periods_per_year) if (self.periods_per_year and self.periods_per_year > 0) else 252
        mean_return = float(np.mean(rets))
        downside = rets[rets < 0.0]
        if downside.size == 0:
            return 0.0
        downside_std = float(np.std(downside, ddof=1))
        if downside_std < 1e-8:
            return 0.0
        period_rf = (self.risk_free_rate / ppy) if ppy > 0 else 0.0
        sortino = (mean_return - period_rf) / downside_std * np.sqrt(ppy)
        
        if self.clip_metrics:
            return float(np.clip(sortino, -10.0, 10.0))
        else:
            return float(sortino)