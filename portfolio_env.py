import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MultiCurrencyPortfolioEnv(gym.Env):
    """
    Multi-Currency Portfolio Rebalancing Environment for Reinforcement Learning
    
    The agent receives observations from multiple cryptocurrencies and outputs
    portfolio weights (percentages) for rebalancing. Rewards include Sharpe ratio,
    returns, drawdown penalties, and other risk-adjusted metrics.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        dfs: Dict[str, pd.DataFrame],
        symbols: List[str],
        starting_balance: float = 10000.0,
        fee_rate: float = 0.001,
        lookback_window: int = 50,
        rebalance_frequency: int = 12,  # Rebalance every N steps (e.g., every hour for 5min data)
        risk_free_rate: float = 0.02,  # Annual risk-free rate for Sharpe ratio
        max_drawdown_penalty: float = 2.0,  # Penalty multiplier for drawdown
        normalize_obs: bool = True
    ):
        """
        Initialize multi-currency portfolio environment
        
        Args:
            dfs: Dictionary of {symbol: DataFrame} with OHLCV + technical indicators
            symbols: List of cryptocurrency symbols to trade
            starting_balance: Initial portfolio value in USD
            fee_rate: Transaction fee rate (0.001 = 0.1%)
            lookback_window: Number of timesteps in observation window
            rebalance_frequency: Steps between rebalancing actions
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            max_drawdown_penalty: Penalty weight for maximum drawdown
            normalize_obs: Whether to normalize observations
        """
        super(MultiCurrencyPortfolioEnv, self).__init__()
        
        self.dfs = dfs
        self.symbols = symbols
        self.num_assets = len(symbols)
        self.starting_balance = starting_balance
        self.fee_rate = fee_rate
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_penalty = max_drawdown_penalty
        self.normalize_obs = normalize_obs
        
        # Find common date range across all symbols
        self._align_dataframes()
        
        # Select numeric feature columns (same across all symbols)
        first_df = self.dfs[symbols[0]]
        self.feature_columns = [col for col in first_df.columns 
                               if first_df[col].dtype in ['float64', 'int64']]
        self.num_features_per_asset = len(self.feature_columns)
        
        # Action space: portfolio weights for each asset (must sum to 1)
        # Each weight is between 0 and 1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_assets,),
            dtype=np.float32
        )
        
        # Observation space: features for all assets + portfolio state
        # Features: lookback_window * num_features * num_assets
        # Portfolio state: current weights (num_assets) + cash ratio (1) + portfolio metrics (3)
        obs_dim = (self.lookback_window * self.num_features_per_asset * self.num_assets + 
                   self.num_assets + 4)  # +4 for cash_ratio, total_value_norm, drawdown, volatility
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Trading state
        self.current_step = 0
        self.portfolio_value = starting_balance
        self.cash_balance = starting_balance
        self.holdings = {symbol: 0.0 for symbol in symbols}  # Amount of each crypto held
        self.weights = {symbol: 0.0 for symbol in symbols}  # Current portfolio weights
        
        # Performance tracking
        self.portfolio_history = []
        self.returns_history = []
        self.weights_history = []
        self.peak_value = starting_balance
        self.max_drawdown = 0.0
        self.trades_executed = 0
        self.total_fees_paid = 0.0
        
        # Steps since last rebalance
        self.steps_since_rebalance = 0
    
    def sample_action(self) -> np.ndarray:
        """
        Sample a valid portfolio allocation (weights that sum to 1.0)
        
        Returns:
            Random portfolio weights that sum to 1.0
        """
        # Generate random weights using Dirichlet distribution
        # This ensures they sum to 1.0
        random_weights = self.np_random.dirichlet(np.ones(self.num_assets))
        return random_weights.astype(np.float32)
        
    def _align_dataframes(self):
        """Ensure all dataframes have the same length and aligned timestamps"""
        min_length = min(len(df) for df in self.dfs.values())
        
        for symbol in self.symbols:
            self.dfs[symbol] = self.dfs[symbol].iloc[:min_length].reset_index(drop=True)
        
        self.max_steps = min_length - self.lookback_window - 1
        
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation including all asset features and portfolio state
        
        Returns:
            Flattened observation vector
        """
        obs_parts = []
        
        # Get features for all assets over lookback window
        for symbol in self.symbols:
            df = self.dfs[symbol]
            start_idx = self.current_step
            end_idx = self.current_step + self.lookback_window
            
            window_data = df.loc[start_idx:end_idx-1, self.feature_columns].values
            
            # Normalize if requested
            if self.normalize_obs and len(window_data) > 0:
                mean = np.mean(window_data, axis=0, keepdims=True)
                std = np.std(window_data, axis=0, keepdims=True) + 1e-8
                window_data = (window_data - mean) / std
            
            obs_parts.append(window_data.flatten())
        
        # Portfolio state
        current_weights = np.array([self.weights[s] for s in self.symbols], dtype=np.float32)
        cash_ratio = self.cash_balance / (self.portfolio_value + 1e-8)
        portfolio_value_norm = self.portfolio_value / self.starting_balance
        current_drawdown = self.max_drawdown
        
        # Calculate recent volatility
        if len(self.returns_history) > 1:
            recent_returns = self.returns_history[-min(20, len(self.returns_history)):]
            volatility = np.std(recent_returns)
        else:
            volatility = 0.0
        
        portfolio_state = np.array([
            *current_weights,
            cash_ratio,
            portfolio_value_norm,
            current_drawdown,
            volatility
        ], dtype=np.float32)
        
        # Concatenate all parts
        obs_parts.append(portfolio_state)
        observation = np.concatenate(obs_parts)
        
        return observation.astype(np.float32)
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current total portfolio value"""
        total_value = self.cash_balance
        
        for symbol in self.symbols:
            if self.holdings[symbol] > 0:
                current_price = self.dfs[symbol].loc[self.current_step + self.lookback_window, 'close']
                total_value += self.holdings[symbol] * current_price
        
        return total_value
    
    def _rebalance_portfolio(self, target_weights: np.ndarray):
        """
        Rebalance portfolio to target weights
        
        Args:
            target_weights: Desired weight for each asset (must sum to ~1.0)
        """
        # Normalize weights to sum to 1
        target_weights = np.clip(target_weights, 0, 1)
        weight_sum = np.sum(target_weights)
        if weight_sum > 0:
            target_weights = target_weights / weight_sum
        else:
            # If all weights are 0, distribute equally
            target_weights = np.ones(self.num_assets) / self.num_assets
        
        current_value = self._calculate_portfolio_value()
        fees_paid = 0.0
        
        # Calculate target values for each asset
        target_values = {symbol: current_value * target_weights[i] 
                        for i, symbol in enumerate(self.symbols)}
        
        # Get current prices
        current_prices = {symbol: self.dfs[symbol].loc[self.current_step + self.lookback_window, 'close']
                         for symbol in self.symbols}
        
        # Calculate current values
        current_values = {symbol: self.holdings[symbol] * current_prices[symbol]
                         for symbol in self.symbols}
        
        # First, sell assets that need to be reduced
        for symbol in self.symbols:
            current_val = current_values[symbol]
            target_val = target_values[symbol]
            
            if current_val > target_val:
                # Sell excess
                amount_to_sell = (current_val - target_val) / current_prices[symbol]
                if amount_to_sell > 0:
                    sell_value = amount_to_sell * current_prices[symbol]
                    fee = sell_value * self.fee_rate
                    self.cash_balance += sell_value - fee
                    self.holdings[symbol] -= amount_to_sell
                    fees_paid += fee
                    self.trades_executed += 1
        
        # Then, buy assets that need to be increased
        for symbol in self.symbols:
            current_val = self.holdings[symbol] * current_prices[symbol]
            target_val = target_values[symbol]
            
            if target_val > current_val:
                # Buy more
                amount_to_buy_value = target_val - current_val
                fee = amount_to_buy_value * self.fee_rate
                total_cost = amount_to_buy_value + fee
                
                if self.cash_balance >= total_cost:
                    amount_to_buy = amount_to_buy_value / current_prices[symbol]
                    self.cash_balance -= total_cost
                    self.holdings[symbol] += amount_to_buy
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
    
    def _calculate_reward(self) -> float:
        """
        Calculate complex reward based on multiple factors:
        - Portfolio returns
        - Sharpe ratio
        - Maximum drawdown penalty
        - Volatility penalty
        - Fee penalty
        """
        current_value = self._calculate_portfolio_value()
        
        # Calculate return
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1]
            period_return = (current_value - prev_value) / (prev_value + 1e-8)
        else:
            period_return = 0.0
        
        self.returns_history.append(period_return)
        
        # Update peak and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / (self.peak_value + 1e-8)
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (use recent returns)
        reward = 0.0
        
        if len(self.returns_history) >= 2:
            recent_returns = np.array(self.returns_history[-min(100, len(self.returns_history)):])
            
            # Annualized return (assuming 5-min intervals: 12 per hour, 24*12=288 per day, ~105120 per year)
            periods_per_year = 105120 / self.rebalance_frequency
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns) + 1e-8
            
            # Sharpe ratio component
            sharpe_ratio = (mean_return - self.risk_free_rate / periods_per_year) / std_return
            
            # Reward components
            return_reward = period_return * 100  # Scale returns
            sharpe_reward = sharpe_ratio * 2  # Sharpe ratio component
            drawdown_penalty = -self.max_drawdown_penalty * drawdown * 10  # Penalize drawdown
            volatility_penalty = -std_return * 5  # Penalize high volatility
            
            # Combine rewards
            reward = return_reward + sharpe_reward + drawdown_penalty + volatility_penalty
        else:
            # Initial steps: just use simple return
            reward = period_return * 100
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Portfolio weights for each asset
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Only rebalance at specified frequency
        if self.steps_since_rebalance >= self.rebalance_frequency:
            fees = self._rebalance_portfolio(action)
            self.steps_since_rebalance = 0
        else:
            fees = 0.0
        
        # Move to next step
        self.current_step += 1
        self.steps_since_rebalance += 1
        
        # Update portfolio value
        current_value = self._calculate_portfolio_value()
        self.portfolio_value = current_value
        self.portfolio_history.append(current_value)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Episode ends if portfolio value drops too much
        if current_value < self.starting_balance * 0.1:  # Lost 90%
            terminated = True
            reward -= 100  # Large penalty for bankruptcy
        
        # Get next observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {
            'portfolio_value': current_value,
            'cash_balance': self.cash_balance,
            'holdings': dict(self.holdings),
            'weights': dict(self.weights),
            'total_return': (current_value - self.starting_balance) / self.starting_balance,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'trades_executed': self.trades_executed,
            'total_fees_paid': self.total_fees_paid,
            'fees_this_step': fees
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from returns history"""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        periods_per_year = 105120 / self.rebalance_frequency
        
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8
        
        excess_return = mean_return - (self.risk_free_rate / periods_per_year)
        sharpe = excess_return / std_return * np.sqrt(periods_per_year)
        
        return sharpe
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset to random starting point if requested
        if options and 'start_step' in options:
            self.current_step = options['start_step']
        else:
            # Random start within valid range
            max_start = self.max_steps - 100  # Leave room for episode
            self.current_step = self.np_random.integers(0, max(1, max_start))
        
        # Reset trading state
        self.portfolio_value = self.starting_balance
        self.cash_balance = self.starting_balance
        self.holdings = {symbol: 0.0 for symbol in self.symbols}
        self.weights = {symbol: 0.0 for symbol in self.symbols}
        
        # Reset performance tracking
        self.portfolio_history = [self.starting_balance]
        self.returns_history = []
        self.weights_history = []
        self.peak_value = self.starting_balance
        self.max_drawdown = 0.0
        self.trades_executed = 0
        self.total_fees_paid = 0.0
        self.steps_since_rebalance = self.rebalance_frequency  # Trigger immediate rebalance
        
        observation = self._get_observation()
        info = {
            'portfolio_value': self.starting_balance,
            'start_step': self.current_step
        }
        
        return observation, info
    
    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            current_value = self._calculate_portfolio_value()
            total_return = (current_value - self.starting_balance) / self.starting_balance
            
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Portfolio Value: ${current_value:.2f} (Return: {total_return*100:.2f}%)")
            print(f"Cash Balance: ${self.cash_balance:.2f}")
            print(f"\nHoldings:")
            for symbol in self.symbols:
                amount = self.holdings[symbol]
                weight = self.weights[symbol]
                price = self.dfs[symbol].loc[self.current_step + self.lookback_window, 'close']
                value = amount * price
                print(f"  {symbol}: {amount:.6f} units @ ${price:.2f} = ${value:.2f} ({weight*100:.1f}%)")
            print(f"\nPerformance Metrics:")
            print(f"  Max Drawdown: {self.max_drawdown*100:.2f}%")
            print(f"  Sharpe Ratio: {self._calculate_sharpe_ratio():.3f}")
            print(f"  Trades Executed: {self.trades_executed}")
            print(f"  Total Fees: ${self.total_fees_paid:.2f}")
            print(f"{'='*60}\n")
    
    def get_final_metrics(self) -> Dict:
        """Get final performance metrics for the episode"""
        current_value = self._calculate_portfolio_value()
        total_return = (current_value - self.starting_balance) / self.starting_balance
        
        # Calculate additional metrics
        if len(self.returns_history) > 0:
            returns_array = np.array(self.returns_history)
            win_rate = np.sum(returns_array > 0) / len(returns_array)
            avg_win = np.mean(returns_array[returns_array > 0]) if np.any(returns_array > 0) else 0
            avg_loss = np.mean(returns_array[returns_array < 0]) if np.any(returns_array < 0) else 0
            profit_factor = abs(avg_win / (avg_loss + 1e-8))
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'final_portfolio_value': current_value,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades_executed': self.trades_executed,
            'total_fees_paid': self.total_fees_paid,
            'fee_percentage': self.total_fees_paid / self.starting_balance
        }
