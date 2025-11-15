import numpy as np
import pandas as pd
import pickle
from portfolio_env import PortfolioEnv
from scipy.optimize import minimize
import os
from multiprocessing import Pool, cpu_count
import gc


def _generate_batch_worker(params):
    """
    Worker function for parallel batch generation.
    Creates its own environment to avoid pickling issues.
    
    Args:
        params: Tuple of (batch_idx, num_batches, samples_per_batch, start_idx, method, gamma)
    
    Returns:
        Generated batch dictionary
    """
    batch_idx, num_batches, samples_per_batch, start_idx, method, gamma = params
    
    print(f"\n=== Batch {batch_idx+1}/{num_batches} (Method: {method}) ===")
    
    try:
        # Create environment for this worker
        env = PortfolioEnv(max_records=100_000)
        
        # Create generator with 1-step lookforward (aligned with new reward function)
        generator = SyntheticDataGenerator(env, lookforward_steps=1)
        
        # Generate batch
        batch = generator.generate_synthetic_batch(
            num_samples=samples_per_batch,
            start_idx=start_idx,
            method=method,
            gamma=gamma
        )
        
        # Clean up to free memory
        del env
        del generator
        gc.collect()
        
        return batch
        
    except Exception as e:
        print(f"Error generating batch {batch_idx+1}: {e}")
        return None


class SyntheticDataGenerator:
    """
    Generate synthetic training data with optimal portfolio allocations
    calculated in advance using historical data.
    
    Now aligned with the new reward function that optimizes for 1-step ahead returns.
    """
    
    def __init__(self, env, lookforward_steps=1):
        self.env = env
        self.lookforward_steps = lookforward_steps
        self.num_assets = len(env.asset_names)
        
    def calculate_optimal_portfolio(self, current_idx, method='sharpe'):
        """
        Calculate optimal portfolio allocation based on future returns.
        
        Args:
            current_idx: Current position in the data
            method: 'sharpe' (risk-adjusted), 'returns' (max return), or 'kelly' (Kelly criterion)
        
        Returns:
            Optimal portfolio weights as numpy array
        """
        if current_idx + self.lookforward_steps >= len(self.env.df):
            # Not enough future data, use equal weighting
            return np.ones(self.num_assets) / self.num_assets
        
        # Get future price changes for each asset
        future_returns = []
        for symbol in self.env.df.keys():
            current_price = self.env.df[symbol].iloc[current_idx]['close']
            future_prices = self.env.df[symbol].iloc[current_idx+1:current_idx+self.lookforward_steps+1]['close'].values
            
            if len(future_prices) == 0:
                future_returns.append(np.zeros(self.lookforward_steps))
            else:
                returns = (future_prices - current_price) / current_price
                # Pad if necessary
                if len(returns) < self.lookforward_steps:
                    returns = np.pad(returns, (0, self.lookforward_steps - len(returns)), 'edge')
                future_returns.append(returns)
        
        future_returns = np.array(future_returns)  # Shape: (num_assets, lookforward_steps)
        
        if method == 'sharpe':
            return self._optimize_sharpe(future_returns)
        elif method == 'returns':
            return self._optimize_returns(future_returns)
        elif method == 'kelly':
            return self._optimize_kelly(future_returns)
        elif method == 'max_profit':
            return self._optimize_max_profit(future_returns)
        elif method == 'downside_protection':
            return self._optimize_downside_protection(future_returns)
        elif method == 'stable_growth':
            return self._optimize_stable_growth(future_returns)
        elif method == 'minimize_rebalancing':
            return self._optimize_minimize_rebalancing(future_returns, current_idx)
        elif method == 'cost_aware_optimization':
            return self._optimize_cost_aware(future_returns, current_idx)
        elif method == 'winner_counting':
            return self._optimize_winner_counting(future_returns)
        elif method == 'calmar_ratio':
            return self._optimize_calmar_ratio(future_returns)
        elif method == 'omega_ratio':
            return self._optimize_omega_ratio(future_returns)
        elif method == 'gradient_ascent':
            return self._optimize_gradient_ascent(future_returns)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_sharpe(self, future_returns):
        """Optimize for maximum Sharpe ratio (risk-adjusted returns)"""
        mean_returns = np.mean(future_returns, axis=1)
        cov_matrix = np.cov(future_returns)
        
        # Add small value to diagonal for numerical stability
        cov_matrix += np.eye(len(cov_matrix)) * 1e-8
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Avoid division by zero
            if portfolio_std < 1e-10:
                return 1e10
            sharpe = portfolio_return / portfolio_std
            return -sharpe  # Negative because we minimize
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        initial_weights = np.ones(self.num_assets) / self.num_assets
        
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 128}
        )
        
        if result.success:
            weights = result.x
            # Ensure weights sum to 1 and are non-negative
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
            return weights
        else:
            # Fallback to equal weighting
            return np.ones(self.num_assets) / self.num_assets
    
    def _optimize_returns(self, future_returns):
        """Optimize for maximum expected returns (no risk consideration)"""
        mean_returns = np.mean(future_returns, axis=1)
        
        # Put all weight in the asset with highest expected return
        # But add some diversification to avoid single asset concentration
        sorted_indices = np.argsort(mean_returns)[::-1]
        
        weights = np.zeros(self.num_assets)
        
        # Top 3 assets get weighted allocation
        if mean_returns[sorted_indices[0]] > 0:
            weights[sorted_indices[0]] = 0.6
            if self.num_assets > 1 and mean_returns[sorted_indices[1]] > 0:
                weights[sorted_indices[1]] = 0.25
            if self.num_assets > 2 and mean_returns[sorted_indices[2]] > 0:
                weights[sorted_indices[2]] = 0.15
        
        # If all returns are negative, equal weighting
        if weights.sum() == 0:
            weights = np.ones(self.num_assets) / self.num_assets
        else:
            weights = weights / weights.sum()
        
        return weights
    
    def _optimize_kelly(self, future_returns):
        """Optimize using Kelly Criterion for growth rate"""
        mean_returns = np.mean(future_returns, axis=1)
        cov_matrix = np.cov(future_returns)
        
        # Add small value to diagonal for numerical stability
        cov_matrix += np.eye(len(cov_matrix)) * 1e-8
        
        try:
            # Kelly optimal weights: f = C^-1 * m
            # where C is covariance matrix and m is mean returns
            inv_cov = np.linalg.inv(cov_matrix)
            kelly_weights = np.dot(inv_cov, mean_returns)
            
            # Normalize to [0, 1] and sum to 1
            kelly_weights = np.clip(kelly_weights, 0, None)  # No negative weights
            if kelly_weights.sum() > 0:
                kelly_weights = kelly_weights / kelly_weights.sum()
            else:
                kelly_weights = np.ones(self.num_assets) / self.num_assets
            
            # Apply leverage constraint (max 1.0 total allocation)
            if kelly_weights.sum() > 1.0:
                kelly_weights = kelly_weights / kelly_weights.sum()
            
            return kelly_weights
        except np.linalg.LinAlgError:
            # If covariance matrix is singular, fall back to equal weighting
            return np.ones(self.num_assets) / self.num_assets
    
    def _optimize_max_profit(self, future_returns):
        """Optimize for maximum absolute profit (aggressive strategy)"""
        mean_returns = np.mean(future_returns, axis=1)
        
        # Calculate total expected return for each asset
        total_returns = np.sum(future_returns, axis=1)
        
        # Sort by total expected profit
        sorted_indices = np.argsort(total_returns)[::-1]
        
        weights = np.zeros(self.num_assets)
        
        # Concentrate heavily on top performer, but diversify slightly
        if total_returns[sorted_indices[0]] > 0:
            weights[sorted_indices[0]] = 0.80  # 80% to best asset
            if self.num_assets > 1 and total_returns[sorted_indices[1]] > 0:
                weights[sorted_indices[1]] = 0.15  # 15% to second best
            if self.num_assets > 2 and total_returns[sorted_indices[2]] > 0:
                weights[sorted_indices[2]] = 0.05  # 5% to third best
        
        # If all returns are negative, put everything in least negative
        if weights.sum() == 0:
            least_negative_idx = sorted_indices[0]
            weights[least_negative_idx] = 1.0
        else:
            weights = weights / weights.sum()
        
        return weights
    
    def _optimize_downside_protection(self, future_returns):
        """Optimize to minimize downside risk and protect against losses"""
        mean_returns = np.mean(future_returns, axis=1)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = np.where(future_returns < 0, future_returns, 0)
        downside_std = np.std(downside_returns, axis=1)
        
        # Calculate sortino-like ratio (return / downside risk)
        sortino_scores = np.zeros(self.num_assets)
        for i in range(self.num_assets):
            if downside_std[i] > 1e-10:
                sortino_scores[i] = mean_returns[i] / downside_std[i]
            else:
                # If no downside risk, use mean return as score
                sortino_scores[i] = mean_returns[i] * 1000
        
        # Prefer assets with positive returns and low downside risk
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_downside = np.sqrt(np.dot(weights**2, downside_std**2))
            
            # Penalize downside risk heavily
            if portfolio_downside > 1e-10:
                score = portfolio_return / portfolio_downside
            else:
                score = portfolio_return * 1000
            
            return -score  # Negative because we minimize
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        initial_weights = np.ones(self.num_assets) / self.num_assets
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 128}
        )
        
        if result.success:
            weights = result.x
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
            return weights
        else:
            # Fallback: avoid assets with highest downside risk
            inverse_downside = 1.0 / (downside_std + 1e-8)
            weights = inverse_downside / inverse_downside.sum()
            return weights
    
    def _optimize_stable_growth(self, future_returns):
        """Optimize for stable, consistent growth without sharp jumps"""
        mean_returns = np.mean(future_returns, axis=1)
        
        # Calculate coefficient of variation (volatility relative to return)
        std_returns = np.std(future_returns, axis=1)
        cv = np.zeros(self.num_assets)
        for i in range(self.num_assets):
            if abs(mean_returns[i]) > 1e-10:
                cv[i] = std_returns[i] / abs(mean_returns[i])
            else:
                cv[i] = 1e10  # High penalty for zero return
        
        # Minimize volatility while maintaining positive returns
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(np.cov(future_returns), weights)))
            
            # Penalize both low returns and high volatility
            # Goal: maximize return, minimize volatility
            if portfolio_return > 1e-10:
                stability_score = portfolio_return - 2.0 * portfolio_std
            else:
                stability_score = -1e10 - portfolio_std
            
            return -stability_score  # Negative because we minimize
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        
        # Start with equal weighting
        initial_weights = np.ones(self.num_assets) / self.num_assets
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 128}
        )
        
        if result.success:
            weights = result.x
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
            
            # Additional constraint: limit single asset to max 40% for diversification
            max_weight = 0.4
            while np.max(weights) > max_weight:
                max_idx = np.argmax(weights)
                excess = weights[max_idx] - max_weight
                weights[max_idx] = max_weight
                
                # Redistribute excess to other assets
                other_indices = np.arange(self.num_assets) != max_idx
                if weights[other_indices].sum() > 0:
                    weights[other_indices] += excess * (weights[other_indices] / weights[other_indices].sum())
                else:
                    weights[other_indices] = excess / (self.num_assets - 1)
                
                weights = weights / weights.sum()
            
            return weights
        else:
            # Fallback: favor low volatility assets
            inverse_std = 1.0 / (std_returns + 1e-8)
            weights = inverse_std / inverse_std.sum()
            return weights
    
    def _optimize_minimize_rebalancing(self, future_returns, current_idx):
        """Optimize while minimizing portfolio rebalancing costs"""
        mean_returns = np.mean(future_returns, axis=1)
        
        if current_idx == 0 or not hasattr(self.env, 'last_allocation'):
            current_weights = np.ones(self.num_assets) / self.num_assets
        else:
            current_weights = np.array(self.env.last_allocation)
            if current_weights.sum() == 0:
                current_weights = np.ones(self.num_assets) / self.num_assets
            else:
                current_weights = current_weights / current_weights.sum()
        
        total_returns = np.sum(future_returns, axis=1)
        
        expected_performance = mean_returns - np.mean(mean_returns)
        
        losing_positions = expected_performance < -0.001
        winning_positions = expected_performance > 0.001
        neutral_positions = ~(losing_positions | winning_positions)
        
        new_weights = current_weights.copy()
        
        losing_weight = np.sum(new_weights[losing_positions])
        
        if losing_weight > 0:
            new_weights[losing_positions] = 0
        
        if np.any(winning_positions):
            winning_indices = np.where(winning_positions)[0]
            winning_returns = expected_performance[winning_indices]
            
            if winning_returns.sum() > 0:
                redistribution = winning_returns / winning_returns.sum()
                new_weights[winning_indices] += losing_weight * redistribution
        else:
            if neutral_positions.sum() > 0:
                new_weights[neutral_positions] += losing_weight / neutral_positions.sum()
        
        neutral_change_threshold = 0.05
        for i in range(self.num_assets):
            if neutral_positions[i]:
                change = abs(new_weights[i] - current_weights[i])
                if change < neutral_change_threshold:
                    new_weights[i] = current_weights[i]
        
        new_weights = np.clip(new_weights, 0, 1)
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
        else:
            new_weights = current_weights
        
        rebalance_cost = np.sum(np.abs(new_weights - current_weights))
        
        if rebalance_cost > 0.5:
            blend_factor = 0.5 / rebalance_cost
            new_weights = current_weights * (1 - blend_factor) + new_weights * blend_factor
            new_weights = new_weights / new_weights.sum()
        
        return new_weights
    
    def _optimize_cost_aware(self, future_returns, current_idx):
        """Optimize portfolio using scipy optimizer while penalizing rebalancing costs"""
        mean_returns = np.mean(future_returns, axis=1)
        cov_matrix = np.cov(future_returns)
        cov_matrix += np.eye(len(cov_matrix)) * 1e-8
        
        if current_idx == 0 or not hasattr(self.env, 'last_allocation'):
            current_weights = np.ones(self.num_assets) / self.num_assets
        else:
            current_weights = np.array(self.env.last_allocation)
            if current_weights.sum() == 0:
                current_weights = np.ones(self.num_assets) / self.num_assets
            else:
                current_weights = current_weights / current_weights.sum()
        
        transaction_cost_rate = 0.00175
        rebalancing_penalty = 2.0
        
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            turnover = np.sum(np.abs(weights - current_weights))
            transaction_costs = turnover * transaction_cost_rate
            rebalancing_cost = turnover * rebalancing_penalty
            
            total_cost = transaction_costs + rebalancing_cost
            
            net_return = portfolio_return - total_cost
            
            if portfolio_std < 1e-10:
                score = net_return
            else:
                score = net_return / portfolio_std
            
            return -score
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        
        result = minimize(
            objective,
            current_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 128}
        )
        
        if result.success:
            weights = result.x
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
            return weights
        else:
            return current_weights
    
    def _optimize_winner_counting(self, future_returns):
        num_assets = future_returns.shape[0]
        lookforward_steps = future_returns.shape[1]
        
        win_counts = np.zeros(num_assets)
        
        for step in range(1, lookforward_steps):
            prev_prices = future_returns[:, step - 1]
            curr_prices = future_returns[:, step]
            
            price_changes = curr_prices - prev_prices
            
            max_change_idx = np.argmax(price_changes)
            max_change = price_changes[max_change_idx]
            
            if max_change > 0:
                win_counts[max_change_idx] += 1
        
        if win_counts.sum() == 0:
            return np.ones(num_assets) / num_assets
        
        weights = win_counts / win_counts.sum()
        
        return weights
    
    def _optimize_calmar_ratio(self, future_returns):
        """
        Optimize for Calmar ratio (return / maximum drawdown).
        Better for strategies concerned with avoiding large losses.
        """
        mean_returns = np.mean(future_returns, axis=1)
        
        # Calculate maximum drawdown for each asset
        max_drawdowns = np.zeros(self.num_assets)
        for i in range(self.num_assets):
            cumulative = np.cumsum(future_returns[i, :])
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdowns[i] = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Optimize portfolio for maximum Calmar ratio
        def negative_calmar(weights):
            portfolio_return = np.dot(weights, mean_returns)
            
            # Calculate portfolio drawdown
            portfolio_returns = np.dot(weights, future_returns)
            portfolio_cumulative = np.cumsum(portfolio_returns)
            running_max = np.maximum.accumulate(portfolio_cumulative)
            drawdown = running_max - portfolio_cumulative
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 1e-10
            
            # Calmar ratio: return / max drawdown
            if max_drawdown < 1e-10:
                calmar = portfolio_return * 1000
            else:
                calmar = portfolio_return / max_drawdown
            
            return -calmar  # Negative because we minimize
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        initial_weights = np.ones(self.num_assets) / self.num_assets
        
        result = minimize(
            negative_calmar,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 128}
        )
        
        if result.success:
            weights = result.x
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
            return weights
        else:
            # Fallback: inverse of max drawdown
            inverse_dd = 1.0 / (max_drawdowns + 1e-8)
            # Only allocate to assets with positive returns
            inverse_dd = inverse_dd * (mean_returns > 0)
            if inverse_dd.sum() > 0:
                weights = inverse_dd / inverse_dd.sum()
            else:
                weights = np.ones(self.num_assets) / self.num_assets
            return weights
    
    def _optimize_omega_ratio(self, future_returns, threshold=0.0):
        """
        Optimize for Omega ratio (probability-weighted ratio of gains vs losses).
        More sophisticated than Sharpe - accounts for entire distribution.
        """
        mean_returns = np.mean(future_returns, axis=1)
        
        def negative_omega(weights):
            portfolio_returns = np.dot(weights, future_returns)
            
            # Gains above threshold
            gains = portfolio_returns - threshold
            positive_gains = gains[gains > 0]
            
            # Losses below threshold
            losses = threshold - portfolio_returns
            positive_losses = losses[losses > 0]
            
            # Omega ratio: sum of gains / sum of losses
            sum_gains = np.sum(positive_gains) if len(positive_gains) > 0 else 0
            sum_losses = np.sum(positive_losses) if len(positive_losses) > 0 else 1e-10
            
            if sum_losses < 1e-10:
                omega = sum_gains * 1000
            else:
                omega = sum_gains / sum_losses
            
            return -omega  # Negative because we minimize
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        initial_weights = np.ones(self.num_assets) / self.num_assets
        
        result = minimize(
            negative_omega,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 128}
        )
        
        if result.success:
            weights = result.x
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
            return weights
        else:
            # Fallback: favor assets with positive skew (more upside than downside)
            skewness = np.array([
                np.mean(((future_returns[i] - np.mean(future_returns[i])) / (np.std(future_returns[i]) + 1e-8)) ** 3)
                for i in range(self.num_assets)
            ])
            
            # Positive skew is good, negative is bad
            positive_skew = np.clip(skewness, 0, None)
            if positive_skew.sum() > 0:
                weights = positive_skew / positive_skew.sum()
            else:
                weights = np.ones(self.num_assets) / self.num_assets
            return weights
    
    def _optimize_gradient_ascent(self, future_returns, learning_rate=0.1, iterations=50):
        """
        Simulate gradient ascent on historical returns.
        Iteratively adjust weights toward better performance.
        """
        mean_returns = np.mean(future_returns, axis=1)
        cov_matrix = np.cov(future_returns)
        cov_matrix += np.eye(len(cov_matrix)) * 1e-8
        
        # Start with equal weights
        weights = np.ones(self.num_assets) / self.num_assets
        
        for iteration in range(iterations):
            # Calculate portfolio return and risk
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            # Gradient of Sharpe ratio with respect to weights
            # ∂(return/std)/∂w ≈ (mean_returns/std) - (return * cov * w)/(std^3)
            if portfolio_std > 1e-10:
                gradient = (mean_returns / portfolio_std) - \
                           (portfolio_return * np.dot(cov_matrix, weights) / (portfolio_std ** 3))
            else:
                gradient = mean_returns
            
            # Update weights
            weights = weights + learning_rate * gradient
            
            # Project back to simplex (ensure sum=1 and non-negative)
            weights = np.clip(weights, 0, None)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(self.num_assets) / self.num_assets
                break
            
            # Adaptive learning rate (reduce as we converge)
            learning_rate *= 0.95
        
        return weights
    
    def generate_synthetic_batch(self, num_samples, start_idx=None, method='sharpe', gamma=0.99):
        """
        Generate a batch of synthetic training data with optimal actions.
        
        Args:
            num_samples: Number of samples to generate
            start_idx: Starting index in the data (random if None)
            method: Optimization method for portfolio allocation
            gamma: Discount factor for returns calculation
        
        Returns:
            Dictionary with 'states', 'actions', 'returns', 'is_random' arrays
        """
        if start_idx is None:
            max_start = len(self.env.df[list(self.env.df.keys())[0]]) - self.lookforward_steps - num_samples - 1
            start_idx = np.random.randint(0, max(1, max_start))
        
        states = []
        actions = []
        rewards = []
        is_random = []
        per_asset_rewards = []
        
        self.env.reset()
        self.env.current_step = start_idx
        state = self.env.get_observation()
        
        print(f"Generating {num_samples} synthetic samples starting at index {start_idx} using {method} optimization...")
        
        for i in range(num_samples):
            optimal_action = self.calculate_optimal_portfolio(
                self.env.current_step, 
                method=method
            )
            
            self.env.last_allocation = optimal_action
            
            next_state, reward, done, info = self.env.step(optimal_action)
            
            asset_rewards = np.zeros(len(self.env.asset_names))
            for idx, symbol in enumerate(self.env.asset_names):
                if symbol in self.env.df and self.env.current_step > 0:
                    current_price = self.env.df[symbol]['close'].iloc[self.env.current_step - 1]
                    prev_price = self.env.df[symbol]['close'].iloc[max(0, self.env.current_step - 2)]
                    
                    if prev_price > 0:
                        price_change = (current_price - prev_price) / prev_price
                        allocation = optimal_action[idx]
                        asset_rewards[idx] = price_change * allocation * 100
            
            states.append(state)
            actions.append(optimal_action)
            rewards.append(reward)
            is_random.append(False)
            per_asset_rewards.append(asset_rewards)
            
            state = next_state
            
            if done or self.env.current_step >= len(self.env.df[list(self.env.df.keys())[0]]) - 1:
                print(f"Episode ended at step {i+1}/{num_samples}")
                break
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{num_samples} samples...")
        
        returns = []
        per_asset_returns_cumulative = []
        G = 0
        asset_G = np.zeros(len(self.env.asset_names))
        
        for i in reversed(range(len(rewards))):
            G = rewards[i] + gamma * G
            asset_G = per_asset_rewards[i] + gamma * asset_G
            returns.insert(0, G)
            per_asset_returns_cumulative.insert(0, asset_G.copy())
        
        returns = np.array(returns, dtype=np.float32)
        per_asset_returns_cumulative = np.array(per_asset_returns_cumulative, dtype=np.float32)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        print(f"Generated {len(states)} samples with mean reward: {np.mean(rewards):.4f}")
        
        return {
            'states': states,
            'actions': actions,
            'returns': returns,
            'is_random': is_random,
            'per_asset_returns': per_asset_returns_cumulative.tolist()
        }
    
    def generate_multiple_batches(self, num_batches, samples_per_batch, method='sharpe', gamma=0.99, save_path=None):
        """
        Generate multiple batches of synthetic data covering different time periods.
        
        Args:
            num_batches: Number of batches to generate
            samples_per_batch: Number of samples per batch
            method: Optimization method
            gamma: Discount factor
            save_path: Path to save the generated data (optional)
        
        Returns:
            List of batch dictionaries
        """
        all_batches = []
        
        total_length = len(self.env.df[list(self.env.df.keys())[0]])
        max_start = total_length - self.lookforward_steps - samples_per_batch - 1
        
        if max_start < 0:
            print(f"Warning: Not enough data for requested samples. Reducing samples per batch.")
            samples_per_batch = total_length - self.lookforward_steps - 10
            max_start = 10
        
        # Prepare save path if provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Distribute batches across the entire dataset
        start_indices = np.linspace(0, max_start, num_batches, dtype=int)
        
        for batch_idx, start_idx in enumerate(start_indices):
            print(f"\n=== Batch {batch_idx+1}/{num_batches} ===")
            batch = self.generate_synthetic_batch(
                num_samples=samples_per_batch,
                start_idx=start_idx,
                method=method,
                gamma=gamma
            )
            all_batches.append(batch)
            
            # Save immediately after generating each batch (append mode)
            if save_path:
                with open(save_path, 'ab') as f:
                    pickle.dump(batch, f)
                print(f"Batch {batch_idx+1} saved to {save_path}")
        
        if save_path:
            print(f"\nAll {num_batches} batches saved to {save_path}")
        
        return all_batches
    
    def generate_diverse_batches(self, num_batches, samples_per_batch, gamma=0.99, save_path=None, num_workers=None):
        """
        Generate batches using different optimization methods for diversity.
        Uses multiprocessing for parallel batch generation.
        
        Args:
            num_batches: Number of batches to generate
            samples_per_batch: Number of samples per batch
            gamma: Discount factor
            save_path: Path to save the generated data
            num_workers: Number of parallel workers (defaults to CPU count - 1)
        
        Returns:
            List of batch dictionaries
        """
        methods = ['sharpe', 'returns', 'kelly', 'max_profit', 'downside_protection', 'stable_growth', 
                   'minimize_rebalancing', 'cost_aware_optimization', 'winner_counting', 
                   'calmar_ratio', 'omega_ratio', 'gradient_ascent']
        
        total_length = len(self.env.df[list(self.env.df.keys())[0]])
        max_start = total_length - self.lookforward_steps - samples_per_batch - 1
        
        if max_start < 0:
            print(f"Warning: Not enough data for requested samples. Reducing samples per batch.")
            samples_per_batch = total_length - self.lookforward_steps - 10
            max_start = 10
        
        # Prepare save path if provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        # Ensure num_workers is an integer
        num_workers = int(num_workers)
        
        print(f"Using {num_workers} worker processes for parallel generation")
        
        # Generate batch parameters
        batch_params = []
        for batch_idx in range(num_batches):
            method = methods[batch_idx % len(methods)]
            start_idx = np.random.randint(0, max(1, max_start))
            batch_params.append((batch_idx, num_batches, samples_per_batch, start_idx, method, gamma))
        
        # Process batches in parallel
        with Pool(processes=num_workers) as pool:
            for batch_idx, batch in enumerate(pool.imap(_generate_batch_worker, batch_params)):
                # Save immediately after generating each batch (append mode)
                if save_path and batch is not None:
                    with open(save_path, 'ab') as f:
                        pickle.dump(batch, f)
                    print(f"Batch {batch_idx+1}/{num_batches} saved to {save_path}")
                
                # Clear memory after each batch
                gc.collect()
        
        if save_path:
            print(f"\nAll {num_batches} batches saved to {save_path}")
        
        # Return empty list to save memory (data is already saved to disk)
        return []


if __name__ == "__main__":
    print("=== SYNTHETIC DATA GENERATOR ===\n")
    
    # Initialize environment
    print("Initializing environment...")
    env = PortfolioEnv(max_records=100_000)
    
    # Create generator with 1-step lookforward (aligned with new reward function)
    generator = SyntheticDataGenerator(env, lookforward_steps=32)
    
    # Configuration
    num_batches = 512
    samples_per_batch = 128
    save_path = "/media/user/HDD 1TB/Data/synthetic_training_data.pkl"
    # save_path = "/media/user/64e260bf-e154-43ad-a261-1ebf373775e6/Data/synthetic_training_data.pkl"
    num_workers = max(1, int(cpu_count() / 2))
    
    # Generate diverse batches using different optimization methods
    print(f"\nGenerating {num_batches} batches with {samples_per_batch} samples each...")
    print(f"Using {num_workers} parallel workers")
    print("Using diverse optimization methods:")
    print("  - Sharpe ratio (risk-adjusted returns)")
    print("  - Returns (maximum expected return)")
    print("  - Kelly criterion (optimal growth rate)")
    print("  - Max profit (aggressive maximum gain)")
    print("  - Downside protection (minimize losses)")
    print("  - Stable growth (consistent returns, low volatility)")
    print("  - Minimize rebalancing (heuristic: offload losers, maximize winners)")
    print("  - Cost-aware optimization (optimizer-based with transaction cost penalties)")
    print("  - Winner counting (count highest price changes across future steps)")
    print("  - Calmar ratio (return/max drawdown - minimize large losses)")
    print("  - Omega ratio (probability-weighted gains vs losses)")
    print("  - Gradient ascent (iterative optimization on historical data)\n")
    
    batches = generator.generate_diverse_batches(
        num_batches=num_batches,
        samples_per_batch=samples_per_batch,
        gamma=0.99,
        save_path=save_path,
        num_workers=num_workers
    )
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Total batches: {num_batches}")
    print(f"Data saved to: {save_path}")
