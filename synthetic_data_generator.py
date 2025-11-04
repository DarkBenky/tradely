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
        
        # Create generator
        generator = SyntheticDataGenerator(env, lookforward_steps=30)
        
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
    """
    
    def __init__(self, env, lookforward_steps=20):
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
        
        # Reset environment to start position
        self.env.reset()
        self.env.current_step = start_idx
        state = self.env.get_observation()
        
        print(f"Generating {num_samples} synthetic samples starting at index {start_idx} using {method} optimization...")
        
        for i in range(num_samples):
            # Calculate optimal portfolio allocation
            optimal_action = self.calculate_optimal_portfolio(
                self.env.current_step, 
                method=method
            )
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(optimal_action)
            
            states.append(state)
            actions.append(optimal_action)
            rewards.append(reward)
            is_random.append(False)  # These are optimal, not random
            
            state = next_state
            
            if done or self.env.current_step >= len(self.env.df[list(self.env.df.keys())[0]]) - 1:
                print(f"Episode ended at step {i+1}/{num_samples}")
                break
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{num_samples} samples...")
        
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        returns = np.array(returns, dtype=np.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        print(f"Generated {len(states)} samples with mean reward: {np.mean(rewards):.4f}")
        
        return {
            'states': states,
            'actions': actions,
            'returns': returns,
            'is_random': is_random
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
        methods = ['sharpe', 'returns', 'kelly']
        
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
    
    # Create generator
    generator = SyntheticDataGenerator(env, lookforward_steps=30)
    
    # Configuration
    num_batches = 1024
    samples_per_batch = 128
    save_path = "/media/user/HDD 1TB/Data/synthetic_training_data.pkl"
    num_workers = max(1, int(cpu_count() / 4))  # Use all cores except one
    
    # Generate diverse batches using different optimization methods
    print(f"\nGenerating {num_batches} batches with {samples_per_batch} samples each...")
    print(f"Using {num_workers} parallel workers")
    print("Using diverse optimization methods: Sharpe ratio, Returns, Kelly criterion\n")
    
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
