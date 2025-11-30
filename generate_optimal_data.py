import numpy as np
import pickle
import os
from portfolio_env import PortfolioEnv, ModelType
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import time
import random
import string

DEBUG = False
NUM_OF_THREADS = 8
DEBUG_SAMPLES = 32_000 * NUM_OF_THREADS
N_CANDIDATES = 12
N_REFINEMENT_ITERATIONS = 12
REFINEMENT_SAMPLES_PER_ITERATION = 3
N_SAMPLES = NUM_OF_THREADS * 256_0000  # Total samples across all threads
OUTPUT_FOLDER = "syntheticData"
OBS_SHAPE = ModelType.TRANSFORMER_SHAPE

def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=length))

class GradientOptimalActionFinder:
    def __init__(self, env, n_candidates=16, n_refinement_iterations=3, refinement_samples=8):
        self.env = env
        self.n_candidates = n_candidates
        self.n_refinement_iterations = n_refinement_iterations
        self.refinement_samples = refinement_samples
        self.action_dim = len(env.asset_names)
    
    def normalize_action(self, action):
        action = np.clip(action, 0.0, 1.0)
        total = np.sum(action)
        if total > 0:
            action = action / total
        else:
            action = np.ones(self.action_dim) / self.action_dim
        return action
    
    def find_optimal_action(self, obs):
        candidates = []
        rewards = []
        
        for _ in range(self.n_candidates):
            action = np.random.dirichlet(np.ones(self.action_dim))
            
            self.env.portfolio = self.env._saved_portfolio.copy()
            self.env.portfolio_value = self.env._saved_portfolio_value
            self.env.step_count = self.env._saved_step_count
            self.env.prev_reward = self.env._saved_prev_reward.copy()
            
            reward = self.env.step_fast(action)
            
            candidates.append(action)
            rewards.append(reward)
        
        best_idx = np.argmax(rewards)
        best_action = candidates[best_idx]
        best_reward = rewards[best_idx]
        
        for iteration in range(self.n_refinement_iterations):
            noise_scale = 0.2 / (iteration + 1)
            
            for _ in range(self.refinement_samples):
                noise = np.random.randn(self.action_dim) * noise_scale
                perturbed_action = self.normalize_action(best_action + noise)
                
                self.env.portfolio = self.env._saved_portfolio.copy()
                self.env.portfolio_value = self.env._saved_portfolio_value
                self.env.step_count = self.env._saved_step_count
                self.env.prev_reward = self.env._saved_prev_reward.copy()
                
                reward = self.env.step_fast(perturbed_action)
                
                candidates.append(perturbed_action)
                rewards.append(reward)
                
                if reward > best_reward:
                    best_action = perturbed_action
                    best_reward = reward
        
        confidence = self._calculate_confidence(candidates, rewards)
        
        return best_action, best_reward, confidence
    
    def _calculate_confidence(self, candidates, rewards):
        candidates = np.array(candidates)
        rewards = np.array(rewards)
        
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        
        if max_reward - min_reward < 1e-6:
            weights = np.ones(len(rewards)) / len(rewards)
        else:
            normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
            exp_rewards = np.exp(normalized_rewards * 5)
            weights = exp_rewards / np.sum(exp_rewards)
        
        weighted_allocations = np.zeros(self.action_dim)
        for i, candidate in enumerate(candidates):
            weighted_allocations += candidate * weights[i]
        
        confidence = np.zeros(self.action_dim)
        for i in range(self.action_dim):
            asset_allocations = candidates[:, i]
            
            high_reward_mask = rewards > np.percentile(rewards, 50)
            low_reward_mask = rewards < np.percentile(rewards, 50)
            
            if np.any(high_reward_mask):
                avg_high = np.mean(asset_allocations[high_reward_mask])
            else:
                avg_high = 0.5
            
            if np.any(low_reward_mask):
                avg_low = np.mean(asset_allocations[low_reward_mask])
            else:
                avg_low = 0.5
            
            confidence[i] = 0.5 + (avg_high - avg_low)
            confidence[i] = np.clip(confidence[i], 0.0, 1.0)
        
        return confidence


def save_sample_incremental(filepath, obs, action, reward, confidence):
    sample = {
        'obs': obs,
        'action': action,
        'reward': reward,
        'confidence': confidence
    }
    
    mode = 'ab' if os.path.exists(filepath) else 'wb'
    with open(filepath, mode) as f:
        pickle.dump(sample, f)


def load_all_samples(filepath):
    if not os.path.exists(filepath):
        return []
    
    samples = []
    with open(filepath, 'rb') as f:
        while True:
            try:
                sample = pickle.load(f)
                samples.append(sample)
            except EOFError:
                break
    
    return samples


def add_observation_noise(obs, noise_level=0.0125, noise_type='gaussian'):
    """
    Add noise to observation to prevent overfitting and encourage pattern learning.
    
    Args:
        obs: observation array
        noise_level: standard deviation of noise relative to data scale
        noise_type: 'gaussian', 'uniform', or 'mixed'
    
    Returns:
        noised observation
    """
    if obs.size == 0:
        return obs
    
    noised_obs = obs.copy()
    
    if noise_type == 'gaussian':
        # Gaussian noise proportional to the data scale
        data_std = np.std(obs, axis=0, keepdims=True)
        data_std = np.where(data_std == 0, 1.0, data_std)
        noise = np.random.normal(0, noise_level, obs.shape) * data_std
        noised_obs = obs + noise
        
    elif noise_type == 'uniform':
        # Uniform noise proportional to data range
        data_range = np.ptp(obs, axis=0, keepdims=True)
        data_range = np.where(data_range == 0, 1.0, data_range)
        noise = np.random.uniform(-noise_level, noise_level, obs.shape) * data_range
        noised_obs = obs + noise
        
    elif noise_type == 'mixed':
        # Combination of gaussian and uniform noise
        data_std = np.std(obs, axis=0, keepdims=True)
        data_std = np.where(data_std == 0, 1.0, data_std)
        
        gaussian_noise = np.random.normal(0, noise_level * 0.7, obs.shape) * data_std
        uniform_noise = np.random.uniform(-noise_level * 0.3, noise_level * 0.3, obs.shape) * data_std
        
        noised_obs = obs + gaussian_noise + uniform_noise
    
    # Ensure no NaN or Inf values after adding noise
    noised_obs = np.nan_to_num(noised_obs, nan=0.0, posinf=1.0, neginf=0.0)
    
    return noised_obs

def generate_optimal_dataset(n_samples=N_SAMPLES, output_file=None, debug=DEBUG, 
                            n_candidates=N_CANDIDATES, n_refinement_iterations=N_REFINEMENT_ITERATIONS, 
                            refinement_samples=REFINEMENT_SAMPLES_PER_ITERATION, thread_id=None, 
                            obs_shape=OBS_SHAPE, add_noise=True, noise_level=0.02, noise_type='mixed'):
    """
    noise parameters:
        add_noise: whether to add noise to observations
        noise_level: amount of noise (0.01 = 1% of data scale, 0.02 = 2%, etc.)
        noise_type: 'gaussian', 'uniform', or 'mixed'
    """
    if output_file is None:
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        random_str = generate_random_string()
        output_file = os.path.join(OUTPUT_FOLDER, f'data_{random_str}.pkl')
    
    thread_prefix = f"[Thread {thread_id}] " if thread_id is not None else ""
    
    if debug:
        n_samples = DEBUG_SAMPLES
        print(f"{thread_prefix}DEBUG MODE: Generating only {n_samples} samples")
    
    if add_noise:
        print(f"{thread_prefix}Adding {noise_type} noise (level={noise_level}) to observations")
    
    env = PortfolioEnv(obs_shape=obs_shape)
    optimizer = GradientOptimalActionFinder(env, n_candidates=n_candidates, 
                                           n_refinement_iterations=n_refinement_iterations, 
                                           refinement_samples=refinement_samples)
    
    if os.path.exists(output_file):
        existing_samples = load_all_samples(output_file)
        print(f"{thread_prefix}Found {len(existing_samples)} existing samples in {output_file}")
        start_idx = len(existing_samples)
    else:
        start_idx = 0
        print(f"{thread_prefix}Creating new dataset: {output_file}")
    
    rewards_history = []
    portfolio_values = []
    
    total_evaluations = n_candidates + (n_refinement_iterations * refinement_samples)
    print(f"\n{thread_prefix}Generating {n_samples} optimal action samples...")
    print(f"{thread_prefix}Each sample: {n_candidates} initial candidates + {n_refinement_iterations} refinement iterations ({refinement_samples} samples each)")
    print(f"{thread_prefix}Total evaluations per sample: {total_evaluations}")
    
    for i in tqdm(range(n_samples)):
        obs = env.reset()
        
        # Add noise to observation for training robustness
        if add_noise:
            noised_obs = add_observation_noise(obs, noise_level=noise_level, noise_type=noise_type)
        else:
            noised_obs = obs
        
        env._saved_portfolio = env.portfolio.copy()
        env._saved_portfolio_value = env.portfolio_value
        env._saved_step_count = env.step_count
        env._saved_prev_reward = env.prev_reward.copy()
        
        optimal_action, expected_reward, confidence = optimizer.find_optimal_action(obs)
        
        env.portfolio = env._saved_portfolio.copy()
        env.portfolio_value = env._saved_portfolio_value
        env.step_count = env._saved_step_count
        env.prev_reward = env._saved_prev_reward.copy()
        
        _, actual_reward, _, info = env.step(optimal_action)
        
        # Save the NOISED observation with optimal action
        save_sample_incremental(output_file, noised_obs, optimal_action, actual_reward, confidence)
        
        rewards_history.append(actual_reward)
        portfolio_values.append(info['portfolio_value'])
        
        if debug and (i + 1) % 10 == 0:
            thread_prefix = f"[Thread {thread_id}] " if thread_id is not None else ""
            avg_reward = np.mean(rewards_history[-10:])
            print(f"\n{thread_prefix}Sample {i+1}/{n_samples}: Reward={actual_reward:.2f}, Avg(10)={avg_reward:.2f}")
    
    print(f"\n{thread_prefix} Generated {n_samples} samples")
    print(f"{thread_prefix}  Saved to {output_file}")
    print(f"{thread_prefix}  Total samples in file: {start_idx + n_samples}")
    print(f"{thread_prefix}  Average reward: ${np.mean(rewards_history):.2f}")
    print(f"{thread_prefix}  Min reward: ${np.min(rewards_history):.2f}")
    print(f"{thread_prefix}  Max reward: ${np.max(rewards_history):.2f}")
    print(f"{thread_prefix}  Std reward: ${np.std(rewards_history):.2f}")
    
    if debug:
        plot_debug_statistics(rewards_history, portfolio_values, output_file)
    
    return rewards_history, portfolio_values


def plot_debug_statistics(rewards_history, portfolio_values, output_file):
    print("\n" + "="*60)
    print("DEBUG: Generating visualizations...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(rewards_history, alpha=0.7, linewidth=1)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Reward per Sample', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Reward ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    window = 20
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards_history)), moving_avg, 
                       color='red', linewidth=2, label=f'MA({window})')
        axes[0, 0].legend()
    
    axes[0, 1].hist(rewards_history, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=np.mean(rewards_history), color='r', linestyle='--', 
                      linewidth=2, label=f'Mean: ${np.mean(rewards_history):.2f}')
    axes[0, 1].axvline(x=np.median(rewards_history), color='g', linestyle='--', 
                      linewidth=2, label=f'Median: ${np.median(rewards_history):.2f}')
    axes[0, 1].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Reward ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    cumulative_rewards = np.cumsum(rewards_history)
    axes[1, 0].plot(cumulative_rewards, linewidth=2, color='green')
    axes[1, 0].set_title('Cumulative Rewards', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Cumulative Reward ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    final_cumulative = cumulative_rewards[-1]
    axes[1, 0].text(0.05, 0.95, f'Total: ${final_cumulative:.2f}', 
                   transform=axes[1, 0].transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    portfolio_returns = [(pv / 10000 - 1) * 100 for pv in portfolio_values]
    axes[1, 1].scatter(range(len(portfolio_returns)), portfolio_returns, alpha=0.5, s=20)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Portfolio Returns per Sample', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Return (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    positive_rewards = sum(1 for r in rewards_history if r > 0)
    win_rate = (positive_rewards / len(rewards_history)) * 100
    axes[1, 1].text(0.05, 0.95, f'Win Rate: {win_rate:.1f}%', 
                   transform=axes[1, 1].transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    plot_filename = output_file.replace('.pkl', '_debug_plot.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\n Saved visualization to {plot_filename}")
    
    print("\nStrategy Statistics:")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Loss Rate: {100-win_rate:.2f}%")
    print(f"  Avg Winning Trade: ${np.mean([r for r in rewards_history if r > 0]):.2f}")
    print(f"  Avg Losing Trade: ${np.mean([r for r in rewards_history if r < 0]):.2f}")
    print(f"  Profit Factor: {abs(sum([r for r in rewards_history if r > 0]) / sum([r for r in rewards_history if r < 0])):.2f}")
    print(f"  Sharpe Ratio (approx): {np.mean(rewards_history) / (np.std(rewards_history) + 1e-8):.3f}")
    
    if final_cumulative > 0:
        print(f"\n STRATEGY IS PROFITABLE (${final_cumulative:.2f} cumulative)")
    else:
        print(f"\n STRATEGY IS NOT PROFITABLE (${final_cumulative:.2f} cumulative)")
    
    plt.show()


def thread_worker(thread_id, n_samples, noise_level=0.0125, noise_type='mixed'):
    """Modified thread worker to support noise parameters."""
    print(f"[Thread {thread_id}] Starting with noise_level={noise_level}, noise_type={noise_type}...")
    rewards, portfolio_values = generate_optimal_dataset(
        n_samples=n_samples,
        thread_id=thread_id,
        add_noise=True,
        noise_level=noise_level,
        noise_type=noise_type
    )
    print(f"[Thread {thread_id}] Completed!")
    return rewards, portfolio_values


if __name__ == "__main__":
    if NUM_OF_THREADS > 1:
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        
        samples_per_thread = N_SAMPLES // NUM_OF_THREADS
        threads = []
        
        print(f"\nStarting {NUM_OF_THREADS} threads, {samples_per_thread} samples each...")
        print(f"Total samples: {NUM_OF_THREADS * samples_per_thread}")
        print(f"Output folder: {OUTPUT_FOLDER}\n")
        
        start_time = time.time()
        
        for i in range(NUM_OF_THREADS):
            thread = threading.Thread(target=thread_worker, args=(i, samples_per_thread))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"All threads completed in {elapsed_time:.2f} seconds")
        print(f"{'='*60}")
    else:
        rewards, portfolio_values = generate_optimal_dataset()
        
        print("\n" + "="*60)
        print("Dataset generation complete!")
        print("="*60)
