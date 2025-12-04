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
NUM_OF_THREADS = 2
DEBUG_SAMPLES = 200 * NUM_OF_THREADS
N_CANDIDATES = 7
N_REFINEMENT_ITERATIONS = 1
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
        np.clip(action, 0.0, 1.0, out=action)
        total = action.sum()
        if total > 0:
            action /= total
        else:
            action[:] = 1.0 / self.action_dim
        return action
    
    def find_optimal_action(self, obs):
        total_candidates = self.n_candidates + (self.n_refinement_iterations * self.refinement_samples)
        candidates = np.empty((total_candidates, self.action_dim), dtype=np.float32)
        rewards = np.empty(total_candidates, dtype=np.float32)
        idx = 0
        
        dirichlet_alpha = np.ones(self.action_dim)
        for _ in range(self.n_candidates):
            action = np.random.dirichlet(dirichlet_alpha).astype(np.float32)
            
            self.env.portfolio = self.env._saved_portfolio.copy()
            self.env.portfolio_value = self.env._saved_portfolio_value
            self.env.step_count = self.env._saved_step_count
            self.env.prev_reward = self.env._saved_prev_reward.copy()
            
            reward = self.env.step_fast(action)
            
            candidates[idx] = action
            rewards[idx] = reward
            idx += 1
        
        best_idx = np.argmax(rewards[:idx])
        best_action = candidates[best_idx].copy()
        best_reward = rewards[best_idx]
        
        for iteration in range(self.n_refinement_iterations):
            noise_scale = 0.2 / (iteration + 1)
            
            for _ in range(self.refinement_samples):
                noise = np.random.randn(self.action_dim).astype(np.float32) * noise_scale
                perturbed_action = self.normalize_action(best_action + noise)
                
                self.env.portfolio = self.env._saved_portfolio.copy()
                self.env.portfolio_value = self.env._saved_portfolio_value
                self.env.step_count = self.env._saved_step_count
                self.env.prev_reward = self.env._saved_prev_reward.copy()
                
                reward = self.env.step_fast(perturbed_action)
                
                candidates[idx] = perturbed_action
                rewards[idx] = reward
                idx += 1
                
                if reward > best_reward:
                    best_action = perturbed_action.copy()
                    best_reward = reward
        
        confidence = self._calculate_confidence(candidates[:idx], rewards[:idx])
        
        return best_action, best_reward, confidence
    
    def _calculate_confidence(self, candidates, rewards):
        min_reward = rewards.min()
        max_reward = rewards.max()
        
        if max_reward - min_reward < 1e-6:
            weights = np.ones(len(rewards), dtype=np.float32) / len(rewards)
        else:
            normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
            exp_rewards = np.exp(normalized_rewards * 5, dtype=np.float32)
            weights = exp_rewards / exp_rewards.sum()
        
        median_reward = np.median(rewards)
        high_reward_mask = rewards > median_reward
        low_reward_mask = rewards < median_reward
        
        confidence = np.full(self.action_dim, 0.5, dtype=np.float32)
        
        if np.any(high_reward_mask) and np.any(low_reward_mask):
            avg_high = candidates[high_reward_mask].mean(axis=0)
            avg_low = candidates[low_reward_mask].mean(axis=0)
            confidence = 0.5 + (avg_high - avg_low)
            np.clip(confidence, 0.0, 1.0, out=confidence)
        
        return confidence


def save_samples_batch(filepath, samples_batch):
    mode = 'ab' if os.path.exists(filepath) else 'wb'
    with open(filepath, mode) as f:
        for sample in samples_batch:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_sample_incremental(filepath, obs, action, reward, confidence):
    sample = {
        'obs': obs,
        'action': action,
        'reward': reward,
        'confidence': confidence
    }
    
    mode = 'ab' if os.path.exists(filepath) else 'wb'
    with open(filepath, mode) as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


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
        data_std = np.std(obs, axis=0, keepdims=True)
        np.maximum(data_std, 1.0, out=data_std, where=(data_std == 0))
        noise = np.random.normal(0, noise_level, obs.shape).astype(obs.dtype) * data_std
        noised_obs += noise
        
    elif noise_type == 'uniform':
        data_range = np.ptp(obs, axis=0, keepdims=True)
        np.maximum(data_range, 1.0, out=data_range, where=(data_range == 0))
        noise = np.random.uniform(-noise_level, noise_level, obs.shape).astype(obs.dtype) * data_range
        noised_obs += noise
        
    elif noise_type == 'mixed':
        data_std = np.std(obs, axis=0, keepdims=True)
        np.maximum(data_std, 1.0, out=data_std, where=(data_std == 0))
        
        gaussian_noise = np.random.normal(0, noise_level * 0.7, obs.shape).astype(obs.dtype) * data_std
        uniform_noise = np.random.uniform(-noise_level * 0.3, noise_level * 0.3, obs.shape).astype(obs.dtype) * data_std
        
        noised_obs += gaussian_noise
        noised_obs += uniform_noise
    
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
    
    interval = random.randint(12, 72)
    reward_scale = 12.0 / interval
    env = PortfolioEnv(obs_shape=obs_shape, rebalancing_interval=interval)
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
    
    from collections import deque
    
    rewards_history = np.empty(n_samples, dtype=np.float32)
    portfolio_values = np.empty(n_samples, dtype=np.float32)
    
    rewards_10 = deque(maxlen=10)
    rewards_50 = deque(maxlen=50)
    rewards_sum = 0.0
    wins = 0
    
    batch_buffer = []
    batch_size = 100
    
    total_evaluations = n_candidates + (n_refinement_iterations * refinement_samples)
    print(f"\n{thread_prefix}Generating {n_samples} optimal action samples...")
    print(f"{thread_prefix}Each sample: {n_candidates} initial candidates + {n_refinement_iterations} refinement iterations ({refinement_samples} samples each)")
    print(f"{thread_prefix}Total evaluations per sample: {total_evaluations}")
    
    for i in tqdm(range(n_samples)):
        obs = env.reset()
        
        if add_noise:
            noised_obs = add_observation_noise(obs, noise_level=noise_level, noise_type=noise_type)
        else:
            noised_obs = obs
        
        env._saved_portfolio = env.portfolio.copy()
        env._saved_portfolio_value = env.portfolio_value
        env._saved_step_count = env.step_count
        env._saved_prev_reward = env.prev_reward.copy()
        
        optimal_action, expected_reward, confidence = optimizer.find_optimal_action(obs)
        
        env.portfolio = env._saved_portfolio
        env.portfolio_value = env._saved_portfolio_value
        env.step_count = env._saved_step_count
        env.prev_reward = env._saved_prev_reward
        
        _, actual_reward, _, info = env.step(optimal_action)
        normalized_reward = actual_reward * reward_scale
        
        sample = {
            'obs': noised_obs,
            'action': optimal_action,
            'reward': normalized_reward,
            'confidence': confidence
        }
        batch_buffer.append(sample)
        
        if len(batch_buffer) >= batch_size:
            save_samples_batch(output_file, batch_buffer)
            batch_buffer.clear()
        
        rewards_history[i] = normalized_reward
        portfolio_values[i] = info['portfolio_value']
        
        rewards_10.append(normalized_reward)
        rewards_50.append(normalized_reward)
        rewards_sum += normalized_reward
        if normalized_reward > 0:
            wins += 1
        
        if (i + 1) % 10 == 0:
            avg_10 = sum(rewards_10) / len(rewards_10)
            avg_50 = sum(rewards_50) / len(rewards_50)
            avg_all = rewards_sum / (i + 1)
            win_rate = (wins / (i + 1)) * 100
            print(f"\n{thread_prefix}Sample {i+1}/{n_samples}: R={normalized_reward:.1f}, "
                  f"Avg10={avg_10:.1f}, Avg50={avg_50:.1f}, AvgAll={avg_all:.1f}, WR={win_rate:.1f}%")
    
    if batch_buffer:
        save_samples_batch(output_file, batch_buffer)
        batch_buffer.clear()
    
    print(f"\n{thread_prefix} Generated {n_samples} samples")
    print(f"{thread_prefix}  Saved to {output_file}")
    print(f"{thread_prefix}  Total samples in file: {start_idx + n_samples}")
    print(f"{thread_prefix}  Average reward: ${rewards_sum / n_samples:.2f}")
    print(f"{thread_prefix}  Min reward: ${rewards_history.min():.2f}")
    print(f"{thread_prefix}  Max reward: ${rewards_history.max():.2f}")
    print(f"{thread_prefix}  Std reward: ${rewards_history.std():.2f}")
    
    if debug:
        plot_debug_statistics(rewards_history, portfolio_values, output_file)
    
    return rewards_history.tolist(), portfolio_values.tolist()


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


def thread_worker(thread_id, n_samples, noise_level=0.00275, noise_type='mixed'):
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
