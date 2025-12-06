import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json
import wandb
import copy
import pickle
from collections import deque

tf.keras.mixed_precision.set_global_policy('mixed_float16')

NUM_PREDICTIONS = 8
REBALANCE_INTERVAL = 12
LEARNING_RATE = 0.00001
BATCH_SIZE = 4
MAX_EPISODES = 10000
STEPS_PER_EPISODE = 256
FEE_RATE = 0.001
GRADIENT_CLIP_NORM = 1.0
ROLLBACK_ON_NAN = True
MONTE_CARLO_ENABLED = True
WANDB_PROJECT = "portfolio-finetune"
BEST_MODEL_PATH = "models/best_pretrain_transformer.weights.h5"
CONFIG_PATH = "models/model_config.json"
NORM_STATS_PATH = "models/normalization_stats.json"
LOSS_WINDOW_SIZE = 5
ENTROPY_BONUS = 0.01
ALLOCATION_LOSS_WEIGHT = 1.0
REWARD_LOSS_WEIGHT = 2.0
WIN_RATE_WEIGHT = 0.3
REWARD_MAXIMIZE_WEIGHT = 0.2
OUTPERFORMANCE_WEIGHT = 0.5
REWARD_CLIP_VALUE = 3.0

REPLAY_BUFFER_SIZE = 50000
REPLAY_BUFFER_PATH = "models/replay_buffer.pkl"
REPLAY_MIN_SIZE = 1000
REPLAY_RATIO = 0.5


class RewardNormalizer:
    def __init__(self, clip_value=REWARD_CLIP_VALUE):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip_value = clip_value
        self.epsilon = 1e-8
    
    def update(self, rewards):
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var + self.epsilon
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
            self.var = m2 / total_count
            self.count = total_count
    
    def normalize(self, rewards):
        std = np.sqrt(self.var + self.epsilon)
        normalized = (rewards - self.mean) / std
        return np.clip(normalized, -self.clip_value, self.clip_value)


class ReplayBuffer:
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.max_size = max_size
        self.observations = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
        self.benchmark_rewards = deque(maxlen=max_size)
    
    def add(self, obs, action, reward, value, benchmark_reward=0.0):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.benchmark_rewards.append(benchmark_reward)
    
    def add_batch(self, obs_batch, action_batch, reward_batch, value_batch, benchmark_batch=None):
        if benchmark_batch is None:
            benchmark_batch = [0.0] * len(obs_batch)
        for i in range(len(obs_batch)):
            self.add(obs_batch[i], action_batch[i], reward_batch[i], value_batch[i], benchmark_batch[i])
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.observations), size=min(batch_size, len(self.observations)), replace=False)
        obs = np.array([self.observations[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        values = np.array([self.values[i] for i in indices])
        benchmark = np.array([self.benchmark_rewards[i] for i in indices])
        return obs, actions, rewards, values, benchmark
    
    def __len__(self):
        return len(self.observations)
    
    def save(self, filepath):
        data = {
            'observations': list(self.observations),
            'actions': list(self.actions),
            'rewards': list(self.rewards),
            'values': list(self.values),
            'benchmark_rewards': list(self.benchmark_rewards)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.observations = deque(data['observations'], maxlen=self.max_size)
            self.actions = deque(data['actions'], maxlen=self.max_size)
            self.rewards = deque(data['rewards'], maxlen=self.max_size)
            self.values = deque(data.get('values', [0.0] * len(self.rewards)), maxlen=self.max_size)
            self.benchmark_rewards = deque(data.get('benchmark_rewards', [0.0] * len(self.rewards)), maxlen=self.max_size)
            return True
        except (EOFError, pickle.UnpicklingError, KeyError):
            print(f"Corrupted replay buffer at {filepath}, starting fresh")
            return False


def load_model_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


class ObservationNormalizer:
    def __init__(self, features_per_timestep):
        self.features_per_timestep = features_per_timestep
        self.mean = np.zeros(features_per_timestep, dtype=np.float64)
        self.var = np.ones(features_per_timestep, dtype=np.float64)
        self.count = 0
        self.epsilon = 1e-8
    
    def normalize(self, obs_batch):
        std = np.sqrt(self.var + self.epsilon)
        return (obs_batch - self.mean) / std
    
    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                stats = json.load(f)
            self.mean = np.array(stats['mean'], dtype=np.float64)
            self.var = np.array(stats['var'], dtype=np.float64)
            self.count = stats['count']
            return True
        return False


def bucket_to_reward_estimate(bucket_probs):
    bucket_centers = np.array([-350, -175, -62.5, -12.5, 12.5, 37.5, 75, 150, 300, 500], dtype=np.float32)
    if np.any(np.isnan(bucket_probs)):
        return 0.0
    result = np.sum(bucket_probs * bucket_centers, axis=-1)
    if np.isnan(result):
        return 0.0
    return result


def check_weights_for_nan(model):
    for w in model.trainable_weights:
        if tf.reduce_any(tf.math.is_nan(w)):
            return True
        if tf.reduce_any(tf.math.is_inf(w)):
            return True
    return False


def save_weights_checkpoint(model):
    weights = [w.numpy() for w in model.trainable_weights]
    return weights


def restore_weights_checkpoint(model, checkpoint):
    for w, saved in zip(model.trainable_weights, checkpoint):
        w.assign(saved)


class OnlineTrainer:
    def __init__(self, model, normalizer, optimizer, replay_buffer=None, uses_scratchpad=False):
        self.model = model
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.reward_normalizer = RewardNormalizer()
        self.replay_buffer = replay_buffer
        self.uses_scratchpad = uses_scratchpad
        self.last_checkpoint = None
        self.episode_stats = {
            'rewards': [],
            'portfolio_values': [],
            'benchmark_values': [],
            'fees_paid': [],
            'allocations': [],
            'turnover': []
        }
    
    def get_action(self, obs, use_monte_carlo=True):
        obs_norm = self.normalizer.normalize(obs.reshape(1, *obs.shape)).astype(np.float32)
        
        if np.any(np.isnan(obs_norm)) or np.any(np.isinf(obs_norm)):
            obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=1.0, neginf=-1.0)
        
        bucket_probs = None
        
        if use_monte_carlo and NUM_PREDICTIONS > 1 and not self.uses_scratchpad:
            allocation_preds = []
            reward_preds = []
            for _ in range(NUM_PREDICTIONS):
                alloc, rew = self.model(obs_norm, training=True)
                alloc_np = alloc.numpy()
                rew_np = rew.numpy()
                if not np.any(np.isnan(alloc_np)) and not np.any(np.isnan(rew_np)):
                    allocation_preds.append(alloc_np)
                    reward_preds.append(rew_np)
            
            if len(allocation_preds) == 0:
                alloc, rew = self.model(obs_norm, training=False)
                allocation = np.nan_to_num(alloc.numpy()[0], nan=1.0/len(alloc.numpy()[0]))
                allocation = allocation / (np.sum(allocation) + 1e-8)
                return allocation, np.zeros_like(allocation), 0.0, 0.0, np.ones(10) / 10.0
            
            allocation_preds = np.array(allocation_preds)
            reward_preds = np.array(reward_preds)
            
            allocation = np.mean(allocation_preds, axis=0)[0]
            allocation_std = np.std(allocation_preds, axis=0)[0]
            
            bucket_probs = np.mean(reward_preds, axis=0)[0]
            pred_reward = bucket_to_reward_estimate(bucket_probs)
            pred_reward_std = np.std([bucket_to_reward_estimate(r[0]) for r in reward_preds])
        else:
            alloc, rew = self.model(obs_norm, training=False)
            allocation = alloc.numpy()[0]
            allocation_std = np.zeros_like(allocation)
            bucket_probs = rew.numpy()[0]
            pred_reward = bucket_to_reward_estimate(bucket_probs)
            pred_reward_std = 0.0
        
        if np.any(np.isnan(allocation)):
            allocation = np.nan_to_num(allocation, nan=1.0/len(allocation))
            allocation = allocation / (np.sum(allocation) + 1e-8)
        
        return allocation, allocation_std, pred_reward, pred_reward_std, bucket_probs
    
    def train_step(self, obs_batch, actions_batch, rewards_batch, values_batch, benchmark_batch=None):
        self.last_checkpoint = save_weights_checkpoint(self.model)
        
        self.reward_normalizer.update(rewards_batch)
        rewards_normalized = self.reward_normalizer.normalize(rewards_batch)
        
        obs_norm = tf.constant(self.normalizer.normalize(obs_batch).astype(np.float32))
        actions_tf = tf.constant(actions_batch.astype(np.float32))
        rewards_tf = tf.constant(rewards_batch.astype(np.float32))
        rewards_norm_tf = tf.constant(rewards_normalized.astype(np.float32))
        
        if benchmark_batch is not None:
            benchmark_tf = tf.constant(benchmark_batch.astype(np.float32))
            excess_returns = rewards_tf - benchmark_tf
        else:
            excess_returns = rewards_tf
        
        bucket_centers = tf.constant(
            np.array([-350, -175, -62.5, -12.5, 12.5, 37.5, 75, 150, 300, 500], dtype=np.float32)
        )
        bucket_boundaries = tf.constant(
            np.array([-250, -100, -25, 0, 25, 50, 100, 200, 400], dtype=np.float32)
        )
        
        with tf.GradientTape() as tape:
            alloc_pred, reward_pred = self.model(obs_norm, training=True)
            
            pred_values = tf.reduce_sum(reward_pred * bucket_centers, axis=-1)
            
            value_mse_loss = tf.reduce_mean(tf.square((pred_values - rewards_tf) / 50.0))
            
            reward_bucket_indices = tf.zeros_like(rewards_tf, dtype=tf.int32)
            for i in range(9):
                reward_bucket_indices = tf.where(
                    rewards_tf >= bucket_boundaries[i],
                    tf.ones_like(reward_bucket_indices) * (i + 1),
                    reward_bucket_indices
                )
            reward_onehot = tf.one_hot(reward_bucket_indices, depth=10)
            bucket_ce_loss = -tf.reduce_mean(tf.reduce_sum(reward_onehot * tf.math.log(tf.clip_by_value(reward_pred, 1e-8, 1.0)), axis=-1))
            
            value_loss = value_mse_loss + bucket_ce_loss
            
            alloc_mse = tf.reduce_mean(tf.square(alloc_pred - actions_tf), axis=-1)
            
            profitable_mask = tf.cast(rewards_tf > 0, tf.float32)
            outperform_mask = tf.cast(excess_returns > 0, tf.float32)
            profitable_and_outperform = profitable_mask * outperform_mask
            
            allocation_weight = tf.where(
                profitable_and_outperform > 0,
                tf.abs(rewards_tf) / 10.0 + 1.0,
                tf.where(
                    profitable_mask > 0,
                    tf.abs(rewards_tf) / 20.0 + 0.5,
                    0.05
                )
            )
            allocation_loss = tf.reduce_mean(alloc_mse * allocation_weight)
            
            positive_bucket_probs = tf.reduce_sum(reward_pred[:, 4:], axis=-1)
            positive_bucket_probs = tf.clip_by_value(positive_bucket_probs, 1e-6, 1.0 - 1e-6)
            win_rate_loss = -tf.reduce_mean(profitable_mask * tf.math.log(positive_bucket_probs) +
                                            (1 - profitable_mask) * tf.math.log(1 - positive_bucket_probs))
            
            profit_loss = -tf.reduce_mean(rewards_tf) / 50.0
            
            profitable_excess = tf.where(rewards_tf > 0, excess_returns, tf.zeros_like(excess_returns))
            outperformance_loss = -tf.reduce_mean(profitable_excess) / 100.0
            
            entropy = -tf.reduce_sum(alloc_pred * tf.math.log(tf.clip_by_value(alloc_pred, 1e-6, 1.0)), axis=-1)
            entropy_bonus = tf.reduce_mean(entropy)
            
            total_loss = (ALLOCATION_LOSS_WEIGHT * allocation_loss + 
                         REWARD_LOSS_WEIGHT * value_loss +
                         WIN_RATE_WEIGHT * win_rate_loss +
                         REWARD_MAXIMIZE_WEIGHT * profit_loss +
                         OUTPERFORMANCE_WEIGHT * outperformance_loss -
                         ENTROPY_BONUS * entropy_bonus)
        
        if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss):
            if ROLLBACK_ON_NAN and self.last_checkpoint is not None:
                restore_weights_checkpoint(self.model, self.last_checkpoint)
                print("NaN in loss, rolled back weights")
            return None, None, None, None, None, None, None, True
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        has_nan_grad = False
        for g in gradients:
            if g is not None and tf.reduce_any(tf.math.is_nan(g)):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            if ROLLBACK_ON_NAN and self.last_checkpoint is not None:
                restore_weights_checkpoint(self.model, self.last_checkpoint)
                print("NaN in gradients, rolled back weights")
            return None, None, None, None, None, None, None, True
        
        gradients, _ = tf.clip_by_global_norm(gradients, GRADIENT_CLIP_NORM)
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        if check_weights_for_nan(self.model):
            if ROLLBACK_ON_NAN and self.last_checkpoint is not None:
                restore_weights_checkpoint(self.model, self.last_checkpoint)
                print("NaN in weights after update, rolled back")
            return None, None, None, None, None, None, None, True
        
        return (float(total_loss.numpy()), 
                float(allocation_loss.numpy()), 
                float(value_loss.numpy()), 
                float(win_rate_loss.numpy()),
                float(profit_loss.numpy()),
                float(outperformance_loss.numpy()),
                float(entropy_bonus.numpy()),
                False)
    
    def run_episode(self, env, max_steps=STEPS_PER_EPISODE, log_steps=True):
        obs = env.reset()
        
        observations = []
        actions = []
        rewards = []
        values = []
        benchmark_rewards = []
        infos = []
        
        total_fees = 0.0
        total_turnover = 0.0
        
        initial_portfolio = env._portfolio_portfolio_value()
        initial_benchmark = env._update_benchmark_value()
        prev_benchmark = initial_benchmark
        
        allocation_sums = np.zeros(len(env.asset_names))
        bucket_centers = np.array([-350, -175, -62.5, -12.5, 12.5, 37.5, 75, 150, 300, 500])
        
        for step in range(max_steps):
            allocation, alloc_std, pred_reward, pred_std, bucket_probs = self.get_action(obs, MONTE_CARLO_ENABLED)
            
            prev_value = env._portfolio_portfolio_value()
            
            obs_next, reward, done, info = env.step(allocation)
            
            current_benchmark = info.get('benchmark_value', prev_benchmark)
            benchmark_reward = current_benchmark - prev_benchmark
            prev_benchmark = current_benchmark
            
            observations.append(obs)
            actions.append(allocation)
            rewards.append(reward)
            values.append(pred_reward)
            benchmark_rewards.append(benchmark_reward)
            infos.append(info)
            
            total_fees += info.get('fees_paid', 0.0)
            if prev_value > 0 and FEE_RATE > 0:
                total_turnover += info.get('fees_paid', 0.0) / (FEE_RATE * prev_value)
            
            allocation_sums += allocation
            
            if log_steps:
                current_portfolio = env._portfolio_portfolio_value()
                step_portfolio_return = (current_portfolio - initial_portfolio) / initial_portfolio * 100
                step_benchmark_return = (current_benchmark - initial_benchmark) / initial_benchmark * 100
                
                max_bucket_idx = int(np.argmax(bucket_probs))
                max_bucket_prob = float(np.max(bucket_probs))
                bucket_entropy = float(-np.sum(bucket_probs * np.log(np.clip(bucket_probs, 1e-8, 1.0))))
                
                step_log = {
                    'step/portfolio_value': current_portfolio,
                    'step/benchmark_value': current_benchmark,
                    'step/portfolio_return': step_portfolio_return,
                    'step/benchmark_return': step_benchmark_return,
                    'step/excess_return': step_portfolio_return - step_benchmark_return,
                    'step/reward': reward,
                    'step/benchmark_reward': benchmark_reward,
                    'step/excess_reward': reward - benchmark_reward,
                    'step/pred_reward': pred_reward,
                    'step/pred_reward_std': pred_std,
                    'step/fees_cumulative': total_fees,
                    'step/bucket_max_prob': max_bucket_prob,
                    'step/bucket_max_idx': max_bucket_idx,
                    'step/bucket_max_center': bucket_centers[max_bucket_idx],
                    'step/bucket_entropy': bucket_entropy
                }
                
                for i in range(len(bucket_probs)):
                    step_log[f'step/bucket_prob_{i}'] = float(bucket_probs[i])
                
                for i, asset in enumerate(env.asset_names):
                    step_log[f'step/alloc_{asset}'] = float(allocation[i])
                    step_log[f'step/alloc_std_{asset}'] = float(alloc_std[i])
                
                wandb.log(step_log)
            
            obs = obs_next
            
            if done:
                break
        
        final_portfolio = env._portfolio_portfolio_value()
        final_benchmark = env._update_benchmark_value()
        
        n_steps = len(observations)
        avg_allocation = allocation_sums / n_steps if n_steps > 0 else allocation_sums
        
        portfolio_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100
        benchmark_return = (final_benchmark - initial_benchmark) / initial_benchmark * 100
        excess_return = portfolio_return - benchmark_return
        
        holding_ratios = {}
        for i, asset in enumerate(env.asset_names):
            holding_ratios[asset] = float(avg_allocation[i])
        
        outperform_rate = sum(1 for r, b in zip(rewards, benchmark_rewards) if r > b) / len(rewards) * 100 if rewards else 0
        
        episode_info = {
            'n_steps': n_steps,
            'initial_portfolio': initial_portfolio,
            'final_portfolio': final_portfolio,
            'initial_benchmark': initial_benchmark,
            'final_benchmark': final_benchmark,
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'total_fees': total_fees,
            'total_turnover': total_turnover,
            'avg_turnover': total_turnover / n_steps if n_steps > 0 else 0,
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards) if rewards else 0,
            'win_rate': sum(1 for r in rewards if r > 0) / len(rewards) * 100 if rewards else 0,
            'outperform_rate': outperform_rate,
            'holding_ratios': holding_ratios
        }
        
        return (np.array(observations), np.array(actions), 
                np.array(rewards), np.array(values), np.array(benchmark_rewards), episode_info)


def finetune():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPUs available: {len(physical_devices)}")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    config = load_model_config()
    
    sequence_length = config['sequence_length']
    features_per_timestep = config['features_per_timestep']
    num_assets = config['num_assets']
    model_type = config.get('model_type', 'temporal-cnn')
    
    from preTrain import (build_model_temporal_cnn, build_model_iterative_refine,
                          build_model, build_model_v2, build_model_lstm,
                          build_model_dense, build_model_cnn, build_model_hybrid_attention)
    
    print(f"Loading model type: {model_type}")
    
    if model_type == 'iterative-refine':
        model = build_model_iterative_refine(
            sequence_length, features_per_timestep, num_assets,
            cnn_filters=config.get('cnn_filters', 512),
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'temporal-cnn':
        model = build_model_temporal_cnn(
            sequence_length, features_per_timestep, num_assets,
            d_model=config['d_model'],
            cnn_filters=config.get('cnn_filters', 512),
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'v2-cnn-transformer-cnn':
        model = build_model_v2(
            sequence_length, features_per_timestep, num_assets,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_blocks=config['num_layers'],
            cnn_filters=config.get('cnn_filters', 512),
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'hybrid-attention':
        model = build_model_hybrid_attention(
            sequence_length, features_per_timestep, num_assets,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            cnn_filters=config.get('cnn_filters', 512),
            dropout_rate=config['dropout_rate']
        )
    else:
        model = build_model(
            sequence_length, features_per_timestep, num_assets,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout_rate=config['dropout_rate']
        )
    
    if os.path.exists(BEST_MODEL_PATH):
        model.load_weights(BEST_MODEL_PATH)
        print(f"Loaded pretrained weights from {BEST_MODEL_PATH}")
    else:
        print("No pretrained model found, starting from scratch")
    
    model.summary()
    
    normalizer = ObservationNormalizer(features_per_timestep)
    if normalizer.load(NORM_STATS_PATH):
        print(f"Loaded normalization stats (count: {normalizer.count})")
    
    optimizer = keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    
    from portfolio_env import PortfolioEnv, ModelType
    env = PortfolioEnv(obs_shape=ModelType.TRANSFORMER_SHAPE, rebalancing_interval=REBALANCE_INTERVAL)
    env.fee_rate = FEE_RATE
    
    replay_buffer = ReplayBuffer(max_size=REPLAY_BUFFER_SIZE)
    if os.path.exists(REPLAY_BUFFER_PATH):
        if replay_buffer.load(REPLAY_BUFFER_PATH):
            print(f"Loaded replay buffer with {len(replay_buffer)} experiences")
    
    trainer = OnlineTrainer(model, normalizer, optimizer, replay_buffer, 
                            uses_scratchpad=(model_type == 'iterative-refine'))
    
    wandb.init(project=WANDB_PROJECT, config={
        'num_predictions': NUM_PREDICTIONS,
        'rebalance_interval': REBALANCE_INTERVAL,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'max_episodes': MAX_EPISODES,
        'steps_per_episode': STEPS_PER_EPISODE,
        'fee_rate': FEE_RATE,
        'gradient_clip_norm': GRADIENT_CLIP_NORM,
        'monte_carlo_enabled': MONTE_CARLO_ENABLED,
        'entropy_bonus': ENTROPY_BONUS,
        'allocation_loss_weight': ALLOCATION_LOSS_WEIGHT,
        'reward_loss_weight': REWARD_LOSS_WEIGHT,
        'reward_clip_value': REWARD_CLIP_VALUE,
        'replay_buffer_size': REPLAY_BUFFER_SIZE,
        'replay_min_size': REPLAY_MIN_SIZE,
        'replay_ratio': REPLAY_RATIO,
        'model_config': config
    })
    
    best_excess_return = -float('inf')
    best_avg_loss = float('inf')
    loss_history = []
    rolling_returns = []
    rolling_window = 20
    
    print(f"\nStarting online finetuning...")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Steps per episode: {STEPS_PER_EPISODE}")
    print(f"  Rebalance interval: {REBALANCE_INTERVAL}")
    print(f"  Fee rate: {FEE_RATE}")
    print(f"  Monte Carlo predictions: {NUM_PREDICTIONS if MONTE_CARLO_ENABLED else 1}")
    print(f"  Replay buffer: {len(replay_buffer)} experiences")
    
    if len(replay_buffer) >= REPLAY_MIN_SIZE:
        warmup_batches = min(100, len(replay_buffer) // BATCH_SIZE)
        print(f"\nWarmup training on {warmup_batches} batches from replay buffer...")
        for i in range(warmup_batches):
            obs_batch, act_batch, rew_batch, val_batch, bench_batch = replay_buffer.sample(BATCH_SIZE)
            result = trainer.train_step(obs_batch, act_batch, rew_batch, val_batch, bench_batch)
            if i % 20 == 0 and result[7] is False:
                print(f"  Warmup batch {i}/{warmup_batches}, loss: {result[0]:.4f}")
        print("Warmup complete\n")
    
    for episode in range(MAX_EPISODES):
        observations, actions, rewards, values, benchmark_rewards, episode_info = trainer.run_episode(env)
        
        replay_buffer.add_batch(observations, actions, rewards, values, benchmark_rewards)
        
        if len(replay_buffer) >= REPLAY_MIN_SIZE:
            total_loss_sum = 0.0
            alloc_loss_sum = 0.0
            value_loss_sum = 0.0
            entropy_sum = 0.0
            n_batches = 0
            nan_count = 0
            
            n_new_batches = max(1, len(observations) // BATCH_SIZE)
            n_replay_batches = int(n_new_batches * REPLAY_RATIO / (1 - REPLAY_RATIO)) if REPLAY_RATIO < 1 else n_new_batches
            
            indices = np.arange(len(observations))
            np.random.shuffle(indices)
            for start_idx in range(0, min(len(observations), n_new_batches * BATCH_SIZE), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(observations))
                if end_idx - start_idx < BATCH_SIZE:
                    continue
                batch_indices = indices[start_idx:end_idx]
                
                obs_batch = observations[batch_indices]
                act_batch = actions[batch_indices]
                rew_batch = rewards[batch_indices]
                val_batch = values[batch_indices]
                bench_batch = benchmark_rewards[batch_indices]
                
                result = trainer.train_step(obs_batch, act_batch, rew_batch, val_batch, bench_batch)
                
                if result[7]:
                    nan_count += 1
                    continue
                
                total_loss, alloc_loss, value_loss, win_loss, rew_max_loss, outperf_loss, entropy = result[:7]
                total_loss_sum += total_loss
                alloc_loss_sum += alloc_loss
                value_loss_sum += value_loss
                entropy_sum += entropy
                n_batches += 1
            
            for _ in range(n_replay_batches):
                obs_batch, act_batch, rew_batch, val_batch, bench_batch = replay_buffer.sample(BATCH_SIZE)
                
                result = trainer.train_step(obs_batch, act_batch, rew_batch, val_batch, bench_batch)
                
                if result[7]:
                    nan_count += 1
                    continue
                
                total_loss, alloc_loss, value_loss, win_loss, rew_max_loss, outperf_loss, entropy = result[:7]
                total_loss_sum += total_loss
                alloc_loss_sum += alloc_loss
                value_loss_sum += value_loss
                entropy_sum += entropy
                n_batches += 1
            
            if n_batches > 0:
                avg_total_loss = total_loss_sum / n_batches
                avg_alloc_loss = alloc_loss_sum / n_batches
                avg_value_loss = value_loss_sum / n_batches
                avg_entropy = entropy_sum / n_batches
            else:
                avg_total_loss = avg_alloc_loss = avg_value_loss = avg_entropy = 0.0
        else:
            avg_total_loss = avg_alloc_loss = avg_value_loss = avg_entropy = 0.0
            nan_count = 0
        
        rolling_returns.append(episode_info['excess_return'])
        if len(rolling_returns) > rolling_window:
            rolling_returns.pop(0)
        rolling_avg_return = np.mean(rolling_returns)
        
        log_data = {
            'episode': episode,
            'n_steps': episode_info['n_steps'],
            'portfolio_return': episode_info['portfolio_return'],
            'benchmark_return': episode_info['benchmark_return'],
            'excess_return': episode_info['excess_return'],
            'rolling_avg_excess_return': rolling_avg_return,
            'total_fees': episode_info['total_fees'],
            'avg_turnover': episode_info['avg_turnover'],
            'total_reward': episode_info['total_reward'],
            'avg_reward': episode_info['avg_reward'],
            'win_rate': episode_info['win_rate'],
            'outperform_rate': episode_info['outperform_rate'],
            'loss/total': avg_total_loss,
            'loss/allocation': avg_alloc_loss,
            'loss/value': avg_value_loss,
            'loss/entropy': avg_entropy,
            'nan_batches': nan_count,
            'replay_buffer_size': len(replay_buffer)
        }
        
        for asset, ratio in episode_info['holding_ratios'].items():
            log_data[f'holdings/{asset}'] = ratio
        
        wandb.log(log_data)
        
        if episode % 10 == 0:
            print(f"\nEpisode {episode}:")
            print(f"  Portfolio: {episode_info['portfolio_return']:.2f}% | Benchmark: {episode_info['benchmark_return']:.2f}% | Excess: {episode_info['excess_return']:.2f}%")
            print(f"  Rolling Avg Excess: {rolling_avg_return:.2f}%")
            print(f"  Fees: ${episode_info['total_fees']:.2f} | Turnover: {episode_info['avg_turnover']*100:.1f}%")
            print(f"  Win Rate: {episode_info['win_rate']:.1f}% | Outperform Rate: {episode_info['outperform_rate']:.1f}%")
            print(f"  Loss: {avg_total_loss:.4f} (A:{avg_alloc_loss:.4f} V:{avg_value_loss:.4f} E:{avg_entropy:.4f})")
            print(f"  Replay buffer: {len(replay_buffer)} experiences")
            if nan_count > 0:
                print(f"  NaN batches rolled back: {nan_count}")
        
        if len(rolling_returns) >= rolling_window and rolling_avg_return > best_excess_return:
            best_excess_return = rolling_avg_return
            wandb.log({'best_excess_return': best_excess_return})
        
        if avg_total_loss > 0:
            loss_history.append(avg_total_loss)
            if len(loss_history) > LOSS_WINDOW_SIZE:
                loss_history.pop(0)
            
            if len(loss_history) >= LOSS_WINDOW_SIZE:
                current_avg_loss = np.mean(loss_history)
                if current_avg_loss < best_avg_loss:
                    best_avg_loss = current_avg_loss
                    model.save_weights(BEST_MODEL_PATH)
                    print(f"  New best model saved! Avg loss: {best_avg_loss:.6f}")
                    wandb.log({'best_avg_loss': best_avg_loss})
        
        if episode % 20 == 0 and episode > 0:
            replay_buffer.save(REPLAY_BUFFER_PATH)
            print(f"  Replay buffer saved ({len(replay_buffer)} experiences)")
    
    replay_buffer.save(REPLAY_BUFFER_PATH)
    print(f"Saved replay buffer with {len(replay_buffer)} experiences")
    
    wandb.finish()
    print("\nFinetuning completed!")
    print(f"Best rolling excess return: {best_excess_return:.2f}%")
    print(f"Best avg loss: {best_avg_loss:.6f}")
    print(f"Model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    finetune()
