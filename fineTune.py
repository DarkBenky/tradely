import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json
import wandb
import pickle
import gc
from collections import deque

from evolve import Population

tf.keras.mixed_precision.set_global_policy('mixed_float16')

NUM_PREDICTIONS = 8
REBALANCE_INTERVAL = 36
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
MAX_EPISODES = 10000
STEPS_PER_EPISODE = 32
FEE_RATE = 0.001
GRADIENT_CLIP_NORM = 0.5
ROLLBACK_ON_NAN = True
MONTE_CARLO_ENABLED = False
WANDB_PROJECT = "portfolio-finetune"
BEST_MODEL_PATH = "models/best_pretrain_transformer.weights.h5"
CONFIG_PATH = "models/model_config.json"
NORM_STATS_PATH = "models/normalization_stats.json"
LOSS_WINDOW_SIZE = 5
ENTROPY_WEIGHT = 0.01
VALUE_LOSS_WEIGHT = 0.5
POLICY_LOSS_WEIGHT = 2.0
REWARD_CLIP_VALUE = 3.0
EXPLORATION_NOISE = 0.1
EXPLORATION_DECAY = 0.999

REPLAY_BUFFER_SIZE = 50000
REPLAY_BUFFER_PATH = "models/replay_buffer.pkl"
REPLAY_MIN_SIZE = 512
REPLAY_RATIO = 0.5

POPULATION_SIZE = 6
ELITE_COUNT = 2
EPISODES_PER_MEMBER = 5
MUTATION_RATE = 0.02
MUTATION_DECAY = 0.995
POPULATION_DIR = "models/population"


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
        
        if bucket_probs is None or np.any(np.isnan(bucket_probs)):
            bucket_probs = np.ones(10) / 10.0
            pred_reward = 0.0
        
        if hasattr(self, 'exploration_noise') and self.exploration_noise > 0:
            noise = np.random.dirichlet(np.ones(len(allocation)) * 0.3)
            allocation = (1 - self.exploration_noise) * allocation + self.exploration_noise * noise
            allocation = allocation / (np.sum(allocation) + 1e-8)
        
        return allocation, allocation_std, pred_reward, pred_reward_std, bucket_probs
    
    def train_step(self, obs_batch, actions_batch, rewards_batch, values_batch, benchmark_batch=None):
        self.last_checkpoint = save_weights_checkpoint(self.model)
        
        self.reward_normalizer.update(rewards_batch)
        
        obs_norm = tf.constant(self.normalizer.normalize(obs_batch).astype(np.float32))
        actions_tf = tf.constant(actions_batch.astype(np.float32))
        rewards_tf = tf.constant(rewards_batch.astype(np.float32))
        
        advantages = (rewards_tf - self.reward_normalizer.mean) / (np.sqrt(self.reward_normalizer.var) + 1e-8)
        advantages = tf.clip_by_value(advantages, -REWARD_CLIP_VALUE, REWARD_CLIP_VALUE)
        
        bucket_centers = tf.constant(
            np.array([-350, -175, -62.5, -12.5, 12.5, 37.5, 75, 150, 300, 500], dtype=np.float32)
        )
        bucket_boundaries = tf.constant(
            np.array([-250, -100, -25, 0, 25, 50, 100, 200, 400], dtype=np.float32)
        )
        
        with tf.GradientTape() as tape:
            alloc_pred, reward_pred = self.model(obs_norm, training=True)
            
            log_probs = tf.reduce_sum(actions_tf * tf.math.log(tf.clip_by_value(alloc_pred, 1e-8, 1.0)), axis=-1)
            policy_loss = -tf.reduce_mean(log_probs * advantages)
            
            pred_values = tf.reduce_sum(reward_pred * bucket_centers, axis=-1)
            value_loss = tf.reduce_mean(tf.square((pred_values - rewards_tf) / 100.0))
            
            reward_bucket_indices = tf.zeros_like(rewards_tf, dtype=tf.int32)
            for i in range(9):
                reward_bucket_indices = tf.where(
                    rewards_tf >= bucket_boundaries[i],
                    tf.ones_like(reward_bucket_indices) * (i + 1),
                    reward_bucket_indices
                )
            reward_onehot = tf.one_hot(reward_bucket_indices, depth=10)
            bucket_ce_loss = -tf.reduce_mean(tf.reduce_sum(reward_onehot * tf.math.log(tf.clip_by_value(reward_pred, 1e-8, 1.0)), axis=-1))
            
            value_loss = value_loss + bucket_ce_loss
            
            entropy = -tf.reduce_sum(alloc_pred * tf.math.log(tf.clip_by_value(alloc_pred, 1e-6, 1.0)), axis=-1)
            entropy_bonus = tf.reduce_mean(entropy)
            
            total_loss = (POLICY_LOSS_WEIGHT * policy_loss + 
                         VALUE_LOSS_WEIGHT * value_loss -
                         ENTROPY_WEIGHT * entropy_bonus)
        
        if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss):
            if ROLLBACK_ON_NAN and self.last_checkpoint is not None:
                restore_weights_checkpoint(self.model, self.last_checkpoint)
            return None, None, None, None, True
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        has_nan_grad = False
        for g in gradients:
            if g is not None and tf.reduce_any(tf.math.is_nan(g)):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            if ROLLBACK_ON_NAN and self.last_checkpoint is not None:
                restore_weights_checkpoint(self.model, self.last_checkpoint)
            return None, None, None, None, True
        
        gradients, _ = tf.clip_by_global_norm(gradients, GRADIENT_CLIP_NORM)
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        if check_weights_for_nan(self.model):
            if ROLLBACK_ON_NAN and self.last_checkpoint is not None:
                restore_weights_checkpoint(self.model, self.last_checkpoint)
            return None, None, None, None, True
        
        return (float(total_loss.numpy()), 
                float(policy_loss.numpy()), 
                float(value_loss.numpy()), 
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
            'holding_ratios': holding_ratios,
            'max_allocation': float(np.max(avg_allocation)),
            'allocation_std': float(np.std(avg_allocation))
        }
        
        return (np.array(observations), np.array(actions), 
                np.array(rewards), np.array(values), np.array(benchmark_rewards), episode_info)


def build_model_from_config(config):
    from preTrain import (build_model_temporal_cnn, build_model_iterative_refine,
                          build_model, build_model_v2, build_model_lstm,
                          build_model_dense, build_model_cnn, build_model_hybrid_attention)
    
    sequence_length = config['sequence_length']
    features_per_timestep = config['features_per_timestep']
    num_assets = config['num_assets']
    model_type = config.get('model_type', 'temporal-cnn')
    
    if model_type == 'iterative-refine':
        return build_model_iterative_refine(
            sequence_length, features_per_timestep, num_assets,
            cnn_filters=config.get('cnn_filters', 512),
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'temporal-cnn':
        return build_model_temporal_cnn(
            sequence_length, features_per_timestep, num_assets,
            d_model=config['d_model'],
            cnn_filters=config.get('cnn_filters', 512),
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'v2-cnn-transformer-cnn':
        return build_model_v2(
            sequence_length, features_per_timestep, num_assets,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_blocks=config['num_layers'],
            cnn_filters=config.get('cnn_filters', 512),
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'hybrid-attention':
        return build_model_hybrid_attention(
            sequence_length, features_per_timestep, num_assets,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            cnn_filters=config.get('cnn_filters', 512),
            dropout_rate=config['dropout_rate']
        )
    else:
        return build_model(
            sequence_length, features_per_timestep, num_assets,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout_rate=config['dropout_rate']
        )


def finetune():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPUs available: {len(physical_devices)}")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    config = load_model_config()
    model_type = config.get('model_type', 'temporal-cnn')
    
    print(f"Loading model type: {model_type}")
    model = build_model_from_config(config)
    
    normalizer = ObservationNormalizer(config['features_per_timestep'])
    if normalizer.load(NORM_STATS_PATH):
        print(f"Loaded normalization stats (count: {normalizer.count})")
    
    from portfolio_env import PortfolioEnv, ModelType
    env = PortfolioEnv(obs_shape=ModelType.TRANSFORMER_SHAPE, rebalancing_interval=REBALANCE_INTERVAL)
    env.fee_rate = FEE_RATE
    
    replay_buffer = ReplayBuffer(max_size=REPLAY_BUFFER_SIZE)
    if os.path.exists(REPLAY_BUFFER_PATH):
        if replay_buffer.load(REPLAY_BUFFER_PATH):
            print(f"Loaded replay buffer with {len(replay_buffer)} experiences")
    
    population = Population(
        population_size=POPULATION_SIZE,
        elite_count=ELITE_COUNT,
        population_dir=POPULATION_DIR,
        mutation_rate=MUTATION_RATE,
        mutation_decay=MUTATION_DECAY
    )
    
    print(f"\nInitializing population of {POPULATION_SIZE} members...")
    population.initialize(model, BEST_MODEL_PATH)
    print("Population initialized")
    
    wandb.init(project=WANDB_PROJECT, config={
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'max_episodes': MAX_EPISODES,
        'steps_per_episode': STEPS_PER_EPISODE,
        'fee_rate': FEE_RATE,
        'gradient_clip_norm': GRADIENT_CLIP_NORM,
        'entropy_weight': ENTROPY_WEIGHT,
        'policy_loss_weight': POLICY_LOSS_WEIGHT,
        'value_loss_weight': VALUE_LOSS_WEIGHT,
        'exploration_noise': EXPLORATION_NOISE,
        'exploration_decay': EXPLORATION_DECAY,
        'replay_buffer_size': REPLAY_BUFFER_SIZE,
        'replay_min_size': REPLAY_MIN_SIZE,
        'replay_ratio': REPLAY_RATIO,
        'population_size': POPULATION_SIZE,
        'elite_count': ELITE_COUNT,
        'episodes_per_member': EPISODES_PER_MEMBER,
        'mutation_rate': MUTATION_RATE,
        'model_config': config
    })
    
    for m in range(POPULATION_SIZE):
        wandb.define_metric(f'member_{m}/return', step_metric='member_step')
        wandb.define_metric(f'member_{m}/value', step_metric='member_step')
    wandb.define_metric('fitness/*', step_metric='generation')
    wandb.define_metric('evolution/*', step_metric='generation')
    
    best_excess_return = -float('inf')
    global_episode = 0
    generation = 0
    
    print(f"\nStarting evolutionary RL training...")
    print(f"  Population: {POPULATION_SIZE} members, {ELITE_COUNT} elites")
    print(f"  Episodes per member: {EPISODES_PER_MEMBER}")
    print(f"  Steps per episode: {STEPS_PER_EPISODE}")
    print(f"  Replay buffer: {len(replay_buffer)} experiences")
    
    while global_episode < MAX_EPISODES:
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")
        
        optimizer = keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
        trainer = OnlineTrainer(model, normalizer, optimizer, replay_buffer,
                                uses_scratchpad=(model_type == 'iterative-refine'))
        
        for member_idx in range(POPULATION_SIZE):
            member = population.members[member_idx]
            
            model.load_weights(member.weight_path)
            trainer.exploration_noise = EXPLORATION_NOISE * (EXPLORATION_DECAY ** global_episode)
            
            member_excess_returns = []
            member_losses = []
            member_step_counter = 0
            
            for ep in range(EPISODES_PER_MEMBER):
                observations, actions, rewards, values, benchmark_rewards, episode_info = trainer.run_episode(env, log_steps=False)
                
                initial_value = episode_info['initial_portfolio']
                cumulative_reward = 0.0
                for step_idx, reward in enumerate(rewards):
                    cumulative_reward += reward
                    current_value = initial_value + cumulative_reward
                    pct_return = (current_value - initial_value) / initial_value * 100
                    
                    wandb.log({
                        f'member_{member_idx}/return': pct_return,
                        f'member_{member_idx}/value': current_value,
                        'member_step': member_step_counter,
                        'current_member': member_idx,
                        'current_generation': generation
                    })
                    member_step_counter += 1
                
                replay_buffer.add_batch(observations, actions, rewards, values, benchmark_rewards)
                
                if len(replay_buffer) >= REPLAY_MIN_SIZE:
                    n_batches = max(1, len(observations) // BATCH_SIZE)
                    
                    indices = np.arange(len(observations))
                    np.random.shuffle(indices)
                    for start_idx in range(0, min(len(observations), n_batches * BATCH_SIZE), BATCH_SIZE):
                        end_idx = min(start_idx + BATCH_SIZE, len(observations))
                        if end_idx - start_idx < BATCH_SIZE:
                            continue
                        batch_indices = indices[start_idx:end_idx]
                        
                        result = trainer.train_step(
                            observations[batch_indices],
                            actions[batch_indices],
                            rewards[batch_indices],
                            values[batch_indices],
                            benchmark_rewards[batch_indices]
                        )
                        if result[0] is not None:
                            member_losses.append(result[0])
                    
                    n_replay = int(n_batches * REPLAY_RATIO / (1 - REPLAY_RATIO)) if REPLAY_RATIO < 1 else n_batches
                    for _ in range(n_replay):
                        obs_batch, act_batch, rew_batch, val_batch, bench_batch = replay_buffer.sample(BATCH_SIZE)
                        result = trainer.train_step(obs_batch, act_batch, rew_batch, val_batch, bench_batch)
                        if result[0] is not None:
                            member_losses.append(result[0])
                
                member_excess_returns.append(episode_info['excess_return'])
                population.increment_episodes(member_idx)
                global_episode += 1
            
            model.save_weights(member.weight_path)
            
            avg_excess = np.mean(member_excess_returns)
            population.update_fitness(member_idx, avg_excess)
            
            avg_loss = np.mean(member_losses) if member_losses else 0.0
            final_portfolio = episode_info['final_portfolio']
            final_return = episode_info['portfolio_return']
            
            print(f"  Member {member_idx}: fitness={member.fitness:.2f}% excess={avg_excess:.2f}% loss={avg_loss:.4f}")
        
        generation_data = {'generation': generation}
        for idx, member in enumerate(population.members):
            generation_data[f'fitness/member_{idx}'] = member.fitness
        wandb.log(generation_data)
        
        evolve_stats = population.evolve(model, save_best_path=BEST_MODEL_PATH)
        
        if evolve_stats['best_fitness'] > best_excess_return:
            best_excess_return = evolve_stats['best_fitness']
        
        wandb.log({
            'evolution/best_fitness': evolve_stats['best_fitness'],
            'evolution/avg_fitness': evolve_stats['avg_fitness'],
            'evolution/worst_fitness': evolve_stats['worst_fitness'],
            'evolution/best_ever_fitness': evolve_stats['best_ever_fitness'],
            'evolution/mutation_rate': evolve_stats['mutation_rate'],
            'evolution/fitness_spread': evolve_stats['best_fitness'] - evolve_stats['worst_fitness'],
            'replay_buffer_size': len(replay_buffer),
            'global_episode': global_episode
        })
        
        print(f"\n  Generation {generation} summary:")
        print(f"    Best: {evolve_stats['best_fitness']:.2f}% | Avg: {evolve_stats['avg_fitness']:.2f}% | Worst: {evolve_stats['worst_fitness']:.2f}%")
        print(f"    Best ever: {evolve_stats['best_ever_fitness']:.2f}%")
        print(f"    Mutation rate: {evolve_stats['mutation_rate']:.4f}")
        print(f"    Replay buffer: {len(replay_buffer)} experiences")
        
        if generation % 5 == 0:
            replay_buffer.save(REPLAY_BUFFER_PATH)
            print(f"    Replay buffer saved")
        
        generation += 1
        
        del optimizer
        del trainer
        gc.collect()
        tf.keras.backend.clear_session()
        model = build_model_from_config(config)
    
    replay_buffer.save(REPLAY_BUFFER_PATH)
    print(f"\nSaved replay buffer with {len(replay_buffer)} experiences")
    
    wandb.finish()
    print("\nEvolutionary RL training completed!")
    print(f"Best fitness: {best_excess_return:.2f}%")
    print(f"Generations: {generation}")
    print(f"Model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    finetune()
