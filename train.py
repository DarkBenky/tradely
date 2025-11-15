import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
from portfolio_env import PortfolioEnv
import os
import wandb
import time
from collections import deque
import pickle
import gc
import random

DEBUG = False

try:
    from feature_reducer import FeatureReducer
    FEATURE_REDUCER = None
    TRAIN_FEATURE_REDUCER = False
    print("Feature reduction will be handled by model's built-in compression layers")
except Exception as e:
    FEATURE_REDUCER = None
    TRAIN_FEATURE_REDUCER = False
    print(f"Using model's built-in feature compression")

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(rate),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class FeatureExtractor(layers.Layer):
    def __init__(self, output_dim, rate=0.1):
        super().__init__()
        self.dense1 = layers.Dense(output_dim * 2, activation='gelu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dense2 = layers.Dense(output_dim, activation='gelu')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(rate * 0.5)
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        return x

def build_model(obs_shape, num_assets, config=None, feature_reducer=None):
    """Build actor-only model (REINFORCE) - no critic"""
    if config is None:
        config = {
            "dropout_rate": 0.15,
            "compression_layers": [8192, 2048],
            "compression_dropout": 0.1,
            "shared_layers": [1024, 768, 512, 384, 256],
            "residual_connections": [True, True, True, False],
            "actor_layers": [192, 128],
        }
    
    compression_layers = config.get("compression_layers", [8192, 2048])
    compression_dropout = config.get("compression_dropout", 0.1)
    shared_layers = config.get("shared_layers", [1024, 768, 512, 384, 256])
    residual_connections = config.get("residual_connections", [True, True, True, False])
    actor_layers = config.get("actor_layers", [192, 128])
    dropout_rate = config.get("dropout_rate", 0.15)
    
    inputs = layers.Input(shape=obs_shape)
    
    if obs_shape[0] > 10000 and compression_layers:
        x = inputs
        for i, units in enumerate(compression_layers):
            x = layers.Dense(units, activation='relu', name=f'compression_layer_{i+1}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(compression_dropout)(x)
    else:
        x = inputs
    
    while len(residual_connections) < len(shared_layers):
        residual_connections.append(False)
    
    for i, units in enumerate(shared_layers):
        use_residual = residual_connections[i] if i < len(residual_connections) else False
        
        if use_residual and i > 0:
            residual = x
            if x.shape[-1] != units:
                residual = layers.Dense(units, activation='linear')(residual)
            
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Dense(units, activation='linear')(x)
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)
        else:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
    
    # Actor head - portfolio allocation
    actor_x = x
    for i, units in enumerate(actor_layers):
        actor_x = layers.Dense(units, activation='relu', name=f'actor_dense_{i+1}')(actor_x)
        if i < len(actor_layers) - 1:
            actor_x = layers.BatchNormalization()(actor_x)
        actor_x = layers.Dropout(dropout_rate)(actor_x)
    actor_output = layers.Dense(num_assets, activation='softmax', name='portfolio_allocation')(actor_x)
    
    # Confidence head - per-asset confidence (0-1 for each asset)
    confidence_x = x
    confidence_x = layers.Dense(128, activation='relu', name='confidence_dense_1')(confidence_x)
    confidence_x = layers.BatchNormalization()(confidence_x)
    confidence_x = layers.Dropout(dropout_rate)(confidence_x)
    confidence_output = layers.Dense(num_assets, activation='sigmoid', name='confidence')(confidence_x)
    
    # Return both outputs: allocation and per-asset confidence
    model = Model(inputs=inputs, outputs=[actor_output, confidence_output])
    return model

def collect_batch_data(env, model, batch_size, gamma, current_step, epsilon=0.1, env_reset_probability=0.05, feature_reducer=None, config=None):
    states, raw_states, actions, rewards, is_random_flags, confidences = [], [], [], [], [], []
    per_asset_returns = []
    step_logs = []
    
    max_reset_attempts = 3
    reset_attempt = 0
    
    while reset_attempt < max_reset_attempts:
        if current_step >= env.max_steps or random.random() < env_reset_probability or reset_attempt > 0:
            state = env.reset()
        else:
            state = env.get_observation()
        
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"WARNING: Invalid state after reset (attempt {reset_attempt + 1}/{max_reset_attempts}), retrying...")
            reset_attempt += 1
            env_reset_probability_temp = 1.0
            continue
        else:
            break
    
    if reset_attempt >= max_reset_attempts:
        print("ERROR: Failed to get valid state after multiple resets, returning empty batch")
        return [], [], [], np.array([]), [], {
            'steps_collected': 0,
            'total_reward': 0.0,
            'portfolio_value': env._portfolio_portfolio_value(),
            'benchmark_value': env._update_benchmark_value(),
            'outperformance': 1.0,
            'done': True,
            'random_actions': 0,
            'policy_actions': 0
        }, []
    
    raw_state = state.copy()
    
    if feature_reducer is not None:
        state = feature_reducer.transform(state.reshape(1, -1))[0]
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"WARNING: Feature reducer produced NaN/Inf, cleaning...")
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
    
    done = False
    steps_collected = 0
    
    print(f"  Collecting {batch_size} steps for batch (epsilon={epsilon:.4f})...")
    
    while steps_collected < batch_size and not done:
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"WARNING: NaN/Inf detected in state at step {steps_collected}, resetting episode")
            state = env.reset()
            raw_state = state.copy()
            if feature_reducer is not None:
                state = feature_reducer.transform(state.reshape(1, -1))[0]
            
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print(f"ERROR: State still invalid after reset, ending batch early")
                done = True
                break
            continue
        
        # Check if this is a rebalancing step (model should make a decision)
        # env.steps_since_last_rebalance will be 0 after reset, or >= rebalance_interval when it's time
        is_rebalancing_step = (env.steps_since_last_rebalance >= env.rebalance_interval or env.last_action is None)
        
        if is_rebalancing_step:
            num_samples = config.get('num_stochastic_samples', 16) if config else 16
            sampled_actions = []
            sampled_confidences = []
            
            for _ in range(num_samples):
                action_probs, confidence = model(state_tensor, training=True)
                sampled_action = tf.random.categorical(tf.math.log(action_probs + 1e-8), 1)
                sampled_action_onehot = tf.one_hot(sampled_action[0], depth=len(env.asset_names))
                mixed_action = 0.7 * action_probs[0].numpy() + 0.3 * sampled_action_onehot.numpy()
                mixed_action = mixed_action / mixed_action.sum()
                sampled_actions.append(mixed_action)
                sampled_confidences.append(confidence[0].numpy())
            
            sampled_actions = np.array(sampled_actions)
            sampled_confidences = np.array(sampled_confidences)
            
            conf_weights = np.mean(sampled_confidences, axis=1)
            conf_weights = conf_weights / (conf_weights.sum() + 1e-8)
            
            action = np.average(sampled_actions, axis=0, weights=conf_weights)
            action = action / action.sum()
            conf_values = np.mean(sampled_confidences, axis=0)
            
            mean_conf = float(np.mean(conf_values))

            print("="*90)
            print(f"Step {current_step + steps_collected}: Mean Confidence = {mean_conf:.4f}")
            print(f"Per-asset confidence: min={np.min(conf_values):.3f}, max={np.max(conf_values):.3f}")
            print("="*90)
            
            equal_weight = np.ones_like(action) / len(action)
            
            for i in range(len(action)):
                conf = conf_values[i]
                blend_factor = 1.0 - conf
                action[i] = conf * action[i] + blend_factor * equal_weight[i]
            
            action = action / action.sum()
            
            if np.any(np.isnan(action)):
                print(f"WARNING: NaN in model output (action), using random action")
                action = env.sample()
                is_random = True
                conf_values = np.zeros_like(action)
                mean_conf = 0.0
            else:
                is_random = False
            
            if random.random() < epsilon and not is_random:
                action = env.sample()
                is_random = True
            elif random.random() < epsilon * 2 and not is_random:
                noise = np.random.normal(0, 0.1, size=action.shape)
                action = action + noise
                action = np.clip(action, 0, 1)
                action = action / action.sum()
        else:
            action = env.sample()  # Dummy action, won't be used
            is_random = False  # Mark as not random (it's a holding period)
            conf_values = np.zeros(len(env.asset_names))  # Dummy confidence
            mean_conf = 0.0
        
        next_state, reward, done, info = env.step(action)
        
        if np.isnan(reward) or np.isinf(reward):
            print(f"WARNING: NaN/Inf reward detected, replacing with 0.0")
            reward = 0.0
        elif abs(reward) > 20:
            print(f"WARNING: Extreme reward detected: {reward:.2f}, clipping to +-10")
            reward = np.clip(reward, -10.0, 10.0)

        env.render()
        
        if is_rebalancing_step:
            asset_rewards = np.zeros(len(env.asset_names))
            portfolio_layout = info.get('portfolio_layout', {})
            
            # Ensure action is flat for indexing
            action_flat = np.array(action).flatten()
            
            for idx, symbol in enumerate(env.asset_names):
                if symbol in env.df:
                    current_price = env.df[symbol]['close'].iloc[env.step_count - 1]
                    prev_price = env.df[symbol]['close'].iloc[max(0, env.step_count - 2)]
                    
                    if prev_price > 0:
                        price_change = (current_price - prev_price) / prev_price
                        allocation = float(action_flat[idx])
                        asset_rewards[idx] = float(price_change * allocation * 100)
        else:
            asset_rewards = np.zeros(len(env.asset_names))
        
        raw_next_state = next_state.copy()
        
        if feature_reducer is not None:
            next_state = feature_reducer.transform(next_state.reshape(1, -1))[0]

        if is_rebalancing_step:
            states.append(state)
            raw_states.append(raw_state)
            actions.append(np.array(action).flatten())
            rewards.append(float(reward))
            confidences.append(np.array(conf_values).flatten())
            is_random_flags.append(bool(is_random))
            per_asset_returns.append(np.array(asset_rewards).flatten())
        
        portfolio_value = info.get('portfolio_value', 1.0)
        current_allocations = {}
        for symbol in env.df.keys():
            holdings = info.get('portfolio_layout', {}).get(symbol, 0)
            price = env.df[symbol]['close'].iloc[env.step_count - 1]
            value = holdings * price
            current_allocations[symbol] = value / portfolio_value if portfolio_value > 0 else 0
        
        cash_allocation = info.get('cash_allocation', 0)
        
        if is_rebalancing_step:
            log_confidence = mean_conf
            log_confidence_min = float(np.min(conf_values))
            log_confidence_max = float(np.max(conf_values))
            log_confidence_std = float(np.std(conf_values))
            log_action_mean = np.mean(action)
            log_action_std = np.std(action)
            log_action_max = np.max(action)
            log_per_asset_confidence = conf_values
            log_per_asset_rewards = asset_rewards
        else:
            log_confidence = None
            log_confidence_min = None
            log_confidence_max = None
            log_confidence_std = None
            allocation_values = list(current_allocations.values()) + [cash_allocation]
            log_action_mean = np.mean(allocation_values) if allocation_values else 0
            log_action_std = np.std(allocation_values) if allocation_values else 0
            log_action_max = np.max(allocation_values) if allocation_values else 0
            log_per_asset_confidence = None
            log_per_asset_rewards = None
        
        step_log = {
            'step_global': current_step + steps_collected,
            'step_batch': steps_collected,
            'reward': reward,
            'confidence': log_confidence,
            'confidence_min': log_confidence_min,
            'confidence_max': log_confidence_max,
            'confidence_std': log_confidence_std,
            'portfolio_value': info.get('portfolio_value', 0),
            'benchmark_value': info.get('benchmark_value', 0),
            'outperformance': info.get('outperformance', 0),
            'action_mean': log_action_mean,
            'action_std': log_action_std,
            'action_max': log_action_max,
            'done': done,
            'portfolio_layout': info.get('portfolio_layout', {}),
            'per_asset_confidence': log_per_asset_confidence,
            'per_asset_rewards': log_per_asset_rewards,
            'was_rebalancing_step': is_rebalancing_step,
            'current_allocations': current_allocations,
            'cash_allocation': cash_allocation,
        }
        
        step_logs.append(step_log)
        
        state = next_state
        raw_state = raw_next_state
        steps_collected += 1
        
        if steps_collected >= 10000:
            print(f"  Warning: Episode exceeded 10000 steps, forcing termination")
            break
    
    returns = np.array([], dtype=np.float32)
    per_asset_returns_cumulative = []
    
    if steps_collected > 0:
        rewards_array = np.array(rewards, dtype=np.float32)
        
        if np.any(np.isnan(rewards_array)):
            print(f"WARNING: NaN in rewards array, replacing with zeros")
            rewards_array = np.nan_to_num(rewards_array, nan=0.0)
            rewards = rewards_array.tolist()
        
        returns = []
        asset_returns_list = []
        G = 0.0
        asset_G = np.zeros(len(env.asset_names))
        
        for i in reversed(range(len(rewards))):
            G = rewards[i] + gamma * G
            asset_G = per_asset_returns[i] + gamma * asset_G
            
            if np.isnan(G) or np.isinf(G):
                print(f"WARNING: NaN/Inf in return calculation at step {i}, resetting to 0")
                G = 0.0
            
            if np.any(np.isnan(asset_G)) or np.any(np.isinf(asset_G)):
                asset_G = np.nan_to_num(asset_G, nan=0.0, posinf=1.0, neginf=-1.0)
            
            returns.insert(0, G)
            asset_returns_list.insert(0, asset_G.copy())
        
        returns = np.array(returns, dtype=np.float32)
        per_asset_returns_cumulative = np.array(asset_returns_list, dtype=np.float32)
        
        if np.any(np.isnan(returns)):
            print("WARNING: NaN in returns, replacing with zeros")
            returns = np.nan_to_num(returns, nan=0.0)

    train_data_dir = "/media/user/HDD 1TB/Data"
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)

    if steps_collected > 0 and len(states) > 0:
        with open(os.path.join(train_data_dir, "training_data.pkl"), 'ab') as f:
            pickle.dump({
                'states': states, 
                'actions': actions, 
                'returns': returns, 
                'is_random': is_random_flags,
                'per_asset_returns': per_asset_returns_cumulative.tolist() if len(per_asset_returns_cumulative) > 0 else []
            }, f)
    
    random_count = sum(is_random_flags)
    policy_count = len(is_random_flags) - random_count
    
    wandb.log({
        'batch/steps_collected': steps_collected,
        'batch/random_actions_count': random_count,
        'batch/policy_actions_count': policy_count,
        'batch/total_reward': sum(rewards) if rewards else 0.0,
        'batch/mean_reward': np.mean(rewards) if rewards else 0.0,
        'batch/std_reward': np.std(rewards) if rewards else 0.0,
        'batch/min_reward': np.min(rewards) if rewards else 0.0,
        'batch/max_reward': np.max(rewards) if rewards else 0.0,
        'batch/mean_return': float(np.mean(returns)) if len(returns) > 0 else 0.0,
        'batch/std_return': float(np.std(returns)) if len(returns) > 0 else 0.0,
    })
    
    if steps_collected == 0:
        batch_info = {
            'steps_collected': 0,
            'total_reward': 0.0,
            'mean_confidence': 0.0,
            'portfolio_value': env._portfolio_portfolio_value(),
            'benchmark_value': env._update_benchmark_value(),
            'outperformance': 1.0,
            'done': done,
            'random_actions': 0,
            'policy_actions': 0
        }
    else:
        all_confidences = np.array(confidences)
        batch_info = {
            'steps_collected': steps_collected,
            'total_reward': sum(rewards),
            'mean_confidence': float(np.mean(all_confidences)) if confidences else 0.0,
            'portfolio_value': info.get('portfolio_value', 1.0) if steps_collected > 0 else 1.0,
            'benchmark_value': info.get('benchmark_value', 1.0) if steps_collected > 0 else 1.0,
            'outperformance': info.get('outperformance', 1.0) if steps_collected > 0 else 1.0,
            'done': done,
            'random_actions': random_count,
            'policy_actions': policy_count
        }
    
    return states, raw_states, actions, returns, confidences, is_random_flags, batch_info, step_logs, per_asset_returns_cumulative

def train_on_batch(model, optimizer, batch_states, batch_raw_states, batch_actions, batch_returns, 
                   batch_confidences, batch_per_asset_returns, is_random_flags, batch_num, feature_reducer=None, 
                   feature_reducer_optimizer=None, clip_ratio=0.2, train_encoder=True):
    states_batch = np.array(batch_states, dtype=np.float32)
    actions_batch = np.array(batch_actions, dtype=np.float32)
    returns_batch = np.array(batch_returns, dtype=np.float32)
    per_asset_returns_batch = np.array(batch_per_asset_returns, dtype=np.float32)
    
    returns_raw = returns_batch.copy()
    
    returns_mean = np.mean(returns_batch)
    returns_std = np.std(returns_batch)
    if returns_std > 1e-6 and not np.isnan(returns_std):
        returns_normalized = (returns_batch - returns_mean) / (returns_std + 1e-8)
    else:
        returns_normalized = returns_batch - returns_mean if not np.isnan(returns_mean) else returns_batch
    
    returns_normalized = np.clip(returns_normalized, -10.0, 10.0)
    
    confidences_batch = np.array(batch_confidences, dtype=np.float32)
    
    asset_returns_mean = np.mean(per_asset_returns_batch, axis=0)
    asset_returns_std = np.std(per_asset_returns_batch, axis=0) + 1e-8
    per_asset_returns_normalized = (per_asset_returns_batch - asset_returns_mean) / asset_returns_std
    per_asset_returns_normalized = np.clip(per_asset_returns_normalized, -5.0, 5.0)
    
    with tf.GradientTape() as tape:
        action_probs, confidence_pred = model(states_batch, training=True)
        action_probs_clipped = tf.clip_by_value(action_probs, 1e-8, 1.0)
        
        log_probs = tf.reduce_sum(actions_batch * tf.math.log(action_probs_clipped), axis=1)
        actor_loss = -tf.reduce_mean(log_probs * returns_normalized)
        
        target_confidence = tf.sigmoid(per_asset_returns_normalized / 2.0)
        target_confidence = tf.cast(target_confidence, confidence_pred.dtype)
        
        confidence_loss = tf.reduce_mean(tf.square(confidence_pred - target_confidence))
        
        entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs_clipped), axis=1))
        
        total_loss = actor_loss + 0.3 * confidence_loss - 0.01 * entropy
    
    grads = tape.gradient(total_loss, model.trainable_variables)
    
    nan_grad_count = sum(1 for g in grads if g is not None and tf.reduce_any(tf.math.is_nan(g)))
    if nan_grad_count > 0:
        print(f"WARNING: {nan_grad_count} gradients contain NaN, skipping update")
        return {
            'batch/loss': 0.0,
            'batch/actor_loss': 0.0,
            'batch/confidence_loss': 0.0,
            'batch/entropy': 0.0,
            'batch/mean_action_prob': 0.0,
            'batch/max_action_prob': 0.0,
            'batch/min_action_prob': 0.0,
            'batch/action_std': 0.0,
            'batch/mean_return': 0.0,
            'batch/std_return': 0.0,
            'batch/mean_confidence': 0.0,
            'batch/grad_norm': 0.0,
            'batch/random_action_ratio': 0.0,
        }
    
    grads = [tf.clip_by_norm(g, 0.5) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    action_probs_np = action_probs.numpy()
    confidence_pred_np = confidence_pred.numpy()
    grad_norm = float(tf.linalg.global_norm(grads).numpy())
    
    metrics = {
        'batch/loss': float(total_loss.numpy()),
        'batch/actor_loss': float(actor_loss.numpy()),
        'batch/confidence_loss': float(confidence_loss.numpy()),
        'batch/entropy': float(entropy.numpy()),
        'batch/mean_action_prob': float(tf.reduce_mean(action_probs).numpy()),
        'batch/max_action_prob': float(tf.reduce_max(action_probs).numpy()),
        'batch/min_action_prob': float(tf.reduce_min(action_probs).numpy()),
        'batch/action_std': float(np.std(action_probs_np)),
        'batch/mean_return_raw': float(np.mean(returns_raw)),
        'batch/std_return_raw': float(np.std(returns_raw)),
        'batch/mean_return_normalized': float(np.mean(returns_normalized)),
        'batch/std_return_normalized': float(np.std(returns_normalized)),
        'batch/mean_confidence': float(np.mean(confidence_pred_np)),
        'batch/std_confidence': float(np.std(confidence_pred_np)),
        'batch/grad_norm': grad_norm,
        'batch/random_action_ratio': float(np.mean(is_random_flags)),
    }
    
    grad_values = [g.numpy() for g in grads if g is not None]
    if grad_values:
        all_grads = np.concatenate([g.flatten() for g in grad_values])
        metrics.update({
            'gradients/mean': float(np.mean(all_grads)),
            'gradients/std': float(np.std(all_grads)),
            'gradients/min': float(np.min(all_grads)),
            'gradients/max': float(np.max(all_grads)),
            'gradients/abs_mean': float(np.mean(np.abs(all_grads))),
        })
        
        if batch_num % 1 == 0:
            metrics['gradients/histogram'] = wandb.Histogram(all_grads)
    
    weight_values = [w.numpy() for w in model.trainable_variables]
    if weight_values:
        all_weights = np.concatenate([w.flatten() for w in weight_values])
        metrics.update({
            'weights/mean': float(np.mean(all_weights)),
            'weights/std': float(np.std(all_weights)),
            'weights/min': float(np.min(all_weights)),
            'weights/max': float(np.max(all_weights)),
            'weights/abs_mean': float(np.mean(np.abs(all_weights))),
        })
        
        if batch_num % 1 == 0:
            metrics['weights/histogram'] = wandb.Histogram(all_weights)
    
    if batch_num % 1 == 0:  # Log every batch
        metrics['action_probs/histogram'] = wandb.Histogram(action_probs_np.flatten())
        metrics['confidence/histogram'] = wandb.Histogram(confidence_pred_np.flatten())
        metrics['returns_raw/histogram'] = wandb.Histogram(returns_raw)
        metrics['returns_normalized/histogram'] = wandb.Histogram(returns_normalized)
    
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)):
            if np.isnan(value):
                print(f"WARNING: NaN detected in metric {key}, replacing with 0.0")
                metrics[key] = 0.0
    
    return metrics

def merge_synthetic_data_files(file1, file2, output_file=None, cleanup=True):
    """
    Merge two synthetic data files into one, cleaning both in the process.
    Handles large files without loading to memory - processes in chunks.
    
    Args:
        file1: Path to first synthetic data file
        file2: Path to second synthetic data file
        output_file: Path for merged file (default: file1 with '_merged' suffix)
        cleanup: If True, remove original files after successful merge
    
    Returns:
        Path to merged file, or None if merge failed
    """
    print(f"\nMERGING SYNTHETIC DATA FILES")
    print(f"{'='*80}")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    
    # Validate input files exist
    if not os.path.exists(file1):
        print(f"ERROR: File 1 does not exist: {file1}")
        return None
    if not os.path.exists(file2):
        print(f"ERROR: File 2 does not exist: {file2}")
        return None
    
    # Set output filename
    if output_file is None:
        output_file = file1.replace('.pkl', '_merged.pkl')
    
    print(f"Output: {output_file}\n")
    
    merged_count = 0
    skipped_count = 0
    total_batches = 0
    
    try:
        with open(output_file, 'wb') as f_out:
            # Process file 1
            print(f"Processing File 1: {file1}")
            file1_batches = 0
            try:
                with open(file1, 'rb') as f_in:
                    while True:
                        try:
                            batch_data = pickle.load(f_in)
                            total_batches += 1
                            file1_batches += 1
                            
                            if isinstance(batch_data, dict):
                                # Clean and merge batch
                                cleaned_batch = _clean_batch_data(batch_data)
                                if cleaned_batch and cleaned_batch['states']:
                                    pickle.dump(cleaned_batch, f_out)
                                    merged_count += len(cleaned_batch['states'])
                                else:
                                    skipped_count += len(batch_data.get('states', []))
                                    
                        except (pickle.UnpicklingError, EOFError) as e:
                            if isinstance(e, pickle.UnpicklingError):
                                print(f"  WARNING: Corrupted record in File 1, skipping...")
                                skipped_count += 1
                                continue
                            else:
                                break
            except Exception as e:
                print(f"ERROR reading File 1: {e}")
                return None
            
            print(f"  File 1: {file1_batches} batches processed\n")
            
            # Process file 2
            print(f"Processing File 2: {file2}")
            file2_batches = 0
            try:
                with open(file2, 'rb') as f_in:
                    while True:
                        try:
                            batch_data = pickle.load(f_in)
                            total_batches += 1
                            file2_batches += 1
                            
                            if isinstance(batch_data, dict):
                                # Clean and merge batch
                                cleaned_batch = _clean_batch_data(batch_data)
                                if cleaned_batch and cleaned_batch['states']:
                                    pickle.dump(cleaned_batch, f_out)
                                    merged_count += len(cleaned_batch['states'])
                                else:
                                    skipped_count += len(batch_data.get('states', []))
                                    
                        except (pickle.UnpicklingError, EOFError) as e:
                            if isinstance(e, pickle.UnpicklingError):
                                print(f"  WARNING: Corrupted record in File 2, skipping...")
                                skipped_count += 1
                                continue
                            else:
                                break
            except Exception as e:
                print(f"ERROR reading File 2: {e}")
                return None
            
            print(f"  File 2: {file2_batches} batches processed\n")
    
    except Exception as e:
        print(f"ERROR writing merged file: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        return None
    
    # Print merge statistics
    print(f"{'='*80}")
    print(f"MERGE STATISTICS")
    print(f"{'='*80}")
    print(f"Total batches: {total_batches}")
    print(f"Clean samples merged: {merged_count}")
    print(f"Corrupted samples skipped: {skipped_count}")
    print(f"Output file: {output_file}")
    
    # Get file sizes
    try:
        file1_size = os.path.getsize(file1) / (1024**2)  # MB
        file2_size = os.path.getsize(file2) / (1024**2)  # MB
        output_size = os.path.getsize(output_file) / (1024**2)  # MB
        print(f"\nFile sizes:")
        print(f"  File 1: {file1_size:.2f} MB")
        print(f"  File 2: {file2_size:.2f} MB")
        print(f"  Total: {file1_size + file2_size:.2f} MB")
        print(f"  Merged: {output_size:.2f} MB (saved {file1_size + file2_size - output_size:.2f} MB)")
    except:
        pass
    
    # Cleanup old files if requested
    if cleanup:
        try:
            os.remove(file1)
            print(f"\nRemoved: {file1}")
        except:
            print(f"\nWARNING: Could not remove {file1}")
        
        try:
            os.remove(file2)
            print(f"Removed: {file2}")
        except:
            print(f"WARNING: Could not remove {file2}")
    
    print(f"\nMerge complete!\n")
    return output_file

def truncate_pickle_at_corruption(pkl_file):
    """
    Find and truncate a pickle file at the first corrupted record.
    This prevents infinite loops when reading corrupted files.
    
    Args:
        pkl_file: Path to pickle file to repair
    
    Returns:
        Number of valid batches saved, or -1 if error
    """
    print(f"\nREPAIRING CORRUPTED PICKLE FILE: {pkl_file}")
    print(f"{'='*80}")
    
    temp_file = pkl_file + '.repaired'
    valid_batches = 0
    
    try:
        with open(pkl_file, 'rb') as f_in:
            with open(temp_file, 'wb') as f_out:
                while True:
                    try:
                        pos = f_in.tell()  # Save position before reading
                        batch_data = pickle.load(f_in)
                        # If successful, write it
                        pickle.dump(batch_data, f_out)
                        valid_batches += 1
                    except (pickle.UnpicklingError, EOFError) as e:
                        if isinstance(e, pickle.UnpicklingError):
                            print(f"Corruption detected at byte position {pos}")
                            print(f"Truncating file: keeping {valid_batches} valid batches")
                            break
                        else:
                            # Normal EOF
                            break
                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        print(f"Truncating file: keeping {valid_batches} valid batches")
                        break
        
        # Replace original with repaired version
        os.replace(temp_file, pkl_file)
        file_size = os.path.getsize(pkl_file) / (1024**2)
        print(f"Repair complete! File size: {file_size:.2f} MB, Valid batches: {valid_batches}")
        print(f"{'='*80}\n")
        return valid_batches
        
    except Exception as e:
        print(f"ERROR repairing file: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return -1


def _clean_batch_data(batch_data):
    """
    Helper function to clean a single batch of data.
    Removes samples with NaN/Inf values.
    
    Args:
        batch_data: Dictionary with 'states', 'actions', 'returns', 'is_random', 'per_asset_returns'
    
    Returns:
        Cleaned batch dictionary, or None if no valid samples
    """
    if not isinstance(batch_data, dict):
        return None
    
    states = batch_data.get('states', [])
    actions = batch_data.get('actions', [])
    returns = batch_data.get('returns', [])
    is_random = batch_data.get('is_random', [])
    per_asset_returns = batch_data.get('per_asset_returns', [])
    
    clean_states = []
    clean_actions = []
    clean_returns = []
    clean_is_random = []
    clean_per_asset = []
    
    for i in range(len(states)):
        state = np.array(states[i], dtype=np.float32)
        action = np.array(actions[i], dtype=np.float32)
        ret = returns[i] if i < len(returns) else 0.0
        is_rand = is_random[i] if i < len(is_random) else False
        per_asset = per_asset_returns[i] if i < len(per_asset_returns) else np.zeros(10)
        
        # Check for NaN/Inf in all fields
        has_nan_state = np.any(np.isnan(state)) or np.any(np.isinf(state))
        has_nan_action = np.any(np.isnan(action)) or np.any(np.isinf(action))
        has_nan_return = np.isnan(ret) or np.isinf(ret)
        
        per_asset_arr = np.array(per_asset, dtype=np.float32)
        has_nan_per_asset = np.any(np.isnan(per_asset_arr)) or np.any(np.isinf(per_asset_arr))
        
        # Skip corrupted samples
        if has_nan_state or has_nan_action or has_nan_return or has_nan_per_asset:
            continue
        
        # Add clean sample to batch
        clean_states.append(states[i])
        clean_actions.append(actions[i])
        clean_returns.append(ret)
        clean_is_random.append(is_rand)
        clean_per_asset.append(per_asset)
    
    # Return cleaned batch if it has data
    if clean_states:
        return {
            'states': clean_states,
            'actions': clean_actions,
            'returns': clean_returns,
            'is_random': clean_is_random,
            'per_asset_returns': clean_per_asset
        }
    return None

def clean_synthetic_data(pkl_file, chunk_size=10000):
    """Clean corrupted NaN/Inf values from synthetic data without loading entire file to memory."""
    print("\nCLEANING SYNTHETIC DATA (removing NaN/Inf values)...")
    
    temp_pkl_file = pkl_file + '.clean'
    cleaned_count = 0
    skipped_count = 0
    total_batches = 0
    
    with open(pkl_file, 'rb') as f_in:
        with open(temp_pkl_file, 'wb') as f_out:
            while True:
                try:
                    batch_data = pickle.load(f_in)
                    total_batches += 1
                    
                    if isinstance(batch_data, dict):
                        states = batch_data.get('states', [])
                        actions = batch_data.get('actions', [])
                        returns = batch_data.get('returns', [])
                        is_random = batch_data.get('is_random', [])
                        per_asset_returns = batch_data.get('per_asset_returns', [])
                        
                        # Filter out samples with NaN/Inf values
                        clean_states = []
                        clean_actions = []
                        clean_returns = []
                        clean_is_random = []
                        clean_per_asset = []
                        
                        for i in range(len(states)):
                            state = np.array(states[i], dtype=np.float32)
                            action = np.array(actions[i], dtype=np.float32)
                            ret = returns[i] if i < len(returns) else 0.0
                            is_rand = is_random[i] if i < len(is_random) else False
                            per_asset = per_asset_returns[i] if i < len(per_asset_returns) else np.zeros(10)
                            
                            # Check for NaN/Inf in state, action, return, and per_asset_returns
                            has_nan_state = np.any(np.isnan(state)) or np.any(np.isinf(state))
                            has_nan_action = np.any(np.isnan(action)) or np.any(np.isinf(action))
                            has_nan_return = np.isnan(ret) or np.isinf(ret)
                            
                            per_asset_arr = np.array(per_asset, dtype=np.float32)
                            has_nan_per_asset = np.any(np.isnan(per_asset_arr)) or np.any(np.isinf(per_asset_arr))
                            
                            if has_nan_state or has_nan_action or has_nan_return or has_nan_per_asset:
                                skipped_count += 1
                                continue
                            
                            # Data is clean, add to batch
                            clean_states.append(states[i])
                            clean_actions.append(actions[i])
                            clean_returns.append(ret)
                            clean_is_random.append(is_rand)
                            clean_per_asset.append(per_asset)
                            cleaned_count += 1
                        
                        # Only write batch if it has data
                        if clean_states:
                            clean_batch = {
                                'states': clean_states,
                                'actions': clean_actions,
                                'returns': clean_returns,
                                'is_random': clean_is_random,
                                'per_asset_returns': clean_per_asset
                            }
                            pickle.dump(clean_batch, f_out)
                            
                except (pickle.UnpicklingError, EOFError) as e:
                    if isinstance(e, pickle.UnpicklingError):
                        print(f"  WARNING: Corrupted pickle record at batch {total_batches}, skipping...")
                        continue
                    else:
                        break
    
    # Replace original file with cleaned version
    os.replace(temp_pkl_file, pkl_file)
    print(f"Data cleaning complete!")
    print(f"  Total batches processed: {total_batches}")
    print(f"  Clean samples saved: {cleaned_count}")
    print(f"  Corrupted samples removed: {skipped_count}")
    print(f"  File updated: {pkl_file}\n")

def behavior_cloning(pkl_file, model, optimizer, chunk_size=10000, batch_size=256, epochs=2):
    print("\nBEHAVIOR CLONING (Supervised Action Learning)")
    print(f"Training whole model to imitate synthetic data actions...")
    print(f"Chunk size: {chunk_size}, Batch size: {batch_size}, epochs: {epochs}")
    
    total_samples = 0
    total_batches = 0
    rollback_count = 0
    best_accuracy = 0.0
    best_loss = float('inf')
    batches_without_improvement = 0
    plateau_patience = 10
    lr_reduction_factor = 0.75
    current_lr = optimizer.learning_rate.numpy()
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    def action_loss(y_true, y_pred):
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-8, 1.0)
        return -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred_clipped), axis=1))
    
    def confidence_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
    
    model.compile(
        optimizer=optimizer,
        loss=[action_loss, confidence_loss],
        loss_weights=[2.5, 0.1],
        metrics=[['accuracy'], ['mae']]
    )
    
    start_time = time.time()
    chunk_num = 0
    
    with open(pkl_file, 'rb') as f:
        chunk_states = []
        chunk_actions = []
        chunk_per_asset_returns = []
        
        while True:
            try:
                batch_data = pickle.load(f)
                if isinstance(batch_data, dict):
                    states = batch_data.get('states', [])
                    actions = batch_data.get('actions', [])
                    per_asset_returns = batch_data.get('per_asset_returns', [])
                    chunk_states.extend(states)
                    chunk_actions.extend(actions)
                    chunk_per_asset_returns.extend(per_asset_returns)
                
                if len(chunk_states) >= chunk_size:
                    chunk_num += 1
                    min_len = min(len(chunk_states), len(chunk_actions), len(chunk_per_asset_returns))
                    chunk_states_arr = np.array(chunk_states[:min_len], dtype=np.float32)
                    chunk_actions_arr = np.array(chunk_actions[:min_len], dtype=np.float32)
                    chunk_per_asset_arr = np.array(chunk_per_asset_returns[:min_len], dtype=np.float32)
                    
                    total_samples += len(chunk_states_arr)
                    print(f"    Chunk {chunk_num}: Training on {len(chunk_states_arr)} samples...")
                    
                    asset_returns_mean = np.mean(chunk_per_asset_arr, axis=0)
                    asset_returns_std = np.std(chunk_per_asset_arr, axis=0) + 1e-8
                    per_asset_normalized = (chunk_per_asset_arr - asset_returns_mean) / asset_returns_std
                    per_asset_normalized = np.clip(per_asset_normalized, -5.0, 5.0)
                    chunk_confidence = 1.0 / (1.0 + np.exp(-per_asset_normalized / 2.0))
                    
                    try:
                        weights_backup = [w.numpy().copy() for w in model.trainable_variables]
                        
                        history = model.fit(
                            chunk_states_arr,
                            [chunk_actions_arr, chunk_confidence],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            shuffle=True
                        )
                        
                        # Check for NaN in weights after training
                        has_nan = False
                        for var in model.trainable_variables:
                            if tf.reduce_any(tf.math.is_nan(var)):
                                has_nan = True
                                break
                        
                        if has_nan:
                            print(f"      WARNING: NaN detected in weights! Rolling back...")
                            for var, backup in zip(model.trainable_variables, weights_backup):
                                var.assign(backup)
                            rollback_count += 1
                            wandb.log({
                                'behavior_cloning/rollback_count': rollback_count,
                                'behavior_cloning/chunk_num': chunk_num,
                            })
                        else:
                            chunk_loss = history.history['portfolio_allocation_loss'][-1]
                            chunk_acc = history.history['portfolio_allocation_accuracy'][-1]
                            chunk_conf_loss = history.history['confidence_loss'][-1]
                            chunk_conf_mae = history.history['confidence_mae'][-1]
                            total_batches += 1

                            improved = False
                            if chunk_acc > best_accuracy:
                                best_accuracy = chunk_acc
                                model.save_weights('models/best_model.weights.h5')
                                print(f"New best accuracy: {chunk_acc:.2%}! Model saved.")
                                improved = True

                            if chunk_loss < best_loss:
                                best_loss = chunk_loss
                                model.save_weights('models/best_model.weights.h5')
                                print(f"New best loss: {chunk_loss:.4f}! Model saved.")
                                improved = True
                            
                            if improved:
                                batches_without_improvement = 0
                            else:
                                batches_without_improvement += 1
                            
                            if batches_without_improvement >= plateau_patience:
                                current_lr *= lr_reduction_factor
                                optimizer.learning_rate.assign(current_lr)
                                print(f"\n{'='*80}")
                                print(f"LEARNING RATE REDUCED ON PLATEAU")
                                print(f"No improvement for {plateau_patience} batches")
                                print(f"New learning rate: {current_lr:.6f} (reduced by {(1-lr_reduction_factor)*100:.0f}%)")
                                print(f"{'='*80}\n")
                                batches_without_improvement = 0
                                
                                wandb.log({
                                    'behavior_cloning/lr_reduction': 1,
                                    'behavior_cloning/learning_rate': current_lr,
                                    'behavior_cloning/batches_without_improvement': batches_without_improvement,
                                })

                            
                            log_dict = {
                                'behavior_cloning/chunk_loss': chunk_loss,
                                'behavior_cloning/chunk_accuracy': chunk_acc,
                                'behavior_cloning/chunk_confidence_loss': chunk_conf_loss,
                                'behavior_cloning/chunk_confidence_mae': chunk_conf_mae,
                                'behavior_cloning/best_accuracy': best_accuracy,
                                'behavior_cloning/best_loss': best_loss,
                                'behavior_cloning/chunk_num': chunk_num,
                                'behavior_cloning/total_batches': total_batches,
                                'behavior_cloning/samples_processed': total_samples,
                                'behavior_cloning/learning_rate': current_lr,
                                'behavior_cloning/batches_without_improvement': batches_without_improvement,
                            }
                            
                            if total_batches % 25 == 0:
                                weight_vals = [w.numpy() for w in model.trainable_variables]
                                all_weights = np.concatenate([w.flatten() for w in weight_vals])
                                log_dict['behavior_cloning/weights_histogram'] = wandb.Histogram(all_weights)
                                log_dict['behavior_cloning/weights_mean'] = float(np.mean(all_weights)),
                                log_dict['behavior_cloning/weights_std'] = float(np.std(all_weights)),
                            
                            wandb.log(log_dict)

                            print(f"Loss: {chunk_loss:.4f}, Accuracy: {chunk_acc:.2%}, Conf Loss: {chunk_conf_loss:.4f}")
                        
                        del weights_backup
                        
                    except Exception as e:
                        print(f"      ERROR training on chunk: {e}")
                        print(f"      Skipping this chunk...")
                    
                    chunk_states = []
                    chunk_actions = []
                    chunk_per_asset_returns = []
                    del chunk_states_arr, chunk_actions_arr, chunk_confidence, chunk_per_asset_arr
                    gc.collect()
            
            except (pickle.UnpicklingError, EOFError):
                break
        
        # Process final chunk if it exists
        if len(chunk_states) > 0:
            chunk_num += 1
            min_len = min(len(chunk_states), len(chunk_actions), len(chunk_per_asset_returns))
            chunk_states_arr = np.array(chunk_states[:min_len], dtype=np.float32)
            chunk_actions_arr = np.array(chunk_actions[:min_len], dtype=np.float32)
            chunk_per_asset_arr = np.array(chunk_per_asset_returns[:min_len], dtype=np.float32)
            
            total_samples += len(chunk_states_arr)
            print(f"    Chunk {chunk_num} (final): Training on {len(chunk_states_arr)} samples...")
            
            asset_returns_mean = np.mean(chunk_per_asset_arr, axis=0)
            asset_returns_std = np.std(chunk_per_asset_arr, axis=0) + 1e-8
            per_asset_normalized = (chunk_per_asset_arr - asset_returns_mean) / asset_returns_std
            per_asset_normalized = np.clip(per_asset_normalized, -5.0, 5.0)
            chunk_confidence = 1.0 / (1.0 + np.exp(-per_asset_normalized / 2.0))
            
            try:
                weights_backup = [w.numpy().copy() for w in model.trainable_variables]
                
                history = model.fit(
                    chunk_states_arr,
                    [chunk_actions_arr, chunk_confidence],
                    batch_size=batch_size,
                    epochs=1,
                    verbose=0,
                    shuffle=True
                )
                
                has_nan = False
                for var in model.trainable_variables:
                    if tf.reduce_any(tf.math.is_nan(var)):
                        has_nan = True
                        break
                
                if has_nan:
                    print(f"      WARNING: NaN detected in weights! Rolling back...")
                    for var, backup in zip(model.trainable_variables, weights_backup):
                        var.assign(backup)
                    rollback_count += 1
                    wandb.log({
                        'behavior_cloning/rollback_count': rollback_count,
                        'behavior_cloning/chunk_num': chunk_num,
                    })
                else:
                    chunk_loss = history.history['portfolio_allocation_loss'][-1]
                    chunk_acc = history.history['portfolio_allocation_accuracy'][-1]
                    chunk_conf_loss = history.history['confidence_loss'][-1]
                    chunk_conf_mae = history.history['confidence_mae'][-1]
                    total_batches += 1
                    
                    improved = False
                    if chunk_acc > best_accuracy:
                        best_accuracy = chunk_acc
                        model.save_weights('models/best_model.weights.h5')
                        print(f"New best accuracy: {chunk_acc:.2%}! Model saved.")
                        improved = True

                    if chunk_loss < best_loss:
                        best_loss = chunk_loss
                        model.save_weights('models/best_model.weights.h5')
                        print(f"New best loss: {chunk_loss:.4f}! Model saved.")
                        improved = True
                    
                    if improved:
                        batches_without_improvement = 0
                    else:
                        batches_without_improvement += 1
                    
                    if batches_without_improvement >= plateau_patience:
                        current_lr *= lr_reduction_factor
                        optimizer.learning_rate.assign(current_lr)
                        print(f"\n{'='*80}")
                        print(f"LEARNING RATE REDUCED ON PLATEAU")
                        print(f"No improvement for {plateau_patience} batches")
                        print(f"New learning rate: {current_lr:.6f} (reduced by {(1-lr_reduction_factor)*100:.0f}%)")
                        print(f"{'='*80}\n")
                        batches_without_improvement = 0
                        
                        wandb.log({
                            'behavior_cloning/lr_reduction': 1,
                            'behavior_cloning/learning_rate': current_lr,
                            'behavior_cloning/batches_without_improvement': batches_without_improvement,
                        })

                    weight_vals = [w.numpy() for w in model.trainable_variables]
                    all_weights = np.concatenate([w.flatten() for w in weight_vals])
                    
                    log_dict = {
                        'behavior_cloning/chunk_loss': chunk_loss,
                        'behavior_cloning/chunk_accuracy': chunk_acc,
                        'behavior_cloning/chunk_confidence_loss': chunk_conf_loss,
                        'behavior_cloning/chunk_confidence_mae': chunk_conf_mae,
                        'behavior_cloning/best_accuracy': best_accuracy,
                        'behavior_cloning/best_loss': best_loss,
                        'behavior_cloning/chunk_num': chunk_num,
                        'behavior_cloning/weights_mean': float(np.mean(all_weights)),
                        'behavior_cloning/weights_std': float(np.std(all_weights)),
                        'behavior_cloning/total_batches': total_batches,
                        'behavior_cloning/samples_processed': total_samples,
                        'behavior_cloning/learning_rate': current_lr,
                        'behavior_cloning/batches_without_improvement': batches_without_improvement,
                    }
                    
                    if total_batches % 25 == 0:
                        log_dict['behavior_cloning/weights_histogram'] = wandb.Histogram(all_weights)
                    
                    wandb.log(log_dict)
                    
                    print(f"Loss: {chunk_loss:.4f}, Accuracy: {chunk_acc:.2%}, Conf Loss: {chunk_conf_loss:.4f}")
                
                del weights_backup
                
            except Exception as e:
                print(f"ERROR training on final chunk: {e}")
            
            del chunk_states_arr, chunk_actions_arr, chunk_confidence, chunk_per_asset_arr
            gc.collect()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"BEHAVIOR CLONING SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Total samples: {total_samples}")
    print(f"Total chunks: {chunk_num}")
    print(f"Best accuracy: {best_accuracy:.2%}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Rollbacks: {rollback_count}")
    print(f"{'='*80}")
    
    gc.collect()
    
    print(f"\n  Behavior cloning complete!")
    print(f"  Total batches: {total_batches}, Total rollbacks: {rollback_count}")
    print(f"  Best accuracy achieved: {best_accuracy:.2%}")
    print(f"  Best model saved to: models/best_model.weights.h5")
    print(f"  Model trained end-to-end with actions as confidence targets.\n")
    
    wandb.log({
        'behavior_cloning/final_best_accuracy': best_accuracy,
        'behavior_cloning/final_best_loss': best_loss,
        'behavior_cloning/final_total_batches': total_batches,
        'behavior_cloning/final_total_rollbacks': rollback_count,
        'behavior_cloning/final_epochs': epochs,
    })
    
    gc.collect()

DISABLE_CHECKING = False

def offline_pretrain(env, model, optimizer, num_episodes=50, gamma=0.99, batch_size=256, epochs=20, chunk_size=10000):
    print("\nOFFLINE PRETRAINING (MEMORY OPTIMIZED)")
    
    pkl_files = [
        '/media/user/HDD 1TB/Data/synthetic_training_data_compressed.pkl',
        '/media/user/HDD 1TB/Data/synthetic_training_data.pkl'
    ]
    
    pkl_file = None
    for f in pkl_files:
        if os.path.exists(f):
            pkl_file = f
            break
    
    if pkl_file is None:
        print("No pretraining data found, skipping pretraining")
        return
    
    # Phase 0: Repair any corrupted pickle file BEFORE cleaning/training
    if not DISABLE_CHECKING:
        print(f"\n{'='*80}")
        print(f"PHASE 0: PICKLE FILE REPAIR (checking for corruption)")
        print(f"{'='*80}")
        truncate_pickle_at_corruption(pkl_file)
    else:
        print("\nPICKLE FILE REPAIR DISABLED, SKIPPING...\n")
    
    # Phase 1: Clean synthetic data first (if epochs > 0)
    if not DISABLE_CHECKING:
        if epochs > 0:
            print(f"\n{'='*80}")
            print(f"PHASE 1: DATA CLEANING (removing NaN/Inf values)")
            print(f"{'='*80}")
            clean_synthetic_data(pkl_file, chunk_size)
    else:
        print("\nDATA CLEANING DISABLED, SKIPPING...\n")
    
    # Phase 1.5: Behavior Cloning (supervised learning on actions)
    print(f"\n{'='*80}")
    print(f"PHASE 1.5: BEHAVIOR CLONING (imitation learning from synthetic data)")
    print(f"{'='*80}")
    behavior_cloning(pkl_file, model, optimizer, chunk_size, batch_size, epochs=epochs)
    
    print(f"{'='*80}")
    print(f"PHASE 2: OFFLINE PRETRAINING (training on clean data)")
    print(f"{'='*80}")
    
    pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate.numpy())
    print(f"Created fresh optimizer for offline pretraining with LR: {pretrain_optimizer.learning_rate.numpy():.6f}")
    
    is_compressed = 'compressed' in pkl_file
    print(f"Loading data from {pkl_file} ({'compressed' if is_compressed else 'raw'})...")
    
    if not is_compressed and FEATURE_REDUCER is None:
        print("Warning: Using raw data but no feature reducer available!")
    
    print(f"Loading data in chunks of {chunk_size} samples...\n")
    
    global_batch_num = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_start = time.time()
        epoch_losses = []
        epoch_grad_norms = []
        epoch_entropies = []
        chunk_num = 0
        total_samples = 0
        
        with open(pkl_file, 'rb') as f:
            chunk_states = []
            chunk_actions = []
            chunk_returns = []
            chunk_per_asset_returns = []
            
            while True:
                try:
                    batch_data = pickle.load(f)
                    if isinstance(batch_data, dict):
                        states_to_add = batch_data.get('states', [])
                        
                        if not is_compressed and FEATURE_REDUCER is not None:
                            states_to_add = [FEATURE_REDUCER.transform(np.array(s).reshape(1, -1))[0].tolist() 
                                           for s in states_to_add]
                        
                        chunk_states.extend(states_to_add)
                        chunk_actions.extend(batch_data.get('actions', []))
                        chunk_returns.extend(batch_data.get('returns', []))
                        
                        per_asset_ret = batch_data.get('per_asset_returns', [])
                        if per_asset_ret:
                            chunk_per_asset_returns.extend(per_asset_ret)
                    
                    if len(chunk_states) >= chunk_size:
                        chunk_num += 1
                        chunk_start = time.time()
                        min_len = min(len(chunk_states), len(chunk_actions), len(chunk_returns))
                        chunk_states = chunk_states[:min_len]
                        chunk_actions = chunk_actions[:min_len]
                        chunk_returns = chunk_returns[:min_len]
                        
                        if chunk_per_asset_returns:
                            chunk_per_asset_returns = chunk_per_asset_returns[:min_len]
                        
                        total_samples += len(chunk_states)
                        print(f"  Chunk {chunk_num}: Training on {len(chunk_states)} samples...")
                        
                        chunk_returns_arr = np.array(chunk_returns, dtype=np.float32)
                        returns_mean = chunk_returns_arr.mean()
                        returns_std = chunk_returns_arr.std()
                        chunk_returns_arr = (chunk_returns_arr - returns_mean) / (returns_std + 1e-8)
                        
                        has_per_asset = len(chunk_per_asset_returns) > 0
                        if has_per_asset:
                            chunk_per_asset_arr = np.array(chunk_per_asset_returns, dtype=np.float32)
                            asset_returns_mean = np.mean(chunk_per_asset_arr, axis=0)
                            asset_returns_std = np.std(chunk_per_asset_arr, axis=0) + 1e-8
                            chunk_per_asset_normalized = (chunk_per_asset_arr - asset_returns_mean) / asset_returns_std
                            chunk_per_asset_normalized = np.clip(chunk_per_asset_normalized, -5.0, 5.0)
                        
                        chunk_losses = []
                        chunk_grad_norms = []
                        chunk_entropies = []
                        
                        indices = np.random.permutation(len(chunk_states))
                        for i in range(0, len(chunk_states), batch_size):
                            global_batch_num += 1
                            batch_indices = indices[i:i+batch_size]
                            batch_states = np.array([chunk_states[idx] for idx in batch_indices], dtype=np.float32)
                            batch_actions = np.array([chunk_actions[idx] for idx in batch_indices], dtype=np.float32)
                            batch_returns = chunk_returns_arr[batch_indices]
                            
                            with tf.GradientTape() as tape:
                                action_probs, confidence_pred = model(batch_states, training=True)
                                action_probs_clipped = tf.clip_by_value(action_probs, 1e-8, 1.0)
                                log_probs = tf.reduce_sum(batch_actions * tf.math.log(action_probs_clipped), axis=1)
                                actor_loss = -tf.reduce_mean(log_probs * batch_returns)
                                
                                if has_per_asset:
                                    batch_per_asset = chunk_per_asset_normalized[batch_indices]
                                    target_confidence = tf.sigmoid(batch_per_asset / 2.0)
                                else:
                                    returns_expanded = tf.expand_dims(batch_returns, axis=1)
                                    target_confidence = tf.sigmoid(returns_expanded / 2.0)
                                
                                target_confidence = tf.cast(target_confidence, confidence_pred.dtype)
                                confidence_loss = tf.reduce_mean(tf.square(confidence_pred - target_confidence))
                                
                                entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs_clipped), axis=1))
                                
                                # Use lower confidence loss weight for synthetic data (reduce gradient explosion risk)
                                confidence_weight = 0.1  # Much lower than online training
                                loss = actor_loss + confidence_weight * confidence_loss - 0.01 * entropy
                                
                                # Clip loss to prevent extreme values
                                loss = tf.clip_by_value(loss, -100.0, 100.0)
                                
                                # Check for NaN before gradient computation
                                if tf.math.is_nan(loss):
                                    loss = tf.constant(0.0, dtype=loss.dtype)
                            
                            grads = tape.gradient(loss, model.trainable_variables)
                            
                            # Replace NaN/Inf gradients with zeros
                            grads = [tf.where(tf.math.is_nan(g) | tf.math.is_inf(g), tf.zeros_like(g), g) 
                                    if g is not None else None for g in grads]
                            
                            has_nonzero_grads = any(tf.reduce_any(g != 0.0).numpy() for g in grads if g is not None)
                            
                            if has_nonzero_grads:
                                grads = [tf.clip_by_value(g, -10.0, 10.0) if g is not None else None for g in grads]
                                grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                                
                                grad_norm = tf.linalg.global_norm(grads).numpy()
                                pretrain_optimizer.apply_gradients(zip(grads, model.trainable_variables))
                            else:
                                print(f"    WARNING: All gradients are zero (NaN/Inf data), skipping batch update")
                                grad_norm = 0.0
                            
                            chunk_losses.append(loss.numpy())
                            chunk_grad_norms.append(grad_norm)
                            chunk_entropies.append(entropy.numpy())
                            epoch_losses.append(loss.numpy())
                            epoch_grad_norms.append(grad_norm)
                            epoch_entropies.append(entropy.numpy())
                            
                            del batch_states, batch_actions, batch_returns, action_probs, confidence_pred, grads
                            
                            wandb.log({
                                "pretrain/batch_num": global_batch_num,
                                "pretrain/batch_loss": chunk_losses[-1],
                                "pretrain/batch_grad_norm": grad_norm,
                                "pretrain/batch_entropy": entropy,
                            })
                        
                        chunk_time = time.time() - chunk_start
                        wandb.log({
                            "pretrain/chunk_num": chunk_num,
                            "pretrain/chunk_epoch": epoch + 1,
                            "pretrain/chunk_loss": np.mean(chunk_losses),
                            "pretrain/chunk_grad_norm": np.mean(chunk_grad_norms),
                            "pretrain/chunk_entropy": np.mean(chunk_entropies),
                            "pretrain/chunk_samples": len(chunk_states),
                            "pretrain/chunk_time": chunk_time,
                            "pretrain/chunk_returns_mean": returns_mean,
                            "pretrain/chunk_returns_std": returns_std,
                        })
                        
                        del chunk_states, chunk_actions, chunk_returns, chunk_returns_arr, chunk_per_asset_returns
                        chunk_states = []
                        chunk_actions = []
                        chunk_returns = []
                        chunk_per_asset_returns = []
                        gc.collect()
                        
                        # Save model after each chunk
                        model.save_weights('models/best_model.weights.h5')
                        print(f"  Model saved after chunk {chunk_num}")
                        
                except (pickle.UnpicklingError, EOFError) as e:
                    if isinstance(e, pickle.UnpicklingError):
                        # Corrupted record - we can't recover from this position
                        # We need to break the loop and process what we have
                        print(f"WARNING: Corrupted pickle record encountered")
                        print(f"Breaking out of file read loop to process remaining data...")
                        break
                    else:
                        # EOFError means end of file, normal termination
                        break
            
            if len(chunk_states) > 0:
                chunk_num += 1
                chunk_start = time.time()
                min_len = min(len(chunk_states), len(chunk_actions), len(chunk_returns))
                chunk_states = chunk_states[:min_len]
                chunk_actions = chunk_actions[:min_len]
                chunk_returns = chunk_returns[:min_len]
                
                if chunk_per_asset_returns:
                    chunk_per_asset_returns = chunk_per_asset_returns[:min_len]
                
                total_samples += len(chunk_states)
                print(f"  Chunk {chunk_num} (final): Training on {len(chunk_states)} samples...")
                
                chunk_returns_arr = np.array(chunk_returns, dtype=np.float32)
                returns_mean = chunk_returns_arr.mean()
                returns_std = chunk_returns_arr.std()
                chunk_returns_arr = (chunk_returns_arr - returns_mean) / (returns_std + 1e-8)
                
                has_per_asset = len(chunk_per_asset_returns) > 0
                if has_per_asset:
                    chunk_per_asset_arr = np.array(chunk_per_asset_returns, dtype=np.float32)
                    asset_returns_mean = np.mean(chunk_per_asset_arr, axis=0)
                    asset_returns_std = np.std(chunk_per_asset_arr, axis=0) + 1e-8
                    chunk_per_asset_normalized = (chunk_per_asset_arr - asset_returns_mean) / asset_returns_std
                    chunk_per_asset_normalized = np.clip(chunk_per_asset_normalized, -5.0, 5.0)
                
                chunk_losses = []
                chunk_grad_norms = []
                chunk_entropies = []
                
                indices = np.random.permutation(len(chunk_states))
                for i in range(0, len(chunk_states), batch_size):
                    global_batch_num += 1
                    batch_indices = indices[i:i+batch_size]
                    batch_states = np.array([chunk_states[idx] for idx in batch_indices], dtype=np.float32)
                    batch_actions = np.array([chunk_actions[idx] for idx in batch_indices], dtype=np.float32)
                    batch_returns = chunk_returns_arr[batch_indices]
                    
                    with tf.GradientTape() as tape:
                        action_probs, confidence_pred = model(batch_states, training=True)
                        action_probs_clipped = tf.clip_by_value(action_probs, 1e-8, 1.0)
                        log_probs = tf.reduce_sum(batch_actions * tf.math.log(action_probs_clipped), axis=1)
                        actor_loss = -tf.reduce_mean(log_probs * batch_returns)
                        
                        if has_per_asset:
                            batch_per_asset = chunk_per_asset_normalized[batch_indices]
                            target_confidence = tf.sigmoid(batch_per_asset / 2.0)
                        else:
                            returns_expanded = tf.expand_dims(batch_returns, axis=1)
                            target_confidence = tf.sigmoid(returns_expanded / 2.0)
                        
                        target_confidence = tf.cast(target_confidence, confidence_pred.dtype)
                        confidence_loss = tf.reduce_mean(tf.square(confidence_pred - target_confidence))
                        
                        entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs_clipped), axis=1))
                        
                        # Use lower confidence loss weight for synthetic data (reduce gradient explosion risk)
                        confidence_weight = 0.1  # Much lower than online training
                        loss = actor_loss + confidence_weight * confidence_loss - 0.01 * entropy
                        
                        # Clip loss to prevent extreme values
                        loss = tf.clip_by_value(loss, -100.0, 100.0)
                        
                        # Check for NaN before gradient computation
                        if tf.math.is_nan(loss):
                            loss = tf.constant(0.0, dtype=loss.dtype)
                    
                    grads = tape.gradient(loss, model.trainable_variables)
                    
                    # Replace NaN/Inf gradients with zeros
                    grads = [tf.where(tf.math.is_nan(g) | tf.math.is_inf(g), tf.zeros_like(g), g) 
                            if g is not None else None for g in grads]
                    
                    has_nonzero_grads = any(tf.reduce_any(g != 0.0).numpy() for g in grads if g is not None)
                    
                    if has_nonzero_grads:
                        grads = [tf.clip_by_value(g, -10.0, 10.0) if g is not None else None for g in grads]
                        grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                        
                        grad_norm = tf.linalg.global_norm(grads).numpy()
                        pretrain_optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    else:
                        print(f"    WARNING: All gradients are zero (NaN/Inf data), skipping batch update")
                        grad_norm = 0.0
                    
                    chunk_losses.append(loss.numpy())
                    chunk_grad_norms.append(grad_norm)
                    chunk_entropies.append(entropy.numpy())
                    epoch_losses.append(loss.numpy())
                    epoch_grad_norms.append(grad_norm)
                    epoch_entropies.append(entropy.numpy())
                    
                    del batch_states, batch_actions, batch_returns, action_probs, confidence_pred, grads
                    
                    if global_batch_num % 10 == 0:
                        wandb.log({
                            "pretrain/batch_num": global_batch_num,
                            "pretrain/batch_loss": chunk_losses[-1],
                            "pretrain/batch_grad_norm": grad_norm,
                            "pretrain/batch_entropy": entropy,
                        })
                
                chunk_time = time.time() - chunk_start
                wandb.log({
                    "pretrain/chunk_num": chunk_num,
                    "pretrain/chunk_epoch": epoch + 1,
                    "pretrain/chunk_loss": np.mean(chunk_losses),
                    "pretrain/chunk_grad_norm": np.mean(chunk_grad_norms),
                    "pretrain/chunk_entropy": np.mean(chunk_entropies),
                    "pretrain/chunk_samples": len(chunk_states),
                    "pretrain/chunk_time": chunk_time,
                    "pretrain/chunk_returns_mean": returns_mean,
                    "pretrain/chunk_returns_std": returns_std,
                })
                
                del chunk_states, chunk_actions, chunk_returns, chunk_returns_arr, chunk_per_asset_returns
                gc.collect()
                
                # Save model after final chunk
                model.save_weights('models/best_model.weights.h5')
                print(f"  Model saved after final chunk {chunk_num}")
                
                # Save model after final chunk
                model.save_weights('models/best_model.weights.h5')
                print(f"  Model saved after final chunk {chunk_num}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        avg_entropy = np.mean(epoch_entropies) if epoch_entropies else 0.0
        
        weight_vals = [w.numpy() for w in model.trainable_variables]
        all_weights = np.concatenate([w.flatten() for w in weight_vals])
        
        wandb.log({
            "pretrain/epoch": epoch + 1,
            "pretrain/epoch_loss": avg_loss,
            "pretrain/epoch_grad_norm": avg_grad_norm,
            "pretrain/epoch_entropy": avg_entropy,
            "pretrain/epoch_time": epoch_time,
            "pretrain/epoch_samples": total_samples,
            "pretrain/epoch_chunks": chunk_num,
            "pretrain/epoch_batches": global_batch_num,
            "pretrain/epoch_weights_mean": float(np.mean(all_weights)),
            "pretrain/epoch_weights_std": float(np.std(all_weights)),
            "pretrain/epoch_weights_min": float(np.min(all_weights)),
            "pretrain/epoch_weights_max": float(np.max(all_weights)),
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Entropy: {avg_entropy:.4f} - Samples: {total_samples} - Time: {epoch_time:.1f}s")
        
        gc.collect()
    
    print("Offline pretraining complete!\n")

if __name__ == "__main__":
    ENABLE_PRETRAINING = True
    
    config = {
        "architecture": "dense_residual",
        "dropout_rate": 0.2,
        "compression_layers": [4096, 4096, 2048, 2048],
        "compression_dropout": 0.1,
        "shared_layers": [2048, 2048, 2048, 2048, 2048, 1024, 1024, 512, 512],
        "residual_connections": [True, True, True, True, True, True, True, True, False],
        "actor_layers": [512, 512, 512, 512, 512, 512, 512, 256, 128],
        "initial_lr": 0.000175,  # Much lower for synthetic data stability
        "pretrain_episodes": 10,
        "pretrain_epochs": 3,
        "pretrain_batch_size": 24,  # Smaller batches for stability
        "pretrain_chunk_size": 3072,
        "online_total_batches": 5000,
        "rebalance_interval": 12,
        "online_batch_size": 24 * 12,
        "gamma": 0.99,
        "lr_patience": 10,
        "lr_decay": 0.8,
        "log_step_frequency": 1,
        "epsilon_start": 0.025,
        "epsilon_end": 0.0001,
        "epsilon_decay": 0.9995,
        "env_reset_probability": 0.125,
        "num_stochastic_samples": 64,
        "pretrain_confidence_loss_weight": 0.1,  # Much lower for synthetic data
    }
    
    wandb.init(project="portfolio-trading-online-batch", config=config)
    config = wandb.config
    
    print("Initializing environment...")
    env = PortfolioEnv(max_records=150_000)
    observation = env.reset()
    
    # Set rebalance interval from config
    env.rebalance_interval = config.rebalance_interval
    print(f"Rebalance interval set to: {env.rebalance_interval} steps ({env.rebalance_interval * 5} minutes)")
    
    obs_shape = observation.shape
    num_assets = len(env.asset_names)
    
    print(f"Raw observation shape: {obs_shape}")
    print("Building model with built-in feature compression...")
    model = build_model(obs_shape, num_assets, config, FEATURE_REDUCER)
    model.summary()
    
    # Calculate and log model architecture details
    total_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    
    # Get layer-wise breakdown
    actor_params = 0
    shared_params = 0
    
    for layer in model.layers:
        layer_params = sum([tf.size(var).numpy() for var in layer.trainable_variables])
        layer_name = layer.name.lower()
        
        if 'actor' in layer_name or 'portfolio_allocation' in layer_name:
            actor_params += layer_params
        else:
            shared_params += layer_params
    
    model_info = {
        "model/total_parameters": int(total_params),
        "model/total_parameters_millions": float(total_params / 1e6),
        "model/actor_parameters": int(actor_params),
        "model/shared_parameters": int(shared_params),
        "model/observation_shape": int(obs_shape[0]),
        "model/num_assets": int(num_assets),
        "model/num_layers": len(model.layers),
        "model/num_trainable_variables": len(model.trainable_variables),
    }
    
    wandb.config.update(model_info)
    wandb.log(model_info)
    
    print(f"\nMODEL ARCHITECTURE SUMMARY")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  - Shared Backbone: {shared_params:,} ({shared_params/total_params*100:.1f}%)")
    print(f"  - Actor Head: {actor_params:,} ({actor_params/total_params*100:.1f}%)")
    print(f"Observation Shape: {obs_shape}")
    print(f"Number of Assets: {num_assets}\n")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.initial_lr)
    
    print(f"Using REINFORCE algorithm (actor-only, policy gradient)")
    print(f"All features compressed within model (no external feature reducer)")
    print(f"Model includes built-in compression layers for large observations\n")
    
    best_model_path = 'models/best_model.weights.h5'

    # Load best model if it exists
    if os.path.exists(best_model_path):
        print(f"\n=== LOADING BEST MODEL FROM {best_model_path} ===")
        try:
            model.load_weights(best_model_path)
            print("Best model loaded successfully!\n")
        except Exception as e:
            print(f"Failed to load best model: {e}")
            print("Starting with fresh model weights\n")
    else:
        print(f"\nNo existing best model found at {best_model_path}")
        print("Starting with fresh model weights\n")
    
    # Validate model weights
    has_nan = False
    for var in model.trainable_variables:
        if tf.reduce_any(tf.math.is_nan(var)):
            has_nan = True
            break
    
    if has_nan:
        print("="*80)
        print("CRITICAL ERROR: Model has NaN weights after loading!")
        # print("ERROR: Model has NaN weights! Rebuilding...")
        print("="*80)
        # model = build_model(obs_shape, num_assets, config, FEATURE_REDUCER)
    else:
        print("Model weights validated: No NaN detected")

    if not os.path.exists('/media/user/HDD 1TB/Data/training_data.pkl') and \
       not os.path.exists('/media/user/HDD 1TB/Data/training_data_compressed.pkl') and \
       not os.path.exists('/media/user/HDD 1TB/Data/synthetic_training_data.pkl') and \
       not os.path.exists('/media/user/HDD 1TB/Data/synthetic_training_data_compressed.pkl'):
        print("No pretraining data found, skipping pretraining")
        ENABLE_PRETRAINING = False
    
    if ENABLE_PRETRAINING and config.pretrain_epochs > 0:
        offline_pretrain(env, model, optimizer, 
                        num_episodes=config.pretrain_episodes, 
                        gamma=config.gamma,
                        batch_size=config.pretrain_batch_size,
                        epochs=config.pretrain_epochs,
                        chunk_size=config.pretrain_chunk_size)
        model.save_weights('models/best_model.weights.h5')
    else:
        print("\nSKIPPING PRETRAINING")
    
    print("\nONLINE TRAINING (BATCH-BY-BATCH)")
    print(f"Batch size: {config.online_batch_size}")
    print(f"Total target batches: {config.online_total_batches}")
    
    best_avg_step_reward = float('-inf')
    no_improvement = 0
    current_lr = config.initial_lr
    global_step = 0
    batch_count = 0
    epsilon = config.epsilon_start
    
    recent_step_rewards = deque(maxlen=10)  # Track avg reward per step
    recent_losses = deque(maxlen=10)
    
    print(f"\nSTARTING BATCH TRAINING LOOP")
    
    start_time = time.time()
    
    while batch_count < config.online_total_batches:
        batch_count += 1
        print(f"\n--- Batch {batch_count}/{config.online_total_batches} ---")
        
        collect_start = time.time()
        states, raw_states, actions, returns, confidences, is_random_flags, batch_info, step_logs, per_asset_returns = collect_batch_data(
            env, model, config.online_batch_size, config.gamma, global_step, epsilon, config.env_reset_probability, FEATURE_REDUCER, config
        )
        collect_time = time.time() - collect_start
        
        steps_collected = len(states)
        global_step += steps_collected
        
        # Calculate average reward per step for this batch
        avg_step_reward = batch_info['total_reward'] / max(1, steps_collected)
        recent_step_rewards.append(avg_step_reward)
        
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        
        # Log all step data
        for i, step_log in enumerate(step_logs):
            if i % config.log_step_frequency == 0:
                log_dict = {
                    "step/global": step_log['step_global'],
                    "step/batch": step_log['step_batch'],
                    "step/reward": step_log['reward'],
                    "step/confidence_mean": step_log['confidence'],
                    "step/confidence_min": step_log['confidence_min'],
                    "step/confidence_max": step_log['confidence_max'],
                    "step/confidence_std": step_log['confidence_std'],
                    "step/portfolio_value": step_log['portfolio_value'],
                    "step/benchmark_value": step_log['benchmark_value'],
                    "step/outperformance": (step_log['outperformance'] - 1) * 100,
                    "step/action_mean": step_log['action_mean'],
                    "step/action_std": step_log['action_std'],
                    "step/action_max": step_log['action_max'],
                    "step/done": step_log['done'],
                    "step/portfolio_layout": step_log['portfolio_layout']
                }
                
                per_asset_conf = step_log['per_asset_confidence']
                if per_asset_conf is not None:
                    for asset_idx, conf_val in enumerate(per_asset_conf):
                        asset_name = env.asset_names[asset_idx] if asset_idx < len(env.asset_names) else f"asset_{asset_idx}"
                        log_dict[f"confidence_per_asset/{asset_name}"] = float(conf_val)
                
                per_asset_rew = step_log['per_asset_rewards']
                if per_asset_rew is not None:
                    for asset_idx, rew_val in enumerate(per_asset_rew):
                        asset_name = env.asset_names[asset_idx] if asset_idx < len(env.asset_names) else f"asset_{asset_idx}"
                        log_dict[f"reward_per_asset/{asset_name}"] = float(rew_val)
                
                wandb.log(log_dict)
        
        train_start = time.time()
        if len(states) > 0:
            batch_metrics = train_on_batch(
                model, optimizer, states, raw_states, actions, returns, 
                confidences, per_asset_returns, is_random_flags, batch_count, None, None,
                clip_ratio=0.2, train_encoder=False
            )
            train_time = time.time() - train_start
            recent_losses.append(batch_metrics['batch/loss'])
            
            # Check for NaN in model weights after training
            has_nan_weights = False
            for var in model.trainable_variables:
                if tf.reduce_any(tf.math.is_nan(var)):
                    has_nan_weights = True
                    break
            
            if has_nan_weights:
                print("CRITICAL: NaN detected in model weights! Reloading best model...")
                if os.path.exists('models/best_model.weights.h5'):
                    model.load_weights('models/best_model.weights.h5')
                    print("Best model reloaded successfully")
                else:
                    print("ERROR: No best model to reload! Training may be unstable")
            
            if batch_count % 10 == 0:
                all_weights = tf.concat([tf.reshape(var, [-1]) for var in model.trainable_variables], axis=0)
                weight_stats = {
                    'model_stats/weight_mean': float(tf.reduce_mean(all_weights).numpy()),
                    'model_stats/weight_std': float(tf.math.reduce_std(all_weights).numpy()),
                    'model_stats/weight_min': float(tf.reduce_min(all_weights).numpy()),
                    'model_stats/weight_max': float(tf.reduce_max(all_weights).numpy()),
                    'model_stats/weight_abs_mean': float(tf.reduce_mean(tf.abs(all_weights)).numpy()),
                }
                
                extreme_weight_fraction = float(tf.reduce_sum(
                    tf.cast(tf.abs(all_weights) > 10.0, tf.float32)
                ).numpy() / tf.size(all_weights, out_type=tf.float32).numpy())
                
                weight_stats['model_stats/extreme_weight_fraction'] = extreme_weight_fraction
                
                wandb.log(weight_stats)
                
                if extreme_weight_fraction > 0.01:
                    print(f"  WARNING: {extreme_weight_fraction*100:.2f}% of weights are extreme (>10)")
            
            if batch_count % 10 == 0:  # Log every 10 batches (weights are large)
                histogram_data = {}
                for layer in model.layers:
                    if len(layer.trainable_variables) > 0:
                        layer_name = layer.name
                        for i, var in enumerate(layer.trainable_variables):
                            var_name = var.name.replace('/', '_').replace(':', '_')
                            histogram_data[f"weights/{layer_name}/{var_name}"] = wandb.Histogram(var.numpy().flatten())
                
                wandb.log(histogram_data)
        else:
            # Empty batch - set all metrics to 0
            batch_metrics = {
                'batch/loss': 0.0, 
                'batch/actor_loss': 0.0,
                'batch/confidence_loss': 0.0,
                'batch/entropy': 0.0,
            }
            train_time = 0.0
        
        # Calculate weighted average of recent step rewards (more weight to recent)
        if len(recent_step_rewards) > 1:
            weights = np.linspace(0.5, 1.0, len(recent_step_rewards))
            weighted_avg_step_reward = np.average(list(recent_step_rewards), weights=weights)
        elif len(recent_step_rewards) == 1:
            weighted_avg_step_reward = recent_step_rewards[0]
        else:
            weighted_avg_step_reward = 0.0
        
        if len(recent_losses) > 1:
            weights = np.linspace(0.5, 1.0, len(recent_losses))
            weighted_avg_loss = np.average(list(recent_losses), weights=weights)
        elif len(recent_losses) == 1:
            weighted_avg_loss = recent_losses[0]
        else:
            weighted_avg_loss = 0.0
        
        batch_metrics.update({
            'batch/number': batch_count,
            'batch/global_step': global_step,
            'batch/steps_collected': steps_collected,
            'batch/total_reward': batch_info['total_reward'],
            'batch/avg_step_reward': avg_step_reward,
            'batch/weighted_avg_step_reward': weighted_avg_step_reward,
            'batch/weighted_avg_loss': weighted_avg_loss,
            'batch/mean_confidence': batch_info['mean_confidence'],
            'batch/portfolio_value': batch_info['portfolio_value'],
            'batch/benchmark_value': batch_info['benchmark_value'],
            'batch/outperformance': (batch_info['outperformance'] - 1) * 100,
            'batch/collection_time': collect_time,
            'batch/training_time': train_time,
            'batch/total_time': collect_time + train_time,
            'batch/steps_per_second': steps_collected / collect_time if collect_time > 0 else 0,
            'batch/learning_rate': current_lr,
            'batch/epsilon': epsilon,
        })
        
        # Add per-asset confidence statistics (mean across all steps in batch)
        if len(confidences) > 0:
            all_confidences = np.array(confidences)  # Shape: (steps, num_assets)
            per_asset_mean_conf = np.mean(all_confidences, axis=0)  # Mean across steps for each asset
            
            for asset_idx, conf_mean in enumerate(per_asset_mean_conf):
                asset_name = env.asset_names[asset_idx] if asset_idx < len(env.asset_names) else f"asset_{asset_idx}"
                batch_metrics[f"batch_confidence_per_asset/{asset_name}"] = float(conf_mean)
        
        wandb.log(batch_metrics)
        
        print(f"Collected {steps_collected} steps in {collect_time:.2f}s")
        
        # Add diagnostic info about rewards and returns
        if len(states) > 0 and len(returns) > 0:
            return_stats = f"Returns: min={np.min(returns):.2f}, max={np.max(returns):.2f}, mean={np.mean(returns):.2f}"
            print(f"  Avg Reward/step: {avg_step_reward:.3f}")
            print(f"  {return_stats}")
        
        if TRAIN_FEATURE_REDUCER:
            print(f"Trained in {train_time:.2f}s, Loss: {batch_metrics['batch/loss']:.4f} (Actor: {batch_metrics['batch/actor_loss']:.4f})")
            print(f"Encoder Loss: {batch_metrics.get('batch/encoder_loss', 0):.4f}")
        else:
            print(f"Trained in {train_time:.2f}s, Loss: {batch_metrics['batch/loss']:.4f} (Actor: {batch_metrics['batch/actor_loss']:.4f})")
        print(f"Reward: {batch_info['total_reward']:.2f}, Avg step reward: {avg_step_reward:.3f}")
        print(f"Weighted avg step reward (recent): {weighted_avg_step_reward:.3f}")

        # Log best model tracking metrics
        wandb.log({
            'best/best_avg_step_reward': best_avg_step_reward,
            'best/weighted_avg_step_reward_recent': weighted_avg_step_reward,
            'best/current_avg_step_reward': avg_step_reward,
        })

        # Save model if weighted average step reward improved (no warmup)
        if weighted_avg_step_reward > best_avg_step_reward:
            best_avg_step_reward = weighted_avg_step_reward
            model.save_weights('models/best_model.weights.h5')
            wandb.save('models/best_model.weights.h5')
            
            no_improvement = 0
            print(f"NEW BEST AVG STEP REWARD: {best_avg_step_reward:.4f}")
        else:
            no_improvement += 1
        
        if no_improvement >= config.lr_patience:
            current_lr *= config.lr_decay
            if current_lr < 1e-6:
                current_lr = 1e-6
            optimizer.learning_rate.assign(current_lr)
            
            print(f"\nLEARNING RATE DECAY TRIGGERED")
            print(f"No improvement for {config.lr_patience} batches")
            print(f"New learning rate: {current_lr:.6f} (reduced by {config.lr_decay}x)\n")
            no_improvement = 0
        
        progress = (batch_count / config.online_total_batches) * 100
        elapsed_time = time.time() - start_time
        batches_per_sec = batch_count / elapsed_time if elapsed_time > 0 else 0
        steps_per_sec = global_step / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nProgress: {progress:.1f}% ({batch_count}/{config.online_total_batches} batches)")
        print(f"Time elapsed: {elapsed_time:.1f}s, Batches/sec: {batches_per_sec:.2f}, Steps/sec: {steps_per_sec:.2f}")
        print(f"Current LR: {current_lr:.6f}, Epsilon: {epsilon:.4f}, Best avg step reward: {best_avg_step_reward:.4f}")
        
        del states, actions, returns
        gc.collect()
    
    total_time = time.time() - start_time
    print(f"\nTRAINING COMPLETE!")
    print(f"Total batches: {batch_count}")
    print(f"Total steps: {global_step}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Best avg step reward: {best_avg_step_reward:.4f}")
    print(f"Best model saved at: models/best_model.weights.h5")
    
    wandb.finish()