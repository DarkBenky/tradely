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
    if config is None:
        config = {
            "dropout_rate": 0.15,
            "compression_layers": [8192, 2048],
            "compression_dropout": 0.1,
            "shared_layers": [1024, 768, 512, 384, 256],
            "residual_connections": [True, True, True, False],
            "actor_layers": [192, 128],
            "critic_layers": [192, 128],
        }
    
    compression_layers = config.get("compression_layers", [8192, 2048])
    compression_dropout = config.get("compression_dropout", 0.1)
    shared_layers = config.get("shared_layers", [1024, 768, 512, 384, 256])
    residual_connections = config.get("residual_connections", [True, True, True, False])
    actor_layers = config.get("actor_layers", [192, 128])
    critic_layers = config.get("critic_layers", [192, 128])
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
    
    actor_x = x
    for i, units in enumerate(actor_layers):
        actor_x = layers.Dense(units, activation='relu', name=f'actor_dense_{i+1}')(actor_x)
        if i < len(actor_layers) - 1:
            actor_x = layers.BatchNormalization()(actor_x)
        actor_x = layers.Dropout(dropout_rate)(actor_x)
    actor_output = layers.Dense(num_assets, activation='softmax', name='portfolio_allocation')(actor_x)
    
    critic_x = x
    for i, units in enumerate(critic_layers):
        critic_x = layers.Dense(units, activation='relu', name=f'critic_dense_{i+1}')(critic_x)
        if i < len(critic_layers) - 1:
            critic_x = layers.BatchNormalization()(critic_x)
        critic_x = layers.Dropout(dropout_rate)(critic_x)
    critic_output = layers.Dense(1, activation='linear', name='value')(critic_x)
    
    model = Model(inputs=inputs, outputs=[actor_output, critic_output])
    return model

def collect_batch_data(env, model, batch_size, gamma, current_step, epsilon=0.1, env_reset_probability=0.05, feature_reducer=None):
    states, raw_states, actions, rewards, values, is_random_flags = [], [], [], [], [], []
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
        return [], [], [], np.array([]), np.array([]), [], [], {
            'steps_collected': 0,
            'total_reward': 0.0,
            'mean_value': 0.0,
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
        
        action_probs, value = model(state_tensor, training=False)
        action = action_probs[0].numpy()
        value = value[0, 0].numpy()
        
        if np.any(np.isnan(action)) or np.isnan(value):
            print(f"WARNING: NaN in model output (action or value), using random action")
            action = env.sample()
            value = 0.0
            is_random = True
        elif np.isinf(value) or abs(value) > 1000:
            print(f"WARNING: Extreme value prediction: {value:.2f}, clipping and using random action")
            value = np.clip(value, -100.0, 100.0)
            action = env.sample()
            is_random = True
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
        
        next_state, reward, done, info = env.step(action)
        
        if np.isnan(reward) or np.isinf(reward):
            print(f"WARNING: NaN/Inf reward detected, replacing with 0.0")
            reward = 0.0
        elif abs(reward) > 20:
            print(f"WARNING: Extreme reward detected: {reward:.2f}, clipping to +-10")
            reward = np.clip(reward, -10.0, 10.0)
        
        env.render()
        
        raw_next_state = next_state.copy()
        
        if feature_reducer is not None:
            next_state = feature_reducer.transform(next_state.reshape(1, -1))[0]

        states.append(state)
        raw_states.append(raw_state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        is_random_flags.append(is_random)
        
        step_log = {
            'step_global': current_step + steps_collected,
            'step_batch': steps_collected,
            'reward': reward,
            'value': value,
            'portfolio_value': info.get('portfolio_value', 0),
            'benchmark_value': info.get('benchmark_value', 0),
            'outperformance': info.get('outperformance', 0),
            'action_mean': np.mean(action),
            'action_std': np.std(action),
            'action_max': np.max(action),
            'done': done,
            'portfolio_layout': info.get('portfolio_layout', {})
        }
        
        step_logs.append(step_log)
        
        state = next_state
        raw_state = raw_next_state
        steps_collected += 1
        
        if steps_collected >= 10000:
            print(f"  Warning: Episode exceeded 10000 steps, forcing termination")
            break
    
    returns = np.array([], dtype=np.float32)
    advantages = np.array([], dtype=np.float32)
    
    if steps_collected > 0:
        rewards_array = np.array(rewards, dtype=np.float32)
        values_array = np.array(values, dtype=np.float32)
        
        if np.any(np.isnan(rewards_array)):
            print(f"WARNING: NaN in rewards array, replacing with zeros")
            rewards_array = np.nan_to_num(rewards_array, nan=0.0)
            rewards = rewards_array.tolist()
        
        if np.any(np.isnan(values_array)):
            print(f"WARNING: NaN in values array, replacing with zeros")
            values_array = np.nan_to_num(values_array, nan=0.0)
            values = values_array.tolist()
        
        returns = []
        advantages = []
        G = 0.0
        A = 0.0
        
        for i in reversed(range(len(rewards))):
            G = rewards[i] + gamma * G
            
            if np.isnan(G) or np.isinf(G):
                print(f"WARNING: NaN/Inf in return calculation at step {i}, resetting to 0")
                G = 0.0
            
            returns.insert(0, G)
            
            next_value = values[i+1] if i+1 < len(values) else 0.0
            td_error = rewards[i] + gamma * next_value - values[i]
            
            if np.isnan(td_error) or np.isinf(td_error):
                print(f"WARNING: NaN/Inf in TD error at step {i}, resetting to 0")
                td_error = 0.0
            
            A = td_error + gamma * 0.95 * A
            
            if np.isnan(A) or np.isinf(A):
                print(f"WARNING: NaN/Inf in advantage calculation at step {i}, resetting to 0")
                A = 0.0
            
            advantages.insert(0, A)
        
        returns = np.array(returns, dtype=np.float32)
        advantages = np.array(advantages, dtype=np.float32)
        
        if np.any(np.isnan(advantages)):
            print("WARNING: NaN in raw advantages, replacing with zeros")
            advantages = np.nan_to_num(advantages, nan=0.0)
        
        if np.any(np.isnan(returns)):
            print("WARNING: NaN in returns, replacing with zeros")
            returns = np.nan_to_num(returns, nan=0.0)
        
        if len(advantages) > 1:
            adv_std = np.std(advantages)
            adv_mean = np.mean(advantages)
            
            if np.isnan(adv_mean) or np.isnan(adv_std):
                print("WARNING: NaN in advantage mean/std, resetting advantages to zero")
                advantages = np.zeros_like(advantages)
            elif adv_std > 1e-8:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                if not np.isnan(adv_mean):
                    advantages = advantages - adv_mean
                else:
                    advantages = np.zeros_like(advantages)
        else:
            if np.isnan(advantages[0]) or np.isinf(advantages[0]):
                advantages[0] = 0.0
        
        if np.any(np.isnan(advantages)) or np.any(np.isinf(advantages)):
            print("WARNING: NaN/Inf in normalized advantages, replacing with zeros")
            advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

    train_data_dir = "/media/user/HDD 1TB/Data"
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)

    if steps_collected > 0 and len(states) > 0:
        with open(os.path.join(train_data_dir, "training_data.pkl"), 'ab') as f:
            pickle.dump({'states': states, 'actions': actions, 'returns': returns, 'advantages': advantages, 'is_random': is_random_flags}, f)
    
    random_count = sum(is_random_flags)
    policy_count = len(is_random_flags) - random_count
    
    if steps_collected == 0:
        batch_info = {
            'steps_collected': 0,
            'total_reward': 0.0,
            'mean_value': 0.0,
            'portfolio_value': env._portfolio_portfolio_value(),
            'benchmark_value': env._update_benchmark_value(),
            'outperformance': 1.0,
            'done': done,
            'random_actions': 0,
            'policy_actions': 0
        }
    else:
        batch_info = {
            'steps_collected': steps_collected,
            'total_reward': sum(rewards),
            'mean_value': np.mean(values) if values else 0.0,
            'portfolio_value': info.get('portfolio_value', 1.0) if steps_collected > 0 else 1.0,
            'benchmark_value': info.get('benchmark_value', 1.0) if steps_collected > 0 else 1.0,
            'outperformance': info.get('outperformance', 1.0) if steps_collected > 0 else 1.0,
            'done': done,
            'random_actions': random_count,
            'policy_actions': policy_count
        }
    
    return states, raw_states, actions, returns, advantages, values, is_random_flags, batch_info, step_logs

def train_on_batch(model, optimizer, batch_states, batch_raw_states, batch_actions, batch_returns, 
                   batch_advantages, is_random_flags, batch_num, feature_reducer=None, 
                   feature_reducer_optimizer=None, clip_ratio=0.2, train_encoder=True):
    states_batch = np.array(batch_states, dtype=np.float32)
    actions_batch = np.array(batch_actions, dtype=np.float32)
    returns_batch = np.array(batch_returns, dtype=np.float32)
    
    returns_mean = np.mean(returns_batch)
    returns_std = np.std(returns_batch)
    if returns_std > 1e-6 and not np.isnan(returns_std):
        returns_batch = (returns_batch - returns_mean) / (returns_std + 1e-8)
    else:
        returns_batch = returns_batch - returns_mean if not np.isnan(returns_mean) else returns_batch
    
    returns_batch = np.clip(returns_batch, -10.0, 10.0)
    
    with tf.GradientTape() as tape:
        action_probs, values = model(states_batch, training=True)
        action_probs_clipped = tf.clip_by_value(action_probs, 1e-8, 1.0)
        
        log_probs = tf.reduce_sum(actions_batch * tf.math.log(action_probs_clipped), axis=1)
        actor_loss = -tf.reduce_mean(log_probs * returns_batch)
        
        values_squeezed = tf.squeeze(values)
        value_loss = tf.reduce_mean(tf.square(returns_batch - values_squeezed))
        
        entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs_clipped), axis=1))
        
        total_loss = actor_loss + 1.0 * value_loss - 0.01 * entropy
    
    grads = tape.gradient(total_loss, model.trainable_variables)
    
    nan_grad_count = sum(1 for g in grads if g is not None and tf.reduce_any(tf.math.is_nan(g)))
    if nan_grad_count > 0:
        print(f"WARNING: {nan_grad_count} gradients contain NaN, skipping update")
        return {
            'batch/loss': 0.0,
            'batch/actor_loss': 0.0,
            'batch/critic_loss': 0.0,
            'batch/entropy': 0.0,
            'batch/mean_action_prob': 0.0,
            'batch/max_action_prob': 0.0,
            'batch/min_action_prob': 0.0,
            'batch/action_std': 0.0,
            'batch/mean_return': 0.0,
            'batch/std_return': 0.0,
            'batch/mean_value': 0.0,
            'batch/grad_norm': 0.0,
            'batch/random_action_ratio': 0.0,
        }
    
    grads = [tf.clip_by_norm(g, 0.5) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    action_probs_np = action_probs.numpy()
    grad_norm = float(tf.linalg.global_norm(grads).numpy())
    
    metrics = {
        'batch/loss': float(total_loss.numpy()),
        'batch/actor_loss': float(actor_loss.numpy()),
        'batch/critic_loss': float(value_loss.numpy()),
        'batch/entropy': float(entropy.numpy()),
        'batch/mean_action_prob': float(tf.reduce_mean(action_probs).numpy()),
        'batch/max_action_prob': float(tf.reduce_max(action_probs).numpy()),
        'batch/min_action_prob': float(tf.reduce_min(action_probs).numpy()),
        'batch/action_std': float(np.std(action_probs_np)),
        'batch/mean_return': float(np.mean(batch_returns)),
        'batch/std_return': float(np.std(batch_returns)),
        'batch/mean_value': float(tf.reduce_mean(values).numpy()),
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
        
        if batch_num % 2 == 0:
            metrics['gradients/histogram'] = wandb.Histogram(all_grads)
    
    if batch_num % 2 == 0:
        metrics['action_probs/histogram'] = wandb.Histogram(action_probs_np.flatten())
        metrics['values/histogram'] = wandb.Histogram(values.numpy().flatten())
        metrics['returns/histogram'] = wandb.Histogram(batch_returns)
    
    for key, value in metrics.items():
        if np.isnan(value):
            print(f"WARNING: NaN detected in metric {key}, replacing with 0.0")
            metrics[key] = 0.0
    
    return metrics

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
    
    is_compressed = 'compressed' in pkl_file
    print(f"Loading data from {pkl_file} ({'compressed' if is_compressed else 'raw'})...")
    
    if not is_compressed and FEATURE_REDUCER is None:
        print("Warning: Using raw data but no feature reducer available!")
    
    print(f"Loading data from {pkl_file} in chunks of {chunk_size} samples...")
    
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
            
            while True:
                try:
                    batch_data = pickle.load(f)
                    if isinstance(batch_data, dict):
                        states_to_add = batch_data.get('states', [])
                        
                        # Apply feature reduction if using raw data
                        if not is_compressed and FEATURE_REDUCER is not None:
                            states_to_add = [FEATURE_REDUCER.transform(np.array(s).reshape(1, -1))[0].tolist() 
                                           for s in states_to_add]
                        
                        chunk_states.extend(states_to_add)
                        chunk_actions.extend(batch_data.get('actions', []))
                        chunk_returns.extend(batch_data.get('returns', []))
                    
                    if len(chunk_states) >= chunk_size:
                        chunk_num += 1
                        chunk_start = time.time()
                        min_len = min(len(chunk_states), len(chunk_actions), len(chunk_returns))
                        chunk_states = chunk_states[:min_len]
                        chunk_actions = chunk_actions[:min_len]
                        chunk_returns = chunk_returns[:min_len]
                        
                        total_samples += len(chunk_states)
                        print(f"  Chunk {chunk_num}: Training on {len(chunk_states)} samples...")
                        
                        chunk_returns_arr = np.array(chunk_returns, dtype=np.float32)
                        returns_mean = chunk_returns_arr.mean()
                        returns_std = chunk_returns_arr.std()
                        chunk_returns_arr = (chunk_returns_arr - returns_mean) / (returns_std + 1e-8)
                        
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
                                action_probs, values = model(batch_states, training=True)
                                action_probs_clipped = tf.clip_by_value(action_probs, 1e-8, 1.0)
                                log_probs = tf.reduce_sum(batch_actions * tf.math.log(action_probs_clipped), axis=1)
                                actor_loss = -tf.reduce_mean(log_probs * batch_returns)
                                
                                values_squeezed = tf.squeeze(values)
                                value_loss = tf.reduce_mean(tf.square(batch_returns - values_squeezed))
                                
                                loss = actor_loss + 0.5 * value_loss
                            
                            grads = tape.gradient(loss, model.trainable_variables)
                            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                            grad_norm = tf.linalg.global_norm(grads).numpy()
                            optimizer.apply_gradients(zip(grads, model.trainable_variables))
                            
                            action_probs_np = action_probs.numpy()
                            entropy = -np.mean(np.sum(action_probs_np * np.log(action_probs_np + 1e-8), axis=1))
                            
                            chunk_losses.append(loss.numpy())
                            chunk_grad_norms.append(grad_norm)
                            chunk_entropies.append(entropy)
                            epoch_losses.append(loss.numpy())
                            epoch_grad_norms.append(grad_norm)
                            epoch_entropies.append(entropy)
                            
                            del batch_states, batch_actions, batch_returns, action_probs, grads
                            
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
                        
                        del chunk_states, chunk_actions, chunk_returns, chunk_returns_arr
                        chunk_states = []
                        chunk_actions = []
                        chunk_returns = []
                        gc.collect()
                        
                except EOFError:
                    break
            
            if len(chunk_states) > 0:
                chunk_num += 1
                chunk_start = time.time()
                min_len = min(len(chunk_states), len(chunk_actions), len(chunk_returns))
                chunk_states = chunk_states[:min_len]
                chunk_actions = chunk_actions[:min_len]
                chunk_returns = chunk_returns[:min_len]
                
                total_samples += len(chunk_states)
                print(f"  Chunk {chunk_num} (final): Training on {len(chunk_states)} samples...")
                
                chunk_returns_arr = np.array(chunk_returns, dtype=np.float32)
                returns_mean = chunk_returns_arr.mean()
                returns_std = chunk_returns_arr.std()
                chunk_returns_arr = (chunk_returns_arr - returns_mean) / (returns_std + 1e-8)
                
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
                        action_probs, values = model(batch_states, training=True)
                        action_probs_clipped = tf.clip_by_value(action_probs, 1e-8, 1.0)
                        log_probs = tf.reduce_sum(batch_actions * tf.math.log(action_probs_clipped), axis=1)
                        actor_loss = -tf.reduce_mean(log_probs * batch_returns)
                        
                        values_squeezed = tf.squeeze(values)
                        value_loss = tf.reduce_mean(tf.square(batch_returns - values_squeezed))
                        
                        loss = actor_loss + 0.5 * value_loss
                    
                    grads = tape.gradient(loss, model.trainable_variables)
                    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                    grad_norm = tf.linalg.global_norm(grads).numpy()
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    
                    action_probs_np = action_probs.numpy()
                    entropy = -np.mean(np.sum(action_probs_np * np.log(action_probs_np + 1e-8), axis=1))
                    
                    chunk_losses.append(loss.numpy())
                    chunk_grad_norms.append(grad_norm)
                    chunk_entropies.append(entropy)
                    epoch_losses.append(loss.numpy())
                    epoch_grad_norms.append(grad_norm)
                    epoch_entropies.append(entropy)
                    
                    del batch_states, batch_actions, batch_returns, action_probs, grads
                    
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
                
                del chunk_states, chunk_actions, chunk_returns, chunk_returns_arr
                gc.collect()
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        avg_entropy = np.mean(epoch_entropies) if epoch_entropies else 0.0
        
        wandb.log({
            "pretrain/epoch": epoch + 1,
            "pretrain/epoch_loss": avg_loss,
            "pretrain/epoch_grad_norm": avg_grad_norm,
            "pretrain/epoch_entropy": avg_entropy,
            "pretrain/epoch_time": epoch_time,
            "pretrain/epoch_samples": total_samples,
            "pretrain/epoch_chunks": chunk_num,
            "pretrain/epoch_batches": global_batch_num,
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Entropy: {avg_entropy:.4f} - Samples: {total_samples} - Time: {epoch_time:.1f}s")
        
        gc.collect()
    
    print("Offline pretraining complete!\n")

if __name__ == "__main__":
    ENABLE_PRETRAINING = True
    
    config = {
        "architecture": "dense_residual",
        "dropout_rate": 0.15,
        "compression_layers": [4096, 2048, 1024],
        "compression_dropout": 0.1,
        "shared_layers": [1024, 768, 512, 384, 256],
        "residual_connections": [True, True, True, False],
        "actor_layers": [256, 256, 256, 256, 256, 256, 128],
        "critic_layers": [256, 256, 256, 256, 256, 256, 128],
        "initial_lr": 0.0001,
        "pretrain_episodes": 10,
        "pretrain_epochs": 0,
        "pretrain_batch_size": 512,
        "pretrain_chunk_size": 512 * 10,
        "online_total_batches": 5000,
        "online_batch_size": 128,
        "gamma": 0.99,
        "lr_patience": 200,
        "lr_decay": 0.8,
        "log_step_frequency": 1,
        "epsilon_start": 0.025,
        "epsilon_end": 0.0001,
        "epsilon_decay": 0.9995,
        "env_reset_probability": 0.25,
    }
    
    wandb.init(project="portfolio-trading-online-batch", config=config)
    config = wandb.config
    
    print("Initializing environment...")
    env = PortfolioEnv(max_records=100_000)
    observation = env.reset()
    
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
    critic_params = 0
    shared_params = 0
    
    for layer in model.layers:
        layer_params = sum([tf.size(var).numpy() for var in layer.trainable_variables])
        layer_name = layer.name.lower()
        
        if 'actor' in layer_name or 'portfolio_allocation' in layer_name:
            actor_params += layer_params
        elif 'critic' in layer_name or 'value' in layer_name:
            critic_params += layer_params
        else:
            shared_params += layer_params
    
    model_info = {
        "model/total_parameters": int(total_params),
        "model/total_parameters_millions": float(total_params / 1e6),
        "model/actor_parameters": int(actor_params),
        "model/critic_parameters": int(critic_params),
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
    print(f"  - Critic Head: {critic_params:,} ({critic_params/total_params*100:.1f}%)")
    print(f"Observation Shape: {obs_shape}")
    print(f"Number of Assets: {num_assets}\n")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.initial_lr)
    
    print(f"Using Adam optimizer with simple policy gradient (no PPO)")
    print(f"All features compressed within model (no external feature reducer)")
    print(f"Model includes built-in compression layers for large observations\n")
    
    pretrained_model_path = 'models/pretrained_model.weights.h5'
    best_model_path = 'models/best_model.weights.h5'

    print(f"\nSTARTING WITH FRESH MODEL (NO PRETRAINED WEIGHTS)")
    print(f"All previous models have been backed up to models_corrupted_backup/")
    print(f"Training from scratch with fixed reward scaling and stability improvements\n")
    
    has_nan = False
    for var in model.trainable_variables:
        if tf.reduce_any(tf.math.is_nan(var)):
            has_nan = True
            break
    
    if has_nan:
        print("ERROR: Freshly initialized model has NaN weights! This should not happen.")
        print("Rebuilding model...")
        model = build_model(obs_shape, num_assets, config, FEATURE_REDUCER)
    else:
        print("Model weights validated: No NaN detected in fresh initialization")
    
    # if os.path.exists(best_model_path):
    #     print(f"\n=== LOADING BEST MODEL FROM {best_model_path} ===")
    #     model.load_weights(best_model_path)
    #     print("Best model loaded successfully!\n")
    # elif os.path.exists(pretrained_model_path):
    #     print(f"\n=== LOADING PRETRAINED MODEL FROM {pretrained_model_path} ===")
    #     model.load_weights(pretrained_model_path)
    #     print("Pretrained model loaded successfully!\n")

    if not os.path.exists('/media/user/HDD 1TB/Data/training_data.pkl') and \
       not os.path.exists('/media/user/HDD 1TB/Data/training_data_compressed.pkl') and \
       not os.path.exists('/media/user/HDD 1TB/Data/synthetic_training_data.pkl') and \
       not os.path.exists('/media/user/HDD 1TB/Data/synthetic_training_data_compressed.pkl'):
        print("No pretraining data found, skipping pretraining")
        ENABLE_PRETRAINING = False
    
    if ENABLE_PRETRAINING:
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
    
    WARMUP_BATCHES = 20
    
    best_avg_reward = float('-inf')
    best_outperformance = 0.0
    no_improvement = 0
    current_lr = config.initial_lr
    global_step = 0
    batch_count = 0
    epsilon = config.epsilon_start
    
    recent_rewards = deque(maxlen=50)
    recent_outperformance = deque(maxlen=50)
    recent_losses = deque(maxlen=50)
    
    print(f"\nSTARTING BATCH TRAINING LOOP")
    
    start_time = time.time()
    
    while batch_count < config.online_total_batches:
        batch_count += 1
        print(f"\n--- Batch {batch_count}/{config.online_total_batches} ---")
        
        collect_start = time.time()
        states, raw_states, actions, returns, advantages, values, is_random_flags, batch_info, step_logs = collect_batch_data(
            env, model, config.online_batch_size, config.gamma, global_step, epsilon, config.env_reset_probability, FEATURE_REDUCER
        )
        collect_time = time.time() - collect_start
        
        steps_collected = len(states)
        global_step += steps_collected
        recent_rewards.append(batch_info['total_reward'])
        recent_outperformance.append(batch_info['outperformance'])
        
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        
        # Log all step data
        for i, step_log in enumerate(step_logs):
            if i % config.log_step_frequency == 0:
                wandb.log({
                    "step/global": step_log['step_global'],
                    "step/batch": step_log['step_batch'],
                    "step/reward": step_log['reward'],
                    "step/value": step_log['value'],
                    "step/portfolio_value": step_log['portfolio_value'],
                    "step/benchmark_value": step_log['benchmark_value'],
                    "step/outperformance": (step_log['outperformance'] - 1) * 100,
                    "step/action_mean": step_log['action_mean'],
                    "step/action_std": step_log['action_std'],
                    "step/action_max": step_log['action_max'],
                    "step/done": step_log['done'],
                    "step/portfolio_layout": step_log['portfolio_layout']
                })
        
        train_start = time.time()
        if len(states) > 0:
            batch_metrics = train_on_batch(
                model, optimizer, states, raw_states, actions, returns, advantages, 
                is_random_flags, batch_count, None, None,
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
            
            if batch_count % 50 == 0:
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
            
            if batch_count % 2 == 0:
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
                'batch/critic_loss': 0.0,
                'batch/entropy': 0.0,
                'batch/encoder_loss': 0.0,
            }
            train_time = 0.0
        
        avg_recent_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0.0
        
        # Weighted average - recent outperformance is more important
        if len(recent_outperformance) > 1:
            weights = np.linspace(0.5, 1.0, len(recent_outperformance))
            avg_recent_outperformance = np.average(list(recent_outperformance), weights=weights)
        elif len(recent_outperformance) == 1:
            avg_recent_outperformance = recent_outperformance[0]
        else:
            avg_recent_outperformance = 1.0  # No outperformance data yet
        
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
            'batch/mean_value': batch_info['mean_value'],
            'batch/avg_recent_reward': avg_recent_reward,
            'batch/avg_recent_outperformance': (avg_recent_outperformance - 1) * 100,
            'batch/weighted_avg_loss': weighted_avg_loss,
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
        
        wandb.log(batch_metrics)
        
        print(f"Collected {steps_collected} steps in {collect_time:.2f}s")
        
        # Add diagnostic info about rewards and values
        if len(states) > 0 and len(returns) > 0:
            reward_mean = batch_info['total_reward'] / max(1, steps_collected)
            return_stats = f"Returns: min={np.min(returns):.2f}, max={np.max(returns):.2f}, mean={np.mean(returns):.2f}"
            value_stats = f"Values: min={np.min(values):.2f}, max={np.max(values):.2f}, mean={np.mean(values):.2f}"
            print(f"  Avg Reward/step: {reward_mean:.3f}")
            print(f"  {return_stats}")
            print(f"  {value_stats}")
        
        if TRAIN_FEATURE_REDUCER:
            print(f"Trained in {train_time:.2f}s, Loss: {batch_metrics['batch/loss']:.4f} (Actor: {batch_metrics['batch/actor_loss']:.4f}, Critic: {batch_metrics['batch/critic_loss']:.4f})")
            print(f"Encoder Loss: {batch_metrics.get('batch/encoder_loss', 0):.4f}")
        else:
            print(f"Trained in {train_time:.2f}s, Loss: {batch_metrics['batch/loss']:.4f} (Actor: {batch_metrics['batch/actor_loss']:.4f}, Critic: {batch_metrics['batch/critic_loss']:.4f})")
        print(f"Reward: {batch_info['total_reward']:.2f}, Outperformance: {(batch_info['outperformance'] - 1) * 100:.2f}%")
        print(f"Recent avg outperformance: {(avg_recent_outperformance - 1) * 100:.2f}%")

        # Log best model tracking metrics
        wandb.log({
            'best/avg_recent_outperformance': (avg_recent_outperformance - 1) * 100,
            'best/best_outperformance': (best_outperformance - 1) * 100,
            'best/current_outperformance': (batch_info['outperformance'] - 1) * 100
        })

        if batch_count <= WARMUP_BATCHES:
            print(f"[WARMUP {batch_count}/{WARMUP_BATCHES}] Ignoring early performance for best model tracking")
            if avg_recent_outperformance > best_outperformance:
                best_outperformance = avg_recent_outperformance
        elif avg_recent_outperformance > best_outperformance:
            best_outperformance = avg_recent_outperformance
            model.save_weights('models/best_model.weights.h5')
            wandb.save('models/best_model.weights.h5')
            
            no_improvement = 0
            print(f"NEW BEST OUTPERFORMANCE: {(best_outperformance - 1) * 100:.2f}%")
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
        print(f"Current LR: {current_lr:.6f}, Epsilon: {epsilon:.4f}, Best outperformance: {(best_outperformance - 1) * 100:.2f}%")
        
        if batch_count % 2 == 0:
            checkpoint_path = f'models/checkpoint_batch{batch_count}.weights.h5'
            model.save_weights(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        del states, actions, returns
        gc.collect()
    
    model.save_weights('models/final_model.weights.h5')
    wandb.save('models/final_model.weights.h5')
    
    total_time = time.time() - start_time
    print(f"\nTRAINING COMPLETE!")
    print(f"Total batches: {batch_count}")
    print(f"Total steps: {global_step}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Best outperformance: {(best_outperformance - 1) * 100:.2f}%")
    print(f"Final model saved!")
    
    wandb.finish()