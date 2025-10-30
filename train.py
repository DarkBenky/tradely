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

DEBUG = False

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_model(obs_shape, num_assets, config=None):
    if config is None:
        config = {
            "num_transformer_blocks": 3,
            "embed_dim": 128,
            "num_heads": 8,
            "ff_dim": 512,
            "dropout_rate": 0.2,
            "dense_hidden": 256,
            "embedding_dim": 512
        }
    
    inputs = layers.Input(shape=obs_shape)
    
    x = layers.Dense(config["embedding_dim"], activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config["dropout_rate"])(x)
    
    seq_len = config["embedding_dim"] // config["embed_dim"]
    x = layers.Reshape((seq_len, config["embed_dim"]))(x)
    
    for _ in range(config["num_transformer_blocks"]):
        x = TransformerBlock(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            rate=config["dropout_rate"]
        )(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(config["dense_hidden"], activation='relu')(x)
    x = layers.Dropout(config["dropout_rate"] * 1.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    
    outputs = layers.Dense(num_assets, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def collect_batch_data(env, model, batch_size, gamma, current_step):
    """Collect exactly one batch of data (batch_size steps)"""
    states, actions, rewards = [], [], []
    step_logs = []
    
    if current_step >= env.max_steps:
        state = env.reset()
    else:
        state = env.get_observation()
    
    done = False
    steps_collected = 0
    
    print(f"  Collecting {batch_size} steps for batch...")
    
    while steps_collected < batch_size and not done:
        # Get action from model
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        action_probs = model(state_tensor, training=False)
        action = action_probs[0].numpy()
        
        # Take step
        next_state, reward, done, info = env.step(action)
        env.render()

        # Store step data
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        # Log step details
        step_log = {
            'step_global': current_step + steps_collected,
            'step_batch': steps_collected,
            'reward': reward,
            'portfolio_value': info.get('portfolio_value', 0),
            'benchmark_value': info.get('benchmark_value', 0),
            'outperformance': info.get('outperformance', 0),
            'action_mean': np.mean(action),
            'action_std': np.std(action),
            'action_max': np.max(action),
            'done': done
        }
        
        wandb.log({
            "step/global": step_log['step_global'],
            "step/batch": step_log['step_batch'],
            "step/reward": step_log['reward'],
            "step/portfolio_value": step_log['portfolio_value'],
            "step/benchmark_value": step_log['benchmark_value'],
            "step/outperformance": (step_log['outperformance'] - 1) * 100,
            "step/action_mean": step_log['action_mean'],
            "step/action_std": step_log['action_std'],
            "step/action_max": step_log['action_max'],
            "step/done": step_log['done'],
            "step/portfolio_layout": info['portfolio_layout']
        })

        step_logs.append(step_log)
        
        state = next_state
        steps_collected += 1
        
        # Safety limit
        if steps_collected >= 10000:
            print(f"  Warning: Episode exceeded 10000 steps, forcing termination")
            break
    
    # Compute returns for the collected steps
    if steps_collected > 0:
        returns = []
        G = 0
        # Compute returns in reverse to handle discounting properly
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns, dtype=np.float32)
        # Normalize returns for this batch only
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # At the end of collect_batch_data:
    if not os.path.exists("trainData"):
        os.makedirs("trainData")

    # Append to single file
    with open("trainData/training_data.pkl", 'ab') as f:
        pickle.dump({'states': states, 'actions': actions, 'returns': returns}, f)
    
    batch_info = {
        'steps_collected': steps_collected,
        'total_reward': sum(rewards),
        'portfolio_value': info.get('portfolio_value', 0) if steps_collected > 0 else 0,
        'benchmark_value': info.get('benchmark_value', 0) if steps_collected > 0 else 0,
        'outperformance': info.get('outperformance', 0) if steps_collected > 0 else 0,
        'done': done
    }
    
    return states, actions, returns, batch_info, step_logs

def train_on_batch(model, optimizer, batch_states, batch_actions, batch_returns, batch_num):
    """Train model on a single batch with detailed logging"""
    states_batch = np.array(batch_states, dtype=np.float32)
    actions_batch = np.array(batch_actions, dtype=np.float32)
    returns_batch = np.array(batch_returns, dtype=np.float32)
    
    with tf.GradientTape() as tape:
        action_probs = model(states_batch, training=True)
        action_probs_clipped = tf.clip_by_value(action_probs, 1e-8, 1.0)
        log_probs = tf.reduce_sum(actions_batch * tf.math.log(action_probs_clipped), axis=1)
        loss = -tf.reduce_mean(log_probs * returns_batch)
    
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Detailed metrics
    action_probs_np = action_probs.numpy()
    metrics = {
        'batch/loss': loss.numpy(),
        'batch/mean_action_prob': tf.reduce_mean(action_probs).numpy(),
        'batch/max_action_prob': tf.reduce_max(action_probs).numpy(),
        'batch/min_action_prob': tf.reduce_min(action_probs).numpy(),
        'batch/action_std': np.std(action_probs_np),
        'batch/entropy': -np.mean(np.sum(action_probs_np * np.log(action_probs_np + 1e-8), axis=1)),
        'batch/mean_return': np.mean(batch_returns),
        'batch/std_return': np.std(batch_returns),
        'batch/grad_norm': tf.linalg.global_norm(grads).numpy() if grads[0] is not None else 0.0,
    }
    
    return metrics

def offline_pretrain(env, model, optimizer, num_episodes=50, gamma=0.99, batch_size=256, epochs=20):
    """Pretrain on saved data"""
    print("\n=== OFFLINE PRETRAINING ===")
    
    train_data_dir = 'trainData'
    
    if not os.path.exists(train_data_dir) or not os.path.isdir(train_data_dir):
        print(f"trainData folder not found, skipping pretraining")
        return
    
    all_states = []
    all_actions = []
    all_returns = []
    
    data_files = [f for f in os.listdir(train_data_dir) if f.endswith(('.npz', '.npy', '.pkl'))]
    
    if not data_files:
        print("No training data files found, skipping pretraining")
        return
    
    print(f"Found {len(data_files)} data files")
    
    for data_file in data_files:
        file_path = os.path.join(train_data_dir, data_file)
        print(f"Loading {data_file}...")
        
        try:
            if data_file.endswith('.npz'):
                data = np.load(file_path, allow_pickle=True)
                all_states.extend(list(data['states']))
                all_actions.extend(list(data['actions']))
                all_returns.extend(data['returns'])
            elif data_file.endswith('.npy'):
                data = np.load(file_path, allow_pickle=True)
                
                if 'states' in data_file.lower():
                    all_states.extend(list(data))
                elif 'actions' in data_file.lower():
                    all_actions.extend(list(data))
                elif 'returns' in data_file.lower():
                    all_returns.extend(data)
            elif data_file.endswith('.pkl'):
                # Handle pickle files - they might contain multiple batches
                with open(file_path, 'rb') as f:
                    while True:
                        try:
                            batch_data = pickle.load(f)
                            if isinstance(batch_data, dict):
                                if 'states' in batch_data:
                                    all_states.extend(batch_data['states'])
                                if 'actions' in batch_data:
                                    all_actions.extend(batch_data['actions'])
                                if 'returns' in batch_data:
                                    all_returns.extend(batch_data['returns'])
                        except EOFError:
                            break
        except Exception as e:
            print(f"Error loading {data_file}: {e}")
            continue
    
    if len(all_states) == 0:
        print("No valid data loaded, skipping pretraining")
        return
    
    # Ensure we have matching lengths
    min_length = min(len(all_states), len(all_actions), len(all_returns))
    all_states = all_states[:min_length]
    all_actions = all_actions[:min_length]
    all_returns = all_returns[:min_length]
    
    print(f"Loaded {len(all_states)} samples from {len(data_files)} files")
    
    # Normalize returns
    all_returns = np.array(all_returns, dtype=np.float32)
    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)
    
    dataset_size = len(all_states)
    print(f"\nTraining on {dataset_size} samples for {epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        indices = np.random.permutation(dataset_size)
        epoch_losses = []
        
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = [all_states[idx] for idx in batch_indices]
            batch_actions = [all_actions[idx] for idx in batch_indices]
            batch_returns = all_returns[batch_indices]
            
            # Simple training for pretraining
            with tf.GradientTape() as tape:
                action_probs = model(batch_states, training=True)
                action_probs_clipped = tf.clip_by_value(action_probs, 1e-8, 1.0)
                log_probs = tf.reduce_sum(batch_actions * tf.math.log(action_probs_clipped), axis=1)
                loss = -tf.reduce_mean(log_probs * batch_returns)
            
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(loss.numpy())
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        
        wandb.log({
            "pretrain/epoch": epoch + 1,
            "pretrain/loss": avg_loss,
            "pretrain/epoch_time": epoch_time,
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.1f}s")
    
    print("Offline pretraining complete!\n")

if __name__ == "__main__":
    ENABLE_PRETRAINING = True
    
    config = {
        "architecture": "transformer",
        "num_transformer_blocks": 5,
        "embed_dim": 256,
        "num_heads": 8,
        "ff_dim": 512,
        "dropout_rate": 0.2,
        "dense_hidden": 256,
        "embedding_dim": 2048,
        "initial_lr": 0.0001,
        "pretrain_episodes": 3,
        "pretrain_epochs": 3,
        "pretrain_batch_size": 256,
        "online_total_batches": 5000,  # Total batches to process
        "online_batch_size": 256,  # Steps per batch
        "gamma": 0.99,
        "lr_patience": 100,  # In batches
        "lr_decay": 0.5,
        "log_step_frequency": 1,  # Log every N steps within batch
    }
    
    wandb.init(project="portfolio-trading-online-batch", config=config)
    config = wandb.config
    
    print("Initializing environment...")
    env = PortfolioEnv(max_records=100_000)
    observation = env.reset()
    
    obs_shape = observation.shape
    num_assets = len(env.asset_names)
    
    print("Building model...")
    model = build_model(obs_shape, num_assets, config)
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.initial_lr)
    
    # Load pretrained model if exists
    pretrained_model_path = 'pretrained_model.weights.h5'
    best_model_path = 'best_model.weights.h5'

    if os.path.exists(best_model_path):
        print(f"\n=== LOADING BEST MODEL FROM {best_model_path} ===")
        model.load_weights(best_model_path)
        print("Best model loaded successfully!\n")
        ENABLE_PRETRAINING = False
    elif os.path.exists(pretrained_model_path):
        print(f"\n=== LOADING PRETRAINED MODEL FROM {pretrained_model_path} ===")
        model.load_weights(pretrained_model_path)
        print("Pretrained model loaded successfully!\n")
        ENABLE_PRETRAINING = False

    # check if pretraining data exists
    if (not os.path.exists('trainData') or not os.path.isdir('trainData')) and not os.path.exists('trainData/train_data.pkl'):
        print("No pretraining data found, skipping pretraining")
        ENABLE_PRETRAINING = False
    
    if ENABLE_PRETRAINING:
        offline_pretrain(env, model, optimizer, 
                        num_episodes=config.pretrain_episodes, 
                        gamma=config.gamma,
                        batch_size=config.pretrain_batch_size,
                        epochs=config.pretrain_epochs)
        model.save_weights('pretrained_model.weights.h5')
    else:
        print("\n=== SKIPPING PRETRAINING ===")
    
    print("\n=== ONLINE TRAINING (BATCH-BY-BATCH) ===")
    print(f"Batch size: {config.online_batch_size}")
    print(f"Total target batches: {config.online_total_batches}")
    
    best_avg_reward = float('-inf')
    no_improvement = 0
    current_lr = config.initial_lr
    global_step = 0
    batch_count = 0
    
    # Track recent performance
    recent_rewards = deque(maxlen=50)  # Track last 50 batches
    
    print(f"\n{'='*80}")
    print("STARTING BATCH TRAINING LOOP")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    while batch_count < config.online_total_batches:
        batch_count += 1
        print(f"\n--- Batch {batch_count}/{config.online_total_batches} ---")
        
        # Collect exactly one batch of data
        collect_start = time.time()
        states, actions, returns, batch_info, step_logs = collect_batch_data(
            env, model, config.online_batch_size, config.gamma, global_step
        )
        collect_time = time.time() - collect_start
        
        # Update global step counter
        steps_collected = len(states)
        global_step += steps_collected
        recent_rewards.append(batch_info['total_reward'])
        
        # Log step-level details (sample some steps to avoid too much logging)
        for i, step_log in enumerate(step_logs):
            if i % config.log_step_frequency == 0:  # Log every Nth step
                wandb.log({
                    "step/global": step_log['step_global'],
                    "step/batch": step_log['step_batch'],
                    "step/reward": step_log['reward'],
                    "step/portfolio_value": step_log['portfolio_value'],
                    "step/benchmark_value": step_log['benchmark_value'],
                    "step/outperformance": (step_log['outperformance'] - 1) * 100,
                    "step/action_mean": step_log['action_mean'],
                    "step/action_std": step_log['action_std'],
                    "step/action_max": step_log['action_max'],
                })
        
        # Train on the collected batch
        train_start = time.time()
        if len(states) > 0:
            batch_metrics = train_on_batch(model, optimizer, states, actions, returns, batch_count)
            train_time = time.time() - train_start
        else:
            batch_metrics = {'batch/loss': 0.0}  # No data collected
            train_time = 0.0
        
        # Calculate performance metrics
        avg_recent_reward = np.mean(recent_rewards) if recent_rewards else batch_info['total_reward']
        
        # Log batch metrics
        batch_metrics.update({
            'batch/number': batch_count,
            'batch/global_step': global_step,
            'batch/steps_collected': steps_collected,
            'batch/total_reward': batch_info['total_reward'],
            'batch/avg_recent_reward': avg_recent_reward,
            'batch/portfolio_value': batch_info['portfolio_value'],
            'batch/benchmark_value': batch_info['benchmark_value'],
            'batch/outperformance': (batch_info['outperformance'] - 1) * 100,
            'batch/collection_time': collect_time,
            'batch/training_time': train_time,
            'batch/total_time': collect_time + train_time,
            'batch/steps_per_second': steps_collected / collect_time if collect_time > 0 else 0,
            'batch/learning_rate': current_lr,
        })
        
        wandb.log(batch_metrics)
        
        print(f"Collected {steps_collected} steps in {collect_time:.2f}s")
        print(f"Trained in {train_time:.2f}s, Loss: {batch_metrics['batch/loss']:.4f}")
        print(f"Reward: {batch_info['total_reward']:.2f}, Recent avg: {avg_recent_reward:.2f}")
        
        # Track best model based on recent performance
        if avg_recent_reward > best_avg_reward:
            best_avg_reward = avg_recent_reward
            model.save_weights('best_model.weights.h5')
            wandb.save('best_model.weights.h5')
            no_improvement = 0
            print(f"★ NEW BEST AVG REWARD: {best_avg_reward:.2f}")
        else:
            no_improvement += 1
        
        # Learning rate decay based on batches
        if no_improvement >= config.lr_patience:
            current_lr *= config.lr_decay
            if current_lr < 1e-6:
                current_lr = 1e-6
            optimizer.learning_rate.assign(current_lr)
            print(f"\n>>> Learning rate reduced to {current_lr:.6f} <<<")
            no_improvement = 0
        
        # Progress reporting
        progress = (batch_count / config.online_total_batches) * 100
        elapsed_time = time.time() - start_time
        batches_per_sec = batch_count / elapsed_time if elapsed_time > 0 else 0
        steps_per_sec = global_step / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nProgress: {progress:.1f}% ({batch_count}/{config.online_total_batches} batches)")
        print(f"Time elapsed: {elapsed_time:.1f}s, Batches/sec: {batches_per_sec:.2f}, Steps/sec: {steps_per_sec:.2f}")
        print(f"Current LR: {current_lr:.6f}, Best avg reward: {best_avg_reward:.2f}")
        
        # Save checkpoint every 100 batches
        if batch_count % 100 == 0:
            checkpoint_path = f'checkpoint_batch{batch_count}.weights.h5'
            model.save_weights(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Final save
    model.save_weights('final_model.weights.h5')
    wandb.save('final_model.weights.h5')
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total batches: {batch_count}")
    print(f"Total steps: {global_step}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Best avg reward: {best_avg_reward:.2f}")
    print(f"Final model saved!")
    
    wandb.finish()

#TODO Add loading of the pickle file