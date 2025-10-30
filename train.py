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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def collect_episode_data(env, model):
    states, actions, rewards = [], [], []
    
    state = env.reset()
    done = False
    
    while not done:
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        action_probs = model(state_tensor, training=False)
        action = action_probs[0].numpy()
        
        next_state, reward, done, info = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    return states, actions, rewards, info

def collect_single_episode(model, gamma):
    env = PortfolioEnv()
    states, actions, rewards, info = collect_episode_data(env, model)
    
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    episode_reward = sum(rewards)
    return states, actions, returns, episode_reward, info

def append_memmap(path, new_data):
    new_data = np.asarray(new_data, dtype=np.float32)
    if not os.path.exists(path):
        # Create new file
        np.save(path, new_data)
        return
    
    # Load existing file shape
    old = np.load(path, mmap_mode='r')
    old_shape = old.shape

    # Only supports appending along axis 0
    new_shape = (old_shape[0] + new_data.shape[0],) + old_shape[1:]

    # Create temporary file
    tmp_path = path + '.tmp'
    mm = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=np.float32, shape=new_shape)

    # Copy existing data (streamed from disk)
    mm[:old_shape[0]] = old
    mm[old_shape[0]:] = new_data
    mm.flush()
    del mm
    os.replace(tmp_path, path)

def train_on_batch(model, optimizer, states, actions, returns):
    states_batch = np.array(states, dtype=np.float32)
    actions_batch = np.array(actions, dtype=np.float32)

    r = np.array(returns, dtype=np.float32)
    
    max_file_size = 10 * 1024 * 1024 * 1024
    
    states_path = 'trainData/states.npy'
    actions_path = 'trainData/actions.npy'
    returns_path = 'trainData/returns.npy'
    
    def get_file_size(path):
        return os.path.getsize(path) if os.path.exists(path) else 0
    
    if get_file_size(states_path) < max_file_size and get_file_size(actions_path) < max_file_size and get_file_size(returns_path) < max_file_size:
        append_memmap(states_path, states)
        append_memmap(actions_path, actions)
        append_memmap(returns_path, r)

    with tf.GradientTape() as tape:
        action_probs = model(states_batch, training=True)
        action_probs_clipped = tf.clip_by_value(action_probs, 1e-8, 1.0)
        log_probs = tf.reduce_sum(actions_batch * tf.math.log(action_probs_clipped), axis=1)
        loss = -tf.reduce_mean(log_probs * returns)
        
        mean_action_prob = tf.reduce_mean(action_probs).numpy()
        max_action_prob = tf.reduce_max(action_probs).numpy()
        min_action_prob = tf.reduce_min(action_probs).numpy()
        
        wandb.log({
            "train_on_batch/batch_loss": loss.numpy(),
            "train_on_batch/mean_action_prob": mean_action_prob,
            "train_on_batch/max_action_prob": max_action_prob,
            "train_on_batch/min_action_prob": min_action_prob,
            "train_on_batch/batch_size": len(states_batch)
        })
    
    grads = tape.gradient(loss, model.trainable_variables)
    grad_norms = [tf.norm(g).numpy() for g in grads if g is not None]
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
    max_grad_norm = np.max(grad_norms) if grad_norms else 0
    
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    wandb.log({
        "train_on_batch/avg_grad_norm": avg_grad_norm,
        "train_on_batch/max_grad_norm": max_grad_norm
    })
    
    return loss.numpy()

def offline_pretrain(env, model, optimizer, num_episodes=50, gamma=0.99, batch_size=256, epochs=20):
    print("\n=== OFFLINE PRETRAINING ===")
    
    train_data_dir = 'trainData'
    
    if os.path.exists(train_data_dir) and os.path.isdir(train_data_dir):
        print(f"Loading training data from {train_data_dir} folder...")
        all_states, all_actions, all_returns = [], [], []
        
        data_files = [f for f in os.listdir(train_data_dir) if f.endswith('.npz') or f.endswith('.npy')]
        if not data_files:
            print(f"No .npz or .npy files found in {train_data_dir}, skipping pretraining")
            return
        
        for data_file in data_files:
            file_path = os.path.join(train_data_dir, data_file)
            print(f"Loading {data_file}...")
            
            if data_file.endswith('.npz'):
                data = np.load(file_path, allow_pickle=True)
                all_states.extend(list(data['states']))
                all_actions.extend(list(data['actions']))
                all_returns.extend(data['returns'])
            elif data_file.endswith('.npy'):
                data = np.load(file_path, allow_pickle=True)
                
                if data.shape == () and isinstance(data.item(), dict):
                    data = data.item()
                    if 'states' in data:
                        all_states.extend(list(data['states']))
                    if 'actions' in data:
                        all_actions.extend(list(data['actions']))
                    if 'returns' in data:
                        all_returns.extend(data['returns'])
                else:
                    if 'states' in data_file.lower():
                        all_states.extend(list(data))
                    elif 'actions' in data_file.lower():
                        all_actions.extend(list(data))
                    elif 'returns' in data_file.lower():
                        all_returns.extend(data)
        
        if len(all_states) == 0:
            print("No valid data loaded, skipping pretraining")
            return
        
        print(f"Loaded {len(all_states)} samples from {len(data_files)} files")
        
        all_returns = np.array(all_returns, dtype=np.float32)
        all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)
    else:
        print(f"trainData folder not found, skipping pretraining")
        return
    
    dataset_size = len(all_states)
    
    print(f"\nTraining on {dataset_size} samples for {epochs} epochs...")
    
    wandb.log({
        "pretrain/dataset_size": dataset_size,
        "pretrain/num_epochs": epochs,
        "pretrain/batch_size": batch_size
    })
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        indices = np.random.permutation(dataset_size)
        epoch_loss = 0
        num_batches = 0
        batch_losses = []
        
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = [all_states[idx] for idx in batch_indices]
            batch_actions = [all_actions[idx] for idx in batch_indices]
            batch_returns = all_returns[batch_indices]
            
            loss = train_on_batch(model, optimizer, batch_states, batch_actions, batch_returns)
            epoch_loss += loss
            batch_losses.append(loss)
            num_batches += 1
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        min_loss = min(batch_losses)
        max_loss = max(batch_losses)
        std_loss = np.std(batch_losses)
        
        elapsed = time.time() - start_time
        remaining_epochs = epochs - (epoch + 1)
        avg_epoch_time = elapsed / (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        
        wandb.log({
            "pretrain/epoch": epoch + 1,
            "pretrain/loss": avg_loss,
            "pretrain/loss_min": min_loss,
            "pretrain/loss_max": max_loss,
            "pretrain/loss_std": std_loss,
            "pretrain/epoch_time": epoch_time,
            "pretrain/samples_per_second": dataset_size / epoch_time,
            "pretrain/eta_seconds": eta_seconds
        })
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {avg_loss:.4f} (±{std_loss:.4f}) - "
              f"Time: {epoch_time:.1f}s - "
              f"ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s")
    
    total_time = time.time() - start_time
    wandb.log({
        "pretrain/total_time": total_time,
        "pretrain/avg_epoch_time": total_time / epochs
    })
    
    print("Offline pretraining complete!\n")

def train_episode(env, model, optimizer, gamma=0.99, batch_size=32, episode_num=0):
    states, actions, rewards = [], [], []
    
    state = env.reset()
    done = False
    episode_reward = 0
    step = 0
    
    while not done:
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            action_probs = model(state_tensor, training=True)
            action = action_probs[0].numpy()
            
        next_state, reward, done, info = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        
        step += 1
        if step % 2 == 0:
            portfolio_total = info['portfolio_value']
            cash_ratio = env.portfolio['cash'] / portfolio_total if portfolio_total > 0 else 0
            
            allocation_log = {
                f"online/ep{episode_num}_step": step,
                f"online/ep{episode_num}_portfolio_value": info['portfolio_value'],
                f"online/ep{episode_num}_benchmark_value": info['benchmark_value'],
                f"online/ep{episode_num}_reward": reward,
                f"online/ep{episode_num}_outperformance": (info['outperformance'] - 1) * 100,
                f"online/ep{episode_num}_cash_ratio": cash_ratio * 100,
            }
            
            for i, symbol in enumerate(env.asset_names):
                asset_value = env.portfolio[symbol] * env._get_current_priced(symbol)
                asset_ratio = asset_value / portfolio_total if portfolio_total > 0 else 0
                allocation_log[f"online/ep{episode_num}_allocation_{symbol}"] = asset_ratio * 100
            
            wandb.log(allocation_log)
        
        state = next_state
        
        if len(states) >= batch_size or done:
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            
            returns = np.array(returns, dtype=np.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            train_on_batch(model, optimizer, states, actions, returns)
            states, actions, rewards = [], [], []
    
    return episode_reward, info

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
        "initial_lr": 0.001,
        "pretrain_episodes": 10,
        "pretrain_epochs": 10,
        "pretrain_batch_size": 256,
        "online_episodes": 500,
        "online_batch_size": 32,
        "gamma": 0.99,
        "lr_patience": 5,
        "lr_decay": 0.5
    }
    
    wandb.init(project="portfolio-trading", config=config)
    config = wandb.config
    
    env = PortfolioEnv(max_records=100000)
    observation = env.reset()
    
    obs_shape = observation.shape
    num_assets = len(env.asset_names)
    
    model = build_model(obs_shape, num_assets, config)
    model.summary()
    plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)
    wandb.save("model_architecture.png")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.initial_lr)
    
    pretrained_model_path = 'pretrained_model.weights.h5'
    if os.path.exists(pretrained_model_path):
        print(f"\n=== LOADING PRETRAINED MODEL FROM {pretrained_model_path} ===")
        model.load_weights(pretrained_model_path)
        print("Pretrained model loaded successfully!\n")
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
    
    print("\n=== ONLINE TRAINING ===")
    best_reward = float('-inf')
    no_improvement = 0
    current_lr = config.initial_lr
    
    for episode in range(config.online_episodes):
        episode_reward, info = train_episode(env, model, optimizer, 
                                            gamma=config.gamma,
                                            batch_size=config.online_batch_size,
                                            episode_num=episode)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            model.save_weights(f'best_model.weights.h5')
            wandb.save('best_model.weights.h5')
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement >= config.lr_patience:
            current_lr *= config.lr_decay
            if current_lr < 1e-6:
                current_lr = 1e-6
            optimizer.learning_rate.assign(current_lr)
            print(f"\n>>> Learning rate reduced to {current_lr:.6f} <<<\n")
            no_improvement = 0
        
        wandb.log({
            "online/episode": episode,
            "online/episode_reward": episode_reward,
            "online/portfolio_value": info['portfolio_value'],
            "online/benchmark_value": info['benchmark_value'],
            "online/outperformance": (info['outperformance'] - 1) * 100,
            "online/learning_rate": current_lr,
            "online/best_reward": best_reward,
            "online/no_improvement_count": no_improvement
        })
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{config.online_episodes} - Reward: {episode_reward:.2f} - "
                  f"Portfolio: ${info['portfolio_value']:.2f} - "
                  f"Benchmark: ${info['benchmark_value']:.2f} - "
                  f"Outperformance: {(info['outperformance']-1)*100:.2f}% - "
                  f"LR: {current_lr:.6f}")
        
        if episode % 50 == 0 and episode > 0:
            model.save_weights(f'checkpoint_ep{episode}.weights.h5')
    
    model.save_weights('final_model.weights.h5')
    wandb.save('final_model.weights.h5')
    print(f"\nTraining complete! Best reward: {best_reward:.2f}")
    wandb.finish()