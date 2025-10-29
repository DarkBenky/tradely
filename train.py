import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
from portfolio_env import PortfolioEnv
import os

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

def build_model(obs_shape, num_assets):
    inputs = layers.Input(shape=obs_shape)
    
    x = layers.Dense(8192, activation='relu')(inputs)
    x = layers.Reshape((-1, 128))(x)
    
    x = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512, rate=0.1)(x)
    x = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512, rate=0.1)(x)
    x = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512, rate=0.1)(x)
    x = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512, rate=0.1)(x)
    x = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512, rate=0.1)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_assets, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_episode(env, model, optimizer, gamma=0.99):
    states, actions, rewards = [], [], []
    
    state = env.reset()
    done = False
    episode_reward = 0
    
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
        
        state = next_state
        
        if len(states) >= 32 or done:
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            
            returns = np.array(returns, dtype=np.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            states_batch = np.array(states, dtype=np.float32)
            actions_batch = np.array(actions, dtype=np.float32)
            
            with tf.GradientTape() as tape:
                action_probs = model(states_batch, training=True)
                
                action_probs_clipped = tf.clip_by_value(action_probs, 1e-8, 1.0)
                log_probs = tf.reduce_sum(actions_batch * tf.math.log(action_probs_clipped), axis=1)
                
                loss = -tf.reduce_mean(log_probs * returns)
            
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            states, actions, rewards = [], [], []
    
    return episode_reward, info

if __name__ == "__main__":
    env = PortfolioEnv()
    observation = env.reset()
    
    obs_shape = observation.shape
    num_assets = len(env.asset_names)
    
    model = build_model(obs_shape, num_assets)
    model.summary()
    plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.summary()
    
    num_episodes = 10
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        episode_reward, info = train_episode(env, model, optimizer)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            model.save_weights(f'best_model_ep{episode}.h5')
        
        print(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f} - "
                f"Portfolio: ${info['portfolio_value']:.2f} - "
                f"Benchmark: ${info['benchmark_value']:.2f} - "
                f"Outperformance: {(info['outperformance']-1)*100:.2f}%")
        
        if episode % 50 == 0 and episode > 0:
            model.save_weights(f'model_checkpoint_ep{episode}.h5')
    
    model.save_weights('final_model.h5')
    print(f"\nTraining complete! Best reward: {best_reward:.2f}")