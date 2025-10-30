import tensorflow as tf
import numpy as np
import pandas as pd
from portfolio_env import PortfolioEnv
from train import build_model
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

def get_feature_names(env):
    """Extract feature names from the environment observation"""
    feature_names = []
    
    exclude_cols = ['timestamp', 'close_time', 'ignore']
    sample_symbol = list(env.df.keys())[0]
    all_available_cols = [col for col in env.df[sample_symbol].columns 
                         if col not in exclude_cols]
    
    aggregation_info = {
        '5m': env.lookback_window_size,
        '15m': env.high_timeframes_count[0],
        '1h': env.high_timeframes_count[1],
        '4h': env.high_timeframes_count[2],
        '1d': env.high_timeframes_count[3]
    }
    
    for symbol in env.asset_names:
        for timeframe, count in aggregation_info.items():
            for col in all_available_cols:
                for step in range(count):
                    feature_names.append(f"{symbol}_{timeframe}_{col}_t{step}")
    
    for symbol in env.asset_names:
        feature_names.append(f"portfolio_holdings_{symbol}")
    
    feature_names.extend([
        'portfolio_cash_pct',
        'portfolio_value_ratio',
        'benchmark_ratio',
        'outperformance'
    ])
    
    return feature_names

def collect_data_samples(env, model, num_episodes=10, max_steps=200):
    """Collect state-action-reward samples from the environment"""
    states = []
    actions = []
    rewards = []
    
    print(f"Collecting data from {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
            action_probs = model(state_tensor, training=False)
            action = action_probs[0].numpy()
            
            next_state, reward, done, info = env.step(action)
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            state = next_state
            steps += 1
        
        states.extend(episode_states)
        actions.extend(episode_actions)
        rewards.extend(episode_rewards)
        
        print(f"  Episode {episode+1}/{num_episodes}: {len(episode_states)} steps, avg reward: {np.mean(episode_rewards):.4f}")
    
    return np.array(states), np.array(actions), np.array(rewards)

def analyze_gradient_importance(model, states, actions, batch_size=10):
    """Analyze feature importance using gradients"""
    print("\nCalculating gradient-based importance...")
    
    num_samples = min(len(states), 100)
    avg_gradients = np.zeros(states.shape[1], dtype=np.float32)
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        state_batch = states[i:end_idx]
        action_batch = actions[i:end_idx]
        
        state_tensor = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(state_tensor)
            predictions = model(state_tensor, training=False)
            loss = tf.reduce_sum(predictions * action_tensor)
        
        grads = tape.gradient(loss, state_tensor)
        avg_gradients += np.sum(np.abs(grads.numpy()), axis=0)
        
        del state_tensor, action_tensor, grads
        tf.keras.backend.clear_session()
        
        if i % 50 == 0:
            print(f"  Processed {i}/{num_samples} samples...")
    
    avg_gradients /= num_samples
    return avg_gradients

def analyze_permutation_importance(states, actions, rewards, sample_size=500):
    """Use permutation importance with a surrogate model"""
    print("\nTraining surrogate model for permutation importance...")
    
    if len(states) > sample_size:
        indices = np.random.choice(len(states), sample_size, replace=False)
        states_sample = states[indices]
        actions_sample = actions[indices]
    else:
        states_sample = states
        actions_sample = actions
    
    action_labels = np.argmax(actions_sample, axis=1)
    
    rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=4)
    rf_model.fit(states_sample, action_labels)
    
    print("Calculating permutation importance...")
    result = permutation_importance(rf_model, states_sample, action_labels, 
                                   n_repeats=2, random_state=42, n_jobs=4)
    
    del rf_model
    
    return result.importances_mean

def create_importance_table(feature_names, gradient_importance, permutation_importance, top_n=50):
    """Create a comprehensive importance table"""
    
    if len(feature_names) != len(gradient_importance):
        print(f"Warning: Feature names ({len(feature_names)}) and gradient importance ({len(gradient_importance)}) length mismatch")
        min_len = min(len(feature_names), len(gradient_importance))
        feature_names = feature_names[:min_len]
        gradient_importance = gradient_importance[:min_len]
        permutation_importance = permutation_importance[:min_len]
    
    gradient_normalized = (gradient_importance - gradient_importance.min()) / (gradient_importance.max() - gradient_importance.min() + 1e-8)
    perm_normalized = (permutation_importance - permutation_importance.min()) / (permutation_importance.max() - permutation_importance.min() + 1e-8)
    
    combined_importance = (gradient_normalized + perm_normalized) / 2
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Gradient_Importance': gradient_importance,
        'Permutation_Importance': permutation_importance,
        'Combined_Importance': combined_importance
    })
    
    df = df.sort_values('Combined_Importance', ascending=False)
    
    print(f"\n{'='*100}")
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print(f"{'='*100}")
    print(f"{'Rank':<6} {'Feature':<50} {'Gradient':<15} {'Permutation':<15} {'Combined':<15}")
    print(f"{'-'*100}")
    
    for idx, row in df.head(top_n).iterrows():
        print(f"{df.index.get_loc(idx)+1:<6} {row['Feature']:<50} {row['Gradient_Importance']:<15.6f} {row['Permutation_Importance']:<15.6f} {row['Combined_Importance']:<15.6f}")
    
    return df

def analyze_by_category(df):
    """Group features by category and analyze"""
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE BY CATEGORY")
    print(f"{'='*80}")
    
    categories = {
        'close': 'Price (Close)',
        'open': 'Price (Open)',
        'high': 'Price (High)',
        'low': 'Price (Low)',
        'volume': 'Volume',
        'rsi': 'RSI',
        'macd': 'MACD',
        'ema': 'EMA',
        'sma': 'SMA',
        'bb': 'Bollinger Bands',
        'atr': 'ATR',
        'stoch': 'Stochastic',
        'adx': 'ADX',
        'obv': 'OBV',
        'mfi': 'MFI',
        'cci': 'CCI',
        'williams': 'Williams %R',
        'vwap': 'VWAP',
        'portfolio': 'Portfolio State',
        'benchmark': 'Benchmark',
        'outperformance': 'Outperformance'
    }
    
    category_importance = {}
    
    for category_key, category_name in categories.items():
        mask = df['Feature'].str.contains(category_key, case=False, na=False)
        if mask.any():
            avg_importance = df[mask]['Combined_Importance'].mean()
            count = mask.sum()
            category_importance[category_name] = {
                'avg_importance': avg_importance,
                'count': count,
                'total_importance': df[mask]['Combined_Importance'].sum()
            }
    
    category_df = pd.DataFrame(category_importance).T
    category_df = category_df.sort_values('avg_importance', ascending=False)
    
    print(f"{'Category':<30} {'Avg Importance':<20} {'Count':<10} {'Total Importance':<20}")
    print(f"{'-'*80}")
    for category, row in category_df.iterrows():
        print(f"{category:<30} {row['avg_importance']:<20.6f} {int(row['count']):<10} {row['total_importance']:<20.6f}")
    
    return category_df

def analyze_by_timeframe(df):
    """Analyze importance by timeframe"""
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE BY TIMEFRAME")
    print(f"{'='*60}")
    
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    timeframe_importance = {}
    
    for tf in timeframes:
        mask = df['Feature'].str.contains(f'_{tf}_', case=False, na=False)
        if mask.any():
            avg_importance = df[mask]['Combined_Importance'].mean()
            count = mask.sum()
            timeframe_importance[tf] = {
                'avg_importance': avg_importance,
                'count': count
            }
    
    if timeframe_importance:
        tf_df = pd.DataFrame(timeframe_importance).T
        tf_df = tf_df.sort_values('avg_importance', ascending=False)
        
        print(f"{'Timeframe':<15} {'Avg Importance':<20} {'Count':<10}")
        print(f"{'-'*60}")
        for tf, row in tf_df.iterrows():
            print(f"{tf:<15} {row['avg_importance']:<20.6f} {int(row['count']):<10}")
        
        return tf_df
    else:
        print("No timeframe-specific features found")
        return None

def save_visualizations(df, category_df, timeframe_df):
    """Create and save visualization plots"""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    top_features = df.head(20)
    axes[0, 0].barh(range(len(top_features)), top_features['Combined_Importance'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels([f[:40] for f in top_features['Feature']], fontsize=8)
    axes[0, 0].set_xlabel('Combined Importance')
    axes[0, 0].set_title('Top 20 Most Important Features')
    axes[0, 0].invert_yaxis()
    
    if category_df is not None:
        axes[0, 1].bar(range(len(category_df)), category_df['avg_importance'])
        axes[0, 1].set_xticks(range(len(category_df)))
        axes[0, 1].set_xticklabels(category_df.index, rotation=45, ha='right', fontsize=8)
        axes[0, 1].set_ylabel('Average Importance')
        axes[0, 1].set_title('Feature Importance by Category')
    
    if timeframe_df is not None:
        axes[1, 0].bar(range(len(timeframe_df)), timeframe_df['avg_importance'])
        axes[1, 0].set_xticks(range(len(timeframe_df)))
        axes[1, 0].set_xticklabels(timeframe_df.index, rotation=0)
        axes[1, 0].set_ylabel('Average Importance')
        axes[1, 0].set_title('Feature Importance by Timeframe')
    
    axes[1, 1].scatter(df['Gradient_Importance'], df['Permutation_Importance'], alpha=0.3)
    axes[1, 1].set_xlabel('Gradient Importance')
    axes[1, 1].set_ylabel('Permutation Importance')
    axes[1, 1].set_title('Gradient vs Permutation Importance')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to 'feature_importance_analysis.png'")
    plt.close()

if __name__ == "__main__":
    print("="*100)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*100)
    
    config = {
        "num_transformer_blocks": 5,
        "embed_dim": 256,
        "num_heads": 8,
        "ff_dim": 512,
        "dropout_rate": 0.2,
        "dense_hidden": 256,
        "embedding_dim": 2048
    }
    
    print("\nInitializing environment...")
    env = PortfolioEnv(max_records=10000)
    observation = env.reset()
    
    obs_shape = observation.shape
    num_assets = len(env.asset_names)
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of assets: {num_assets}")
    
    print("\nBuilding model...")
    model = build_model(obs_shape, num_assets, config)
    
    model_path = 'best_model.weights.h5'
    if not os.path.exists(model_path):
        model_path = 'pretrained_model.weights.h5'
    
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        model.load_weights(model_path)
    else:
        print("Warning: No trained model found. Using untrained model for analysis.")
    
    print("\nExtracting feature names...")
    feature_names = get_feature_names(env)
    print(f"Total features: {len(feature_names)}")
    
    states, actions, rewards = collect_data_samples(env, model, num_episodes=2, max_steps=100)
    
    print(f"\nCollected {len(states)} samples")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")
    
    gradient_importance = analyze_gradient_importance(model, states, actions)
    
    permutation_imp = analyze_permutation_importance(states, actions, rewards, sample_size=5000)
    
    importance_df = create_importance_table(feature_names, gradient_importance, permutation_imp, top_n=50)
    
    category_df = analyze_by_category(importance_df)
    
    timeframe_df = analyze_by_timeframe(importance_df)
    
    print("\nSaving results to CSV...")
    importance_df.to_csv('feature_importance_full.csv', index=False)
    print("Saved 'feature_importance_full.csv'")
    
    if category_df is not None:
        category_df.to_csv('feature_importance_by_category.csv')
        print("Saved 'feature_importance_by_category.csv'")
    
    if timeframe_df is not None:
        timeframe_df.to_csv('feature_importance_by_timeframe.csv')
        print("Saved 'feature_importance_by_timeframe.csv'")
    
    save_visualizations(importance_df, category_df, timeframe_df)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("\nGenerated files:")
    print("  - feature_importance_full.csv")
    print("  - feature_importance_by_category.csv")
    print("  - feature_importance_by_timeframe.csv")
    print("  - feature_importance_analysis.png")
