import tensorflow as tf
import numpy as np
import pandas as pd
from portfolio_env import PortfolioEnv
from train import build_model
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def get_feature_names(env):
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
    
    crypto_symbols = list(env.df.keys())
    
    for symbol in crypto_symbols:
        for timeframe, count in aggregation_info.items():
            for col in all_available_cols:
                for step in range(count):
                    feature_names.append(f"{symbol}_{timeframe}_{col}_t{step}")
    
    for symbol in crypto_symbols:
        feature_names.append(f"portfolio_holdings_{symbol}")
        feature_names.append(f"avg_price_ratio_{symbol}")
        feature_names.append(f"pl_percent_{symbol}")
    
    feature_names.extend([
        'portfolio_cash_pct',
        'portfolio_value_ratio',
        'benchmark_ratio',
        'outperformance'
    ])
    
    return feature_names

def collect_data_samples(env, model, num_episodes=3, max_steps=50):
    states = []
    actions = []
    
    print(f"Collecting data from {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        
        for step in range(max_steps):
            state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
            action_probs = model(state_tensor, training=False)
            action = action_probs[0].numpy()
            
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            
            if done:
                break
            
            state = next_state
        
        print(f"  Episode {episode+1}/{num_episodes}: collected {len(states)} samples")
    
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)

def fast_gradient_importance(model, states, actions, batch_size=32):
    print("\nCalculating gradient-based importance...")
    
    num_samples = min(len(states), 150)
    indices = np.random.choice(len(states), num_samples, replace=False)
    states_sample = states[indices]
    actions_sample = actions[indices]
    
    all_gradients = []
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        state_batch = states_sample[i:end_idx]
        action_batch = actions_sample[i:end_idx]
        
        state_tensor = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(state_tensor)
            predictions = model(state_tensor, training=False)
            loss = tf.reduce_sum(predictions * action_tensor)
        
        grads = tape.gradient(loss, state_tensor)
        all_gradients.append(np.abs(grads.numpy()))
        
        if (i // batch_size) % 2 == 0:
            print(f"  Processed {end_idx}/{num_samples} samples")
    
    avg_gradients = np.mean(np.concatenate(all_gradients, axis=0), axis=0)
    
    return avg_gradients

def fast_variance_importance(states, sample_size=500):
    print("\nCalculating variance-based importance...")
    
    if len(states) > sample_size:
        indices = np.random.choice(len(states), sample_size, replace=False)
        states_sample = states[indices]
    else:
        states_sample = states
    
    variances = np.var(states_sample, axis=0)
    
    return variances

def create_importance_table(feature_names, gradient_importance, variance_importance, top_n=100):
    
    grad_normalized = (gradient_importance - gradient_importance.min()) / (gradient_importance.max() - gradient_importance.min() + 1e-8)
    var_normalized = (variance_importance - variance_importance.min()) / (variance_importance.max() - variance_importance.min() + 1e-8)
    
    combined_importance = 0.7 * grad_normalized + 0.3 * var_normalized
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Gradient_Importance': gradient_importance,
        'Variance_Importance': variance_importance,
        'Combined_Score': combined_importance
    })
    
    df = df.sort_values('Combined_Score', ascending=False)
    
    print(f"\n{'='*110}")
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print(f"{'='*110}")
    print(f"{'Rank':<6} {'Feature':<60} {'Gradient':<15} {'Variance':<15} {'Combined':<15}")
    print(f"{'-'*110}")
    
    for idx, (i, row) in enumerate(df.head(top_n).iterrows(), 1):
        print(f"{idx:<6} {row['Feature']:<60} {row['Gradient_Importance']:<15.6f} {row['Variance_Importance']:<15.6f} {row['Combined_Score']:<15.6f}")
    
    return df

def analyze_by_category(df):
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
        'ma_': 'MA',
        'bb': 'Bollinger Bands',
        'atr': 'ATR',
        'vwap': 'VWAP',
        'portfolio_holdings': 'Portfolio Holdings',
        'avg_price_ratio': 'Avg Price Ratio',
        'pl_percent': 'P/L Percent',
        'benchmark': 'Benchmark',
        'outperformance': 'Outperformance'
    }
    
    category_importance = {}
    
    for category_key, category_name in categories.items():
        mask = df['Feature'].str.contains(category_key, case=False, na=False)
        if mask.any():
            avg_importance = df[mask]['Combined_Score'].mean()
            count = mask.sum()
            top_feature = df[mask].iloc[0]['Feature'] if len(df[mask]) > 0 else 'N/A'
            category_importance[category_name] = {
                'avg_importance': avg_importance,
                'count': count,
                'top_feature': top_feature
            }
    
    category_df = pd.DataFrame(category_importance).T
    category_df = category_df.sort_values('avg_importance', ascending=False)
    
    print(f"{'Category':<30} {'Avg Score':<15} {'Count':<10} {'Top Feature':<40}")
    print(f"{'-'*80}")
    for category, row in category_df.iterrows():
        print(f"{category:<30} {row['avg_importance']:<15.6f} {int(row['count']):<10} {str(row['top_feature'])[:40]}")
    
    return category_df

def analyze_by_timeframe(df):
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE BY TIMEFRAME")
    print(f"{'='*60}")
    
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    timeframe_importance = {}
    
    for tf in timeframes:
        mask = df['Feature'].str.contains(f'_{tf}_', case=False, na=False)
        if mask.any():
            avg_importance = df[mask]['Combined_Score'].mean()
            count = mask.sum()
            timeframe_importance[tf] = {
                'avg_importance': avg_importance,
                'count': count
            }
    
    if timeframe_importance:
        tf_df = pd.DataFrame(timeframe_importance).T
        tf_df = tf_df.sort_values('avg_importance', ascending=False)
        
        print(f"{'Timeframe':<15} {'Avg Score':<20} {'Count':<10}")
        print(f"{'-'*60}")
        for tf, row in tf_df.iterrows():
            print(f"{tf:<15} {row['avg_importance']:<20.6f} {int(row['count']):<10}")
        
        return tf_df
    else:
        print("No timeframe features found")
        return None

def save_quick_visualizations(df, category_df, timeframe_df):
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    top_features = df.head(30)
    axes[0, 0].barh(range(len(top_features)), top_features['Combined_Score'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels([f[:35] for f in top_features['Feature']], fontsize=7)
    axes[0, 0].set_xlabel('Combined Importance Score')
    axes[0, 0].set_title('Top 30 Most Important Features')
    axes[0, 0].invert_yaxis()
    
    if category_df is not None:
        axes[0, 1].bar(range(len(category_df)), category_df['avg_importance'])
        axes[0, 1].set_xticks(range(len(category_df)))
        axes[0, 1].set_xticklabels(category_df.index, rotation=45, ha='right', fontsize=7)
        axes[0, 1].set_ylabel('Average Importance')
        axes[0, 1].set_title('Feature Importance by Category')
    
    if timeframe_df is not None:
        axes[1, 0].bar(range(len(timeframe_df)), timeframe_df['avg_importance'], color='green')
        axes[1, 0].set_xticks(range(len(timeframe_df)))
        axes[1, 0].set_xticklabels(timeframe_df.index, rotation=0)
        axes[1, 0].set_ylabel('Average Importance')
        axes[1, 0].set_title('Feature Importance by Timeframe')
    
    axes[1, 1].hist(df['Combined_Score'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Combined Importance Score')
    axes[1, 1].set_ylabel('Number of Features')
    axes[1, 1].set_title('Distribution of Feature Importance Scores')
    axes[1, 1].axvline(df['Combined_Score'].median(), color='red', linestyle='--', label='Median')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('feature_importance_fast.png', dpi=200, bbox_inches='tight')
    print("Saved visualization to 'feature_importance_fast.png'")
    plt.close()

if __name__ == "__main__":
    print("="*100)
    print("FAST FEATURE IMPORTANCE ANALYSIS (Optimized for Large Feature Sets)")
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
    print(f"Total features: {obs_shape[0]}")
    
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
    print(f"Expected features from naming: {len(feature_names)}")
    
    states, actions = collect_data_samples(env, model, num_episodes=3, max_steps=50)
    
    print(f"\nCollected {len(states)} samples")
    print(f"Actual state shape: {states.shape}")
    
    actual_feature_count = states.shape[1]
    if len(feature_names) != actual_feature_count:
        print(f"\nAdjusting feature names: {len(feature_names)} -> {actual_feature_count}")
        
        if len(feature_names) < actual_feature_count:
            for i in range(len(feature_names), actual_feature_count):
                feature_names.append(f"feature_{i}")
        else:
            feature_names = feature_names[:actual_feature_count]
        
        print(f"Adjusted to {len(feature_names)} feature names")
    
    gradient_importance = fast_gradient_importance(model, states, actions, batch_size=32)
    variance_importance = fast_variance_importance(states, sample_size=500)
    assert len(feature_names) == len(gradient_importance) == len(variance_importance), \
        f"Length mismatch: features={len(feature_names)}, gradient={len(gradient_importance)}, variance={len(variance_importance)}"
    
    importance_df = create_importance_table(feature_names, gradient_importance, variance_importance, top_n=100)
    category_df = analyze_by_category(importance_df)
    timeframe_df = analyze_by_timeframe(importance_df)
    
    print("\nSaving results to CSV...")
    importance_df.to_csv('feature_importance_fast.csv', index=False)
    print("Saved 'feature_importance_fast.csv'")
    
    importance_df.head(500).to_csv('feature_importance_top500.csv', index=False)
    print("Saved 'feature_importance_top500.csv'")
    
    if category_df is not None:
        category_df.to_csv('feature_importance_by_category.csv')
        print("Saved 'feature_importance_by_category.csv'")
    
    if timeframe_df is not None:
        timeframe_df.to_csv('feature_importance_by_timeframe.csv')
        print("Saved 'feature_importance_by_timeframe.csv'")
    
    save_quick_visualizations(importance_df, category_df, timeframe_df)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)