import pandas as pd
import numpy as np
from portfolio_env import MultiCurrencyPortfolioEnv
import os

def test_portfolio_env():
    """Test the multi-currency portfolio environment"""
    
    print("Testing Multi-Currency Portfolio Environment")
    print("=" * 60)
    
    # Load data for multiple symbols
    data_dir = 'data'
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT', 'XRPUSDT', 'DOGEUSDT']
    
    # Check which files exist
    available_symbols = []
    dfs = {}
    
    for symbol in symbols:
        filepath = os.path.join(data_dir, f'{symbol}_combined_data.csv')
        if os.path.exists(filepath):
            print(f"Loading {symbol}...")
            df = pd.read_csv(filepath)
            
            # Parse timestamp/datetime column and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
            
            # Keep only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_cols]
            # Drop rows with missing raw OHLCV data, fill other missing values
            raw_cols = ['open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            df = df.dropna(subset=[col for col in raw_cols if col in df.columns])
            df = df.fillna(0)
            
            if len(df) > 100:  # Ensure enough data
                dfs[symbol] = df
                available_symbols.append(symbol)
                print(f"  Loaded {len(df)} rows, {len(df.columns)} features")
        else:
            print(f"  {symbol} data not found, skipping...")
    
    if len(available_symbols) < 2:
        print("\nError: Need at least 2 symbols with data to test portfolio environment")
        print("Please run getData.py first to fetch data")
        return
    
    print(f"\nUsing {len(available_symbols)} symbols: {available_symbols}")
    
    # Create environment
    print("\nCreating environment...")
    env = MultiCurrencyPortfolioEnv(
        dfs=dfs,
        symbols=available_symbols,
        starting_balance=10000.0,
        fee_rate=0.001,
        lookback_window=50,
        rebalance_frequency=12,  # Rebalance every 12 steps (1 hour)
        risk_free_rate=0.02,
        max_drawdown_penalty=2.0,
        normalize_obs=True
    )
    
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Max steps: {env.max_steps}")
    
    # Test random episode
    print("\n" + "=" * 60)
    print("Testing Random Agent")
    print("=" * 60)
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial portfolio value: ${info['portfolio_value']:.2f}")
    
    episode_reward = 0
    done = False
    step = 0
    
    while not done and step < 100:  # Limit to 100 steps for testing
        # Random action: random portfolio weights that sum to 1.0
        action = env.sample_action()
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        step += 1
        
        # Print every 20 steps
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Action (weights): {action}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Portfolio Value: ${info['portfolio_value']:.2f}")
            print(f"  Total Return: {info['total_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {info['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {info['max_drawdown']*100:.2f}%")
            print(f"  Current Weights: {[f'{v:.2f}' for v in info['weights'].values()]}")
    
    # Get final metrics
    print("\n" + "=" * 60)
    print("Episode Complete")
    print("=" * 60)
    
    final_metrics = env.get_final_metrics()
    print(f"\nFinal Performance:")
    print(f"  Total Steps: {step}")
    print(f"  Episode Reward: {episode_reward:.2f}")
    print(f"  Final Portfolio Value: ${final_metrics['final_portfolio_value']:.2f}")
    print(f"  Total Return: {final_metrics['total_return']*100:.2f}%")
    print(f"  Max Drawdown: {final_metrics['max_drawdown']*100:.2f}%")
    print(f"  Sharpe Ratio: {final_metrics['sharpe_ratio']:.3f}")
    print(f"  Win Rate: {final_metrics['win_rate']*100:.1f}%")
    print(f"  Avg Win: {final_metrics['avg_win']*100:.3f}%")
    print(f"  Avg Loss: {final_metrics['avg_loss']*100:.3f}%")
    print(f"  Profit Factor: {final_metrics['profit_factor']:.2f}")
    print(f"  Trades Executed: {final_metrics['trades_executed']}")
    print(f"  Total Fees: ${final_metrics['total_fees_paid']:.2f} ({final_metrics['fee_percentage']*100:.2f}%)")
    
    # Test equal weight strategy
    print("\n" + "=" * 60)
    print("Testing Equal Weight Strategy (Baseline)")
    print("=" * 60)
    
    obs, info = env.reset()
    episode_reward = 0
    done = False
    step = 0
    
    # Equal weights for all assets
    equal_weight_action = np.ones(len(available_symbols)) / len(available_symbols)
    
    while not done and step < 100:
        obs, reward, terminated, truncated, info = env.step(equal_weight_action)
        episode_reward += reward
        done = terminated or truncated
        step += 1
    
    final_metrics = env.get_final_metrics()
    print(f"\nEqual Weight Strategy Results:")
    print(f"  Total Steps: {step}")
    print(f"  Episode Reward: {episode_reward:.2f}")
    print(f"  Final Portfolio Value: ${final_metrics['final_portfolio_value']:.2f}")
    print(f"  Total Return: {final_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {final_metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {final_metrics['max_drawdown']*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("Environment test complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_portfolio_env()
