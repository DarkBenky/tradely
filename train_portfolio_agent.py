import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from portfolio_env import MultiCurrencyPortfolioEnv
import torch
import matplotlib.pyplot as plt
from datetime import datetime


class PortfolioMetricsCallback(BaseCallback):
    """Custom callback to log portfolio-specific metrics during training"""
    
    def __init__(self, eval_env, eval_freq=1000, verbose=1):
        super(PortfolioMetricsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_returns = []
        self.episode_sharpe_ratios = []
        self.episode_drawdowns = []
        self.best_sharpe = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            obs = self.eval_env.reset()[0]
            done = False
            step = 0
            
            while not done and step < 500:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                step += 1
            
            metrics = self.eval_env.get_final_metrics()
            
            self.episode_returns.append(metrics['total_return'])
            self.episode_sharpe_ratios.append(metrics['sharpe_ratio'])
            self.episode_drawdowns.append(metrics['max_drawdown'])
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Evaluation at step {self.n_calls}")
                print(f"  Return: {metrics['total_return']*100:.2f}%")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
                print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
                print(f"{'='*60}\n")
            
            # Save best model based on Sharpe ratio
            if metrics['sharpe_ratio'] > self.best_sharpe:
                self.best_sharpe = metrics['sharpe_ratio']
                self.model.save(f"{self.model.logger.dir}/best_model_sharpe_{metrics['sharpe_ratio']:.3f}")
                if self.verbose > 0:
                    print(f"New best Sharpe ratio! Model saved.")
        
        return True


def load_data(symbols, data_dir='data'):
    """Load data for all symbols"""
    dfs = {}
    
    for symbol in symbols:
        filepath = os.path.join(data_dir, f'{symbol}_combined_data.csv')
        if os.path.exists(filepath):
            print(f"Loading {symbol}...")
            df = pd.read_csv(filepath)
            
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
            
            # Keep only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_cols]
            df = df.dropna()
            
            if len(df) > 500:
                dfs[symbol] = df
                print(f"  Loaded {len(df)} rows, {len(df.columns)} features")
        else:
            print(f"  {symbol} data not found")
    
    return dfs


def create_env(dfs, symbols, starting_balance=10000, is_eval=False):
    """Create portfolio environment"""
    env = MultiCurrencyPortfolioEnv(
        dfs=dfs,
        symbols=symbols,
        starting_balance=starting_balance,
        fee_rate=0.001,
        lookback_window=50,
        rebalance_frequency=12,  # Every hour (12 * 5min)
        risk_free_rate=0.02,
        max_drawdown_penalty=2.0,
        normalize_obs=True
    )
    
    if not is_eval:
        env = Monitor(env)
    
    return env


def train_portfolio_agent(
    symbols,
    algorithm='PPO',
    total_timesteps=100000,
    learning_rate=0.0003,
    batch_size=64,
    n_steps=2048,
    starting_balance=10000,
    save_dir='models/portfolio',
    log_dir='logs/portfolio'
):
    """
    Train a multi-currency portfolio rebalancing agent
    
    Args:
        symbols: List of cryptocurrency symbols to trade
        algorithm: 'PPO', 'A2C', or 'SAC'
        total_timesteps: Total training steps
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        n_steps: Steps per environment per update (for on-policy algorithms)
        starting_balance: Initial portfolio balance
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
    """
    
    print("=" * 60)
    print("Multi-Currency Portfolio RL Training")
    print("=" * 60)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    dfs = load_data(symbols)
    
    if len(dfs) < 2:
        raise ValueError("Need at least 2 symbols with data. Run getData.py first.")
    
    available_symbols = list(dfs.keys())
    print(f"\nTraining with {len(available_symbols)} symbols: {available_symbols}")
    
    # Split data into train and validation
    print("\nSplitting data into train/validation sets...")
    train_dfs = {}
    val_dfs = {}
    
    for symbol in available_symbols:
        df = dfs[symbol]
        split_idx = int(len(df) * 0.8)
        train_dfs[symbol] = df.iloc[:split_idx].copy()
        val_dfs[symbol] = df.iloc[split_idx:].copy()
        print(f"  {symbol}: {len(train_dfs[symbol])} train, {len(val_dfs[symbol])} val")
    
    # Create environments
    print("\nCreating environments...")
    train_env = create_env(train_dfs, available_symbols, starting_balance)
    eval_env = create_env(val_dfs, available_symbols, starting_balance, is_eval=True)
    
    print(f"Observation space: {train_env.observation_space.shape}")
    print(f"Action space: {train_env.action_space.shape}")
    
    # Create model
    print(f"\nInitializing {algorithm} agent...")
    
    # Policy kwargs for better performance
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn=torch.nn.ReLU,
    )
    
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir
        )
    elif algorithm == 'A2C':
        model = A2C(
            'MlpPolicy',
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            train_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=0.99,
            tau=0.005,
            ent_coef='auto',
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix=f'{algorithm}_portfolio'
    )
    
    metrics_callback = PortfolioMetricsCallback(
        eval_env=eval_env,
        eval_freq=5000,
        verbose=1
    )
    
    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("Monitor progress with: tensorboard --logdir=" + log_dir)
    print()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, metrics_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_model_path = os.path.join(save_dir, f'{algorithm}_portfolio_final')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Evaluate final model
    print("\n" + "=" * 60)
    print("Evaluating Final Model")
    print("=" * 60)
    
    obs = eval_env.reset()[0]
    done = False
    step = 0
    episode_reward = 0
    
    portfolio_values = []
    actions_history = []
    
    while not done and step < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        portfolio_values.append(info['portfolio_value'])
        actions_history.append(info['weights'])
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}: Portfolio Value = ${info['portfolio_value']:.2f}, Return = {info['total_return']*100:.2f}%")
    
    final_metrics = eval_env.get_final_metrics()
    
    print("\n" + "=" * 60)
    print("Final Evaluation Results")
    print("=" * 60)
    print(f"Total Steps: {step}")
    print(f"Episode Reward: {episode_reward:.2f}")
    print(f"Final Portfolio Value: ${final_metrics['final_portfolio_value']:.2f}")
    print(f"Total Return: {final_metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {final_metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {final_metrics['win_rate']*100:.1f}%")
    print(f"Profit Factor: {final_metrics['profit_factor']:.2f}")
    print(f"Trades Executed: {final_metrics['trades_executed']}")
    print(f"Total Fees: ${final_metrics['total_fees_paid']:.2f}")
    
    # Plot results
    plot_results(portfolio_values, actions_history, available_symbols, save_dir)
    
    return model, final_metrics


def plot_results(portfolio_values, actions_history, symbols, save_dir):
    """Plot training results"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Portfolio value over time
    axes[0].plot(portfolio_values)
    axes[0].set_title('Portfolio Value Over Time')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Portfolio weights over time
    if len(actions_history) > 0:
        weights_array = np.array([[w[s] for s in symbols] for w in actions_history])
        for i, symbol in enumerate(symbols):
            axes[1].plot(weights_array[:, i], label=symbol, alpha=0.7)
        axes[1].set_title('Portfolio Weights Over Time')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Weight')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'evaluation_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
        'XRPUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'AVAXUSDT'
    ]
    
    ALGORITHM = 'PPO'  # Options: 'PPO', 'A2C', 'SAC'
    TOTAL_TIMESTEPS = 100000  # Adjust based on your needs
    STARTING_BALANCE = 10000
    
    # Train agent
    model, metrics = train_portfolio_agent(
        symbols=SYMBOLS,
        algorithm=ALGORITHM,
        total_timesteps=TOTAL_TIMESTEPS,
        learning_rate=0.0003,
        batch_size=64,
        n_steps=2048,
        starting_balance=STARTING_BALANCE,
        save_dir=f'models/portfolio_{ALGORITHM.lower()}',
        log_dir=f'logs/portfolio_{ALGORITHM.lower()}'
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
