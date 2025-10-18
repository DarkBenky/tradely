import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from portfolio_env import AggressiveMultiCurrencyPortfolioEnv
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import wandb
import json


class AggressivePortfolioMetricsCallback(BaseCallback):
    
    def __init__(self, eval_env, eval_freq=1000, verbose=1, use_wandb=True, symbols=None):
        super(AggressivePortfolioMetricsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_returns = []
        self.episode_sharpe_ratios = []
        self.episode_drawdowns = []
        self.best_outperformance = -np.inf
        self.use_wandb = use_wandb
        self.symbols = symbols if symbols else []
        
    def _on_step(self) -> bool:
        if self.use_wandb and hasattr(self.locals.get('infos', [{}])[0], '__iter__'):
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0 and 'portfolio_value' in infos[0]:
                info = infos[0]
                log_data = {
                    'train/step': self.n_calls,
                    'train/portfolio_value': info.get('portfolio_value', 0),
                    'train/cash_balance': info.get('cash_balance', 0),
                    'train/total_return': info.get('total_return', 0),
                    'train/reward': self.locals.get('rewards', [0])[0],
                    'train/max_drawdown': info.get('max_drawdown', 0),
                    'train/sharpe_ratio': info.get('sharpe_ratio', 0),
                    'train/benchmark_value': info.get('benchmark_value', 0),
                    'train/benchmark_outperformance': info.get('benchmark_outperformance', 0),
                    'train/future_profit_potential': info.get('future_profit_potential', 0),
                    'train/portfolio_volatility': info.get('portfolio_volatility', 0),
                    'train/diversification_ratio': info.get('diversification_ratio', 0),
                    'train/value_at_risk': info.get('value_at_risk', 0),
                }
                
                cash_ratio = info.get('cash_balance', 0) / (info.get('portfolio_value', 1) + 1e-8)
                log_data['train/cash_ratio'] = cash_ratio
                
                weights = info.get('weights', {})
                if weights and self.symbols:
                    for symbol in self.symbols:
                        weight = weights.get(symbol, 0)
                        log_data[f'train/weight_{symbol}'] = weight
                
                # Log momentum and trend signals
                momentum_signals = info.get('momentum_signals', {})
                trend_signals = info.get('trend_signals', {})
                if momentum_signals and self.symbols:
                    for symbol in self.symbols:
                        momentum = momentum_signals.get(symbol, 0)
                        trend = trend_signals.get(symbol, 0)
                        log_data[f'train/momentum_{symbol}'] = momentum
                        log_data[f'train/trend_{symbol}'] = trend
                
                wandb.log(log_data)
        
        if self.n_calls % self.eval_freq == 0:
            obs = self.eval_env.reset()[0]
            done = False
            step = 0
            
            episode_portfolio_values = []
            episode_weights = []
            episode_cash_balances = []
            episode_benchmark_values = []
            episode_risk_metrics = []
            
            while not done and step < 500:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_portfolio_values.append(info.get('portfolio_value', 0))
                episode_weights.append(info.get('weights', {}))
                episode_cash_balances.append(info.get('cash_balance', 0))
                episode_benchmark_values.append(info.get('benchmark_value', 0))
                episode_risk_metrics.append({
                    'volatility': info.get('portfolio_volatility', 0),
                    'diversification': info.get('diversification_ratio', 0),
                    'var': info.get('value_at_risk', 0),
                    'outperformance': info.get('benchmark_outperformance', 0)
                })
                step += 1
            
            metrics = self.eval_env.get_final_metrics()
            
            self.episode_returns.append(metrics['total_return'])
            self.episode_sharpe_ratios.append(metrics['sharpe_ratio'])
            self.episode_drawdowns.append(metrics['max_drawdown'])
            
            if self.use_wandb:
                log_dict = {
                    'eval/step': self.n_calls,
                    'eval/total_return': metrics['total_return'],
                    'eval/sharpe_ratio': metrics['sharpe_ratio'],
                    'eval/max_drawdown': metrics['max_drawdown'],
                    'eval/win_rate': metrics['win_rate'],
                    'eval/profit_factor': metrics.get('profit_factor', 0),
                    'eval/trades_executed': metrics.get('trades_executed', 0),
                    'eval/total_fees': metrics.get('total_fees_paid', 0),
                    'eval/final_portfolio_value': metrics['final_portfolio_value'],
                    'eval/best_outperformance_so_far': self.best_outperformance,
                    'eval/benchmark_return': metrics.get('benchmark_return', 0),
                    'eval/outperformance': metrics.get('outperformance', 0),
                    'eval/sortino_ratio': metrics.get('sortino_ratio', 0),
                    'eval/calmar_ratio': metrics.get('calmar_ratio', 0),
                    'eval/portfolio_volatility': metrics.get('portfolio_volatility', 0),
                    'eval/diversification_ratio': metrics.get('diversification_ratio', 0),
                    'eval/value_at_risk': metrics.get('value_at_risk', 0),
                    'eval/avg_win': metrics.get('avg_win', 0),
                    'eval/avg_loss': metrics.get('avg_loss', 0),
                }
                
                if episode_cash_balances:
                    final_cash = episode_cash_balances[-1]
                    final_portfolio_value = episode_portfolio_values[-1] if episode_portfolio_values else 1
                    cash_ratio = final_cash / (final_portfolio_value + 1e-8)
                    log_dict['eval/cash_balance'] = final_cash
                    log_dict['eval/cash_ratio'] = cash_ratio
                
                if episode_weights and self.symbols:
                    final_weights = episode_weights[-1]
                    for symbol in self.symbols:
                        weight = final_weights.get(symbol, 0)
                        log_dict[f'eval/weight_{symbol}'] = weight
                
                # Portfolio value vs benchmark chart
                if len(episode_portfolio_values) > 0 and len(episode_benchmark_values) > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(episode_portfolio_values, label='Portfolio', linewidth=2)
                    plt.plot(episode_benchmark_values, label='Benchmark', linewidth=2, linestyle='--')
                    plt.title('Portfolio vs Benchmark Value')
                    plt.xlabel('Step')
                    plt.ylabel('Value ($)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    log_dict['eval/portfolio_vs_benchmark'] = wandb.Image(plt)
                    plt.close()
                
                # Risk metrics chart
                if len(episode_risk_metrics) > 0:
                    plt.figure(figsize=(12, 10))
                    
                    plt.subplot(4, 1, 1)
                    volatilities = [rm['volatility'] for rm in episode_risk_metrics]
                    plt.plot(volatilities, color='red')
                    plt.title('Portfolio Volatility Over Time')
                    plt.ylabel('Volatility')
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(4, 1, 2)
                    diversifications = [rm['diversification'] for rm in episode_risk_metrics]
                    plt.plot(diversifications, color='green')
                    plt.title('Diversification Ratio Over Time')
                    plt.ylabel('Diversification')
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(4, 1, 3)
                    vars = [rm['var'] for rm in episode_risk_metrics]
                    plt.plot(vars, color='orange')
                    plt.title('Value at Risk (95%) Over Time')
                    plt.ylabel('VaR')
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(4, 1, 4)
                    outperformance = [rm['outperformance'] for rm in episode_risk_metrics]
                    plt.plot(outperformance, color='purple')
                    plt.title('Benchmark Outperformance Over Time')
                    plt.ylabel('Outperformance')
                    plt.xlabel('Step')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    log_dict['eval/risk_metrics'] = wandb.Image(plt)
                    plt.close()
                
                # Asset allocation chart
                if episode_weights and self.symbols and len(episode_weights) > 0:
                    plt.figure(figsize=(12, 6))
                    weights_array = np.array([[w.get(s, 0) for s in self.symbols] for w in episode_weights])
                    for i, symbol in enumerate(self.symbols):
                        plt.plot(weights_array[:, i], label=symbol, alpha=0.7)
                    plt.title('Asset Allocation Over Time')
                    plt.xlabel('Step')
                    plt.ylabel('Weight')
                    plt.legend(loc='upper right')
                    plt.grid(True, alpha=0.3)
                    plt.ylim(0.0, 1.0)  # No negative weights
                    log_dict['eval/allocation_chart'] = wandb.Image(plt)
                    plt.close()
                
                wandb.log(log_dict)
            
            if self.verbose > 0:
                print(f"\n{'='*80}")
                print(f"Evaluation at step {self.n_calls}")
                print(f"  Return: {metrics['total_return']*100:.2f}%")
                print(f"  Benchmark Return: {metrics.get('benchmark_return', 0)*100:.2f}%")
                print(f"  Outperformance: {metrics.get('outperformance', 0)*100:.2f}%")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
                print(f"  Volatility: {metrics.get('portfolio_volatility', 0)*100:.2f}%")
                print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
                print(f"{'='*80}\n")
            
            # Focus on outperformance as the main metric for aggressive strategy
            current_outperformance = metrics.get('outperformance', 0)
            if current_outperformance > self.best_outperformance and current_outperformance > 0:
                self.best_outperformance = current_outperformance
                model_path = f"{self.model.logger.dir}/best_model_outperformance_{current_outperformance:.3f}"
                self.model.save(model_path)
                
                if self.use_wandb:
                    wandb.log({'eval/new_best_outperformance': current_outperformance})
                    
                    import os
                    zip_file = model_path + ".zip"
                    if os.path.exists(zip_file):
                        artifact = wandb.Artifact(
                            name=f"model-best-outperformance",
                            type="model",
                            description=f"Best model with outperformance {current_outperformance:.3f}"
                        )
                        artifact.add_file(zip_file)
                        wandb.log_artifact(artifact)
                
                if self.verbose > 0:
                    print(f"New best outperformance! Model saved.")
        
        return True


def load_data(symbols, data_dir='data'):
    dfs = {}
    
    for symbol in symbols:
        filepath = os.path.join(data_dir, f'{symbol}_combined_data.csv')
        if os.path.exists(filepath):
            print(f"Loading {symbol}...")
            df = pd.read_csv(filepath)
            
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_cols]
            df = df.dropna()
            
            if len(df) > 500:
                dfs[symbol] = df
                print(f"  Loaded {len(df)} rows, {len(df.columns)} features")
        else:
            print(f"  {symbol} data not found")
    
    return dfs


def create_aggressive_env(dfs, symbols, starting_balance=10000, is_eval=False,
               fee_rate=0.001, lookback_window=50, rebalance_frequency=6,
               risk_free_rate=0.02, max_drawdown_penalty=1.0, normalize_obs=True,
               reward_horizon=1, benchmark_weight=3.0, future_profit_weight=2.0,
               future_window=8, risk_adjustment=False, dynamic_penalties=False,
               use_attention_weights=True, volatility_scaling=False,
               momentum_lookback=10, concentration_limit=0.7, 
               trend_following=True):  # Removed leverage_limit
    env = AggressiveMultiCurrencyPortfolioEnv(
        dfs=dfs,
        symbols=symbols,
        starting_balance=starting_balance,
        fee_rate=fee_rate,
        lookback_window=lookback_window,
        rebalance_frequency=rebalance_frequency,
        risk_free_rate=risk_free_rate,
        max_drawdown_penalty=max_drawdown_penalty,
        normalize_obs=normalize_obs,
        reward_horizon=reward_horizon,
        benchmark_weight=benchmark_weight,
        future_profit_weight=future_profit_weight,
        future_window=future_window,
        risk_adjustment=risk_adjustment,
        dynamic_penalties=dynamic_penalties,
        use_attention_weights=use_attention_weights,
        volatility_scaling=volatility_scaling,
        momentum_lookback=momentum_lookback,
        concentration_limit=concentration_limit,
        trend_following=trend_following
        # Removed leverage_limit
    )
    
    if not is_eval:
        env = Monitor(env)
    
    return env


def train_aggressive_portfolio_agent(
    symbols,
    algorithm='PPO',
    total_timesteps=200000,
    learning_rate=0.0003,
    batch_size=64,
    n_steps=2048,
    starting_balance=10000,
    save_dir='models/aggressive_portfolio',
    log_dir='logs/aggressive_portfolio',
    use_wandb=True,
    wandb_project='crypto-portfolio-rl-aggressive',
    wandb_run_name=None,
    fee_rate=0.001,
    lookback_window=50,
    rebalance_frequency=6,
    risk_free_rate=0.02,
    max_drawdown_penalty=1.0,
    normalize_obs=True,
    reward_horizon=1,
    benchmark_weight=3.0,
    future_profit_weight=2.0,
    future_window=8,
    risk_adjustment=False,
    dynamic_penalties=False,
    use_attention_weights=True,
    volatility_scaling=False,
    network_size='large',
    momentum_lookback=10,
    concentration_limit=0.7,
    trend_following=True
    # Removed leverage_limit
):
    
    print("=" * 80)
    print("AGGRESSIVE Multi-Currency Portfolio RL Training")
    print("=" * 80)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("\nLoading data...")
    dfs = load_data(symbols)
    
    if len(dfs) < 2:
        raise ValueError("Need at least 2 symbols with data. Run getData.py first.")
    
    available_symbols = list(dfs.keys())
    print(f"\nTraining with {len(available_symbols)} symbols: {available_symbols}")
    
    print("\nSplitting data into train/validation sets...")
    train_dfs = {}
    val_dfs = {}
    
    for symbol in available_symbols:
        df = dfs[symbol]
        split_idx = int(len(df) * 0.8)
        train_dfs[symbol] = df.iloc[:split_idx].copy()
        val_dfs[symbol] = df.iloc[split_idx:].copy()
        print(f"  {symbol}: {len(train_dfs[symbol])} train, {len(val_dfs[symbol])} val")
    
    if use_wandb:
        if wandb_run_name is None:
            wandb_run_name = f"{algorithm}_aggressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                'algorithm': algorithm,
                'total_timesteps': total_timesteps,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'n_steps': n_steps,
                'starting_balance': starting_balance,
                'num_symbols': len(available_symbols),
                'symbols': available_symbols,
                'train_samples': sum(len(df) for df in train_dfs.values()),
                'val_samples': sum(len(df) for df in val_dfs.values()),
                'fee_rate': fee_rate,
                'lookback_window': lookback_window,
                'rebalance_frequency': rebalance_frequency,
                'risk_free_rate': risk_free_rate,
                'max_drawdown_penalty': max_drawdown_penalty,
                'normalize_obs': normalize_obs,
                'reward_horizon': reward_horizon,
                'benchmark_weight': benchmark_weight,
                'future_profit_weight': future_profit_weight,
                'future_window': future_window,
                'risk_adjustment': risk_adjustment,
                'dynamic_penalties': dynamic_penalties,
                'use_attention_weights': use_attention_weights,
                'volatility_scaling': volatility_scaling,
                'network_size': network_size,
                'momentum_lookback': momentum_lookback,
                'concentration_limit': concentration_limit,
                'trend_following': trend_following,
            },
            tags=[algorithm, 'portfolio', 'crypto', 'aggressive', f'net_{network_size}'],
            notes=f"Training {algorithm} agent on {len(available_symbols)} cryptocurrencies with AGGRESSIVE strategy"
        )
        print(f"\nWandB initialized: {wandb.run.url}")
    
    print("\nCreating aggressive environments...")
    train_env = create_aggressive_env(
        train_dfs, available_symbols, starting_balance,
        fee_rate=fee_rate, lookback_window=lookback_window,
        rebalance_frequency=rebalance_frequency, risk_free_rate=risk_free_rate,
        max_drawdown_penalty=max_drawdown_penalty, normalize_obs=normalize_obs,
        reward_horizon=reward_horizon, benchmark_weight=benchmark_weight,
        future_profit_weight=future_profit_weight, future_window=future_window,
        risk_adjustment=risk_adjustment, dynamic_penalties=dynamic_penalties,
        use_attention_weights=use_attention_weights, volatility_scaling=volatility_scaling,
        momentum_lookback=momentum_lookback, concentration_limit=concentration_limit,
        trend_following=trend_following
    )
    eval_env = create_aggressive_env(
        val_dfs, available_symbols, starting_balance, is_eval=True,
        fee_rate=fee_rate, lookback_window=lookback_window,
        rebalance_frequency=rebalance_frequency, risk_free_rate=risk_free_rate,
        max_drawdown_penalty=max_drawdown_penalty, normalize_obs=normalize_obs,
        reward_horizon=reward_horizon, benchmark_weight=benchmark_weight,
        future_profit_weight=future_profit_weight, future_window=future_window,
        risk_adjustment=risk_adjustment, dynamic_penalties=dynamic_penalties,
        use_attention_weights=use_attention_weights, volatility_scaling=volatility_scaling,
        momentum_lookback=momentum_lookback, concentration_limit=concentration_limit,
        trend_following=trend_following
    )
    
    print(f"Observation space: {train_env.observation_space.shape}")
    print(f"Action space: {train_env.action_space.shape}")
    
    # Test observation creation to verify shape
    test_obs, _ = train_env.reset()
    print(f"Test observation shape: {test_obs.shape}")
    
    if use_wandb:
        wandb.config.update({
            'observation_dim': train_env.observation_space.shape[0],
            'action_dim': train_env.action_space.shape[0],
            'actual_observation_dim': test_obs.shape[0],
        })
    
    print(f"\nInitializing {algorithm} agent with {network_size} network...")
    
    network_configs = {
        'small': dict(pi=[128, 64], vf=[128, 64]),
        'medium': dict(pi=[256, 128], vf=[256, 128]),
        'large': dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        'xlarge': dict(pi=[1024, 512, 256], vf=[1024, 512, 256]),
        'xxlarge': dict(pi=[2048, 1024, 512], vf=[2048, 1024, 512])
    }
    
    if network_size not in network_configs:
        print(f"Unknown network size '{network_size}', defaulting to 'large'")
        network_size = 'large'
    
    net_arch = network_configs[network_size]
    print(f"Network architecture: Policy={net_arch['pi']}, Value={net_arch['vf']}")
    
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=torch.nn.ReLU,
    )
    
    if use_wandb:
        wandb.config.update({
            'network_pi_layers': net_arch['pi'],
            'network_vf_layers': net_arch['vf'],
        })
    
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
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix=f'{algorithm}_aggressive_portfolio'
    )
    
    metrics_callback = AggressivePortfolioMetricsCallback(
        eval_env=eval_env,
        eval_freq=5000,
        verbose=1,
        use_wandb=use_wandb,
        symbols=available_symbols
    )
    
    print(f"\nStarting AGGRESSIVE training for {total_timesteps} timesteps...")
    print("Monitor progress with: tensorboard --logdir=" + log_dir)
    if use_wandb:
        print(f"Monitor progress with W&B: {wandb.run.url}")
    print()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, metrics_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    print("\n" + "=" * 80)
    print("Evaluating Final Aggressive Model")
    print("=" * 80)
    
    obs = eval_env.reset()[0]
    done = False
    step = 0
    episode_reward = 0
    
    portfolio_values = []
    benchmark_values = []
    actions_history = []
    risk_metrics_history = []
    
    while not done and step < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        portfolio_values.append(info['portfolio_value'])
        benchmark_values.append(info['benchmark_value'])
        actions_history.append(info['weights'])
        risk_metrics_history.append({
            'volatility': info.get('portfolio_volatility', 0),
            'diversification': info.get('diversification_ratio', 0),
            'var': info.get('value_at_risk', 0),
            'outperformance': info.get('benchmark_outperformance', 0)
        })
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}: Portfolio = ${info['portfolio_value']:.2f}, "
                  f"Benchmark = ${info['benchmark_value']:.2f}, "
                  f"Outperformance = {info.get('benchmark_outperformance', 0)*100:.2f}%")
    
    final_metrics = eval_env.get_final_metrics()
    
    print("\n" + "=" * 80)
    print("Final Aggressive Evaluation Results")
    print("=" * 80)
    print(f"Total Steps: {step}")
    print(f"Episode Reward: {episode_reward:.2f}")
    print(f"Final Portfolio Value: ${final_metrics['final_portfolio_value']:.2f}")
    print(f"Final Benchmark Value: ${final_metrics['final_benchmark_value']:.2f}")
    print(f"Total Return: {final_metrics['total_return']*100:.2f}%")
    print(f"Benchmark Return: {final_metrics['benchmark_return']*100:.2f}%")
    print(f"Outperformance: {final_metrics['outperformance']*100:.2f}%")
    print(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {final_metrics.get('sortino_ratio', 0):.3f}")
    print(f"Calmar Ratio: {final_metrics.get('calmar_ratio', 0):.3f}")
    print(f"Max Drawdown: {final_metrics['max_drawdown']*100:.2f}%")
    print(f"Portfolio Volatility: {final_metrics.get('portfolio_volatility', 0)*100:.2f}%")
    print(f"Diversification Ratio: {final_metrics.get('diversification_ratio', 0)*100:.1f}%")
    print(f"Value at Risk (95%): {final_metrics.get('value_at_risk', 0)*100:.2f}%")
    print(f"Win Rate: {final_metrics['win_rate']*100:.1f}%")
    print(f"Profit Factor: {final_metrics['profit_factor']:.2f}")
    print(f"Trades Executed: {final_metrics['trades_executed']}")
    print(f"Total Fees: ${final_metrics['total_fees_paid']:.2f}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outperformance_str = f"{final_metrics['outperformance']*100:.1f}".replace('.', '_').replace('-', 'neg')
    return_str = f"{final_metrics['total_return']*100:.1f}".replace('.', '_').replace('-', 'neg')
    sharpe_str = f"{final_metrics['sharpe_ratio']:.2f}".replace('.', '_')
    
    final_model_name = f"{algorithm}_aggressive_outperf{outperformance_str}pct_ret{return_str}pct_sharpe{sharpe_str}_{timestamp}"
    final_model_path = os.path.join(save_dir, final_model_name)
    model.save(final_model_path)
    
    config_data = {
        'model_info': {
            'filename': final_model_name,
            'algorithm': algorithm,
            'timestamp': timestamp,
            'total_timesteps': total_timesteps,
            'strategy': 'aggressive'
        },
        'performance': {
            'sharpe_ratio': float(final_metrics['sharpe_ratio']),
            'sortino_ratio': float(final_metrics.get('sortino_ratio', 0)),
            'calmar_ratio': float(final_metrics.get('calmar_ratio', 0)),
            'total_return': float(final_metrics['total_return']),
            'benchmark_return': float(final_metrics['benchmark_return']),
            'outperformance': float(final_metrics['outperformance']),
            'max_drawdown': float(final_metrics['max_drawdown']),
            'portfolio_volatility': float(final_metrics.get('portfolio_volatility', 0)),
            'diversification_ratio': float(final_metrics.get('diversification_ratio', 0)),
            'value_at_risk': float(final_metrics.get('value_at_risk', 0)),
            'win_rate': float(final_metrics['win_rate']),
            'profit_factor': float(final_metrics['profit_factor']),
            'final_portfolio_value': float(final_metrics['final_portfolio_value']),
            'final_benchmark_value': float(final_metrics['final_benchmark_value']),
            'trades_executed': int(final_metrics['trades_executed']),
            'total_fees_paid': float(final_metrics['total_fees_paid']),
        },
        'training_config': {
            'symbols': available_symbols,
            'starting_balance': starting_balance,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'n_steps': n_steps,
        },
        'aggressive_environment_config': {
            'fee_rate': fee_rate,
            'lookback_window': lookback_window,
            'rebalance_frequency': rebalance_frequency,
            'risk_free_rate': risk_free_rate,
            'max_drawdown_penalty': max_drawdown_penalty,
            'normalize_obs': normalize_obs,
            'reward_horizon': reward_horizon,
            'benchmark_weight': benchmark_weight,
            'future_profit_weight': future_profit_weight,
            'future_window': future_window,
            'risk_adjustment': risk_adjustment,
            'dynamic_penalties': dynamic_penalties,
            'use_attention_weights': use_attention_weights,
            'volatility_scaling': volatility_scaling,
            'momentum_lookback': momentum_lookback,
            'concentration_limit': concentration_limit,
            'trend_following': trend_following,
        },
        'network': {
            'size': network_size,
            'pi_layers': net_arch['pi'],
            'vf_layers': net_arch['vf'],
        }
    }
    
    config_path = final_model_path + '_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\nModel saved to: {final_model_path}.zip")
    print(f"Config saved to: {config_path}")
    
    if use_wandb:
        artifact = wandb.Artifact(
            name=f"model-aggressive-final-{timestamp}",
            type="model",
            description=f"Final aggressive {algorithm} model - Outperformance: {final_metrics['outperformance']*100:.1f}%, "
                       f"Return: {final_metrics['total_return']*100:.1f}%, "
                       f"Sharpe: {final_metrics['sharpe_ratio']:.2f}"
        )
        artifact.add_file(final_model_path + ".zip")
        artifact.add_file(config_path)
        wandb.log_artifact(artifact)
    
    if use_wandb:
        wandb.log({
            'final/total_steps': step,
            'final/episode_reward': episode_reward,
            'final/portfolio_value': final_metrics['final_portfolio_value'],
            'final/benchmark_value': final_metrics['final_benchmark_value'],
            'final/total_return': final_metrics['total_return'],
            'final/benchmark_return': final_metrics['benchmark_return'],
            'final/outperformance': final_metrics['outperformance'],
            'final/sharpe_ratio': final_metrics['sharpe_ratio'],
            'final/sortino_ratio': final_metrics.get('sortino_ratio', 0),
            'final/calmar_ratio': final_metrics.get('calmar_ratio', 0),
            'final/max_drawdown': final_metrics['max_drawdown'],
            'final/portfolio_volatility': final_metrics.get('portfolio_volatility', 0),
            'final/diversification_ratio': final_metrics.get('diversification_ratio', 0),
            'final/value_at_risk': final_metrics.get('value_at_risk', 0),
            'final/win_rate': final_metrics['win_rate'],
            'final/profit_factor': final_metrics['profit_factor'],
            'final/trades_executed': final_metrics['trades_executed'],
            'final/total_fees': final_metrics['total_fees_paid'],
        })
    
    plot_aggressive_results(portfolio_values, benchmark_values, actions_history, 
                           risk_metrics_history, available_symbols, save_dir, use_wandb)
    
    if use_wandb:
        wandb.finish()
    
    return model, final_metrics


def plot_aggressive_results(portfolio_values, benchmark_values, actions_history, 
                           risk_metrics_history, symbols, save_dir, use_wandb=True):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Portfolio vs Benchmark
    axes[0, 0].plot(portfolio_values, label='Portfolio', linewidth=2)
    axes[0, 0].plot(benchmark_values, label='Benchmark', linewidth=2, linestyle='--')
    axes[0, 0].set_title('Portfolio vs Benchmark Value')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Asset allocation
    if len(actions_history) > 0:
        weights_array = np.array([[w[s] for s in symbols] for w in actions_history])
        for i, symbol in enumerate(symbols):
            axes[0, 1].plot(weights_array[:, i], label=symbol, alpha=0.7)
        axes[0, 1].set_title('Portfolio Weights Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0.0, 1.0)  # No negative weights
    
    # Risk metrics
    if len(risk_metrics_history) > 0:
        volatilities = [rm['volatility'] for rm in risk_metrics_history]
        diversifications = [rm['diversification'] for rm in risk_metrics_history]
        vars = [rm['var'] for rm in risk_metrics_history]
        outperformance = [rm['outperformance'] for rm in risk_metrics_history]
        
        axes[1, 0].plot(volatilities, color='red')
        axes[1, 0].set_title('Portfolio Volatility')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Volatility')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(diversifications, color='green')
        axes[1, 1].set_title('Diversification Ratio')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Diversification')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 0].plot(vars, color='orange')
        axes[2, 0].set_title('Value at Risk (95%)')
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('VaR')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(outperformance, color='purple')
        axes[2, 1].set_title('Benchmark Outperformance')
        axes[2, 1].set_xlabel('Step')
        axes[2, 1].set_ylabel('Outperformance')
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'aggressive_evaluation_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nAggressive plot saved to: {plot_path}")
    
    if use_wandb:
        wandb.log({'final/aggressive_evaluation_plot': wandb.Image(plot_path)})
    
    plt.close()


if __name__ == "__main__":
    try:
        from config import *
        print("Loaded configuration from config.py")
        
        # Override for aggressive strategy
        AGGRESSIVE_CONFIG = {
            'ALGORITHM': 'PPO',
            'TOTAL_TIMESTEPS': 200_000,
            'STARTING_BALANCE': 10000,
            'FEE_RATE': 0.001,
            'LOOKBACK_WINDOW': 50,
            'REBALANCE_FREQUENCY': 6,
            'RISK_FREE_RATE': 0.02,
            'MAX_DRAWDOWN_PENALTY': 1.0,
            'NORMALIZE_OBS': True,
            'REWARD_HORIZON': 1,
            'BENCHMARK_WEIGHT': 3.0,
            'FUTURE_PROFIT_WEIGHT': 2.0,
            'FUTURE_WINDOW': 8,
            'RISK_ADJUSTMENT': False,
            'DYNAMIC_PENALTIES': False,
            'USE_ATTENTION_WEIGHTS': True,
            'VOLATILITY_SCALING': False,
            'NETWORK_SIZE': 'large',
            'LEARNING_RATE': 0.0003,
            'BATCH_SIZE': 64,
            'N_STEPS': 2048,
            'MOMENTUM_LOOKBACK': 10,
            'CONCENTRATION_LIMIT': 0.7,
            'TREND_FOLLOWING': True
            # Removed LEVERAGE_LIMIT
        }
        
    except ImportError:
        print("No config.py found, using aggressive default configuration")
        SYMBOLS = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
            'XRPUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'AVAXUSDT'
        ]
        AGGRESSIVE_CONFIG = {
            'ALGORITHM': 'PPO',
            'TOTAL_TIMESTEPS': 200_000,
            'STARTING_BALANCE': 10000,
            'FEE_RATE': 0.001,
            'LOOKBACK_WINDOW': 60,
            'REBALANCE_FREQUENCY': 3,
            'RISK_FREE_RATE': -0.04,
            'MAX_DRAWDOWN_PENALTY': 1.0,
            'NORMALIZE_OBS': True,
            'REWARD_HORIZON': 1,
            'BENCHMARK_WEIGHT': 3.0,
            'FUTURE_PROFIT_WEIGHT': 5.0,
            'FUTURE_WINDOW': 8,
            'RISK_ADJUSTMENT': False,
            'DYNAMIC_PENALTIES': False,
            'USE_ATTENTION_WEIGHTS': True,
            'VOLATILITY_SCALING': False,
            'NETWORK_SIZE': 'large',
            'LEARNING_RATE': 0.0003,
            'BATCH_SIZE': 64,
            'N_STEPS': 2048,
            'MOMENTUM_LOOKBACK': 10,
            'CONCENTRATION_LIMIT': 0.7,
            'TREND_FOLLOWING': True
            # Removed LEVERAGE_LIMIT
        }
    
    model, metrics = train_aggressive_portfolio_agent(
        symbols=SYMBOLS,
        algorithm=AGGRESSIVE_CONFIG['ALGORITHM'],
        total_timesteps=AGGRESSIVE_CONFIG['TOTAL_TIMESTEPS'],
        learning_rate=AGGRESSIVE_CONFIG['LEARNING_RATE'],
        batch_size=AGGRESSIVE_CONFIG['BATCH_SIZE'],
        n_steps=AGGRESSIVE_CONFIG['N_STEPS'],
        starting_balance=AGGRESSIVE_CONFIG['STARTING_BALANCE'],
        save_dir=f'models/aggressive_portfolio_{AGGRESSIVE_CONFIG["ALGORITHM"].lower()}',
        log_dir=f'logs/aggressive_portfolio_{AGGRESSIVE_CONFIG["ALGORITHM"].lower()}',
        fee_rate=AGGRESSIVE_CONFIG['FEE_RATE'],
        lookback_window=AGGRESSIVE_CONFIG['LOOKBACK_WINDOW'],
        rebalance_frequency=AGGRESSIVE_CONFIG['REBALANCE_FREQUENCY'],
        risk_free_rate=AGGRESSIVE_CONFIG['RISK_FREE_RATE'],
        max_drawdown_penalty=AGGRESSIVE_CONFIG['MAX_DRAWDOWN_PENALTY'],
        normalize_obs=AGGRESSIVE_CONFIG['NORMALIZE_OBS'],
        reward_horizon=AGGRESSIVE_CONFIG['REWARD_HORIZON'],
        benchmark_weight=AGGRESSIVE_CONFIG['BENCHMARK_WEIGHT'],
        future_profit_weight=AGGRESSIVE_CONFIG['FUTURE_PROFIT_WEIGHT'],
        future_window=AGGRESSIVE_CONFIG['FUTURE_WINDOW'],
        risk_adjustment=AGGRESSIVE_CONFIG['RISK_ADJUSTMENT'],
        dynamic_penalties=AGGRESSIVE_CONFIG['DYNAMIC_PENALTIES'],
        use_attention_weights=AGGRESSIVE_CONFIG['USE_ATTENTION_WEIGHTS'],
        volatility_scaling=AGGRESSIVE_CONFIG['VOLATILITY_SCALING'],
        network_size=AGGRESSIVE_CONFIG['NETWORK_SIZE'],
        momentum_lookback=AGGRESSIVE_CONFIG['MOMENTUM_LOOKBACK'],
        concentration_limit=AGGRESSIVE_CONFIG['CONCENTRATION_LIMIT'],
        trend_following=AGGRESSIVE_CONFIG['TREND_FOLLOWING']
    )
    
    print("\n" + "=" * 80)
    print("Aggressive Training Complete!")
    print("=" * 80)