import os
import json
from stable_baselines3 import PPO, A2C, SAC
import glob


def list_available_models(save_dir='models/portfolio_ppo'):
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} not found")
        return []
    
    config_files = glob.glob(os.path.join(save_dir, '*_config.json'))
    
    if not config_files:
        print(f"No models with configs found in {save_dir}")
        return []
    
    models = []
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        model_path = config_file.replace('_config.json', '.zip')
        if os.path.exists(model_path):
            models.append({
                'path': model_path,
                'config_path': config_file,
                'config': config
            })
    
    return models


def print_model_info(model_info):
    config = model_info['config']
    
    print("\n" + "=" * 70)
    print(f"Model: {os.path.basename(model_info['path'])}")
    print("=" * 70)
    
    print("\nPERFORMANCE:")
    perf = config['performance']
    print(f"  Sharpe Ratio:      {perf['sharpe_ratio']:.3f}")
    print(f"  Total Return:      {perf['total_return']*100:+.2f}%")
    print(f"  Max Drawdown:      {perf['max_drawdown']*100:.2f}%")
    print(f"  Win Rate:          {perf['win_rate']*100:.1f}%")
    print(f"  Profit Factor:     {perf['profit_factor']:.2f}")
    print(f"  Final Value:       ${perf['final_portfolio_value']:.2f}")
    print(f"  Trades:            {perf['trades_executed']}")
    print(f"  Total Fees:        ${perf['total_fees_paid']:.2f}")
    
    print("\nTRAINING CONFIG:")
    train = config['training_config']
    print(f"  Algorithm:         {config['model_info']['algorithm']}")
    print(f"  Timesteps:         {config['model_info']['total_timesteps']:,}")
    print(f"  Learning Rate:     {train['learning_rate']}")
    print(f"  Batch Size:        {train['batch_size']}")
    print(f"  Symbols:           {', '.join(train['symbols'][:3])}... ({len(train['symbols'])} total)")
    
    print("\nENVIRONMENT CONFIG:")
    env = config['environment_config']
    print(f"  Fee Rate:          {env['fee_rate']*100:.2f}%")
    print(f"  Lookback Window:   {env['lookback_window']}")
    print(f"  Rebalance Freq:    {env['rebalance_frequency']}")
    print(f"  Risk-Free Rate:    {env['risk_free_rate']*100:.2f}%")
    
    print("\nREWARD WEIGHTS:")
    weights = config['reward_weights']
    print(f"  Return:            {weights['return_weight']}")
    print(f"  Sharpe:            {weights['sharpe_weight']}")
    print(f"  Drawdown:          {weights['drawdown_weight']}")
    print(f"  Volatility:        {weights['volatility_weight']}")
    
    print("\nNETWORK:")
    net = config['network']
    print(f"  Size:              {net['size']}")
    print(f"  Policy Layers:     {net['pi_layers']}")
    print(f"  Value Layers:      {net['vf_layers']}")
    
    print("\nCREATED:")
    print(f"  Timestamp:         {config['model_info']['timestamp']}")


def load_model(model_path, algorithm='PPO'):
    if not model_path.endswith('.zip'):
        model_path += '.zip'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    config_path = model_path.replace('.zip', '_config.json')
    config = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        algorithm = config['model_info']['algorithm']
        print(f"Loaded config from: {config_path}")
    
    print(f"Loading {algorithm} model from: {model_path}")
    
    if algorithm == 'PPO':
        model = PPO.load(model_path)
    elif algorithm == 'A2C':
        model = A2C.load(model_path)
    elif algorithm == 'SAC':
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print("Model loaded successfully!")
    return model, config


def compare_models(save_dir='models/portfolio_ppo'):
    models = list_available_models(save_dir)
    
    if not models:
        return
    
    print("\n" + "=" * 70)
    print(f"FOUND {len(models)} MODEL(S) IN {save_dir}")
    print("=" * 70)
    
    models_sorted = sorted(models, key=lambda x: x['config']['performance']['sharpe_ratio'], reverse=True)
    
    print("\n{:<5} {:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Rank", "Model", "Sharpe", "Return%", "Drawdown%", "Win Rate%"
    ))
    print("-" * 70)
    
    for i, model_info in enumerate(models_sorted, 1):
        perf = model_info['config']['performance']
        filename = os.path.basename(model_info['path']).replace('.zip', '')
        if len(filename) > 25:
            filename = filename[:22] + '...'
        
        print("{:<5} {:<25} {:>10.3f} {:>9.1f}% {:>9.1f}% {:>9.1f}%".format(
            i,
            filename,
            perf['sharpe_ratio'],
            perf['total_return'] * 100,
            perf['max_drawdown'] * 100,
            perf['win_rate'] * 100
        ))
    
    return models_sorted


if __name__ == "__main__":
    print("\nMODEL MANAGER")
    print("=" * 70)
    
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python load_model.py list [directory]          - List all models")
        print("  python load_model.py info <model_path>         - Show model details")
        print("  python load_model.py compare [directory]       - Compare all models")
        print("\nExamples:")
        print("  python load_model.py list")
        print("  python load_model.py list models/portfolio_a2c")
        print("  python load_model.py info models/portfolio_ppo/PPO_sharpe2_15_ret15_3pct_20251014_230145.zip")
        print("  python load_model.py compare")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == 'list':
        save_dir = sys.argv[2] if len(sys.argv) > 2 else 'models/portfolio_ppo'
        models = list_available_models(save_dir)
        
        if models:
            print(f"\nFound {len(models)} model(s):")
            for i, model_info in enumerate(models, 1):
                perf = model_info['config']['performance']
                print(f"\n{i}. {os.path.basename(model_info['path'])}")
                print(f"   Sharpe: {perf['sharpe_ratio']:.3f} | Return: {perf['total_return']*100:+.1f}% | Drawdown: {perf['max_drawdown']*100:.1f}%")
    
    elif command == 'info':
        if len(sys.argv) < 3:
            print("Error: Please provide model path")
            print("Usage: python load_model.py info <model_path>")
            sys.exit(1)
        
        model_path = sys.argv[2]
        config_path = model_path.replace('.zip', '_config.json')
        
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print_model_info({'path': model_path, 'config': config})
    
    elif command == 'compare':
        save_dir = sys.argv[2] if len(sys.argv) > 2 else 'models/portfolio_ppo'
        compare_models(save_dir)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'list', 'info', or 'compare'")
