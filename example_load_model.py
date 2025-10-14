from load_model import load_model, compare_models
from train_portfolio_agent import create_env, load_data
import numpy as np


print("\n" + "=" * 70)
print("EXAMPLE: Loading and Using a Saved Model")
print("=" * 70)

best_models = compare_models('models/portfolio_ppo')

if not best_models:
    print("\nNo models found. Train a model first using train_portfolio_agent.py")
    exit()

print("\nLoading the best model (highest Sharpe ratio)...")
best_model_info = best_models[0]

model, config = load_model(best_model_info['path'])

print("\nModel loaded! Here's what you can do:\n")

print("1️⃣  Make predictions:")
print("   obs = env.reset()[0]")
print("   action, _ = model.predict(obs, deterministic=True)")
print()

print("2️⃣  Re-create the exact environment:")
env_config = config['environment_config']
train_config = config['training_config']
print(f"   symbols = {train_config['symbols']}")
print(f"   dfs = load_data(symbols)")
print(f"   env = create_env(")
print(f"       dfs, symbols,")
print(f"       starting_balance={train_config['starting_balance']},")
print(f"       fee_rate={env_config['fee_rate']},")
print(f"       lookback_window={env_config['lookback_window']},")
print(f"       rebalance_frequency={env_config['rebalance_frequency']},")
print(f"       risk_free_rate={env_config['risk_free_rate']},")
print(f"       max_drawdown_penalty={env_config['max_drawdown_penalty']},")
print(f"       normalize_obs={env_config['normalize_obs']}")
print(f"   )")
print()

print("3️⃣  Continue training from this checkpoint:")
print("   model.learn(total_timesteps=50000)")
print()

print("4️⃣  Test on new data:")
print("   obs = test_env.reset()[0]")
print("   done = False")
print("   while not done:")
print("       action, _ = model.predict(obs, deterministic=True)")
print("       obs, reward, terminated, truncated, info = test_env.step(action)")
print("       done = terminated or truncated")
print("   metrics = test_env.get_final_metrics()")
print()

print("\nTIP: You can load any model by its path:")
print("   model, config = load_model('models/portfolio_ppo/PPO_sharpe2_15_ret15_3pct_20251014_230145.zip')")
print()
