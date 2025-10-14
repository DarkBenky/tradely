# Cryptocurrency Portfolio RL Trading

Reinforcement Learning for multi-currency portfolio management using PPO/A2C/SAC.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train
```bash
python train_portfolio_agent.py
```

Training stops automatically at timesteps configured in `config.py`.
Models saved with performance metrics in filename.

### 2. Compare Models
```bash
python load_model.py compare
```

### 3. Load Model
```python
from load_model import load_model

model, config = load_model('models/portfolio_ppo/MODEL_NAME.zip')
action, _ = model.predict(obs, deterministic=True)
```

## Configuration

Edit `config.py` to adjust:

**Training:**
- `TOTAL_TIMESTEPS` - Training duration
- `LEARNING_RATE`, `BATCH_SIZE`, `N_STEPS`
- `NETWORK_SIZE` - small/medium/large/xlarge (for 6GB VRAM use medium)

**Environment:**
- `FEE_RATE` - Transaction fees (0.001 = 0.1%)
- `LOOKBACK_WINDOW` - Historical data window
- `REBALANCE_FREQUENCY` - Steps between rebalancing
- `REWARD_HORIZON` - Look-ahead for reward calculation

**Rewards:**
- `REWARD_RETURN_WEIGHT` - Profit focus
- `REWARD_SHARPE_WEIGHT` - Risk-adjusted returns
- `REWARD_DRAWDOWN_WEIGHT` - Drawdown penalty
- `REWARD_VOLATILITY_WEIGHT` - Volatility penalty

## Model Files

Models saved as: `PPO_sharpe2_15_ret15_3pct_TIMESTAMP.zip`
- Sharpe ratio: 2.15
- Return: 15.3%
- Config: `*_config.json` (contains all parameters for reproduction)

## Monitoring

- W&B: Dashboard URL printed at training start
- TensorBoard: `tensorboard --logdir=logs/portfolio_ppo`

## Load Model Commands

```bash
python load_model.py list                    # List all models
python load_model.py compare                 # Compare by Sharpe
python load_model.py info MODEL_PATH.zip     # Full model details
```

## Key Files

- `config.py` - All configuration parameters
- `train_portfolio_agent.py` - Training script
- `load_model.py` - Model management utilities
- `portfolio_env.py` - RL environment
- `getData.py` - Data collection from Binance

