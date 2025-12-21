# Cross-Asset Relative Value Transformer

Implementation of a Transformer-based model for predicting relative performance of assets vs an index.

## Architecture

**Input**: `(batch, 11 assets, 128 timesteps, 4 features)`
- 1 index + 10 tradable assets
- 128 hour window (hourly data)
- Features: log_return, relative_log_return, rolling_volatility, market_cap_weight

**Output**: `(batch, 10)` - Relative performance scores for tradable assets

**Model Structure**:
1. Time Encoder: Multi-head attention over time dimension per asset
2. Asset Encoder: Multi-head attention across assets
3. Output Head: Linear projection to scores

## Files

- `crossAssetData.py`: Data loading, feature engineering, normalization
- `crossAssetModel.py`: Transformer architecture (time + asset encoders)
- `trainCrossAsset.py`: Training loop with cross-sectional MSE + ranking loss
- `evaluateCrossAsset.py`: Evaluation metrics and backtesting

## Usage

### Train Model
```bash
python trainCrossAsset.py
```

### Evaluate
```bash
python evaluateCrossAsset.py eval
python evaluateCrossAsset.py backtest
python evaluateCrossAsset.py analyze
```

## Key Features

- Cross-sectional normalization (assets normalized relative to each other)
- Combined loss: MSE + ranking loss for better ordering
- Long/short portfolio strategy (top 3 long, bottom 3 short)
- Rank correlation metric for prediction quality

## Strategy

Predict relative returns → Rank assets → Long top performers, Short bottom performers
