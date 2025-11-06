# Trading AI Training Pipeline

## Architecture Overview

```
Market Data (BTCUSDT, ETHUSDT, SOLUSDT, etc.)
5min OHLCV -> Technical indicators (134,980 features)
         |
         v
+-------------------------------------------------------------------+
|         FEATURE REDUCER (Autoencoder - End-to-End Trained)        |
|  +-------------------------------------------------------------+  |
|  |  Encoder: 134,980 -> 3,222 -> 2,148 (latent space)          |  |
|  |  Decoder: 2,148 -> 3,222 -> 134,980 (reconstruction)        |  |
|  +-------------------------------------------------------------+  |
|  Compression: 62.8x                                               |
|  Memory efficient: Adafactor optimizer                            |
|  Mixed precision (float16)                                        |
|  TRAINED WITH POLICY: Learns trading-relevant features            |
+-------|-----------------------------------------------------------+
        |
        | Compressed features (2,148 dims)
        v
+--------------------------------------------------------------------+
|           PPO POLICY NETWORK (Actor-Critic)                        |
|  +--------------------------------------------------------------+  |
|  |  Input: 2,148 features                                       |  |
|  |    |                                                         |  |
|  |  Feature Extractor: -> 768 -> 384                            |  |
|  |    |                                                         |  |
|  |  Transformer Blocks (8x):                                    |  |
|  |    Multi-head attention (12 heads, 192 dim)                  |  |
|  |    Feed-forward network (768 dim)                            |  |
|  |    |                                                         |  |
|  |  +-----------------+-----------------+                       |  |
|  |  |  Actor Head     |  Critic Head    |                       |  |
|  |  |  (Portfolio %)  |  (Value Est.)   |                       |  |
|  |  |  11 assets      |  1 scalar       |                       |  |
|  |  +-----------------+-----------------+                       |  |
|  +--------------------------------------------------------------+  |
+-------|------------------------------------------------------------+
        |
        v
+--------------------------------------------------------------------+
|                    PPO TRAINING LOOP                               |
|                                                                    |
|  1. Collect Batch (128 steps):                                     |
|     Policy actions (95%) + Random exploration (5%)                 |
|     Calculate rewards, values, advantages (GAE lambda=0.95)        |
|                                                                    |
|  2. Train Step (JOINT OPTIMIZATION):                               |
|                                                                    |
|     Policy Loss = Actor Loss + 0.5 x Critic Loss - 0.01 x Entropy  |
|                                                                    |
|     Actor Loss (Clipped):                                          |
|       min(ratio x A, clip(ratio,0.8,1.2) x A)                      |
|       where ratio = pi_new/pi_old                                  |
|                                                                    |
|     Critic Loss:                                                   |
|       MSE(value_predicted, returns)                                |
|                                                                    |
|     Encoder Loss (END-TO-END):                                     |
|       0.7 x Policy Loss + 0.3 x Reconstruction Loss                |
|       -> Feature extractor learns what helps trading               |
|                                                                    |
|  3. Update:                                                        |
|     Gradient clipping (max_norm=1.0)                               |
|     Adam optimizer (policy_lr=0.0005, encoder_lr=0.00005)          |
|     Epsilon decay (0.075 -> 0.02)                                  |
|                                                                    |
|  4. Evaluate & Save:                                               |
|     Track rolling reward (50 batches)                              |
|     Save best model + trained encoder when improved                |
|     Reduce LR if no improvement (100 batches)                      |
+--------------------------------------------------------------------+

+--------------------------------------------------------------------+
|                    KEY METRICS                                     |
+--------------------------------------------------------------------+
|  Portfolio Value vs Benchmark                                      |
|  Sharpe Ratio, Max Drawdown                                        |
|  Policy Entropy (exploration)                                      |
|  Advantage Mean/Std                                                |
|  Clip Fraction (PPO diagnostic)                                    |
|  Feature Reconstruction Accuracy (encoder quality)                 |
|  Encoder Loss (feature learning progress)                          |
+--------------------------------------------------------------------+
```

## Training Flow

```
START
  |
  +-> [1] PRETRAIN (Optional)
  |     +- Load synthetic/recorded data
  |     +- Train on historical batches (10 epochs)
  |     +- Save pretrained weights
  |
  +-> [2] ONLINE TRAINING (5000 batches)
  |     |
  |     +-> Collect Batch
  |     |     +- Compress with Feature Reducer
  |     |     +- Run policy (Actor output)
  |     |     +- Execute trades in environment
  |     |     +- Calculate GAE advantages
  |     |
  |     +-> Train Policy (PPO) + Encoder (End-to-End)
  |     |     +- Actor loss (clipped objective)
  |     |     +- Critic loss (value prediction)
  |     |     +- Entropy bonus
  |     |     +- Encoder loss (policy + reconstruction)
  |     |     +- Backprop through encoder
  |     |
  |     +-> Update & Log
  |           +- Save if best performance
  |           +- Decay epsilon
  |           +- Adjust learning rate
  |
  +-> SAVE FINAL MODEL + TRAINED ENCODER
```

## Key Improvements

1. Feature Compression: 134K -> 2K features (62x smaller)
   - Faster training, less memory
   - Autoencoder learns trading-relevant features

2. End-to-End Training: Encoder + Policy trained together
   - Feature extractor learns what matters for trading
   - Not just generic compression, but task-specific
   - Gradients flow from rewards back to feature extraction

3. PPO Algorithm: More stable than vanilla PG
   - Clipped objective prevents large updates
   - Value function reduces variance
   - Entropy bonus maintains exploration

4. Actor-Critic: Dual-head architecture
   - Actor: Portfolio allocation decisions
   - Critic: Value estimation for advantage

5. Memory Optimization:
   - Mixed precision (float16)
   - Adafactor optimizer (no momentum storage)
   - Streaming data loading

## Files

- feature_reducer.py - Autoencoder training/inference
- train.py - PPO policy + end-to-end encoder training
- portfolio_env.py - Trading environment
- getData.py - Market data collection

## Usage

```bash
# 1. Train feature reducer (optional pretraining)
python feature_reducer.py

# 2. Train trading policy (will train encoder end-to-end)
python train.py
```
