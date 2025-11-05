"""
NEW REWARD SYSTEM - QUICK REFERENCE
====================================

PHILOSOPHY: Simple and focused - maximize actual next-step profits

FORMULA:
--------
reward = portfolio_weighted_return * 1000 + outperformance_bonus * 500

Where:
  portfolio_weighted_return = sum(weight[asset] * next_step_return[asset])
  outperformance_bonus = portfolio_return - benchmark_return
  
Clipped to: [-100, 100]

EXAMPLE SCENARIOS:
------------------

Scenario 1: Perfect Prediction
- Portfolio: 100% in BTC
- BTC goes up 1% next step
- Benchmark goes up 0.5% (mixed allocation)
→ reward = (1.0 * 0.01) * 1000 + (0.01 - 0.005) * 500
→ reward = 10 + 2.5 = 12.5

Scenario 2: Wrong Allocation
- Portfolio: 100% in ETH  
- ETH goes down 1% next step
- Benchmark goes up 0.2%
→ reward = (1.0 * -0.01) * 1000 + (-0.01 - 0.002) * 500
→ reward = -10 + (-6) = -16

Scenario 3: Too Much Cash
- Portfolio: 50% BTC, 50% cash
- BTC goes up 2% next step
- Benchmark goes up 1% (full allocation)
→ reward = (0.5 * 0.02 + 0.5 * 0) * 1000 + (0.01 - 0.01) * 500
→ reward = 10 + 0 = 10
(vs. 20 if fully invested - opportunity cost!)

WHAT THE AGENT LEARNS:
-----------------------
✓ Allocate to assets that will rise
✓ Avoid assets that will fall
✓ Minimize cash (0% return)
✓ Beat the benchmark
✓ Make actual profit (not complex metrics)

COMPARISON TO OLD SYSTEM:
--------------------------
OLD: 6+ weighted components, 30-step lookforward, ~150 lines
NEW: 2 simple components, 1-step lookforward, ~45 lines

OLD COMPONENTS (removed):
- Portfolio value change tracking
- Sharpe ratio calculation
- Unrealized P&L quality scores
- Concentration penalties
- Complex cash management rules
- Position sizing heuristics
- Win rate bonuses

NEW COMPONENTS (kept):
- Direct profit maximization
- Benchmark comparison

TRAINING TIPS:
--------------
1. Regenerate synthetic data with lookforward_steps=1
2. Agent should learn faster (clearer signal)
3. Watch for overtrading (rebalance_threshold prevents this)
4. Monitor actual portfolio returns, not just reward values

CODE LOCATIONS:
---------------
Reward function:     portfolio_env.py::_calculate_reward() [line ~555]
Future profit calc:  portfolio_env.py::_calculate_future_profit() [line ~373]
Step function:       portfolio_env.py::step() [line ~607]
"""
