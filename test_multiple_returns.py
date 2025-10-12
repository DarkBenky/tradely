import pandas as pd
import numpy as np
from ta_indicators import calculate_volume_nodes, calculate_time_nodes, calculate_trade_nodes

# Create test data with enough rows
np.random.seed(42)
test_data = {
    'datetime': pd.date_range('2023-01-01', periods=100, freq='5min'),
    'open': np.random.uniform(40000, 50000, 100),
    'high': np.random.uniform(45000, 55000, 100),
    'low': np.random.uniform(35000, 45000, 100),
    'close': np.random.uniform(40000, 50000, 100),
    'volume': np.random.uniform(100, 1000, 100),
    'number_of_trades': np.random.randint(50, 200, 100)
}

# Ensure high > low > open, close
for i in range(100):
    low = test_data['low'][i]
    high = test_data['high'][i]
    if high <= low:
        high = low + np.random.uniform(1000, 5000)
        test_data['high'][i] = high
    
    # Ensure open and close are between low and high
    test_data['open'][i] = np.random.uniform(low, high)
    test_data['close'][i] = np.random.uniform(low, high)

df = pd.DataFrame(test_data)
df = df.set_index('datetime')

print("Testing volume nodes with multiple returns...")
try:
    vol_results = calculate_volume_nodes(df, window_size=12, num_bins=20, multiple_returns=3)
    print(f"Volume nodes returned {len(vol_results)} columns:")
    for col in sorted(vol_results.keys()):
        print(f"  - {col}: shape {vol_results[col].shape}")
        # Show some sample values
        non_nan_values = vol_results[col].dropna()
        if len(non_nan_values) > 0:
            print(f"    Sample values: {non_nan_values[:3].tolist()}")
        else:
            print(f"    All NaN values")
    print()
except Exception as e:
    print(f"Error in volume nodes: {e}")
    import traceback
    traceback.print_exc()

print("Testing time nodes with multiple returns...")
try:
    time_results = calculate_time_nodes(df, window_size=12, num_bins=20, multiple_returns=3)
    print(f"Time nodes returned {len(time_results)} columns:")
    for col in sorted(time_results.keys()):
        print(f"  - {col}: shape {time_results[col].shape}")
        # Show some sample values
        non_nan_values = time_results[col].dropna()
        if len(non_nan_values) > 0:
            print(f"    Sample values: {non_nan_values[:3].tolist()}")
        else:
            print(f"    All NaN values")
    print()
except Exception as e:
    print(f"Error in time nodes: {e}")
    import traceback
    traceback.print_exc()

print("Testing trade nodes with multiple returns...")
try:
    trade_results = calculate_trade_nodes(df, window_size=12, num_bins=20, multiple_returns=3)
    print(f"Trade nodes returned {len(trade_results)} columns:")
    for col in sorted(trade_results.keys()):
        print(f"  - {col}: shape {trade_results[col].shape}")
        # Show some sample values
        non_nan_values = trade_results[col].dropna()
        if len(non_nan_values) > 0:
            print(f"    Sample values: {non_nan_values[:3].tolist()}")
        else:
            print(f"    All NaN values")
    print()
except Exception as e:
    print(f"Error in trade nodes: {e}")
    import traceback
    traceback.print_exc()