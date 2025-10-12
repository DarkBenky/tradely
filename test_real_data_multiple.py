import pandas as pd
import os
from ta_indicators import calculate_all_indicators

# Test with existing BTCUSDT data if available
data_file = 'data/BTCUSDT_5min_data.csv'

if os.path.exists(data_file):
    print(f"Loading {data_file}...")
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Use only first 50 rows for quick test
    df_small = df.head(50).copy()
    
    print(f"Data shape: {df_small.shape}")
    print(f"Columns: {list(df_small.columns)}")
    
    # Test with small dataset
    dfs_test = {'BTCUSDT': df_small}
    
    print("\nTesting multiple returns with real data...")
    results = calculate_all_indicators(dfs_test, window_size=12, multiple_returns=3)
    
    if 'BTCUSDT' in results:
        btc_result = results['BTCUSDT']
        print(f"\nBTC result shape: {btc_result.shape}")
        
        # Show columns related to nodes
        node_columns = [col for col in btc_result.columns if 'htn' in col or 'hvn' in col or 'ltn' in col or 'lvn' in col or 'trade_' in col]
        print(f"\nNode-related columns ({len(node_columns)}):")
        for col in sorted(node_columns):
            non_nan_count = btc_result[col].notna().sum()
            print(f"  - {col}: {non_nan_count}/{len(btc_result)} non-NaN values")
            if non_nan_count > 0:
                unique_values = btc_result[col].dropna().nunique()
                print(f"    Unique values: {unique_values}")
                if unique_values <= 5:  # Show actual values if few unique ones
                    unique_vals = btc_result[col].dropna().unique()[:3]
                    print(f"    Sample values: {unique_vals}")
    
else:
    print(f"File {data_file} not found. Please run getData.py first to fetch data.")