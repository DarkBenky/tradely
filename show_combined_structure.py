import pandas as pd
import numpy as np
from ta_indicators import calculate_all_indicators

# Create minimal test data that matches Binance format
np.random.seed(42)
test_data = {
    'datetime': pd.date_range('2023-01-01', periods=50, freq='5min'),
    'open': np.random.uniform(40000, 50000, 50),
    'high': np.random.uniform(45000, 55000, 50),
    'low': np.random.uniform(35000, 45000, 50),
    'close': np.random.uniform(40000, 50000, 50),
    'volume': np.random.uniform(100, 1000, 50),
    'close_time': pd.date_range('2023-01-01', periods=50, freq='5min'),
    'quote_asset_volume': np.random.uniform(1000000, 10000000, 50),
    'number_of_trades': np.random.randint(50, 200, 50),
    'taker_buy_base_asset_volume': np.random.uniform(50, 500, 50),
    'taker_buy_quote_asset_volume': np.random.uniform(500000, 5000000, 50),
    'ignore': np.zeros(50)
}

# Ensure high > low and open/close between them
for i in range(50):
    low = test_data['low'][i]
    high = test_data['high'][i]
    if high <= low:
        high = low + np.random.uniform(1000, 5000)
        test_data['high'][i] = high
    
    test_data['open'][i] = np.random.uniform(low, high)
    test_data['close'][i] = np.random.uniform(low, high)

df = pd.DataFrame(test_data)
df = df.set_index('datetime')

# Test with one symbol
dfs_test = {'BTCUSDT': df}

print("Calculating all indicators to show combined file structure...")
print("=" * 60)

try:
    results = calculate_all_indicators(dfs_test, window_size=12, multiple_returns=5)
    
    if 'BTCUSDT' in results:
        btc_result = results['BTCUSDT']
        
        print(f"COMBINED FILE STRUCTURE:")
        print(f"Total columns: {len(btc_result.columns)}")
        print(f"Total rows: {len(btc_result)}")
        print()
        
        print("COLUMN LIST:")
        print("-" * 40)
        
        # Group columns by type
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'close_time', 
                        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                        'taker_buy_quote_asset_volume', 'ignore']
        
        ma_cols = [col for col in btc_result.columns if col.startswith('ma_') or col.startswith('ema_')]
        indicator_cols = [col for col in btc_result.columns if col in ['rsi', 'atr', 'vwap']]
        macd_cols = [col for col in btc_result.columns if 'macd' in col]
        bb_cols = [col for col in btc_result.columns if col.startswith('bb_')]
        vol_node_cols = [col for col in btc_result.columns if 'hvn' in col or 'lvn' in col or 'volume_profile' in col]
        time_node_cols = [col for col in btc_result.columns if 'htn' in col or 'ltn' in col or 'time_profile' in col]
        trade_node_cols = [col for col in btc_result.columns if 'trade_' in col]
        
        print("1. ORIGINAL BINANCE DATA:")
        for col in original_cols:
            if col in btc_result.columns:
                print(f"   {col}")
        
        print(f"\n2. MOVING AVERAGES ({len(ma_cols)} columns):")
        for col in sorted(ma_cols):
            print(f"   {col}")
            
        print(f"\n3. BASIC INDICATORS ({len(indicator_cols)} columns):")
        for col in sorted(indicator_cols):
            print(f"   {col}")
            
        print(f"\n4. MACD INDICATORS ({len(macd_cols)} columns):")
        for col in sorted(macd_cols):
            print(f"   {col}")
            
        print(f"\n5. BOLLINGER BANDS ({len(bb_cols)} columns):")
        for col in sorted(bb_cols):
            print(f"   {col}")
            
        print(f"\n6. VOLUME NODES ({len(vol_node_cols)} columns):")
        for col in sorted(vol_node_cols):
            print(f"   {col}")
            
        print(f"\n7. TIME NODES ({len(time_node_cols)} columns):")
        for col in sorted(time_node_cols):
            print(f"   {col}")
            
        print(f"\n8. TRADE NODES ({len(trade_node_cols)} columns):")
        for col in sorted(trade_node_cols):
            print(f"   {col}")
        
        print(f"\nTOTAL: {len(btc_result.columns)} columns")
        
        # Show sample of first few rows
        print(f"\nSAMPLE DATA (first 3 rows, selected columns):")
        print("-" * 60)
        sample_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'ma_5', 'ma_10', 'hvn_level_1', 'trade_htn_level_1']
        available_sample_cols = [col for col in sample_cols if col in btc_result.columns]
        if available_sample_cols:
            print(btc_result[available_sample_cols].head(3))
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()