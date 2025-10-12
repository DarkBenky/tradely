import pandas as pd
from ta_indicators import calculate_all_indicators

def test_with_existing_data():
    """Test windowed calculations with existing BTCUSDT data"""
    try:
        # Load existing data
        btc_data = pd.read_csv('data/BTCUSDT_5min_data.csv')
        print(f"Loaded BTCUSDT data: {btc_data.shape}")
        print("Columns:", list(btc_data.columns))
        
        # Test with different window sizes
        test_data = {'BTCUSDT': btc_data}
        
        for window_size in [30, 50, 100]:
            print(f"\n=== Testing BTCUSDT with window_size = {window_size} ===")
            
            results = calculate_all_indicators(test_data, window_size=window_size)
            
            if 'BTCUSDT' in results:
                result_df = results['BTCUSDT']
                print(f"Original data shape: {btc_data.shape}")
                print(f"Result shape: {result_df.shape}")
                print(f"Row count match: {len(result_df) == len(btc_data)}")
                
                # Show indicator columns
                original_cols = set(btc_data.columns)
                indicator_cols = [col for col in result_df.columns if col not in original_cols]
                print(f"Added {len(indicator_cols)} indicator columns:")
                for col in indicator_cols[:10]:  # Show first 10
                    non_nan_count = result_df[col].notna().sum()
                    print(f"  {col}: {non_nan_count}/{len(result_df)} non-NaN values")
                
                # Save sample for inspection
                sample_file = f'data/BTCUSDT_windowed_{window_size}.csv'
                result_df.to_csv(sample_file, index=False)
                print(f"Saved sample to: {sample_file}")
                
    except FileNotFoundError:
        print("BTCUSDT data file not found. Please run getData.py first to fetch data.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_existing_data()