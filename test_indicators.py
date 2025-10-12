import pandas as pd
import numpy as np
from ta_indicators import calculate_all_indicators
import warnings
warnings.filterwarnings('ignore')

def test_windowed_calculations():
    """Test the new windowed indicator calculations"""
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    
    # Generate realistic OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.uniform(0, 2, 100)
    low_prices = close_prices - np.random.uniform(0, 2, 100)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    volumes = np.random.randint(1000, 10000, 100)
    trades = np.random.randint(10, 100, 100)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes,
        'close_time': dates + pd.Timedelta('5min'),
        'quote_asset_volume': volumes * close_prices,
        'number_of_trades': trades,
        'taker_buy_base_asset_volume': volumes * 0.6,
        'taker_buy_quote_asset_volume': volumes * close_prices * 0.6,
    })
    
    print("Sample data shape:", sample_data.shape)
    print("Sample data columns:", list(sample_data.columns))
    
    # Test with different window sizes
    test_data = {'TEST': sample_data}
    
    for window_size in [20, 30, 50]:
        print(f"\n=== Testing window_size = {window_size} ===")
        try:
            results = calculate_all_indicators(test_data, window_size=window_size)
            
            if 'TEST' in results:
                result_df = results['TEST']
                print(f"Result shape: {result_df.shape}")
                print(f"Number of columns: {len(result_df.columns)}")
                print(f"Original rows: {len(sample_data)}, Result rows: {len(result_df)}")
                
                # Check if we have the same number of rows
                if len(result_df) == len(sample_data):
                    print("✅ Row count matches!")
                else:
                    print("❌ Row count mismatch!")
                
                # Show some indicator columns
                indicator_cols = [col for col in result_df.columns if col not in sample_data.columns]
                print(f"Indicator columns added: {len(indicator_cols)}")
                print("First few indicators:", indicator_cols[:5])
                
                # Check for NaN patterns
                nan_counts = result_df.isnull().sum()
                indicators_with_nans = nan_counts[nan_counts > 0]
                if len(indicators_with_nans) > 0:
                    print("Columns with NaN values:")
                    for col, count in indicators_with_nans.head().items():
                        print(f"  {col}: {count} NaNs")
                
        except Exception as e:
            print(f"❌ Error with window_size {window_size}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_windowed_calculations()