import pandas as pd
import os
import numpy as np

def load_and_align_data(data_dir='data'):
    data_files = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('_combined_data.csv'):
            symbol = filename.replace('_combined_data.csv', '')
            if symbol == 'MATICUSDT':  # Skip empty dataframes
                continue
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            data_files[symbol] = df
    
    print(f"Loaded {len(data_files)} data files")
    
    # Find common timestamps across all symbols
    common_timestamps = None
    for symbol, df in data_files.items():
        if common_timestamps is None:
            common_timestamps = set(df.index)
        else:
            common_timestamps = common_timestamps.intersection(set(df.index))
    
    common_timestamps = sorted(list(common_timestamps))
    print(f"Found {len(common_timestamps)} common timestamps")
    
    # Align each dataframe to common timestamps and remove timestamp index
    aligned_data = {}
    for symbol, df in data_files.items():
        aligned_df = df.loc[common_timestamps].reset_index(drop=True)
        aligned_data[symbol] = aligned_df
    
    return aligned_data


# Load and align data
print("Loading and aligning data...")
data = load_and_align_data()

print(f"\nAligned data ready for training:")
for symbol, df in data.items():
    print(f"  {symbol}: {len(df)} rows, {len(df.columns)} columns")
    print(data[symbol].head(1))