from binance.client import Client
import pandas as pd
import datetime
from ta_indicators import calculate_all_indicators
from tqdm import tqdm
import gc
import os

def fetch_historical_data(symbol, interval, years=3):
    try:
        client = Client()

        print(f"Fetching data for {symbol}...")
    
        klines = client.get_historical_klines(symbol, interval, start_str=f"{years} year ago UTC")

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])


        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df = df.astype({
            "open": "float64", "high": "float64", "low": "float64",
            "close": "float64", "volume": "float64",
            "quote_asset_volume": "float64", "number_of_trades": "int64",
            "taker_buy_base_asset_volume": "float64",
            "taker_buy_quote_asset_volume": "float64"
        })
        print(f"Successfully fetched {len(df)} records for {symbol}")
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


def process_symbol_individually(symbol, interval, years=1, window_size=12*4, multiple_returns=5):
    print(f"\n{'='*60}")
    print(f"Processing {symbol}...")
    print(f"{'='*60}")
    
    df = fetch_historical_data(symbol, interval, years=years)
    if df is None:
        print(f"Skipping {symbol} due to fetch error")
        return None
    
    print(f"Calculating technical indicators for {symbol}...")
    temp_data = {symbol: df}
    combined_data = calculate_all_indicators(temp_data, window_size=window_size, multiple_returns=multiple_returns)
    
    if symbol in combined_data:
        combined_df = combined_data[symbol]
        
        filename = f"data/{symbol}_combined_data.csv"
        combined_df.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(combined_df.columns)} columns ({len(combined_df)} rows)")
        
        complete_filename = f"data/{symbol}_complete_data.csv"
        combined_df.to_csv(complete_filename, index=True)
        print(f"Saved {complete_filename}")
        
        ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                     'close_time', 'quote_asset_volume', 'number_of_trades',
                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        if all(col in combined_df.columns for col in ohlcv_cols):
            raw_filename = f"data/{symbol}_5min_data.csv"
            combined_df[ohlcv_cols].to_csv(raw_filename, index=False)
            print(f"Saved {raw_filename} (raw OHLCV data only)")
        
        del df, temp_data, combined_data, combined_df
        gc.collect()
        
        return symbol
    else:
        print(f"Failed to process {symbol}")
        return None


if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", 
               "DOGEUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT", "AVAXUSDT"]
    interval = Client.KLINE_INTERVAL_5MINUTE
    
    print("="*60)
    print("MEMORY-OPTIMIZED DATA COLLECTION")
    print("="*60)
    print(f"Processing {len(symbols)} symbols individually to reduce memory usage")
    print("Each symbol will be: fetched → processed → saved → memory freed")
    print()
    
    successful = []
    failed = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        result = process_symbol_individually(symbol, interval, years=3, window_size=12*4, multiple_returns=5)
        
        if result:
            successful.append(result)
        else:
            failed.append(symbol)
    
    print("\n" + "="*60)
    print("Aligning timestamps across all assets...")
    print("="*60)
    
    if len(successful) > 1:
        print("Finding common timestamps...")
        common_timestamps = None
        
        for symbol in successful:
            filename = f"data/{symbol}_combined_data.csv"
            df = pd.read_csv(filename, usecols=['timestamp'])
            timestamps_set = set(df['timestamp'].values)
            
            if common_timestamps is None:
                common_timestamps = timestamps_set
            else:
                common_timestamps = common_timestamps.intersection(timestamps_set)
            
            del df
            gc.collect()
        
        common_timestamps = sorted(list(common_timestamps))
        print(f"Found {len(common_timestamps)} common timestamps across all assets")
        
        print("\nAligning each asset to common timestamps...")
        for symbol in successful:
            filename = f"data/{symbol}_combined_data.csv"
            df = pd.read_csv(filename)
            
            original_len = len(df)
            df = df[df['timestamp'].isin(common_timestamps)]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df.to_csv(filename, index=False)
            print(f"{symbol}: {original_len} -> {len(df)} rows")
            
            complete_filename = f"data/{symbol}_complete_data.csv"
            df.to_csv(complete_filename, index=True)
            
            ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                         'close_time', 'quote_asset_volume', 'number_of_trades',
                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            if all(col in df.columns for col in ohlcv_cols):
                raw_filename = f"data/{symbol}_5min_data.csv"
                df[ohlcv_cols].to_csv(raw_filename, index=False)
            
            del df
            gc.collect()
        
        print(f"\nAll assets aligned to {len(common_timestamps)} common timestamps")
    
    print("\n" + "="*60)
    print("Calculating inter-symbol correlations...")
    print("="*60)
    
    try:
        correlation_data = {}
        for symbol in successful:
            filename = f"data/{symbol}_combined_data.csv"
            df = pd.read_csv(filename, usecols=['close'])
            correlation_data[symbol] = df['close']
        
        if len(correlation_data) > 1:
            corr_df_data = pd.DataFrame(correlation_data)
            correlations = corr_df_data.corr()
            
            corr_pairs = []
            for i in range(len(correlations.columns)):
                for j in range(i+1, len(correlations.columns)):
                    pair = f"{correlations.columns[i]}-{correlations.columns[j]}"
                    corr_value = correlations.iloc[i, j]
                    corr_pairs.append({'Pair': pair, 'Correlation': corr_value})
            
            corr_result = pd.DataFrame(corr_pairs)
            corr_result.to_csv('correlations.csv', index=False)
            print(f"Saved correlations.csv ({len(corr_pairs)} pairs)")
        
        del correlation_data, corr_df_data, correlations
    except Exception as e:
        print(f"Error calculating correlations: {e}")
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"Successfully processed: {len(successful)} symbols")
    if successful:
        print(f"  {', '.join(successful)}")
    if failed:
        print(f"Failed: {len(failed)} symbols")
        print(f"  {', '.join(failed)}")
    print(f"\nTotal: {len(successful)}/{len(symbols)} symbols completed")
    print("="*60)

