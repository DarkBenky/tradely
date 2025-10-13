from binance.client import Client
import pandas as pd
import datetime
from ta_indicators import calculate_all_indicators
from tqdm import tqdm

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
    
def save_combined_data_to_files(combined_data, symbols):
    """Save combined OHLCV data and technical indicators to CSV files"""
    print("Saving combined data and indicators to CSV files...")
    
    # Save correlations
    correlations = combined_data.get('correlations', {})
    if correlations:
        corr_df = pd.DataFrame(list(correlations.items()), columns=['Pair', 'Correlation'])
        corr_df.to_csv('correlations.csv', index=False)
        print("Saved correlations.csv")
    
    # Save combined data for each symbol
    for symbol in symbols:
        if symbol in combined_data:
            combined_df = combined_data[symbol]

            # Diagnostic: check node-based indicator columns
            print(f"Diagnostics for {symbol} node indicators:")
            # Volume nodes
            for prefix in ['hvn_level_', 'hvn_strength_', 'lvn_level_']:
                for i in range(1, 6):
                    col = f"{prefix}{i}"
                    if col in combined_df.columns:
                        total = combined_df[col].sum(skipna=True)
                        print(f"  {col}: present, sum={total}")
                    else:
                        print(f"  {col}: MISSING")
            # Time nodes
            for prefix in ['htn_level_', 'htn_strength_', 'ltn_level_']:
                for i in range(1, 6):
                    col = f"{prefix}{i}"
                    if col in combined_df.columns:
                        total = combined_df[col].sum(skipna=True)
                        print(f"  {col}: present, sum={total}")
                    else:
                        print(f"  {col}: MISSING")
            # Trade nodes
            for prefix in ['trade_htn_level_', 'trade_htn_strength_', 'trade_ltn_level_']:
                for i in range(1, 6):
                    col = f"{prefix}{i}"
                    if col in combined_df.columns:
                        total = combined_df[col].sum(skipna=True)
                        print(f"  {col}: present, sum={total}")
                    else:
                        print(f"  {col}: MISSING")
            # Profiles
            for col in ['volume_profile', 'time_profile', 'trade_profile']:
                if col in combined_df.columns:
                    total = combined_df[col].sum(skipna=True)
                    print(f"  {col}: present, sum={total}")
                else:
                    print(f"  {col}: MISSING")
            # Save to file with both OHLCV and indicators
            filename = f"data/{symbol}_combined_data.csv"
            combined_df.to_csv(filename, index=False)
            print(f"Saved {filename} with {len(combined_df.columns)} columns ({len(combined_df)} rows)")
            
            # Also save as _complete_data.csv for backward compatibility
            complete_filename = f"data/{symbol}_complete_data.csv"
            combined_df.to_csv(complete_filename, index=False)
            
            # Also save just the raw OHLCV data for backward compatibility
            ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                         'close_time', 'quote_asset_volume', 'number_of_trades',
                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            if all(col in combined_df.columns for col in ohlcv_cols):
                raw_filename = f"data/{symbol}_5min_data.csv"
                combined_df[ohlcv_cols].to_csv(raw_filename, index=False)
                print(f"Saved {raw_filename} (raw OHLCV data only)")



if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT", "AVAXUSDT", "SHIBUSDT", "TRXUSDT", "UNIUSDT", "ATOMUSDT", "LINKUSDT", "XLMUSDT", "ETCUSDT", "FILUSDT", "ICPUSDT"]
    interval = Client.KLINE_INTERVAL_5MINUTE
    
    print("Starting data collection...")
    all_data = {}
    
    # Use tqdm for progress bar
    for symbol in tqdm(symbols, desc="Fetching data", unit="symbol"):
        df = fetch_historical_data(symbol, interval, years=1)
        if df is not None:
            all_data[symbol] = df
    
    print("Raw data fetched, proceeding to calculate indicators...")
    
    # Calculate all technical indicators
    print("Calculating technical indicators...")
    combined_data = calculate_all_indicators(all_data, window_size=12*4, multiple_returns=5)
    
    # Save combined data to files
    save_combined_data_to_files(combined_data, list(all_data.keys()))
    
    print(f"Data collection and analysis complete! Processed {len(all_data)} symbols.")

