import yfinance as yf
import numpy as np
import json
import os

INTERVAL = '1h'
BUCKETS = 24

CryptoTickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'BNB-USD', 'XLM-USD', 'DOGE-USD', 'SOL-USD', 'MATIC-USD', 'AVAX-USD', 'SHIB-USD']

all_close = []
all_open = []
all_high = []
all_low = []
all_volume = []

os.makedirs('tickerData', exist_ok=True)

tickers = json.loads(open('tickers.json').read())
crypto_tickers = [{'Symbol': symbol} for symbol in CryptoTickers]
all_tickers = tickers + crypto_tickers

for i, ticker in enumerate(all_tickers):
    try:
        t = ticker['Symbol']
        print(f"Processing {i+1}/{len(all_tickers)}: {t}")
        t = yf.Ticker(t)
        hist = t.history(interval=INTERVAL, period='730d')
        
        if len(hist) < 2:
            continue
            
        close_changes = hist['Close'].pct_change(fill_method=None) * 100
        open_changes = hist['Open'].pct_change(fill_method=None) * 100
        high_changes = hist['High'].pct_change(fill_method=None) * 100
        low_changes = hist['Low'].pct_change(fill_method=None) * 100
        volume_changes = hist['Volume'].pct_change(fill_method=None) * 100
        
        close_changes = close_changes.replace([np.inf, -np.inf], np.nan)
        open_changes = open_changes.replace([np.inf, -np.inf], np.nan)
        high_changes = high_changes.replace([np.inf, -np.inf], np.nan)
        low_changes = low_changes.replace([np.inf, -np.inf], np.nan)
        volume_changes = volume_changes.replace([np.inf, -np.inf], np.nan)
        
        all_close.extend(close_changes.dropna().values)
        all_open.extend(open_changes.dropna().values)
        all_high.extend(high_changes.dropna().values)
        all_low.extend(low_changes.dropna().values)
        all_volume.extend(volume_changes.dropna().values)
        
        data_to_save = {
            'Open': {str(k): v for k, v in open_changes.to_dict().items() if not (np.isnan(v) or np.isinf(v))},
            'High': {str(k): v for k, v in high_changes.to_dict().items() if not (np.isnan(v) or np.isinf(v))},
            'Low': {str(k): v for k, v in low_changes.to_dict().items() if not (np.isnan(v) or np.isinf(v))},
            'Close': {str(k): v for k, v in close_changes.to_dict().items() if not (np.isnan(v) or np.isinf(v))},
            'Volume': {str(k): v for k, v in volume_changes.to_dict().items() if not (np.isnan(v) or np.isinf(v))}
        }

        with open(f"tickerData/{ticker['Symbol']}.json", "w") as f:
            json.dump(data_to_save, f)
    except Exception as e:
        print(f"Error processing {ticker['Symbol']}: {e}")

all_close = np.array(all_close)
all_open = np.array(all_open)
all_high = np.array(all_high)
all_low = np.array(all_low)
all_volume = np.array(all_volume)

all_close = all_close[np.isfinite(all_close)]
all_open = all_open[np.isfinite(all_open)]
all_high = all_high[np.isfinite(all_high)]
all_low = all_low[np.isfinite(all_low)]
all_volume = all_volume[np.isfinite(all_volume)]

percentiles = np.linspace(0, 100, BUCKETS + 1)

close_buckets = np.percentile(all_close, percentiles)
open_buckets = np.percentile(all_open, percentiles)
high_buckets = np.percentile(all_high, percentiles)
low_buckets = np.percentile(all_low, percentiles)
volume_buckets = np.percentile(all_volume, percentiles)

bucket_config = {
    'num_buckets': BUCKETS,
    'close': {
        'buckets': close_buckets.tolist(),
        'min': float(np.min(all_close)),
        'max': float(np.max(all_close)),
        'mean': float(np.mean(all_close)),
        'std': float(np.std(all_close)),
        'median': float(np.median(all_close)),
        'q25': float(np.percentile(all_close, 25)),
        'q75': float(np.percentile(all_close, 75))
    },
    'open': {
        'buckets': open_buckets.tolist(),
        'min': float(np.min(all_open)),
        'max': float(np.max(all_open)),
        'mean': float(np.mean(all_open)),
        'std': float(np.std(all_open)),
        'median': float(np.median(all_open)),
        'q25': float(np.percentile(all_open, 25)),
        'q75': float(np.percentile(all_open, 75))
    },
    'high': {
        'buckets': high_buckets.tolist(),
        'min': float(np.min(all_high)),
        'max': float(np.max(all_high)),
        'mean': float(np.mean(all_high)),
        'std': float(np.std(all_high)),
        'median': float(np.median(all_high)),
        'q25': float(np.percentile(all_high, 25)),
        'q75': float(np.percentile(all_high, 75))
    },
    'low': {
        'buckets': low_buckets.tolist(),
        'min': float(np.min(all_low)),
        'max': float(np.max(all_low)),
        'mean': float(np.mean(all_low)),
        'std': float(np.std(all_low)),
        'median': float(np.median(all_low)),
        'q25': float(np.percentile(all_low, 25)),
        'q75': float(np.percentile(all_low, 75))
    },
    'volume': {
        'buckets': volume_buckets.tolist(),
        'min': float(np.min(all_volume)),
        'max': float(np.max(all_volume)),
        'mean': float(np.mean(all_volume)),
        'std': float(np.std(all_volume)),
        'median': float(np.median(all_volume)),
        'q25': float(np.percentile(all_volume, 25)),
        'q75': float(np.percentile(all_volume, 75))
    }
}

with open('bucket_config.json', 'w') as f:
    json.dump(bucket_config, f, indent=2)

print("\n=== Distribution Statistics ===")
print(f"\nClose: mean={bucket_config['close']['mean']:.4f}, std={bucket_config['close']['std']:.4f}")
print(f"Open: mean={bucket_config['open']['mean']:.4f}, std={bucket_config['open']['std']:.4f}")
print(f"High: mean={bucket_config['high']['mean']:.4f}, std={bucket_config['high']['std']:.4f}")
print(f"Low: mean={bucket_config['low']['mean']:.4f}, std={bucket_config['low']['std']:.4f}")
print(f"Volume: mean={bucket_config['volume']['mean']:.4f}, std={bucket_config['volume']['std']:.4f}")
print("\nBucket configuration saved to bucket_config.json")