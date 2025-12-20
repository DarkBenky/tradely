import yfinance as yf
import numpy as np
import json
import os

CryptoTickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'BNB-USD', 'XLM-USD', 'DOGE-USD', 'SOL-USD', 'MATIC-USD', 'AVAX-USD', 'SHIB-USD']

TIMEFRAMES = [
    {'interval': '5m', 'period': '60d'},
    {'interval': '1h', 'period': '730d'},
    {'interval': '4h', 'period': '730d'},
    {'interval': '1d', 'period': 'max'}
]

os.makedirs('tickerData', exist_ok=True)

tickers = json.loads(open('tickers.json').read())
crypto_tickers = [{'Symbol': symbol} for symbol in CryptoTickers]
all_tickers = tickers + crypto_tickers

for i, ticker in enumerate(all_tickers):
    for timeframe in TIMEFRAMES:
        try:
            t = ticker['Symbol']
            interval = timeframe['interval']
            period = timeframe['period']
            
            print(f"Processing {i+1}/{len(all_tickers)}: {t} [{interval}]")
            t = yf.Ticker(t)
            hist = t.history(interval=interval, period=period)
            
            if len(hist) < 2:
                continue
            
            close_pct = hist['Close'].pct_change(fill_method=None) * 100
            open_pct = hist['Open'].pct_change(fill_method=None) * 100
            high_pct = hist['High'].pct_change(fill_method=None) * 100
            low_pct = hist['Low'].pct_change(fill_method=None) * 100
            volume_pct = hist['Volume'].pct_change(fill_method=None) * 100
            
            data_to_save = {
                'Open': {str(k): v for k, v in open_pct.to_dict().items() if not (np.isnan(v) or np.isinf(v))},
                'High': {str(k): v for k, v in high_pct.to_dict().items() if not (np.isnan(v) or np.isinf(v))},
                'Low': {str(k): v for k, v in low_pct.to_dict().items() if not (np.isnan(v) or np.isinf(v))},
                'Close': {str(k): v for k, v in close_pct.to_dict().items() if not (np.isnan(v) or np.isinf(v))},
                'Volume': {str(k): v for k, v in volume_pct.to_dict().items() if not (np.isnan(v) or np.isinf(v))}
            }

            filename = f"{ticker['Symbol']}_{interval.replace('h', 'H').replace('d', 'D')}.json"
            with open(f"tickerData/{filename}", "w") as f:
                json.dump(data_to_save, f)
            print(f"Saved {filename} with {len(data_to_save['Close'])} records")
        except Exception as e:
            print(f"Error processing {ticker['Symbol']} [{interval}]: {e}")

print("\nData download complete!")