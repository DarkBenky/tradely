import json
import numpy as np
import os
from typing import Tuple, List

ASSETS = 10
WINDOW_SIZE = 128
HORIZON = 128
FEATURES = 4

def get_top_assets(n_assets: int = 10) -> List[str]:
    file_list = [f for f in os.listdir('tickerData') if f.endswith('_1H.json')]
    
    asset_scores = []
    for file in file_list:
        try:
            with open(os.path.join('tickerData', file), 'r') as f:
                data = json.load(f)
            if len(data['Close']) > WINDOW_SIZE + HORIZON:
                asset_scores.append((file, len(data['Close'])))
        except:
            continue
    
    asset_scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in asset_scores[:n_assets]]

def load_price_data(filename: str) -> np.ndarray:
    with open(os.path.join('tickerData', filename), 'r') as f:
        data = json.load(f)
    
    min_len = min(len(data['Close']), len(data['High']), len(data['Low']), len(data['Volume']))
    
    close = np.array(list(data['Close'].values())[:min_len])
    high = np.array(list(data['High'].values())[:min_len])
    low = np.array(list(data['Low'].values())[:min_len])
    volume = np.array(list(data['Volume'].values())[:min_len])
    
    close_cumsum = np.cumsum(close)
    high_cumsum = np.cumsum(high)
    low_cumsum = np.cumsum(low)
    
    close_cumsum = np.clip(close_cumsum, -100, 100)
    high_cumsum = np.clip(high_cumsum, -100, 100)
    low_cumsum = np.clip(low_cumsum, -100, 100)
    
    prices = np.exp(close_cumsum / 100)
    highs = np.exp(high_cumsum / 100)
    lows = np.exp(low_cumsum / 100)
    volumes = np.abs(volume)
    
    return np.stack([prices, highs, lows, volumes], axis=1)

def calculate_features(prices: np.ndarray, index_prices: np.ndarray, 
                       market_caps: np.ndarray) -> np.ndarray:
    T = prices.shape[0]
    features = np.zeros((T, FEATURES))
    
    log_returns = np.zeros(T)
    log_returns[1:] = np.log(prices[1:, 0] / prices[:-1, 0])
    
    index_log_returns = np.zeros(T)
    index_log_returns[1:] = np.log(index_prices[1:, 0] / index_prices[:-1, 0])
    
    relative_log_returns = log_returns - index_log_returns
    
    rolling_vol = np.zeros(T)
    for i in range(20, T):
        rolling_vol[i] = np.std(log_returns[i-20:i])
    
    total_cap = market_caps.sum()
    weights = market_caps / total_cap if total_cap > 0 else np.ones_like(market_caps) / len(market_caps)
    
    features[:, 0] = log_returns
    features[:, 1] = relative_log_returns
    features[:, 2] = rolling_vol
    features[:, 3] = weights[0]
    
    return features

def normalize_features(x: np.ndarray) -> np.ndarray:
    batch, assets, time, features = x.shape
    
    for f in range(features):
        for a in range(assets):
            mean = x[:, a, :, f].mean()
            std = x[:, a, :, f].std()
            if std > 0:
                x[:, a, :, f] = (x[:, a, :, f] - mean) / std
    
    for t in range(time):
        mean = x[:, :, t, :].mean(axis=1, keepdims=True)
        x[:, :, t, :] -= mean
    
    return x

def generate_labels(all_prices: np.ndarray, window_end: int, 
                    horizon: int) -> np.ndarray:
    current_prices = all_prices[:, window_end, 0]
    future_prices = all_prices[:, min(window_end + horizon, all_prices.shape[1] - 1), 0]
    
    future_returns = np.log(future_prices / current_prices)
    
    index_return = future_returns[0]
    relative_returns = future_returns[1:] - index_return
    
    relative_returns = relative_returns - relative_returns.mean()
    
    return relative_returns

def create_cross_asset_dataset(batch_size: int, index_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
    top_assets = get_top_assets(ASSETS + 1)
    
    if index_file is None:
        index_file = top_assets[0]
        asset_files = top_assets[1:ASSETS+1]
    else:
        asset_files = top_assets[:ASSETS]
    
    index_prices = load_price_data(index_file)
    asset_prices_list = [load_price_data(f) for f in asset_files]
    
    min_length = min([p.shape[0] for p in [index_prices] + asset_prices_list])
    
    index_prices = index_prices[:min_length]
    asset_prices_list = [p[:min_length] for p in asset_prices_list]
    
    all_prices = np.stack([index_prices] + asset_prices_list, axis=0)
    
    max_start = min_length - WINDOW_SIZE - HORIZON
    if max_start <= 0:
        raise ValueError(f"Not enough data: min_length={min_length}, need {WINDOW_SIZE + HORIZON}")
    
    X_batch = []
    y_batch = []
    
    market_caps = np.random.uniform(1e9, 1e12, ASSETS + 1)
    
    for _ in range(batch_size):
        start_idx = np.random.randint(0, max_start)
        end_idx = start_idx + WINDOW_SIZE
        
        window_prices = all_prices[:, start_idx:end_idx, :]
        
        features_list = []
        for i in range(ASSETS + 1):
            feats = calculate_features(
                window_prices[i],
                window_prices[0],
                market_caps
            )
            features_list.append(feats)
        
        sample_features = np.stack(features_list, axis=0)
        X_batch.append(sample_features)
        
        labels = generate_labels(all_prices, end_idx - 1, HORIZON)
        y_batch.append(labels)
    
    X = np.array(X_batch)
    y = np.array(y_batch)
    
    X = normalize_features(X)
    
    return X.astype(np.float32), y.astype(np.float32)
