import pandas as pd
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalysis:
    """
    Technical Analysis indicators with rolling window calculations and smoothing filters
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    def _ensure_same_length(self, series: pd.Series, original_length: int) -> pd.Series:
        """Ensure series has same length as original data by padding with NaN if necessary"""
        if len(series) < original_length:
            # Pad at the beginning with NaN
            padding = pd.Series([np.nan] * (original_length - len(series)))
            return pd.concat([padding, series]).reset_index(drop=True)
        elif len(series) > original_length:
            # Truncate to match original length
            return series.iloc[:original_length].reset_index(drop=True)
        return series.reset_index(drop=True)
    
    def _apply_smoothing(self, data: np.ndarray, method: str = 'gaussian', sigma: float = 1.0) -> np.ndarray:
        """Apply smoothing filter to make discrete data more continuous"""
        if len(data) < 3:
            return data
            
        # Remove NaN values for smoothing
        mask = ~np.isnan(data)
        if not mask.any():
            return data
            
        if method == 'gaussian':
            # Gaussian filter for smooth continuous curves
            smoothed = ndimage.gaussian_filter1d(data[mask], sigma=sigma)
        elif method == 'savgol':
            # Savitzky-Golay filter for preserving features while smoothing
            window_length = min(21, len(data[mask]) if len(data[mask]) % 2 == 1 else len(data[mask]) - 1)
            if window_length >= 3:
                smoothed = savgol_filter(data[mask], window_length, 3)
            else:
                smoothed = data[mask]
        else:
            # Simple moving average
            window = min(5, len(data[mask]))
            smoothed = pd.Series(data[mask]).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        result = data.copy()
        result[mask] = smoothed
        return result

def calculate_correlations(dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Calculate correlations between close price & volume, and between currency pairs"""
    correlations = {}
    
    for symbol, df in dfs.items():
        # Close price vs Volume correlation
        if len(df) > 1:
            corr = df['close'].corr(df['volume'])
            correlations[f"{symbol}_close_volume_corr"] = corr if not np.isnan(corr) else 0.0
    
    # Currency pair correlations
    symbols = list(dfs.keys())
    for i, symbol1 in enumerate(symbols):
        for symbol2 in symbols[i+1:]:
            if symbol1 in dfs and symbol2 in dfs:
                df1, df2 = dfs[symbol1], dfs[symbol2]
                min_len = min(len(df1), len(df2))
                if min_len > 1:
                    close1 = df1['close'].iloc[-min_len:].values
                    close2 = df2['close'].iloc[-min_len:].values
                    corr = np.corrcoef(close1, close2)[0, 1]
                    correlations[f"{symbol1}_{symbol2}_corr"] = corr if not np.isnan(corr) else 0.0
    
    return correlations

def calculate_moving_averages(df: pd.DataFrame, window_size: int = 50, 
                            periods: List[int] = [5, 10, 20, 50]) -> Dict[str, pd.Series]:
    """Calculate various moving averages using rolling window approach"""
    ta = TechnicalAnalysis(window_size)
    mas = {}
    original_length = len(df)
    
    # Filter periods to be within window_size
    valid_periods = [p for p in periods if p <= window_size]
    
    for period in valid_periods:
        if len(df) >= period:
            ma = df['close'].rolling(window=period).mean()
            mas[f'ma_{period}'] = ta._ensure_same_length(ma, original_length)
        else:
            mas[f'ma_{period}'] = pd.Series([np.nan] * original_length)
    
    # Exponential Moving Averages
    for period in valid_periods:
        if len(df) >= period:
            ema = df['close'].ewm(span=period).mean()
            mas[f'ema_{period}'] = ta._ensure_same_length(ema, original_length)
        else:
            mas[f'ema_{period}'] = pd.Series([np.nan] * original_length)
    
    return mas

def calculate_rsi(df: pd.DataFrame, window_size: int = 50, period: int = 14) -> pd.Series:
    """Calculate RSI using rolling window approach"""
    ta = TechnicalAnalysis(window_size)
    original_length = len(df)
    
    if len(df) < period + 1 or period > window_size:
        return pd.Series([np.nan] * original_length)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return ta._ensure_same_length(rsi, original_length)

def calculate_macd(df: pd.DataFrame, window_size: int = 50, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD using rolling window approach"""
    ta = TechnicalAnalysis(window_size)
    original_length = len(df)
    
    # Adjust parameters if they exceed window size
    fast = min(fast, window_size // 3)
    slow = min(slow, window_size // 2)
    signal = min(signal, window_size // 4)
    
    if len(df) < slow + signal or slow > window_size:
        return {
            'macd_line': pd.Series([np.nan] * original_length),
            'macd_signal': pd.Series([np.nan] * original_length),
            'macd_histogram': pd.Series([np.nan] * original_length)
        }
    
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    
    return {
        'macd_line': ta._ensure_same_length(macd_line, original_length),
        'macd_signal': ta._ensure_same_length(macd_signal, original_length),
        'macd_histogram': ta._ensure_same_length(macd_histogram, original_length)
    }

def calculate_bollinger_bands(df: pd.DataFrame, window_size: int = 50, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands using rolling window approach"""
    ta = TechnicalAnalysis(window_size)
    original_length = len(df)
    
    # Adjust period if it exceeds window size
    period = min(period, window_size)
    
    if len(df) < period or period > window_size:
        return {
            'bb_upper': pd.Series([np.nan] * original_length),
            'bb_middle': pd.Series([np.nan] * original_length),
            'bb_lower': pd.Series([np.nan] * original_length)
        }
    
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return {
        'bb_upper': ta._ensure_same_length(upper, original_length),
        'bb_middle': ta._ensure_same_length(middle, original_length),
        'bb_lower': ta._ensure_same_length(lower, original_length)
    }

def calculate_keltner_channels(df: pd.DataFrame, period: int = 20, multiplier: float = 2.0, 
                             target_length: int = 360) -> Dict[str, pd.Series]:
    """Calculate Keltner Channels with fixed length"""
    ta = TechnicalAnalysis()
    
    if len(df) < period:
        return {
            'kc_upper': pd.Series([np.nan] * target_length),
            'kc_middle': pd.Series([np.nan] * target_length),
            'kc_lower': pd.Series([np.nan] * target_length)
        }
    
    middle = df['close'].ewm(span=period).mean()
    atr = calculate_atr(df, period)
    upper = middle + (atr * multiplier)
    lower = middle - (atr * multiplier)
    
    return {
        'kc_upper': ta._ensure_fixed_length(upper, target_length),
        'kc_middle': ta._ensure_fixed_length(middle, target_length),
        'kc_lower': ta._ensure_fixed_length(lower, target_length)
    }

def calculate_atr(df: pd.DataFrame, window_size: int = 50, period: int = 14) -> pd.Series:
    """Calculate Average True Range using rolling window approach"""
    ta = TechnicalAnalysis(window_size)
    original_length = len(df)
    
    # Adjust period if it exceeds window size
    period = min(period, window_size)
    
    if len(df) < 2 or period > window_size:
        return pd.Series([np.nan] * original_length)
    
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift())
    low_close_prev = np.abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return ta._ensure_same_length(atr, original_length)

def calculate_obv(df: pd.DataFrame, target_length: int = 360) -> pd.Series:
    """Calculate On-Balance Volume with fixed length"""
    ta = TechnicalAnalysis()
    
    if len(df) < 2:
        return pd.Series([np.nan] * target_length)
    
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return ta._ensure_fixed_length(obv, target_length)

def calculate_vwap(df: pd.DataFrame, window_size: int = 50) -> pd.Series:
    """Calculate Volume Weighted Average Price using rolling window approach"""
    ta = TechnicalAnalysis(window_size)
    original_length = len(df)
    
    if len(df) < 1:
        return pd.Series([np.nan] * original_length)
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    # Use rolling window for VWAP calculation
    vwap = (typical_price * df['volume']).rolling(window=window_size).sum() / df['volume'].rolling(window=window_size).sum()
    
    return ta._ensure_same_length(vwap, original_length)

def calculate_cci(df: pd.DataFrame, period: int = 20, target_length: int = 360) -> pd.Series:
    """Calculate Commodity Channel Index with fixed length"""
    ta = TechnicalAnalysis()
    
    if len(df) < period:
        return pd.Series([np.nan] * target_length)
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    
    return ta._ensure_fixed_length(cci, target_length)

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, 
                        target_length: int = 360) -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator with fixed length"""
    ta = TechnicalAnalysis()
    
    if len(df) < k_period:
        return {
            'stoch_k': pd.Series([np.nan] * target_length),
            'stoch_d': pd.Series([np.nan] * target_length)
        }
    
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    
    k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return {
        'stoch_k': ta._ensure_fixed_length(k_percent, target_length),
        'stoch_d': ta._ensure_fixed_length(d_percent, target_length)
    }

def calculate_adx(df: pd.DataFrame, period: int = 14, target_length: int = 360) -> Dict[str, pd.Series]:
    """Calculate Average Directional Index with fixed length"""
    ta = TechnicalAnalysis()
    
    if len(df) < period + 1:
        return {
            'adx': pd.Series([np.nan] * target_length),
            'di_plus': pd.Series([np.nan] * target_length),
            'di_minus': pd.Series([np.nan] * target_length)
        }
    
    # Calculate True Range and Directional Movement
    tr = calculate_atr(df, 1)  # True Range for 1 period
    
    dm_plus = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']), 
                       np.maximum(df['high'] - df['high'].shift(), 0), 0)
    dm_minus = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()), 
                        np.maximum(df['low'].shift() - df['low'], 0), 0)
    
    # Smooth the values
    tr_smooth = pd.Series(tr).rolling(window=period).mean()
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=period).mean()
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=period).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)
    
    # Calculate ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()
    
    return {
        'adx': ta._ensure_fixed_length(adx, target_length),
        'di_plus': ta._ensure_fixed_length(di_plus, target_length),
        'di_minus': ta._ensure_fixed_length(di_minus, target_length)
    }

def calculate_fibonacci_retracements(df: pd.DataFrame, target_length: int = 360) -> Dict[str, pd.Series]:
    """Calculate Fibonacci retracement levels with fixed length"""
    ta = TechnicalAnalysis()
    
    if len(df) < 2:
        return {f'fib_{level}': pd.Series([np.nan] * target_length) 
                for level in ['0', '236', '382', '500', '618', '786', '1000']}
    
    # Find swing high and low over rolling window
    window = min(50, len(df))
    rolling_high = df['high'].rolling(window=window).max()
    rolling_low = df['low'].rolling(window=window).min()
    
    fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_names = ['0', '236', '382', '500', '618', '786', '1000']
    
    result = {}
    for level, name in zip(fib_levels, fib_names):
        fib_price = rolling_low + (rolling_high - rolling_low) * level
        result[f'fib_{name}'] = ta._ensure_fixed_length(fib_price, target_length)
    
    return result

def calculate_pivot_points(df: pd.DataFrame, target_length: int = 360) -> Dict[str, pd.Series]:
    """Calculate Pivot Points with fixed length"""
    ta = TechnicalAnalysis()
    
    if len(df) < 1:
        return {
            'pivot': pd.Series([np.nan] * target_length),
            'r1': pd.Series([np.nan] * target_length),
            'r2': pd.Series([np.nan] * target_length),
            's1': pd.Series([np.nan] * target_length),
            's2': pd.Series([np.nan] * target_length)
        }
    
    # Calculate pivot point (yesterday's HLC average)
    pivot = (df['high'].shift() + df['low'].shift() + df['close'].shift()) / 3
    
    # Resistance and Support levels
    r1 = 2 * pivot - df['low'].shift()
    r2 = pivot + (df['high'].shift() - df['low'].shift())
    s1 = 2 * pivot - df['high'].shift()
    s2 = pivot - (df['high'].shift() - df['low'].shift())
    
    return {
        'pivot': ta._ensure_fixed_length(pivot, target_length),
        'r1': ta._ensure_fixed_length(r1, target_length),
        'r2': ta._ensure_fixed_length(r2, target_length),
        's1': ta._ensure_fixed_length(s1, target_length),
        's2': ta._ensure_fixed_length(s2, target_length)
    }

def calculate_ichimoku_cloud(df: pd.DataFrame, target_length: int = 360) -> Dict[str, pd.Series]:
    """Calculate Ichimoku Cloud with fixed length"""
    ta = TechnicalAnalysis()
    
    if len(df) < 52:
        return {
            'tenkan_sen': pd.Series([np.nan] * target_length),
            'kijun_sen': pd.Series([np.nan] * target_length),
            'senkou_span_a': pd.Series([np.nan] * target_length),
            'senkou_span_b': pd.Series([np.nan] * target_length),
            'chikou_span': pd.Series([np.nan] * target_length)
        }
    
    # Tenkan-sen (Conversion Line): 9-period high-low average
    tenkan_sen = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    
    # Kijun-sen (Base Line): 26-period high-low average
    kijun_sen = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    
    # Senkou Span A (Leading Span A): Average of Tenkan and Kijun, shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): 52-period high-low average, shifted 26 periods ahead
    senkou_span_b = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Close price shifted 26 periods back
    chikou_span = df['close'].shift(-26)
    
    return {
        'tenkan_sen': ta._ensure_fixed_length(tenkan_sen, target_length),
        'kijun_sen': ta._ensure_fixed_length(kijun_sen, target_length),
        'senkou_span_a': ta._ensure_fixed_length(senkou_span_a, target_length),
        'senkou_span_b': ta._ensure_fixed_length(senkou_span_b, target_length),
        'chikou_span': ta._ensure_fixed_length(chikou_span, target_length)
    }

def calculate_heikin_ashi(df: pd.DataFrame, target_length: int = 360) -> Dict[str, pd.Series]:
    """Calculate Heikin-Ashi candles with fixed length"""
    ta = TechnicalAnalysis()
    
    if len(df) < 1:
        return {
            'ha_open': pd.Series([np.nan] * target_length),
            'ha_high': pd.Series([np.nan] * target_length),
            'ha_low': pd.Series([np.nan] * target_length),
            'ha_close': pd.Series([np.nan] * target_length)
        }
    
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    
    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)
    
    return {
        'ha_open': ta._ensure_fixed_length(ha_open, target_length),
        'ha_high': ta._ensure_fixed_length(ha_high, target_length),
        'ha_low': ta._ensure_fixed_length(ha_low, target_length),
        'ha_close': ta._ensure_fixed_length(ha_close, target_length)
    }

def calculate_renko_charts(df: pd.DataFrame, brick_size: float = None, target_length: int = 360) -> Dict[str, pd.Series]:
    """Calculate Renko chart data with fixed length"""
    ta = TechnicalAnalysis()
    
    if len(df) < 2:
        return {
            'renko_open': pd.Series([np.nan] * target_length),
            'renko_close': pd.Series([np.nan] * target_length),
            'renko_direction': pd.Series([np.nan] * target_length)
        }
    
    if brick_size is None:
        # Auto-calculate brick size as 1% of average price
        brick_size = df['close'].mean() * 0.01
    
    renko_prices = [df['close'].iloc[0]]
    renko_directions = [1]  # 1 for up, -1 for down
    
    current_brick = df['close'].iloc[0]
    
    for price in df['close'].iloc[1:]:
        if price >= current_brick + brick_size:
            # Up brick(s)
            while price >= current_brick + brick_size:
                current_brick += brick_size
                renko_prices.append(current_brick)
                renko_directions.append(1)
        elif price <= current_brick - brick_size:
            # Down brick(s)
            while price <= current_brick - brick_size:
                current_brick -= brick_size
                renko_prices.append(current_brick)
                renko_directions.append(-1)
    
    # Create series aligned with original data
    renko_open = pd.Series([renko_prices[0]] + renko_prices[:-1], index=df.index[:len(renko_prices)])
    renko_close = pd.Series(renko_prices, index=df.index[:len(renko_prices)])
    renko_direction = pd.Series(renko_directions, index=df.index[:len(renko_directions)])
    
    # Extend to match original dataframe length
    if len(renko_close) < len(df):
        last_price = renko_close.iloc[-1] if len(renko_close) > 0 else df['close'].iloc[0]
        last_direction = renko_direction.iloc[-1] if len(renko_direction) > 0 else 1
        
        remaining_index = df.index[len(renko_close):]
        renko_open = pd.concat([renko_open, pd.Series([last_price] * len(remaining_index), index=remaining_index)])
        renko_close = pd.concat([renko_close, pd.Series([last_price] * len(remaining_index), index=remaining_index)])
        renko_direction = pd.concat([renko_direction, pd.Series([last_direction] * len(remaining_index), index=remaining_index)])
    
    return {
        'renko_open': ta._ensure_fixed_length(renko_open, target_length),
        'renko_close': ta._ensure_fixed_length(renko_close, target_length),
        'renko_direction': ta._ensure_fixed_length(renko_direction, target_length)
    }

def calculate_volume_nodes(df: pd.DataFrame, window_size: int = 50, num_bins: int = 25, 
                          smoothing_method: str = 'gaussian', sigma: float = 2.0, multiple_returns: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate High Volume Nodes (HVN) and Low Volume Nodes (LVN) with smoothing filters
    using rolling window approach to make them more continuous rather than discrete
    """
    ta = TechnicalAnalysis(window_size)
    original_length = len(df)
    
    if len(df) < 2:
        return {
            'hvn_levels': pd.Series([np.nan] * original_length),
            'lvn_levels': pd.Series([np.nan] * original_length),
            'volume_profile': pd.Series([np.nan] * original_length),
            'hvn_strength': pd.Series([np.nan] * original_length)
        }
    
    # Create price bins
    price_min, price_max = df['low'].min(), df['high'].max()
    price_bins = np.linspace(price_min, price_max, num_bins)
    
    # Calculate volume at each price level
    volume_profile = np.zeros(len(price_bins))
    
    for i in range(len(df)):
        # For each candle, distribute volume across price range
        low, high, volume = df['low'].iloc[i], df['high'].iloc[i], df['volume'].iloc[i]
        
        # Find bins that overlap with this candle's price range
        bin_mask = (price_bins >= low) & (price_bins <= high)
        if bin_mask.any():
            # Distribute volume evenly across overlapping bins
            volume_per_bin = volume / bin_mask.sum()
            volume_profile[bin_mask] += volume_per_bin
    
    # Apply smoothing to make volume profile continuous
    volume_profile_smooth = ta._apply_smoothing(volume_profile, method=smoothing_method, sigma=sigma)
    
    # Identify HVN and LVN
    # HVN: peaks in volume profile (high volume areas)
    # LVN: valleys in volume profile (low volume areas)
    
    # Find local maxima and minima
    from scipy.signal import find_peaks
    
    hvn_indices, _ = find_peaks(volume_profile_smooth, height=np.percentile(volume_profile_smooth, 70))
    lvn_indices, _ = find_peaks(-volume_profile_smooth, height=-np.percentile(volume_profile_smooth, 30))
    
    # Create time series aligned with original data
    hvn_levels = pd.Series(index=df.index, dtype=float)
    lvn_levels = pd.Series(index=df.index, dtype=float)
    volume_profile_series = pd.Series(index=df.index, dtype=float)
    hvn_strength = pd.Series(index=df.index, dtype=float)
    
    # Get top HVN and LVN levels
    hvn_prices = price_bins[hvn_indices] if len(hvn_indices) > 0 else np.array([])
    hvn_strengths = volume_profile_smooth[hvn_indices] if len(hvn_indices) > 0 else np.array([])
    lvn_prices = price_bins[lvn_indices] if len(lvn_indices) > 0 else np.array([])
    lvn_strengths = volume_profile_smooth[lvn_indices] if len(lvn_indices) > 0 else np.array([])
    
    # Sort by strength and get top N
    if len(hvn_prices) > 0:
        hvn_sorted_idx = np.argsort(hvn_strengths)[::-1]  # Descending order
        hvn_prices = hvn_prices[hvn_sorted_idx][:multiple_returns]
        hvn_strengths = hvn_strengths[hvn_sorted_idx][:multiple_returns]
    
    if len(lvn_prices) > 0:
        lvn_sorted_idx = np.argsort(lvn_strengths)  # Ascending order (weakest first for LVN)
        lvn_prices = lvn_prices[lvn_sorted_idx][:multiple_returns]
        lvn_strengths = lvn_strengths[lvn_sorted_idx][:multiple_returns]
    
    # Create multiple columns for each return
    result = {}
    
    # Volume profile for all rows
    volume_profile_series = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        price_bin_idx = np.argmin(np.abs(price_bins - current_price))
        volume_profile_series.iloc[i] = volume_profile_smooth[price_bin_idx]
    result['volume_profile'] = ta._ensure_same_length(volume_profile_series, original_length)
    
    # Create multiple HVN and LVN columns
    for n in range(multiple_returns):
        hvn_series = pd.Series([np.nan] * original_length)
        hvn_strength_series = pd.Series([np.nan] * original_length)
        lvn_series = pd.Series([np.nan] * original_length)
        
        if n < len(hvn_prices):
            hvn_series[:] = hvn_prices[n]  # Same level for all rows
            hvn_strength_series[:] = hvn_strengths[n]
        
        if n < len(lvn_prices):
            lvn_series[:] = lvn_prices[n]  # Same level for all rows
            
        result[f'hvn_level_{n+1}'] = hvn_series
        result[f'hvn_strength_{n+1}'] = hvn_strength_series
        result[f'lvn_level_{n+1}'] = lvn_series
    
    return result

def calculate_time_nodes(df: pd.DataFrame, window_size: int = 50, num_bins: int = 25, 
                        smoothing_method: str = 'gaussian', sigma: float = 2.0, multiple_returns: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate High Time Nodes (HTN) and Low Time Nodes (LTN) - time spent at price levels
    with smoothing filters using rolling window approach to make them more continuous
    """
    ta = TechnicalAnalysis(window_size)
    original_length = len(df)
    
    if len(df) < 2:
        return {
            'htn_levels': pd.Series([np.nan] * original_length),
            'ltn_levels': pd.Series([np.nan] * original_length),
            'time_profile': pd.Series([np.nan] * original_length),
            'htn_strength': pd.Series([np.nan] * original_length)
        }
    
    # Create price bins
    price_min, price_max = df['low'].min(), df['high'].max()
    price_bins = np.linspace(price_min, price_max, num_bins)
    
    # Calculate time spent at each price level
    time_profile = np.zeros(len(price_bins))
    
    for i in range(len(df)):
        # For each candle, add time spent across price range
        low, high = df['low'].iloc[i], df['high'].iloc[i]
        
        # Find bins that overlap with this candle's price range
        bin_mask = (price_bins >= low) & (price_bins <= high)
        if bin_mask.any():
            # Add 1 time unit to each overlapping bin (could weight by OHLC)
            time_profile[bin_mask] += 1
    
    # Apply smoothing to make time profile continuous
    time_profile_smooth = ta._apply_smoothing(time_profile, method=smoothing_method, sigma=sigma)
    
    # Find High Time Nodes and Low Time Nodes
    from scipy.signal import find_peaks
    
    htn_indices, _ = find_peaks(time_profile_smooth, height=np.percentile(time_profile_smooth, 70))
    ltn_indices, _ = find_peaks(-time_profile_smooth, height=-np.percentile(time_profile_smooth, 30))
    
    # Create time series aligned with original data
    htn_levels = pd.Series(index=df.index, dtype=float)
    ltn_levels = pd.Series(index=df.index, dtype=float)
    time_profile_series = pd.Series(index=df.index, dtype=float)
    htn_strength = pd.Series(index=df.index, dtype=float)
    
    # Get top HTN and LTN levels
    htn_prices = price_bins[htn_indices] if len(htn_indices) > 0 else np.array([])
    htn_strengths = time_profile_smooth[htn_indices] if len(htn_indices) > 0 else np.array([])
    ltn_prices = price_bins[ltn_indices] if len(ltn_indices) > 0 else np.array([])
    ltn_strengths = time_profile_smooth[ltn_indices] if len(ltn_indices) > 0 else np.array([])
    
    # Sort by strength and get top N
    if len(htn_prices) > 0:
        htn_sorted_idx = np.argsort(htn_strengths)[::-1]  # Descending order
        htn_prices = htn_prices[htn_sorted_idx][:multiple_returns]
        htn_strengths = htn_strengths[htn_sorted_idx][:multiple_returns]
    
    if len(ltn_prices) > 0:
        ltn_sorted_idx = np.argsort(ltn_strengths)  # Ascending order (weakest first for LTN)
        ltn_prices = ltn_prices[ltn_sorted_idx][:multiple_returns]
        ltn_strengths = ltn_strengths[ltn_sorted_idx][:multiple_returns]
    
    # Create multiple columns for each return
    result = {}
    
    # Time profile for all rows
    time_profile_series = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        price_bin_idx = np.argmin(np.abs(price_bins - current_price))
        time_profile_series.iloc[i] = time_profile_smooth[price_bin_idx]
    result['time_profile'] = ta._ensure_same_length(time_profile_series, original_length)
    
    # Create multiple HTN and LTN columns
    for n in range(multiple_returns):
        htn_series = pd.Series([np.nan] * original_length)
        htn_strength_series = pd.Series([np.nan] * original_length)
        ltn_series = pd.Series([np.nan] * original_length)
        
        if n < len(htn_prices):
            htn_series[:] = htn_prices[n]  # Same level for all rows
            htn_strength_series[:] = htn_strengths[n]
        
        if n < len(ltn_prices):
            ltn_series[:] = ltn_prices[n]  # Same level for all rows
            
        result[f'htn_level_{n+1}'] = htn_series
        result[f'htn_strength_{n+1}'] = htn_strength_series
        result[f'ltn_level_{n+1}'] = ltn_series
    
    return result

def calculate_trade_nodes(df: pd.DataFrame, window_size: int = 50, num_bins: int = 25, 
                         smoothing_method: str = 'gaussian', sigma: float = 2.0, multiple_returns: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate High Trade Nodes (HTN) and Low Trade Nodes (LTN) - number of trades at price levels
    with smoothing filters using rolling window approach to make them more continuous
    """
    ta = TechnicalAnalysis(window_size)
    original_length = len(df)
    
    if len(df) < 2 or 'number_of_trades' not in df.columns:
        return {
            'trade_htn_levels': pd.Series([np.nan] * original_length),
            'trade_ltn_levels': pd.Series([np.nan] * original_length),
            'trade_profile': pd.Series([np.nan] * original_length),
            'trade_htn_strength': pd.Series([np.nan] * original_length)
        }
    
    # Create price bins
    price_min, price_max = df['low'].min(), df['high'].max()
    price_bins = np.linspace(price_min, price_max, num_bins)
    
    # Calculate number of trades at each price level
    trade_profile = np.zeros(len(price_bins))
    
    for i in range(len(df)):
        # For each candle, distribute trades across price range
        low, high, trades = df['low'].iloc[i], df['high'].iloc[i], df['number_of_trades'].iloc[i]
        
        # Find bins that overlap with this candle's price range
        bin_mask = (price_bins >= low) & (price_bins <= high)
        if bin_mask.any():
            # Distribute trades evenly across overlapping bins
            trades_per_bin = trades / bin_mask.sum()
            trade_profile[bin_mask] += trades_per_bin
    
    # Apply smoothing to make trade profile continuous
    trade_profile_smooth = ta._apply_smoothing(trade_profile, method=smoothing_method, sigma=sigma)
    
    # Find High Trade Nodes and Low Trade Nodes
    from scipy.signal import find_peaks
    
    trade_htn_indices, _ = find_peaks(trade_profile_smooth, height=np.percentile(trade_profile_smooth, 70))
    trade_ltn_indices, _ = find_peaks(-trade_profile_smooth, height=-np.percentile(trade_profile_smooth, 30))
    
    # Create time series aligned with original data
    trade_htn_levels = pd.Series(index=df.index, dtype=float)
    trade_ltn_levels = pd.Series(index=df.index, dtype=float)
    trade_profile_series = pd.Series(index=df.index, dtype=float)
    trade_htn_strength = pd.Series(index=df.index, dtype=float)
    
    # Get top Trade HTN and LTN levels
    trade_htn_prices = price_bins[trade_htn_indices] if len(trade_htn_indices) > 0 else np.array([])
    trade_htn_strengths = trade_profile_smooth[trade_htn_indices] if len(trade_htn_indices) > 0 else np.array([])
    trade_ltn_prices = price_bins[trade_ltn_indices] if len(trade_ltn_indices) > 0 else np.array([])
    trade_ltn_strengths = trade_profile_smooth[trade_ltn_indices] if len(trade_ltn_indices) > 0 else np.array([])
    
    # Sort by strength and get top N
    if len(trade_htn_prices) > 0:
        trade_htn_sorted_idx = np.argsort(trade_htn_strengths)[::-1]  # Descending order
        trade_htn_prices = trade_htn_prices[trade_htn_sorted_idx][:multiple_returns]
        trade_htn_strengths = trade_htn_strengths[trade_htn_sorted_idx][:multiple_returns]
    
    if len(trade_ltn_prices) > 0:
        trade_ltn_sorted_idx = np.argsort(trade_ltn_strengths)  # Ascending order (weakest first for LTN)
        trade_ltn_prices = trade_ltn_prices[trade_ltn_sorted_idx][:multiple_returns]
        trade_ltn_strengths = trade_ltn_strengths[trade_ltn_sorted_idx][:multiple_returns]
    
    # Create multiple columns for each return
    result = {}
    
    # Trade profile for all rows
    trade_profile_series = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        price_bin_idx = np.argmin(np.abs(price_bins - current_price))
        trade_profile_series.iloc[i] = trade_profile_smooth[price_bin_idx]
    result['trade_profile'] = ta._ensure_same_length(trade_profile_series, original_length)
    
    # Create multiple Trade HTN and LTN columns
    for n in range(multiple_returns):
        trade_htn_series = pd.Series([np.nan] * original_length)
        trade_htn_strength_series = pd.Series([np.nan] * original_length)
        trade_ltn_series = pd.Series([np.nan] * original_length)
        
        if n < len(trade_htn_prices):
            trade_htn_series[:] = trade_htn_prices[n]  # Same level for all rows
            trade_htn_strength_series[:] = trade_htn_strengths[n]
        
        if n < len(trade_ltn_prices):
            trade_ltn_series[:] = trade_ltn_prices[n]  # Same level for all rows
            
        result[f'trade_htn_level_{n+1}'] = trade_htn_series
        result[f'trade_htn_strength_{n+1}'] = trade_htn_strength_series
        result[f'trade_ltn_level_{n+1}'] = trade_ltn_series
    
    return result

def calculate_all_indicators(dfs: Dict[str, pd.DataFrame], window_size: int = 50, multiple_returns: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Calculate all technical indicators for all symbols using rolling window approach
    
    Args:
        dfs: Dictionary of {symbol: DataFrame} containing OHLCV data
        window_size: Window size for rolling calculations
        
    Returns:
        Dictionary containing combined data and indicators for all symbols plus correlations
    """
    results = {}
    
    print("Calculating correlations...")
    # Calculate correlations first (between symbols and close/volume)
    correlations = calculate_correlations(dfs)
    results['correlations'] = correlations
    
    # Calculate indicators for each symbol
    for symbol, df in dfs.items():
        print(f"Calculating indicators for {symbol}...")
        
        # Start with original OHLCV data
        combined_df = df.copy()
        
        # Basic indicators
        mas = calculate_moving_averages(df, window_size=window_size)
        for key, value in mas.items():
            combined_df[key] = value
            
        combined_df['rsi'] = calculate_rsi(df, window_size=window_size)
        
        macd_data = calculate_macd(df, window_size=window_size)
        for key, value in macd_data.items():
            combined_df[key] = value
            
        bb_data = calculate_bollinger_bands(df, window_size=window_size)
        for key, value in bb_data.items():
            combined_df[key] = value
            
        # Add more indicators
        combined_df['atr'] = calculate_atr(df, window_size=window_size)
        combined_df['vwap'] = calculate_vwap(df, window_size=window_size)
        
        # Add volume, time, and trade nodes with reduced parameters for window_size
        try:
            vol_nodes = calculate_volume_nodes(df, window_size=window_size, num_bins=min(25, window_size//2), multiple_returns=multiple_returns)
            for key, value in vol_nodes.items():
                combined_df[key] = value
        except Exception as e:
            print(f"Warning: Could not calculate volume nodes for {symbol}: {e}")
            
        try:
            time_nodes = calculate_time_nodes(df, window_size=window_size, num_bins=min(25, window_size//2), multiple_returns=multiple_returns)
            for key, value in time_nodes.items():
                combined_df[key] = value
        except Exception as e:
            print(f"Warning: Could not calculate time nodes for {symbol}: {e}")
            
        try:
            trade_nodes = calculate_trade_nodes(df, window_size=window_size, num_bins=min(25, window_size//2), multiple_returns=multiple_returns)
            for key, value in trade_nodes.items():
                combined_df[key] = value
        except Exception as e:
            print(f"Warning: Could not calculate trade nodes for {symbol}: {e}")
        
        results[symbol] = combined_df
        print(f"Completed {symbol}: {len(combined_df.columns)} total columns (OHLCV + indicators)")
    
    print(f"All indicators calculated for {len(dfs)} symbols")
    return results
