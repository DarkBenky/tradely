import json
import numpy as np
import os
from params import WINDOW_SIZE

def create_training_set(number_of_samples: int, number_of_files: int):
    X_samples = []
    y_next = []
    y_half = []
    y_full = []
    
    file_list = [f for f in os.listdir('tickerData') if f.endswith('.json')]
    samples_per_file = number_of_samples // number_of_files
    
    files_processed = 0
    file_attempts = 0
    max_attempts = number_of_files * 3
    
    while files_processed < number_of_files and file_attempts < max_attempts:
        file_attempts += 1
        selected_file = np.random.choice(file_list)
        
        try:
            with open(os.path.join('tickerData', selected_file), 'r') as f:
                data = json.load(f)
        except:
            continue
        
        close_prices = np.array(list(data['Close'].values()))
        open_prices = np.array(list(data['Open'].values()))
        high_prices = np.array(list(data['High'].values()))
        low_prices = np.array(list(data['Low'].values()))
        volume_data = np.array(list(data['Volume'].values()))
        
        min_len = min(len(close_prices), len(open_prices), len(high_prices), len(low_prices), len(volume_data))
        close_prices = close_prices[:min_len]
        open_prices = open_prices[:min_len]
        high_prices = high_prices[:min_len]
        low_prices = low_prices[:min_len]
        volume_data = volume_data[:min_len]
        
        min_length = WINDOW_SIZE * 2
        if min_len < min_length:
            continue
        
        files_processed += 1
        max_start_index = min_len - WINDOW_SIZE * 2
        
        if max_start_index <= 0:
            continue

        for i in range(samples_per_file):
            start_index = np.random.randint(0, max(1, max_start_index))
            end_index = start_index + WINDOW_SIZE

            x_close = close_prices[start_index:end_index]
            x_open = open_prices[start_index:end_index]
            x_high = high_prices[start_index:end_index]
            x_low = low_prices[start_index:end_index]
            x_volume = volume_data[start_index:end_index]
            
            X_sample = np.stack([x_close, x_open, x_high, x_low, x_volume], axis=-1)
            X_samples.append(X_sample)
            
            half_window = WINDOW_SIZE // 2
            
            y_next_val = close_prices[end_index:end_index + 1].sum()
            y_half_val = close_prices[end_index:end_index + half_window].sum()
            y_full_val = close_prices[end_index:end_index + WINDOW_SIZE].sum()
            
            y_next.append(y_next_val)
            y_half.append(y_half_val)
            y_full.append(y_full_val)
    
    X = np.array(X_samples)
    y_next = np.array(y_next)
    y_half = np.array(y_half)
    y_full = np.array(y_full)
    
    return X, (y_next, y_half, y_full)

if __name__ == "__main__":
    X, (y_next, y_half, y_full) = create_training_set(128, 10)
    print("X shape:", X.shape)
    print("y_next shape:", y_next.shape)
    print("y_half shape:", y_half.shape)
    print("y_full shape:", y_full.shape)
    print("\nFirst X sample shape:", X[0].shape)
    print("First y_next:", y_next[0])
    print("First y_half:", y_half[0])
    print("First y_full:", y_full[0])

