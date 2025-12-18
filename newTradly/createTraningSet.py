import json
import numpy as np
import os
from params import WINDOW_SIZE, NEXT_PRICE_PREDICTION, NEXT_PRICE_PREDICTION_1, NEXT_PRICE_PREDICTION_2

def import_bucket_config(file_path='bucket_config.json'):
    with open(file_path, 'r') as f:
        bucket_config = json.load(f)
    return bucket_config

def value_to_distribution(value, buckets):
    num_buckets = len(buckets) - 1
    distribution = np.zeros(num_buckets)
    
    bucket_idx = np.digitize(value, buckets) - 1
    bucket_idx = np.clip(bucket_idx, 0, num_buckets - 1)
    
    if bucket_idx == 0:
        lower_bound = buckets[0]
        upper_bound = buckets[1]
    elif bucket_idx == num_buckets - 1:
        lower_bound = buckets[-2]
        upper_bound = buckets[-1]
    else:
        lower_bound = buckets[bucket_idx]
        upper_bound = buckets[bucket_idx + 1]
    
    bucket_range = upper_bound - lower_bound
    if bucket_range > 0:
        position = (value - lower_bound) / bucket_range
        position = np.clip(position, 0, 1)
    else:
        position = 0.5
    
    distribution[bucket_idx] = 1.0 - abs(position - 0.5) * 0.5
    
    if position < 0.5 and bucket_idx > 0:
        distribution[bucket_idx - 1] = abs(position - 0.5) * 0.5
    elif position > 0.5 and bucket_idx < num_buckets - 1:
        distribution[bucket_idx + 1] = abs(position - 0.5) * 0.5
    
    distribution = distribution / distribution.sum()
    
    return distribution

def create_training_set(number_of_samples: int, number_of_files: int):
    bucket_config = import_bucket_config()
    
    X_samples = []
    y_next = []
    y_next_1 = []
    y_next_2 = []
    
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
        
        min_length = WINDOW_SIZE + NEXT_PRICE_PREDICTION_2
        if min_len < min_length:
            continue
        
        files_processed += 1
        max_start_index = min_len - WINDOW_SIZE - NEXT_PRICE_PREDICTION_2
        
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
            
            y_sum = close_prices[end_index:end_index + NEXT_PRICE_PREDICTION].sum()
            y_sum_1 = close_prices[end_index:end_index + int(NEXT_PRICE_PREDICTION_1)].sum()
            y_sum_2 = close_prices[end_index:end_index + int(NEXT_PRICE_PREDICTION_2)].sum()
            
            y_normalized = y_sum / NEXT_PRICE_PREDICTION
            y_normalized_1 = y_sum_1 / NEXT_PRICE_PREDICTION_1
            y_normalized_2 = y_sum_2 / NEXT_PRICE_PREDICTION_2
            
            y_dist = value_to_distribution(y_normalized, bucket_config['close']['buckets'])
            y_dist_1 = value_to_distribution(y_normalized_1, bucket_config['close']['buckets'])
            y_dist_2 = value_to_distribution(y_normalized_2, bucket_config['close']['buckets'])
            
            y_next.append(y_dist)
            y_next_1.append(y_dist_1)
            y_next_2.append(y_dist_2)
    
    X = np.array(X_samples)
    y_next = np.array(y_next)
    y_next_1 = np.array(y_next_1)
    y_next_2 = np.array(y_next_2)
    
    return X, (y_next, y_next_1, y_next_2)

if __name__ == "__main__":
    X, (y_next, y_next_1, y_next_2) = create_training_set(128, 10)
    print("X shape:", X.shape)
    print("y_next shape:", y_next.shape)
    print("y_next_1 shape:", y_next_1.shape)
    print("y_next_2 shape:", y_next_2.shape)

    print("\nFirst X sample shape:", X[0].shape)
    print("First y_next distribution:", y_next[0])
    print("Sum of first y_next distribution:", y_next[0].sum())

