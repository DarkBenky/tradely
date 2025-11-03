import pickle
import os

def count_records_in_pickle(file_path):
    """
    Count the number of records (batches and total samples) in a pickle file.
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        Dictionary with batch count and total sample count
    """
    if not os.path.exists(file_path):
        return {'batches': 0, 'samples': 0, 'error': 'File not found'}
    
    batch_count = 0
    total_samples = 0
    
    try:
        with open(file_path, 'rb') as f:
            while True:
                try:
                    batch = pickle.load(f)
                    batch_count += 1
                    
                    # Count samples in this batch
                    if isinstance(batch, dict):
                        if 'states' in batch:
                            total_samples += len(batch['states'])
                        elif 'actions' in batch:
                            total_samples += len(batch['actions'])
                    
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error reading batch {batch_count + 1}: {e}")
                    break
        
        return {
            'batches': batch_count,
            'samples': total_samples,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
    
    except Exception as e:
        return {'batches': 0, 'samples': 0, 'error': str(e)}

def print_data_summary(label, file_path):
    """Print a formatted summary of data in a pickle file"""
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"File: {file_path}")
    
    if not os.path.exists(file_path):
        print("Status: File does not exist")
        return
    
    stats = count_records_in_pickle(file_path)
    
    if 'error' in stats and stats['error'] != 'File not found':
        print(f"Status: Error - {stats['error']}")
        return
    
    print(f"Status: OK")
    print(f"Batches: {stats['batches']:,}")
    print(f"Total Samples: {stats['samples']:,}")
    print(f"File Size: {stats['file_size_mb']:.2f} MB")
    
    if stats['batches'] > 0:
        avg_samples_per_batch = stats['samples'] / stats['batches']
        print(f"Average Samples per Batch: {avg_samples_per_batch:.1f}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DATA RECORD COUNTER")
    print("="*80)
    
    # Paths to data files
    synthetic_data_path = "/media/user/HDD 1TB/Data/synthetic_training_data.pkl"
    recorded_data_path = "/media/user/HDD 1TB/Data/training_data.pkl"
    
    # Count synthetic data
    print_data_summary("SYNTHETIC DATA", synthetic_data_path)
    
    # Count recorded data
    print_data_summary("RECORDED DATA", recorded_data_path)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    synthetic_stats = count_records_in_pickle(synthetic_data_path)
    recorded_stats = count_records_in_pickle(recorded_data_path)
    
    total_batches = synthetic_stats.get('batches', 0) + recorded_stats.get('batches', 0)
    total_samples = synthetic_stats.get('samples', 0) + recorded_stats.get('samples', 0)
    total_size_mb = synthetic_stats.get('file_size_mb', 0) + recorded_stats.get('file_size_mb', 0)
    
    print(f"Total Batches: {total_batches:,}")
    print(f"Total Samples: {total_samples:,}")
    print(f"Total Size: {total_size_mb:.2f} MB")
    print(f"\nSynthetic Ratio: {(synthetic_stats.get('samples', 0) / total_samples * 100) if total_samples > 0 else 0:.1f}%")
    print(f"Recorded Ratio: {(recorded_stats.get('samples', 0) / total_samples * 100) if total_samples > 0 else 0:.1f}%")
