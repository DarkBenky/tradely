import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import pickle
import os
import wandb
import time
import gc
from portfolio_env import PortfolioEnv
from train import build_model, train_on_batch

def load_synthetic_data_chunked(file_path, max_batches=None):
    """
    Load synthetic batches from pickle file in a memory-efficient way.
    
    Args:
        file_path: Path to the pickle file
        max_batches: Maximum number of batches to load (None for all)
    
    Returns:
        List of batch dictionaries
    """
    batches = []
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return batches
    
    print(f"Loading synthetic data from {file_path}...")
    
    batch_count = 0
    with open(file_path, 'rb') as f:
        while True:
            try:
                batch = pickle.load(f)
                batches.append(batch)
                batch_count += 1
                
                if max_batches and batch_count >= max_batches:
                    print(f"Reached max_batches limit ({max_batches}), stopping load")
                    break
                    
            except EOFError:
                break
    
    total_samples = sum(len(batch['states']) for batch in batches)
    print(f"Loaded {len(batches)} batches with {total_samples} total samples")
    
    return batches

def train_on_synthetic_data(model, optimizer, batches, epochs=10, batch_size=64, max_samples=None):
    """
    Train model on synthetic data with optimal actions.
    Memory-efficient: processes data in chunks without loading all at once.
    
    Args:
        model: The neural network model
        optimizer: TensorFlow optimizer
        batches: List of batch dictionaries from synthetic generator
        epochs: Number of training epochs
        batch_size: Mini-batch size for training
        max_samples: Maximum number of samples to use (None for all)
    
    Returns:
        Training metrics
    """
    print(f"\n=== TRAINING ON SYNTHETIC DATA ===")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    # Combine all batches into single arrays
    all_states = []
    all_actions = []
    all_returns = []
    all_is_random = []
    
    sample_count = 0
    for batch in batches:
        batch_samples = len(batch['states'])
        
        if max_samples and sample_count + batch_samples > max_samples:
            # Take only partial batch to reach max_samples
            remaining = max_samples - sample_count
            all_states.extend(batch['states'][:remaining])
            all_actions.extend(batch['actions'][:remaining])
            all_returns.extend(batch['returns'][:remaining])
            all_is_random.extend(batch['is_random'][:remaining])
            break
        
        all_states.extend(batch['states'])
        all_actions.extend(batch['actions'])
        all_returns.extend(batch['returns'])
        all_is_random.extend(batch['is_random'])
        sample_count += batch_samples
    
    all_states = np.array(all_states, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    all_returns = np.array(all_returns, dtype=np.float32)
    all_is_random = np.array(all_is_random, dtype=np.float32)
    
    total_samples = len(all_states)
    print(f"Total training samples: {total_samples}")
    
    global_batch_num = 0
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_start = time.time()
        epoch_losses = []
        epoch_entropies = []
        epoch_grad_norms = []
        
        # Shuffle data each epoch
        indices = np.random.permutation(total_samples)
        
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        for i in range(0, total_samples, batch_size):
            global_batch_num += 1
            batch_indices = indices[i:i+batch_size]
            
            batch_states = all_states[batch_indices]
            batch_actions = all_actions[batch_indices]
            batch_returns = all_returns[batch_indices]
            batch_is_random = all_is_random[batch_indices]
            
            # Train on this mini-batch
            metrics = train_on_batch(
                model, optimizer, 
                batch_states, batch_actions, batch_returns, 
                batch_is_random, global_batch_num
            )
            
            epoch_losses.append(metrics['batch/loss'])
            epoch_entropies.append(metrics['batch/entropy'])
            epoch_grad_norms.append(metrics['batch/grad_norm'])
            
            # Log every 10 batches
            if global_batch_num % 10 == 0:
                wandb.log({
                    'synthetic/batch_num': global_batch_num,
                    'synthetic/batch_loss': metrics['batch/loss'],
                    'synthetic/batch_entropy': metrics['batch/entropy'],
                    'synthetic/batch_grad_norm': metrics['batch/grad_norm'],
                    'synthetic/mean_action_prob': metrics['batch/mean_action_prob'],
                })
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        avg_entropy = np.mean(epoch_entropies)
        avg_grad_norm = np.mean(epoch_grad_norms)
        
        wandb.log({
            'synthetic/epoch': epoch + 1,
            'synthetic/epoch_loss': avg_loss,
            'synthetic/epoch_entropy': avg_entropy,
            'synthetic/epoch_grad_norm': avg_grad_norm,
            'synthetic/epoch_time': epoch_time,
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Entropy: {avg_entropy:.4f} - Time: {epoch_time:.1f}s")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'synthetic_checkpoint_epoch{epoch+1}.weights.h5'
            model.save_weights(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    print("\nSynthetic training complete!")
    
    return {
        'total_epochs': epochs,
        'total_batches': global_batch_num,
        'final_loss': avg_loss,
        'final_entropy': avg_entropy
    }

def iterative_synthetic_training(model, optimizer, synthetic_path, cycles=5, epochs_per_cycle=10, max_batches=None, max_samples=None):
    """
    Repeatedly train on synthetic data for multiple cycles.
    This allows the model to see the optimal actions many times.
    
    Args:
        model: The neural network model
        optimizer: TensorFlow optimizer
        synthetic_path: Path to synthetic data file
        cycles: Number of training cycles
        epochs_per_cycle: Epochs per cycle
        max_batches: Maximum number of batches to load (None for all)
        max_samples: Maximum number of samples to use per cycle (None for all)
    
    Returns:
        Training history
    """
    print(f"\n{'='*80}")
    print("ITERATIVE SYNTHETIC TRAINING")
    print(f"{'='*80}")
    print(f"Cycles: {cycles}")
    print(f"Epochs per cycle: {epochs_per_cycle}")
    
    # Load synthetic data once
    batches = load_synthetic_data_chunked(synthetic_path, max_batches=max_batches)
    
    if len(batches) == 0:
        print("No synthetic data found!")
        return None
    
    history = []
    
    for cycle in range(cycles):
        print(f"\n{'='*80}")
        print(f"CYCLE {cycle+1}/{cycles}")
        print(f"{'='*80}")
        
        cycle_start = time.time()
        
        # Train on synthetic data
        metrics = train_on_synthetic_data(
            model, optimizer, batches, 
            epochs=epochs_per_cycle, 
            batch_size=64,
            max_samples=max_samples
        )
        
        cycle_time = time.time() - cycle_start
        metrics['cycle'] = cycle + 1
        metrics['cycle_time'] = cycle_time
        history.append(metrics)
        
        # Save model after each cycle
        model.save_weights(f'synthetic_cycle{cycle+1}.weights.h5')
        wandb.save(f'synthetic_cycle{cycle+1}.weights.h5')
        
        wandb.log({
            'synthetic/cycle': cycle + 1,
            'synthetic/cycle_time': cycle_time,
            'synthetic/cycle_final_loss': metrics['final_loss'],
        })
        
        print(f"\nCycle {cycle+1}/{cycles} complete - Time: {cycle_time:.1f}s - Final loss: {metrics['final_loss']:.4f}")
        
        # Clear memory
        gc.collect()
    
    return history


if __name__ == "__main__":
    config = {
        "architecture": "transformer_v2",
        "num_transformer_blocks": 4,
        "embed_dim": 192,
        "num_heads": 12,
        "ff_dim": 768,
        "dropout_rate": 0.15,
        "feature_extractor_dim": 512,
        "num_attention_blocks": 2,
        "learning_rate": 0.0003,
        "training_cycles": 10,
        "epochs_per_cycle": 5,
        "batch_size": 64,
        "max_batches": 10,
        "max_samples": None,
    }
    
    wandb.init(project="portfolio-trading-synthetic", config=config)
    config = wandb.config
    
    print("=== SYNTHETIC DATA TRAINING ===\n")
    
    # Initialize environment (just to get observation shape)
    print("Initializing environment...")
    env = PortfolioEnv(max_records=100_000)
    observation = env.reset()
    obs_shape = observation.shape
    num_assets = len(env.asset_names)
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of assets: {num_assets}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(obs_shape, num_assets, config)
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    # Load existing model if available
    pretrained_model_path = 'best_model.weights.h5'
    if os.path.exists(pretrained_model_path):
        print(f"\nLoading existing model from {pretrained_model_path}...")
        model.load_weights(pretrained_model_path)
        print("Model loaded successfully!")
    
    # Path to synthetic data
    synthetic_data_path = "/media/user/HDD 1TB/Data/synthetic_training_data.pkl"
    
    # Train iteratively on synthetic data
    history = iterative_synthetic_training(
        model, optimizer, synthetic_data_path,
        cycles=config.training_cycles,
        epochs_per_cycle=config.epochs_per_cycle,
        max_batches=config.max_batches,
        max_samples=config.max_samples
    )
    
    if history:
        # Save final model as synthetic_trained_model.weights.h5
        final_model_path = 'synthetic_trained_model.weights.h5'
        model.save_weights(final_model_path)
        wandb.save(final_model_path)
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"Total cycles: {len(history)}")
        print(f"Final loss: {history[-1]['final_loss']:.4f}")
        print(f"Model saved to: {final_model_path}")
        print(f"\nNote: To use this model, copy it to best_model.weights.h5")
    
    wandb.finish()
