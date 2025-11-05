import numpy as np
import pickle
import os
import sys
import wandb

# Set GPU BEFORE importing tensorflow - MUST be first
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only RTX 3090 (GPU 1)
    # Re-execute the script with the environment variable set
    os.execv(sys.executable, [sys.executable] + sys.argv)

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Enable mixed precision for memory savings
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision enabled: {policy.name} (uses float16 for computations, float32 for variables)")

# Configure GPU memory growth and allow memory oversubscription
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Allow memory growth (allocate as needed)
            tf.config.experimental.set_memory_growth(gpu, True)
            
        # Set soft memory limit (allow spilling to system RAM if needed)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=20480 * 2)]  # 40GB soft limit
        )
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        print(f"Memory limit: 20GB (can use system RAM as fallback)")
        print(f"Note: When CUDA_VISIBLE_DEVICES=1, physical GPU 1 (RTX 3090) appears as GPU:0")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

class FeatureReducer:
    """
    Autoencoder-based feature reducer that learns compressed representations.
    Train once, use forever - works with all future data.
    """
    
    def __init__(self, target_dim=2048):
        self.target_dim = target_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.input_dim = None
        self.is_fitted = False
        
    def _build_autoencoder(self, input_dim):
        """Build autoencoder architecture - balanced compression and memory efficiency"""
        self.input_dim = input_dim
        
        # Encoder - balanced compression
        encoder_input = keras.Input(shape=(input_dim,))
        
        # First layer - moderate compression
        x = keras.layers.Dense(int(self.target_dim * 1.5), activation='relu')(encoder_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)
        
        # Bottleneck - 2048 dims for good reconstruction
        encoded = keras.layers.Dense(self.target_dim, activation='relu', name='encoded')(x)
        
        self.encoder = keras.Model(encoder_input, encoded, name='encoder')
        
        # Decoder - mirror encoder structure
        decoder_input = keras.Input(shape=(self.target_dim,))
        
        # First decoder layer
        x = keras.layers.Dense(int(self.target_dim * 1.5), activation='relu')(decoder_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)
        
        # Output layer
        decoded = keras.layers.Dense(input_dim, activation='linear')(x)
        
        self.decoder = keras.Model(decoder_input, decoded, name='decoder')
        
        # Full autoencoder
        autoencoder_input = keras.Input(shape=(input_dim,))
        encoded_out = self.encoder(autoencoder_input)
        decoded_out = self.decoder(encoded_out)
        
        self.autoencoder = keras.Model(autoencoder_input, decoded_out, name='autoencoder')
        
        # Custom accuracy metric: percentage of values reconstructed within 10% error
        def reconstruction_accuracy(y_true, y_pred):
            """Percentage of values where |predicted - actual| / |actual| < 0.1"""
            # Cast to float32 for mixed precision compatibility
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            relative_error = tf.abs((y_pred - y_true) / (tf.abs(y_true) + 1e-8))
            within_threshold = tf.cast(relative_error < 0.1, tf.float32)
            return tf.reduce_mean(within_threshold)
        
        # Compile with mixed precision to save memory
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0015),
            loss='mse',
            metrics=['mae', reconstruction_accuracy]
        )

        self.autoencoder.summary()

        # log model size

        wandb.log({
            "model/autoencoder_params": self.autoencoder.count_params(),
        })
        
        print(f"\nAutoencoder architecture (memory-efficient, 2048 dims):")
        print(f"  Input: {input_dim} dims")
        print(f"  Encoder: {input_dim} → {int(self.target_dim * 1.5)} → {self.target_dim}")
        print(f"  Decoder: {self.target_dim} → {int(self.target_dim * 1.5)} → {input_dim}")
        print(f"  Compression ratio: {input_dim / self.target_dim:.2f}x")
        print(f"  Balanced: good accuracy with manageable memory usage")
        
    def fit(self, data_samples, epochs=100, batch_size=64, validation_split=0.1, use_wandb=True, incremental=False):
        """
        Train autoencoder on sample data to learn compression.
        
        Args:
            data_samples: List of observation arrays
            epochs: Training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            use_wandb: Whether to log to wandb
            incremental: If True, continues training from existing weights
        """
        if not incremental:
            print(f"\n{'='*80}")
            print(f"TRAINING AUTOENCODER FEATURE REDUCER")
            print(f"{'='*80}\n")
        
        print(f"Training on {len(data_samples)} samples...")
        
        data_array = np.array(data_samples, dtype=np.float32)
        print(f"Original dimension: {data_array.shape[1]}")
        
        # Initialize wandb (only on first call, not incremental)
        if use_wandb and not incremental:
            try:
                import wandb
                wandb.init(
                    project="tradely-feature-reducer",
                    config={
                        "architecture": "autoencoder",
                        "input_dim": data_array.shape[1],
                        "target_dim": self.target_dim,
                        "compression_ratio": data_array.shape[1] / self.target_dim,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "validation_split": validation_split,
                        "samples": len(data_samples),
                        "incremental": True
                    }
                )
                print("Wandb logging enabled")
            except:
                print("Wandb not available, skipping logging")
                use_wandb = False
        
        # Standardize - fit scaler only on first batch, transform on subsequent
        if not self.is_fitted or not incremental:
            print("Fitting scaler on data...")
            data_scaled = self.scaler.fit_transform(data_array)
        else:
            print("Transforming data with existing scaler...")
            data_scaled = self.scaler.transform(data_array)
        
        # Build autoencoder only if not already built
        if self.autoencoder is None:
            print("Building autoencoder...")
            self._build_autoencoder(data_array.shape[1])
        else:
            print("Using existing autoencoder architecture...")
        
        # Train
        print(f"\nTraining for {epochs} epochs...")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_reconstruction_accuracy',  # Monitor accuracy instead of loss
                patience=8,  # More patience for accuracy improvements
                restore_best_weights=True,
                mode='max',  # Maximize accuracy
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_reconstruction_accuracy',  # Monitor accuracy
                factor=0.5,  # Reduce LR by half
                patience=3,  # Reduce quickly if no improvement
                min_lr=1e-7,
                mode='max',  # Maximize accuracy
                verbose=1
            )
        ]
        
        # Add wandb callback if enabled (including incremental training)
        if use_wandb or (not incremental and use_wandb):
            try:
                import wandb
                
                # Check if wandb run is active (for incremental training)
                if wandb.run is None and not use_wandb:
                    # Don't add callback if no active run
                    pass
                else:
                    # Custom callback for wandb logging (avoids compatibility issues)
                    class SimpleWandbCallback(keras.callbacks.Callback):
                        def __init__(self, batch_num=0):
                            super().__init__()
                            self.batch_num = batch_num
                            
                        def on_epoch_end(self, epoch, logs=None):
                            if logs and wandb.run is not None:
                                log_dict = {
                                    "epoch/epoch": epoch,
                                    "epoch/loss": logs.get("loss"),
                                    "epoch/val_loss": logs.get("val_loss"),
                                    "epoch/mae": logs.get("mae"),
                                    "epoch/val_mae": logs.get("val_mae"),
                                    "epoch/accuracy": logs.get("reconstruction_accuracy"),
                                    "epoch/val_accuracy": logs.get("val_reconstruction_accuracy"),
                                    "epoch/learning_rate": float(self.model.optimizer.learning_rate)
                                }
                                # Add batch number for incremental training
                                if self.batch_num > 0:
                                    log_dict["epoch/batch_num"] = self.batch_num
                                wandb.log(log_dict)
                    
                    # Pass batch number if incremental
                    batch_num = 0
                    if incremental and hasattr(self, '_current_batch_num'):
                        batch_num = self._current_batch_num
                    
                    callbacks.append(SimpleWandbCallback(batch_num=batch_num))
                    if not incremental:
                        print("Added custom wandb callback")
            except Exception as e:
                if not incremental:
                    print(f"Warning: Could not add wandb callback: {e}")
                use_wandb = False
        
        history = self.autoencoder.fit(
            data_scaled, data_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        
        # Evaluate reconstruction quality
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_mae = history.history['mae'][-1]
        final_accuracy = history.history.get('reconstruction_accuracy', [0])[-1]
        final_val_accuracy = history.history.get('val_reconstruction_accuracy', [0])[-1]
        
        # Return validation accuracy for tracking best model
        self._last_val_accuracy = final_val_accuracy
        
        # Calculate reconstruction quality on sample
        sample_indices = np.random.choice(len(data_scaled), min(100, len(data_scaled)), replace=False)
        sample_data = data_scaled[sample_indices]
        reconstructed = self.autoencoder.predict(sample_data, verbose=0)
        reconstruction_error = np.mean(np.abs(sample_data - reconstructed))
        relative_error = reconstruction_error / (np.abs(sample_data).mean() + 1e-8)
        
        # Calculate actual accuracy (% within 10% of true value)
        sample_relative_error = np.abs((reconstructed - sample_data) / (np.abs(sample_data) + 1e-8))
        actual_accuracy = np.mean(sample_relative_error < 0.1)
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Final training loss: {final_loss:.6f}")
        print(f"Final validation loss: {final_val_loss:.6f}")
        print(f"Mean absolute error: {final_mae:.6f}")
        print(f"Reconstruction accuracy: {final_accuracy:.2%} (train) / {final_val_accuracy:.2%} (val)")
        print(f"Sample accuracy: {actual_accuracy:.2%}")
        print(f"Reconstruction error: {reconstruction_error:.6f}")
        print(f"Relative error: {relative_error:.2%}")
        print(f"Reduced to {self.target_dim} dimensions")
        
        # Log final metrics to wandb
        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "final/train_loss": final_loss,
                    "final/val_loss": final_val_loss,
                    "final/mae": final_mae,
                    "final/train_accuracy": final_accuracy,
                    "final/val_accuracy": final_val_accuracy,
                    "final/sample_accuracy": actual_accuracy,
                    "final/reconstruction_error": reconstruction_error,
                    "final/relative_error": relative_error,
                    "final/compression_ratio": data_array.shape[1] / self.target_dim
                })
                wandb.finish()
            except:
                pass
        
        return self
    
    def transform(self, observations):
        """
        Transform observations to reduced dimension using encoder.
        
        Args:
            observations: Single observation or batch [batch_size, obs_dim]
        
        Returns:
            Compressed observations [batch_size, reduced_dim]
        """
        if not self.is_fitted:
            raise ValueError("FeatureReducer not fitted. Call fit() first.")
        
        if len(observations.shape) == 1:
            observations = observations.reshape(1, -1)
        
        scaled = self.scaler.transform(observations)
        reduced = self.encoder.predict(scaled, verbose=0)
        
        return reduced.astype(np.float32)
    
    def reconstruct(self, observations):
        """
        Reconstruct original observations from compressed (for testing quality).
        
        Args:
            observations: Single observation or batch [batch_size, obs_dim]
        
        Returns:
            Reconstructed observations [batch_size, obs_dim]
        """
        if not self.is_fitted:
            raise ValueError("FeatureReducer not fitted. Call fit() first.")
        
        if len(observations.shape) == 1:
            observations = observations.reshape(1, -1)
        
        scaled = self.scaler.transform(observations)
        reconstructed = self.autoencoder.predict(scaled, verbose=0)
        reconstructed = self.scaler.inverse_transform(reconstructed)
        
        return reconstructed.astype(np.float32)
    
    def save(self, filepath='feature_reducer.h5'):
        """Save fitted reducer"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted reducer")
        
        # Save weights and scaler separately
        base_path = filepath.replace('.h5', '').replace('.pkl', '')
        
        self.encoder.save(f"{base_path}_encoder.h5")
        self.decoder.save(f"{base_path}_decoder.h5")
        
        with open(f"{base_path}_scaler.pkl", 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'target_dim': self.target_dim,
                'input_dim': self.input_dim
            }, f)
        
        print(f"\nFeature reducer saved:")
        print(f"  Encoder: {base_path}_encoder.h5")
        print(f"  Decoder: {base_path}_decoder.h5")
        print(f"  Scaler: {base_path}_scaler.pkl")
        print(f"  Compression: {self.input_dim} → {self.target_dim} dimensions")
    
    @classmethod
    def load(cls, filepath='feature_reducer.h5'):
        """Load pre-trained reducer"""
        base_path = filepath.replace('.h5', '').replace('.pkl', '')
        
        if not os.path.exists(f"{base_path}_encoder.h5"):
            raise FileNotFoundError(f"{base_path}_encoder.h5 not found")
        
        # Load models
        encoder = keras.models.load_model(f"{base_path}_encoder.h5")
        decoder = keras.models.load_model(f"{base_path}_decoder.h5")
        
        # Load scaler
        with open(f"{base_path}_scaler.pkl", 'rb') as f:
            data = pickle.load(f)
        
        reducer = cls(target_dim=data['target_dim'])
        reducer.encoder = encoder
        reducer.decoder = decoder
        reducer.scaler = data['scaler']
        reducer.input_dim = data['input_dim']
        reducer.is_fitted = True
        
        # Rebuild full autoencoder
        autoencoder_input = keras.Input(shape=(reducer.input_dim,))
        encoded_out = reducer.encoder(autoencoder_input)
        decoded_out = reducer.decoder(encoded_out)
        reducer.autoencoder = keras.Model(autoencoder_input, decoded_out, name='autoencoder')
        
        print(f"\nFeature reducer loaded from {base_path}")
        print(f"  Dimension: {reducer.input_dim} → {reducer.target_dim}")
        print(f"  Compression ratio: {reducer.input_dim / reducer.target_dim:.2f}x")
        
        return reducer


def train_reducer_incrementally(input_path, reducer_path='feature_reducer.h5', 
                                 samples_per_batch=2048, epochs_per_batch=10, 
                                 max_batches=None, use_wandb=True):
    """
    Train feature reducer incrementally by loading batches from file.
    Memory-efficient: doesn't load entire dataset at once.
    
    Args:
        input_path: Path to raw training data
        reducer_path: Path to save feature reducer
        samples_per_batch: Number of samples to load and train on at once
        epochs_per_batch: Epochs to train on each batch
        max_batches: Maximum number of batches to process (None = all)
        use_wandb: Whether to log to wandb
    """
    print(f"\n{'='*80}")
    print("INCREMENTAL FEATURE REDUCER TRAINING")
    print(f"{'='*80}\n")
    print(f"Input: {input_path}")
    print(f"Samples per batch: {samples_per_batch}")
    print(f"Epochs per batch: {epochs_per_batch}")
    
    # Initialize wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="tradely-feature-reducer-incremental",
                config={
                    "architecture": "autoencoder",
                    "target_dim": 2048,
                    "samples_per_batch": samples_per_batch,
                    "epochs_per_batch": epochs_per_batch,
                    "training_mode": "incremental"
                }
            )
            print("Wandb logging enabled\n")
        except:
            print("Wandb not available, skipping logging\n")
            use_wandb = False
    
    reducer = None
    batch_num = 0
    total_samples = 0
    best_accuracy = 0.0  # Track best accuracy
    
    with open(input_path, 'rb') as f:
        sample_buffer = []
        
        while True:
            try:
                batch = pickle.load(f)
                
                if isinstance(batch, dict) and 'states' in batch:
                    sample_buffer.extend(batch['states'])
                    
                    # When buffer has enough samples, train
                    while len(sample_buffer) >= samples_per_batch:
                        batch_num += 1
                        
                        # Check max_batches limit
                        if max_batches is not None and batch_num > max_batches:
                            print(f"\nReached max_batches limit ({max_batches})")
                            break
                        
                        # Take samples_per_batch samples
                        training_samples = sample_buffer[:samples_per_batch]
                        sample_buffer = sample_buffer[samples_per_batch:]
                        
                        print(f"\n{'='*60}")
                        print(f"BATCH {batch_num} - Training on {len(training_samples)} samples")
                        print(f"{'='*60}")
                        
                        # First batch: create and fit reducer
                        if reducer is None:
                            print("Creating new feature reducer...")
                            reducer = FeatureReducer(target_dim=2148)  # Fixed from 2248
                            reducer._current_batch_num = batch_num
                            reducer.fit(training_samples, 
                                       epochs=epochs_per_batch, 
                                       batch_size=24,  # Increased for better gradient estimates
                                       validation_split=0.2,
                                       use_wandb=use_wandb,
                                       incremental=False)
                        else:
                            # Subsequent batches: incremental training
                            print("Continuing incremental training...")
                            reducer._current_batch_num = batch_num
                            reducer.fit(training_samples,
                                       epochs=epochs_per_batch,
                                       batch_size=24,  # Increased for better gradient estimates
                                       validation_split=0.1,
                                       use_wandb=True,  # Changed to True to enable logging!
                                       incremental=True)
                        
                        total_samples += len(training_samples)
                        
                        # Get validation accuracy from last training
                        current_accuracy = getattr(reducer, '_last_val_accuracy', 0.0)
                        
                        # Log batch metrics including accuracy
                        if use_wandb:
                            try:
                                import wandb
                                wandb.log({
                                    "batch/number": batch_num,
                                    "batch/samples": len(training_samples),
                                    "batch/total_samples": total_samples,
                                    "batch/val_accuracy": current_accuracy,
                                    "batch/best_accuracy": best_accuracy,
                                })
                            except:
                                pass
                        
                        print(f"Total samples processed: {total_samples:,}")
                        print(f"Batch validation accuracy: {current_accuracy:.2%}")
                        
                        # Save model if accuracy improved
                        if current_accuracy > best_accuracy:
                            best_accuracy = current_accuracy
                            best_model_path = reducer_path.replace('.pkl', '_best.pkl').replace('.h5', '_best.h5')
                            reducer.save(best_model_path)
                            print(f"★ NEW BEST ACCURACY: {best_accuracy:.2%} - Model saved!")
                            
                            # Also log to wandb
                            if use_wandb:
                                try:
                                    import wandb
                                    wandb.log({
                                        "best/accuracy": best_accuracy,
                                        "best/batch_num": batch_num,
                                        "best/total_samples": total_samples,
                                    })
                                except:
                                    pass
                        else:
                            print(f"Best accuracy remains: {best_accuracy:.2%}")
                        
                        del training_samples
                        
                    if max_batches is not None and batch_num >= max_batches:
                        break
                        
            except EOFError:
                # Train on remaining samples if any
                if len(sample_buffer) > 0 and reducer is not None:
                    batch_num += 1
                    print(f"\n{'='*60}")
                    print(f"FINAL BATCH {batch_num} - Training on {len(sample_buffer)} samples")
                    print(f"{'='*60}")
                    
                    reducer.fit(sample_buffer,
                               epochs=epochs_per_batch,
                               batch_size=32,  # Reduced for memory
                               validation_split=0.1,
                               use_wandb=False,
                               incremental=True)
                    
                    total_samples += len(sample_buffer)
                break
    
    if reducer is None:
        raise ValueError("No training data found in file!")
    
    # Save final model (always save the last state)
    reducer.save(reducer_path)
    
    print(f"\n{'='*80}")
    print("INCREMENTAL TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total batches: {batch_num}")
    print(f"Total samples: {total_samples:,}")
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Final model saved: {reducer_path}")
    best_model_path = reducer_path.replace('.pkl', '_best.pkl').replace('.h5', '_best.h5')
    print(f"Best model saved: {best_model_path}")
    
    if use_wandb:
        try:
            import wandb
            wandb.log({
                "final/total_batches": batch_num,
                "final/total_samples": total_samples,
                "final/best_accuracy": best_accuracy,
            })
            wandb.finish()
        except:
            pass
    
    return reducer


def preprocess_training_data(input_path, output_path, reducer_path='feature_reducer.h5', batch_size=1024):
    """
    Preprocess training data in streaming mode: train autoencoder (if needed) and transform all data.
    Memory-efficient: processes data in small batches without loading entire file.
    
    Args:
        input_path: Path to raw training data
        output_path: Path to save compressed data
        reducer_path: Path to save/load feature reducer
        batch_size: Number of samples to process at once
    """
    print(f"\n{'='*80}")
    print("PREPROCESSING TRAINING DATA (STREAMING MODE)")
    print(f"{'='*80}\n")
    
    # Check if reducer exists
    base_path = reducer_path.replace('.h5', '').replace('.pkl', '')
    encoder_exists = os.path.exists(f"{base_path}_encoder.h5")
    
    if encoder_exists:
        print(f"Loading existing feature reducer from {reducer_path}")
        reducer = FeatureReducer.load(reducer_path)
        fit_reducer = False
    else:
        print("Creating new feature reducer...")
        reducer = FeatureReducer(target_dim=2048)
        fit_reducer = True
    
    # If need to fit, collect samples first
    if fit_reducer:
        print(f"\nCollecting diverse samples from all available data sources...")
        sample_states = []
        
        # Collect from current file (prioritize what we're about to process)
        print(f"  [1] Collecting from {os.path.basename(input_path)}...")
        samples_from_current = 0
        with open(input_path, 'rb') as f:
            batch_count = 0
            while samples_from_current < 1024:
                try:
                    batch = pickle.load(f)
                    batch_count += 1
                    # Skip some batches to get diversity across the file
                    if batch_count % 5 == 0 and isinstance(batch, dict) and 'states' in batch:
                        states = batch['states'][:min(50, 1024 - samples_from_current)]
                        sample_states.extend(states)
                        samples_from_current += len(states)
                except EOFError:
                    break
        print(f"      Collected {samples_from_current} samples")
        
        # Try to collect from other data source for diversity
        other_paths = []
        synthetic_raw = "/media/user/HDD 1TB/Data/synthetic_training_data.pkl"
        recorded_raw = "/media/user/HDD 1TB/Data/training_data.pkl"
        
        if input_path != synthetic_raw and os.path.exists(synthetic_raw):
            other_paths.append(("synthetic", synthetic_raw))
        if input_path != recorded_raw and os.path.exists(recorded_raw):
            other_paths.append(("recorded", recorded_raw))
        
        for source_name, source_path in other_paths:
            print(f"  [2] Collecting from {source_name} data...")
            samples_from_other = 0
            try:
                with open(source_path, 'rb') as f:
                    batch_count = 0
                    while samples_from_other < 1024:
                        try:
                            batch = pickle.load(f)
                            batch_count += 1
                            # Skip batches for diversity
                            if batch_count % 5 == 0 and isinstance(batch, dict) and 'states' in batch:
                                states = batch['states'][:min(50, 1024 - samples_from_other)]
                                sample_states.extend(states)
                                samples_from_other += len(states)
                        except EOFError:
                            break
                print(f"      Collected {samples_from_other} samples")
            except Exception as e:
                print(f"      Could not read {source_name} data: {e}")
        
        # Shuffle to mix synthetic and real data
        print(f"\n  Total samples collected: {len(sample_states)}")
        print(f"  Shuffling for diversity...")
        import random
        random.shuffle(sample_states)
        
        if len(sample_states) < 500:
            raise ValueError(f"Not enough samples for training! Got {len(sample_states)}, need at least 500")
        
        reducer.fit(sample_states, epochs=100, batch_size=64, validation_split=0.1)
        reducer.save(reducer_path)
        del sample_states
    
    # Transform all data in streaming mode
    print(f"\nTransforming data to reduced dimensions (batch_size={batch_size})...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_batches = 0
    total_samples = 0
    
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        sample_buffer = []
        action_buffer = []
        return_buffer = []
        is_random_buffer = []
        
        while True:
            try:
                batch = pickle.load(f_in)
                total_batches += 1
                
                if isinstance(batch, dict):
                    # Add to buffers
                    sample_buffer.extend(batch.get('states', []))
                    action_buffer.extend(batch.get('actions', []))
                    return_buffer.extend(batch.get('returns', []))
                    is_random_buffer.extend(batch.get('is_random', []))
                    
                    # Process when buffer reaches batch_size
                    while len(sample_buffer) >= batch_size:
                        # Take batch_size samples
                        states_chunk = sample_buffer[:batch_size]
                        actions_chunk = action_buffer[:batch_size]
                        returns_chunk = return_buffer[:batch_size]
                        is_random_chunk = is_random_buffer[:batch_size]
                        
                        # Remove from buffers
                        sample_buffer = sample_buffer[batch_size:]
                        action_buffer = action_buffer[batch_size:]
                        return_buffer = return_buffer[batch_size:]
                        is_random_buffer = is_random_buffer[batch_size:]
                        
                        # Transform states
                        states_array = np.array(states_chunk, dtype=np.float32)
                        transformed_states = reducer.transform(states_array)
                        
                        # Save transformed batch
                        new_batch = {
                            'states': transformed_states.tolist(),
                            'actions': actions_chunk,
                            'returns': returns_chunk,
                            'is_random': is_random_chunk
                        }
                        
                        pickle.dump(new_batch, f_out)
                        total_samples += len(states_chunk)
                        
                        if total_samples % 10240 == 0:
                            print(f"  Processed {total_samples:,} samples...")
                        
                        del states_array, transformed_states, new_batch
                
            except EOFError:
                # Process remaining samples in buffer
                if len(sample_buffer) > 0:
                    print(f"\n  Processing final {len(sample_buffer)} samples...")
                    states_array = np.array(sample_buffer, dtype=np.float32)
                    transformed_states = reducer.transform(states_array)
                    
                    new_batch = {
                        'states': transformed_states.tolist(),
                        'actions': action_buffer,
                        'returns': return_buffer,
                        'is_random': is_random_buffer
                    }
                    
                    pickle.dump(new_batch, f_out)
                    total_samples += len(sample_buffer)
                
                break
    
    print(f"\nPreprocessed data saved to {output_path}")
    print(f"Total batches read: {total_batches:,}")
    print(f"Total samples processed: {total_samples:,}")
    print(f"Reduced dimension: {reducer.target_dim}")
    print(f"Compression ratio: ~{reducer.input_dim / reducer.target_dim:.2f}x")


if __name__ == "__main__":
    import sys
    
    # Paths
    synthetic_raw = "/media/user/HDD 1TB/Data/synthetic_training_data.pkl"
    synthetic_compressed = "/media/user/HDD 1TB/Data/synthetic_training_data_compressed.pkl"
    
    recorded_raw = "/media/user/HDD 1TB/Data/training_data.pkl"
    recorded_compressed = "/media/user/HDD 1TB/Data/training_data_compressed.pkl"
    
    reducer_path = "feature_reducer.pkl"
    
    print("Feature Reducer - Incremental Training")
    print("=" * 80)
    
    # Check if old models exist and ask to remove
    base_path = reducer_path.replace('.h5', '').replace('.pkl', '')
    old_files = [
        f"{base_path}_encoder.h5",
        f"{base_path}_decoder.h5",
        f"{base_path}_scaler.pkl"
    ]
    
    existing_old_files = [f for f in old_files if os.path.exists(f)]
    
    if existing_old_files:
        print("\n WARNING: Old model files found:")
        for f in existing_old_files:
            print(f"  - {f}")
        
        response = input("\nDo you want to DELETE old models and train from scratch? (yes/no): ").lower().strip()
        
        if response in ['yes', 'y']:
            print("\n Deleting old model files...")
            for f in existing_old_files:
                try:
                    os.remove(f)
                    print(f"  ✓ Deleted: {f}")
                except Exception as e:
                    print(f"  ✗ Could not delete {f}: {e}")
            print()
        else:
            print("\n Keeping old models. Exiting to prevent conflicts.")
            print("   Either delete them manually or rename them before training.")
            sys.exit(0)
    
    # Check which files exist
    has_synthetic = os.path.exists(synthetic_raw)
    has_recorded = os.path.exists(recorded_raw)
    
    if not has_synthetic and not has_recorded:
        print("No training data found!")
        sys.exit(1)
    
    # Choose training mode
    print("\nTraining mode: INCREMENTAL (memory-efficient)")
    print("  - Loads 2048 samples at a time")
    print("  - Trains for 10 epochs per batch")
    print("  - Continues training across batches\n")
    
    # Train incrementally on synthetic data (prioritize synthetic for diversity)
    training_source = None
    if has_synthetic:
        print("Using synthetic data for training (more diverse)")
        training_source = synthetic_raw
    elif has_recorded:
        print("Using recorded data for training")
        training_source = recorded_raw
    
    if training_source:
        print(f"\nTraining feature reducer on {os.path.basename(training_source)}...")
        reducer = train_reducer_incrementally(
            input_path=training_source,
            reducer_path=reducer_path,
            samples_per_batch=512,  # Reduced for memory
            epochs_per_batch=10,
            max_batches=None,  # Train on all data
            use_wandb=True
        )
        
        print(f"\n{'='*80}")
        print("REDUCER TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"Feature reducer saved to: {reducer_path}")
        
        # Now compress both datasets using trained reducer
        print("\n\nCompressing datasets with trained reducer...")
        
        if has_synthetic:
            print("\n[1/2] Compressing synthetic data...")
            preprocess_training_data(synthetic_raw, synthetic_compressed, reducer_path)
        
        if has_recorded:
            print("\n[2/2] Compressing recorded data...")
            preprocess_training_data(recorded_raw, recorded_compressed, reducer_path)
        
        print(f"\n{'='*80}")
        print("ALL PREPROCESSING COMPLETE!")
        print(f"{'='*80}")
        print("Use this reducer for all future data - no retraining needed!")
        
        if has_synthetic:
            print(f"\nSynthetic data compressed: {synthetic_compressed}")
        if has_recorded:
            print(f"Recorded data compressed: {recorded_compressed}")
    else:
        print("No training data available!")
