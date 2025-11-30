# TODO: Use mixed precision training for speedup and memory savings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
import wandb
import json

D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 5
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
SAMPLES_PER_FILE = 2048
BEST_MODEL_PATH = "models/best_pretrain_transformer.weights.h5"
CONFIG_PATH = "models/model_config.json"
NORM_STATS_PATH = "models/normalization_stats.json"
DATA_FOLDER = "syntheticData"
LOG_INTERVAL = 1
WANDB_PROJECT = "portfolio-pretrain"
EPOCHS_PER_CHUNK = 5


class ObservationNormalizer:
    def __init__(self, features_per_timestep):
        self.features_per_timestep = features_per_timestep
        self.mean = np.zeros(features_per_timestep, dtype=np.float64)
        self.var = np.ones(features_per_timestep, dtype=np.float64)
        self.count = 0
        self.epsilon = 1e-8
    
    def update(self, obs_batch):
        batch_mean = np.mean(obs_batch, axis=(0, 1))
        batch_var = np.var(obs_batch, axis=(0, 1))
        batch_count = obs_batch.shape[0] * obs_batch.shape[1]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        
        self.count = total_count
    
    def normalize(self, obs_batch):
        std = np.sqrt(self.var + self.epsilon)
        return (obs_batch - self.mean) / std
    
    def save(self, filepath):
        stats = {
            'mean': self.mean.tolist(),
            'var': self.var.tolist(),
            'count': int(self.count)
        }
        with open(filepath, 'w') as f:
            json.dump(stats, f)
    
    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                stats = json.load(f)
            self.mean = np.array(stats['mean'], dtype=np.float64)
            self.var = np.array(stats['var'], dtype=np.float64)
            self.count = stats['count']
            return True
        return False


def get_data_specs_from_file(filepath):
    with open(filepath, 'rb') as f:
        sample = pickle.load(f)
    
    obs_shape = sample['obs'].shape
    num_assets = len(sample['action'])
    
    if len(obs_shape) == 1:
        print(f"Warning: Flat observation shape detected {obs_shape}")
        print("Model expects transformer shape (sequence_length, features_per_timestep)")
        raise ValueError("Data must be in TRANSFORMER_SHAPE format")
    
    sequence_length = obs_shape[0]
    features_per_timestep = obs_shape[1]
    
    return sequence_length, features_per_timestep, num_assets


def scan_data_folder(data_folder):
    data_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.pkl')])
    if not data_files:
        raise ValueError(f"No data files found in {data_folder}")
    
    first_file = os.path.join(data_folder, data_files[0])
    sequence_length, features_per_timestep, num_assets = get_data_specs_from_file(first_file)
    
    print(f"Detected data specs from {data_files[0]}:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Features per timestep: {features_per_timestep}")
    print(f"  Number of assets: {num_assets}")
    
    return sequence_length, features_per_timestep, num_assets, data_files


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout_rate):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            layers.Dense(dim_feedforward, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class PortfolioTransformer(keras.Model):
    def __init__(self, sequence_length, features_per_timestep, d_model=D_MODEL, 
                 num_heads=NHEAD, num_layers=NUM_ENCODER_LAYERS, 
                 dim_feedforward=DIM_FEEDFORWARD, dropout_rate=DROPOUT, num_assets=11):
        super(PortfolioTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_assets = num_assets
        self.sequence_length = sequence_length
        self.features_per_timestep = features_per_timestep
        
        self.input_projection = layers.Dense(d_model)
        
        self.pos_encoding = self.add_weight(
            name='pos_encoding',
            shape=(1, sequence_length, d_model),
            initializer='random_normal',
            trainable=True
        )
        
        self.encoder_layers = [
            TransformerEncoderBlock(d_model, num_heads, dim_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.global_pool = layers.GlobalAveragePooling1D()
        
        self.allocation_head = keras.Sequential([
            layers.Dense(dim_feedforward // 2, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_assets),
            layers.Softmax()
        ])
        
        self.confidence_head = keras.Sequential([
            layers.Dense(dim_feedforward // 2, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_assets, activation='sigmoid')
        ])
    
    def call(self, x, training=False):
        x = self.input_projection(x)
        x = x + self.pos_encoding
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)
        
        pooled = self.global_pool(x)
        
        allocation = self.allocation_head(pooled)
        confidence = self.confidence_head(pooled)
        
        return allocation, confidence
    
    def count_parameters(self):
        return sum([tf.size(w).numpy() for w in self.trainable_weights])


def load_samples_from_file(filepath, offset=0, max_samples=None):
    samples = []
    with open(filepath, 'rb') as f:
        count = 0
        loaded = 0
        while True:
            try:
                sample = pickle.load(f)
                if count >= offset and (max_samples is None or loaded < max_samples):
                    samples.append(sample)
                    loaded += 1
                count += 1
                if max_samples and loaded >= max_samples:
                    break
            except EOFError:
                break
    return samples


def create_batch(samples):
    obs_list = []
    action_list = []
    confidence_list = []
    
    for sample in samples:
        obs_list.append(sample['obs'])
        action_list.append(sample['action'])
        confidence_list.append(sample['confidence'])
    
    obs_batch = np.array(obs_list, dtype=np.float32)
    action_batch = np.array(action_list, dtype=np.float32)
    confidence_batch = np.array(confidence_list, dtype=np.float32)
    
    assert len(obs_batch.shape) == 3, f"obs_batch should be 3D (batch, seq, features), got shape {obs_batch.shape}"
    assert len(action_batch.shape) == 2, f"action_batch should be 2D (batch, num_assets), got shape {action_batch.shape}"
    assert len(confidence_batch.shape) == 2, f"confidence_batch should be 2D (batch, num_assets), got shape {confidence_batch.shape}"
    
    return obs_batch, action_batch, confidence_batch


def compute_loss(model, obs_batch, action_batch, confidence_batch, training=True):
    pred_allocation, pred_confidence = model(obs_batch, training=training)
    
    assert pred_allocation.shape == action_batch.shape, f"pred_allocation shape {pred_allocation.shape} != action_batch shape {action_batch.shape}"
    assert pred_confidence.shape == confidence_batch.shape, f"pred_confidence shape {pred_confidence.shape} != confidence_batch shape {confidence_batch.shape}"
    
    allocation_loss = tf.reduce_mean(tf.square(pred_allocation - action_batch))
    
    confidence_loss = tf.reduce_mean(tf.square(pred_confidence - confidence_batch))
    
    epsilon = 1e-7
    pred_allocation_safe = tf.clip_by_value(pred_allocation, epsilon, 1.0 - epsilon)
    action_batch_safe = tf.clip_by_value(action_batch, epsilon, 1.0 - epsilon)
    
    ce_loss = -tf.reduce_mean(tf.reduce_sum(action_batch_safe * tf.math.log(pred_allocation_safe), axis=1))
    
    top_k = 3
    _, target_top_indices = tf.nn.top_k(action_batch, k=top_k)
    pred_at_target_tops = tf.gather(pred_allocation, target_top_indices, batch_dims=1)
    target_at_target_tops = tf.gather(action_batch, target_top_indices, batch_dims=1)
    top_k_loss = tf.reduce_mean(tf.square(pred_at_target_tops - target_at_target_tops))
    
    total_loss = allocation_loss + 0.5 * confidence_loss + 0.3 * ce_loss + 0.5 * top_k_loss
    
    pred_allocation_argmax = tf.argmax(pred_allocation, axis=1)
    action_batch_argmax = tf.argmax(action_batch, axis=1)
    allocation_accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_allocation_argmax, action_batch_argmax), tf.float32))
    
    pred_confidence_argmax = tf.argmax(pred_confidence, axis=1)
    confidence_batch_argmax = tf.argmax(confidence_batch, axis=1)
    confidence_accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_confidence_argmax, confidence_batch_argmax), tf.float32))
    
    return total_loss, allocation_loss, confidence_loss, ce_loss, top_k_loss, allocation_accuracy, confidence_accuracy


@tf.function
def train_step(model, optimizer, obs_batch, action_batch, confidence_batch):
    with tf.GradientTape() as tape:
        total_loss, allocation_loss, confidence_loss, ce_loss, top_k_loss, allocation_accuracy, confidence_accuracy = compute_loss(
            model, obs_batch, action_batch, confidence_batch, training=True
        )
    
    gradients = tape.gradient(total_loss, model.trainable_weights)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    return total_loss, allocation_loss, confidence_loss, ce_loss, top_k_loss, allocation_accuracy, confidence_accuracy


def save_model_and_config(model, sequence_length, features_per_timestep, num_assets, filepath, config_path):
    model.save_weights(filepath)
    
    config = {
        'sequence_length': int(sequence_length),
        'features_per_timestep': int(features_per_timestep),
        'num_assets': int(num_assets),
        'd_model': D_MODEL,
        'num_heads': NHEAD,
        'num_layers': NUM_ENCODER_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout_rate': DROPOUT
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_model_from_config(config_path, weights_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = PortfolioTransformer(
        sequence_length=config['sequence_length'],
        features_per_timestep=config['features_per_timestep'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout_rate=config['dropout_rate'],
        num_assets=config['num_assets']
    )
    
    dummy_input = tf.random.normal((1, config['sequence_length'], config['features_per_timestep']))
    _ = model(dummy_input, training=False)
    
    model.load_weights(weights_path)
    
    return model, config


def train():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPUs available: {len(physical_devices)}")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        print("No GPU detected, using CPU")
    
    print(f"\nScanning data folder: {DATA_FOLDER}")
    sequence_length, features_per_timestep, num_assets, data_files = scan_data_folder(DATA_FOLDER)
    
    config = {
        'sequence_length': sequence_length,
        'features_per_timestep': features_per_timestep,
        'd_model': D_MODEL,
        'nhead': NHEAD,
        'num_encoder_layers': NUM_ENCODER_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'num_assets': num_assets,
        'samples_per_file': SAMPLES_PER_FILE,
        'epochs_per_chunk': EPOCHS_PER_CHUNK,
        'log_interval': LOG_INTERVAL
    }
    
    wandb.init(project=WANDB_PROJECT, config=config)
    
    if os.path.exists(CONFIG_PATH) and os.path.exists(BEST_MODEL_PATH):
        print(f"\nLoading best model from {BEST_MODEL_PATH}")
        model, loaded_config = load_model_from_config(CONFIG_PATH, BEST_MODEL_PATH)
        
        if (loaded_config['sequence_length'] != sequence_length or 
            loaded_config['features_per_timestep'] != features_per_timestep or
            loaded_config['num_assets'] != num_assets):
            print("WARNING: Data specs changed! Creating new model...")
            model = PortfolioTransformer(
                sequence_length=sequence_length,
                features_per_timestep=features_per_timestep,
                d_model=D_MODEL,
                num_heads=NHEAD,
                num_layers=NUM_ENCODER_LAYERS,
                dim_feedforward=DIM_FEEDFORWARD,
                dropout_rate=DROPOUT,
                num_assets=num_assets
            )
        else:
            print("Model loaded successfully!")
    else:
        print("\nCreating new model...")
        model = PortfolioTransformer(
            sequence_length=sequence_length,
            features_per_timestep=features_per_timestep,
            d_model=D_MODEL,
            num_heads=NHEAD,
            num_layers=NUM_ENCODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout_rate=DROPOUT,
            num_assets=num_assets
        )
    
    dummy_input = tf.random.normal((1, sequence_length, features_per_timestep))
    _ = model(dummy_input, training=False)
    
    model.summary()

    num_params = model.count_parameters()
    model_size_mb = num_params * 4 / (1024 * 1024)
    
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB (assuming float32)")
    
    wandb.config.update({
        'num_parameters': num_params,
        'model_size_mb': model_size_mb
    })
    
    normalizer = ObservationNormalizer(features_per_timestep)
    if normalizer.load(NORM_STATS_PATH):
        print(f"Loaded normalization stats from {NORM_STATS_PATH} (count: {normalizer.count})")
    else:
        print("Starting with fresh normalization statistics")
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    
    os.makedirs("models", exist_ok=True)
    
    best_loss = float('inf')
    best_allocation_accuracy = 0.0
    best_confidence_accuracy = 0.0
    
    print(f"Found {len(data_files)} data files")
    
    global_step = 0
    first_batch_logged = False
    
    for epoch in range(1000):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*60}")
        
        for file_idx, filename in enumerate(data_files):
            filepath = os.path.join(DATA_FOLDER, filename)
            print(f"\nProcessing {filename} ({file_idx + 1}/{len(data_files)})")
            
            chunk_offset = 0
            chunk_num = 0
            
            while True:
                samples = load_samples_from_file(filepath, offset=chunk_offset, max_samples=SAMPLES_PER_FILE)
                if not samples:
                    break
                
                chunk_num += 1
                chunk_offset += len(samples)
                print(f"  Chunk {chunk_num}: Loaded {len(samples)} samples (offset: {chunk_offset - len(samples)})")
                
                for chunk_epoch in range(EPOCHS_PER_CHUNK):
                    backup_weights = [w.numpy() for w in model.trainable_weights]
                    
                    num_batches = len(samples) // BATCH_SIZE
                    epoch_loss = 0.0
                    epoch_alloc_acc = 0.0
                    epoch_conf_acc = 0.0
                    nan_detected = False
                    
                    for batch_idx in range(num_batches):
                        batch_samples = samples[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
                        
                        try:
                            obs_batch, action_batch, confidence_batch = create_batch(batch_samples)
                            
                            normalizer.update(obs_batch)
                            obs_batch_norm = normalizer.normalize(obs_batch).astype(np.float32)
                            
                            if not first_batch_logged:
                                print(f"\n  First batch shapes:")
                                print(f"    obs_batch: {obs_batch.shape}")
                                print(f"    action_batch: {action_batch.shape}")
                                print(f"    confidence_batch: {confidence_batch.shape}")
                                print(f"    obs_batch raw stats: min={obs_batch.min():.4f}, max={obs_batch.max():.4f}, mean={obs_batch.mean():.4f}")
                                print(f"    obs_batch norm stats: min={obs_batch_norm.min():.4f}, max={obs_batch_norm.max():.4f}, mean={obs_batch_norm.mean():.4f}")
                                first_batch_logged = True
                            
                            total_loss, allocation_loss, confidence_loss, ce_loss, top_k_loss, allocation_accuracy, confidence_accuracy = train_step(
                                model, optimizer, obs_batch_norm, action_batch, confidence_batch
                            )
                            
                            if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss):
                                print(f"\n    NaN/Inf detected in loss at step {global_step}!")
                                print("    Rolling back to previous state...")
                                for w, backup in zip(model.trainable_weights, backup_weights):
                                    w.assign(backup)
                                nan_detected = True
                                break
                            
                            epoch_loss += total_loss.numpy()
                            epoch_alloc_acc += allocation_accuracy.numpy()
                            epoch_conf_acc += confidence_accuracy.numpy()
                            global_step += 1
                            
                            if global_step % LOG_INTERVAL == 0:
                                wandb.log({
                                    'loss': total_loss.numpy(),
                                    'allocation_loss': allocation_loss.numpy(),
                                    'confidence_loss': confidence_loss.numpy(),
                                    'ce_loss': ce_loss.numpy(),
                                    'top_k_loss': top_k_loss.numpy(),
                                    'allocation_accuracy': allocation_accuracy.numpy(),
                                    'confidence_accuracy': confidence_accuracy.numpy(),
                                    'step': global_step
                                })
                                
                                print(f"  Step {global_step}: Loss={total_loss.numpy():.6f}, "
                                      f"Alloc={allocation_loss.numpy():.6f}, "
                                      f"Conf={confidence_loss.numpy():.6f}, "
                                      f"CE={ce_loss.numpy():.6f}, "
                                      f"TopK={top_k_loss.numpy():.6f}, "
                                      f"AllocAcc={allocation_accuracy.numpy():.4f}, "
                                      f"ConfAcc={confidence_accuracy.numpy():.4f}")
                        
                        except Exception as e:
                            print(f"\n  Error during training at step {global_step}: {e}")
                            print("  Rolling back to previous state...")
                            for w, backup in zip(model.trainable_weights, backup_weights):
                                w.assign(backup)
                            nan_detected = True
                            break
                    
                    if nan_detected:
                        print(f"  Skipping to next chunk due to NaN/error")
                        break
                    
                    if num_batches > 0:
                        avg_loss = epoch_loss / num_batches
                        avg_alloc_acc = epoch_alloc_acc / num_batches
                        avg_conf_acc = epoch_conf_acc / num_batches
                        print(f"  Chunk {chunk_num} Epoch {chunk_epoch + 1}/{EPOCHS_PER_CHUNK}: Avg loss={avg_loss:.6f}, AllocAcc={avg_alloc_acc:.4f}, ConfAcc={avg_conf_acc:.4f}")
                        
                        model_improved = False
                        
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            model_improved = True
                        
                        if avg_alloc_acc > best_allocation_accuracy:
                            best_allocation_accuracy = avg_alloc_acc
                            model_improved = True
                        
                        if avg_conf_acc > best_confidence_accuracy:
                            best_confidence_accuracy = avg_conf_acc
                            model_improved = True
                        
                        if model_improved:
                            save_model_and_config(model, sequence_length, features_per_timestep, 
                                                num_assets, BEST_MODEL_PATH, CONFIG_PATH)
                            normalizer.save(NORM_STATS_PATH)
                            print(f"    New best model saved! Loss: {best_loss:.6f}, AllocAcc: {best_allocation_accuracy:.4f}, ConfAcc: {best_confidence_accuracy:.4f}")
                            wandb.log({'best_loss': best_loss, 'best_allocation_accuracy': best_allocation_accuracy, 'best_confidence_accuracy': best_confidence_accuracy, 'step': global_step})
    
    normalizer.save(NORM_STATS_PATH)
    wandb.finish()
    print("\nTraining completed!")


if __name__ == "__main__":
    train()