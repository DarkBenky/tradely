import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
import wandb
from pathlib import Path
import json

D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
SAMPLES_PER_FILE = 1000
BEST_MODEL_PATH = "models/best_pretrain_transformer.h5"
CONFIG_PATH = "models/model_config.json"
DATA_FOLDER = "syntheticData"
LOG_INTERVAL = 10
WANDB_PROJECT = "portfolio-pretrain"


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


def load_samples_from_file(filepath, max_samples=None):
    samples = []
    with open(filepath, 'rb') as f:
        count = 0
        while True:
            if max_samples and count >= max_samples:
                break
            try:
                sample = pickle.load(f)
                samples.append(sample)
                count += 1
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
    
    return obs_batch, action_batch, confidence_batch


def compute_loss(model, obs_batch, action_batch, confidence_batch, training=True):
    pred_allocation, pred_confidence = model(obs_batch, training=training)
    
    allocation_loss = tf.reduce_mean(tf.square(pred_allocation - action_batch))
    
    confidence_loss = tf.reduce_mean(tf.square(pred_confidence - confidence_batch))
    
    kl_loss = tf.reduce_mean(
        tf.keras.losses.kl_divergence(action_batch, pred_allocation)
    )
    
    total_loss = allocation_loss + 0.5 * confidence_loss + 0.3 * kl_loss
    
    return total_loss, allocation_loss, confidence_loss, kl_loss


@tf.function
def train_step(model, optimizer, obs_batch, action_batch, confidence_batch):
    with tf.GradientTape() as tape:
        total_loss, allocation_loss, confidence_loss, kl_loss = compute_loss(
            model, obs_batch, action_batch, confidence_batch, training=True
        )
    
    gradients = tape.gradient(total_loss, model.trainable_weights)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    return total_loss, allocation_loss, confidence_loss, kl_loss


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
        'num_assets': num_assets
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
    
    num_params = model.count_parameters()
    model_size_mb = num_params * 4 / (1024 * 1024)
    
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB (assuming float32)")
    
    wandb.config.update({
        'num_parameters': num_params,
        'model_size_mb': model_size_mb
    })
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    
    os.makedirs("models", exist_ok=True)
    
    best_loss = float('inf')
    
    print(f"Found {len(data_files)} data files")
    
    global_step = 0
    
    for epoch in range(1000):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*60}")
        
        for file_idx, filename in enumerate(data_files):
            filepath = os.path.join(DATA_FOLDER, filename)
            print(f"\nLoading {filename} ({file_idx + 1}/{len(data_files)})")
            
            samples = load_samples_from_file(filepath, max_samples=SAMPLES_PER_FILE)
            if not samples:
                print(f"No samples in {filename}, skipping")
                continue
            
            print(f"Loaded {len(samples)} samples")
            
            backup_weights = [w.numpy() for w in model.trainable_weights]
            
            num_batches = len(samples) // BATCH_SIZE
            epoch_loss = 0.0
            nan_detected = False
            
            for batch_idx in range(num_batches):
                batch_samples = samples[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
                
                try:
                    obs_batch, action_batch, confidence_batch = create_batch(batch_samples)
                    
                    total_loss, allocation_loss, confidence_loss, kl_loss = train_step(
                        model, optimizer, obs_batch, action_batch, confidence_batch
                    )
                    
                    if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss):
                        print(f"\nNaN/Inf detected in loss at step {global_step}!")
                        print("Rolling back to previous state...")
                        for w, backup in zip(model.trainable_weights, backup_weights):
                            w.assign(backup)
                        nan_detected = True
                        break
                    
                    epoch_loss += total_loss.numpy()
                    global_step += 1
                    
                    if global_step % LOG_INTERVAL == 0:
                        wandb.log({
                            'loss': total_loss.numpy(),
                            'allocation_loss': allocation_loss.numpy(),
                            'confidence_loss': confidence_loss.numpy(),
                            'kl_loss': kl_loss.numpy(),
                            'step': global_step
                        })
                        
                        print(f"Step {global_step}: Loss={total_loss.numpy():.6f}, "
                              f"Alloc={allocation_loss.numpy():.6f}, "
                              f"Conf={confidence_loss.numpy():.6f}, "
                              f"KL={kl_loss.numpy():.6f}")
                
                except Exception as e:
                    print(f"\nError during training at step {global_step}: {e}")
                    print("Rolling back to previous state...")
                    for w, backup in zip(model.trainable_weights, backup_weights):
                        w.assign(backup)
                    nan_detected = True
                    break
            
            if nan_detected:
                print(f"Skipping to next file due to NaN/error")
                continue
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"Average loss for {filename}: {avg_loss:.6f}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_and_config(model, sequence_length, features_per_timestep, 
                                        num_assets, BEST_MODEL_PATH, CONFIG_PATH)
                    print(f"New best model saved! Loss: {best_loss:.6f}")
                    wandb.log({'best_loss': best_loss, 'step': global_step})
    
    wandb.finish()
    print("\nTraining completed!")


if __name__ == "__main__":
    train()