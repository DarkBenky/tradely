import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
import wandb
import json
from wandb.integration.keras import WandbMetricsLogger

D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 5
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
SAMPLES_PER_FILE = 4096
BEST_MODEL_PATH = "models/best_pretrain_transformer.weights.h5"
CONFIG_PATH = "models/model_config.json"
NORM_STATS_PATH = "models/normalization_stats.json"
DATA_FOLDER = "syntheticData"
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
    def __init__(self, d_model, num_heads, dim_feedforward, dropout_rate, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        
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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dim_feedforward': self.dim_feedforward,
            'dropout_rate': self.dropout_rate
        })
        return config


def build_model(sequence_length, features_per_timestep, num_assets, d_model=D_MODEL, 
                num_heads=NHEAD, num_layers=NUM_ENCODER_LAYERS, 
                dim_feedforward=DIM_FEEDFORWARD, dropout_rate=DROPOUT):
    
    inputs = layers.Input(shape=(sequence_length, features_per_timestep))
    
    x = layers.Dense(d_model)(inputs)
    
    pos_encoding = tf.Variable(
        tf.random.normal([1, sequence_length, d_model], stddev=0.02),
        trainable=True, name='pos_encoding'
    )
    x = x + pos_encoding
    
    for i in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, dim_feedforward, dropout_rate, name=f'encoder_{i}')(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    alloc_hidden = layers.Dense(dim_feedforward // 2, activation='relu')(x)
    alloc_hidden = layers.Dropout(dropout_rate)(alloc_hidden)
    allocation_output = layers.Dense(num_assets, activation='softmax', name='allocation')(alloc_hidden)
    
    conf_hidden = layers.Dense(dim_feedforward // 2, activation='relu')(x)
    conf_hidden = layers.Dropout(dropout_rate)(conf_hidden)
    confidence_output = layers.Dense(num_assets, activation='sigmoid', name='confidence')(conf_hidden)
    
    model = keras.Model(inputs=inputs, outputs=[allocation_output, confidence_output])
    
    return model


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


def prepare_data(samples, normalizer):
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
    
    normalizer.update(obs_batch)
    obs_batch_norm = normalizer.normalize(obs_batch).astype(np.float32)
    
    return obs_batch_norm, action_batch, confidence_batch


class TopKAccuracy(keras.metrics.Metric):
    def __init__(self, k=3, name='top_k_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        true_top = tf.argsort(y_true, axis=-1, direction='DESCENDING')[:, :self.k]
        pred_top = tf.argsort(y_pred, axis=-1, direction='DESCENDING')[:, :self.k]
        
        matches = tf.reduce_sum(tf.cast(
            tf.equal(tf.expand_dims(pred_top, 2), tf.expand_dims(true_top, 1)), tf.float32
        ), axis=[1, 2])
        accuracy = matches / tf.cast(self.k, tf.float32)
        
        self.total.assign_add(tf.reduce_sum(accuracy))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        return self.total / (self.count + 1e-7)
    
    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


def allocation_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    
    epsilon = 1e-7
    y_pred_safe = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_true_safe = tf.clip_by_value(y_true, epsilon, 1.0 - epsilon)
    ce = -tf.reduce_mean(tf.reduce_sum(y_true_safe * tf.math.log(y_pred_safe), axis=1))
    
    top_k = 3
    _, target_top_indices = tf.nn.top_k(y_true, k=top_k)
    pred_at_tops = tf.gather(y_pred, target_top_indices, batch_dims=1)
    true_at_tops = tf.gather(y_true, target_top_indices, batch_dims=1)
    top_k_loss = tf.reduce_mean(tf.square(pred_at_tops - true_at_tops))
    
    return mse + 0.3 * ce + 0.5 * top_k_loss


def save_model_config(sequence_length, features_per_timestep, num_assets, config_path):
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
    
    wandb.init(project=WANDB_PROJECT, config={
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
        'epochs_per_chunk': EPOCHS_PER_CHUNK
    })
    
    model = build_model(sequence_length, features_per_timestep, num_assets)
    
    if os.path.exists(BEST_MODEL_PATH):
        print(f"\nLoading weights from {BEST_MODEL_PATH}")
        try:
            model.load_weights(BEST_MODEL_PATH)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Could not load weights: {e}")
    
    model.summary()
    
    optimizer = keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'allocation': allocation_loss,
            'confidence': 'mse'
        },
        loss_weights={
            'allocation': 1.0,
            'confidence': 0.5
        },
        metrics={
            'allocation': ['mae', TopKAccuracy(k=1, name='top1_acc'), TopKAccuracy(k=3, name='top3_acc')],
            'confidence': ['mae']
        }
    )
    
    normalizer = ObservationNormalizer(features_per_timestep)
    if normalizer.load(NORM_STATS_PATH):
        print(f"Loaded normalization stats (count: {normalizer.count})")
    else:
        print("Starting with fresh normalization statistics")
    
    os.makedirs("models", exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        WandbMetricsLogger()
    ]
    
    print(f"\nFound {len(data_files)} data files")
    
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
                print(f"  Chunk {chunk_num}: {len(samples)} samples")
                
                X, y_alloc, y_conf = prepare_data(samples, normalizer)
                
                if chunk_num == 1 and file_idx == 0 and epoch == 0:
                    print(f"    X shape: {X.shape}, stats: min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
                
                model.fit(
                    X,
                    {'allocation': y_alloc, 'confidence': y_conf},
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS_PER_CHUNK,
                    callbacks=callbacks,
                    verbose=1
                )
                
                normalizer.save(NORM_STATS_PATH)
                save_model_config(sequence_length, features_per_timestep, num_assets, CONFIG_PATH)
    
    wandb.finish()
    print("\nTraining completed!")


if __name__ == "__main__":
    train()