import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
import wandb
import json
from wandb.integration.keras import WandbMetricsLogger
import matplotlib.pyplot as plt

tf.keras.mixed_precision.set_global_policy('mixed_float16')

MODEL_VERSIONS = ['v1-transformer', 'v2-cnn-transformer-cnn']

MODEL_TYPE = MODEL_VERSIONS[1]
TEST = False
NUM_PREDICTIONS = 8
D_MODEL = 512
NHEAD = 16
NUM_BLOCKS = 12
CNN_FILTERS = 512
CNN_KERNEL = 5
DROPOUT = 0.2
LEARNING_RATE = 0.000175
BATCH_SIZE = 2
ACCUMULATION_STEPS = 8
GRADIENT_CLIP_NORM = 1.0
SAMPLES_PER_FILE = 512
BEST_MODEL_PATH = "models/best_pretrain_transformer.weights.h5"
CONFIG_PATH = "models/model_config.json"
NORM_STATS_PATH = "models/normalization_stats.json"
DATA_FOLDER = "syntheticData"
WANDB_PROJECT = "portfolio-pretrain"
EPOCHS_PER_CHUNK = 5
TEST_STEPS = 64
SAVE_AVG_WINDOW = 5
TOP_1_WEIGHT = 0.3
TOP_K_WEIGHT = 0.5
REWARD_WEIGHT = 0.2

REWARD_BUCKETS = [-250,-100, -25, 0, 25, 50, 100, 200, 400]
NUM_REWARD_BUCKETS = len(REWARD_BUCKETS) + 1


FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
    'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'ma_5', 'ma_10', 'ma_20', 'ema_5', 'ema_10', 'ema_20',
    'rsi', 'macd_line', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'vwap',
    'volume_profile', 'hvn_level_1', 'hvn_strength_1', 'lvn_level_1',
    'hvn_level_2', 'hvn_strength_2', 'lvn_level_2',
    'hvn_level_3', 'hvn_strength_3', 'lvn_level_3',
    'hvn_level_4', 'hvn_strength_4', 'lvn_level_4',
    'hvn_level_5', 'hvn_strength_5', 'lvn_level_5',
    'time_profile', 'htn_level_1', 'htn_strength_1', 'ltn_level_1',
    'htn_level_2', 'htn_strength_2', 'ltn_level_2',
    'htn_level_3', 'htn_strength_3', 'ltn_level_3',
    'htn_level_4', 'htn_strength_4', 'ltn_level_4',
    'htn_level_5', 'htn_strength_5', 'ltn_level_5',
    'trade_profile', 'trade_htn_level_1', 'trade_htn_strength_1', 'trade_ltn_level_1',
    'trade_htn_level_2', 'trade_htn_strength_2', 'trade_ltn_level_2',
    'trade_htn_level_3', 'trade_htn_strength_3', 'trade_ltn_level_3',
    'trade_htn_level_4', 'trade_htn_strength_4', 'trade_ltn_level_4',
    'trade_htn_level_5', 'trade_htn_strength_5', 'trade_ltn_level_5'
]

ASSET_NAMES = ['ADAUSDT', 'AVAXUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 
               'DOTUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT', 'XRPUSDT']

PORTFOLIO_STATE_NAMES = ['cash_pct', 'portfolio_value_ratio', 'benchmark_ratio'] + \
                        [f'{asset}_holdings_pct' for asset in ASSET_NAMES]


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
    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(d_model * 2, activation='gelu'),
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
            'dropout_rate': self.dropout_rate
        })
        return config


class CNNTransformerBlock(layers.Layer):
    def __init__(self, filters, kernel_size, d_model, num_heads, dropout_rate, axis='time', **kwargs):
        super(CNNTransformerBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.axis = axis
        
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same', activation='gelu')
        self.bn1 = layers.BatchNormalization()
        
        self.transformer = TransformerEncoderBlock(d_model, num_heads, dropout_rate)
        
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same', activation='gelu')
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        if self.axis == 'feature':
            x = tf.transpose(x, [0, 2, 1])
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        
        x = self.transformer(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout(x, training=training)
        
        if self.axis == 'feature':
            x = tf.transpose(x, [0, 2, 1])
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'axis': self.axis
        })
        return config


def build_model_v2(sequence_length, features_per_timestep, num_assets,
                   d_model=D_MODEL, num_heads=NHEAD, num_blocks=NUM_BLOCKS,
                   cnn_filters=CNN_FILTERS, cnn_kernel=CNN_KERNEL, dropout_rate=DROPOUT):
    
    inputs = layers.Input(shape=(sequence_length, features_per_timestep))
    
    x = layers.Dense(d_model)(inputs)
    
    for i in range(num_blocks):
        axis = 'time' if i % 2 == 0 else 'feature'
        x = CNNTransformerBlock(
            cnn_filters, cnn_kernel, d_model, num_heads, dropout_rate,
            axis=axis, name=f'cnn_transformer_block_{i}'
        )(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(d_model, activation='gelu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    alloc_logits = layers.Dense(num_assets, dtype='float32')(x)
    allocation_output = layers.Activation('softmax', dtype='float32', name='allocation')(alloc_logits)
    
    reward_logits = layers.Dense(NUM_REWARD_BUCKETS, dtype='float32')(x)
    reward_output = layers.Activation('softmax', dtype='float32', name='reward')(reward_logits)
    
    model = keras.Model(inputs=inputs, outputs=[allocation_output, reward_output])
    
    return model


def build_model(sequence_length, features_per_timestep, num_assets, d_model=D_MODEL, 
                num_heads=NHEAD, num_layers=NUM_BLOCKS,
                dim_feedforward=D_MODEL*2, dropout_rate=DROPOUT):
    
    inputs = layers.Input(shape=(sequence_length, features_per_timestep))
    
    x = layers.Dense(d_model)(inputs)
    
    pos_encoding = tf.Variable(
        tf.random.normal([1, sequence_length, d_model], stddev=0.02),
        trainable=True, name='pos_encoding'
    )
    x = x + pos_encoding
    
    for i in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, dropout_rate, name=f'encoder_{i}')(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    alloc_hidden = layers.Dense(dim_feedforward // 2, activation='gelu')(x)
    alloc_hidden = layers.Dropout(dropout_rate)(alloc_hidden)
    alloc_logits = layers.Dense(num_assets, dtype='float32')(alloc_hidden)
    allocation_output = layers.Activation('softmax', dtype='float32', name='allocation')(alloc_logits)
    
    reward_hidden = layers.Dense(dim_feedforward // 2, activation='gelu')(x)
    reward_hidden = layers.Dropout(dropout_rate)(reward_hidden)
    reward_logits = layers.Dense(NUM_REWARD_BUCKETS, dtype='float32')(reward_hidden)
    reward_output = layers.Activation('softmax', dtype='float32', name='reward')(reward_logits)
    
    model = keras.Model(inputs=inputs, outputs=[allocation_output, reward_output])
    
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


def reward_to_bucket(rewards):
    bucket_indices = np.zeros(len(rewards), dtype=np.int32)
    for i, threshold in enumerate(REWARD_BUCKETS):
        bucket_indices += (rewards >= threshold).astype(np.int32)
    return bucket_indices


def bucket_to_reward_estimate(bucket_probs):
    bucket_centers = np.array([-350, -175, -62.5, -12.5, 12.5, 37.5, 75, 150, 300, 500], dtype=np.float32)
    return np.sum(bucket_probs * bucket_centers, axis=-1)


def prepare_data(samples, normalizer):
    obs_list = []
    action_list = []
    reward_list = []
    
    for sample in samples:
        obs_list.append(sample['obs'])
        action_list.append(sample['action'])
        reward_list.append(sample.get('reward', 0.0))
    
    obs_batch = np.array(obs_list, dtype=np.float32)
    action_batch = np.array(action_list, dtype=np.float32)
    rewards_raw = np.array(reward_list, dtype=np.float32)
    
    bucket_indices = reward_to_bucket(rewards_raw)
    reward_onehot = np.eye(NUM_REWARD_BUCKETS, dtype=np.float32)[bucket_indices]
    
    normalizer.update(obs_batch)
    obs_batch_norm = normalizer.normalize(obs_batch).astype(np.float32)
    
    return obs_batch_norm, action_batch, reward_onehot


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
    epsilon = 1e-7
    y_pred_safe = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_true_safe = tf.clip_by_value(y_true, epsilon, 1.0 - epsilon)
    
    _, top1_indices = tf.nn.top_k(y_true, k=1)
    pred_at_top1 = tf.gather(y_pred, top1_indices, batch_dims=1)
    true_at_top1 = tf.gather(y_true, top1_indices, batch_dims=1)
    top1_loss = tf.reduce_mean(tf.square(pred_at_top1 - true_at_top1))
    
    _, top3_indices = tf.nn.top_k(y_true, k=3)
    pred_at_top3 = tf.gather(y_pred, top3_indices, batch_dims=1)
    true_at_top3 = tf.gather(y_true, top3_indices, batch_dims=1)
    top3_loss = tf.reduce_mean(tf.square(pred_at_top3 - true_at_top3))
    
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    
    total_alloc = TOP_1_WEIGHT + TOP_K_WEIGHT
    w1 = TOP_1_WEIGHT / total_alloc
    w3 = TOP_K_WEIGHT / total_alloc
    
    return w1 * top1_loss + w3 * top3_loss + 0.1 * mse


def save_model_config(sequence_length, features_per_timestep, num_assets, config_path):
    config = {
        'sequence_length': int(sequence_length),
        'features_per_timestep': int(features_per_timestep),
        'num_assets': int(num_assets),
        'd_model': D_MODEL,
        'num_heads': NHEAD,
        'num_layers': NUM_BLOCKS,
        'cnn_filters': CNN_FILTERS,
        'cnn_kernel': CNN_KERNEL,
        'dropout_rate': DROPOUT,
        'model_type': MODEL_TYPE
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


class RollingAverageCallback(keras.callbacks.Callback):
    def __init__(self, model_path, window_size=SAVE_AVG_WINDOW):
        super().__init__()
        self.model_path = model_path
        self.window_size = window_size
        self.loss_history = []
        self.best_avg_loss = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        current_loss = logs.get('loss', float('inf'))
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) >= self.window_size:
            avg_loss = np.mean(self.loss_history[-self.window_size:])
            
            if avg_loss < self.best_avg_loss:
                self.best_avg_loss = avg_loss
                self.model.save_weights(self.model_path)
                print(f"\nModel saved - avg loss over {self.window_size} epochs: {avg_loss:.6f}")
                
                wandb.log({
                    'best_avg_loss': self.best_avg_loss,
                    'current_avg_loss': avg_loss
                })


class DetailedLoggingCallback(keras.callbacks.Callback):
    def __init__(self, log_activations_every=20, log_weights_every=100):
        super().__init__()
        self.batch_count = 0
        self.log_activations_every = log_activations_every
        self.log_weights_every = log_weights_every
        self.sample_input = None
        self.sample_output = None
        self.feature_labels = self._build_feature_labels()
    
    def _build_feature_labels(self):
        labels = []
        for asset in ASSET_NAMES:
            for feat in FEATURE_NAMES:
                labels.append(f'{asset}_{feat}')
        for pstate in PORTFOLIO_STATE_NAMES:
            labels.append(pstate)
        return labels
    
    def set_sample_input(self, sample_input, sample_output=None):
        self.sample_input = sample_input
        self.sample_output = sample_output
    
    def _create_heatmap(self, data, title, xlabel='Features', ylabel='Sequence'):
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, aspect='auto', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig
    
    def _log_activations(self):
        if self.sample_input is None:
            return
        
        sample = self.sample_input[:2]
        log_data = {}
        
        x = sample
        for layer in self.model.layers:
            if isinstance(layer, layers.InputLayer):
                continue
            
            x = layer(x, training=False)
            
            if isinstance(layer, CNNTransformerBlock):
                act_np = x.numpy()
                name = layer.name
                
                log_data[f'activation/{name}/mean'] = float(np.mean(act_np))
                log_data[f'activation/{name}/std'] = float(np.std(act_np))
                log_data[f'activation/{name}/min'] = float(np.min(act_np))
                log_data[f'activation/{name}/max'] = float(np.max(act_np))
                
                sample_act = act_np[0]
                if sample_act.shape[1] > 64:
                    step = sample_act.shape[1] // 64
                    sample_act = sample_act[:, ::step]
                if sample_act.shape[0] > 64:
                    step = sample_act.shape[0] // 64
                    sample_act = sample_act[::step, :]
                
                fig = self._create_heatmap(sample_act, name)
                log_data[f'activation/{name}/heatmap'] = wandb.Image(fig)
                plt.close(fig)
            
            if isinstance(layer, layers.GlobalAveragePooling1D):
                break
        
        if log_data:
            wandb.log(log_data)
        
        wandb.log(log_data)
    
    def _log_weight_importance(self):
        log_data = {}
        
        input_layer = None
        for layer in self.model.layers:
            if isinstance(layer, layers.Dense) and hasattr(layer, 'kernel') and layer.kernel is not None:
                if layer.kernel.shape[0] == len(self.feature_labels):
                    input_layer = layer
                    break
        
        if input_layer is not None and hasattr(input_layer, 'kernel'):
            kernel = input_layer.kernel.numpy()
            importance = np.mean(np.abs(kernel), axis=1)
            
            n_features = len(FEATURE_NAMES)
            n_assets = len(ASSET_NAMES)
            
            asset_importance = np.zeros(n_assets)
            for i, asset in enumerate(ASSET_NAMES):
                start_idx = i * n_features
                end_idx = start_idx + n_features
                if end_idx <= len(importance):
                    asset_importance[i] = np.mean(importance[start_idx:end_idx])
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(range(n_assets), asset_importance)
            ax.set_xticks(range(n_assets))
            ax.set_xticklabels(ASSET_NAMES, rotation=45, ha='right')
            ax.set_ylabel('Mean abs weight')
            ax.set_title('Input importance by asset')
            plt.tight_layout()
            log_data['weights/input_asset_importance'] = wandb.Image(fig)
            plt.close(fig)
            
            feature_importance = np.zeros(n_features)
            for i in range(n_features):
                indices = [a * n_features + i for a in range(n_assets) if a * n_features + i < len(importance)]
                if indices:
                    feature_importance[i] = np.mean(importance[indices])
            
            top_k = 20
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            top_names = [FEATURE_NAMES[i] for i in top_indices]
            top_values = feature_importance[top_indices]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(top_names)), top_values)
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names)
            ax.set_xlabel('Mean abs weight')
            ax.set_title(f'Top {top_k} feature importance')
            ax.invert_yaxis()
            plt.tight_layout()
            log_data['weights/input_feature_importance'] = wandb.Image(fig)
            plt.close(fig)
        
        for layer in self.model.layers:
            if isinstance(layer, CNNTransformerBlock):
                for sublayer in [layer.conv1, layer.conv2]:
                    if hasattr(sublayer, 'kernel'):
                        kernel = sublayer.kernel.numpy()
                        importance = np.mean(np.abs(kernel), axis=(0, 1))
                        
                        log_data[f'weights/{layer.name}_{sublayer.name}_mean'] = float(np.mean(np.abs(kernel)))
                        log_data[f'weights/{layer.name}_{sublayer.name}_std'] = float(np.std(kernel))
        
        wandb.log(log_data)
    
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            return
        
        self.batch_count += 1
        
        log_data = {'batch': self.batch_count}
        for key, value in logs.items():
            log_data[f'batch_{key}'] = float(value)
        
        wandb.log(log_data)
        
        if self.batch_count % self.log_activations_every == 0:
            self._log_activations()
        
        if self.batch_count % self.log_weights_every == 0:
            self._log_weight_importance()
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        log_data = {'epoch': epoch}
        for key, value in logs.items():
            log_data[f'epoch_{key}'] = float(value)
        
        wandb.log(log_data)


def train():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPUs available: {len(physical_devices)}")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            try:
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
                )
            except:
                pass
    else:
        print("No GPU detected, using CPU")
    
    print(f"\nScanning data folder: {DATA_FOLDER}")
    sequence_length, features_per_timestep, num_assets, data_files = scan_data_folder(DATA_FOLDER)
    
    wandb.init(project=WANDB_PROJECT, config={
        'model_type': MODEL_TYPE,
        'sequence_length': sequence_length,
        'features_per_timestep': features_per_timestep,
        'd_model': D_MODEL,
        'nhead': NHEAD,
        'num_blocks': NUM_BLOCKS,
        'cnn_filters': CNN_FILTERS,
        'cnn_kernel': CNN_KERNEL,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'accumulation_steps': ACCUMULATION_STEPS,
        'effective_batch_size': BATCH_SIZE * ACCUMULATION_STEPS,
        'gradient_clip_norm': GRADIENT_CLIP_NORM,
        'num_assets': num_assets,
        'samples_per_file': SAMPLES_PER_FILE,
        'epochs_per_chunk': EPOCHS_PER_CHUNK,
        'save_avg_window': SAVE_AVG_WINDOW,
        'top_1_weight': TOP_1_WEIGHT,
        'top_k_weight': TOP_K_WEIGHT,
        'reward_weight': REWARD_WEIGHT,
        'reward_buckets': REWARD_BUCKETS,
        'num_reward_buckets': NUM_REWARD_BUCKETS
    })
    
    if MODEL_TYPE == 'v2-cnn-transformer-cnn':
        model = build_model_v2(sequence_length, features_per_timestep, num_assets)
    else:
        model = build_model(sequence_length, features_per_timestep, num_assets)
    
    if os.path.exists(BEST_MODEL_PATH):
        print(f"\nLoading weights from {BEST_MODEL_PATH}")
        try:
            model.load_weights(BEST_MODEL_PATH)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Could not load weights: {e}")
    
    model.summary()
    
    num_params = sum([tf.reduce_prod(w.shape).numpy() for w in model.trainable_weights])
    model_size_mb = num_params * 2 / (1024 * 1024)
    print(f"\nModel parameters: {num_params:,}")
    print(f"Model size (float16): {model_size_mb:.2f} MB")
    
    optimizer = keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'allocation': allocation_loss,
            'reward': keras.losses.CategoricalCrossentropy()
        },
        loss_weights={
            'allocation': TOP_K_WEIGHT,
            'reward': REWARD_WEIGHT
        },
        metrics={
            'allocation': ['mae', TopKAccuracy(k=1, name='top1_acc'), TopKAccuracy(k=3, name='top3_acc')],
            'reward': ['accuracy']
        }
    )
    
    normalizer = ObservationNormalizer(features_per_timestep)
    if normalizer.load(NORM_STATS_PATH):
        print(f"Loaded normalization stats (count: {normalizer.count})")
    else:
        print("Starting with fresh normalization statistics")
    
    os.makedirs("models", exist_ok=True)
    
    rolling_avg_callback = RollingAverageCallback(BEST_MODEL_PATH, SAVE_AVG_WINDOW)
    detailed_logging_callback = DetailedLoggingCallback(log_activations_every=500, log_weights_every=5000)
    
    callbacks = [
        rolling_avg_callback,
        detailed_logging_callback,
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TerminateOnNaN()
    ]
    
    print(f"\nFound {len(data_files)} data files")
    
    for epoch in range(1000):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*60}")
        
        shuffled_files = data_files.copy()
        np.random.shuffle(shuffled_files)
        
        for file_idx, filename in enumerate(shuffled_files):
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
                
                X, y_alloc, y_reward = prepare_data(samples, normalizer)
                
                detailed_logging_callback.set_sample_input(X)
                
                if chunk_num == 1 and file_idx == 0 and epoch == 0:
                    print(f"    X shape: {X.shape}, stats: min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
                
                num_samples = len(X)
                indices = np.arange(num_samples)
                
                for sub_epoch in range(EPOCHS_PER_CHUNK):
                    np.random.shuffle(indices)
                    epoch_loss = 0.0
                    epoch_alloc_loss = 0.0
                    epoch_reward_loss = 0.0
                    num_batches = 0
                    
                    for start_idx in range(0, num_samples, BATCH_SIZE * ACCUMULATION_STEPS):
                        batch_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
                        accum_loss = 0.0
                        accum_alloc_loss = 0.0
                        accum_reward_loss = 0.0
                        
                        for accum_step in range(ACCUMULATION_STEPS):
                            step_start = start_idx + accum_step * BATCH_SIZE
                            step_end = min(step_start + BATCH_SIZE, num_samples)
                            
                            if step_start >= num_samples:
                                break
                            
                            batch_indices = indices[step_start:step_end]
                            X_batch = tf.constant(X[batch_indices])
                            y_alloc_batch = tf.constant(y_alloc[batch_indices])
                            y_reward_batch = tf.constant(y_reward[batch_indices])
                            
                            with tf.GradientTape() as tape:
                                alloc_pred, reward_pred = model(X_batch, training=True)
                                
                                alloc_loss = allocation_loss(y_alloc_batch, alloc_pred)
                                reward_loss = keras.losses.categorical_crossentropy(y_reward_batch, reward_pred)
                                reward_loss = tf.reduce_mean(reward_loss)
                                
                                total_loss = (TOP_K_WEIGHT * alloc_loss + REWARD_WEIGHT * reward_loss) / ACCUMULATION_STEPS
                            
                            gradients = tape.gradient(total_loss, model.trainable_variables)
                            
                            for i, grad in enumerate(gradients):
                                if grad is not None:
                                    batch_gradients[i] = batch_gradients[i] + grad
                            
                            accum_loss += total_loss.numpy() * ACCUMULATION_STEPS
                            accum_alloc_loss += alloc_loss.numpy()
                            accum_reward_loss += reward_loss.numpy()
                            
                            del X_batch, y_alloc_batch, y_reward_batch
                            del alloc_pred, reward_pred, alloc_loss, reward_loss, total_loss
                            del gradients
                        
                        batch_gradients, _ = tf.clip_by_global_norm(batch_gradients, GRADIENT_CLIP_NORM)
                        
                        model.optimizer.apply_gradients(zip(batch_gradients, model.trainable_variables))
                        
                        del batch_gradients
                        
                        epoch_loss += accum_loss
                        epoch_alloc_loss += accum_alloc_loss
                        epoch_reward_loss += accum_reward_loss
                        num_batches += 1
                        
                        detailed_logging_callback.batch_count += 1
                        if detailed_logging_callback.batch_count % detailed_logging_callback.log_activations_every == 0:
                            detailed_logging_callback._log_activations()
                        if detailed_logging_callback.batch_count % detailed_logging_callback.log_weights_every == 0:
                            detailed_logging_callback._log_weight_importance()
                        
                        wandb.log({
                            'batch': detailed_logging_callback.batch_count,
                            'batch_loss': accum_loss / ACCUMULATION_STEPS,
                            'batch_allocation_loss': accum_alloc_loss / ACCUMULATION_STEPS,
                            'batch_reward_loss': accum_reward_loss / ACCUMULATION_STEPS
                        })
                    
                    if num_batches > 0:
                        avg_loss = epoch_loss / num_batches
                        avg_alloc_loss = epoch_alloc_loss / num_batches
                        avg_reward_loss = epoch_reward_loss / num_batches
                        
                        print(f"    Epoch {sub_epoch+1}/{EPOCHS_PER_CHUNK} - loss: {avg_loss:.4f} - alloc_loss: {avg_alloc_loss:.4f} - reward_loss: {avg_reward_loss:.4f}")
                        
                        wandb.log({
                            'epoch_loss': avg_loss,
                            'epoch_allocation_loss': avg_alloc_loss,
                            'epoch_reward_loss': avg_reward_loss
                        })
                        
                        rolling_avg_callback.loss_history.append(avg_loss)
                        if len(rolling_avg_callback.loss_history) >= rolling_avg_callback.window_size:
                            window_avg = np.mean(rolling_avg_callback.loss_history[-rolling_avg_callback.window_size:])
                            if window_avg < rolling_avg_callback.best_avg_loss:
                                rolling_avg_callback.best_avg_loss = window_avg
                                model.save_weights(rolling_avg_callback.model_path)
                                print(f"    Model saved - avg loss: {window_avg:.6f}")
                
                normalizer.save(NORM_STATS_PATH)
                save_model_config(sequence_length, features_per_timestep, num_assets, CONFIG_PATH)
    
    wandb.finish()
    print("\nTraining completed!")


def evaluate_model():
    from portfolio_env import PortfolioEnv, ModelType
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(BEST_MODEL_PATH):
        print("No trained model found. Creating new model...")
        
        print(f"Scanning data folder: {DATA_FOLDER}")
        sequence_length, features_per_timestep, num_assets, _ = scan_data_folder(DATA_FOLDER)
        
        if MODEL_TYPE == 'v2-cnn-transformer-cnn':
            model = build_model_v2(sequence_length, features_per_timestep, num_assets)
        else:
            model = build_model(sequence_length, features_per_timestep, num_assets)
        
        os.makedirs("models", exist_ok=True)
        model.save_weights(BEST_MODEL_PATH)
        save_model_config(sequence_length, features_per_timestep, num_assets, CONFIG_PATH)
        
        normalizer = ObservationNormalizer(features_per_timestep)
        normalizer.save(NORM_STATS_PATH)
        
        print(f"Created new model and saved to {BEST_MODEL_PATH}")
        print("Note: Model is untrained, results will be random.\n")
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    sequence_length = config['sequence_length']
    features_per_timestep = config['features_per_timestep']
    num_assets = config['num_assets']
    model_type = config.get('model_type', 'v1-transformer')
    
    if model_type == 'v2-cnn-transformer-cnn':
        model = build_model_v2(
            sequence_length, features_per_timestep, num_assets,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_blocks=config['num_layers'],
            cnn_filters=config.get('cnn_filters', CNN_FILTERS),
            cnn_kernel=config.get('cnn_kernel', CNN_KERNEL),
            dropout_rate=config['dropout_rate']
        )
    else:
        model = build_model(
            sequence_length, features_per_timestep, num_assets,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout_rate=config['dropout_rate']
        )
    model.load_weights(BEST_MODEL_PATH)
    print("Model loaded successfully")
    
    normalizer = ObservationNormalizer(features_per_timestep)
    if not normalizer.load(NORM_STATS_PATH):
        print("Warning: No normalization stats found")
    
    env = PortfolioEnv(obs_shape=ModelType.TRANSFORMER_SHAPE)
    
    portfolio_values = []
    benchmark_values = []
    rewards = []
    predicted_rewards = []
    predicted_reward_stds = []
    allocations_history = []
    timestamps = []
    
    obs = env.get_observation()
    initial_portfolio = env._portfolio_portfolio_value()
    initial_benchmark = env._update_benchmark_value()
    
    portfolio_values.append(initial_portfolio)
    benchmark_values.append(initial_benchmark)
    timestamps.append(0)
    
    print(f"\nRunning evaluation for {TEST_STEPS} steps...")
    print(f"Initial portfolio value: ${initial_portfolio:.2f}")
    print(f"Initial benchmark value: ${initial_benchmark:.2f}")
    print(f"Asset names: {env.asset_names}")
    
    for step in range(TEST_STEPS):
        obs_norm = normalizer.normalize(obs.reshape(1, *obs.shape)).astype(np.float32)
        
        allocation_preds = []
        reward_preds = []
        for _ in range(NUM_PREDICTIONS):
            alloc, rew = model(obs_norm, training=True)
            allocation_preds.append(alloc.numpy())
            reward_preds.append(rew.numpy())
        
        allocation_preds = np.array(allocation_preds)
        reward_preds = np.array(reward_preds)
        
        allocation = np.mean(allocation_preds, axis=0)[0]
        mean_bucket_probs = np.mean(reward_preds, axis=0)[0]
        pred_reward = bucket_to_reward_estimate(mean_bucket_probs)
        pred_reward_std = np.std([bucket_to_reward_estimate(r[0]) for r in reward_preds])
        
        predicted_rewards.append(pred_reward)
        predicted_reward_stds.append(pred_reward_std)
        allocations_history.append(allocation.copy())
        
        obs, reward, done, info = env.step(allocation)
        rewards.append(reward)
        
        portfolio_val = env._portfolio_portfolio_value()
        benchmark_val = env._update_benchmark_value()
        portfolio_values.append(portfolio_val)
        benchmark_values.append(benchmark_val)
        timestamps.append(step + 1)
        
        if step % 1 == 0:
            print(f"Step {step}: Portfolio=${portfolio_val:.2f}, Benchmark=${benchmark_val:.2f}, Reward={reward:.2f}")
        
        if done:
            print(f"Episode ended at step {step}: {info}")
            break
    
    portfolio_values = np.array(portfolio_values)
    benchmark_values = np.array(benchmark_values)
    rewards = np.array(rewards)
    predicted_rewards = np.array(predicted_rewards)
    predicted_reward_stds = np.array(predicted_reward_stds)
    allocations_history = np.array(allocations_history)
    
    final_portfolio = portfolio_values[-1]
    final_benchmark = benchmark_values[-1]
    
    portfolio_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100
    benchmark_return = (final_benchmark - initial_benchmark) / initial_benchmark * 100
    excess_return = portfolio_return - benchmark_return
    
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
    
    portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252 * 12)
    benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252 * 12)
    
    risk_free_rate = 0.02
    sharpe_ratio = (np.mean(portfolio_returns) * 252 * 12 - risk_free_rate) / (portfolio_volatility + 1e-8)
    
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (running_max - portfolio_values) / running_max
    max_drawdown = np.max(drawdowns) * 100
    
    winning_trades = np.sum(rewards > 0)
    total_trades = len(rewards)
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nPerformance Metrics:")
    print(f"  Initial Portfolio Value: ${initial_portfolio:.2f}")
    print(f"  Final Portfolio Value:   ${final_portfolio:.2f}")
    print(f"  Portfolio Return:        {portfolio_return:.2f}%")
    print(f"  Benchmark Return:        {benchmark_return:.2f}%")
    print(f"  Excess Return:           {excess_return:.2f}%")
    print(f"\nRisk Metrics:")
    print(f"  Portfolio Volatility:    {portfolio_volatility*100:.2f}%")
    print(f"  Benchmark Volatility:    {benchmark_volatility*100:.2f}%")
    print(f"  Sharpe Ratio:            {sharpe_ratio:.3f}")
    print(f"  Max Drawdown:            {max_drawdown:.2f}%")
    print(f"\nTrading Metrics:")
    print(f"  Total Steps:             {total_trades}")
    print(f"  Winning Steps:           {winning_trades}")
    print(f"  Win Rate:                {win_rate:.2f}%")
    print(f"  Total Reward:            {np.sum(rewards):.2f}")
    print(f"  Average Reward:          {np.mean(rewards):.2f}")
    print("="*60)
    
    os.makedirs("evaluation_results", exist_ok=True)
    
    reward_corr = np.corrcoef(predicted_rewards, rewards)[0, 1] if len(rewards) > 1 else 0
    print(f"\nReward Prediction Correlation: {reward_corr:.3f}")
    print(f"  Predicted Reward Mean: {np.mean(predicted_rewards):.2f}")
    print(f"  Actual Reward Mean: {np.mean(rewards):.2f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(timestamps, portfolio_values, label='Model Portfolio', linewidth=2, color='blue', zorder=3)
    ax1.plot(timestamps, benchmark_values, label='Benchmark', linewidth=2, alpha=0.7, color='gray')
    
    pred_rewards_plot = np.concatenate([[0], predicted_rewards])
    pred_stds_plot = np.concatenate([[0], predicted_reward_stds])
    
    norm_rewards = (pred_rewards_plot - np.min(pred_rewards_plot)) / (np.max(pred_rewards_plot) - np.min(pred_rewards_plot) + 1e-8)
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0.8, 0.2, 0.2), (0.9, 0.9, 0.2), (0.2, 0.8, 0.2)]
    cmap = LinearSegmentedColormap.from_list('reward_cmap', colors, N=100)
    
    band_scale = np.abs(pred_rewards_plot) * 0.5 + pred_stds_plot * 2
    upper_band = portfolio_values + band_scale
    lower_band = portfolio_values - band_scale
    
    for i in range(len(timestamps) - 1):
        color = cmap(norm_rewards[i])
        ax1.fill_between([timestamps[i], timestamps[i+1]], 
                        [lower_band[i], lower_band[i+1]], 
                        [upper_band[i], upper_band[i+1]], 
                        color=color, alpha=0.4)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(pred_rewards_plot), vmax=np.max(pred_rewards_plot)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, label='Predicted Reward', shrink=0.8)
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Value with Reward Confidence Band')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(predicted_rewards, label='Predicted', linewidth=1.5, alpha=0.8)
    ax2.plot(rewards, label='Actual', linewidth=1.5, alpha=0.8)
    ax2.fill_between(range(len(predicted_rewards)), 
                     predicted_rewards - predicted_reward_stds,
                     predicted_rewards + predicted_reward_stds,
                     alpha=0.3, label='Uncertainty')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward ($)')
    ax2.set_title(f'Predicted vs Actual Rewards (Corr: {reward_corr:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    avg_allocations = np.mean(allocations_history, axis=0)
    bars = ax3.bar(range(len(env.asset_names)), avg_allocations)
    ax3.set_xticks(range(len(env.asset_names)))
    ax3.set_xticklabels(env.asset_names, rotation=45, ha='right')
    ax3.set_ylabel('Average Allocation')
    ax3.set_title('Average Asset Allocation')
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = axes[1, 1]
    
    cumsum_pred = np.cumsum(predicted_rewards)
    cumsum_actual = np.cumsum(rewards)
    
    ax4.plot(cumsum_pred, label='Cumulative Predicted', linewidth=2)
    ax4.plot(cumsum_actual, label='Cumulative Actual', linewidth=2, alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Cumulative Reward ($)')
    ax4.set_title('Cumulative Predicted vs Actual Rewards')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax4_twin = ax4.twinx()
    cumulative_returns = (portfolio_values[1:] / initial_portfolio - 1) * 100
    ax4_twin.plot(cumulative_returns, label='Portfolio Return %', linewidth=1, color='green', alpha=0.5)
    ax4_twin.set_ylabel('Portfolio Return (%)', color='green')
    
    plt.tight_layout()
    plt.savefig('evaluation_results/performance_summary.png', dpi=150)
    plt.close()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    ax1 = axes[0]
    for i, asset_name in enumerate(env.asset_names):
        ax1.plot(allocations_history[:, i], label=asset_name, alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Allocation')
    ax1.set_title('Asset Allocation Over Time')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(predicted_reward_stds, linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Prediction Uncertainty')
    ax2.set_title('Model Prediction Uncertainty Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/allocation_analysis.png', dpi=150)
    plt.close()
    
    results = {
        'initial_portfolio': float(initial_portfolio),
        'final_portfolio': float(final_portfolio),
        'portfolio_return': float(portfolio_return),
        'benchmark_return': float(benchmark_return),
        'excess_return': float(excess_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'volatility': float(portfolio_volatility),
        'win_rate': float(win_rate),
        'total_steps': int(total_trades),
        'total_reward': float(np.sum(rewards)),
        'avg_reward': float(np.mean(rewards)),
        'reward_prediction_correlation': float(reward_corr),
        'avg_predicted_reward': float(np.mean(predicted_rewards)),
        'avg_prediction_uncertainty': float(np.mean(predicted_reward_stds))
    }
    
    with open('evaluation_results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCharts saved to evaluation_results/")
    print("  - performance_summary.png")
    print("  - allocation_analysis.png")
    print("  - metrics.json")


if __name__ == "__main__":
    if TEST:
        evaluate_model()
    else:
        train()
