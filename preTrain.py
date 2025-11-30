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

TEST = False
NUM_PREDICTIONS = 32
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 5
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
SAMPLES_PER_FILE = 128
BEST_MODEL_PATH = "models/best_pretrain_transformer.weights.h5"
CONFIG_PATH = "models/model_config.json"
NORM_STATS_PATH = "models/normalization_stats.json"
DATA_FOLDER = "syntheticData"
WANDB_PROJECT = "portfolio-pretrain"
EPOCHS_PER_CHUNK = 1
TEST_STEPS = 256


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
    
    reward_hidden = layers.Dense(dim_feedforward // 2, activation='relu')(x)
    reward_hidden = layers.Dropout(dropout_rate)(reward_hidden)
    reward_output = layers.Dense(1, activation='linear', name='reward')(reward_hidden)
    
    model = keras.Model(inputs=inputs, outputs=[allocation_output, confidence_output, reward_output])
    
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
    reward_list = []
    
    for sample in samples:
        obs_list.append(sample['obs'])
        action_list.append(sample['action'])
        confidence_list.append(sample['confidence'])
        reward_list.append(sample.get('reward', 0.0))
    
    obs_batch = np.array(obs_list, dtype=np.float32)
    action_batch = np.array(action_list, dtype=np.float32)
    confidence_batch = np.array(confidence_list, dtype=np.float32)
    reward_batch = np.array(reward_list, dtype=np.float32).reshape(-1, 1)
    
    normalizer.update(obs_batch)
    obs_batch_norm = normalizer.normalize(obs_batch).astype(np.float32)
    
    return obs_batch_norm, action_batch, confidence_batch, reward_batch


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


class BestMetricsCallback(keras.callbacks.Callback):
    def __init__(self, model_path, metrics_to_track):
        super().__init__()
        self.model_path = model_path
        self.metrics_to_track = metrics_to_track
        self.best_values = {}
        for metric, mode in metrics_to_track.items():
            self.best_values[metric] = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        improved = []
        for metric, mode in self.metrics_to_track.items():
            if metric not in logs:
                continue
            
            current = logs[metric]
            if mode == 'min' and current < self.best_values[metric]:
                self.best_values[metric] = current
                improved.append(metric)
            elif mode == 'max' and current > self.best_values[metric]:
                self.best_values[metric] = current
                improved.append(metric)
        
        if improved:
            self.model.save_weights(self.model_path)
            print(f"\nModel saved - improved: {', '.join(improved)}")
            
            wandb.log({f'best_{k}': v for k, v in self.best_values.items()})


class DetailedLoggingCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_count = 0
    
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            return
        
        self.batch_count += 1
        
        log_data = {'batch': self.batch_count}
        for key, value in logs.items():
            log_data[f'batch_{key}'] = float(value)
        
        wandb.log(log_data)
    
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
            'confidence': 'mse',
            'reward': 'mse'
        },
        loss_weights={
            'allocation': 1.0,
            'confidence': 0.5,
            'reward': 0.3
        },
        metrics={
            'allocation': ['mae', TopKAccuracy(k=1, name='top1_acc'), TopKAccuracy(k=3, name='top3_acc')],
            'confidence': ['mae'],
            'reward': ['mae']
        }
    )
    
    normalizer = ObservationNormalizer(features_per_timestep)
    if normalizer.load(NORM_STATS_PATH):
        print(f"Loaded normalization stats (count: {normalizer.count})")
    else:
        print("Starting with fresh normalization statistics")
    
    os.makedirs("models", exist_ok=True)
    
    metrics_to_track = {
        'loss': 'min',
        'allocation_loss': 'min',
        'confidence_loss': 'min',
        'reward_loss': 'min',
        'allocation_mae': 'min',
        'allocation_top1_acc': 'max',
        'allocation_top3_acc': 'max',
        'confidence_mae': 'min',
        'reward_mae': 'min'
    }
    
    best_metrics_callback = BestMetricsCallback(BEST_MODEL_PATH, metrics_to_track)
    detailed_logging_callback = DetailedLoggingCallback()
    
    callbacks = [
        best_metrics_callback,
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
                
                X, y_alloc, y_conf, y_reward = prepare_data(samples, normalizer)
                
                if chunk_num == 1 and file_idx == 0 and epoch == 0:
                    print(f"    X shape: {X.shape}, stats: min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
                
                model.fit(
                    X,
                    {'allocation': y_alloc, 'confidence': y_conf, 'reward': y_reward},
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS_PER_CHUNK,
                    callbacks=callbacks,
                    verbose=1
                )
                
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
        print("No trained model found. Training first...")
        print("Set TEST=False and run training, or the model will be created now.\n")
        
        print(f"Scanning data folder: {DATA_FOLDER}")
        sequence_length, features_per_timestep, num_assets, _ = scan_data_folder(DATA_FOLDER)
        
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
    
    model = build_model(
        sequence_length, features_per_timestep, num_assets,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
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
    confidence_history = []
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
        confidence_preds = []
        reward_preds = []
        for _ in range(NUM_PREDICTIONS):
            alloc, conf, rew = model(obs_norm, training=True)
            allocation_preds.append(alloc.numpy())
            confidence_preds.append(conf.numpy())
            reward_preds.append(rew.numpy())
        
        allocation_preds = np.array(allocation_preds)
        confidence_preds = np.array(confidence_preds)
        reward_preds = np.array(reward_preds)
        
        allocation = np.mean(allocation_preds, axis=0)[0]
        confidence = np.mean(confidence_preds, axis=0)[0]
        pred_reward = np.mean(reward_preds)
        pred_reward_std = np.std(reward_preds)
        
        predicted_rewards.append(pred_reward)
        predicted_reward_stds.append(pred_reward_std)
        allocations_history.append(allocation.copy())
        confidence_history.append(confidence.copy())
        
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
    confidence_history = np.array(confidence_history)
    
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
    avg_confidence = np.mean(confidence_history, axis=1)
    ax2.plot(avg_confidence, linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Model Confidence Over Time')
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
