import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from createTraningSet import create_training_set
from params import WINDOW_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, LSTM_LAYERS, DROPOUT_RATE, MODEL
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=20480)]
        )
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

def create_lstm_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    x = inputs
    for i, units in enumerate(LSTM_LAYERS):
        return_sequences = i < len(LSTM_LAYERS) - 1
        x = layers.LSTM(units, return_sequences=return_sequences)(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
    
    output_next = layers.Dense(1, name='next')(x)
    output_half = layers.Dense(1, name='half')(x)
    output_full = layers.Dense(1, name='full')(x)
    
    model = keras.Model(inputs=inputs, outputs=[output_next, output_half, output_full])
    
    return model

def create_gru_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    x = inputs
    for i, units in enumerate(LSTM_LAYERS):
        return_sequences = i < len(LSTM_LAYERS) - 1
        x = layers.GRU(units, return_sequences=return_sequences)(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
    
    output_next = layers.Dense(1, name='next')(x)
    output_half = layers.Dense(1, name='half')(x)
    output_full = layers.Dense(1, name='full')(x)
    
    model = keras.Model(inputs=inputs, outputs=[output_next, output_half, output_full])
    
    return model

def create_transformer_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Dense(LSTM_LAYERS[0])(inputs)
    
    for i, units in enumerate(LSTM_LAYERS):
        num_heads = min(8, units // 64)
        if num_heads < 1:
            num_heads = 1
        
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units // num_heads,
            dropout=DROPOUT_RATE
        )(x, x)
        attn_output = layers.Dropout(DROPOUT_RATE)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        ffn = keras.Sequential([
            layers.Dense(units * 4, activation='relu'),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(units)
        ])
        ffn_output = ffn(x)
        ffn_output = layers.Dropout(DROPOUT_RATE)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        if i < len(LSTM_LAYERS) - 1:
            x = layers.Dense(LSTM_LAYERS[i + 1])(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    output_next = layers.Dense(1, name='next')(x)
    output_half = layers.Dense(1, name='half')(x)
    output_full = layers.Dense(1, name='full')(x)
    
    model = keras.Model(inputs=inputs, outputs=[output_next, output_half, output_full])
    
    return model

def create_prediction_charts(model, X_test, y_test_next, y_test_half, y_test_full):
    pred_next, pred_half, pred_full = model.predict(X_test, verbose=0)
    
    num_samples = min(5, len(X_test))
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        axes[sample_idx, 0].bar(['Predicted', 'Real'], [pred_next[sample_idx][0], y_test_next[sample_idx]], alpha=0.7)
        axes[sample_idx, 0].set_title(f'Sample {sample_idx + 1} - Next Value')
        axes[sample_idx, 0].set_ylabel('Cumsum Price Change')
        axes[sample_idx, 0].grid(True, alpha=0.3)
        
        axes[sample_idx, 1].bar(['Predicted', 'Real'], [pred_half[sample_idx][0], y_test_half[sample_idx]], alpha=0.7)
        axes[sample_idx, 1].set_title(f'Sample {sample_idx + 1} - Half Window')
        axes[sample_idx, 1].set_ylabel('Cumsum Price Change')
        axes[sample_idx, 1].grid(True, alpha=0.3)
        
        axes[sample_idx, 2].bar(['Predicted', 'Real'], [pred_full[sample_idx][0], y_test_full[sample_idx]], alpha=0.7)
        axes[sample_idx, 2].set_title(f'Sample {sample_idx + 1} - Full Window')
        axes[sample_idx, 2].set_ylabel('Cumsum Price Change')
        axes[sample_idx, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_charts.png', dpi=100, bbox_inches='tight')
    print("Saved prediction_charts.png")
    
    wandb.log({"prediction_charts": wandb.Image("prediction_charts.png")})
    plt.close()

def train():
    wandb.init(
        project="tradely",
        config={
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "architecture": MODEL,
            "optimizer": "adamw",
            "layers": LSTM_LAYERS,
            "dropout_rate": DROPOUT_RATE
        }
    )
    
    print("Generating training data...")
    X_train, (y_next, y_half, y_full) = create_training_set(25_000, 5_000)
    print(f"Training data shape: {X_train.shape}")
    print(f"Target shapes: {y_next.shape}, {y_half.shape}, {y_full.shape}")
    
    print("\nGenerating validation data...")
    X_val, (y_val_next, y_val_half, y_val_full) = create_training_set(2000, 20)
    
    input_shape = (WINDOW_SIZE, 5)
    
    print(f"\nCreating {MODEL} model with input shape {input_shape}...")
    
    if MODEL == "LSTM":
        model = create_lstm_model(input_shape)
    elif MODEL == "GRU":
        model = create_gru_model(input_shape)
    elif MODEL == "Transformer":
        model = create_transformer_model(input_shape)
    else:
        raise ValueError(f"Unknown model type: {MODEL}")
    
    optimizer = keras.optimizers.AdamW(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss={'next': 'mse', 'half': 'mse', 'full': 'mse'},
        metrics={'next': ['mae'], 'half': ['mae'], 'full': ['mae']}
    )
    
    model.summary()
    
    callbacks = [
        WandbMetricsLogger(),
        WandbModelCheckpoint('best_model.keras'),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    print("\nStarting training...")
    history = model.fit(
        X_train,
        {'next': y_next, 'half': y_half, 'full': y_full},
        validation_data=(X_val, {'next': y_val_next, 'half': y_val_half, 'full': y_val_full}),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save('final_model.keras')
    print("\nTraining complete. Model saved.")
    
    print("\nGenerating prediction charts...")
    create_prediction_charts(model, X_val, y_val_next, y_val_half, y_val_full)
    
    wandb.finish()

if __name__ == "__main__":
    train()
