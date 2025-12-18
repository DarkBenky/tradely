import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from createTraningSet import create_training_set
from params import WINDOW_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, NEXT_PRICE_PREDICTION, NEXT_PRICE_PREDICTION_1, NEXT_PRICE_PREDICTION_2
import json
import matplotlib.pyplot as plt

def create_lstm_model(input_shape, num_buckets):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    
    output_next = layers.Dense(num_buckets, activation='softmax', name='next')(x)
    output_next_1 = layers.Dense(num_buckets, activation='softmax', name='next_1')(x)
    output_next_2 = layers.Dense(num_buckets, activation='softmax', name='next_2')(x)
    
    model = keras.Model(inputs=inputs, outputs=[output_next, output_next_1, output_next_2])
    
    return model

def distribution_to_value(distribution, bucket_config):
    buckets = np.array(bucket_config['close']['buckets'])
    bucket_centers = (buckets[:-1] + buckets[1:]) / 2
    predicted_value = np.sum(distribution * bucket_centers)
    return predicted_value

def create_prediction_charts(model, X_test, y_test_next, y_test_next_1, y_test_next_2):
    bucket_config = json.load(open('bucket_config.json'))
    
    pred_next, pred_next_1, pred_next_2 = model.predict(X_test, verbose=0)
    
    num_samples = min(5, len(X_test))
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        pred_val_next = distribution_to_value(pred_next[sample_idx], bucket_config)
        pred_val_next_1 = distribution_to_value(pred_next_1[sample_idx], bucket_config)
        pred_val_next_2 = distribution_to_value(pred_next_2[sample_idx], bucket_config)
        
        real_val_next = distribution_to_value(y_test_next[sample_idx], bucket_config)
        real_val_next_1 = distribution_to_value(y_test_next_1[sample_idx], bucket_config)
        real_val_next_2 = distribution_to_value(y_test_next_2[sample_idx], bucket_config)
        
        buckets = np.array(bucket_config['close']['buckets'])
        bucket_centers = (buckets[:-1] + buckets[1:]) / 2
        
        axes[sample_idx, 0].bar(range(len(pred_next[sample_idx])), pred_next[sample_idx], alpha=0.7, label='Predicted')
        axes[sample_idx, 0].bar(range(len(y_test_next[sample_idx])), y_test_next[sample_idx], alpha=0.7, label='Real')
        axes[sample_idx, 0].axvline(np.argmax(pred_next[sample_idx]), color='r', linestyle='--', label=f'Pred: {pred_val_next:.2f}%')
        axes[sample_idx, 0].axvline(np.argmax(y_test_next[sample_idx]), color='g', linestyle='--', label=f'Real: {real_val_next:.2f}%')
        axes[sample_idx, 0].set_title(f'Sample {sample_idx + 1} - Next Hour')
        axes[sample_idx, 0].set_xlabel('Bucket')
        axes[sample_idx, 0].set_ylabel('Probability')
        axes[sample_idx, 0].legend()
        axes[sample_idx, 0].grid(True, alpha=0.3)
        
        axes[sample_idx, 1].bar(range(len(pred_next_1[sample_idx])), pred_next_1[sample_idx], alpha=0.7, label='Predicted')
        axes[sample_idx, 1].bar(range(len(y_test_next_1[sample_idx])), y_test_next_1[sample_idx], alpha=0.7, label='Real')
        axes[sample_idx, 1].axvline(np.argmax(pred_next_1[sample_idx]), color='r', linestyle='--', label=f'Pred: {pred_val_next_1:.2f}%')
        axes[sample_idx, 1].axvline(np.argmax(y_test_next_1[sample_idx]), color='g', linestyle='--', label=f'Real: {real_val_next_1:.2f}%')
        axes[sample_idx, 1].set_title(f'Sample {sample_idx + 1} - Next 12 Hours')
        axes[sample_idx, 1].set_xlabel('Bucket')
        axes[sample_idx, 1].set_ylabel('Probability')
        axes[sample_idx, 1].legend()
        axes[sample_idx, 1].grid(True, alpha=0.3)
        
        axes[sample_idx, 2].bar(range(len(pred_next_2[sample_idx])), pred_next_2[sample_idx], alpha=0.7, label='Predicted')
        axes[sample_idx, 2].bar(range(len(y_test_next_2[sample_idx])), y_test_next_2[sample_idx], alpha=0.7, label='Real')
        axes[sample_idx, 2].axvline(np.argmax(pred_next_2[sample_idx]), color='r', linestyle='--', label=f'Pred: {pred_val_next_2:.2f}%')
        axes[sample_idx, 2].axvline(np.argmax(y_test_next_2[sample_idx]), color='g', linestyle='--', label=f'Real: {real_val_next_2:.2f}%')
        axes[sample_idx, 2].set_title(f'Sample {sample_idx + 1} - Next 30 Days')
        axes[sample_idx, 2].set_xlabel('Bucket')
        axes[sample_idx, 2].set_ylabel('Probability')
        axes[sample_idx, 2].legend()
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
            "architecture": "LSTM",
            "optimizer": "adam"
        }
    )
    
    print("Generating training data...")
    X_train, (y_next, y_next_1, y_next_2) = create_training_set(100_000, 5000)
    print(f"Training data shape: {X_train.shape}")
    print(f"Target shapes: {y_next.shape}, {y_next_1.shape}, {y_next_2.shape}")
    
    print("\nGenerating validation data...")
    X_val, (y_val_next, y_val_next_1, y_val_next_2) = create_training_set(2000, 20)
    
    num_buckets = y_next.shape[1]
    input_shape = (WINDOW_SIZE, 5)
    
    print(f"\nCreating model with input shape {input_shape} and {num_buckets} buckets...")
    model = create_lstm_model(input_shape, num_buckets)
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss={'next': 'categorical_crossentropy', 'next_1': 'categorical_crossentropy', 'next_2': 'categorical_crossentropy'},
        metrics={'next': ['accuracy'], 'next_1': ['accuracy'], 'next_2': ['accuracy']}
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
        {'next': y_next, 'next_1': y_next_1, 'next_2': y_next_2},
        validation_data=(X_val, {'next': y_val_next, 'next_1': y_val_next_1, 'next_2': y_val_next_2}),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save('final_model.keras')
    print("\nTraining complete. Model saved.")
    
    print("\nGenerating prediction charts...")
    create_prediction_charts(model, X_val, y_val_next, y_val_next_1, y_val_next_2)
    
    wandb.finish()

if __name__ == "__main__":
    train()
