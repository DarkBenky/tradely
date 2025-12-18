import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from createTraningSet import create_training_set
from params import WINDOW_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE

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
    
    wandb.finish()

if __name__ == "__main__":
    train()
