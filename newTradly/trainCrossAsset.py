import tensorflow as tf
from tensorflow import keras
import numpy as np
import wandb
from wandb.integration.keras import WandbMetricsLogger
from crossAssetData import create_cross_asset_dataset, ASSETS, WINDOW_SIZE
from crossAssetModel import create_cross_asset_model
import os

D_MODEL = 512
TIME_LAYERS = 6
ASSET_LAYERS = 4
NUM_HEADS = 8
FF_DIM = 2048
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
TRAIN_BATCHES = 100
VAL_BATCHES = 20

def cross_sectional_mse(y_true, y_pred):
    y_true_mean = tf.reduce_mean(y_true, axis=-1, keepdims=True)
    y_pred_mean = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean
    
    return tf.reduce_mean(tf.square(y_true_centered - y_pred_centered))

def ranking_loss(y_true, y_pred):
    batch_size = tf.shape(y_true)[0]
    n_assets = tf.shape(y_true)[1]
    
    y_true_diff = tf.expand_dims(y_true, 2) - tf.expand_dims(y_true, 1)
    y_pred_diff = tf.expand_dims(y_pred, 2) - tf.expand_dims(y_pred, 1)
    
    sign_agreement = tf.sign(y_true_diff) * y_pred_diff
    
    loss = tf.reduce_mean(tf.maximum(0.0, -sign_agreement + 0.1))
    
    return loss

def combined_loss(y_true, y_pred):
    mse = cross_sectional_mse(y_true, y_pred)
    rank = ranking_loss(y_true, y_pred)
    return mse + 0.5 * rank

def train_cross_asset_model():
    os.makedirs('models', exist_ok=True)
    
    wandb.init(project="tradely-cross-asset", config={
        "d_model": D_MODEL,
        "time_layers": TIME_LAYERS,
        "asset_layers": ASSET_LAYERS,
        "num_heads": NUM_HEADS,
        "ff_dim": FF_DIM,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "assets": ASSETS,
        "window_size": WINDOW_SIZE
    })
    
    model = create_cross_asset_model(
        n_assets=ASSETS + 1,
        d_model=D_MODEL,
        time_layers=TIME_LAYERS,
        asset_layers=ASSET_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    )

    optimizer = keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[cross_sectional_mse]
    )
    
    print("\nInitializing model with dummy forward pass...")
    X_init, _ = create_cross_asset_dataset(1)
    _ = model(X_init, training=False)
    
    print("\n" + "="*60)
    model.summary()
    print("="*60 + "\n")

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Pre-generating training and validation data...")
    train_data = []
    for _ in range(TRAIN_BATCHES):
        train_data.append(create_cross_asset_dataset(BATCH_SIZE))
    
    val_data = []
    for _ in range(VAL_BATCHES):
        val_data.append(create_cross_asset_dataset(BATCH_SIZE))
    
    print(f"Generated {len(train_data)} training batches and {len(val_data)} validation batches\n")
    
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
        
        train_losses = []
        for batch, (X_train, y_train) in enumerate(train_data):
            try:
                with tf.GradientTape() as tape:
                    y_pred = model(X_train, training=True)
                    loss = combined_loss(y_train, y_pred)
                
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
                train_losses.append(loss.numpy())
                
                if batch % 10 == 0:
                    print(f"Batch {batch}/{TRAIN_BATCHES} - Loss: {loss.numpy():.6f}")
            
            except Exception as e:
                print(f"Error in training batch {batch}: {e}")
                continue
        
        val_losses = []
        for batch, (X_val, y_val) in enumerate(val_data):
            try:
                y_pred = model(X_val, training=False)
                loss = combined_loss(y_val, y_pred)
                val_losses.append(loss.numpy())
            except Exception as e:
                print(f"Error in validation batch {batch}: {e}")
                continue
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save('models/cross_asset_best.keras')
            print(f"New best model saved! Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 5 == 0:
            print("\nRegenerating data to prevent overfitting...")
            train_data = []
            for _ in range(TRAIN_BATCHES):
                train_data.append(create_cross_asset_dataset(BATCH_SIZE))
            
            val_data = []
            for _ in range(VAL_BATCHES):
                val_data.append(create_cross_asset_dataset(BATCH_SIZE))
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    model.save('models/cross_asset_final.keras')
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    wandb.finish()

if __name__ == "__main__":
    train_cross_asset_model()
