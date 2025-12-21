import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.Variable(pe[np.newaxis, :, :].astype(np.float32), trainable=False)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

class TimeEncoder(layers.Layer):
    def __init__(self, d_model=128, num_heads=4, num_layers=2, ff_dim=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.input_projection = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append([
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads),
                layers.LayerNormalization(),
                layers.Dense(ff_dim, activation='relu'),
                layers.Dense(d_model),
                layers.LayerNormalization(),
                layers.Dropout(dropout)
            ])
    
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        features = tf.shape(x)[2]
        
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        for attn, norm1, ff1, ff2, norm2, dropout in self.transformer_blocks:
            attn_out = attn(x, x, training=training)
            x = norm1(x + attn_out)
            
            ff_out = ff2(ff1(x))
            ff_out = dropout(ff_out, training=training)
            x = norm2(x + ff_out)
        
        return x[:, -1, :]

class AssetEncoder(layers.Layer):
    def __init__(self, d_model=128, num_heads=4, num_layers=2, ff_dim=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append([
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads),
                layers.LayerNormalization(),
                layers.Dense(ff_dim, activation='relu'),
                layers.Dense(d_model),
                layers.LayerNormalization(),
                layers.Dropout(dropout)
            ])
    
    def call(self, x, training=False):
        for attn, norm1, ff1, ff2, norm2, dropout in self.transformer_blocks:
            attn_out = attn(x, x, training=training)
            x = norm1(x + attn_out)
            
            ff_out = ff2(ff1(x))
            ff_out = dropout(ff_out, training=training)
            x = norm2(x + ff_out)
        
        return x

class CrossAssetTransformer(keras.Model):
    def __init__(self, n_assets=11, d_model=128, time_layers=2, asset_layers=2, 
                 num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        
        self.time_encoder = TimeEncoder(d_model, num_heads, time_layers, ff_dim, dropout)
        self.asset_encoder = AssetEncoder(d_model, num_heads, asset_layers, ff_dim, dropout)
        
        self.output_head = layers.Dense(1)
    
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        n_assets = tf.shape(x)[1]
        time_steps = tf.shape(x)[2]
        features = tf.shape(x)[3]
        
        x_reshaped = tf.reshape(x, [-1, time_steps, features])
        
        time_encoded = self.time_encoder(x_reshaped, training=training)
        
        asset_embeddings = tf.reshape(time_encoded, [batch_size, n_assets, self.d_model])
        
        asset_encoded = self.asset_encoder(asset_embeddings, training=training)
        
        asset_scores = asset_encoded[:, 1:, :]
        
        outputs = self.output_head(asset_scores)
        outputs = tf.squeeze(outputs, axis=-1)
        
        return outputs

def create_cross_asset_model(n_assets=11, d_model=128, time_layers=2, asset_layers=2,
                              num_heads=4, ff_dim=512, dropout=0.1):
    model = CrossAssetTransformer(
        n_assets=n_assets,
        d_model=d_model,
        time_layers=time_layers,
        asset_layers=asset_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout
    )
    
    return model
