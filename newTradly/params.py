LOOKBACK_DAYS = 30
WINDOW_SIZE = LOOKBACK_DAYS * 24
NEXT_PRICE_PREDICTION = 24
NEXT_PRICE_PREDICTION_1 = 24 * (LOOKBACK_DAYS / 2)
NEXT_PRICE_PREDICTION_2 = 24 * LOOKBACK_DAYS
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

MODELS = ["LSTM", "GRU", "Transformer"]

MODEL = MODELS[2]
LSTM_LAYERS = [2048,1048,1048,1048,512, 64, 32]
DROPOUT_RATE = 0.2
DENSE_ACTIVATION = 'softmax'

TIMEFRAMES = [
    {'interval': '5m', 'period': '60d', 'minutes': 5},
    {'interval': '1h', 'period': '730d', 'minutes': 60},
    {'interval': '4h', 'period': '730d', 'minutes': 240},
    {'interval': '1d', 'period': 'max', 'minutes': 1440}
]