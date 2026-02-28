"""
LSTM Model — Gender Detection from MFCC Sequences
Input:  (batch, 128, 40)  — MFCC time series
Output: (batch, 2)        — [male_prob, female_prob]
"""

import numpy as np
import os

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    TF_OK = True
except ImportError:
    TF_OK = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_gender.h5')


def build_lstm(input_shape=(128, 40), num_classes=2):
    """Build Bidirectional LSTM architecture."""
    inp = layers.Input(shape=input_shape)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, x, name='BiLSTM_Gender')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_lstm(X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    """Train LSTM and save model."""
    if not TF_OK:
        print("TensorFlow not installed. Skipping LSTM training.")
        return None

    model = build_lstm()
    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1
    )
    print(f"\nLSTM saved to: {MODEL_PATH}")
    return model, history


def load_lstm():
    """Load saved LSTM model."""
    if not TF_OK:
        return None
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None


def predict_lstm(model, mfcc_input):
    """
    mfcc_input: numpy array shape (1, 128, 40)
    Returns: (label, confidence)
    """
    if model is None:
        return None, 0.0
    proba  = model.predict(mfcc_input, verbose=0)[0]
    idx    = int(np.argmax(proba))
    label  = "FEMALE" if idx == 1 else "MALE"
    conf   = float(proba[idx]) * 100
    return label, conf
