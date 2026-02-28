"""
CNN Model — Gender Detection from Mel Spectrograms
Input:  (batch, 64, 128, 1)  — Mel spectrogram
Output: (batch, 2)           — [male_prob, female_prob]
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

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_gender.h5')


def build_cnn(input_shape=(64, 128, 1), num_classes=2):
    """Build CNN architecture."""
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, x, name='CNN_Gender')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_cnn(X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    """Train CNN and save model."""
    if not TF_OK:
        print("TensorFlow not installed. Skipping CNN training.")
        return None

    model = build_cnn()
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
    print(f"\nCNN saved to: {MODEL_PATH}")
    return model, history


def load_cnn():
    """Load saved CNN model."""
    if not TF_OK:
        return None
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None


def predict_cnn(model, mel_input):
    """
    mel_input: numpy array shape (1, 64, 128, 1)
    Returns: (label, confidence)
    """
    if model is None:
        return None, 0.0
    proba  = model.predict(mel_input, verbose=0)[0]
    idx    = int(np.argmax(proba))
    label  = "FEMALE" if idx == 1 else "MALE"
    conf   = float(proba[idx]) * 100
    return label, conf
