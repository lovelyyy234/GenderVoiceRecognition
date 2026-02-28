"""
train_deep.py — Train CNN and/or LSTM models for gender recognition

Usage:
  python train_deep.py --model cnn
  python train_deep.py --model lstm
  python train_deep.py --model both

This script:
1. Loads voice.csv
2. Generates synthetic mel spectrograms / MFCC sequences from acoustic features
3. Trains CNN (mel) and/or LSTM (MFCC) with early stopping
4. Saves models to /models/

NOTE: For best results, collect real WAV files per speaker labeled male/female
and use extract_mel_spectrogram() / extract_mfcc_sequence() from feature_extractor.py
"""

import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} found.")
    TF_OK = True
except ImportError:
    print("ERROR: TensorFlow not installed.")
    print("Run:  pip install tensorflow")
    sys.exit(1)

from modules.cnn_model  import build_cnn,  MODEL_PATH as CNN_PATH
from modules.lstm_model import build_lstm, MODEL_PATH as LSTM_PATH

FEATURE_LABELS = ['mode','minfun','maxdom','Q25','Q75','IQR','meanfun','median','skew']
N_MELS         = 64
N_MFCC         = 40
TIME_STEPS     = 128
DATA_FILE      = 'data/voice.csv'


def load_dataset():
    data = pd.read_csv(DATA_FILE)
    X    = data[FEATURE_LABELS].values
    y    = data['label'].replace({'male': 0, 'female': 1}).astype(int).values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    return X_sc, y


def features_to_mel(features_scaled):
    """
    Convert scaled acoustic features to synthetic mel spectrogram.
    Since we don't have raw WAV files here, we synthesize plausible
    spectrograms from acoustic feature values.
    In production: use extract_mel_spectrogram() on real WAV files.
    """
    n = len(features_scaled)
    mels = np.zeros((n, N_MELS, TIME_STEPS, 1), dtype=np.float32)

    for i, feat in enumerate(features_scaled):
        # Use feature values to shape a plausible spectrogram
        meanfun_norm = (feat[6] + 2) / 4.0      # normalize ~[-2,2] → [0,1]
        energy       = abs(feat[0]) / 3.0

        base = np.random.randn(N_MELS, TIME_STEPS).astype(np.float32) * 0.1

        # Low frequency band (voice fundamental)
        low_band = max(0, int(meanfun_norm * 20))
        base[low_band:low_band+8, :] += 0.6 + energy * 0.4

        # Harmonics
        for h in range(1, 4):
            hband = min(N_MELS-1, low_band + h*8)
            base[hband:hband+4, :] += (0.4 - h*0.1) * energy

        # Normalize
        mn = base.min(); mx = base.max()
        if mx > mn:
            base = (base - mn) / (mx - mn)

        mels[i, :, :, 0] = base

    return mels


def features_to_mfcc(features_scaled):
    """
    Convert scaled acoustic features to synthetic MFCC sequences.
    In production: use extract_mfcc_sequence() on real WAV files.
    """
    n = len(features_scaled)
    mfccs = np.zeros((n, TIME_STEPS, N_MFCC), dtype=np.float32)

    for i, feat in enumerate(features_scaled):
        # Create MFCC-like sequence based on feature values
        base_mfcc = feat[:min(9, N_MFCC)]  # use our 9 features as base
        full_mfcc = np.zeros(N_MFCC)
        full_mfcc[:len(base_mfcc)] = base_mfcc

        # Add time variation
        seq = np.tile(full_mfcc, (TIME_STEPS, 1))
        seq += np.random.randn(TIME_STEPS, N_MFCC).astype(np.float32) * 0.15

        # Temporal smoothing
        for t in range(1, TIME_STEPS):
            seq[t] = 0.85 * seq[t-1] + 0.15 * seq[t]

        mfccs[i] = seq

    return mfccs


def train_cnn_model(X, y):
    print("\n" + "="*50)
    print("  Training CNN Model")
    print("="*50)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"  Converting {len(X_train)} training samples to Mel spectrograms...")
    mel_train = features_to_mel(X_train)
    mel_val   = features_to_mel(X_val)
    print(f"  Train shape: {mel_train.shape}")

    model = build_cnn(input_shape=(N_MELS, TIME_STEPS, 1))
    model.summary()

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        ModelCheckpoint(CNN_PATH, save_best_only=True, verbose=0)
    ]

    history = model.fit(
        mel_train, y_train,
        validation_data=(mel_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    val_acc = max(history.history.get('val_accuracy', [0])) * 100
    print(f"\n  CNN Best Val Accuracy: {val_acc:.2f}%")
    print(f"  Model saved: {CNN_PATH}")
    return model


def train_lstm_model(X, y):
    print("\n" + "="*50)
    print("  Training BiLSTM Model")
    print("="*50)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"  Converting {len(X_train)} training samples to MFCC sequences...")
    mfcc_train = features_to_mfcc(X_train)
    mfcc_val   = features_to_mfcc(X_val)
    print(f"  Train shape: {mfcc_train.shape}")

    model = build_lstm(input_shape=(TIME_STEPS, N_MFCC))
    model.summary()

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        ModelCheckpoint(LSTM_PATH, save_best_only=True, verbose=0)
    ]

    history = model.fit(
        mfcc_train, y_train,
        validation_data=(mfcc_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    val_acc = max(history.history.get('val_accuracy', [0])) * 100
    print(f"\n  BiLSTM Best Val Accuracy: {val_acc:.2f}%")
    print(f"  Model saved: {LSTM_PATH}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deep learning models')
    parser.add_argument('--model', choices=['cnn','lstm','both'], default='both',
                        help='Which model to train')
    args = parser.parse_args()

    print(f"\nGPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Dataset: {len(X)} samples ({sum(y==0)} male, {sum(y==1)} female)")

    if args.model in ('cnn', 'both'):
        train_cnn_model(X, y)

    if args.model in ('lstm', 'both'):
        train_lstm_model(X, y)

    print("\nTraining complete! Restart the Flask app to use new models.")
    print("  python app.py")
