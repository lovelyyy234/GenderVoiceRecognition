"""
Emotion Detection Module
Detects 7 emotions: neutral, happy, sad, angry, fearful, disgust, surprised
Uses MFCC + statistical features with a trained classifier.
"""

import numpy as np
import os
import pickle

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

EMOTIONS      = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
EMOTION_EMOJI = {
    'Neutral':   '😐',
    'Happy':     '😊',
    'Sad':       '😢',
    'Angry':     '😠',
    'Fearful':   '😨',
    'Disgust':   '🤢',
    'Surprised': '😲'
}
SAMPLE_RATE  = 22050
MODEL_PATH   = os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion_model.pkl')
SCALER_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion_scaler.pkl')


def extract_emotion_features(y, sr=SAMPLE_RATE):
    """Extract rich feature set for emotion detection from audio array."""
    features = []

    # MFCCs (mean + std of each coefficient)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(mfccs.mean(axis=1).tolist())
    features.extend(mfccs.std(axis=1).tolist())

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(chroma.mean(axis=1).tolist())

    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(spec_centroid.mean()))

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(spec_rolloff.mean()))

    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(float(spec_bandwidth.mean()))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(float(zcr.mean()))

    # RMS energy
    rms = librosa.feature.rms(y=y)
    features.append(float(rms.mean()))

    # Tonnetz (tonal centroid features)
    try:
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        features.extend(tonnetz.mean(axis=1).tolist())
    except Exception:
        features.extend([0.0] * 6)

    return np.array(features)


def extract_emotion_features_from_file(filepath):
    """Extract emotion features from a WAV file."""
    if not LIBROSA_OK:
        raise RuntimeError("librosa not installed: pip install librosa")
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    return extract_emotion_features(y, sr)


def train_emotion_model_synthetic():
    """
    Train emotion model on synthetic/augmented data when RAVDESS is unavailable.
    Uses acoustic feature heuristics to create meaningful synthetic training data.
    In production: replace with real RAVDESS dataset.
    """
    np.random.seed(42)
    n_per_class = 200
    n_features  = 40*2 + 12 + 1 + 1 + 1 + 1 + 1 + 6  # = 103

    X_list, y_list = [], []

    # Each emotion has characteristic acoustic fingerprint
    emotion_profiles = {
        0: {'energy': 0.3, 'zcr': 0.05, 'centroid': 2000, 'mfcc_mean': 0.0},   # Neutral
        1: {'energy': 0.7, 'zcr': 0.12, 'centroid': 3500, 'mfcc_mean': 2.0},   # Happy
        2: {'energy': 0.2, 'zcr': 0.03, 'centroid': 1200, 'mfcc_mean': -2.0},  # Sad
        3: {'energy': 0.9, 'zcr': 0.18, 'centroid': 4000, 'mfcc_mean': 3.0},   # Angry
        4: {'energy': 0.5, 'zcr': 0.08, 'centroid': 2800, 'mfcc_mean': 1.0},   # Fearful
        5: {'energy': 0.4, 'zcr': 0.06, 'centroid': 1800, 'mfcc_mean': -1.0},  # Disgust
        6: {'energy': 0.8, 'zcr': 0.15, 'centroid': 3800, 'mfcc_mean': 2.5},   # Surprised
    }

    for emotion_idx, profile in emotion_profiles.items():
        for _ in range(n_per_class):
            feat = np.zeros(n_features)
            # MFCC means (0-39)
            feat[0:40] = np.random.normal(
                profile['mfcc_mean'], 1.5, 40)
            # MFCC stds (40-79)
            feat[40:80] = np.abs(np.random.normal(2.0, 0.5, 40))
            # Chroma (80-91)
            feat[80:92] = np.random.uniform(0.3, 0.8, 12)
            # Spectral centroid (92)
            feat[92] = profile['centroid'] + np.random.normal(0, 200)
            # Spectral rolloff (93)
            feat[93] = profile['centroid'] * 1.5 + np.random.normal(0, 300)
            # Bandwidth (94)
            feat[94] = 1500 + np.random.normal(0, 200)
            # ZCR (95)
            feat[95] = profile['zcr'] + np.random.normal(0, 0.01)
            # RMS energy (96)
            feat[96] = profile['energy'] + np.random.normal(0, 0.05)
            # Tonnetz (97-102)
            feat[97:103] = np.random.normal(0, 0.1, 6)

            X_list.append(feat)
            y_list.append(emotion_idx)

    X = np.array(X_list)
    y = np.array(y_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_scaled, y)

    # Save
    with open(MODEL_PATH,  'wb') as f: pickle.dump(model,  f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    print(f"Emotion model saved. NOTE: Trained on synthetic data.")
    print(f"For production, train on RAVDESS dataset for real accuracy.")
    return model, scaler


def load_emotion_model():
    """Load saved emotion model and scaler."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH,  'rb') as f: model  = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        return model, scaler
    return None, None


def predict_emotion(model, scaler, features):
    """
    Predict emotion from extracted features.
    features: numpy array of shape (n_features,)
    Returns: (emotion_label, emoji, confidence, all_probas_dict)
    """
    if model is None:
        return 'Neutral', '😐', 50.0, {}

    feat_scaled = scaler.transform(features.reshape(1, -1))
    proba       = model.predict_proba(feat_scaled)[0]
    idx         = int(np.argmax(proba))
    label       = EMOTIONS[idx]
    emoji       = EMOTION_EMOJI[label]
    confidence  = float(proba[idx]) * 100

    all_probas = {EMOTIONS[i]: float(proba[i]) * 100
                  for i in range(len(EMOTIONS))}

    return label, emoji, confidence, all_probas
