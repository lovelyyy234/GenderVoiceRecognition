"""
Feature Extraction Module
- Classical acoustic features (for ML models)
- Mel Spectrogram (for CNN)
- MFCC sequences (for LSTM)
"""

import numpy as np
import pandas as pd
from scipy.stats import skew

# ── Try librosa ──────────────────────────────────────────────
try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

FEATURE_LABELS = ['mode','minfun','maxdom','Q25','Q75','IQR','meanfun','median','skew']
SAMPLE_RATE    = 22050
N_MELS         = 64
N_MFCC         = 40
MAX_TIME_STEPS = 128   # fixed length for LSTM input
CHUNK          = 2048


def extract_classical_features(filepath):
    """Extract 9 acoustic features used by ML models."""
    if not LIBROSA_OK:
        raise RuntimeError("librosa required: pip install librosa")
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    if len(y) == 0:
        return None

    # Fundamental frequency via pyin
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_voiced = f0[voiced_flag] / 1000.0  # convert to kHz
    except Exception:
        f0_voiced = np.array([0.15])

    if len(f0_voiced) == 0:
        f0_voiced = np.array([0.15])

    freqs = f0_voiced.tolist()

    # Spectral features
    spec  = np.abs(librosa.stft(y, n_fft=CHUNK))
    freqv = librosa.fft_frequencies(sr=sr, n_fft=CHUNK) / 1000.0

    # Weighted frequency distribution
    power = spec.mean(axis=1)
    if power.sum() > 0:
        weights = power / power.sum()
    else:
        weights = np.ones(len(power)) / len(power)

    sorted_idx = np.argsort(freqv)
    cumulative = np.cumsum(weights[sorted_idx])
    Q25  = float(freqv[sorted_idx][np.searchsorted(cumulative, 0.25)])
    Q75  = float(freqv[sorted_idx][np.searchsorted(cumulative, 0.75)])
    IQR  = Q75 - Q25

    mode_val   = float(pd.Series(freqs).mode()[0])
    minfun     = float(np.min(f0_voiced))
    maxdom     = float(freqv[np.argmax(power)])
    meanfun    = float(np.mean(f0_voiced))
    median_val = float(np.median(freqs))
    skw        = float(skew(freqs)) if len(freqs) > 1 else 0.0

    return [[mode_val, minfun, maxdom, Q25, Q75, IQR, meanfun, median_val, skw]]


def extract_mel_spectrogram(filepath, fixed_length=128):
    """Extract Mel spectrogram for CNN input. Returns shape (N_MELS, fixed_length, 1)."""
    if not LIBROSA_OK:
        raise RuntimeError("librosa required: pip install librosa")
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    mel   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate to fixed length
    if mel_db.shape[1] < fixed_length:
        pad = fixed_length - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode='constant')
    else:
        mel_db = mel_db[:, :fixed_length]

    # Normalize to [0,1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db[..., np.newaxis]   # (64, 128, 1)


def extract_mfcc_sequence(filepath, fixed_length=MAX_TIME_STEPS):
    """Extract MFCC sequence for LSTM input. Returns shape (fixed_length, N_MFCC)."""
    if not LIBROSA_OK:
        raise RuntimeError("librosa required: pip install librosa")
    y, sr  = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)  # (40, T)
    mfccs  = mfccs.T   # (T, 40)

    if mfccs.shape[0] < fixed_length:
        pad   = fixed_length - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0,pad),(0,0)), mode='constant')
    else:
        mfccs = mfccs[:fixed_length, :]

    # Normalize
    mu  = mfccs.mean(axis=0)
    std = mfccs.std(axis=0) + 1e-8
    return (mfccs - mu) / std    # (128, 40)


def extract_features_from_array(y_array, sr=SAMPLE_RATE):
    """Extract mel spectrogram and MFCC from a numpy audio array (for mic input)."""
    # Mel
    mel    = librosa.feature.melspectrogram(y=y_array, sr=sr, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < 128:
        mel_db = np.pad(mel_db, ((0,0),(0,128-mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :128]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_in = mel_db[np.newaxis, ..., np.newaxis]  # (1, 64, 128, 1)

    # MFCC
    mfccs = librosa.feature.mfcc(y=y_array, sr=sr, n_mfcc=N_MFCC).T
    if mfccs.shape[0] < MAX_TIME_STEPS:
        mfccs = np.pad(mfccs, ((0, MAX_TIME_STEPS-mfccs.shape[0]),(0,0)), mode='constant')
    else:
        mfccs = mfccs[:MAX_TIME_STEPS, :]
    mu  = mfccs.mean(axis=0); std = mfccs.std(axis=0) + 1e-8
    mfcc_in = ((mfccs-mu)/std)[np.newaxis]   # (1, 128, 40)

    return mel_in, mfcc_in
