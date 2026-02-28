"""
Speaker Identification Module
- Registers new speakers with voice embeddings
- Identifies unknown speaker from voice
- Uses MFCC-based speaker embeddings + cosine similarity
"""

import numpy as np
import os
import json
import pickle
from datetime import datetime

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

SPEAKERS_DB   = os.path.join(os.path.dirname(__file__), '..', 'models', 'speakers.json')
EMBEDDINGS_DB = os.path.join(os.path.dirname(__file__), '..', 'models', 'embeddings.pkl')
SAMPLE_RATE   = 22050
N_MFCC        = 40
THRESHOLD     = 0.75   # cosine similarity threshold for identification


def extract_speaker_embedding(y, sr=SAMPLE_RATE):
    """
    Extract a compact speaker embedding vector from audio.
    Uses MFCC statistics + pitch + energy as speaker fingerprint.
    """
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfccs.mean(axis=1)
    mfcc_std  = mfccs.std(axis=1)
    mfcc_delta = librosa.feature.delta(mfccs).mean(axis=1)

    # Spectral
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()

    # Pitch (fundamental frequency)
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_voiced = f0[voiced]
        f0_mean = float(f0_voiced.mean()) if len(f0_voiced) > 0 else 150.0
        f0_std  = float(f0_voiced.std())  if len(f0_voiced) > 0 else 20.0
    except Exception:
        f0_mean, f0_std = 150.0, 20.0

    # Energy
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    embedding = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_delta,
        [centroid, bandwidth, rolloff, f0_mean, f0_std, rms, zcr]
    ])

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def extract_embedding_from_file(filepath):
    """Extract speaker embedding from a WAV file."""
    if not LIBROSA_OK:
        raise RuntimeError("librosa not installed: pip install librosa")
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    return extract_speaker_embedding(y, sr)


def _load_db():
    """Load speakers database."""
    if os.path.exists(SPEAKERS_DB):
        with open(SPEAKERS_DB, 'r') as f:
            return json.load(f)
    return {}


def _save_db(db):
    """Save speakers database."""
    with open(SPEAKERS_DB, 'w') as f:
        json.dump(db, f, indent=2)


def _load_embeddings():
    """Load embeddings dictionary."""
    if os.path.exists(EMBEDDINGS_DB):
        with open(EMBEDDINGS_DB, 'rb') as f:
            return pickle.load(f)
    return {}


def _save_embeddings(embs):
    """Save embeddings dictionary."""
    with open(EMBEDDINGS_DB, 'wb') as f:
        pickle.dump(embs, f)


def register_speaker(name, audio_files):
    """
    Register a new speaker from one or more audio files.
    Averages embeddings from all provided files for robustness.

    Args:
        name: Speaker name string
        audio_files: list of file paths OR list of numpy arrays

    Returns: success message
    """
    if not LIBROSA_OK:
        raise RuntimeError("librosa not installed")

    embeddings_list = []
    for item in audio_files:
        if isinstance(item, str):
            y, sr = librosa.load(item, sr=SAMPLE_RATE, mono=True)
        else:
            y, sr = item, SAMPLE_RATE
        emb = extract_speaker_embedding(y, sr)
        embeddings_list.append(emb)

    # Mean embedding
    mean_emb = np.mean(embeddings_list, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    # Save
    db   = _load_db()
    embs = _load_embeddings()

    speaker_id = f"spk_{len(db)+1:03d}"
    db[speaker_id] = {
        'name':       name,
        'registered': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'n_samples':  len(audio_files)
    }
    embs[speaker_id] = mean_emb

    _save_db(db)
    _save_embeddings(embs)

    return f"Speaker '{name}' registered as {speaker_id} ({len(audio_files)} sample(s))"


def identify_speaker(audio_input):
    """
    Identify speaker from audio.

    Args:
        audio_input: file path (str) OR numpy array

    Returns:
        dict with keys: name, speaker_id, confidence, all_scores
    """
    if not LIBROSA_OK:
        raise RuntimeError("librosa not installed")

    db   = _load_db()
    embs = _load_embeddings()

    if not db:
        return {
            'name':       'Unknown (No speakers registered)',
            'speaker_id': None,
            'confidence': 0.0,
            'all_scores': {}
        }

    # Extract query embedding
    if isinstance(audio_input, str):
        y, sr = librosa.load(audio_input, sr=SAMPLE_RATE, mono=True)
    else:
        y, sr = audio_input, SAMPLE_RATE

    query_emb = extract_speaker_embedding(y, sr)

    # Cosine similarity with all registered speakers
    scores = {}
    for spk_id, emb in embs.items():
        sim = float(np.dot(query_emb, emb))   # both are L2-normalized
        scores[spk_id] = sim

    best_id  = max(scores, key=scores.get)
    best_sim = scores[best_id]

    if best_sim >= THRESHOLD:
        name       = db[best_id]['name']
        confidence = best_sim * 100
    else:
        name       = 'Unknown Speaker'
        confidence = (1 - best_sim) * 100

    all_scores = {
        db[sid]['name']: round(scores[sid] * 100, 1)
        for sid in scores
    }

    return {
        'name':       name,
        'speaker_id': best_id if best_sim >= THRESHOLD else None,
        'confidence': round(confidence, 1),
        'all_scores': all_scores,
        'matched':    best_sim >= THRESHOLD
    }


def list_speakers():
    """Return list of all registered speakers."""
    db = _load_db()
    return [{'id': sid, **info} for sid, info in db.items()]


def delete_speaker(speaker_id):
    """Remove a speaker from the database."""
    db   = _load_db()
    embs = _load_embeddings()
    if speaker_id in db:
        name = db[speaker_id]['name']
        del db[speaker_id]
        del embs[speaker_id]
        _save_db(db)
        _save_embeddings(embs)
        return f"Speaker '{name}' deleted."
    return "Speaker not found."
