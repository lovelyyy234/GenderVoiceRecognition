"""
Gender Voice Recognition System V3
Flask Web Application
Routes:
  GET  /                  — Main dashboard
  POST /predict           — Predict from uploaded WAV
  POST /register_speaker  — Register new speaker
  GET  /speakers          — List all speakers
  DELETE /speaker/<id>    — Delete speaker
  GET  /history           — Session history (JSON)
  GET  /model_stats       — Model accuracy stats (JSON)
"""

import os, sys, json, uuid, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import numpy as np

# ── modules ──────────────────────────────────────────────────
from modules.ml_gender       import train_ml_models, load_ml_models, predict_ml
from modules.emotion_detector import (load_emotion_model, train_emotion_model_synthetic,
                                       extract_emotion_features_from_file, predict_emotion)
from modules.speaker_id      import (register_speaker, identify_speaker,
                                      list_speakers, delete_speaker)

try:
    from modules.feature_extractor import extract_classical_features, extract_mel_spectrogram, extract_mfcc_sequence
    from modules.cnn_model  import load_cnn,  predict_cnn
    from modules.lstm_model import load_lstm, predict_lstm
    DL_OK = True
except Exception:
    DL_OK = False

# ── Flask setup ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'gender_voice_v3_secret_2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

# ── Global model state ────────────────────────────────────────
ML_RESULTS = {}
ML_SCALER  = None
ML_BEST    = None
CNN_MODEL  = None
LSTM_MODEL = None
EMOTION_MODEL  = None
EMOTION_SCALER = None
HISTORY    = []   # in-memory session history

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def estimate_age_group(meanfun_hz):
    if meanfun_hz > 250:   return "Child"
    elif meanfun_hz > 165: return "Young / Adult"
    elif meanfun_hz > 100: return "Adult"
    else:                  return "Older Adult"

# ── Load all models on startup ────────────────────────────────
def init_models():
    global ML_RESULTS, ML_SCALER, ML_BEST, CNN_MODEL, LSTM_MODEL
    global EMOTION_MODEL, EMOTION_SCALER

    print("\n" + "="*55)
    print("  Initializing Gender Voice Recognition V3")
    print("="*55)

    # ML Models
    print("\n[1/4] ML Models...")
    ML_RESULTS, ML_SCALER, ML_BEST = load_ml_models()
    if ML_RESULTS is None:
        print("  Training ML models from scratch...")
        ML_RESULTS, ML_SCALER, ML_BEST = train_ml_models()
    else:
        print(f"  Loaded saved ML models. Best: {ML_BEST}")

    # CNN
    if DL_OK:
        print("\n[2/4] CNN Model...")
        CNN_MODEL = load_cnn()
        if CNN_MODEL:
            print("  CNN model loaded.")
        else:
            print("  CNN model not found. Train with: python train_deep.py --model cnn")

        # LSTM
        print("\n[3/4] LSTM Model...")
        LSTM_MODEL = load_lstm()
        if LSTM_MODEL:
            print("  LSTM model loaded.")
        else:
            print("  LSTM model not found. Train with: python train_deep.py --model lstm")
    else:
        print("\n[2/4] Deep Learning (TensorFlow not installed - skipped)")
        print("[3/4] LSTM skipped")

    # Emotion Model
    print("\n[4/4] Emotion Model...")
    EMOTION_MODEL, EMOTION_SCALER = load_emotion_model()
    if EMOTION_MODEL is None:
        print("  Training emotion model (synthetic data)...")
        EMOTION_MODEL, EMOTION_SCALER = train_emotion_model_synthetic()
    else:
        print("  Emotion model loaded.")

    print("\n" + "="*55)
    print("  All models ready. App running at http://127.0.0.1:5000")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    ml_stats = []
    if ML_RESULTS:
        for name, data in ML_RESULTS.items():
            ml_stats.append({
                'name':     name,
                'accuracy': round(data['accuracy'], 2),
                'std':      round(data['std'], 2),
                'best':     name == ML_BEST
            })
    return render_template('index.html',
                            ml_stats=ml_stats,
                            best_model=ML_BEST,
                            dl_available=DL_OK,
                            cnn_loaded=CNN_MODEL is not None,
                            lstm_loaded=LSTM_MODEL is not None)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file. Upload WAV/MP3/OGG/FLAC'}), 400

    selected_model = request.form.get('model', 'ml')

    # Save file temporarily
    fname    = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    filepath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(filepath)

    result = {}
    try:
        # ── Gender Prediction ────────────────────────────────
        gender, gender_conf, model_used = "UNKNOWN", 0.0, "none"

        if selected_model == 'cnn' and CNN_MODEL:
            mel    = extract_mel_spectrogram(filepath)
            gender, gender_conf = predict_cnn(CNN_MODEL, mel[np.newaxis])
            model_used = "CNN"

        elif selected_model == 'lstm' and LSTM_MODEL:
            mfcc   = extract_mfcc_sequence(filepath)
            gender, gender_conf = predict_lstm(LSTM_MODEL, mfcc[np.newaxis])
            model_used = "BiLSTM"

        else:  # Default: ML
            feats = extract_classical_features(filepath)
            if feats:
                gender, gender_conf = predict_ml(
                    ML_RESULTS[ML_BEST]['model'], ML_SCALER, feats)
                model_used = f"ML ({ML_BEST})"
                # Age group from meanfun (index 6)
                meanfun_hz = feats[0][6] * 1000
                result['age_group'] = estimate_age_group(meanfun_hz)

        result['gender']     = gender
        result['confidence'] = round(gender_conf, 1)
        result['model_used'] = model_used

        # ── Emotion Detection ────────────────────────────────
        try:
            emotion_feats = extract_emotion_features_from_file(filepath)
            emo, emoji, emo_conf, all_probas = predict_emotion(
                EMOTION_MODEL, EMOTION_SCALER, emotion_feats)
            result['emotion']         = emo
            result['emotion_emoji']   = emoji
            result['emotion_conf']    = round(emo_conf, 1)
            result['emotion_probas']  = {k: round(v, 1) for k,v in all_probas.items()}
        except Exception as e:
            result['emotion']       = 'N/A'
            result['emotion_emoji'] = '❓'
            result['emotion_conf']  = 0.0
            result['emotion_probas']= {}

        # ── Speaker Identification ───────────────────────────
        try:
            spk = identify_speaker(filepath)
            result['speaker']      = spk['name']
            result['speaker_conf'] = spk['confidence']
            result['speaker_matched'] = spk['matched']
            result['speaker_scores']  = spk['all_scores']
        except Exception as e:
            result['speaker']      = 'N/A'
            result['speaker_conf'] = 0.0
            result['speaker_matched'] = False
            result['speaker_scores']  = {}

        # ── Save to history ──────────────────────────────────
        import datetime
        entry = {
            'time':        datetime.datetime.now().strftime("%H:%M:%S"),
            'filename':    file.filename,
            'gender':      result.get('gender', '?'),
            'confidence':  result.get('confidence', 0),
            'emotion':     result.get('emotion', 'N/A'),
            'speaker':     result.get('speaker', 'N/A'),
            'model':       model_used
        }
        HISTORY.insert(0, entry)
        if len(HISTORY) > 100:
            HISTORY.pop()

    except Exception as e:
        result['error'] = str(e)
    finally:
        try: os.remove(filepath)
        except: pass

    return jsonify(result)


@app.route('/register_speaker', methods=['POST'])
def api_register_speaker():
    name  = request.form.get('name', '').strip()
    files = request.files.getlist('files')

    if not name:
        return jsonify({'error': 'Speaker name is required'}), 400
    if not files:
        return jsonify({'error': 'At least one audio file required'}), 400

    saved = []
    try:
        for f in files:
            if allowed_file(f.filename):
                fname = secure_filename(f"{uuid.uuid4().hex}_{f.filename}")
                path  = os.path.join(UPLOAD_FOLDER, fname)
                f.save(path)
                saved.append(path)

        msg = register_speaker(name, saved)
        return jsonify({'message': msg, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for p in saved:
            try: os.remove(p)
            except: pass


@app.route('/speakers', methods=['GET'])
def api_speakers():
    return jsonify(list_speakers())


@app.route('/speaker/<speaker_id>', methods=['DELETE'])
def api_delete_speaker(speaker_id):
    msg = delete_speaker(speaker_id)
    return jsonify({'message': msg})


@app.route('/history', methods=['GET'])
def api_history():
    return jsonify(HISTORY)


@app.route('/model_stats', methods=['GET'])
def api_model_stats():
    stats = []
    if ML_RESULTS:
        for name, data in ML_RESULTS.items():
            stats.append({
                'name': name,
                'accuracy': round(data['accuracy'], 2),
                'std': round(data['std'], 2),
                'best': name == ML_BEST
            })
    return jsonify({
        'ml_models': stats,
        'best_ml': ML_BEST,
        'cnn_available': CNN_MODEL is not None,
        'lstm_available': LSTM_MODEL is not None
    })


@app.route('/clear_history', methods=['POST'])
def clear_history():
    HISTORY.clear()
    return jsonify({'message': 'History cleared'})


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_models()
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
