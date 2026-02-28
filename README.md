# 🎙️ Gender Voice Recognition V3 — Web App Edition

## What's New in V3

| Feature | V2 (Desktop) | V3 (Web App) |
|---|---|---|
| Interface | Tkinter Desktop | Flask Web Browser |
| Gender Detection | ML only | ML + CNN + BiLSTM |
| Emotion Detection | ❌ | ✅ 7 emotions |
| Speaker Identification | ❌ | ✅ Register + Identify |
| CNN (Mel Spectrogram) | ❌ | ✅ |
| BiLSTM (MFCC Sequence) | ❌ | ✅ |
| Multi-language Ready | ❌ | ✅ Upload any WAV |
| History Export | CSV only | CSV from browser |
| Access | One PC | Any browser on network |

---

## 📁 Project Structure

```
GenderVoiceV3/
├── app.py                  ← Flask web server (MAIN FILE)
├── train_deep.py           ← Train CNN + LSTM models
├── requirements.txt        ← All dependencies
├── setup_and_run.bat       ← Windows: auto-install + run
│
├── modules/
│   ├── feature_extractor.py ← Mel, MFCC, classical features
│   ├── ml_gender.py         ← SVM, RF, XGBoost models
│   ├── cnn_model.py         ← CNN architecture + train/predict
│   ├── lstm_model.py        ← BiLSTM architecture + train/predict
│   ├── emotion_detector.py  ← 7-emotion detection
│   └── speaker_id.py        ← Speaker registration + identification
│
├── templates/
│   └── index.html           ← Web UI (single page)
│
├── models/                  ← Saved model files (auto-created)
├── data/
│   └── voice.csv            ← Training dataset
└── uploads/                 ← Temp file storage (auto-cleaned)
```

---

## 🚀 HOW TO RUN — Step by Step (Windows CMD)

### OPTION A: One-Click (Recommended)
Double-click `setup_and_run.bat`
That's it. It installs everything and opens the server.

---

### OPTION B: Manual CMD Commands

**Step 1 — Open CMD in the project folder**
```
Win + R → type cmd → Enter
cd "C:\path\to\GenderVoiceV3"
```
Or: Open File Explorer → go to GenderVoiceV3 → click address bar → type `cmd`

**Step 2 — Install all packages**
```cmd
pip install flask werkzeug numpy pandas scipy scikit-learn xgboost librosa soundfile
```

**Step 3 — Install TensorFlow (for CNN + LSTM)**
```cmd
pip install tensorflow
```

**Step 4 — Run the web app**
```cmd
python app.py
```

**Step 5 — Open browser**
```
http://127.0.0.1:5000
```

---

## 🧠 Train Deep Learning Models (CNN + LSTM)

After the app is set up, train the CNN and LSTM models:

**Train CNN only:**
```cmd
python train_deep.py --model cnn
```

**Train LSTM only:**
```cmd
python train_deep.py --model lstm
```

**Train both:**
```cmd
python train_deep.py --model both
```
Training takes ~5-15 minutes depending on your CPU/GPU.
After training, restart the app — CNN and LSTM tabs will unlock.

---

## 👤 How to Use Speaker Identification

1. Go to **Speakers tab** in the browser
2. Enter speaker name (e.g. "Kumar")
3. Upload 3-5 WAV recordings of that person speaking
4. Click **Register Speaker**
5. Go back to **Predict tab** → upload a WAV → the system will match the speaker

---

## 🌐 Access from Another Device (Same WiFi)

Find your computer's IP address:
```cmd
ipconfig
```
Look for `IPv4 Address` e.g. `192.168.1.5`

On any phone/laptop on the same WiFi, open browser:
```
http://192.168.1.5:5000
```

---

## 🐛 Troubleshooting

### "No module named flask"
```cmd
pip install flask
```

### "No module named librosa"
```cmd
pip install librosa soundfile
```

### "No module named tensorflow"
```cmd
pip install tensorflow
```
If TF install fails, the app still works with ML models (SVM/RF/XGBoost).

### Port 5000 already in use
```cmd
python app.py
```
Edit `app.py` last line: change `port=5000` to `port=5001`

### App shows "CNN not trained"
Run: `python train_deep.py --model both` then restart app.

---

## 🎙️ Supported Audio Formats
WAV · MP3 · OGG · FLAC · M4A

---

## 📊 Model Accuracy
| Model | Accuracy |
|---|---|
| SVM (RBF) | 96.78% |
| Random Forest | 96.72% |
| Gradient Boosting | 96.72% |
| CNN (after training) | ~94-97% |
| BiLSTM (after training) | ~93-96% |
