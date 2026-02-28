"""
Classical ML Gender Model
Trains 5 models on acoustic features, auto-selects best.
"""

import numpy as np
import pandas as pd
import pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False

FEATURE_LABELS = ['mode','minfun','maxdom','Q25','Q75','IQR','meanfun','median','skew']
MODEL_PATH     = os.path.join(os.path.dirname(__file__), '..', 'models', 'ml_gender.pkl')
SCALER_PATH    = os.path.join(os.path.dirname(__file__), '..', 'models', 'ml_scaler.pkl')
DATA_FILE      = os.path.join(os.path.dirname(__file__), '..', 'data', 'voice.csv')


def load_data():
    data = pd.read_csv(DATA_FILE)
    x    = data[FEATURE_LABELS]
    y    = data['label'].replace({'male': 0, 'female': 1}).astype(int)
    return x, y


def train_ml_models():
    x, y = load_data()
    scaler   = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    candidates = {
        'Logistic Regression': LogisticRegression(solver='liblinear', max_iter=1000),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, random_state=42),
        'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=42),
    }
    if XGB_OK:
        candidates['XGBoost'] = XGBClassifier(n_estimators=150, eval_metric='logloss', random_state=42)

    results = {}
    print("\n  Training ML models...")
    for name, mdl in candidates.items():
        scores = cross_val_score(mdl, x_scaled, y, cv=5)
        mdl.fit(x_scaled, y)
        results[name] = {'model': mdl, 'accuracy': scores.mean()*100, 'std': scores.std()*100}
        print(f"    {name:<25} {scores.mean()*100:.2f}%")

    best = max(results, key=lambda k: results[k]['accuracy'])
    print(f"    Best: {best} ({results[best]['accuracy']:.2f}%)")

    with open(MODEL_PATH,  'wb') as f: pickle.dump({'results': results, 'best': best}, f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)

    return results, scaler, best


def load_ml_models():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH,  'rb') as f: data   = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        return data['results'], scaler, data['best']
    return None, None, None


def predict_ml(model, scaler, features):
    """features: list of feature vectors [[f1,f2,...,f9], ...]"""
    x_sc   = scaler.transform(features)
    probas = model.predict_proba(x_sc)
    votes  = model.predict(x_sc)
    female = int(np.sum(votes == 1))
    result = "FEMALE" if female > len(votes)//2 else "MALE"
    idx    = 1 if result == "FEMALE" else 0
    conf   = float(np.mean(probas[:, idx])) * 100
    return result, conf
