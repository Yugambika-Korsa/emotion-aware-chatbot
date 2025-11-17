# src/audio_emotion.py
import numpy as np
import librosa
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "audio_emotion_model.pkl")

def extract_features(y, sr):
    """Return feature vector for an audio time-series."""
    if y.ndim > 1:
        y = librosa.to_mono(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    spec_ct = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_ct_mean = np.mean(spec_ct, axis=1)
    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
    except Exception:
        tonnetz_mean = np.zeros(6)

    features = np.concatenate([
        mfcc_mean, mfcc_std,
        chroma_mean,
        spec_ct_mean,
        tonnetz_mean
    ])
    return features

def prepare_audio_file(path, sr=22050, duration=None):
    y, sr = librosa.load(path, sr=sr, duration=duration)
    return y, sr

class AudioEmotionModel:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        if os.path.exists(self.model_path):
            try:
                self.model = load(self.model_path)
            except Exception:
                self.model = None

    def train(self, X, y, save_path=None):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        print("Validation accuracy:", acc)
        self.model = clf
        path = save_path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump(clf, path)
        print("Saved model to", path)
        return acc

    def predict(self, features):
        if self.model is None:
            raise RuntimeError("No audio model loaded. Train or place model at: " + self.model_path)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.model.predict(features)[0]

    def predict_proba(self, features):
        if self.model is None:
            raise RuntimeError("No audio model loaded.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.model.predict_proba(features)[0]
