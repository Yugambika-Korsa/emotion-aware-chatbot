# src/utils.py
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import tempfile

def record(duration=4, sr=16000, channels=1):
    """
    Record audio from the default microphone and save to a temporary WAV file.
    Returns the path to the saved WAV.
    """
    duration = int(duration)
    sr = int(sr)
    print(f"Recording {duration}s at {sr}Hz...")
    rec = sd.rec(int(duration * sr), samplerate=sr, channels=channels, dtype="float32")
    sd.wait()
    rec = rec.reshape(-1) if channels == 1 else rec
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, rec, sr)
    return tmp.name

def save_array(y: np.ndarray, sr: int, path: str):
    """
    Save numpy audio array to a WAV file.
    """
    sf.write(path, y, sr)

def load_audio(path: str, sr: int = 22050, duration: float = None):
    """
    Convenience wrapper around librosa.load (keeps dependency isolation).
    If you want to avoid importing librosa here, app code can call librosa directly.
    """
    try:
        import librosa
    except Exception as e:
        raise RuntimeError("librosa is required to use load_audio. Install requirements.txt") from e
    y, sr = librosa.load(path, sr=sr, duration=duration)
    return y, sr
