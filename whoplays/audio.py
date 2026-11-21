import librosa


def load_audio(path, sr, offset=0.0, duration=None):
    x, _ = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)
    return x
