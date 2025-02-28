import librosa
import numpy as np


def load_and_resample_audio(
    audio_path: str, target_sample_rate: int, **kwargs
) -> np.ndarray:
    wave_data, _ = librosa.load(audio_path, sr=target_sample_rate, **kwargs)
    return wave_data
