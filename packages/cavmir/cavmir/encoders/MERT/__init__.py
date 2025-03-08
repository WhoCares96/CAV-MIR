import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from cavmir.audio import load_and_resample_audio
from cavmir.encoders.interface import AudioEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_ID_REPOSITORY_MAP = {
    "mert_v1_95m": "m-a-p/MERT-v1-95M",
}


class MERTEncoder(AudioEncoder):
    def __init__(self, model_id: str = "mert_v1_95m"):
        self.model_id = model_id
        model_repository = MODEL_ID_REPOSITORY_MAP[model_id]

        self.model = AutoModel.from_pretrained(
            model_repository, trust_remote_code=True
        ).to(device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_repository, trust_remote_code=True
        )
        self.target_sample_rate = self.processor.sampling_rate

    def embed_audio_file(self, audio_path: str, **kwargs) -> np.ndarray:
        return_segments = kwargs.get("return_segments", False)

        wave_data = load_and_resample_audio(
            audio_path, self.target_sample_rate, **kwargs
        )

        inputs = self.processor(
            wave_data, sampling_rate=self.target_sample_rate, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = self.model(
                **inputs, output_hidden_states=False
            )  # adapt if we want to test hidden states

        last_hidden_state = outputs.last_hidden_state

        # by default, we return the mean of the last hidden states
        if not return_segments:
            last_hidden_state = last_hidden_state.mean(-2)

        return last_hidden_state.cpu().numpy()
