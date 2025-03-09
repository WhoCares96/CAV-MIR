import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoProcessor,
)

from cavmir.audio import load_and_resample_audio
from cavmir.encoders.interface import AudioEncoder

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_ID_REPOSITORY_MAP = {
    "whisper_large_turbo": "openai/whisper-large-v3-turbo",
}


class WhisperEncoder(AudioEncoder):
    def __init__(self, model_id: str = "whisper_large_turbo"):
        self.model_id = model_id
        model_repository = MODEL_ID_REPOSITORY_MAP[model_id]

        self.model = AutoModel.from_pretrained(
            model_repository,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(model_repository)

        self.target_sample_rate = self.processor.feature_extractor.sampling_rate

    def embed_audio_file(self, audio_path: str, **kwargs) -> np.ndarray:
        return_segments = kwargs.get("return_segments", False)

        wave_data = load_and_resample_audio(
            audio_path, self.target_sample_rate, **kwargs
        )

        inputs = (
            self.processor(
                wave_data, sampling_rate=self.target_sample_rate, return_tensors="pt"
            )["input_features"]
            .to(torch_dtype)
            .to(device)
        )
        with torch.no_grad():
            last_hidden_state = self.model.encoder(
                inputs, output_hidden_states=False
            ).last_hidden_state

        # by default, we return the mean of the last hidden states
        if not return_segments:
            last_hidden_state = last_hidden_state.mean(-2)

        return last_hidden_state.cpu().numpy().astype(np.float32)
