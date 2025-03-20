import numpy as np
import torch
from muq import MuQ, MuQMuLan

from cavmir.audio import load_and_resample_audio
from cavmir.encoders.interface import AudioEncoder

device = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_ID_REPOSITORY_MAP = {
    "muq_large_msd": "OpenMuQ/MuQ-large-msd-iter",
    "muq_mulan_large": "OpenMuQ/MuQ-MuLan-large",
}


class MuQEncoder(AudioEncoder):
    def __init__(self, model_id: str = "muq_large_msd"):
        self.model_id = model_id
        model_repository = MODEL_ID_REPOSITORY_MAP[model_id]

        match self.model_id:
            case "muq_large_msd":
                self.model = (
                    MuQ.from_pretrained(
                        model_repository,
                    )
                    .to(device)
                    .eval()
                )
            case "muq_mulan_large":
                self.model = (
                    MuQMuLan.from_pretrained(
                        model_repository,
                    )
                    .to(device)
                    .eval()
                )
            case _:
                raise ValueError(f"Model ID {model_id} not recognized.")

        self.target_sample_rate = 24000

    def embed_audio_file(self, audio_path: str, **kwargs) -> np.ndarray:
        return_segments = kwargs.get("return_segments", False)

        wave_data = load_and_resample_audio(audio_path, self.target_sample_rate)

        inputs = torch.tensor(wave_data).unsqueeze(0).to(device)

        if self.model_id == "muq_large_msd":
            options = {"output_hidden_states": False}
        else:
            options = {}

        with torch.no_grad():
            outputs = self.model(inputs, **options)

        match self.model_id:
            case "muq_large_msd":
                outputs = outputs.last_hidden_state
            case "muq_mulan_large":
                outputs = outputs[None]

        print(outputs.shape)

        # by default, we return the mean of the last hidden states
        if not return_segments:
            outputs = outputs.mean(-2)

        return outputs.cpu().numpy()
