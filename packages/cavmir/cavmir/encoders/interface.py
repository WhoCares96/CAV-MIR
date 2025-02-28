import abc

import numpy as np


class AudioEncoder(abc.ABC):
    @abc.abstractmethod
    def embed_audio_file(self, audio_path: str, **kwargs) -> np.ndarray:
        pass
