import numpy as np

@dataclass
class TrainingSample:
    id: str
    embedding: np.ndarray
    target: np.ndarray

    def validate_attributes(self):
        if not isinstance(self.id, str):
            raise ValueError(f"id must be `str` (is `{type(self.id)}`)")
        if not isinstance(self.embedding, np.ndarray):
            raise ValueError(
                f"embedding must be `np.ndarray` (is `{type(self.embedding)}`)"
            )
        if not isinstance(self.target, np.ndarray):
            raise ValueError(f"target must be `np.ndarray` (is `{type(self.target)}`)")

