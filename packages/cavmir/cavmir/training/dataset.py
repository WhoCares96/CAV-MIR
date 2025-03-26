import os
from dataclasses import dataclass
from io import BytesIO
from typing import Callable

import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader


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


def custom_webdataset_decoder(key, data):
    key = key.split(".")[-1]

    if key in {"npz", "npy"}:
        return np.load(BytesIO(data), allow_pickle=False)  # Explicitly disable pickle
    else:
        raise NotImplementedError(f"No decoding strategy for key: {key}")


def create_dataloader_from_webdataset_path(
    path_or_file: str,
    batch_size: int,
    data_decoder: Callable = custom_webdataset_decoder,
    shuffle: bool = False,  # Optional parameter for shuffling
    num_workers: int = 4,  # Use more workers for parallel loading
) -> DataLoader:
    """
    Create a torch DataLoader from a WebDataset file or directory.

    Parameters
    ----------
    path_or_file: str
        The path to the WebDataset file(s) or directory.
    batch_size: int
        The batch size to use.
    data_decoder: Callable
        The function to use to decode data.
    shuffle: bool
        Whether to shuffle dataset shards.
    num_workers: int
        Number of DataLoader workers (recommended: 4-8 for large data).

    Returns
    -------
    DataLoader
        A PyTorch DataLoader.
    """

    if os.path.isdir(path_or_file):
        all_webdataset_files = [
            os.path.join(path_or_file, file)
            for file in os.listdir(path_or_file)
            if file.endswith(".tar")
        ]
    elif os.path.isfile(path_or_file):
        all_webdataset_files = [path_or_file]
    else:
        raise ValueError("Invalid path provided.")

    dataset = wds.WebDataset(all_webdataset_files, shardshuffle=shuffle).decode(
        data_decoder
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    return dataloader


def create_webdataset(samples: list[TrainingSample], output_dir: str) -> None:
    """
    Create a WebDataset from a list of TrainingSample objects.

    Parameters
    ----------
    samples: list[TrainingSample]
        List of TrainingSample objects.
    output_dir: str
        Path to store the dataset.
    """

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    with wds.TarWriter(output_dir) as tar_writer:
        for sample in samples:
            buffer = BytesIO()
            np.savez_compressed(
                buffer, embedding=sample.embedding, target=sample.target
            )
            tar_writer.write({"__key__": sample.id, "npz": buffer.getvalue()})
