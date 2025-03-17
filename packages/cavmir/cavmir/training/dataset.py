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
    """
    Custom decoder implementation for cavmir webdatasets.
    """

    key = key.split(".")[-1]

    match key:
        case "npz":
            return np.load(BytesIO(data))
        case "npy":
            return np.load(BytesIO(data))
        case _:
            raise NotImplementedError(f"No decoding strategy for key: {key}")


def create_dataloader_from_webdataset_path(
    path_or_file: str,
    batch_size: int,
    data_decoder: Callable = custom_webdataset_decoder,
) -> DataLoader:
    """
    Create a torch DataLoader from a path containing webdataset file(s).

    Parameters
    ----------
    path_or_file: str
        The path to the webdataset file(s) or a single webdataset file.
    batch_size: int
        The batch size to use for the DataLoader.
    data_decoder: Callable
        The function to use to decode the data from the webdataset.

    Returns
    -------
    DataLoader
        A torch DataLoader object containing the data from the webdataset

    Raises
    ------
    ValueError
        If an invalid path_to_files is provided.
    """

    if os.path.isdir(path_or_file):
        all_webdataset_files = [
            os.path.join(path_or_file, file)
            for file in filter(
                lambda file: file.endswith(".tar"), os.listdir(path_or_file)
            )
            if file.endswith(".tar")
        ]
    elif os.path.isfile(path_or_file):
        all_webdataset_files = [path_or_file]
    else:
        raise ValueError("Invalid path_to_files provided.")

    dataset = wds.WebDataset(all_webdataset_files, shardshuffle=False).decode(
        data_decoder
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    return dataloader


def create_webdataset(samples: list[TrainingSample], output_dir) -> None:
    """
    Create a webdataset from a list of TrainingSample objects.

    Parameters
    ----------
    samples: list[TrainingSample]
        A list of TrainingSample objects to write to the webdataset.

    output_dir: str
        Path to store the webdataset to.

    """

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    with open(output_dir, "wb") as webdataset_obj:
        with wds.TarWriter(webdataset_obj) as tar_writer:
            for sample in samples:
                tar_writer.write(
                    {
                        "__key__": sample.id,
                        "npz": {
                            "embedding": sample.embedding,
                            "target": sample.target,
                        },
                    }
                )
