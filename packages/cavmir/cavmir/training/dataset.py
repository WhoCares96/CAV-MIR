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
    path_to_files: str,
    batch_size: int,
    data_decoder: Callable = custom_webdataset_decoder,
) -> DataLoader:
    """
    Create a torch DataLoader from a path containing webdataset file(s).
    """

    if os.path.isdir(path_to_files):
        all_webdataset_files = [
            os.path.join(path_to_files, file)
            for file in filter(
                lambda file: file.endswith(".tar"), os.listdir(path_to_files)
            )
            if file.endswith(".tar")
        ]
    elif os.path.isfile(path_to_files):
        all_webdataset_files = [path_to_files]
    else:
        raise ValueError("Invalid path_to_files provided.")

    dataset = wds.WebDataset(all_webdataset_files).decode(data_decoder)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    return dataloader


def create_webdataset(samples: list[TrainingSample], output_dir) -> None:
    """
    Create a webdataset from a list of TrainingSample objects.
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
