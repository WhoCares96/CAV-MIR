"""
Collection of utility functions specific to how we process the training data, evaluation and tcav scoring.
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from fsspec import AbstractFileSystem


def create_embedding_path(song_id: str, embedding_prefix: str, encoder_id: str) -> str:
    return os.path.join(embedding_prefix, encoder_id, f"{song_id}.{encoder_id}.npy")


def load_embedding(
    song_id: str, embedding_prefix: str, encoder_id: str, fs: AbstractFileSystem
) -> np.ndarray:
    embedding_path = create_embedding_path(song_id, embedding_prefix, encoder_id)
    embedding = np.load(io.BytesIO(fs.cat(embedding_path)))[0]

    return embedding


def load_embeddings(
    dataset: pd.DataFrame,
    embedding_prefix: str,
    encoder_id: str,
    fs: AbstractFileSystem,
) -> np.ndarray:
    with ThreadPoolExecutor() as executor:
        return np.array(
            list(
                executor.map(
                    lambda row: load_embedding(
                        row.song_id, embedding_prefix, encoder_id, fs
                    ),
                    dataset.itertuples(),
                )
            )
        )
