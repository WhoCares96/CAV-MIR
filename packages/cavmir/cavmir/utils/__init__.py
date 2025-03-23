"""
Collection of utility functions specific to how we process the training data, evaluation and tcav scoring.
"""

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import pandas as pd
import torch
from fsspec import AbstractFileSystem
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from torch.utils.data import DataLoader

from cavmir.training.dataset import (
    TrainingSample,
    create_dataloader_from_webdataset_path,
    create_webdataset,
)
from cavmir.training.evaluate import evaluate_cav_model
from cavmir.training.fit import fit_cav_model
from cavmir.training.network import CAVNetwork


def create_embedding_path(song_id: str, embedding_prefix: str, encoder_id: str) -> str:
    return os.path.join(embedding_prefix, encoder_id, f"{song_id}.{encoder_id}.npy")


def load_embedding(
    song_id: str, embedding_prefix: str, encoder_id: str, fs: AbstractFileSystem
) -> np.ndarray:
    embedding_path = create_embedding_path(song_id, embedding_prefix, encoder_id)
    embedding = np.load(io.BytesIO(fs.cat(embedding_path)))[0]

    return embedding


def load_embeddings(
    song_ids: list[str],
    embedding_prefix: str,
    encoder_id: str,
    fs: AbstractFileSystem,
) -> np.ndarray:
    with ThreadPoolExecutor() as executor:
        return np.array(
            list(
                executor.map(
                    lambda song_id: load_embedding(
                        song_id, embedding_prefix, encoder_id, fs
                    ),
                    song_ids,
                )
            )
        )


def cache_df(cache_dir: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_file = os.path.join(
                cache_dir,
                f"{func.__name__}_{args[0]}_{args[1]}_{args[2]}.pkl",
            )

            if os.path.exists(cache_file):
                return pd.read_pickle(cache_file)
            else:
                result = func(*args, **kwargs)
                result.to_pickle(cache_file)
                return result

        return wrapper

    return decorator


@cache_df(cache_dir="/tmp")
def load_df_and_embeddings(
    project_name: str,
    dataset_type: Literal["train", "test"],
    encoder_id: str,
    dataset_prefix: str,
    embedding_prefix: str,
    fs: AbstractFileSystem,
) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(dataset_prefix, f"{dataset_type}_dataset_{project_name}.csv")
    )
    embeddings = load_embeddings(df.song_id.tolist(), embedding_prefix, encoder_id, fs)
    df["embedding"] = embeddings.tolist()

    return df


def create_training_samples_from_df(df: pd.DataFrame) -> list[TrainingSample]:
    training_samples = []

    for row in df.itertuples():
        training_sample = TrainingSample(
            id=str(row.song_id),
            embedding=np.array(row.embedding),
            target=np.array([row.target]),
        )

        training_sample.validate_attributes()
        training_samples.append(training_sample)

    return training_samples


def create_in_memory_test_dataloader(
    df: pd.DataFrame, batch_size: int | None = None
) -> DataLoader:
    embeddings = np.asarray(df["embedding"].values)
    labels = df["target"].values

    samples = [
        {
            "npz": {
                "embedding": torch.tensor(embedding, dtype=torch.float),
                "target": torch.tensor(label, dtype=torch.float)[None],
            }
        }
        for embedding, label in zip(embeddings, labels)
    ]

    if not batch_size:
        batch_size = len(df)

    return DataLoader(samples, batch_size=batch_size, shuffle=False)


def create_subset_for_training(
    df: pd.DataFrame,
    training_size: int,
    validation_size: int,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a subset of the dataframe for training and validation.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to create the subset from.
    training_size : int
        The size of the training set
    random_state : int | None, optional
        The random state to use for sampling, by default None
    shuffle : bool, optional
        Whether to shuffle the data, by default True

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The training and validation dataframes
    """

    def equally_sample_from_df(
        df: pd.DataFrame, n: int, random_state: int
    ) -> pd.DataFrame:
        df_positive = df[df["target"] == 1]
        df_negative = df[df["target"] == 0]

        df_positive_sample = df_positive.sample(n=n // 2, random_state=random_state)
        df_negative_sample = df_negative.sample(n=n // 2, random_state=random_state)

        return pd.concat([df_positive_sample, df_negative_sample])

    if random_state is None:
        random_state = np.random.randint(0, 2**32 - 1)

    df_train = equally_sample_from_df(df, training_size, random_state)

    df_val = equally_sample_from_df(
        df.drop(df_train.index), validation_size, random_state
    )

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=random_state)
        df_val = df_val.sample(frac=1, random_state=random_state)

    return df_train, df_val


def get_CAV_logistic(X: np.ndarray, y: np.ndarray, random_state: int) -> np.ndarray:
    lr = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=1000,
        random_state=random_state,
    )
    lr.fit(X, y)
    return np.atleast_2d(lr.coef_)


def lda_one_cav(
    random_state: int,
    df: pd.DataFrame,
    project_name: str,
    training_sample_size: int,
    embedding_dim: int,
    test_dataloader: torch.utils.data.DataLoader,
    plot_evaluation: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Perform one training run for a LDA model consisting of:
    - Creating a training and validation dataset
    - Training the model
    - Evaluating the model
    - displaying the evaluation results
    - Returning the CAV vector
    """

    df_train, _ = create_subset_for_training(
        df=df,
        training_size=training_sample_size,
        validation_size=0,
        random_state=random_state,
        shuffle=True,
    )

    train_embeddings = np.array([np.array(x) for x in df_train.embedding.values])
    train_targets = np.array([np.array(x) for x in df_train.target.values])

    cav = get_CAV_logistic(train_embeddings, train_targets, random_state)

    model = CAVNetwork(
        input_shape=embedding_dim,
        target_shape=1,
        dropout_rate=0.0,
    )
    model.set_concept_activation_vector(cav)

    evaluation_metrics = evaluate_cav_model(
        model=model,
        test_dataloader=test_dataloader,
        true_label_name=project_name,
        loss_history_dir=None,
        plot_evaluation=plot_evaluation,
    )

    return cav, evaluation_metrics


def train_one_cav(
    random_state: int,
    df: pd.DataFrame,
    project_name: str,
    training_sample_count: int,
    validation_sample_count: int,
    epochs: int,
    learning_rate: float,
    embedding_dim: int,
    dropout_rate: float,
    df_test: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """Perform one training run for a CAV model consisting of:
    - Creating a training and validation dataset
    - Training the model
    - Evaluating the model
    - displaying the evaluation results
    - Returning the CAV vector
    """

    if validation_sample_count is None:
        validation_sample_count = -1

    df_train, df_val = create_subset_for_training(
        df=df,
        training_size=training_sample_count,
        validation_size=validation_sample_count,
        random_state=random_state,
        shuffle=True,
    )

    train_dataloader = create_in_memory_test_dataloader(df_train)
    val_dataloader = create_in_memory_test_dataloader(df_val)
    test_dataloader = create_in_memory_test_dataloader(df_test)

    model = CAVNetwork(
        input_shape=embedding_dim,
        target_shape=1,
        dropout_rate=dropout_rate,
    )

    fit_cav_model(
        model=model,
        train_dataset=train_dataloader,
        val_dataset=val_dataloader,
        out_files_dir=f"trainings/{project_name}/",
        num_epochs=epochs,
        learning_rate=learning_rate,
        verbose_steps=100,
    )

    evaluation_metrics = evaluate_cav_model(
        model=model,
        test_dataloader=test_dataloader,
        true_label_name=project_name,
        loss_history_dir=f"trainings/{project_name}/loss_history.json",
        plot_evaluation=True,
    )

    cav_vector = model.get_concept_activation_vector()

    return cav_vector, evaluation_metrics


def store_cav_vector_array(
    data: np.ndarray | list[np.ndarray],
    file_name: str,
    encoder_id: str,
    project_name: str,
):
    os.makedirs(
        os.path.join(
            "trainings",
            encoder_id,
            project_name,
        ),
        exist_ok=True,
    )

    np.save(
        os.path.join(
            "trainings",
            encoder_id,
            project_name,
            file_name,
        ),
        np.array(data),
    )


def store_evaluation_metrics(
    data: dict | list[dict], file_name: str, encoder_id: str, project_name: str
):
    os.makedirs(
        os.path.join(
            "trainings",
            encoder_id,
            project_name,
        ),
        exist_ok=True,
    )

    json.dump(
        data,
        open(
            os.path.join("trainings", encoder_id, project_name, file_name),
            "w",
        ),
    )
