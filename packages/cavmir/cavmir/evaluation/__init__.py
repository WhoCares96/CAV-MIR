import numpy as np


def calculate_tcav_score(
    cav_weight: np.ndarray, cav_bias: float | np.ndarray, target_embeddings: np.ndarray
) -> float:
    """
    Calculate the TCAV score for a given CAV vector and target embeddings.

    Both, the CAV vector and the target embeddings must be from the same embedding space.
    The target embeddings should be from a balanced dataset.

    Parameters
    ----------
    cav_vector : np.ndarray
        The CAV vector. Shape: (n_features) or (1, n_features)
    target_embeddings : np.ndarray
        The target embeddings. Shape: (n_samples, n_features)

    Returns
    -------
    float
        The TCAV score.
    """
    target_activations = (
        np.dot(target_embeddings, np.atleast_2d(cav_weight).T) + cav_bias
    )
    tcav_score = np.sum(target_activations > 0) / len(target_activations)

    return tcav_score
