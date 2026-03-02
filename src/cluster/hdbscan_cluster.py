"""HDBSCAN clustering for website style grouping."""

import hdbscan
import numpy as np

import config


def cluster_embeddings(
    reduced: np.ndarray,
    min_cluster_size: int = config.HDBSCAN_MIN_CLUSTER_SIZE,
    cluster_selection_method: str = config.HDBSCAN_CLUSTER_SELECTION,
) -> tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    """Run HDBSCAN on reduced embeddings.

    Args:
        reduced: (N, D) array of UMAP-reduced embeddings.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        cluster_selection_method: 'eom' or 'leaf'.

    Returns:
        (labels, probabilities, clusterer) where labels[i]=-1 means noise.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    clusterer.fit(reduced)
    return clusterer.labels_, clusterer.probabilities_, clusterer


def get_cluster_stats(labels: np.ndarray) -> dict:
    """Compute cluster statistics."""
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    unique_clusters = set(labels) - {-1}
    n_clusters = len(unique_clusters)
    noise_ratio = n_noise / n_total if n_total > 0 else 0

    cluster_sizes = {}
    for cid in sorted(unique_clusters):
        cluster_sizes[int(cid)] = int((labels == cid).sum())

    return {
        "n_total": n_total,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "cluster_sizes": cluster_sizes,
    }
