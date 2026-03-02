"""UMAP dimensionality reduction for embeddings."""

import numpy as np
import umap

import config


def _safe_umap(embeddings: np.ndarray, n_components: int,
               metric: str, min_dist: float, n_neighbors: int,
               random_state: int) -> np.ndarray:
    """Run UMAP with automatic parameter clamping for small datasets."""
    n_samples = embeddings.shape[0]
    # Clamp n_components and n_neighbors to what the dataset supports
    n_components = min(n_components, n_samples - 1)
    n_neighbors = min(n_neighbors, n_samples - 1)
    if n_components < 1:
        n_components = 1
    if n_neighbors < 2:
        n_neighbors = 2

    # Use random init for small datasets to avoid spectral decomposition issues
    init = "random" if n_samples < 20 else "spectral"

    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        random_state=random_state,
        init=init,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


def reduce_for_clustering(embeddings: np.ndarray,
                          n_components: int = config.UMAP_N_COMPONENTS_CLUSTER,
                          metric: str = config.UMAP_METRIC,
                          min_dist: float = config.UMAP_MIN_DIST_CLUSTER,
                          n_neighbors: int = config.UMAP_N_NEIGHBORS,
                          random_state: int = 42) -> np.ndarray:
    """Reduce embeddings to n_components dims for HDBSCAN input.

    Returns:
        (N, n_components) array of reduced coordinates.
    """
    return _safe_umap(embeddings, n_components, metric, min_dist,
                      n_neighbors, random_state)


def reduce_for_visualization(embeddings: np.ndarray,
                             n_components: int = config.UMAP_N_COMPONENTS_VIZ,
                             metric: str = config.UMAP_METRIC,
                             min_dist: float = config.UMAP_MIN_DIST_VIZ,
                             n_neighbors: int = config.UMAP_N_NEIGHBORS,
                             random_state: int = 42) -> np.ndarray:
    """Reduce embeddings to 2D for visualization.

    Returns:
        (N, 2) array of 2D coordinates.
    """
    return _safe_umap(embeddings, n_components, metric, min_dist,
                      n_neighbors, random_state)
