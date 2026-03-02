"""UMAP dimensionality reduction for embeddings."""

import numpy as np
import umap

import config


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
    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


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
    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)
