"""Cosine similarity search over CLIP embeddings."""

import numpy as np


def cosine_similarity_matrix(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and all embeddings.

    Args:
        query: (D,) L2-normalized query vector.
        embeddings: (N, D) L2-normalized embedding matrix.

    Returns:
        (N,) array of cosine similarities.
    """
    # Since both are L2-normalized, cosine sim = dot product
    return embeddings @ query


def top_k_similar(query: np.ndarray, embeddings: np.ndarray,
                  site_ids: list[int], k: int = 10) -> list[dict]:
    """Find top-K most similar items.

    Returns:
        List of dicts with 'site_id', 'score', 'rank'.
    """
    scores = cosine_similarity_matrix(query, embeddings)
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "site_id": site_ids[idx],
            "score": float(scores[idx]),
            "rank": rank,
        })
    return results
