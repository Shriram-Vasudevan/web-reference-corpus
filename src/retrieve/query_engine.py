"""Query interface for style-based website retrieval."""

from collections import Counter
from pathlib import Path

import numpy as np

import config
from src.embed.clip_embedder import CLIPEmbedder
from src.retrieve.similarity import top_k_similar
from src.utils.storage import (
    get_connection, get_all_embeddings, get_site_cluster,
    get_style_label, get_latest_run_id, get_site_by_url,
    get_embedding,
)


class QueryEngine:
    """Unified query interface for text, image, and URL-based retrieval."""

    def __init__(self, conn=None, embedder: CLIPEmbedder | None = None):
        self.conn = conn or get_connection()
        self.embedder = embedder
        self._site_ids = None
        self._embeddings = None
        self.run_id = get_latest_run_id(self.conn)

    def _ensure_embedder(self):
        if self.embedder is None:
            self.embedder = CLIPEmbedder()

    def _load_embeddings(self):
        if self._embeddings is None:
            self._site_ids, self._embeddings = get_all_embeddings(self.conn)
        return self._site_ids, self._embeddings

    def query_by_text(self, text: str, top_k: int = config.DEFAULT_TOP_K) -> dict:
        """Query by natural language description (e.g., 'dark minimal SaaS')."""
        self._ensure_embedder()
        site_ids, embeddings = self._load_embeddings()
        query_vec = self.embedder.embed_text(text)
        results = top_k_similar(query_vec, embeddings, site_ids, k=top_k)
        return self._enrich_results(results, query_type="text", query=text)

    def query_by_image(self, image_path: str | Path,
                       top_k: int = config.DEFAULT_TOP_K) -> dict:
        """Query by an image file (screenshot)."""
        self._ensure_embedder()
        site_ids, embeddings = self._load_embeddings()
        query_vec = self.embedder.embed_image(image_path)
        results = top_k_similar(query_vec, embeddings, site_ids, k=top_k)
        return self._enrich_results(results, query_type="image", query=str(image_path))

    def query_by_url(self, url: str, top_k: int = config.DEFAULT_TOP_K) -> dict:
        """Query by a URL already in the database."""
        site = get_site_by_url(self.conn, url)
        if not site:
            return {"error": f"URL not found in database: {url}"}

        vec = get_embedding(self.conn, site["id"])
        if vec is None:
            return {"error": f"No embedding found for: {url}"}

        site_ids, embeddings = self._load_embeddings()
        results = top_k_similar(vec, embeddings, site_ids, k=top_k + 1)
        # Remove self from results
        results = [r for r in results if r["site_id"] != site["id"]][:top_k]
        return self._enrich_results(results, query_type="url", query=url)

    def _enrich_results(self, results: list[dict], query_type: str,
                        query: str) -> dict:
        """Add site metadata, cluster info, and style labels to results."""
        enriched = []
        cluster_votes = Counter()

        for r in results:
            site = self.conn.execute(
                "SELECT * FROM sites WHERE id=?", (r["site_id"],)
            ).fetchone()
            if site is None:
                continue

            entry = {
                "rank": r["rank"],
                "score": round(r["score"], 4),
                "url": site["url"],
                "domain": site["domain"],
                "category_hint": site["category_hint"],
                "screenshot_path": site["screenshot_path"],
            }

            # Add cluster info
            if self.run_id:
                cid = get_site_cluster(self.conn, self.run_id, r["site_id"])
                entry["cluster_id"] = cid
                if cid is not None and cid >= 0:
                    cluster_votes[cid] += 1
                    label = get_style_label(self.conn, self.run_id, cid)
                    if label:
                        entry["style_label"] = label["umbrella_label"]

            enriched.append(entry)

        # Dominant style from majority vote
        dominant_style = None
        if cluster_votes:
            dominant_cid = cluster_votes.most_common(1)[0][0]
            label = get_style_label(self.conn, self.run_id, dominant_cid) if self.run_id else None
            if label:
                dominant_style = {
                    "cluster_id": dominant_cid,
                    "umbrella_label": label["umbrella_label"],
                    "visual_density": label["visual_density"],
                    "color_mode": label["color_mode"],
                    "typography_style": label["typography_style"],
                    "layout_structure": label["layout_structure"],
                    "motion_intensity": label["motion_intensity"],
                    "visual_energy": label["visual_energy"],
                }

        return {
            "query_type": query_type,
            "query": query,
            "run_id": self.run_id,
            "dominant_style": dominant_style,
            "results": enriched,
        }
