"""Query interfaces for visual style and schema-based reference retrieval."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import json
from pathlib import Path
import re

import numpy as np

import config
from src.embed.clip_embedder import CLIPEmbedder
from src.retrieve.similarity import top_k_similar
from src.utils.storage import (
    get_connection,
    get_all_embeddings,
    get_site_cluster,
    get_style_label,
    get_latest_run_id,
    get_site_by_url,
    get_embedding,
)


class QueryEngine:
    """Unified query interface for text/image/url and reference records."""

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
        results = [r for r in results if r["site_id"] != site["id"]][:top_k]
        return self._enrich_results(results, query_type="url", query=url)

    def query_reference_records(
        self,
        query: str,
        top_k: int = 8,
        records_path: str | Path = "outputs/reference_records.v1.json",
        stack: str | None = None,
        constraints: str | None = None,
        risk_focus: str | None = None,
        min_review_status: str = "reviewed",
    ) -> dict:
        """Rank schema-v1 records via hybrid lexical+label scoring."""
        records_file = Path(records_path)
        if not records_file.exists():
            return {"error": f"Records file not found: {records_file}"}

        labels_file = Path("labels.v1.json")
        if not labels_file.exists():
            return {"error": "labels.v1.json not found at repo root."}

        records_payload = json.loads(records_file.read_text(encoding="utf-8"))
        records = records_payload.get("records", records_payload)
        if not isinstance(records, list):
            return {"error": "Invalid records payload; expected list or {'records': [...]}"}

        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        ranking_weights = labels.get("weights", {}).get("ranking", {})
        policies = labels.get("policies", {})

        q_tokens = self._tokenize(" ".join(filter(None, [query, stack, constraints, risk_focus])))
        desired_tests = self._desired_test_types(risk_focus)
        min_review_rank = self._review_rank(min_review_status)

        ranked = []
        for record in records:
            governance = record.get("governance", {})
            if self._review_rank(governance.get("review_status", "draft")) < min_review_rank:
                continue

            semantic = self._semantic_score(q_tokens, record)
            label_match = self._label_match_score(record, stack=stack, risk_focus=risk_focus)
            test_coverage = self._test_coverage_score(record, desired_tests)
            freshness = self._freshness_score(record)
            incident_proven = self._incident_proven_score(record)
            simplicity = self._simplicity_score(record)

            base = (
                ranking_weights.get("semantic", 0.35) * semantic
                + ranking_weights.get("label_match", 0.25) * label_match
                + ranking_weights.get("test_coverage", 0.15) * test_coverage
                + ranking_weights.get("freshness", 0.10) * freshness
                + ranking_weights.get("incident_proven", 0.10) * incident_proven
                + ranking_weights.get("simplicity", 0.05) * simplicity
            )

            confidence = record.get("labels", {}).get("confidence", "bronze")
            confidence_mul = labels.get("weights", {}).get("confidence_multiplier", {}).get(confidence, 0.70)
            final = base * confidence_mul

            ranked.append(
                {
                    "id": record.get("id"),
                    "title": record.get("title"),
                    "score": round(final, 4),
                    "confidence": confidence,
                    "corpus": record.get("corpus"),
                    "quality_axes": record.get("labels", {}).get("quality_axes", []),
                    "solution": record.get("solution", ""),
                    "invariants": record.get("contracts", {}).get("invariants", []),
                    "failure_modes": record.get("contracts", {}).get("failure_modes", []),
                    "tests": record.get("artifacts", {}).get("tests", []),
                    "anti_patterns": record.get("anti_patterns", []),
                    "bug_class": record.get("labels", {}).get("bug_class", []),
                    "source_ref": record.get("evidence", {}).get("source_ref", ""),
                }
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        max_per_bug_class = int(policies.get("max_results_per_bug_class", 2))
        selected = self._apply_diversity_cap(ranked, max_per_bug_class=max_per_bug_class, top_k=top_k)

        low_confidence = all(r.get("confidence") == "bronze" for r in selected) if selected else True
        return {
            "mode": "reference",
            "query": query,
            "stack": stack,
            "constraints": constraints,
            "risk_focus": risk_focus,
            "records_path": str(records_file),
            "low_confidence_context": low_confidence,
            "results": selected,
        }

    def _enrich_results(self, results: list[dict], query_type: str,
                        query: str) -> dict:
        """Add site metadata, cluster info, and style labels to visual results."""
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

            if self.run_id:
                cid = get_site_cluster(self.conn, self.run_id, r["site_id"])
                entry["cluster_id"] = cid
                if cid is not None and cid >= 0:
                    cluster_votes[cid] += 1
                    label = get_style_label(self.conn, self.run_id, cid)
                    if label:
                        entry["style_label"] = f"{label['page_type']} / {label['visual_style']}"
                        entry["quality_score"] = label["quality_score"]

            enriched.append(entry)

        dominant_style = None
        if cluster_votes:
            dominant_cid = cluster_votes.most_common(1)[0][0]
            label = get_style_label(self.conn, self.run_id, dominant_cid) if self.run_id else None
            if label:
                dominant_style = {
                    "cluster_id": dominant_cid,
                    "page_type": label["page_type"],
                    "visual_style": label["visual_style"],
                    "quality_score": label["quality_score"],
                    "industry": label["industry"],
                    "color_mode": label["color_mode"],
                    "layout_pattern": label["layout_pattern"],
                    "typography_style": label["typography_style"],
                }

        return {
            "query_type": query_type,
            "query": query,
            "run_id": self.run_id,
            "dominant_style": dominant_style,
            "results": enriched,
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9_+-]+", text.lower()) if len(t) > 1}

    def _semantic_score(self, query_tokens: set[str], record: dict) -> float:
        retrieval = record.get("retrieval", {})
        text = " ".join(
            [
                str(record.get("title", "")),
                str(record.get("summary", "")),
                str(retrieval.get("embedding_text", "")),
                " ".join(retrieval.get("keywords", [])),
            ]
        )
        r_tokens = self._tokenize(text)
        if not query_tokens or not r_tokens:
            return 0.0
        inter = len(query_tokens.intersection(r_tokens))
        union = len(query_tokens.union(r_tokens))
        return inter / union if union else 0.0

    def _label_match_score(self, record: dict, stack: str | None, risk_focus: str | None) -> float:
        labels = record.get("labels", {})
        frameworks = {f.lower() for f in labels.get("frameworks", [])}
        quality_axes = {q.lower() for q in labels.get("quality_axes", [])}
        stack_tokens = self._tokenize(stack or "")
        risk_tokens = self._tokenize(risk_focus or "")

        framework_overlap = len(stack_tokens.intersection(frameworks)) / max(1, len(stack_tokens))
        risk_overlap = len(risk_tokens.intersection(quality_axes)) / max(1, len(risk_tokens))

        if not stack_tokens and not risk_tokens:
            return 0.5
        return min(1.0, (0.6 * framework_overlap) + (0.4 * risk_overlap))

    def _desired_test_types(self, risk_focus: str | None) -> set[str]:
        focus = self._tokenize(risk_focus or "")
        desired = {"unit", "integration"}
        if "security" in focus:
            desired.add("security")
        if "performance" in focus:
            desired.add("load")
        if "a11y" in focus:
            desired.add("a11y")
        return desired

    def _test_coverage_score(self, record: dict, desired_tests: set[str]) -> float:
        tests = record.get("artifacts", {}).get("tests", [])
        if not tests:
            return 0.0
        present = {t.get("kind", "").lower() for t in tests if isinstance(t, dict)}
        if not desired_tests:
            return 0.5
        return len(present.intersection(desired_tests)) / len(desired_tests)

    def _freshness_score(self, record: dict) -> float:
        retrieval = record.get("retrieval", {})
        governance = record.get("governance", {})
        dates = [retrieval.get("freshness"), governance.get("last_verified_at")]
        parsed = [self._parse_date(d) for d in dates if d]
        parsed = [d for d in parsed if d]
        if not parsed:
            return 0.5

        newest = max(parsed)
        age_days = (date.today() - newest).days
        if age_days <= 30:
            return 1.0
        if age_days <= 90:
            return 0.8
        if age_days <= 180:
            return 0.6
        if age_days <= 365:
            return 0.4
        return 0.2

    def _incident_proven_score(self, record: dict) -> float:
        source_type = record.get("evidence", {}).get("source_type", "")
        if source_type in {"incident", "production"}:
            return 1.0
        if source_type in {"oss", "internal"}:
            return 0.5
        return 0.2

    def _simplicity_score(self, record: dict) -> float:
        code = record.get("artifacts", {}).get("code", [])
        if not code:
            return 0.9
        total = sum(len(c.get("snippet", "")) for c in code if isinstance(c, dict))
        return max(0.1, 1.0 - min(total / 20000.0, 0.9))

    @staticmethod
    def _review_rank(status: str) -> int:
        return {"draft": 0, "reviewed": 1, "approved": 2}.get(status, 0)

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None
        try:
            if "T" in value:
                return datetime.fromisoformat(value).date()
            return date.fromisoformat(value)
        except ValueError:
            return None

    def _apply_diversity_cap(self, ranked: list[dict], max_per_bug_class: int, top_k: int) -> list[dict]:
        selected = []
        bug_counts: Counter[str] = Counter()
        for item in ranked:
            bug_classes = item.get("bug_class") or ["_none"]
            if any(bug_counts[b] >= max_per_bug_class for b in bug_classes):
                continue
            selected.append(item)
            for bug in bug_classes:
                bug_counts[bug] += 1
            if len(selected) >= top_k:
                break
        return selected
