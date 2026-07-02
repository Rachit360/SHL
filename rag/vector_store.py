"""
Vector Store
------------
Singleton wrapper around the persisted ChromaDB collection.
Loaded once at API startup — no re-embedding on every request.
"""

import json
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings

DATA_DIR         = Path(__file__).parent.parent / "data"
CHROMA_DIR       = DATA_DIR / "chroma_db"
ASSESSMENTS_PATH = DATA_DIR / "assessments.json"
COLLECTION_NAME  = "shl_assessments"


class VectorStore:
    def __init__(self):
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"ChromaDB not found at {CHROMA_DIR}. "
                "Run: python rag/embedder.py"
            )
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_collection(COLLECTION_NAME)
        self._assessment_map = self._load_assessment_map()
        print(f"[vector_store] Loaded {self._collection.count()} assessments")

    def _load_assessment_map(self) -> dict[str, dict]:
        if not ASSESSMENTS_PATH.exists():
            return {}
        with open(ASSESSMENTS_PATH, "r", encoding="utf-8") as f:
            assessments = json.load(f)
        return {a["name"].lower(): a for a in assessments}

    def query(self, embedding: list[float], n_results: int = 20) -> list[dict]:
        n = min(n_results, self._collection.count())
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["metadatas", "distances", "documents"],
        )

        enriched = []
        for i, meta in enumerate(results["metadatas"][0]):
            distance = results["distances"][0][i]
            score    = round(1.0 - distance, 4)
            full     = self._assessment_map.get(meta["name"].lower(), {})

            # duration: stored as dict in JSON
            duration = full.get("duration", {})
            if not isinstance(duration, dict):
                duration = {"minutes": None, "display": str(duration)}

            # job_levels: stored as list in JSON
            job_levels = full.get("job_levels", [])
            if not isinstance(job_levels, list):
                job_levels = [job_levels] if job_levels else []

            enriched.append({
                "name":                    meta.get("name", ""),
                "url":                     meta.get("url", ""),
                "test_type_code":          meta.get("test_type_code", "K"),
                "test_type_label":         meta.get("test_type_label", ""),
                "category":                meta.get("category", ""),
                "description":             full.get("description", meta.get("description", "")),
                "job_levels":              job_levels,
                "duration":                duration,
                "remote_testing":          full.get("remote_testing", True),
                "adaptive_irt":            full.get("adaptive_irt", False),
                "score":                   score,
                "skills_measured":         full.get("skills_measured", []),
                "keywords":                full.get("keywords", []),
                "recommend_for":           full.get("recommend_for", []),
                "business_problems_solved":full.get("business_problems_solved", []),
                "languages":               full.get("languages", []),
            })

        return enriched

    def get_by_names(self, names: list[str]) -> list[dict]:
        results = []
        for name in names:
            entry = self._assessment_map.get(name.lower())
            if entry:
                duration = entry.get("duration", {})
                if not isinstance(duration, dict):
                    duration = {"minutes": None, "display": str(duration)}
                job_levels = entry.get("job_levels", [])
                if not isinstance(job_levels, list):
                    job_levels = [job_levels] if job_levels else []
                results.append({
                    "name":           entry.get("name", ""),
                    "url":            entry.get("url", ""),
                    "test_type_code": entry.get("test_type_code", "K"),
                    "test_type_label":entry.get("test_type_label", ""),
                    "description":    entry.get("description", ""),
                    "job_levels":     job_levels,
                    "duration":       duration,
                    "remote_testing": entry.get("remote_testing", True),
                    "adaptive_irt":   entry.get("adaptive_irt", False),
                    "score":          1.0,
                    "skills_measured":         entry.get("skills_measured", []),
                    "keywords":                entry.get("keywords", []),
                    "recommend_for":           entry.get("recommend_for", []),
                    "business_problems_solved":entry.get("business_problems_solved", []),
                })
        return results

    def count(self) -> int:
        return self._collection.count()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()
