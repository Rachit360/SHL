"""
Retriever
---------
Encodes a query with BGE model and fetches top-k candidates from ChromaDB.
"""

import re
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from rag.vector_store import VectorStore, get_vector_store

EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    print(f"[retriever] Loading model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


class Retriever:
    def __init__(self):
        self._model = get_model()
        self._store: VectorStore = get_vector_store()

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        top_k     = max(1, min(top_k, 10))
        prefixed  = BGE_QUERY_PREFIX + query.strip()
        embedding = self._model.encode(
            prefixed, normalize_embeddings=True
        ).tolist()
        candidates = self._store.query(embedding=embedding, n_results=top_k * 3)
        candidates = [c for c in candidates if c.get("url")]
        return candidates[:top_k]

    def retrieve_for_comparison(self, names: list[str]) -> list[dict]:
        exact  = self._store.get_by_names(names)
        found  = {e["name"].lower() for e in exact}
        fallback = []
        for name in names:
            if name.lower() not in found:
                results = self.retrieve(name, top_k=1)
                fallback.extend(results)
        return exact + fallback


def get_retriever() -> Retriever:
    if not hasattr(get_retriever, "_instance"):
        get_retriever._instance = Retriever()
    return get_retriever._instance
