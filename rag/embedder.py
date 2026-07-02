"""
Embedder
--------
Generates embeddings for all 234 assessments using BAAI/bge-small-en-v1.5
and stores them in ChromaDB (persisted to disk at data/chroma_db/).

Usage:
    python rag/embedder.py           # embed all
    python rag/embedder.py --reset   # wipe and re-embed
"""

import argparse
import json
import re
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DATA_DIR         = Path(__file__).parent.parent / "data"
ASSESSMENTS_PATH = DATA_DIR / "assessments.json"
CHROMA_DIR       = DATA_DIR / "chroma_db"
COLLECTION_NAME  = "shl_assessments"
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
BATCH_SIZE       = 64


def load_assessments() -> list[dict]:
    if not ASSESSMENTS_PATH.exists():
        print(f"[embedder] ERROR: {ASSESSMENTS_PATH} not found.")
        print("[embedder] Run: copy scraper/output/assessments.json data/assessments.json")
        sys.exit(1)
    with open(ASSESSMENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[embedder] Loaded {len(data)} assessments")
    return data


def make_id(name: str, index: int = 0) -> str:
    slug = re.sub(r"[^a-z0-9_-]", "_", name.lower())[:60]
    return f"{index:03d}_{slug}"


def build_embed_text(a: dict) -> str:
    """
    Use pre-built embedding_text from enriched dataset (clean_catalog.py).
    Falls back to constructing inline if field is missing.
    """
    if a.get("embedding_text"):
        return a["embedding_text"]
    parts = [
        a.get("name", ""),
        a.get("test_type_label", ""),
        a.get("description", ""),
        " ".join(a.get("skills_measured", [])),
        " ".join(a.get("keywords", [])),
        " ".join(a.get("recommend_for", [])),
        " ".join(a.get("business_problems_solved", [])),
        " ".join(a.get("job_levels", [])),
        "Remote testing available" if a.get("remote_testing") else "",
    ]
    return " | ".join(p for p in parts if p)


def build_metadata(a: dict) -> dict:
    """Flat metadata for ChromaDB (strings, ints, bools only — no lists)."""
    return {
        "name":             a.get("name", ""),
        "url":              a.get("url", ""),
        "test_type_code":   a.get("test_type_code", "K"),
        "test_type_label":  a.get("test_type_label", ""),
        "description":      a.get("description", "")[:500],  # ChromaDB metadata limit
        "job_levels":       ", ".join(a.get("job_levels", [])),
        "languages":        ", ".join(a.get("languages", [])[:5]),
        "remote_testing":   a.get("remote_testing", True),
        "adaptive_irt":     a.get("adaptive_irt", False),
    }


def main(reset: bool = False):
    assessments = load_assessments()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[embedder] Deleted existing collection")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Skip already-embedded
    existing_ids = set()
    if collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])
        print(f"[embedder] {len(existing_ids)} already embedded")

    to_embed = [(i, a) for i, a in enumerate(assessments) if make_id(a.get("name",""), i) not in existing_ids]
    if not to_embed:
        print(f"[embedder] All {collection.count()} assessments already embedded.")
        return

    print(f"[embedder] Loading model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"[embedder] Embedding {len(to_embed)} assessments ...")
    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i: i + BATCH_SIZE]
        texts = [build_embed_text(a) for _, a in batch]
        ids   = [make_id(a.get("name", f"item_{idx}"), idx) for idx, a in batch]
        metas = [build_metadata(a) for _, a in batch]

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
        end = min(i + BATCH_SIZE, len(to_embed))
        print(f"[embedder] Stored batch {i // BATCH_SIZE + 1} ({end}/{len(to_embed)})")

    print(f"[embedder] Done. Collection has {collection.count()} items.")
    print(f"[embedder] ChromaDB persisted at {CHROMA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    main(reset=args.reset)