"""
Catalog Lookup
--------------
Deterministic, non-fuzzy assessment name matching for the COMPARE state.

The vector retriever is semantic and can confuse similarly-named
assessments (e.g. "Verify Numerical" vs "Verify Numerical Reasoning").
This module loads the catalog directly and does exact (case/whitespace
-insensitive) matching first, only falling back to "similar name"
detection to flag ambiguity — never to silently substitute a guess.
"""

import difflib
import json
import re
from functools import lru_cache

CATALOG_PATH = "data/assessments.json"


@lru_cache(maxsize=1)
def _load_catalog() -> list[dict]:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _exact_match(name: str, catalog: list[dict]) -> dict | None:
    target = _normalize(name)
    for a in catalog:
        if _normalize(a.get("name", "")) == target:
            return a
    return None


def _similar_matches(name: str, catalog: list[dict], cutoff: float = 0.72) -> list[dict]:
    names = [a.get("name", "") for a in catalog]
    close = difflib.get_close_matches(name, names, n=5, cutoff=cutoff)
    close_set = {c.lower() for c in close}
    return [a for a in catalog if a.get("name", "").lower() in close_set]


def resolve_comparison_targets(candidate_names: list[str]) -> dict:
    """
    Resolve user-mentioned assessment names against the real catalog.

    Returns one of:
      {"status": "ok", "assessments": [dict, dict]}
          — every candidate had exactly one exact match. Safe to compare.
      {"status": "ambiguous", "name": str, "candidates": [dict, ...]}
          — a candidate had no exact match but multiple similar names
            exist in the catalog. Caller should ask the user to clarify,
            not guess.
      {"status": "not_found", "name": str}
          — a candidate had no exact match and no similar matches either.
            Caller must not fabricate an assessment for this.
    """
    catalog = _load_catalog()
    resolved = []

    for name in candidate_names:
        exact = _exact_match(name, catalog)
        if exact:
            resolved.append(exact)
            continue

        similar = _similar_matches(name, catalog)
        if len(similar) >= 2:
            return {"status": "ambiguous", "name": name, "candidates": similar}
        if len(similar) == 1:
            # Single close match, not exact — still don't auto-substitute.
            # Treat as ambiguous-of-one so the caller confirms rather than
            # silently comparing a different assessment than the user typed.
            return {"status": "ambiguous", "name": name, "candidates": similar}

        return {"status": "not_found", "name": name}

    # De-dupe in case the user named the same assessment twice
    seen_urls = set()
    deduped = []
    for a in resolved:
        url = a.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            deduped.append(a)

    return {"status": "ok", "assessments": deduped}
