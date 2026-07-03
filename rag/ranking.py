"""
Ranking
-------
Weighted multi-factor scoring for SHL assessment recommendations.

Replaces pure keyword/semantic ordering with a combined score across
independent signals: technical match, role relevance, seniority match,
behavioral match, requested test type, adaptive/remote/language fit,
and semantic similarity from the vector retriever.

This module is pure scoring logic — it does not call any LLM or make
network requests, and does not touch the SHL catalog data itself beyond
reading fields already present on each assessment dict. Safe to unit test
in isolation.
"""

import re

# ── Weights — must sum to 1.0 ────────────────────────────────────────────
# Tune these if evaluation traces show a signal is over/under-weighted.
WEIGHTS = {
    "technical_skill":   0.20,
    "job_role":          0.15,
    "seniority":         0.10,
    "behavioral":        0.15,
    "test_type_match":   0.10,
    "adaptive":          0.05,
    "remote":            0.05,
    "language":          0.05,
    "semantic":          0.15,
}

_SENIORITY_LEVEL_MAP = {
    "entry":     ["entry level", "graduate", "entry"],
    "mid":       ["professional individual contributor", "mid", "experienced professional"],
    "senior":    ["senior", "manager", "director"],
    "executive": ["executive", "director", "vp", "c-suite"],
}

_ADAPTIVE_HINT = re.compile(r"\badaptive\b|\bIRT\b", re.IGNORECASE)
_REMOTE_HINT   = re.compile(r"\bremote\b|\bwork\s+from\s+home\b|\bvirtual\b", re.IGNORECASE)
_LANGUAGE_HINT = re.compile(
    r"\b(english|spanish|french|german|mandarin|chinese|japanese|"
    r"portuguese|italian|dutch|hindi|arabic|korean|russian)\b",
    re.IGNORECASE,
)


def _text_overlap_score(needle_terms: list[str], haystack: str) -> float:
    """Fraction of needle_terms found (case-insensitive) inside haystack.
    Returns 0.0 if needle_terms is empty."""
    if not needle_terms:
        return 0.0
    haystack_lower = haystack.lower()
    hits = sum(1 for term in needle_terms if term and term.lower() in haystack_lower)
    return hits / len(needle_terms)


def _get_semantic_score(assessment: dict) -> float:
    """Pull a 0-1 similarity score from whatever key the retriever attached.
    ChromaDB wrappers vary — check common key names, and convert distance
    (lower=better) into similarity (higher=better) if that's what's present.
    Falls back to a neutral 0.5 if no score is attached, so this signal
    never zeroes out an otherwise-strong candidate."""
    for key in ("score", "similarity", "relevance_score"):
        if key in assessment and isinstance(assessment[key], (int, float)):
            val = float(assessment[key])
            return max(0.0, min(1.0, val))
    for key in ("distance", "cosine_distance"):
        if key in assessment and isinstance(assessment[key], (int, float)):
            dist = float(assessment[key])
            # cosine distance is typically 0 (identical) to 2 (opposite)
            return max(0.0, min(1.0, 1.0 - (dist / 2.0)))
    return 0.5


def _seniority_score(assessment: dict, seniority: str | None) -> float:
    if not seniority:
        return 0.5  # neutral — no signal to judge by
    seniority = seniority.lower()
    target_terms = _SENIORITY_LEVEL_MAP.get(seniority, [])
    levels_raw = assessment.get("job_levels", [])
    levels = levels_raw if isinstance(levels_raw, list) else [str(levels_raw)]
    levels_text = " ".join(levels).lower()
    if not levels_text or "all levels" in levels_text:
        return 0.5
    return 1.0 if any(t in levels_text for t in target_terms) else 0.2


def _test_type_score(assessment: dict, requested_codes: list[str]) -> float:
    if not requested_codes:
        return 0.5
    code = str(assessment.get("test_type_code", "")).upper()
    return 1.0 if code in [c.upper() for c in requested_codes] else 0.0


def _behavioral_score(assessment: dict, behavioral_signal: bool) -> float:
    code = str(assessment.get("test_type_code", "")).upper()
    is_behavioral_type = code in ("P", "M")
    if behavioral_signal:
        return 1.0 if is_behavioral_type else 0.3
    # No behavioral signal in the request — don't penalize behavioral
    # assessments outright, just don't favor them either.
    return 0.5


def _adaptive_score(assessment: dict, text: str) -> float:
    if not _ADAPTIVE_HINT.search(text):
        return 0.5  # not requested — neutral
    return 1.0 if assessment.get("adaptive_irt") else 0.0


def _remote_score(assessment: dict, text: str) -> float:
    if not _REMOTE_HINT.search(text):
        return 0.5
    return 1.0 if assessment.get("remote_testing") else 0.0


def _language_score(assessment: dict, text: str) -> float:
    match = _LANGUAGE_HINT.search(text)
    if not match:
        return 0.5
    wanted = match.group(1).lower()
    langs_raw = assessment.get("languages", [])
    langs = langs_raw if isinstance(langs_raw, list) else [str(langs_raw)]
    langs_text = " ".join(langs).lower()
    return 1.0 if wanted in langs_text else 0.0


def score_assessment(
    assessment: dict,
    constraints: dict,
    last_user_text: str,
    behavioral_signal: bool = False,
) -> float:
    """Compute a single weighted 0-1 score for one assessment."""
    job_role = constraints.get("job_role") or ""
    seniority = constraints.get("seniority")
    requested_codes = constraints.get("test_type_codes") or []
    keywords = constraints.get("keywords") or []

    searchable = " ".join([
        str(assessment.get("name", "")),
        str(assessment.get("description", "")),
        " ".join(assessment.get("skills_measured", []) or []),
        " ".join(assessment.get("keywords", []) or []),
    ])

    recommend_for_text = " ".join(assessment.get("recommend_for", []) or [])

    technical_terms = keywords + ([job_role] if job_role else [])
    signals = {
        "technical_skill": _text_overlap_score(technical_terms, searchable),
        "job_role":        _text_overlap_score([job_role] if job_role else [], recommend_for_text + " " + searchable),
        "seniority":       _seniority_score(assessment, seniority),
        "behavioral":      _behavioral_score(assessment, behavioral_signal),
        "test_type_match": _test_type_score(assessment, requested_codes),
        "adaptive":        _adaptive_score(assessment, last_user_text),
        "remote":          _remote_score(assessment, last_user_text),
        "language":        _language_score(assessment, last_user_text),
        "semantic":        _get_semantic_score(assessment),
    }

    total = sum(signals[k] * WEIGHTS[k] for k in WEIGHTS)
    return total


def rank_pool(
    pool: list[dict],
    constraints: dict,
    last_user_text: str,
    behavioral_signal: bool = False,
    top_k: int = 10,
) -> list[dict]:
    """Score every assessment in the pool and return the top_k, sorted
    descending by score. Attaches the computed score under '_rank_score'
    for logging/debugging — this key is internal only and is stripped
    before anything reaches the API response."""
    scored = [
        (score_assessment(a, constraints, last_user_text, behavioral_signal), a)
        for a in pool
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    ranked = []
    for score, a in scored[:top_k]:
        item = dict(a)
        item["_rank_score"] = round(score, 4)
        ranked.append(item)
    return ranked
