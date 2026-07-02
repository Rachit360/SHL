"""
merge_catalogs.py
-----------------
Since the two datasets contain genuinely different assessment versions
(official has newer "(New)" versions, ours has older "Fundamentals" versions),
we use the official catalog as the PRIMARY source and our enriched dataset
as a SECONDARY source for semantic enrichment only.

Strategy:
  1. Official master_catalog.json (317) = ground truth
  2. Match by fuzzy stem name where possible
  3. For matched entries: overlay our enrichment fields
  4. For unmatched official entries: add with empty enrichment
  5. Do NOT include our-only entries (they're older/deprecated versions)
     UNLESS they have no official equivalent at all

Input:
  scraper/output/master_catalog.json
  data/assessments.json

Output:
  scraper/output/final_assessments.json
  data/assessments.json
"""

import json
import re
import sys
from pathlib import Path

MASTER_PATH   = Path(__file__).parent / "output" / "master_catalog.json"
ENRICHED_PATH = Path(__file__).parent.parent / "data" / "assessments.json"
OUTPUT_PATH   = Path(__file__).parent / "output" / "final_assessments.json"
DATA_OUTPUT   = Path(__file__).parent.parent / "data" / "assessments.json"


def normalize(name: str) -> str:
    """Strict normalization for exact matching."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def stem(name: str) -> str:
    """Aggressive normalization for fuzzy matching."""
    name = name.lower()
    # Remove version markers
    name = re.sub(r"\(new\)|\(new version\)", "", name)
    name = re.sub(r"\b(new|fundamentals|advanced|basic|premium)\b", "", name)
    name = re.sub(r"\b(short form|long form)\b", "", name)
    name = re.sub(r"\d+\.\d+|\bv\d+\b", "", name)
    name = re.sub(r"\bopq\s*-\s*", "opq ", name)
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def build_embedding_text(entry: dict) -> str:
    parts = [
        entry.get("name", ""),
        entry.get("test_type_label", ""),
        entry.get("category", ""),
        entry.get("description", "")[:300],
        " ".join(entry.get("skills_measured", [])),
        " ".join(entry.get("keywords", [])),
        " ".join(entry.get("recommend_for", [])),
        " ".join(entry.get("business_problems_solved", [])),
        " ".join(entry.get("job_levels", [])),
        f"Duration: {entry.get('duration', {}).get('display', '')}" if entry.get('duration', {}).get('display') else "",
        "Remote testing available" if entry.get("remote_testing") else "",
        "Adaptive IRT" if entry.get("adaptive_irt") else "",
    ]
    return " | ".join(p for p in parts if p)


def merge_entry(official: dict, enriched: dict | None) -> dict:
    merged = {
        "entity_id":       official.get("entity_id", ""),
        "name":            official.get("name", ""),
        "url":             official.get("url", ""),
        "description":     official.get("description", ""),
        "test_type_code":  official.get("test_type_code", "K"),
        "test_type_label": official.get("test_type_label", ""),
        "category":        official.get("category", ""),
        "keys":            official.get("keys", []),
        "job_levels":      official.get("job_levels", []),
        "languages":       official.get("languages", []),
        "duration":        official.get("duration", {"minutes": None, "display": ""}),
        "remote_testing":  official.get("remote_testing", True),
        "adaptive_irt":    official.get("adaptive_irt", False),
        # Enrichment fields
        "skills_measured":          enriched.get("skills_measured", []) if enriched else [],
        "keywords":                 enriched.get("keywords", []) if enriched else [],
        "recommend_for":            enriched.get("recommend_for", []) if enriched else [],
        "business_problems_solved": enriched.get("business_problems_solved", []) if enriched else [],
    }

    # Fill gaps from enriched if official is missing data
    if enriched:
        if not merged["description"] and enriched.get("description"):
            merged["description"] = enriched["description"]
        if not merged["job_levels"] and enriched.get("job_levels"):
            merged["job_levels"] = enriched["job_levels"]
        if not merged["languages"] and enriched.get("languages"):
            merged["languages"] = enriched["languages"]

    merged["embedding_text"] = build_embedding_text(merged)
    return merged


def main():
    if not MASTER_PATH.exists():
        print(f"[merge] ERROR: {MASTER_PATH} not found. Run filter_catalog.py first.")
        sys.exit(1)
    if not ENRICHED_PATH.exists():
        print(f"[merge] ERROR: {ENRICHED_PATH} not found.")
        sys.exit(1)

    with open(MASTER_PATH, encoding="utf-8") as f:
        master = json.load(f)
    with open(ENRICHED_PATH, encoding="utf-8") as f:
        enriched_raw = json.load(f)

    print(f"[merge] Official catalog : {len(master)} assessments")
    print(f"[merge] Enriched dataset : {len(enriched_raw)} assessments")

    # Build lookup maps from enriched dataset
    enriched_exact = {normalize(a["name"]): a for a in enriched_raw}
    enriched_stem  = {stem(a["name"]): a for a in enriched_raw}

    matched_exact  = 0
    matched_fuzzy  = 0
    unmatched      = 0
    merged_list    = []

    for official in master:
        norm = normalize(official["name"])
        st   = stem(official["name"])

        # Try exact match first
        enriched = enriched_exact.get(norm)
        if enriched:
            matched_exact += 1
        else:
            # Try stem/fuzzy match
            enriched = enriched_stem.get(st)
            if enriched:
                matched_fuzzy += 1
            else:
                unmatched += 1

        merged_list.append(merge_entry(official, enriched))

    # Stats
    type_dist   = {}
    has_dur     = sum(1 for a in merged_list if a.get("duration", {}).get("minutes"))
    has_skills  = sum(1 for a in merged_list if a.get("skills_measured"))
    has_rec     = sum(1 for a in merged_list if a.get("recommend_for"))
    has_biz     = sum(1 for a in merged_list if a.get("business_problems_solved"))

    for a in merged_list:
        c = a.get("test_type_code", "?")
        type_dist[c] = type_dist.get(c, 0) + 1

    print(f"\n[merge] Results:")
    print(f"  Exact matches        : {matched_exact}")
    print(f"  Fuzzy stem matches   : {matched_fuzzy}")
    print(f"  No match (official)  : {unmatched}")
    print(f"  Total final          : {len(merged_list)}")
    print(f"  With real URL        : {sum(1 for a in merged_list if a.get('url'))}")
    print(f"  With duration        : {has_dur}")
    print(f"  With skills_measured : {has_skills}")
    print(f"  With recommend_for   : {has_rec}")
    print(f"  With business_probs  : {has_biz}")
    print(f"  Type distribution    : {dict(sorted(type_dist.items()))}")

    print(f"\n[merge] Sample merged entries:")
    for a in merged_list[:3]:
        print(f"  {a['name']}")
        print(f"    url={a['url'][:60]}")
        print(f"    duration={a['duration']['display']} | type={a['test_type_code']}")
        print(f"    skills={a['skills_measured'][:3]}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=2, ensure_ascii=False)
    print(f"\n[merge] Saved → {OUTPUT_PATH}")

    with open(DATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=2, ensure_ascii=False)
    print(f"[merge] Saved → {DATA_OUTPUT}")


if __name__ == "__main__":
    main()
