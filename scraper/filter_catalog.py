"""
filter_catalog.py
-----------------
Filters the official SHL product catalog (377 items) to extract
only valid Individual Test Solutions (actual assessments).

Removes:
  - Reports (candidate, manager, leadership, development, team, profile, narrative)
  - Profile cards
  - 360 group/team reports
  - Interpretation reports
  - Any product whose name contains report/profile/cards indicator words

Input:  shl_product_catalog.json (377 items)
Output: scraper/output/master_catalog.json

Usage:
    python scraper/filter_catalog.py
"""

import json
import re
import sys
from pathlib import Path

INPUT_PATH  = Path(__file__).parent / "shl_product_catalog.json"
OUTPUT_PATH = Path(__file__).parent / "output" / "master_catalog.json"

# ── Words that definitively mark a product as a report/artifact ─────────────
REPORT_EXACT_WORDS = [
    "report", "profile", "cards", "planner", "narrative",
    "pack", "tips", "interpretation",
]

# ── Name patterns that indicate a report even without above words ────────────
REPORT_NAME_PATTERNS = [
    r"\breport\b",
    r"\bprofile\b",
    r"\bcandidate report\b",
    r"\bmanager report\b",
    r"\bleadership report\b",
    r"\bdevelopment report\b",
    r"\bteam report\b",
    r"\bnarrative\b",
    r"\baction planner\b",
    r"\binterpretation\b",
    r"\bprofile cards?\b",
    r"\bprofiler cards?\b",
    r"\b360 .* report\b",
    r"\bgroup report\b",
    r"\bunlocking potential\b",
]

# ── Names that LOOK like reports but ARE valid assessments ───────────────────
# These are edge cases we explicitly keep
FORCE_KEEP = {
    "sql server reporting services (ssrs) (new)",  # K test, not a report
    "assessment and development center exercises",  # Exercise, not a report
    "virtual assessment and development centers",   # Simulation
    "global skills assessment",                     # Core assessment
}

# ── Names that LOOK like assessments but are NOT ─────────────────────────────
FORCE_REMOVE = {
    "sales profiler cards",
    "universal competency framework profiler cards (44)",
    "mq profile",                    # personality profile report
    "opq profile report",
    "opq user report",
    "opq user and managers report",
    "pjm selection report",
    "pjm development report",
    "remoteworkq manager report",
    "remoteworkq participant report",
    "verify g+ - ability test report",
    "verify g+ - candidate report",
    "verify interactive ability report",
    "verify interactive g+ candidate report",
    "verify interactive g+ report",
    "360 digital report",
    "hipo assessment report 1.0",
    "hipo assessment report 2.0",
    "hipo unlocking potential report 2.0",
}


def load_catalog() -> list[dict]:
    if not INPUT_PATH.exists():
        print(f"[filter] ERROR: {INPUT_PATH} not found.")
        print("[filter] Place shl_product_catalog.json in scraper/ folder.")
        sys.exit(1)

    content = INPUT_PATH.read_text(encoding="utf-8")
    # Fix malformed JSON (literal newlines inside string values)
    content = re.sub(r'(?<=: ")([^"]*)\n([^"]*?)(?=")', r'\1 \2', content)
    data = json.loads(content)
    print(f"[filter] Loaded {len(data)} products from catalog")
    return data


def is_report(item: dict) -> bool:
    """Return True if this item is a report/artifact, not an assessment."""
    name       = item.get("name", "")
    name_lower = name.lower().strip()

    # Force keep override
    if name_lower in FORCE_KEEP:
        return False

    # Force remove override
    if name_lower in FORCE_REMOVE:
        return True

    # Check against report name patterns
    for pattern in REPORT_NAME_PATTERNS:
        if re.search(pattern, name_lower):
            return True

    return False


def normalize_keys(keys: list[str]) -> tuple[str, str]:
    """
    Convert keys list to primary test_type_code and label.
    Takes the first/most specific key if multiple exist.
    """
    KEY_MAP = {
        "Knowledge & Skills":            ("K", "Knowledge & Skills"),
        "Personality & Behavior":        ("P", "Personality & Behavior"),
        "Simulations":                   ("S", "Simulations"),
        "Ability & Aptitude":            ("A", "Ability & Aptitude"),
        "Competencies":                  ("C", "Competencies"),
        "Biodata & Situational Judgment":("B", "Biodata & Situational Judgement"),
        "Development & 360":             ("D", "Development & 360"),
        "Assessment Exercises":          ("E", "Assessment Exercises"),
        "Motivation":                    ("M", "Motivation"),
    }
    for key in keys:
        if key in KEY_MAP:
            return KEY_MAP[key]
    return ("K", "Knowledge & Skills")


def parse_duration(duration_str: str) -> dict:
    """Parse '30 minutes' → {"minutes": 30, "display": "30 minutes"}"""
    if not duration_str or duration_str.strip() == "":
        return {"minutes": None, "display": ""}
    match = re.search(r"(\d+)", duration_str)
    minutes = int(match.group(1)) if match else None
    return {"minutes": minutes, "display": duration_str.strip()}


def clean_assessment(item: dict) -> dict:
    """Convert official catalog entry into clean assessment record."""
    keys = item.get("keys", [])
    code, label = normalize_keys(keys)

    return {
        "entity_id":       item.get("entity_id", ""),
        "name":            item.get("name", "").strip(),
        "url":             item.get("link", "").strip(),
        "description":     item.get("description", "").strip(),
        "test_type_code":  code,
        "test_type_label": label,
        "category":        label,
        "keys":            keys,
        "job_levels":      item.get("job_levels", []),
        "languages":       item.get("languages", []),
        "duration":        parse_duration(item.get("duration", "")),
        "remote_testing":  item.get("remote", "").lower() == "yes",
        "adaptive_irt":    item.get("adaptive", "").lower() == "yes",
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = load_catalog()

    assessments = []
    removed     = []

    for item in data:
        if is_report(item):
            removed.append(item["name"])
        else:
            assessments.append(clean_assessment(item))

    # Deduplicate by name
    seen = set()
    unique = []
    for a in assessments:
        key = a["name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)

    # Stats
    type_dist = {}
    for a in unique:
        c = a["test_type_code"]
        type_dist[c] = type_dist.get(c, 0) + 1

    print(f"\n[filter] Results:")
    print(f"  Total input        : {len(data)}")
    print(f"  Valid assessments  : {len(unique)}")
    print(f"  Removed (reports)  : {len(removed)}")
    print(f"  Type distribution  : {dict(sorted(type_dist.items()))}")
    print(f"\n[filter] Removed items:")
    for r in removed:
        print(f"  - {r}")

    print(f"\n[filter] Sample assessments:")
    for a in unique[:5]:
        print(f"  {a['name']} | {a['test_type_code']} | {a['duration']['display']} | remote={a['remote_testing']}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    print(f"\n[filter] Saved {len(unique)} assessments → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
