"""
validate_dataset.py
--------------------
Validates final_assessments.json for completeness and correctness.
Produces a validation report and exits with error code if critical issues found.

Checks:
  - Duplicate names
  - Missing required fields (name, url, test_type_code)
  - Empty descriptions
  - Invalid/missing URLs
  - Missing job levels
  - Missing duration
  - Empty enrichment fields (skills, keywords, recommend_for)
  - Invalid test type codes

Usage:
    python scraper/validate_dataset.py
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

INPUT_PATH  = Path(__file__).parent / "output" / "final_assessments.json"
REPORT_PATH = Path(__file__).parent / "output" / "validation_report.txt"

VALID_TYPE_CODES = {"A", "B", "C", "D", "E", "K", "M", "P", "S"}


def validate(data: list[dict]) -> dict:
    issues = {
        "critical": [],   # Must fix — breaks the system
        "warnings": [],   # Should fix — degrades quality
        "info": [],       # FYI only
    }
    stats = {}

    # ── Duplicate names ──────────────────────────────────────────────────────
    names = [a.get("name", "").lower() for a in data]
    name_counts = Counter(names)
    dupes = {n: c for n, c in name_counts.items() if c > 1}
    if dupes:
        for name, count in dupes.items():
            issues["critical"].append(f"DUPLICATE: '{name}' appears {count} times")

    # ── Per-entry checks ─────────────────────────────────────────────────────
    missing_name        = []
    missing_url         = []
    invalid_url         = []
    missing_desc        = []
    short_desc          = []
    missing_type        = []
    invalid_type        = []
    missing_job_levels  = []
    missing_duration    = []
    missing_skills      = []
    missing_keywords    = []
    missing_rec_for     = []
    missing_biz         = []
    missing_embedding   = []

    for i, a in enumerate(data):
        name = a.get("name", "").strip()
        idx  = f"[{i}] {name or '(no name)'}"

        # Critical
        if not name:
            missing_name.append(idx)
        if not a.get("url"):
            missing_url.append(idx)
        elif not a["url"].startswith("https://www.shl.com"):
            invalid_url.append(f"{idx} → {a['url'][:60]}")

        code = a.get("test_type_code", "")
        if not code:
            missing_type.append(idx)
        elif code not in VALID_TYPE_CODES:
            invalid_type.append(f"{idx} → code='{code}'")

        # Warnings
        desc = a.get("description", "").strip()
        if not desc:
            missing_desc.append(idx)
        elif len(desc) < 30:
            short_desc.append(f"{idx} ({len(desc)} chars)")

        if not a.get("job_levels"):
            missing_job_levels.append(idx)

        dur = a.get("duration", {})
        if not dur.get("minutes") and not dur.get("display"):
            missing_duration.append(idx)

        if not a.get("embedding_text"):
            missing_embedding.append(idx)

        # Info (enrichment)
        if not a.get("skills_measured"):
            missing_skills.append(name)
        if not a.get("keywords"):
            missing_keywords.append(name)
        if not a.get("recommend_for"):
            missing_rec_for.append(name)
        if not a.get("business_problems_solved"):
            missing_biz.append(name)

    # Classify issues
    for lst, label, severity in [
        (missing_name,       "Missing name",           "critical"),
        (missing_url,        "Missing URL",            "critical"),
        (invalid_url,        "Non-SHL URL",            "critical"),
        (missing_type,       "Missing test_type_code", "critical"),
        (invalid_type,       "Invalid test_type_code", "critical"),
        (missing_desc,       "Missing description",    "warnings"),
        (short_desc,         "Description < 30 chars", "warnings"),
        (missing_job_levels, "Missing job_levels",     "warnings"),
        (missing_embedding,  "Missing embedding_text", "warnings"),
        (missing_duration,   "Missing duration",       "info"),
        (missing_skills,     "No skills_measured",     "info"),
        (missing_keywords,   "No keywords",            "info"),
        (missing_rec_for,    "No recommend_for",       "info"),
        (missing_biz,        "No business_problems",   "info"),
    ]:
        if lst:
            issues[severity].append(
                f"{label}: {len(lst)} items" +
                (f" — e.g. {lst[0]}" if lst else "")
            )

    # Stats
    type_dist = Counter(a.get("test_type_code", "?") for a in data)
    stats = {
        "total":             len(data),
        "with_url":          sum(1 for a in data if a.get("url")),
        "with_description":  sum(1 for a in data if a.get("description")),
        "with_duration":     sum(1 for a in data if a.get("duration", {}).get("minutes")),
        "with_job_levels":   sum(1 for a in data if a.get("job_levels")),
        "with_skills":       sum(1 for a in data if a.get("skills_measured")),
        "with_rec_for":      sum(1 for a in data if a.get("recommend_for")),
        "with_biz_problems": sum(1 for a in data if a.get("business_problems_solved")),
        "remote_testing":    sum(1 for a in data if a.get("remote_testing")),
        "adaptive_irt":      sum(1 for a in data if a.get("adaptive_irt")),
        "type_distribution": dict(sorted(type_dist.items())),
    }

    return {"issues": issues, "stats": stats}


def main():
    if not INPUT_PATH.exists():
        print(f"[validate] ERROR: {INPUT_PATH} not found. Run merge_catalogs.py first.")
        sys.exit(1)

    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"[validate] Validating {len(data)} assessments ...")
    result = validate(data)
    issues = result["issues"]
    stats  = result["stats"]

    # Build report
    lines = []
    lines.append("=" * 60)
    lines.append("SHL ASSESSMENT ADVISOR — VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append(f"\nTotal assessments: {stats['total']}")
    lines.append("\n── STATISTICS ──────────────────────────────────────────")
    for k, v in stats.items():
        if k != "type_distribution":
            pct = f" ({v/stats['total']*100:.0f}%)" if isinstance(v, int) and k != "total" else ""
            lines.append(f"  {k:<25}: {v}{pct}")
    lines.append(f"\n  type_distribution:")
    for code, count in stats["type_distribution"].items():
        lines.append(f"    {code}: {count}")

    lines.append("\n── CRITICAL ISSUES (must fix) ───────────────────────────")
    if issues["critical"]:
        for issue in issues["critical"]:
            lines.append(f"  ✗ {issue}")
    else:
        lines.append("  ✓ No critical issues")

    lines.append("\n── WARNINGS (should fix) ────────────────────────────────")
    if issues["warnings"]:
        for issue in issues["warnings"]:
            lines.append(f"  ⚠ {issue}")
    else:
        lines.append("  ✓ No warnings")

    lines.append("\n── INFO (enrichment gaps) ───────────────────────────────")
    if issues["info"]:
        for issue in issues["info"]:
            lines.append(f"  ℹ {issue}")
    else:
        lines.append("  ✓ All enrichment fields populated")

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)
    print(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n[validate] Report saved → {REPORT_PATH}")

    if issues["critical"]:
        print(f"\n[validate] ✗ FAILED — {len(issues['critical'])} critical issue(s). Fix before deploying.")
        sys.exit(1)
    else:
        print(f"\n[validate] ✓ PASSED — Dataset is ready for embedding.")
        sys.exit(0)


if __name__ == "__main__":
    main()
