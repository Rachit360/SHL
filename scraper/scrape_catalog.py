"""
SHL Catalog Scraper v2
-----------------------
Extracts all 234 Individual Test Solution assessments from the SHL online portal
via DataTable JS extraction. Captures name, description, languages, and filter IDs.
Also discovers the product URL pattern by clicking the first row.

Usage:
    python scrape_catalog.py
Output:
    output/raw_assessments.json
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

CATALOG_URL = "https://online.shl.com/gb/en-us/products?orderby=none&page=1&producttypes=1"
OUTPUT_PATH = Path(__file__).parent / "output" / "raw_assessments.json"

EXTRACT_JS = """
() => {
    try {
        const table = $('#myTable').DataTable();
        const rows = table.rows().data().toArray();
        return { success: true, data: rows, count: rows.length };
    } catch (e) {
        return { success: false, error: e.toString(), data: [] };
    }
}
"""


def strip_html(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(separator=" ").strip()


def extract_href(html_fragment: str) -> str:
    if not html_fragment:
        return ""
    soup = BeautifulSoup(html_fragment, "html.parser")
    anchor = soup.find("a")
    if not anchor:
        return ""
    href = anchor.get("href", "")
    if href.startswith("/"):
        return f"https://online.shl.com{href}"
    return href


def parse_languages(raw: str) -> list[str]:
    """Parse 'en-US,en-US,fr-FR,fr-FR' → ['en-US', 'fr-FR'] (deduped)."""
    if not raw or raw.strip().upper() == "NULL":
        return []
    seen = set()
    result = []
    for lang in raw.split(","):
        lang = lang.strip()
        if lang and lang not in seen:
            seen.add(lang)
            result.append(lang)
    return result


def parse_filter_ids(raw: str) -> list[int]:
    """Parse '1,3,4,7' → [1, 3, 4, 7]. Returns [] for NULL."""
    if not raw or str(raw).strip().upper() == "NULL":
        return []
    try:
        return [int(x.strip()) for x in str(raw).split(",") if x.strip().isdigit()]
    except Exception:
        return []


def parse_row(row: list) -> dict:
    """
    Actual column structure (confirmed by inspect):
        col[0]: Product type icon (img) — skip
        col[1]: Assessment name (<b>Name</b>)
        col[2]: Description (plain text, may be truncated)
        col[3]: Languages (comma-separated locale codes)
        col[4]: Job level filter IDs
        col[5]: Filter IDs set A (test type or industry)
        col[6]: Filter IDs set B
        col[7]: Product type ID (always "1")
        col[8]: Filter IDs set C (adaptive/IRT related?)
    """
    if not isinstance(row, list) or len(row) < 2:
        return {}

    raw_name_html = str(row[1]) if len(row) > 1 else ""
    raw_desc      = str(row[2]) if len(row) > 2 else ""
    raw_langs     = str(row[3]) if len(row) > 3 else ""
    raw_col4      = str(row[4]) if len(row) > 4 else "NULL"
    raw_col5      = str(row[5]) if len(row) > 5 else "NULL"
    raw_col6      = str(row[6]) if len(row) > 6 else "NULL"
    raw_col7      = str(row[7]) if len(row) > 7 else "NULL"
    raw_col8      = str(row[8]) if len(row) > 8 else "NULL"

    name = strip_html(raw_name_html)
    # Description may already be plain text
    description = raw_desc.strip() if raw_desc.strip().upper() != "NULL" else ""

    return {
        "name":        name,
        "url":         "",          # populated later by URL discovery
        "description": description,
        "languages":   parse_languages(raw_langs),
        "job_level_ids": parse_filter_ids(raw_col4),
        "filter_ids_a":  parse_filter_ids(raw_col5),
        "filter_ids_b":  parse_filter_ids(raw_col6),
        "product_type":  raw_col7.strip(),
        "filter_ids_c":  parse_filter_ids(raw_col8),
        # Raw HTML for debugging
        "_raw_name_html": raw_name_html,
    }


async def discover_product_url(page) -> str:
    """
    Click the first table row to see if it navigates to a product detail page.
    Returns the URL pattern base if found, empty string otherwise.
    """
    try:
        # Try clicking the first row name/link
        first_row = await page.query_selector("#myTable tbody tr:first-child td:nth-child(2)")
        if not first_row:
            return ""

        async with page.expect_navigation(timeout=5_000, wait_until="domcontentloaded") as nav:
            await first_row.click()

        resp = await nav.value
        url = page.url
        print(f"[scraper] Row click navigated to: {url}")

        # Navigate back
        await page.go_back(wait_until="networkidle", timeout=15_000)
        await page.wait_for_timeout(2_000)

        return url
    except Exception as e:
        print(f"[scraper] Row click did not navigate (rows may not be clickable): {e}")
        return ""


async def try_get_product_urls(page, names: list[str]) -> dict[str, str]:
    """
    Try to find individual product page URLs by checking if rows have hrefs
    or if clicking opens a modal/new page.
    Returns dict of name → url (empty if not found).
    """
    urls = {}

    # Method 1: Check if any anchors exist inside table rows
    anchors = await page.evaluate("""
    () => {
        const rows = document.querySelectorAll('#myTable tbody tr');
        const results = [];
        rows.forEach(row => {
            const a = row.querySelector('a');
            const name = row.querySelector('b');
            results.push({
                name: name ? name.innerText.trim() : '',
                href: a ? a.href : ''
            });
        });
        return results.slice(0, 5);
    }
    """)
    print(f"[scraper] Anchor check in table rows: {anchors[:3]}")

    # Method 2: Check for data attributes on rows that might encode URLs
    data_attrs = await page.evaluate("""
    () => {
        const rows = document.querySelectorAll('#myTable tbody tr');
        const results = [];
        rows.forEach(row => {
            const attrs = {};
            for (const attr of row.attributes) {
                attrs[attr.name] = attr.value;
            }
            results.push(attrs);
        });
        return results.slice(0, 3);
    }
    """)
    print(f"[scraper] Row data attributes: {data_attrs[:2]}")

    return urls


async def scrape() -> list[dict]:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        print(f"[scraper] Loading {CATALOG_URL} ...")
        await page.goto(CATALOG_URL, wait_until="networkidle", timeout=60_000)

        print("[scraper] Waiting for DataTable ...")
        try:
            await page.wait_for_function(
                "() => typeof $ !== 'undefined' && $('#myTable').length > 0 && "
                "$.fn.dataTable && $.fn.dataTable.isDataTable('#myTable')",
                timeout=30_000,
                polling=500,
            )
        except PlaywrightTimeout:
            print("[scraper] DataTable wait timed out — attempting extraction anyway")

        await page.wait_for_timeout(2_000)

        # Check URL pattern via row click
        print("[scraper] Probing for product URLs ...")
        await try_get_product_urls(page, [])

        print("[scraper] Executing DataTable extraction ...")
        result = await page.evaluate(EXTRACT_JS)

        if not result.get("success"):
            print(f"[scraper] JS extraction failed: {result.get('error')}")
            sys.exit(1)

        rows_raw = result.get("data", [])
        print(f"[scraper] DataTable returned {result.get('count', 0)} rows")

        await browser.close()

    assessments = []
    skipped = 0
    for i, row in enumerate(rows_raw):
        parsed = parse_row(row)
        if not parsed.get("name"):
            skipped += 1
            continue
        assessments.append(parsed)

    print(f"[scraper] Parsed {len(assessments)} assessments ({skipped} skipped)")
    print(f"[scraper] Sample entry 0: name={assessments[0]['name']!r}")
    print(f"[scraper] Sample entry 0: desc={assessments[0]['description'][:80]!r}")
    print(f"[scraper] Sample entry 0: job_levels={assessments[0]['job_level_ids']}")
    print(f"[scraper] Sample entry 0: filter_a={assessments[0]['filter_ids_a']}")
    print(f"[scraper] Sample entry 0: filter_b={assessments[0]['filter_ids_b']}")
    print(f"[scraper] Sample entry 0: filter_c={assessments[0]['filter_ids_c']}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    print(f"[scraper] Saved → {OUTPUT_PATH}")
    return assessments


if __name__ == "__main__":
    asyncio.run(scrape())
