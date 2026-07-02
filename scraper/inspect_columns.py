"""
Column Inspector
----------------
Shows ALL columns of the first 3 rows from the DataTable
and extracts filter dropdown labels to map numeric IDs → meaning.

Usage:
    python inspect_columns.py
"""

import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

URL = "https://online.shl.com/gb/en-us/products?orderby=none&page=1&producttypes=1"
OUT = Path("output")
OUT.mkdir(exist_ok=True)


async def inspect():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        print(f"[inspect] Loading {URL} ...")
        await page.goto(URL, wait_until="networkidle", timeout=60_000)
        await page.wait_for_timeout(3_000)

        # 1. Get ALL raw columns for first 5 rows
        raw_rows = await page.evaluate("""
        () => {
            const dt = $('#myTable').DataTable();
            return dt.rows().data().toArray().slice(0, 5);
        }
        """)

        print("\n=== RAW ROW DATA (first 5 rows, all columns) ===")
        for i, row in enumerate(raw_rows):
            print(f"\n--- Row {i} ---")
            if isinstance(row, list):
                for j, cell in enumerate(row):
                    print(f"  col[{j}]: {str(cell)[:200]}")
            elif isinstance(row, dict):
                for k, v in row.items():
                    print(f"  key[{k!r}]: {str(v)[:200]}")

        # 2. Extract ALL filter dropdown options to map numeric IDs
        filters = await page.evaluate("""
        () => {
            const result = {};
            // Get all select/filter elements on the page
            document.querySelectorAll('select').forEach(sel => {
                const label = sel.id || sel.name || sel.className;
                result[label] = Array.from(sel.options).map(o => ({
                    value: o.value,
                    text: o.text.trim()
                }));
            });

            // Also look for checkbox filters or list filters
            document.querySelectorAll('[data-filter], [data-type]').forEach(el => {
                result['data-attrs'] = result['data-attrs'] || [];
                result['data-attrs'].push({
                    tag: el.tagName,
                    dataFilter: el.dataset.filter,
                    dataType: el.dataset.type,
                    text: el.innerText.slice(0, 50)
                });
            });

            // Look for filter labels with numbers
            document.querySelectorAll('label, .filter-option, [class*="filter"]').forEach(el => {
                const text = el.innerText.trim();
                if (text) {
                    result['filter-labels'] = result['filter-labels'] || [];
                    result['filter-labels'].push(text.slice(0, 80));
                }
            });

            return result;
        }
        """)

        print("\n=== FILTER DROPDOWNS & OPTIONS ===")
        print(json.dumps(filters, indent=2)[:3000])

        # 3. Check if there are individual product page links anywhere
        links = await page.evaluate("""
        () => Array.from(document.querySelectorAll('a')).map(a => ({
            text: a.innerText.trim().slice(0, 60),
            href: a.href
        })).filter(l => l.href.includes('product') || l.href.includes('assessment'))
        .slice(0, 20)
        """)
        print("\n=== PRODUCT LINKS ON PAGE ===")
        for l in links:
            print(f"  {l['text']!r} → {l['href']}")

        # 4. Check DataTable column headers
        headers = await page.evaluate("""
        () => Array.from(document.querySelectorAll('#myTable th')).map(th => th.innerText.trim())
        """)
        print(f"\n=== TABLE HEADERS ===")
        for i, h in enumerate(headers):
            print(f"  col[{i}]: {h!r}")

        # 5. Save full page source for manual inspection
        html = await page.content()
        (OUT / "online_shl_page.html").write_text(html, encoding="utf-8")
        print(f"\n[inspect] Full page HTML saved → output/online_shl_page.html")

        await browser.close()
        print("\n[inspect] Done. Paste output above.")


if __name__ == "__main__":
    asyncio.run(inspect())