"""
SHL Catalog Diagnostic
----------------------
Run this BEFORE the scraper to see what's actually on the page.
Saves a screenshot + HTML dump so we can fix the real scraper.

Usage:
    python diagnose.py
"""

import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

URL = "https://www.shl.com/solutions/products/product-catalog/"
OUT = Path("output")
OUT.mkdir(exist_ok=True)


async def diagnose():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        print(f"[diag] Loading {URL} ...")
        await page.goto(URL, wait_until="networkidle", timeout=60_000)

        # Wait a bit extra for JS to settle
        await page.wait_for_timeout(5_000)

        # 1. Screenshot
        await page.screenshot(path=str(OUT / "screenshot.png"), full_page=True)
        print("[diag] Screenshot saved → output/screenshot.png")

        # 2. Full HTML
        html = await page.content()
        (OUT / "page.html").write_text(html, encoding="utf-8")
        print(f"[diag] HTML saved → output/page.html ({len(html):,} chars)")

        # 3. Check JavaScript environment
        js_check = await page.evaluate("""
        () => ({
            has_jquery:    typeof $ !== 'undefined',
            jquery_ver:    typeof $ !== 'undefined' ? ($.fn && $.fn.jquery) : null,
            has_datatables: typeof $.fn !== 'undefined' && typeof $.fn.dataTable !== 'undefined',
            tables_on_page: Array.from(document.querySelectorAll('table')).map(t => ({
                id:       t.id,
                classes:  t.className,
                rows:     t.querySelectorAll('tr').length,
                headers:  Array.from(t.querySelectorAll('th')).map(th => th.innerText.trim()),
            })),
            all_table_ids: Array.from(document.querySelectorAll('table')).map(t => t.id),
            url: window.location.href,
            title: document.title,
        })
        """)
        print("\n[diag] JavaScript environment:")
        print(json.dumps(js_check, indent=2))
        (OUT / "js_check.json").write_text(
            json.dumps(js_check, indent=2), encoding="utf-8"
        )

        # 4. Try all possible DataTable selectors
        dt_probe = await page.evaluate("""
        () => {
            const results = {};
            const tableIds = Array.from(document.querySelectorAll('table')).map(t => t.id).filter(Boolean);
            for (const id of tableIds) {
                try {
                    if (typeof $ !== 'undefined' && $.fn && $.fn.dataTable) {
                        const dt = $(`#${id}`).DataTable();
                        results[id] = { success: true, rows: dt.rows().count() };
                    } else {
                        results[id] = { success: false, reason: 'no DataTable' };
                    }
                } catch(e) {
                    results[id] = { success: false, reason: e.toString() };
                }
            }
            return results;
        }
        """)
        print("\n[diag] DataTable probe per table ID:")
        print(json.dumps(dt_probe, indent=2))

        # 5. Count visible rows in ANY table
        row_counts = await page.evaluate("""
        () => Array.from(document.querySelectorAll('table')).map(t => ({
            id: t.id || '(no id)',
            visible_rows: t.querySelectorAll('tbody tr').length,
            sample_cell: t.querySelector('tbody td') ? t.querySelector('tbody td').innerText.slice(0, 80) : null,
        }))
        """)
        print("\n[diag] Row counts per table:")
        print(json.dumps(row_counts, indent=2))

        # 6. Look for any assessment-like links
        links = await page.evaluate("""
        () => {
            const anchors = Array.from(document.querySelectorAll('a[href*="/solutions/products/"]'));
            return anchors.slice(0, 20).map(a => ({
                text: a.innerText.trim(),
                href: a.href,
            }));
        }
        """)
        print(f"\n[diag] Sample product links found: {len(links)}")
        for l in links[:10]:
            print(f"  {l['text']!r} → {l['href']}")

        await browser.close()
        print("\n[diag] Done. Check output/ folder for screenshot.png and page.html")
        print("[diag] Share the output above so we can fix the scraper.")


if __name__ == "__main__":
    asyncio.run(diagnose())