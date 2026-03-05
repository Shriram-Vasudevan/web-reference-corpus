#!/usr/bin/env python3
"""Step 1: Capture screenshots for all source website URLs."""

import argparse
import asyncio
import csv
import sys
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import get_connection, init_db, upsert_site, mark_captured, mark_failed, get_pending_sites, get_captured_sites
from src.capture.screenshotter import capture_batch

console = Console()


def load_website_urls() -> list[dict]:
    """Load source website URLs from CSV."""
    websites = []
    with open(config.WEBSITE_URLS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["url"].strip()
            if not url.startswith("http"):
                url = f"https://{url}"
            websites.append({
                "url": url,
                "domain": urlparse(url).netloc.replace("www.", ""),
                "category_hint": row.get("category_hint", "").strip() or None,
            })
    return websites


def main():
    parser = argparse.ArgumentParser(description="Capture website screenshots")
    parser.add_argument("--resume", action="store_true", help="Only capture pending sites")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of captures (0=all)")
    parser.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENT_BROWSERS)
    args = parser.parse_args()

    conn = get_connection()
    init_db(conn)

    # Load and register source URLs
    websites = load_website_urls()
    for website in websites:
        upsert_site(conn, website["url"], website["domain"], website["category_hint"])

    # Determine which sites to capture
    if args.resume:
        pending = get_pending_sites(conn)
        console.print(f"[yellow]Resume mode:[/] {len(pending)} pending sites")
    else:
        # Reset all to pending for fresh capture
        pending = get_pending_sites(conn)
        already = get_captured_sites(conn)
        console.print(f"[blue]{len(already)} already captured, {len(pending)} pending[/]")
        if not pending and already:
            console.print("[green]All sites already captured. Use --resume to retry failures.[/]")
            pending = []

    if args.limit > 0:
        pending = pending[:args.limit]

    if not pending:
        console.print("[green]Nothing to capture.[/]")
        conn.close()
        return

    # Build task list
    tasks = [
        {"url": row["url"], "site_id": row["id"], "category_hint": row["category_hint"]}
        for row in pending
    ]

    # Progress tracking
    results = {"success": 0, "failed": 0}
    progress = tqdm(total=len(tasks), desc="Capturing screenshots")

    def on_success(site_id, path):
        mark_captured(conn, site_id, path)
        results["success"] += 1
        progress.update(1)

    def on_failure(site_id, error):
        mark_failed(conn, site_id, error)
        results["failed"] += 1
        progress.update(1)

    console.print(f"\n[bold]Capturing {len(tasks)} sites[/] (concurrency={args.concurrency})\n")

    asyncio.run(
        capture_batch(
            tasks,
            max_concurrent=args.concurrency,
            on_success=on_success,
            on_failure=on_failure,
        )
    )

    progress.close()
    conn.close()

    console.print(f"\n[bold green]Done![/] {results['success']} captured, {results['failed']} failed")


if __name__ == "__main__":
    main()
