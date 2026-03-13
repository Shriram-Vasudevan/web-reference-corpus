#!/usr/bin/env python3
"""Step 0 (one-time): Import existing Superpowering-Agents catalog into the local DB.

The web-reference-corpus DB is empty but SA already has fully processed data:
  - reference_data/style_catalog.json   (98 clusters with descriptors)
  - reference_data/site_id_order.txt    (site_id → embedding row index)
  - reference_data/screenshots/         (PNG per domain)
  - index_data/domain_to_row.json       (domain → row_idx)

This script imports that data so that scripts 04-07 can all run normally.
After running this, the DB will contain sites, clusters, and style_labels
for the run_id derived from the catalog, and outputs/ will have the files
that 05/06/07 expect.

Usage:
    python3 scripts/00_import_from_sa.py --sa-dir /path/to/Superpowering-Agents
    python3 scripts/00_import_from_sa.py  # auto-discovers SA dir next to this repo
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import track

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import get_connection, init_db, transaction

console = Console()

RUN_ID = "imported_from_sa"


def _find_sa_dir(hint: str | None) -> Path:
    if hint:
        return Path(hint).resolve()
    # Try sibling directory
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "Superpowering-Agents",
    ]
    for c in candidates:
        if c.exists() and (c / "reference_data" / "style_catalog.json").exists():
            return c
    console.print("[red]Could not auto-discover Superpowering-Agents directory. Use --sa-dir.[/]")
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Import SA catalog into local DB")
    parser.add_argument("--sa-dir", type=str, default=None, help="Path to Superpowering-Agents repo")
    parser.add_argument("--force", action="store_true", help="Re-import even if run already exists")
    args = parser.parse_args()

    sa_dir = _find_sa_dir(args.sa_dir)
    console.print(f"[bold]SA directory:[/] {sa_dir}")

    catalog_path = sa_dir / "reference_data" / "style_catalog.json"
    domain_to_row_path = sa_dir / "index_data" / "domain_to_row.json"
    screenshots_dir = sa_dir / "reference_data" / "screenshots"
    sa_site_id_order = sa_dir / "reference_data" / "site_id_order.txt"

    for p in [catalog_path, domain_to_row_path, screenshots_dir]:
        if not p.exists():
            console.print(f"[red]Missing: {p}[/]")
            raise SystemExit(1)

    with open(catalog_path) as f:
        catalog = json.load(f)
    with open(domain_to_row_path) as f:
        domain_to_row: dict[str, int] = json.load(f)

    styles = catalog.get("styles", [])
    console.print(f"[bold]Catalog:[/] {len(styles)} clusters, {sum(len(s['members']) for s in styles)} members")

    conn = get_connection()
    init_db(conn)

    # Check if already imported
    existing = conn.execute(
        "SELECT COUNT(*) FROM clusters WHERE run_id=?", (RUN_ID,)
    ).fetchone()[0]
    if existing and not args.force:
        console.print(f"[yellow]Run '{RUN_ID}' already exists ({existing} cluster assignments). Use --force to re-import.[/]")
        conn.close()
        return

    if existing and args.force:
        with transaction(conn):
            conn.execute("DELETE FROM clusters WHERE run_id=?", (RUN_ID,))
            conn.execute("DELETE FROM style_labels WHERE run_id=?", (RUN_ID,))
            conn.execute("DELETE FROM sites")

    # ── Insert sites ──────────────────────────────────────────────────────────
    console.print("\n[bold]Inserting sites...[/]")
    all_domains: list[tuple[str, str, str, str]] = []  # (url, domain, category_hint, screenshot_path)

    for style in styles:
        for member in style["members"]:
            domain = member["domain"]
            url = member.get("url") or f"https://{domain}"
            category_hint = member.get("category_hint") or ""
            ss_filename = domain.replace(".", "_") + ".png"
            ss_path = str(screenshots_dir / ss_filename)
            all_domains.append((url, domain, category_hint, ss_path))

    with transaction(conn):
        for url, domain, category_hint, ss_path in all_domains:
            conn.execute(
                """INSERT OR IGNORE INTO sites (url, domain, category_hint, screenshot_path, status, captured_at)
                   VALUES (?, ?, ?, ?, 'captured', CURRENT_TIMESTAMP)""",
                (url, domain, category_hint, ss_path if Path(ss_path).exists() else None),
            )

    total_sites = conn.execute("SELECT COUNT(*) FROM sites").fetchone()[0]
    console.print(f"  Inserted {total_sites} sites")

    # Build domain → site_id lookup
    domain_to_site_id: dict[str, int] = {
        r["domain"]: r["id"]
        for r in conn.execute("SELECT id, domain FROM sites").fetchall()
    }

    # ── Insert clusters ───────────────────────────────────────────────────────
    console.print("[bold]Inserting cluster assignments...[/]")
    cluster_rows: list[tuple] = []

    for style in styles:
        cluster_id = style["cluster_id"]
        for member in style["members"]:
            domain = member["domain"]
            site_id = domain_to_site_id.get(domain)
            if site_id is not None:
                cluster_rows.append((site_id, RUN_ID, cluster_id, 1.0))

    with transaction(conn):
        conn.executemany(
            "INSERT OR REPLACE INTO clusters (site_id, run_id, cluster_id, probability) VALUES (?,?,?,?)",
            cluster_rows,
        )
    console.print(f"  Inserted {len(cluster_rows)} cluster assignments")

    # ── Insert style labels ───────────────────────────────────────────────────
    console.print("[bold]Inserting style labels...[/]")
    label_rows = 0

    for style in track(styles, description="  Labels..."):
        cluster_id = style["cluster_id"]
        descriptor = style.get("descriptor", {})
        if not descriptor:
            continue

        conn.execute(
            """INSERT OR REPLACE INTO style_labels
               (cluster_id, run_id, page_type, visual_style, quality_score,
                industry, color_mode, layout_pattern, typography_style,
                design_era, target_audience, distinguishing_features, raw_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cluster_id, RUN_ID,
                descriptor.get("page_type", ""),
                descriptor.get("visual_style", ""),
                descriptor.get("quality_score", 0),
                descriptor.get("industry", ""),
                descriptor.get("color_mode", ""),
                descriptor.get("layout_pattern", ""),
                descriptor.get("typography_style", ""),
                descriptor.get("design_era", ""),
                descriptor.get("target_audience", ""),
                descriptor.get("distinguishing_features", "")
                if isinstance(descriptor.get("distinguishing_features"), str)
                else ", ".join(descriptor.get("distinguishing_features") or []),
                "",  # no raw_response for imported labels
            ),
        )
        label_rows += 1

    conn.commit()
    console.print(f"  Inserted {label_rows} style labels")

    # ── Build site_id_order.txt ───────────────────────────────────────────────
    # Maps site_id → embedding row index so 07_export_index can resolve row_idx.
    console.print("\n[bold]Writing outputs/site_id_order.txt...[/]")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    order_path = config.OUTPUTS_DIR / "site_id_order.txt"

    # We need a complete site_id_order.txt that covers every site_id in the DB.
    # Build it from domain_to_row (domain → row_idx) and domain_to_site_id.
    # site_id_order.txt format: one site_id per line, line N is row N.
    max_row = max(domain_to_row.values()) if domain_to_row else 0
    row_to_site_id: list[int | None] = [None] * (max_row + 1)
    for domain, row_idx in domain_to_row.items():
        site_id = domain_to_site_id.get(domain)
        if site_id is not None and row_idx <= max_row:
            row_to_site_id[row_idx] = site_id

    # Fill any gaps with 0 (sentinel) — gaps are embedding rows with no matching domain
    with open(order_path, "w") as f:
        for sid in row_to_site_id:
            f.write(f"{sid if sid is not None else 0}\n")
    console.print(f"  Written {len(row_to_site_id)} rows -> {order_path}")

    # ── Copy style_catalog.json to outputs/ ────────────────────────────────────
    dest_catalog = config.OUTPUTS_DIR / "style_catalog.json"
    shutil.copy2(catalog_path, dest_catalog)
    console.print(f"[bold]Copied style_catalog.json -> {dest_catalog}[/]")

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(f"\n[bold green]Import complete![/]")
    console.print(f"  run_id: {RUN_ID}")
    console.print(f"  Sites:    {total_sites}")
    console.print(f"  Clusters: {len(cluster_rows)} assignments across {len(styles)} clusters")
    console.print(f"  Labels:   {label_rows}")
    console.print(f"\n[bold]Next steps:[/]")
    console.print("  python3 scripts/05_reclassify_industry.py")
    console.print("  python3 scripts/06_build_industry_profiles.py")
    console.print("  python3 scripts/07_export_index.py --output-dir /path/to/SA/index_data")

    conn.close()


if __name__ == "__main__":
    main()
