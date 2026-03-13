#!/usr/bin/env python3
"""Step 7: Export catalog_index.json and domain_to_row.json for Superpowering-Agents.

Flattens style_catalog.json clusters into per-member records with embedding row
indices resolved via the DB site_id_order, so downstream consumers don't need to
re-embed or do any data processing.

Also merges DB-side fields (industry_confidence, business_model, brand_tier,
industry_style_profile) on top of the catalog descriptor, and copies
industry_style_profiles.json to the output directory if it exists.

Usage:
    ./.venv/bin/python scripts/07_export_index.py
    ./.venv/bin/python scripts/07_export_index.py --output-dir /path/to/Superpowering-Agents/index_data
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import get_connection, init_db, get_latest_run_id, get_all_style_labels

console = Console()


def _load_site_id_to_row() -> dict[int, int]:
    """Build site_id → embedding row index from site_id_order.txt."""
    order_path = config.OUTPUTS_DIR / "site_id_order.txt"
    if not order_path.exists():
        console.print(f"[red]Missing {order_path}. Run 02_embed.py first.[/]")
        raise SystemExit(1)

    mapping = {}
    with open(order_path) as f:
        for row_idx, line in enumerate(f):
            site_id = int(line.strip())
            mapping[site_id] = row_idx
    return mapping


def _load_site_id_to_domain(conn) -> dict[int, str]:
    """Build site_id → domain from the DB."""
    rows = conn.execute("SELECT id, domain FROM sites WHERE status='captured'").fetchall()
    return {r["id"]: r["domain"] for r in rows}


def main():
    parser = argparse.ArgumentParser(description="Export catalog_index.json and domain_to_row.json")
    parser.add_argument("--run-id", type=str, default=None, help="Clustering run ID (default: latest)")
    parser.add_argument("--output-dir", type=str, default=str(config.OUTPUTS_DIR), help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    init_db(conn)

    run_id = args.run_id or get_latest_run_id(conn)
    if not run_id:
        console.print("[red]No clustering runs found. Run 03_cluster.py first.[/]")
        conn.close()
        return

    console.print(f"[bold]Using run_id:[/] {run_id}")

    # Load the style catalog (legacy format, same as what SA uses)
    catalog_path = config.OUTPUTS_DIR / "style_catalog.json"
    if not catalog_path.exists():
        console.print(f"[red]Missing {catalog_path}. Run export_for_llm.py --format legacy first.[/]")
        conn.close()
        return

    with open(catalog_path) as f:
        catalog = json.load(f)

    # Build lookup tables
    site_id_to_row = _load_site_id_to_row()
    site_id_to_domain = _load_site_id_to_domain(conn)
    domain_to_site_id = {v: k for k, v in site_id_to_domain.items()}

    console.print(f"  {len(site_id_to_row)} embedding rows, {len(site_id_to_domain)} captured sites")

    # Build DB label lookup: cluster_id → full label dict (includes new fields)
    db_labels = get_all_style_labels(conn, run_id)
    cluster_to_db_label: dict[int, dict] = {
        row["cluster_id"]: dict(row) for row in db_labels
    }
    console.print(f"  {len(cluster_to_db_label)} clusters with DB labels")

    # Flatten clusters into per-member records with row_idx
    catalog_index: list[dict] = []
    domain_to_row: dict[str, int] = {}
    missing = 0

    for style in catalog["styles"]:
        cluster_id = style["cluster_id"]
        # Start with catalog descriptor, then overlay DB fields (newer/richer)
        descriptor = dict(style.get("descriptor", {}))
        db_label = cluster_to_db_label.get(cluster_id, {})

        # Overlay DB-side fields, preferring DB values when present
        db_overlay_fields = [
            "industry", "industry_confidence", "business_model",
            "brand_tier", "industry_style_profile",
            "color_mode", "layout_pattern", "typography_style",
            "design_era", "target_audience", "distinguishing_features",
            "quality_score", "visual_style", "page_type",
        ]
        for field in db_overlay_fields:
            if field in db_label and db_label[field] is not None:
                descriptor[field] = db_label[field]

        for member in style["members"]:
            domain = member["domain"]
            site_id = domain_to_site_id.get(domain)

            if site_id is None or site_id not in site_id_to_row:
                missing += 1
                continue

            row_idx = site_id_to_row[site_id]
            domain_to_row[domain] = row_idx

            catalog_index.append({
                "domain": domain,
                "cluster_id": cluster_id,
                "category_hint": member.get("category_hint", ""),
                **descriptor,
                "row_idx": row_idx,
            })

    # Save catalog_index.json and domain_to_row.json
    catalog_index_path = output_dir / "catalog_index.json"
    domain_to_row_path = output_dir / "domain_to_row.json"

    with open(catalog_index_path, "w") as f:
        json.dump(catalog_index, f, indent=2)

    with open(domain_to_row_path, "w") as f:
        json.dump(domain_to_row, f, indent=2)

    console.print(f"[bold green]Exported {len(catalog_index)} entries -> {catalog_index_path}[/]")
    console.print(f"[bold green]Exported {len(domain_to_row)} domain mappings -> {domain_to_row_path}[/]")

    # Copy industry_style_profiles.json if it exists
    profiles_src = config.OUTPUTS_DIR / "industry_style_profiles.json"
    if profiles_src.exists():
        profiles_dst = output_dir / "industry_style_profiles.json"
        shutil.copy2(profiles_src, profiles_dst)
        console.print(f"[bold green]Copied industry_style_profiles.json -> {profiles_dst}[/]")
    else:
        console.print("[yellow]  industry_style_profiles.json not found — run 06_build_industry_profiles.py first[/]")

    if missing:
        console.print(f"[yellow]  {missing} members skipped (no screenshot or embedding)[/]")

    conn.close()


if __name__ == "__main__":
    main()
