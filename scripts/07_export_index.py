#!/usr/bin/env python3
"""Step 7: Export catalog_index.json and domain_to_row.json for Superpowering-Agents.

Flattens style_catalog.json clusters into per-member records with embedding row
indices resolved via the DB site_id_order, so downstream consumers don't need to
re-embed or do any data processing.

Usage:
    ./.venv/bin/python scripts/07_export_index.py
    ./.venv/bin/python scripts/07_export_index.py --output-dir /path/to/Superpowering-Agents/index_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import get_connection, init_db, get_latest_run_id

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

    # Flatten clusters into per-member records with row_idx
    catalog_index: list[dict] = []
    domain_to_row: dict[str, int] = {}
    missing = 0

    for style in catalog["styles"]:
        cluster_id = style["cluster_id"]
        descriptor = style.get("descriptor", {})

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

    # Save outputs
    catalog_index_path = output_dir / "catalog_index.json"
    domain_to_row_path = output_dir / "domain_to_row.json"

    with open(catalog_index_path, "w") as f:
        json.dump(catalog_index, f, indent=2)

    with open(domain_to_row_path, "w") as f:
        json.dump(domain_to_row, f, indent=2)

    console.print(f"[bold green]Exported {len(catalog_index)} entries -> {catalog_index_path}[/]")
    console.print(f"[bold green]Exported {len(domain_to_row)} domain mappings -> {domain_to_row_path}[/]")

    if missing:
        console.print(f"[yellow]  {missing} members skipped (no screenshot or embedding)[/]")

    conn.close()


if __name__ == "__main__":
    main()
