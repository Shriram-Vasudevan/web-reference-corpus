#!/usr/bin/env python3
"""Step 4: Label clusters using Claude VLM."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import (
    get_connection, init_db, get_latest_run_id, get_cluster_ids,
    get_cluster_members, store_style_label, get_style_label,
)
from src.cluster.label_clusters import label_cluster

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Label clusters with Claude VLM")
    parser.add_argument("--run-id", type=str, default=None, help="Clustering run ID (default: latest)")
    parser.add_argument("--dry-run", action="store_true", help="Preview prompts without calling API")
    parser.add_argument("--cluster", type=int, default=None, help="Label only a specific cluster")
    parser.add_argument("--force", action="store_true", help="Re-label even if already labeled")
    args = parser.parse_args()

    conn = get_connection()
    init_db(conn)

    run_id = args.run_id or get_latest_run_id(conn)
    if not run_id:
        console.print("[red]No clustering runs found. Run 03_cluster.py first.[/]")
        conn.close()
        return

    console.print(f"[bold]Using run_id:[/] {run_id}")

    # Get clusters to label
    if args.cluster is not None:
        cluster_ids = [args.cluster]
    else:
        cluster_ids = get_cluster_ids(conn, run_id)

    if not cluster_ids:
        console.print("[red]No clusters found for this run.[/]")
        conn.close()
        return

    console.print(f"[bold]Found {len(cluster_ids)} clusters to label[/]\n")

    results_table = Table(title="Cluster Labels")
    results_table.add_column("Cluster", justify="right")
    results_table.add_column("Size", justify="right")
    results_table.add_column("Page Type")
    results_table.add_column("Visual Style")
    results_table.add_column("Quality", justify="center")
    results_table.add_column("Industry")

    for cid in cluster_ids:
        # Skip if already labeled (unless --force)
        if not args.force:
            existing = get_style_label(conn, run_id, cid)
            if existing:
                results_table.add_row(
                    str(cid), "—",
                    f"[dim]{existing['page_type']}[/] (cached)",
                    existing["visual_style"] or "—",
                    str(existing["quality_score"]) if existing["quality_score"] else "—",
                    existing["industry"] or "—",
                )
                continue

        members = get_cluster_members(conn, run_id, cid)
        paths = [m["screenshot_path"] for m in members if m["screenshot_path"]]

        if not paths:
            console.print(f"  [yellow]Cluster {cid}: no screenshots available[/]")
            continue

        console.print(f"  Labeling cluster {cid} ({len(paths)} members)...")

        label_data, raw = label_cluster(paths, dry_run=args.dry_run)

        if args.dry_run:
            console.print(f"    [dim]Dry run — {label_data.get('n_images', 0)} images would be sent[/]")
            console.print(f"    [dim]Prompt preview: {label_data.get('prompt_preview', '')[:200]}...[/]")
            results_table.add_row(str(cid), str(len(paths)), "(dry run)", "—", "—")
        else:
            store_style_label(conn, cid, run_id, label_data, raw)
            results_table.add_row(
                str(cid), str(len(paths)),
                label_data.get("page_type", "—"),
                label_data.get("visual_style", "—"),
                str(label_data.get("quality_score", "—")),
                label_data.get("industry", "—"),
            )

    console.print()
    console.print(results_table)
    conn.close()


if __name__ == "__main__":
    main()
