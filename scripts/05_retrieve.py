#!/usr/bin/env python3
"""Step 5: Query interface for finding visually similar websites."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieve.query_engine import QueryEngine

console = Console()


def display_results(result: dict):
    """Pretty-print query results."""
    if "error" in result:
        console.print(f"[red]{result['error']}[/]")
        return

    console.print(f"\n[bold]Query:[/] {result['query']} ({result['query_type']})")
    console.print(f"[dim]Run ID: {result.get('run_id', 'N/A')}[/]")

    # Dominant style
    ds = result.get("dominant_style")
    if ds:
        console.print(f"\n[bold cyan]Dominant Style:[/] {ds.get('umbrella_label', 'N/A')}")
        console.print(f"  Density: {ds.get('visual_density', '—')} | "
                       f"Color: {ds.get('color_mode', '—')} | "
                       f"Typography: {ds.get('typography_style', '—')}")
        console.print(f"  Layout: {ds.get('layout_structure', '—')} | "
                       f"Motion: {ds.get('motion_intensity', '—')} | "
                       f"Energy: {ds.get('visual_energy', '—')}")

    # Results table
    table = Table(title="\nTop Results")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Domain")
    table.add_column("Category")
    table.add_column("Style")
    table.add_column("Cluster", justify="right")

    for r in result["results"]:
        table.add_row(
            str(r["rank"]),
            f"{r['score']:.3f}",
            r["domain"],
            r.get("category_hint", "—") or "—",
            r.get("style_label", "—"),
            str(r.get("cluster_id", "—")),
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Query website style database")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text query (e.g., 'dark minimal SaaS')")
    group.add_argument("--image", type=str, help="Path to image file")
    group.add_argument("--url", type=str, help="URL already in the database")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    engine = QueryEngine()

    if args.text:
        result = engine.query_by_text(args.text, top_k=args.top_k)
    elif args.image:
        result = engine.query_by_image(args.image, top_k=args.top_k)
    elif args.url:
        result = engine.query_by_url(args.url, top_k=args.top_k)

    if args.json:
        console.print_json(json.dumps(result, indent=2, default=str))
    else:
        display_results(result)


if __name__ == "__main__":
    main()
