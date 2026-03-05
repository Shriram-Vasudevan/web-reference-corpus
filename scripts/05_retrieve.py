#!/usr/bin/env python3
"""Step 5: Query interface for finding visually similar websites or reference records."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieve.query_engine import QueryEngine

console = Console()


def display_visual_results(result: dict):
    """Pretty-print visual query results."""
    if "error" in result:
        console.print(f"[red]{result['error']}[/]")
        return

    console.print(f"\n[bold]Query:[/] {result['query']} ({result['query_type']})")
    console.print(f"[dim]Run ID: {result.get('run_id', 'N/A')}[/]")

    ds = result.get("dominant_style")
    if ds:
        console.print(f"\n[bold cyan]Dominant Style:[/] {ds.get('page_type', 'N/A')} / {ds.get('visual_style', 'N/A')}")
        console.print(f"  Quality: {ds.get('quality_score', '—')}/5 | "
                       f"Industry: {ds.get('industry', '—')} | "
                       f"Color: {ds.get('color_mode', '—')}")
        console.print(f"  Layout: {ds.get('layout_pattern', '—')} | "
                       f"Typography: {ds.get('typography_style', '—')}")

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


def display_reference_results(result: dict):
    """Pretty-print reference retrieval results."""
    if "error" in result:
        console.print(f"[red]{result['error']}[/]")
        return

    console.print(f"\n[bold]Reference Query:[/] {result['query']}")
    console.print(f"[dim]Records: {result.get('records_path', 'N/A')}[/]")
    if result.get("low_confidence_context"):
        console.print("[yellow]Low-confidence context: all top results are bronze.[/]")

    table = Table(title="\nTop Reference Records")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("ID")
    table.add_column("Corpus")
    table.add_column("Confidence")
    table.add_column("Axes")

    for i, r in enumerate(result.get("results", []), start=1):
        table.add_row(
            str(i),
            f"{r['score']:.3f}",
            r.get("id", "—"),
            r.get("corpus", "—"),
            r.get("confidence", "—"),
            ",".join(r.get("quality_axes", [])[:3]) or "—",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Query website style database or reference corpus")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text query (e.g., 'dark minimal SaaS')")
    group.add_argument("--image", type=str, help="Path to image file")
    group.add_argument("--url", type=str, help="URL already in the database")
    group.add_argument("--reference-query", type=str, help="Reference corpus query")

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    parser.add_argument("--records-path", type=str, default="outputs/reference_records.v1.json")
    parser.add_argument("--stack", type=str, default=None)
    parser.add_argument("--constraints", type=str, default=None)
    parser.add_argument("--risk-focus", type=str, default=None)
    parser.add_argument("--min-review-status", type=str, choices=["draft", "reviewed", "approved"], default="reviewed")
    args = parser.parse_args()

    engine = QueryEngine()

    if args.text:
        result = engine.query_by_text(args.text, top_k=args.top_k)
    elif args.image:
        result = engine.query_by_image(args.image, top_k=args.top_k)
    elif args.url:
        result = engine.query_by_url(args.url, top_k=args.top_k)
    else:
        result = engine.query_reference_records(
            query=args.reference_query,
            top_k=args.top_k,
            records_path=args.records_path,
            stack=args.stack,
            constraints=args.constraints,
            risk_focus=args.risk_focus,
            min_review_status=args.min_review_status,
        )

    if args.json:
        console.print_json(json.dumps(result, indent=2, default=str))
    else:
        if result.get("mode") == "reference":
            display_reference_results(result)
        else:
            display_visual_results(result)


if __name__ == "__main__":
    main()
