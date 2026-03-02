#!/usr/bin/env python3
"""Export the full style catalog as JSON for LLM consumption."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import (
    get_connection, init_db, get_latest_run_id, get_cluster_ids,
    get_cluster_members, get_all_style_labels,
)

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Export style catalog JSON")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(config.OUTPUTS_DIR / "style_catalog.json"))
    args = parser.parse_args()

    conn = get_connection()
    init_db(conn)

    run_id = args.run_id or get_latest_run_id(conn)
    if not run_id:
        console.print("[red]No clustering runs found.[/]")
        conn.close()
        return

    console.print(f"[bold]Exporting catalog for run_id:[/] {run_id}")

    labels = get_all_style_labels(conn, run_id)
    cluster_ids = get_cluster_ids(conn, run_id)

    catalog = {
        "run_id": run_id,
        "n_clusters": len(cluster_ids),
        "styles": [],
    }

    # Build label lookup
    label_map = {l["cluster_id"]: l for l in labels}

    for cid in cluster_ids:
        members = get_cluster_members(conn, run_id, cid)
        member_data = [
            {
                "url": m["url"],
                "domain": m["domain"],
                "category_hint": m["category_hint"],
            }
            for m in members
        ]

        style_entry = {
            "cluster_id": cid,
            "n_members": len(members),
            "example_urls": [m["url"] for m in members[:5]],
            "members": member_data,
        }

        label = label_map.get(cid)
        if label:
            style_entry["descriptor"] = {
                "umbrella_label": label["umbrella_label"],
                "substyle_traits": label["substyle_traits"],
                "visual_density": label["visual_density"],
                "color_mode": label["color_mode"],
                "typography_style": label["typography_style"],
                "layout_structure": label["layout_structure"],
                "motion_intensity": label["motion_intensity"],
                "visual_energy": label["visual_energy"],
            }

        catalog["styles"].append(style_entry)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(catalog, f, indent=2)

    console.print(f"[bold green]Exported {len(catalog['styles'])} styles → {output_path}[/]")

    # Summary
    for style in catalog["styles"]:
        desc = style.get("descriptor", {})
        label = desc.get("umbrella_label", "unlabeled")
        console.print(f"  Cluster {style['cluster_id']}: [cyan]{label}[/] ({style['n_members']} sites)")

    conn.close()


if __name__ == "__main__":
    main()
