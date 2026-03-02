#!/usr/bin/env python3
"""Step 3: UMAP reduction + HDBSCAN clustering."""

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import get_connection, init_db, get_all_embeddings, store_clusters
from src.embed.reduction import reduce_for_clustering, reduce_for_visualization
from src.cluster.hdbscan_cluster import cluster_embeddings, get_cluster_stats

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Cluster website embeddings")
    parser.add_argument("--min-cluster-size", type=int, default=config.HDBSCAN_MIN_CLUSTER_SIZE)
    parser.add_argument("--n-components", type=int, default=config.UMAP_N_COMPONENTS_CLUSTER)
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID (default: auto)")
    args = parser.parse_args()

    run_id = args.run_id or uuid.uuid4().hex[:8]

    conn = get_connection()
    init_db(conn)

    # Load embeddings
    site_ids, embeddings = get_all_embeddings(conn)
    if len(site_ids) == 0:
        console.print("[red]No embeddings found. Run 02_embed.py first.[/]")
        conn.close()
        return

    console.print(f"[bold]Loaded {len(site_ids)} embeddings[/] (shape: {embeddings.shape})")

    # UMAP reduction for clustering
    console.print(f"[yellow]Running UMAP → {args.n_components}D for clustering...[/]")
    reduced_cluster = reduce_for_clustering(
        embeddings, n_components=args.n_components
    )
    np.save(config.UMAP_20D_PATH, reduced_cluster)

    # UMAP reduction for visualization
    console.print("[yellow]Running UMAP → 2D for visualization...[/]")
    reduced_viz = reduce_for_visualization(embeddings)
    np.save(config.UMAP_2D_PATH, reduced_viz)

    # HDBSCAN clustering
    console.print(f"[yellow]Running HDBSCAN (min_cluster_size={args.min_cluster_size})...[/]")
    labels, probabilities, _ = cluster_embeddings(
        reduced_cluster, min_cluster_size=args.min_cluster_size
    )

    # Stats
    stats = get_cluster_stats(labels)

    # Warn if noise is too high
    if stats["noise_ratio"] > config.NOISE_WARN_THRESHOLD:
        console.print(
            f"[bold red]WARNING:[/] Noise ratio {stats['noise_ratio']:.1%} exceeds "
            f"threshold {config.NOISE_WARN_THRESHOLD:.0%}. "
            "Consider lowering min_cluster_size."
        )

    # Display results
    table = Table(title=f"Clustering Results (run_id={run_id})")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total sites", str(stats["n_total"]))
    table.add_row("Clusters", str(stats["n_clusters"]))
    table.add_row("Noise points", str(stats["n_noise"]))
    table.add_row("Noise ratio", f"{stats['noise_ratio']:.1%}")
    console.print(table)

    # Cluster size breakdown
    size_table = Table(title="Cluster Sizes")
    size_table.add_column("Cluster ID")
    size_table.add_column("Size")
    for cid, size in sorted(stats["cluster_sizes"].items()):
        size_table.add_row(str(cid), str(size))
    console.print(size_table)

    # Store in DB
    store_clusters(conn, run_id, site_ids, labels, probabilities)
    console.print(f"\n[bold green]Stored clustering run:[/] {run_id}")

    conn.close()


if __name__ == "__main__":
    main()
