#!/usr/bin/env python3
"""Step 2: Generate CLIP embeddings for all captured screenshots."""

import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import get_connection, init_db, get_captured_sites, store_embedding, get_all_embeddings
from src.embed.clip_embedder import CLIPEmbedder

console = Console()


def main():
    conn = get_connection()
    init_db(conn)

    sites = get_captured_sites(conn)
    if not sites:
        console.print("[red]No captured sites found. Run 01_capture.py first.[/]")
        conn.close()
        return

    console.print(f"[bold]Found {len(sites)} captured sites[/]")

    # Check which already have embeddings
    existing_ids, _ = get_all_embeddings(conn)
    existing_set = set(existing_ids)
    to_embed = [s for s in sites if s["id"] not in existing_set]

    if not to_embed:
        console.print("[green]All sites already have embeddings.[/]")
    else:
        console.print(f"[yellow]Embedding {len(to_embed)} new screenshots...[/]")

        embedder = CLIPEmbedder()
        console.print(f"[dim]Using device: {embedder.device}[/]")
        model_name = f"{config.CLIP_MODEL_NAME}/{config.CLIP_PRETRAINED}"

        # Batch embed
        paths = [s["screenshot_path"] for s in to_embed]
        site_ids = [s["id"] for s in to_embed]

        # Process in batches with progress
        batch_size = config.EMBED_BATCH_SIZE
        for i in tqdm(range(0, len(paths), batch_size), desc="Embedding batches"):
            batch_paths = paths[i:i + batch_size]
            batch_ids = site_ids[i:i + batch_size]
            vectors = embedder.embed_images_batch(batch_paths, batch_size=len(batch_paths))
            for sid, vec in zip(batch_ids, vectors):
                store_embedding(conn, sid, vec, model_name)

    # Export bulk numpy file
    all_ids, all_vectors = get_all_embeddings(conn)
    np.save(config.EMBEDDINGS_PATH, all_vectors)
    console.print(f"\n[bold green]Done![/] Embeddings shape: {all_vectors.shape}")
    console.print(f"Saved to: {config.EMBEDDINGS_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
