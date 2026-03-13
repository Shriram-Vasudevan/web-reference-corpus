#!/usr/bin/env python3
"""Step 5: Secondary industry reclassification pass using focused VLM prompts.

Targets clusters where industry confidence is low or industry is a coarse catch-all
(technology | general | saas) and re-runs a dedicated industry-only classification.
This is cheaper than relabeling entire clusters — it sends the same screenshots but
asks only about industry, business_model, and brand_tier.

Usage:
    ./.venv/bin/python scripts/05_reclassify_industry.py
    ./.venv/bin/python scripts/05_reclassify_industry.py --threshold 0.7   # reclassify if confidence < 0.7
    ./.venv/bin/python scripts/05_reclassify_industry.py --industries technology saas general
    ./.venv/bin/python scripts/05_reclassify_industry.py --cluster 12 --force
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.prompt_templates import INDUSTRY_RECLASSIFY_SYSTEM, INDUSTRY_RECLASSIFY_USER
from src.utils.storage import (
    get_connection, init_db, get_latest_run_id, get_cluster_ids,
    get_cluster_members, get_style_label, update_industry_fields,
)

load_dotenv()
console = Console()

# Coarse catch-all labels that are candidates for refinement
COARSE_INDUSTRIES = {"technology", "saas", "general", "finance", "health", "education", "media"}

DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_MAX_SAMPLES = 5


def _encode_image(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def reclassify_cluster_industry(
    screenshot_paths: list[str],
    current_industry: str,
    current_confidence: float,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    max_retries: int = config.LABEL_MAX_RETRIES,
) -> dict:
    """Run a focused industry-only VLM pass on a cluster."""
    if len(screenshot_paths) > max_samples:
        paths = random.sample(screenshot_paths, max_samples)
    else:
        paths = screenshot_paths

    content = []
    for path in paths:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _encode_image(path),
            },
        })
    content.append({
        "type": "text",
        "text": INDUSTRY_RECLASSIFY_USER.format(
            count=len(paths),
            current_industry=current_industry,
            current_confidence=current_confidence,
        ),
    })

    client = anthropic.Anthropic()

    for attempt in range(1, max_retries + 1):
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=512,
            system=INDUSTRY_RECLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        raw_text = response.content[0].text.strip()

        try:
            json_text = raw_text
            if json_text.startswith("```"):
                lines = json_text.split("\n")
                json_text = "\n".join(lines[1:-1])
            parsed = json.loads(json_text)

            required = ["industry", "industry_confidence", "business_model", "brand_tier"]
            if all(k in parsed for k in required):
                parsed["industry_confidence"] = float(parsed["industry_confidence"])
                return parsed

        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt < max_retries:
                continue

    return {
        "industry": current_industry,
        "industry_confidence": current_confidence,
        "business_model": "unknown",
        "brand_tier": "unknown",
        "reasoning": "parse_error",
    }


def should_reclassify(label: dict, threshold: float, target_industries: list[str] | None) -> bool:
    """Decide whether a cluster warrants a reclassification pass."""
    industry = label.get("industry") or "general"
    confidence = label.get("industry_confidence")

    # If target_industries is specified, only reclassify those
    if target_industries and industry not in target_industries:
        return False

    # Always reclassify if no confidence score yet (old labels)
    if confidence is None:
        return True

    # Reclassify if below confidence threshold
    if float(confidence) < threshold:
        return True

    # Reclassify coarse labels even at moderate confidence
    if industry in COARSE_INDUSTRIES and float(confidence) < 0.9:
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Secondary industry reclassification pass")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--cluster", type=int, default=None, help="Reclassify one specific cluster")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Reclassify clusters with confidence below this value")
    parser.add_argument("--industries", nargs="+", default=None,
                        help="Only reclassify clusters with these current industry labels")
    parser.add_argument("--force", action="store_true",
                        help="Reclassify all clusters regardless of current confidence")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show which clusters would be reclassified without calling the API")
    args = parser.parse_args()

    conn = get_connection()
    init_db(conn)

    run_id = args.run_id or get_latest_run_id(conn)
    if not run_id:
        console.print("[red]No clustering runs found. Run 03_cluster.py first.[/]")
        conn.close()
        return

    console.print(f"[bold]Using run_id:[/] {run_id}")

    cluster_ids = [args.cluster] if args.cluster is not None else get_cluster_ids(conn, run_id)
    console.print(f"[bold]Checking {len(cluster_ids)} clusters[/]")

    results_table = Table(title="Industry Reclassification")
    results_table.add_column("Cluster", justify="right")
    results_table.add_column("Old Industry")
    results_table.add_column("Old Conf", justify="right")
    results_table.add_column("New Industry")
    results_table.add_column("New Conf", justify="right")
    results_table.add_column("Business Model")
    results_table.add_column("Brand Tier")
    results_table.add_column("Action")

    reclassified = 0
    skipped = 0

    for cid in cluster_ids:
        label = get_style_label(conn, run_id, cid)
        if not label:
            continue

        label_dict = dict(label)
        current_industry = label_dict.get("industry") or "general"
        current_confidence = label_dict.get("industry_confidence")
        conf_display = f"{current_confidence:.2f}" if current_confidence is not None else "n/a"

        if not args.force and not should_reclassify(label_dict, args.threshold, args.industries):
            skipped += 1
            continue

        if args.dry_run:
            results_table.add_row(
                str(cid), current_industry, conf_display,
                "—", "—", "—", "—", "[yellow]would reclassify[/]",
            )
            reclassified += 1
            continue

        members = get_cluster_members(conn, run_id, cid)
        paths = [m["screenshot_path"] for m in members if m["screenshot_path"]]

        if not paths:
            console.print(f"  [yellow]Cluster {cid}: no screenshots[/]")
            continue

        console.print(f"  Reclassifying cluster {cid} (was: {current_industry}, conf={conf_display})...")

        result = reclassify_cluster_industry(
            paths,
            current_industry=current_industry,
            current_confidence=current_confidence or 0.5,
        )

        new_industry = result["industry"]
        new_conf = result["industry_confidence"]
        business_model = result.get("business_model", "unknown")
        brand_tier = result.get("brand_tier", "unknown")

        update_industry_fields(
            conn, cid, run_id,
            industry=new_industry,
            industry_confidence=new_conf,
            business_model=business_model,
            brand_tier=brand_tier,
        )

        changed = new_industry != current_industry
        action = "[green]updated[/]" if changed else "[dim]confirmed[/]"

        results_table.add_row(
            str(cid),
            current_industry, conf_display,
            new_industry, f"{new_conf:.2f}",
            business_model, brand_tier,
            action,
        )
        reclassified += 1

    console.print()
    console.print(results_table)
    console.print(f"\n[bold]Reclassified:[/] {reclassified}  [bold]Skipped (already confident):[/] {skipped}")

    if args.dry_run:
        console.print("[dim]Dry run — no changes written.[/]")

    conn.close()


if __name__ == "__main__":
    main()
