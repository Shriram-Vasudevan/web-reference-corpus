#!/usr/bin/env python3
"""Export style catalog JSON in legacy or schema-v1 format."""

from __future__ import annotations

import argparse
from datetime import date
import json
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import (
    get_connection,
    init_db,
    get_latest_run_id,
    get_cluster_ids,
    get_cluster_members,
    get_all_style_labels,
)

console = Console()


def _risk_level_from_quality(score: int) -> str:
    if score <= 2:
        return "high"
    if score == 3:
        return "medium"
    return "low"


def _confidence_from_quality(score: int) -> str:
    if score >= 4:
        return "gold"
    if score == 3:
        return "silver"
    return "bronze"


def _build_schema_record(run_id: str, cid: int, members: list[dict], label: dict | None) -> dict:
    label = dict(label) if label is not None else {}
    today = date.today().isoformat()
    page_type = label.get("page_type", "unknown")
    visual_style = label.get("visual_style", "unknown")
    quality_score = int(label.get("quality_score", 0) or 0)

    example_urls = [m["url"] for m in members[:5]]
    keywords = list({
        page_type.lower(),
        visual_style.lower(),
        (label.get("industry") or "general").lower(),
        (label.get("layout_pattern") or "layout").lower(),
        (label.get("typography_style") or "typography").lower(),
    })

    summary = (
        f"Reference style cluster {cid} with {len(members)} members. "
        f"Primary pattern: {page_type} / {visual_style}."
    )
    solution = (
        f"Use this cluster as a UI reference baseline for {page_type}. "
        f"Style cues: visual_style={visual_style}, color_mode={label.get('color_mode') or 'unknown'}, "
        f"layout_pattern={label.get('layout_pattern') or 'unknown'}, typography_style={label.get('typography_style') or 'unknown'}. "
        f"Distinguishing features: {label.get('distinguishing_features') or 'n/a'}. "
        f"Example URLs: {', '.join(example_urls)}"
    )

    return {
        "id": f"cluster:{run_id}:{cid}",
        "corpus": "dom_semantics",
        "title": f"Cluster {cid}: {page_type} / {visual_style}",
        "summary": summary,
        "problem": "Need a production-derived web UI reference pattern for planning structure and styling.",
        "solution": solution,
        "anti_patterns": [],
        "labels": {
            "domains": ["frontend"],
            "frameworks": ["web"],
            "risk_level": _risk_level_from_quality(quality_score),
            "quality_axes": ["maintainability", "a11y"],
            "edge_cases": ["empty_data"],
            "bug_class": ["state_desync"],
            "scenario": ["happy_path"],
            "interaction": ["navigation"],
            "confidence": _confidence_from_quality(quality_score),
        },
        "artifacts": {
            "code": [],
            "tests": [],
            "dom": [],
        },
        "contracts": {
            "inputs": [
                {
                    "name": "ui_goal",
                    "type": "string",
                    "constraints": "Must align with page intent and target audience.",
                }
            ],
            "outputs": [
                {
                    "name": "ui_reference_plan",
                    "type": "object",
                    "guarantees": "Includes structure and visual style cues.",
                }
            ],
            "invariants": [
                "Reference keeps semantic structure and readability intact.",
            ],
            "failure_modes": [
                "Overfitting to style references without adapting to product constraints.",
            ],
        },
        "evidence": {
            "source_type": "internal",
            "source_ref": f"style_labels:{run_id}:{cid}",
            "validation": ["manual_review_pass"],
            "metrics": {
                "after": {
                    "quality_score": quality_score,
                    "cluster_members": len(members),
                }
            },
        },
        "retrieval": {
            "keywords": [k for k in keywords if k and k != "unknown"],
            "embedding_text": f"{summary} {solution}",
            "hard_filters": [f"run_id={run_id}", f"cluster_id={cid}"],
            "freshness": today,
        },
        "governance": {
            "owner": "web-reference-corpus",
            "review_status": "reviewed",
            "version": "v1.0.0",
            "last_verified_at": today,
        },
    }


def _validate_records_with_schema(records: list[dict], schema_path: Path) -> tuple[bool, str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return False, "jsonschema is not installed; skipping schema validation."

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = []
    for i, record in enumerate(records):
        rec_errors = sorted(validator.iter_errors(record), key=lambda e: e.path)
        for err in rec_errors:
            errors.append(f"record[{i}] {list(err.path)}: {err.message}")

    if errors:
        return False, "\n".join(errors[:20])
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Export style catalog JSON")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--format", type=str, choices=["legacy", "schema-v1"], default="legacy")
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.OUTPUTS_DIR / "style_catalog.json"),
    )
    parser.add_argument("--schema", type=str, default="schema.v1.json")
    parser.add_argument("--validate", action="store_true", help="Validate schema-v1 exports against --schema")
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
    label_map = {l["cluster_id"]: l for l in labels}

    if args.format == "legacy":
        payload = {
            "run_id": run_id,
            "n_clusters": len(cluster_ids),
            "styles": [],
        }

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
                    "page_type": label["page_type"],
                    "visual_style": label["visual_style"],
                    "quality_score": label["quality_score"],
                    "industry": label["industry"],
                    "color_mode": label["color_mode"],
                    "layout_pattern": label["layout_pattern"],
                    "typography_style": label["typography_style"],
                    "design_era": label["design_era"],
                    "target_audience": label["target_audience"],
                    "distinguishing_features": label["distinguishing_features"],
                }

            payload["styles"].append(style_entry)
    else:
        records = []
        for cid in cluster_ids:
            members = get_cluster_members(conn, run_id, cid)
            records.append(_build_schema_record(run_id, cid, members, label_map.get(cid)))

        if args.validate:
            schema_path = Path(args.schema)
            if not schema_path.exists():
                console.print(f"[red]Schema file not found: {schema_path}[/]")
                conn.close()
                raise SystemExit(1)
            valid, message = _validate_records_with_schema(records, schema_path)
            if not valid:
                console.print(f"[red]Schema validation failed:[/]\n{message}")
                conn.close()
                raise SystemExit(1)

        payload = {
            "version": "v1.0.0",
            "generated_at": date.today().isoformat(),
            "run_id": run_id,
            "records": records,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.format == "legacy":
        console.print(f"[bold green]Exported {len(payload['styles'])} styles -> {output_path}[/]")
    else:
        console.print(f"[bold green]Exported {len(payload['records'])} schema-v1 records -> {output_path}[/]")

    conn.close()


if __name__ == "__main__":
    main()
