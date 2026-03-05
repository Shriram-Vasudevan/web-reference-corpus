#!/usr/bin/env python3
"""Validate reference records against schema.v1.json and labels.v1.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

console = Console()


def load_records(input_path: Path) -> list[dict]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = payload.get("records", payload)
    if not isinstance(records, list):
        raise ValueError("Input must be a list or an object containing 'records'.")
    return records


def validate_schema(records: list[dict], schema: dict) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        raise RuntimeError("jsonschema is required. Install with: pip install jsonschema")

    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for i, record in enumerate(records):
        for err in sorted(validator.iter_errors(record), key=lambda e: e.path):
            path = ".".join(str(x) for x in err.path)
            errors.append(f"record[{i}] {path or '<root>'}: {err.message}")
    return errors


def validate_taxonomy(records: list[dict], labels: dict) -> list[str]:
    dims = labels.get("dimensions", {})
    errors: list[str] = []

    label_keys = [
        "domains",
        "quality_axes",
        "scenario",
        "interaction",
        "edge_cases",
        "bug_class",
    ]

    for i, record in enumerate(records):
        labels_obj = record.get("labels", {})

        corpus = record.get("corpus")
        if corpus not in dims.get("corpus", []):
            errors.append(f"record[{i}] corpus: unknown value '{corpus}'")

        for key in ["risk_level", "confidence"]:
            value = labels_obj.get(key)
            if value not in dims.get(key, []):
                errors.append(f"record[{i}] labels.{key}: unknown value '{value}'")

        for key in label_keys:
            allowed = set(dims.get(key, []))
            values = labels_obj.get(key, [])
            if not isinstance(values, list):
                errors.append(f"record[{i}] labels.{key}: must be a list")
                continue
            for v in values:
                if v not in allowed:
                    errors.append(f"record[{i}] labels.{key}: unknown value '{v}'")

        validations = record.get("evidence", {}).get("validation", [])
        allowed_validations = set(dims.get("validation_signals", []))
        for v in validations:
            if v not in allowed_validations:
                errors.append(f"record[{i}] evidence.validation: unknown value '{v}'")

        review_status = record.get("governance", {}).get("review_status")
        if review_status not in dims.get("review_status", []):
            errors.append(f"record[{i}] governance.review_status: unknown value '{review_status}'")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate schema-v1 reference corpus payload")
    parser.add_argument("--input", type=str, default="outputs/reference_records.v1.json")
    parser.add_argument("--schema", type=str, default="schema.v1.json")
    parser.add_argument("--labels", type=str, default="labels.v1.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    schema_path = Path(args.schema)
    labels_path = Path(args.labels)

    for path in [input_path, schema_path, labels_path]:
        if not path.exists():
            console.print(f"[red]Missing file:[/] {path}")
            return 1

    try:
        records = load_records(input_path)
    except Exception as exc:
        console.print(f"[red]Failed loading input:[/] {exc}")
        return 1

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    try:
        schema_errors = validate_schema(records, schema)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        return 1

    taxonomy_errors = validate_taxonomy(records, labels)

    all_errors = schema_errors + taxonomy_errors
    if all_errors:
        console.print(f"[red]Validation failed:[/] {len(all_errors)} issue(s)")
        for err in all_errors[:100]:
            console.print(f"  - {err}")
        if len(all_errors) > 100:
            console.print(f"  ... truncated {len(all_errors) - 100} more")
        return 1

    console.print(f"[bold green]Validation passed[/] ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
