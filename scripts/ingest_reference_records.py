#!/usr/bin/env python3
"""Ingest reference records from JSON/JSONL/CSV into schema-v1 payloads."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def _today() -> str:
    return date.today().isoformat()


def _tokenize(text: str) -> list[str]:
    out = []
    for part in text.replace("|", ",").replace(";", ",").split(","):
        token = part.strip()
        if token:
            out.append(token)
    return out


def _parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return s
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return value
    return value


def _as_list(value: Any) -> list[Any]:
    value = _parse_maybe_json(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return _tokenize(value)
    return [value]


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slug_id(prefix: str, record: dict[str, Any]) -> str:
    seed = "|".join(
        [
            _as_str(record.get("title", "")),
            _as_str(record.get("summary", "")),
            _as_str(record.get("problem", "")),
            _as_str(record.get("solution", "")),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _load_input(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if isinstance(payload.get("records"), list):
                return payload["records"]
            if isinstance(payload.get("styles"), list):
                return payload["styles"]
        raise ValueError("Unsupported JSON format; expected list, {records:[...]}, or {styles:[...]}")

    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
        return rows

    if suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    raise ValueError(f"Unsupported input extension: {suffix}")


def _legacy_style_to_record(style: dict[str, Any], default_owner: str) -> dict[str, Any]:
    descriptor = style.get("descriptor", {}) or {}
    members = style.get("members", []) or []
    cluster_id = style.get("cluster_id", "unknown")

    page_type = _as_str(descriptor.get("page_type"), "unknown")
    visual_style = _as_str(descriptor.get("visual_style"), "unknown")
    quality_score = _as_int(descriptor.get("quality_score"), 3)

    risk_level = "low" if quality_score >= 4 else ("medium" if quality_score == 3 else "high")
    confidence = "gold" if quality_score >= 4 else ("silver" if quality_score == 3 else "bronze")

    urls = [_as_str(m.get("url")) for m in members[:5] if isinstance(m, dict)]
    summary = f"Cluster {cluster_id} style reference for {page_type} / {visual_style}."

    return {
        "id": f"legacy-style:{cluster_id}",
        "corpus": "dom_semantics",
        "title": f"Cluster {cluster_id}: {page_type} / {visual_style}",
        "summary": summary,
        "problem": "Need a web reference baseline based on observed production UI patterns.",
        "solution": f"Use as style baseline with cues from industry={_as_str(descriptor.get('industry'), 'general')} and layout={_as_str(descriptor.get('layout_pattern'), 'unknown')}. Examples: {', '.join(urls)}",
        "anti_patterns": [],
        "labels": {
            "domains": ["frontend"],
            "frameworks": ["web"],
            "risk_level": risk_level,
            "quality_axes": ["maintainability", "a11y"],
            "edge_cases": ["empty_data"],
            "bug_class": ["state_desync"],
            "scenario": ["happy_path"],
            "interaction": ["navigation"],
            "confidence": confidence,
        },
        "artifacts": {"code": [], "tests": [], "dom": []},
        "contracts": {
            "inputs": [{"name": "ui_goal", "type": "string", "constraints": "Align with user intent."}],
            "outputs": [{"name": "ui_reference_plan", "type": "object", "guarantees": "Contains structure and visual cues."}],
            "invariants": ["Maintain semantic clarity and readable hierarchy."],
            "failure_modes": ["Blindly copying style without adapting to product constraints."],
        },
        "evidence": {
            "source_type": "internal",
            "source_ref": f"legacy_style_catalog:{cluster_id}",
            "validation": ["manual_review_pass"],
            "metrics": {"after": {"quality_score": quality_score, "cluster_members": len(members)}},
        },
        "retrieval": {
            "keywords": [page_type.lower(), visual_style.lower()],
            "embedding_text": summary,
            "hard_filters": [f"cluster_id={cluster_id}"],
            "freshness": _today(),
        },
        "governance": {
            "owner": default_owner,
            "review_status": "reviewed",
            "version": "v1.0.0",
            "last_verified_at": _today(),
        },
    }


def _normalize_record(
    raw: dict[str, Any],
    labels: dict[str, Any],
    default_owner: str,
    default_review_status: str,
    infer_from_legacy: bool,
) -> dict[str, Any]:
    if infer_from_legacy and "descriptor" in raw and "cluster_id" in raw:
        return _legacy_style_to_record(raw, default_owner)

    dims = labels.get("dimensions", {})

    labels_in = raw.get("labels", {}) or {}
    if not isinstance(labels_in, dict):
        labels_in = {}

    # Flat aliases from CSV/jsonl
    aliases = {
        "bug_classes": "bug_class",
        "quality_axis": "quality_axes",
        "domain": "domains",
        "framework": "frameworks",
    }
    for old, new in aliases.items():
        if old in labels_in and new not in labels_in:
            labels_in[new] = labels_in[old]
        if old in raw and new not in labels_in:
            labels_in[new] = raw.get(old)

    def enum_pick(value: Any, allowed: list[str], default: str) -> str:
        v = _as_str(value, default).strip()
        return v if v in allowed else default

    def enum_list(value: Any, allowed: list[str], default: list[str]) -> list[str]:
        out = []
        for item in _as_list(value):
            s = _as_str(item).strip()
            if s in allowed:
                out.append(s)
        if not out:
            out = default[:]
        return list(dict.fromkeys(out))

    risk_allowed = dims.get("risk_level", ["low", "medium", "high"])
    confidence_allowed = dims.get("confidence", ["gold", "silver", "bronze"])

    labels_norm = {
        "domains": enum_list(labels_in.get("domains", raw.get("domains")), dims.get("domains", []), ["fullstack"]),
        "frameworks": list(dict.fromkeys(_as_list(labels_in.get("frameworks", raw.get("frameworks", "web"))) or ["web"])),
        "risk_level": enum_pick(labels_in.get("risk_level", raw.get("risk_level")), risk_allowed, "medium"),
        "quality_axes": enum_list(labels_in.get("quality_axes", raw.get("quality_axes")), dims.get("quality_axes", []), ["correctness"]),
        "edge_cases": enum_list(labels_in.get("edge_cases", raw.get("edge_cases")), dims.get("edge_cases", []), ["partial_failure"]),
        "bug_class": enum_list(labels_in.get("bug_class", raw.get("bug_class")), dims.get("bug_class", []), ["logic_error"]),
        "scenario": enum_list(labels_in.get("scenario", raw.get("scenario")), dims.get("scenario", []), ["degraded"]),
        "interaction": enum_list(labels_in.get("interaction", raw.get("interaction")), dims.get("interaction", []), ["navigation"]),
        "confidence": enum_pick(labels_in.get("confidence", raw.get("confidence")), confidence_allowed, "bronze"),
    }

    artifacts_in = raw.get("artifacts", {}) or {}
    if not isinstance(artifacts_in, dict):
        artifacts_in = {}

    contracts_in = raw.get("contracts", {}) or {}
    if not isinstance(contracts_in, dict):
        contracts_in = {}

    evidence_in = raw.get("evidence", {}) or {}
    if not isinstance(evidence_in, dict):
        evidence_in = {}

    retrieval_in = raw.get("retrieval", {}) or {}
    if not isinstance(retrieval_in, dict):
        retrieval_in = {}

    governance_in = raw.get("governance", {}) or {}
    if not isinstance(governance_in, dict):
        governance_in = {}

    title = _as_str(raw.get("title"), "Untitled Reference")
    summary = _as_str(raw.get("summary"), "Reference pattern for implementation guidance.")
    problem = _as_str(raw.get("problem"), "Need a robust implementation approach under real-world constraints.")
    solution = _as_str(raw.get("solution"), "Apply validated patterns and include tests for edge cases.")

    record = {
        "id": _as_str(raw.get("id")) or _slug_id("ingest", raw),
        "corpus": _as_str(raw.get("corpus"), "edge_case_pattern"),
        "title": title,
        "summary": summary,
        "problem": problem,
        "solution": solution,
        "anti_patterns": list(dict.fromkeys(_as_list(raw.get("anti_patterns", [])))),
        "labels": labels_norm,
        "artifacts": {
            "code": _as_list(artifacts_in.get("code", [])),
            "tests": _as_list(artifacts_in.get("tests", [])),
            "dom": _as_list(artifacts_in.get("dom", [])),
            **({"state_graph": artifacts_in["state_graph"]} if "state_graph" in artifacts_in else {}),
        },
        "contracts": {
            "inputs": _as_list(contracts_in.get("inputs", [])),
            "outputs": _as_list(contracts_in.get("outputs", [])),
            "invariants": [
                _as_str(x) for x in _as_list(contracts_in.get("invariants", ["Critical correctness and safety invariants are explicit."]))
            ],
            "failure_modes": [
                _as_str(x) for x in _as_list(contracts_in.get("failure_modes", ["Timeouts, partial failures, and invalid input are handled."]))
            ],
        },
        "evidence": {
            "source_type": _as_str(evidence_in.get("source_type"), "internal"),
            "source_ref": _as_str(evidence_in.get("source_ref"), _as_str(raw.get("source_ref"), "ingestion")),
            "validation": [
                v
                for v in _as_list(evidence_in.get("validation", ["manual_review_pass"]))
                if isinstance(v, str)
            ]
            or ["manual_review_pass"],
            "metrics": _parse_maybe_json(evidence_in.get("metrics", {})) or {},
        },
        "retrieval": {
            "keywords": list(dict.fromkeys([_as_str(x) for x in _as_list(retrieval_in.get("keywords", [])) if _as_str(x)]))
            or list(dict.fromkeys(_tokenize(f"{title},{summary}")))[:8],
            "embedding_text": _as_str(retrieval_in.get("embedding_text"), f"{title}. {summary} {problem} {solution}"),
            "hard_filters": list(dict.fromkeys([_as_str(x) for x in _as_list(retrieval_in.get("hard_filters", [])) if _as_str(x)])),
            "freshness": _as_str(retrieval_in.get("freshness"), _today()),
        },
        "governance": {
            "owner": _as_str(governance_in.get("owner"), default_owner),
            "review_status": _as_str(governance_in.get("review_status"), default_review_status),
            "version": _as_str(governance_in.get("version"), "v1.0.0"),
            "last_verified_at": _as_str(governance_in.get("last_verified_at"), _today()),
        },
    }

    return record


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_schema(records: list[dict[str, Any]], schema: dict[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for i, record in enumerate(records):
        for err in sorted(validator.iter_errors(record), key=lambda e: e.path):
            path = ".".join(str(x) for x in err.path)
            errors.append(f"record[{i}] {path or '<root>'}: {err.message}")
    return errors


def _validate_taxonomy(records: list[dict[str, Any]], labels: dict[str, Any]) -> list[str]:
    dims = labels.get("dimensions", {})
    errors: list[str] = []

    list_dims = ["domains", "quality_axes", "scenario", "interaction", "edge_cases", "bug_class"]

    for i, record in enumerate(records):
        corpus = record.get("corpus")
        if corpus not in dims.get("corpus", []):
            errors.append(f"record[{i}] corpus: unknown value '{corpus}'")

        lab = record.get("labels", {}) or {}
        for scalar in ["risk_level", "confidence"]:
            if lab.get(scalar) not in dims.get(scalar, []):
                errors.append(f"record[{i}] labels.{scalar}: unknown value '{lab.get(scalar)}'")

        for key in list_dims:
            vals = lab.get(key, [])
            if not isinstance(vals, list):
                errors.append(f"record[{i}] labels.{key}: must be list")
                continue
            allowed = set(dims.get(key, []))
            for v in vals:
                if v not in allowed:
                    errors.append(f"record[{i}] labels.{key}: unknown value '{v}'")

        val_signals = record.get("evidence", {}).get("validation", [])
        for v in val_signals:
            if v not in set(dims.get("validation_signals", [])):
                errors.append(f"record[{i}] evidence.validation: unknown value '{v}'")

        review_status = record.get("governance", {}).get("review_status")
        if review_status not in dims.get("review_status", []):
            errors.append(f"record[{i}] governance.review_status: unknown value '{review_status}'")

    return errors


def _merge_records(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {str(r.get("id")): r for r in existing}
    for r in incoming:
        merged[str(r.get("id"))] = r
    return list(merged.values())


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", payload)
    if not isinstance(records, list):
        return []
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest JSON/JSONL/CSV into schema-v1 records")
    parser.add_argument("--input", required=True, type=str, help="Input file: .json, .jsonl, or .csv")
    parser.add_argument("--output", type=str, default="outputs/reference_records.v1.json")
    parser.add_argument("--schema", type=str, default="schema.v1.json")
    parser.add_argument("--labels", type=str, default="labels.v1.json")
    parser.add_argument("--owner", type=str, default="web-reference-corpus")
    parser.add_argument("--review-status", type=str, default="reviewed", choices=["draft", "reviewed", "approved"])
    parser.add_argument("--merge", action="store_true", help="Merge into existing output by id")
    parser.add_argument("--strict", action="store_true", help="Fail if any record is invalid")
    parser.add_argument("--infer-legacy", action="store_true", help="Convert legacy style_catalog styles to schema records")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    schema_path = Path(args.schema)
    labels_path = Path(args.labels)

    for path in [input_path, schema_path, labels_path]:
        if not path.exists():
            console.print(f"[red]Missing file:[/] {path}")
            return 1

    raw_rows = _load_input(input_path)
    labels = _load_json(labels_path)
    schema = _load_json(schema_path)

    normalized = [
        _normalize_record(
            row,
            labels=labels,
            default_owner=args.owner,
            default_review_status=args.review_status,
            infer_from_legacy=args.infer_legacy,
        )
        for row in raw_rows
    ]

    existing = _load_existing_records(output_path) if args.merge else []
    records = _merge_records(existing, normalized) if args.merge else normalized

    schema_errors = _validate_schema(records, schema)
    taxonomy_errors = _validate_taxonomy(records, labels)
    all_errors = schema_errors + taxonomy_errors

    if all_errors and args.strict:
        console.print(f"[red]Validation failed:[/] {len(all_errors)} issue(s)")
        for err in all_errors[:100]:
            console.print(f"  - {err}")
        if len(all_errors) > 100:
            console.print(f"  ... truncated {len(all_errors) - 100} more")
        return 1

    if all_errors and not args.strict:
        console.print(f"[yellow]Validation issues:[/] {len(all_errors)} issue(s). Writing output anyway (non-strict mode).")
        for err in all_errors[:30]:
            console.print(f"  - {err}")
        if len(all_errors) > 30:
            console.print(f"  ... truncated {len(all_errors) - 30} more")

    payload = {
        "version": "v1.0.0",
        "generated_at": _today(),
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(input_path),
        "record_count": len(records),
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    console.print(f"[bold green]Wrote {len(records)} records[/] -> {output_path}")
    if all_errors:
        console.print("[yellow]Completed with validation warnings.[/]")
    else:
        console.print("[bold green]Schema + taxonomy validation passed.[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
