# web-reference-corpus

This repository builds and maintains a structured reference corpus of real websites and style clusters.

The corpus is intended for agent systems and MCP servers that need better visual grounding before code generation. Instead of asking an LLM to generate UI from text alone, you retrieve concrete references (screenshots + descriptors) and use those references as context for planning and generation.

## What this is

- A data and tooling pipeline for web reference records.
- A schema (`schema.v1.json`) and taxonomy (`labels.v1.json`) for consistent, machine-readable records.
- Scripts to export, ingest, validate, and retrieve reference records.

## What this enables

- Better visual fidelity in generated websites by grounding generation in concrete references.
- More repeatable retrieval via explicit labels (page type, style, layout, typography, risk/confidence).
- Cleaner handoff to MCP/agent tools that need:
  - retrieval-ready reference metadata
  - schema-validated records
  - explicit governance/quality fields

In practice (as used in the `Superpowering-Agents` MCP workflow), this corpus supports:
- reference-backed context building (`superpower_context`)
- targeted retrieval (`superpower_retrieve`)
- reference-guided code generation (`superpower_generate`)
- primitive selection and planning layers on top of retrieved references

## Core workflow

```bash
# 1) Export style clusters into schema-v1 records
./.venv/bin/python scripts/export_for_llm.py \
  --format schema-v1 \
  --output outputs/reference_records.v1.json \
  --validate

# 2) Ingest external JSON/JSONL/CSV into schema-v1
./.venv/bin/python scripts/ingest_reference_records.py \
  --input path/to/external.jsonl \
  --output outputs/reference_records.v1.json \
  --merge \
  --strict

# 3) Validate corpus explicitly (schema + taxonomy)
./.venv/bin/python scripts/validate_corpus.py \
  --input outputs/reference_records.v1.json \
  --schema schema.v1.json \
  --labels labels.v1.json

# 4) Query reference records for downstream agent/MCP usage
./.venv/bin/python scripts/05_retrieve.py \
  --reference-query "robust form submit with timeout handling" \
  --records-path outputs/reference_records.v1.json \
  --risk-focus correctness \
  --stack web \
  --top-k 5 --json
```

## Repository map

- `scripts/export_for_llm.py`: exports cluster data into legacy or `schema-v1` records.
- `scripts/ingest_reference_records.py`: imports external records and merges into corpus output.
- `scripts/validate_corpus.py`: validates records against schema + controlled labels.
- `scripts/05_retrieve.py`: retrieves top matching references for downstream prompting/orchestration.
- `schema.v1.json`: record contract for corpus entries.
- `labels.v1.json`: controlled vocabulary for corpus dimensions.

## Current status

- This is an actively refined corpus and workflow.
- The main goal is practical generation quality: better first-pass website outputs by grounding generation in real reference examples.
