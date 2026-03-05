# web-reference-corpus

A data pipeline that captures, embeds, clusters, and labels screenshots from real websites to produce a structured reference corpus. The corpus is used by [Superpowering-Agents](https://github.com/Shriram-Vasudevan/Superpowering-Agents) to ground LLM code generation in concrete visual references instead of relying on the model's training data defaults.

## What this produces

The pipeline outputs three artifacts that downstream consumers (like Superpowering-Agents) use directly:

| Artifact | Description |
|---|---|
| `data/screenshots/` | 3,660 viewport-sized PNGs of real websites (1440x900) |
| `data/all_embeddings.npy` | CLIP ViT-B/32 embeddings matrix (3660, 512), L2-normalized |
| `outputs/style_catalog.json` | 98 style clusters, each with a Claude-generated descriptor (`page_type`, `visual_style`, `color_mode`, `layout_pattern`, `typography_style`, `quality_score`, `industry`, `design_era`, `target_audience`, `distinguishing_features`) and member domains |
| `outputs/catalog_index.json` | Flattened per-site records with cluster descriptors and embedding row indices — the primary retrieval index |
| `outputs/domain_to_row.json` | Domain-to-embedding-row mapping for vector search |

Superpowering-Agents copies these files into its `reference_data/` and `index_data/` directories. From there, the MCP server uses metadata filtering + CLIP vector search to retrieve the best-matching references for any given prompt.

## Data pipeline

The core pipeline runs in order:

```bash
# 1) Build website URL list from award sites, curated lists, etc.
./.venv/bin/python scripts/00_build_seeds.py

# 2) Capture viewport screenshots (Playwright, cookie dismissal, networkidle wait)
./.venv/bin/python scripts/01_capture.py

# 3) Embed all screenshots with CLIP ViT-B/32 (open_clip)
./.venv/bin/python scripts/02_embed.py

# 4) Cluster embeddings (UMAP dimensionality reduction + HDBSCAN)
./.venv/bin/python scripts/03_cluster.py

# 5) Label each cluster with Claude VLM (page_type, visual_style, quality_score, etc.)
./.venv/bin/python scripts/04_label.py

# 6) (Optional) Visualize clusters in 2D
./.venv/bin/python scripts/06_visualize.py

# 7) Export catalog_index.json and domain_to_row.json for Superpowering-Agents
./.venv/bin/python scripts/07_export_index.py
```

After the pipeline completes, copy the following to Superpowering-Agents:
- `data/screenshots/` -> `reference_data/screenshots/`
- `data/all_embeddings.npy` -> `reference_data/all_embeddings.npy`
- `outputs/style_catalog.json` -> `reference_data/style_catalog.json`
- `outputs/catalog_index.json` -> `index_data/catalog_index.json`
- `outputs/domain_to_row.json` -> `index_data/domain_to_row.json`

Step 7 replaces the need to run `build_index.py` in Superpowering-Agents (which previously re-embedded all screenshots to compute row mappings). The corpus now exports everything SA needs directly.

## Retrieval

```bash
# Query the corpus directly (standalone, outside of the MCP workflow)
./.venv/bin/python scripts/05_retrieve.py \
  --query "dark minimal portfolio with large typography" \
  --top-k 5
```

## Schema v1 (experimental)

The repo includes `schema.v1.json` and `labels.v1.json` as an experimental structured record format, along with scripts to export, ingest, and validate records against it. This schema was designed for general software engineering reference records (with fields like `bug_class`, `edge_cases`, `interaction` patterns) and does not yet align well with the visual design data this corpus actually contains. The v1 records in `outputs/reference_records.v1.json` are not currently consumed by Superpowering-Agents.

```bash
# Export clusters to v1 records
./.venv/bin/python scripts/export_for_llm.py --format schema-v1 --output outputs/reference_records.v1.json --validate

# Ingest external records
./.venv/bin/python scripts/ingest_reference_records.py --input path/to/external.jsonl --output outputs/reference_records.v1.json --merge --strict

# Validate against schema + labels
./.venv/bin/python scripts/validate_corpus.py --input outputs/reference_records.v1.json --schema schema.v1.json --labels labels.v1.json
```

## Eval

An eval pipeline compares baseline (no references) vs reference-backed code generation outputs using Claude as a judge. The eval infrastructure is functional but the current sample size is small (3 cases). Results should be treated as directional, not statistically significant.

```bash
# Auto-generate baseline and reference outputs
./.venv/bin/python scripts/build_eval_outputs.py \
  --prompts-file evals/prompts.example.jsonl \
  --baseline-cmd 'cd /path/to/runner && ./run_baseline.sh {prompt_json}' \
  --reference-cmd 'cd /path/to/runner && ./run_reference.sh {prompt_json}' \
  --baseline-out evals/baseline_outputs.generated.jsonl \
  --reference-out evals/reference_outputs.generated.jsonl

# Run the eval judge
./.venv/bin/python scripts/eval_reference_impact.py \
  --prompts-file evals/prompts.example.jsonl \
  --baseline-file evals/baseline_outputs.generated.jsonl \
  --reference-file evals/reference_outputs.generated.jsonl \
  --output-json outputs/eval_reference_impact.json \
  --output-md outputs/eval_reference_impact.md
```

## Repository map

### Data pipeline (run in order)
- `scripts/00_build_seeds.py` — builds the website URL list from award sites and curated sources
- `scripts/01_capture.py` — captures viewport screenshots with Playwright
- `scripts/02_embed.py` — generates CLIP embeddings for all screenshots
- `scripts/03_cluster.py` — UMAP reduction + HDBSCAN clustering
- `scripts/04_label.py` — Claude VLM labeling of each cluster (page_type, visual_style, etc.)
- `scripts/05_retrieve.py` — standalone similarity search over the corpus
- `scripts/06_visualize.py` — 2D cluster visualization
- `scripts/07_export_index.py` — exports `catalog_index.json` and `domain_to_row.json` for Superpowering-Agents

### Source modules
- `src/capture/screenshotter.py` — Playwright screenshot engine with cookie dismissal
- `src/capture/cookie_dismiss.py` — cookie banner detection and dismissal
- `src/embed/clip_embedder.py` — CLIP embedding extraction (open_clip ViT-B/32)
- `src/cluster/hdbscan_cluster.py` — HDBSCAN clustering
- `src/retrieve/similarity.py` — cosine similarity search

### Schema and eval tooling
- `scripts/export_for_llm.py` — exports cluster data to v1 schema records
- `scripts/ingest_reference_records.py` — imports external records into the corpus
- `scripts/validate_corpus.py` — validates records against schema + controlled labels
- `scripts/build_eval_outputs.py` — generates baseline/reference outputs for eval
- `scripts/eval_reference_impact.py` — runs Claude-as-judge comparison eval
- `schema.v1.json` — experimental record schema (not yet aligned with visual design data)
- `labels.v1.json` — controlled vocabulary for schema dimensions

### Config and data
- `config.py` — central configuration (paths, model settings, capture params)
- `data/sources/website_urls.csv` — primary website URL list used for capture
- `data/sources/website_urls_large.csv` — expanded website URL list generated by the builder
- `data/sources/website_urls_original.csv` — original curated URL list snapshot
- `data/screenshots/` — captured website screenshots (3,660 PNGs)
- `data/all_embeddings.npy` — CLIP embedding matrix
- `data/website_styles.db` — SQLite database of site metadata
- `outputs/style_catalog.json` — clustered + labeled catalog (primary output)

## Current status

- The data pipeline is stable and produces the three artifacts Superpowering-Agents needs.
- The v1 schema needs to be reworked to reflect visual design dimensions rather than software engineering patterns before it can replace the current style_catalog.json format.
- The eval pipeline works but needs more test cases to produce meaningful results.
