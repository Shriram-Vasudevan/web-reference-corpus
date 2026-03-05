# Retrieval Scoring Spec v1

This document defines retrieval and ranking for the Web Reference Corpus.

## Objectives
- Prioritize records that are semantically relevant and operationally safe.
- Prefer validated patterns over anecdotal snippets.
- Ensure result diversity across failure classes.

## Query Contract
Input query should be normalized into:
- `task`: user intent in one sentence.
- `stack`: frameworks/runtime/datastore.
- `constraints`: latency, security, cost, compliance, deployment model.
- `risk_focus`: correctness/security/performance/a11y/operability emphasis.

## Candidate Generation
1. Build lexical query from `task + stack + constraints + risk_focus`.
2. Apply hard filters where known:
- framework/runtime compatibility
- domain (frontend/backend/infra)
- max acceptable risk level
- minimum review status (`reviewed` in production mode)
3. Retrieve top `K1` by BM25 over `title`, `summary`, `labels`, `keywords`.
4. Retrieve top `K2` by vector similarity over `retrieval.embedding_text`.
5. Union candidates and deduplicate by `id`.

## Scoring
For each candidate `c`:

- `semantic(c)`: normalized vector relevance score in `[0,1]`.
- `label_match(c)`: weighted overlap score between query facets and `labels` in `[0,1]`.
- `test_coverage(c)`: proportion of desired test types present in `artifacts.tests` in `[0,1]`.
- `freshness(c)`: decay score based on `retrieval.freshness` and `governance.last_verified_at` in `[0,1]`.
- `incident_proven(c)`: `1` if `evidence.source_type=incident` or `production`, else `0.5` for `oss/internal`, else `0.2`.
- `simplicity(c)`: inverse complexity proxy based on snippet size and number of moving parts in `[0,1]`.

Final score:

```text
score(c) =
  0.35 * semantic(c) +
  0.25 * label_match(c) +
  0.15 * test_coverage(c) +
  0.10 * freshness(c) +
  0.10 * incident_proven(c) +
  0.05 * simplicity(c)
```

Then multiply by confidence factor:

```text
score'(c) = score(c) * confidence_multiplier(labels.confidence)
```

Where confidence multipliers are:
- `gold=1.0`
- `silver=0.85`
- `bronze=0.70`

## Diversity and Safety Re-Rank
1. Sort by `score'(c)` descending.
2. Enforce diversity cap: max 2 results per `bug_class`.
3. Keep at least one result with `quality_axes` containing `correctness`.
4. If query includes security constraints, keep at least one result with `quality_axes` containing `security`.
5. If all selected results are below `silver`, emit warning metadata: `low_confidence_context=true`.

## Output Assembly
Return each result as a bundle:
- `solution`
- `contracts.invariants`
- `contracts.failure_modes`
- `artifacts.tests` (at least one where available)
- `anti_patterns` (optional but preferred)

Do not return bare code snippets without invariants and failure modes.

## Fallback Policy
When no `gold` records match:
1. Return best `silver/bronze` candidates.
2. Add mandatory test generation prompt:
- include timeout path
- include partial failure path
- include concurrency path where relevant
3. Mark response with `recommendation_mode=draft`.

## Suggested Defaults
- `K1=40` (BM25)
- `K2=40` (vector)
- candidate union cap: `80`
- final returned references: `6-10`

## Telemetry
Track per query:
- hit rate by corpus
- median top-1 score
- percent of selected records by confidence tier
- downstream acceptance rate in generated PRs
- post-merge regression rate for tasks using retrieved references
