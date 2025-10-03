# EvalOps: Artifacts and Run Management

This page explains the artifact layout, results schema expectations, and run indexing.

## Results schema
- Single JSON file containing: manifest, metrics (aggregates), and optional records.
- Versioned via `results.schema.json` with semver (v1). Use `openeval validate-result`.

## Artifacts layout
- runs/<timestamp>.json — primary result file
- runs/index.json — aggregated index built by `openeval runs collect --dir runs`
- artifacts/ — optional extras (records.csv, prompts.jsonl, logs)

## Indexing conventions
- Each run entry contains: file, created_at, task, dataset fingerprint, adapter/model, metrics summary, timing, error_rate, cache_hit_rate.
- Use `openeval runs collect` to update the index; dashboard reads from index.json.

## Exporting
- CLI: `openeval export --format csv|json --out out/` to convert run files.
- UI: export buttons on run-detail and leaderboard pages.
