# Core Concepts

## Plugins
- Task: builds prompts and postprocesses outputs.
- Dataset: yields Examples with id/input/reference/meta.
- Adapter: wraps a model API or local model; generate(prompt) -> str.
- Metric: compute(predictions, references) -> mapping.

## Specs
- JSON/YAML describing task, dataset, adapter, metrics, and run options.
- Supports short names via the registry.

## Reproducibility
- Seeds, manifests (env + packages), dataset/spec hashing, lockfiles.

## Artifacts & Dashboard
- Runs emit results.json and per-record traces when enabled.
- Web dashboard offers leaderboard and run details.
