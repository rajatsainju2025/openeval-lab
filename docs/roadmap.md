# Roadmap (Aug–Oct 2025)

## Core contracts
- Tighten Task/Dataset/Adapter/Metric docs with examples and invariants. (D1)
- Publish results JSON Schema v1 with semver and CLI validation. (D2)

## Metrics
- Calibration error (ECE), confidence summaries (interfaces + docs first). (D2)
- Robustness sweeps (noise/perturbation); paired bootstrap built-ins. (D3)
- Optional fairness/toxicity checks via extras. (D4)

## Backends
- vLLM adapter and latency-aware batching utilities. (D3)
- API adapters parity (OpenAI/Anthropic/local OpenAI-compatible). (D2)

## UX/DevEx
- Spec validation improvements and human-friendly errors. (D1)
- Registry metadata and browsing (done), richer descriptions. (D1)
- Tutorial, concepts, SOTA, reimagining docs (done), sample gallery. (D1)

## EvalOps
- Curated benchmark suites and export formats. (D3)
- Remote registry sync and library index. (D4)

## Success criteria
- A spec validated by `openeval validate` yields schema-valid results.json with manifest and artifacts.
- Users can compare two runs with `openeval compare --paired-bootstrap`.
- Minimal dashboard renders any schema-valid results; export CSV/JSON from UI.
- Optional extras install cleanly; core stays lightweight.
