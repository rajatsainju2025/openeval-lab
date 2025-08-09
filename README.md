# OpenEval Lab

[![CI](https://github.com/rajatsainju2025/openeval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rajatsainju2025/openeval-lab/actions/workflows/ci.yml)

An extensible, reproducible evaluation framework for LLMs and multimodal agents.

- Plugin-based tasks, datasets, adapters, and metrics
- JSON/YAML experiment specs with version pinning
- Deterministic seeding and artifact logging
- Simple CLI and minimal web dashboard
- MIT License

Quickstart:
- Install: `pip install -e '.[dev]'`
- Run example: `openeval run examples/qa_spec.json --records --artifacts artifacts`
- View dashboard: `uvicorn openeval.web.app:app --reload`

Optional extras:
- OpenAI adapter: `pip install -e '.[openai]'`
- Hugging Face datasets: `pip install -e '.[hf]'`
- Advanced metrics (SacreBLEU, BERTScore): `pip install -e '.[metrics]'`

Dashboard & artifacts:
- Pass `--records` to include per-example outputs in results.
- Use `--artifacts DIR` to write outputs to a directory.
- The dashboard reads `results.json` from the CWD; copy your artifact there to preview.

Multi-run leaderboard:
- Every run can be saved with `--artifacts runs` to write timestamped `runs/<ts>.json`.
- New command `openeval runs collect --dir runs` aggregates all `*.json` into `runs/index.json`.
- Dashboard page `/leaderboard` compares metrics, latency, adapter, dataset fingerprint, spec hash, and optional run_name.

Day 1 features (concurrency & reliability):
- New CLI flags: `--concurrency N`, `--max-retries N`, `--request-timeout SEC`.
- Core now executes requests concurrently with retry+timeout, and tracks error rate in timing.

Examples:
- QA task on JSONL and CSV: see `examples/`
- Use OpenAI adapter: `pip install -e '.[openai]'` and set `OPENAI_API_KEY`, then `openeval run examples/qa_openai_spec.json --records --artifacts runs` (costs may apply)
- Advanced metrics: `examples/qa_metrics_spec.json` (requires `pip install -e '.[metrics]'`)

Demo: leaderboard workflow
- Install metrics: `pip install -e '.[metrics]'`
- Run variants named and saved: `openeval run examples/qa_metrics_spec.json --run-name bleu+bertscore --records --artifacts runs`
- Aggregate: `openeval runs collect --dir runs`
- Start dashboard: `uvicorn openeval.web.app:app --reload` then open http://localhost:8000/leaderboard

Reproducibility
- Each result includes a manifest (python/platform/packages) and dataset/spec hashes.
- Create a lockfile: `openeval lock --from runs/<ts>.json --out openeval-lock.json`.

Planning & tracking
- 10-day contribution plan: see `docs/10-day-contribution-plan.md`.
- Project board (GitHub Projects): https://github.com/rajatsainju2025/openeval-lab/projects – use label `plan-10d` on issues.

Goals:
- Reproducible, configurable evals for LLMs/agents
- Clean plug-in APIs; small, testable units
- Minimal but useful dashboard

Roadmap:
- [ ] Core abstractions (Task, Dataset, Adapter, Metric)
- [ ] CLI: `openeval run <spec>`
- [ ] JSON Schema for spec validation
- [ ] Built-in tasks (QA, code, web), metrics (accuracy, BLEU, BERTScore), adapters (OpenAI, local)
- [ ] Dataset loaders (HF Datasets, local)
- [ ] Reproducibility: lockfiles, hashes
- [ ] Web dashboard (FastAPI + SvelteKit or Streamlit-lite)

## OpenAI Example Usage and Safety Notes

When using the OpenAI adapter, be aware of the following:

- You need to set the `OPENAI_API_KEY` environment variable with your OpenAI API key.
- Usage of the OpenAI API may incur costs, depending on your usage and OpenAI's pricing.
- Ensure that your usage complies with OpenAI's use case policy and guidelines.
