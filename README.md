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
- View dashboard: `openeval web --reload` (then open http://localhost:8000)
 - Diagnose env: `openeval doctor` or JSON: `openeval doctor --json`

Optional extras:
- OpenAI adapter: `pip install -e '.[openai]'`
- Hugging Face datasets: `pip install -e '.[hf]'`
- Advanced metrics (SacreBLEU, BERTScore, ROUGE-L): `pip install -e '.[metrics]'`

Caching (Day 2):
- Flags: `--cache off|read|write|rw`, `--cache-dir .openeval-cache`, `--cache-ttl SECONDS`.
- Example: `openeval run examples/qa_spec.json --cache rw --cache-dir .openeval-cache --records`.
- Stats: hit/miss/hit-rate recorded under `timing.*`; dashboard shows cache hit rate and marks cached records.

Summarization + ROUGE-L (Day 3):
- Try summarization: `openeval run examples/sum_spec.json --records --artifacts runs`
- Datasets: `examples/sum_toy.jsonl`
- Metric: ROUGE-L via `openeval.metrics.rouge.ROUGEL` (install `.[metrics]`)

Spec tools:
- Print schema: `openeval schema`
- Validate a spec: `openeval validate examples/qa_spec.json`
- Use short names via registry: `task: qa`, `dataset: jsonl`, `adapter: echo`, `metrics: [{"name": "exact_match"}]`.
- Render prompts (debugging): `openeval write_out examples/qa_spec.json --limit 5` or write JSONL with `--out prompts.jsonl`.
- More examples: see `examples/qa_write_out.md`.

Dashboard & artifacts:
- Pass `--records` to include per-example outputs in results.
- Use `--artifacts DIR` to write outputs to a directory.
- The dashboard reads `results.json` from the CWD; copy your artifact there to preview.
- Leaderboard rows link to detailed run pages at `/run/<file>` with paginated records.

Multi-run leaderboard:
- Every run can be saved with `--artifacts runs` to write timestamped `runs/<ts>.json`.
- New command `openeval runs collect --dir runs` aggregates all `*.json` into `runs/index.json`.
- Dashboard page `/leaderboard` compares metrics, latency, error rate, cache hit rate, adapter, dataset fingerprint, spec hash, and optional run_name.
- Columns include avg latency, throughput, error rate, and cache hit rate; sort client-side.

Day 1 features (concurrency & reliability):
- New CLI flags: `--concurrency N`, `--max-retries N`, `--request-timeout SEC`.
- Core now executes requests concurrently with retry+timeout, and tracks error rate in timing.

Examples:
- QA task on JSONL and CSV: see `examples/`
- Use OpenAI adapter: `pip install -e '.[openai]'` and set `OPENAI_API_KEY`, then `openeval run examples/qa_openai_spec.json --records --artifacts runs` (costs may apply)
- Advanced metrics: `examples/qa_metrics_spec.json` (requires `pip install -e '.[metrics]'`)
- LLM-as-a-judge: `examples/qa_judge_spec.json` (set `OPENAI_API_KEY` and install `.[openai]`).

Demo: leaderboard workflow
- Install metrics: `pip install -e '.[metrics]'`
- Run variants named and saved: `openeval run examples/qa_metrics_spec.json --run-name bleu+bertscore --records --artifacts runs`
- Aggregate: `openeval runs collect --dir runs`
- Start dashboard: `openeval web --reload` then open http://localhost:8000/leaderboard
- Click a run filename to open its detail page at `/run/<file>`.

LLM-as-a-Judge metric
- Add to metrics: `{ "name": "llm_judge", "kwargs": { "judge_adapter": "openai-chat", "judge_kwargs": {"model": "gpt-4o-mini"}, "max_examples": 200 } }`.
- Uses balanced position calibration (A/B and B/A). Outputs `win_rate`, `wins`, `losses`, `ties`.

Statistical Analysis
- Enable bootstrap confidence intervals: `openeval run examples/qa_spec.json --statistical`
- Compare runs with significance testing: `openeval compare-runs runA.json runB.json --bootstrap 1000`
- Analyze evaluation biases: `openeval analyze-bias examples/qa_spec.json`

Cost Tracking
- Automatic cost calculation for OpenAI API usage
- Cost summary included in evaluation manifests
- Support for multiple models with different pricing

Advanced Metrics
- Calibration Error (ECE): measures model confidence calibration
- Confidence Intervals: statistical bounds on performance metrics
- Diversity Metrics: text generation diversity analysis
- Perplexity: language model evaluation metric

High-Throughput Inference
- vLLM adapter for optimized GPU inference: `openeval.adapters.vllm_adapter.VLLMAdapter`
- Tensor parallelism and memory optimization
- Faster evaluation on large datasets

Few-Shot Evaluation
- Support for few-shot prompting in tasks
- Automatic prompt construction with examples
- Configurable example selection and formatting

Bias Detection
- Positional bias analysis for multiple-choice tasks
- Prompt sensitivity testing
- Automated recommendations for bias mitigation

Reproducibility
- Each result includes a manifest (python/platform/packages) and dataset/spec hashes; manifest now includes the current git commit when available.
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


## Documentation

- Getting started: `docs/tutorial.md`
- Core concepts: `docs/concepts.md`
- Core contracts: `docs/contracts.md`
- EvalOps (artifacts & runs): `docs/evalops.md`
- SOTA references: `docs/sota.md`
- Critique: `docs/critique.md`
- Next-phase plan: `docs/next-phase-plan.md`
- Roadmap: `docs/roadmap.md`
- ICML-style paper: `ICML_PAPER.md`
- Contribution guide: `CONTRIBUTING.md`
