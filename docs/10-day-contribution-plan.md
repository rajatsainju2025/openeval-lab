# 10-Day Contribution Plan (Minimum 10–15 contributions per day)

Plan start: 2025-08-09
Plan end:   2025-08-18

Guidelines
- Target 12+ atomic commits/day (small, reviewable changes). Prefer 1 logical change per commit.
- Keep green CI at all times. Run lint/format/tests before each push.
- Update README and examples as features evolve.
- Use conventional commits (feat, fix, docs, chore, refactor, test, perf).
- After each milestone, aggregate runs and verify dashboard.

Day 1 (2025-08-09) – Concurrency, retries, timeouts for adapters
1) feat(utils): add retry/backoff helper (exponential jitter) and timeout context.
2) feat(adapters): shared request executor with asyncio/semaphore + sync wrapper.
3) feat(cli): add --concurrency, --max-retries, --request-timeout options to run.
4) refactor(openai): use executor; surface adapter_kwargs for concurrency.
5) test(utils): unit tests for retry/backoff behavior.
6) test(openai): mock client; verify retries and timeout handling.
7) docs(README): document new flags with OpenAI example.
8) docs(examples): add examples/qa_openai_concurrency.json.
9) feat(core): track request error counts in manifest.timing/errors.
10) web(index): show error rate and request stats if present.
11) web(leaderboard): add column toggle for error rate.
12) chore: ruff rules for asyncio/safety; update pre-commit hooks.
13) ci: add matrix run for Python 3.10–3.12 (non-required).

Day 2 (2025-08-10) – Prediction cache
1) feat(cache): add file-based cache (SQLite or JSONL) keyed by adapter/model/prompt hash.
2) feat(cli): add --cache-dir and --cache-mode (off, read, write, rw) to run.
3) refactor(core): integrate cache read/write around adapter.generate.
4) feat(core): manifest.cache with hits/misses/ratio.
5) test(cache): unit tests for cache hit/miss and invalidation by model/adapter.
6) docs(README): caching usage and caveats; deterministic prompts guidance.
7) docs(examples): add examples/qa_cache_spec.json.
8) web(index): show cache hit ratio and annotate cached records.
9) web(leaderboard): filter/sort by cache hit ratio.
10) perf: fast prompt hashing utilizing blake3 (optional dependency).
11) chore: optional extra [perf] for blake3 and sqlite3 helpers.
12) ci: cache test artifacts between steps (actions/cache for .pytest_cache).

Day 3 (2025-08-11) – Summarization task + ROUGE
1) feat(tasks): add SummarizationTask with prompt template/options.
2) feat(metrics): add ROUGE-L via rouge-score (optional in [metrics]).
3) docs(examples): examples/sum_jsonl.jsonl dataset and examples/sum_spec.json.
4) test(task): summarization smoke tests w/ EchoAdapter.
5) docs(README): summarization quickstart.
6) web(index): show long reference/prediction truncation with tooltips.
7) web: add task badge/chip styles.
8) feat(metrics): add length/conciseness heuristic metric.
9) test(metrics): unit tests for new metrics.
10) chore: typing improvements across tasks and metrics.
11) ci: ensure optional metrics import errors don’t fail core tests.
12) docs: update schema examples to include new task and metrics.

Day 4 (2025-08-12) – LLM-judge metric (reference-free + pairwise)
1) feat(metrics): LLMJudge (reference-free) using an adapter; safe defaults.
2) feat(metrics): PairwiseJudge for A/B comparisons; JSONL pair dataset loader.
3) test(metrics): mock adapter to avoid network; determinism by seed.
4) feat(cli): add --judge-adapter override in run for judge metrics.
5) docs(examples): judge_spec.json with mock adapter; instructions for OpenAI.
6) web(run-detail): new route /run/<file> to inspect full records.
7) web: highlight judge scores and disagreements.
8) docs(README): warnings on cost/safety for judge metrics.
9) refactor(core): per-record metadata extensibility for metric-specific fields.
10) test(web): template rendering tests via Starlette TestClient.
11) chore: split templates into partials; basic Tailwind-lite CSS utilities.
12) perf: lazy load records on run-detail page.

Day 5 (2025-08-13) – Dashboard depth & UX
1) web: add pagination and search for records on run-detail.
2) web: CSV/JSON export of records and metrics.
3) web(leaderboard): persistent sort/filter via query string.
4) web: quick compare view for two runs (diff predicted vs reference).
5) feat(core): artifact bundling option (write records.csv alongside JSON).
6) docs: dashboard screenshots and usage guide.
7) test(web): snapshot tests for templates.
8) perf: template minify step; static asset cache headers.
9) chore: add favicon and basic branding.
10) docs: CONTRIBUTING update for UI contributions.
11) refactor(web): small components for metric chips and kv blocks.
12) ci: upload built demo artifacts as workflow artifacts.

Day 6 (2025-08-14) – Spec schema & validation
1) feat(spec): strengthen JSON Schema; enums for known tasks/adapters/metrics.
2) feat(cli): openeval validate <spec> command with rich error reporting.
3) feat(spec): defaulting rules for seed/records/artifacts/options.
4) test(spec): schema validation tests (valid/invalid examples).
5) docs(examples): invalid_spec_examples/ for users.
6) docs(README): validation command usage.
7) feat(core): structured errors for missing/invalid components.
8) chore: internal registry map for known plugin paths.
9) test: registry resolution tests; dotted vs colon import forms.
10) web: show spec hash and validation status on run-detail.
11) refactor: type hints and Protocols refined; mypy config added.
12) ci: add mypy step; fail on new type errors.

Day 7 (2025-08-15) – HF Datasets polish
1) feat(hf): add split, streaming, and shuffle seed options.
2) feat(hf): include dataset revision/sha in manifest.
3) test(hf): mock small dataset; ensure len/iter behavior.
4) docs(examples): examples/qa_hf_spec.json with split/train.
5) docs(README): HF caching notes and token permissions.
6) perf: batch prefetch for adapters; early prompt construction.
7) refactor: dataset fingerprinting utilities consolidated.
8) web: dataset pill shows provider icon (HF/local).
9) test: dataset fingerprint tests.
10) chore: optional extra [hf] documented in pyproject classifiers.
11) ci: conditional install [hf] for HF tests.
12) docs: troubleshooting guide for HF rate limits.

Day 8 (2025-08-16) – Reproducibility hardening
1) feat(core): capture git commit/branch in manifest.
2) feat(cli): openeval lock verify <lockfile> command.
3) feat(cli): run --use-lock <lockfile> to pin env/model/dataset.
4) feat(core): include pip freeze in manifest (first 100 lines) or separate artifact.
5) test(lock): verification tests; hash mismatches produce warnings/errors.
6) docs(README): full reproducibility recipe.
7) web: show verification status and lockfile download.
8) chore: add make targets for lock/verify in project scripts.
9) ci: ensure lockfile round-trip on example runs.
10) refactor: consistent timestamp formatting (UTC ISO8601).
11) perf: optional gzip for records.json artifact.
12) docs: security notes for secrets and API keys.

Day 9 (2025-08-17) – Code generation task & metrics
1) feat(tasks): CodeGenTask with prompt template and tests.
2) feat(metrics): pass@k (Monte Carlo) and simple exact compile+run checker (py only, sandboxed).
3) test(metrics): isolated temp dirs; safe execution with timeouts.
4) docs(examples): examples/codegen_spec.json with toy problems.
5) docs(README): codegen section and safety disclaimers.
6) web: show pass@k and error breakdowns.
7) refactor: task registry split by domain (nlp, code, vision placeholder).
8) chore: add tox or nox for multi-env testing.
9) ci: add job to run codegen tests only (tagged slow).
10) perf: reuse adapter sessions where possible.
11) docs: roadmap update and milestone checklist.
12) test: end-to-end smoke across all tasks via cli runner.

Day 10 (2025-08-18) – Polish, QA, and release prep
1) docs: CHANGELOG.md for all new features.
2) chore: bump version to 0.1.0.
3) ci: coverage badge and thresholds, codecov (optional).
4) docs: Issue/PR templates and labels.
5) docs: architecture overview diagram.
6) feat(cli): --version prints rich env (openeval, python, platform).
7) web: about page with links to docs/spec.
8) refactor: remove dead code; tighten lint rules.
9) test: increase coverage for edge cases.
10) perf: micro-optimizations in core loop (string ops).
11) packaging: sdist/wheel build and twine check (no publish).
12) examples: refresh and verify all specs and datasets.
13) final: run leaderboard with multiple fresh runs and update screenshots.

Daily execution checklist (repeat each day)
- [ ] Create a branch for the day’s work (e.g., feat/day-1-concurrency).
- [ ] Break features into 10–15 atomic commits with clear messages.
- [ ] Keep tests green locally; push frequently.
- [ ] Open a PR; self-review; merge after CI passes.
- [ ] Run example specs; aggregate runs; verify dashboard pages.
- [ ] Update README/examples/docs as applicable.
- [ ] Log learnings and next-day adjustments at the top of this file.

Adjustment policy
- If scope changes or blockers arise, replace any item with an equivalent-sized contribution (docs/tests/refactors/UX polish) to maintain 10–15 contributions/day while keeping mainline stable.
