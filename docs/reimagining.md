# Reimagining OpenEval Lab (Sep 2025)

This document critiques the previous plans and proposes the next phase of changes, inspired by current best practices (HELM, lm-evaluation-harness, emerging EvalOps).

## Critique of previous plans

- Scope drift: Mixed dashboard, agent-level tracing, and diverse metrics created breadth without a crisp, documented core contract for tasks/adapters/datasets.
- Spec usability: YAML/JSON specs exist, but discoverability of short names and component docs was limited.
- Reproducibility: Good manifesting and lockfiles, but dataset path handling and example validation were brittle in some contexts.
- Research alignment: Strong feature set, but not clearly mapped to HELM’s multi-metric rigor (calibration, robustness, fairness) and harness’s config-first ergonomics.
- CI/UX: Tests are solid. More dev ergonomics (registry inspection, example validation) help onboarding.

## SOTA takeaways (high level)

- HELM (Bommasani et al., 2022–2023): Multi-metric, broad scenario coverage, transparency with released prompts/outputs.
- lm-evaluation-harness: Config-first tasks, Jinja templating, caching, multi-backend; thriving community.
- LMSYS Arena/Arena-Hard (2024): Pairwise human preference evaluation; handle position bias; emphasize uncertainty.
- MTEB: Suite design and artifact standards for embeddings; inspiration for curated suites and exports.
- Principle: Keep core small, precise, and schema-first; move heavy deps to extras.

## Next-phase objectives

- Sharpen the core contracts and docs (registry metadata, tutorial, design docs).
- Make specs and examples plug-and-play (validation command, robust paths).
- Add calibration/robustness hooks later; start with interfaces and docs now.
- Keep tests green; favor incremental, low-risk commits.

## Design principles (2025)
- Contracts-first: document the Task/Dataset/Adapter/Metric invariants and error modes.
- Schema-first: versioned results schema with CLI validation and compare tools.
- Minimal UI: basic web views; prioritize CLI exports and run diffs.
- Extensible extras: calibration/robustness/fairness as optional installs.

## Reading pointers (selection)
- HELM: https://arxiv.org/abs/2211.09110 (TMLR 2023)
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- LMSYS Arena: Arena-Hard and pairwise judging (2024) overview
- MTEB: https://arxiv.org/abs/2210.07316

## Concrete near-term changes

- Registry metadata and CLI inspection (done).
- Tutorial and quickstart path (done).
- Example validation utility in CLI.
- Programmatic registry tests for stability.
- Docs: SOTA insights and roadmap wiring.

## Mid-term road map

- Calibration/uncertainty: add calibration error metrics and confidence-aware scoring.
- Robustness sweeps: text noise/perturbation runners and paired bootstrap tests (partially done).
- Fairness/toxicity: opt-in checks to align with HELM dimensions.
- Backend breadth: vLLM and API latency-aware batching.
- EvalOps: artifact schema, remote registry sync, curated suite definitions.
