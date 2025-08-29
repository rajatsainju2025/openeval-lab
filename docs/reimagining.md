# Reimagining OpenEval Lab (Aug 2025)

This document critiques the previous plans and proposes the next phase of changes, inspired by current best practices (HELM, lm-evaluation-harness, emerging EvalOps).

## Critique of previous plans

- Scope drift: Mixed dashboard, agent-level tracing, and diverse metrics created breadth without a crisp, documented core contract for tasks/adapters/datasets.
- Spec usability: YAML/JSON specs exist, but discoverability of short names and component docs was limited.
- Reproducibility: Good manifesting and lockfiles, but dataset path handling and example validation were brittle in some contexts.
- Research alignment: Strong feature set, but not clearly mapped to HELM’s multi-metric rigor (calibration, robustness, fairness) and harness’s config-first ergonomics.
- CI/UX: Tests are solid. More dev ergonomics (registry inspection, example validation) help onboarding.

## SOTA takeaways (high level)

- HELM: Multi-metric, scenario coverage, transparency, and open prompts/outputs to enable comparability.
- lm-evaluation-harness: Config-based tasks, prompt templating (Jinja), strong caching and backend flexibility (APIs, HF, vLLM), community scale.
- Light-weight evals: Prefer simple, reproducible tasks with clear invariants; lean on extras for heavy deps.

## Next-phase objectives

- Sharpen the core contracts and docs (registry metadata, tutorial, design docs).
- Make specs and examples plug-and-play (validation command, robust paths).
- Add calibration/robustness hooks later; start with interfaces and docs now.
- Keep tests green; favor incremental, low-risk commits.

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
