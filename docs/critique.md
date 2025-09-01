# Critique of Previous Plans (through Aug 2025)

This document summarizes strengths, gaps, and risks observed in the prior plans and implementation notes.

## Strengths
- Ambition with clear user stories (CLI-first, registry, dashboard, spec validation).
- Emphasis on reproducibility (locks, manifests) and caching.
- Breadth of tasks and metrics, including judge metrics and code execution.
- Solid test coverage and examples to guide new users.

## Gaps and risks
- Core contracts not crystal clear: Task/Dataset/Adapter/Metric interfaces weren’t documented with minimal examples and invariants.
- Spec ergonomics: Short-name registry exists, but discoverability and validation messages were uneven.
- Statistical rigor: Paired bootstrap and calibration hooks are referenced but not consistently exposed in CLI or docs.
- Scope creep: Dashboard ambitions vs. minimal, reliable artifact exports and CLI-first ergonomics.
- EvalOps: Artifact schema and run indexing exist but aren’t standardized for third-party tools.
- Reliability: Retries/timeouts covered; rate-limit strategies and cost-awareness not surfaced in defaults.

## Recommendations
- Document the core contracts with 1-page “contracts” and inline examples, including expected error modes.
- Ship a single, predictable JSON results schema with JSON Schema + versioning, and add CLI validation.
- Expose statistical utilities (paired bootstrap, CI) as first-class flags, with defaults and warnings.
- Keep dashboard minimal; prioritize artifact exports and CLI diffs/compare.
- Add explicit EvalOps doc: file layout, run aggregation, indexing conventions.
- Provide cost/rate-limit guidance and safe defaults for API adapters.

## Success criteria (next 6 weeks)
- Users can:
  - run: `openeval run spec.json` and get a validated results.json matching the schema.
  - compare: `openeval compare runA.json runB.json --paired-bootstrap`.
  - browse: minimal web view that renders any schema-valid results.
- Docs contain: quickstart, contracts, SOTA context, EvalOps, troubleshooting.
