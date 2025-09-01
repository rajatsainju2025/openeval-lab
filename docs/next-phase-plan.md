# Next Phase Plan (Sep 2025)

Focus: clarity, comparability, and low-friction onboarding.

## Objectives (3–4 weeks)
- Contracts: one-page docs for Task/Dataset/Adapter/Metric with code snippets.
- Results schema v1: JSON Schema + semver, CLI validate/print.
- Compare CLI: `openeval compare A.json B.json --paired-bootstrap`.
- EvalOps docs: artifacts layout, run indexing, CSV/JSON export.
- Tutorial revamp and example gallery.

## Milestones
- Week 1: Contracts + tutorial + schema draft.
- Week 2: CLI validate improvements; compare CLI (paired bootstrap).
- Week 3: EvalOps docs; examples refresh; minimal dashboard export buttons.
- Week 4: Backends parity review; extras install polish; cut 0.1.0.

## Non-goals
- Heavy dashboard rebuilds; keep minimal.
- Adding many new tasks/metrics without contracts and tests.

## Risks and mitigations
- Scope creep: keep PRs small and schema-first.
- Breaking changes: gate behind semver and feature flags.
- Optional deps: extras only, core remains light.
