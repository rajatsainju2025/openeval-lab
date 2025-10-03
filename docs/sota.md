# SOTA Evaluation References (Aug–Sep 2025)

A living note of relevant frameworks and papers to guide design choices.

## HELM (Holistic Evaluation of Language Models)
- Multi-metric: accuracy, calibration, robustness, fairness, toxicity, efficiency.
- Transparent prompts/outputs and scenario definitions for comparability.
- Strong emphasis on methodology and uncertainty.

Ref: HELM (2022+, ongoing updates)

## EleutherAI lm-evaluation-harness
- Config-first; Jinja templating; caching; diverse backends (HF, vLLM, APIs).
- Large community task library; strong reproducibility culture.
- Practical batching and GPU guidance; GGUF and quantization support.

Repo: https://github.com/EleutherAI/lm-evaluation-harness

## LMSYS Arena & Arena-Hard
- Human preference evaluations at scale; pairwise A/B judging.
- Position bias handling and statistical reporting challenges.

## MTEB (Massive Text Embedding Benchmark)
- Multi-task embedding evaluation; clear protocols and leaderboards.
- Lessons for task suites and artifact exports.

## Robustness & Safety Evaluations
- Prompt sensitivity, adversarial prompting, jailbreak detection.
- Toxicity/fairness stress tests and red-teaming methodologies.

## Code Evaluation (HumanEval/MBPP and variants)
- pass@k metrics; contamination concerns; execution safety practices.

## Directions for OpenEval Lab
- Adopt multi-metric hooks (calibration, robustness, fairness) as opt-in extras.
- Keep the core minimal but precise (contracts, schema, CLI); move heavy deps to extras.
- Improve reproducibility: artifact schema, open prompts/outputs, curated suites.
- Offer judge metrics with bias mitigation (balanced positions) and uncertainty reporting.
