# SOTA Evaluation References (Aug 2025)

This living note captures relevant takeaways and links.

## HELM (Holistic Evaluation of Language Models)
- Multi-metric evaluation: accuracy, calibration, robustness, fairness, toxicity, efficiency.
- Transparent prompts and outputs for comparability.
- Standardization across scenarios.

Refs: https://arxiv.org/abs/2211.09110

## EleutherAI lm-evaluation-harness
- Config-first; prompt templating (Jinja); caching; many backends (HF, vLLM, APIs).
- Task groups, write-out, result logging, integrations (W&B, Zeno).
- Practical guidance for batching, GPUs, GGUF.

Repo: https://github.com/EleutherAI/lm-evaluation-harness

## Directions for OpenEval Lab
- Adopt multi-metric hooks (calibration, robustness, fairness) as opt-in extras.
- Keep simple core with strong registry and specs; lean on extras for heavy deps.
- Improve reproducibility: artifact schema, open prompts/outputs, curated suites.