# OpenEval Lab

An extensible, reproducible evaluation framework for LLMs and multimodal agents.

- Plugin-based tasks, datasets, adapters, and metrics
- JSON/YAML experiment specs with version pinning
- Deterministic seeding and artifact logging
- Simple CLI and minimal web dashboard
- MIT License

Roadmap:
- [ ] Core abstractions (Task, Dataset, Adapter, Metric)
- [ ] CLI: `openeval run <spec>`
- [ ] JSON Schema for spec validation
- [ ] Built-in tasks (QA, code, web), metrics (accuracy, BLEU, BERTScore), adapters (OpenAI, local)
- [ ] Dataset loaders (HF Datasets, local)
- [ ] Reproducibility: lockfiles, hashes
- [ ] Web dashboard (FastAPI + SvelteKit or Streamlit-lite)
