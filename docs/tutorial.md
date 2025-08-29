# OpenEval Lab Tutorial: Getting Started

This short tutorial helps you run your first evaluation and explore the registry and CLI tools.

## 1. Install

Create a virtual environment and install the package in editable mode:

1) python -m venv .venv
2) source .venv/bin/activate
3) pip install -e .[dev,metrics]

Optional extras: `openai`, `hf`.

## 2. Explore the registry

List items:

- openeval registry-list task
- openeval registry-list dataset
- openeval registry-list adapter
- openeval registry-list metric

Inspect a specific item:

- openeval registry-info metric rouge_l

## 3. Validate a spec

Use one of the examples:

- openeval validate examples/qa_spec.json

## 4. Run an evaluation

Run with the echo adapter (no API keys required):

- openeval run examples/qa_toy.jsonl --adapter echo --metrics exact_match token_f1

## 5. View results

Check `runs/` and `results.json`. Start the web dashboard:

- openeval web --open

## 6. Next steps

- Try the summarization examples.
- Use the OpenAI or HF adapters for real models.
- Explore experiment tracking and compare runs.
