# OpenEval Lab Tutorial: Getting Started

This short tutorial gets you from install to your first run and the minimal web dashboard.

## 1) Install

Create a virtual environment and install in editable mode with developer tools:

1) python -m venv .venv
2) source .venv/bin/activate
3) pip install -e .[dev]

Optional extras: metrics (`.[metrics]`), OpenAI (`.[openai]`), HuggingFace datasets (`.[hf]`).

## 2) Explore the registry

List available components:

- openeval registry-list task
- openeval registry-list dataset
- openeval registry-list adapter
- openeval registry-list metric

Inspect a specific item (e.g., ROUGE-L metric):

- openeval registry-info metric rouge_l

## 3) Validate a spec

Use one of the bundled examples:

- openeval validate examples/qa_spec.json

## 4) Run your first evaluation (echo adapter)

The echo adapter is offline and free. Try a tiny QA toy dataset:

- openeval run examples/qa_spec.json --adapter echo --records --artifacts runs

Optional flags you can try:
- `--robustness-noise 0.05` to perturb inputs and record a robustness slice.
- `--calibration` to attempt calibration metadata if the adapter exposes logprobs.

Calibration notes:
- The CLI computes a simple Expected Calibration Error (ECE) using exp(mean token logprob) as a confidence proxy for short completions.
- This requires adapters that implement `generate_with_logprobs` (OpenAI-style logprobs or equivalent).
- It caps at ~512 samples for speed; results are reported under a `calibration` block in the run JSON.

Flags used:
- `--records` writes per-example outputs.
- `--artifacts runs` saves a timestamped run file in `runs/`.

## 5) View results

- ls runs/  # see timestamped JSON
- openeval web --reload  # open the dashboard (http://localhost:8000)
- Navigate to /leaderboard and click a run to view records.

## 6) Next steps

- Try summarization: `openeval run examples/sum_spec.json --records --artifacts runs`
- Install metrics: `pip install -e .[metrics]` to enable ROUGE/BERTScore/SacreBLEU.
- Use OpenAI: `pip install -e .[openai]` and set `OPENAI_API_KEY`.
- Troubleshoot: `openeval doctor` or `openeval doctor --json` for CI.
