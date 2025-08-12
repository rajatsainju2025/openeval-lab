# Prompt rendering with write_out

Render prompts for a spec to debug templates and dataset formatting.

Examples:

- Preview first 5 prompts in terminal:

  ```bash
  openeval write_out examples/qa_spec.json --limit 5
  ```

- Save all prompts to JSONL:

  ```bash
  openeval write_out examples/qa_spec.json --out prompts.jsonl
  ```

Each JSONL row includes: `id`, `input`, `reference`, `prompt`.
