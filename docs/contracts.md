# Core Contracts (Task, Dataset, Adapter, Metric)

This page defines the minimal contracts and invariants with concise examples.

## Task
- Input: dataset examples
- Output: prompts/inputs for adapter; parses adapter outputs into predictions
- Invariants: deterministic given seed; no side effects

## Dataset
- Iterable of examples; supports len() when finite
- Fingerprintable (provider, id, revision/sha, split); shown in manifest

## Adapter
- generate(inputs, **kwargs) -> outputs
- May support loglikelihood(inputs, choices) for MCQ tasks
- Respect concurrency, retries, timeouts; surface model name/version

## Metric
- score(predictions, references, **kwargs) -> dict of metric_name -> value
- Must be deterministic given inputs

## Example (QA task, echo adapter)

```json
{
  "task": "qa",
  "dataset": {"name": "jsonl", "path": "examples/qa_toy.jsonl"},
  "adapter": {"name": "echo"},
  "metrics": [{"name": "exact_match"}, {"name": "token_f1"}],
  "options": {"seed": 7, "records": true}
}
```

Result keys (subset):
- manifest: env, versions, git, dataset fingerprint, spec hash
- metrics: aggregate metrics with optional CI
- records: per-example inputs/outputs (when records=true)

## Error modes
- Spec validation errors: explain missing/unknown components with suggestions
- Adapter failures: record error and continue when possible; track error rate
- Metric errors: namespaced under metric; skip metric if dependency missing
