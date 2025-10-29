# Quickstart Guide

Get from zero to your first evaluation in 5 minutes! ⏱️

## Prerequisites

- **Python 3.9+** (check with `python --version`)
- **pip** for package installation
- **Git** (optional, for cloning)

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install openeval-lab
```

### Option 2: Development Install

```bash
# Clone the repository
git clone https://github.com/rajatsainju2025/openeval-lab.git
cd openeval-lab

# Install in development mode with all extras
pip install -e '.[dev]'
```

## Your First Evaluation

### 1. Verify Installation

```bash
openeval --version
openeval doctor
```

The `doctor` command checks your environment and reports any issues.

### 2. Run Example Evaluation

```bash
# Run a simple QA evaluation
openeval run examples/qa_spec.json --verbose

# Results are saved to runs/ directory by default
```

### 3. View Results

```bash
# List result files
ls runs/

# View JSON results (use jq for pretty printing)
cat runs/qa_*.json | jq .
```

### 4. Try More Examples

```bash
# Multimodal evaluation
openeval run examples/multimodal_spec.json

# Code generation evaluation
openeval run examples/code_spec.json

# With custom output path
openeval run examples/qa_spec.json --output my_results.json
```

## Understanding the Spec File

A spec file defines your evaluation. Here's a minimal example (`my_eval.json`):

```json
{
  "task": {
    "type": "qa",
    "prompt": "Answer the question: {{question}}"
  },
  "dataset": {
    "type": "jsonl",
    "path": "questions.jsonl"
  },
  "adapter": {
    "type": "echo"
  },
  "metrics": [
    {"type": "exact_match"}
  ]
}
```

Your data file (`questions.jsonl`):

```jsonl
{"question": "What is 2+2?", "answer": "4"}
{"question": "Capital of France?", "answer": "Paris"}
```

Run it:

```bash
openeval run my_eval.json --verbose
```

## Using the Makefile (Development)

If you cloned the repo, use the Makefile for common tasks:

```bash
# Install dependencies
make install

# Run tests
make test

# Run quickstart demo
make quickstart

# Format code
make format

# Lint code
make lint
```

## Exploring More Features

### Caching

Enable caching to speed up repeated evaluations:

```bash
openeval run spec.json --cache .mycache
```

### Concurrency

Adjust concurrent requests for rate limits:

```bash
openeval run spec.json --max-concurrent 5
```

### Performance Profiling

Use profiling to identify bottlenecks:

```python
from openeval import profile_time, PerformanceTimer

@profile_time
def my_evaluation():
    # Your evaluation code
    pass
```

See [PERFORMANCE.md](performance.md) for detailed profiling guide.

### Launch Web Dashboard

```bash
openeval web --port 8000 --reload
```

Visit http://localhost:8000 to view the dashboard.

## Common Issues

### Issue: Import errors

**Solution**: Ensure you installed all dependencies:

```bash
pip install -e '.[dev,openai,metrics,hf]'
```

### Issue: Cache permission errors

**Solution**: Use a custom cache directory:

```bash
openeval run spec.json --cache ~/my-cache
```

### Issue: Rate limits

**Solution**: Reduce concurrency:

```bash
openeval run spec.json --max-concurrent 2 --request-timeout 60
```

## Next Steps

- 📖 Read the [Tutorial](tutorial.md) for detailed walkthroughs
- 🏗️ Check [Architecture](../Architecture.md) to understand the framework
- 🎯 Browse [Examples](../examples/) for more spec files
- 🚀 See [Performance Guide](performance.md) for optimization tips
- 💬 Join discussions on GitHub Issues

## Quick Reference

```bash
# Core commands
openeval run <spec>              # Run evaluation
openeval validate <spec>         # Validate spec file
openeval doctor                  # Check environment
openeval version                 # Show version

# Registry commands
openeval registry list           # List available components
openeval registry info <type>    # Show component details

# Utilities
openeval schema                  # Print spec JSON schema
openeval init <file>             # Create starter spec
openeval web                     # Launch dashboard

# Help
openeval --help                  # Show all commands
openeval run --help              # Command-specific help
```

---

**Need help?** Open an issue on [GitHub](https://github.com/rajatsainju2025/openeval-lab/issues) or check the [documentation](index.md).
