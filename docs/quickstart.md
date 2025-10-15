# Quickstart

This quickstart gets you from clone to first evaluation in minutes.

## Prerequisites
- Python 3.9+
- pip

## Steps

```bash
# Install dev deps
pip install -e '.[dev]'

# Run a QA demo evaluation
openeval run examples/qa_spec.json --records --artifacts runs/qa-demo

# View results
ls runs/qa-demo
```

Alternatively, use the Makefile shortcut:

```bash
make quickstart
```

If you run into issues, try:

```bash
openeval doctor --json
```
