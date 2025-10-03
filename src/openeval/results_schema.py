from __future__ import annotations

from typing import Any, Dict, List, Tuple


# A pragmatic, permissive schema reflecting current result payloads
RESULTS_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://openeval.dev/schemas/results.json",
    "title": "OpenEval Results",
    "type": "object",
    "required": ["task", "dataset", "adapter", "metrics", "size"],
    "additionalProperties": True,
    "properties": {
        "task": {"type": "string"},
        "dataset": {"type": "string"},
        "adapter": {"type": "string"},
        "size": {"type": "integer", "minimum": 0},
        "seed": {"type": ["integer", "null"]},
        "run_name": {"type": "string"},
        "spec_path": {"type": "string"},
        "spec_hash_sha256": {"type": "string"},
        "dataset_path": {"type": "string"},
        "dataset_hash_sha256": {"type": "string"},
        "metrics": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": [
                    {"type": "number"},
                    {"type": "string"},
                    {"type": "integer"},
                    {"type": "boolean"},
                    {"type": "null"},
                ],
                "properties": {"error": {"type": "string"}},
            },
        },
        "timing": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "avg_latency_ms": {"type": ["number", "null"]},
                "total_seconds": {"type": ["number", "null"]},
                "throughput_eps": {"type": ["number", "null"]},
                "error_rate": {"type": ["number", "null"]},
                "cache_hit_rate": {"type": ["number", "null"]},
            },
        },
        "manifest": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "openeval_version": {"type": ["string", "null"]},
                "python": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {"version": {"type": "string"}, "executable": {"type": "string"}},
                },
                "platform": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "system": {"type": "string"},
                        "release": {"type": "string"},
                        "machine": {"type": "string"},
                    },
                },
            },
        },
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "id": {},
                    "input": {},
                    "reference": {},
                    "prompt": {"type": ["string", "null"]},
                    "prediction": {},
                    "latency_ms": {"type": ["number", "null"]},
                    "cached": {"type": ["boolean", "null"]},
                    "error": {"type": ["string", "null"]},
                    "trace": {"type": ["array", "null"]},
                },
            },
        },
    },
}


def validate_results_payload(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Lightweight validation without jsonschema dependency.

    Checks required top-level fields and a few key types, returning (ok, errors).
    The schema remains permissive to avoid breaking existing artifacts.
    """
    errors: List[str] = []

    # Required keys
    for key in ("task", "dataset", "adapter", "metrics", "size"):
        if key not in data:
            errors.append(f"missing required key: {key}")

    # Basic type checks (best-effort, permissive)
    if "size" in data and not isinstance(data["size"], int):
        errors.append("size must be an integer")

    if "metrics" in data and not isinstance(data["metrics"], dict):
        errors.append("metrics must be an object")

    if "timing" in data and not isinstance(data["timing"], dict):
        errors.append("timing must be an object if present")

    if "records" in data and not isinstance(data["records"], list):
        errors.append("records must be an array if present")

    return (len(errors) == 0), errors
