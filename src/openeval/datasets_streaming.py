"""Streaming dataset iterators to avoid materialization.

Replaces list(dataset) with generator iteration.
"""

from typing import Iterator, Any, Dict


def stream_jsonl_dataset(file_path: str) -> Iterator[Dict[str, Any]]:
    """Stream JSONL dataset line by line without materializing."""
    import json

    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stream_csv_dataset(file_path: str) -> Iterator[Dict[str, Any]]:
    """Stream CSV dataset without materializing."""
    import csv

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def chunk_stream(iterator: Iterator[Any], chunk_size: int) -> Iterator[list]:
    """Chunk a streaming iterator into batches."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = ["stream_jsonl_dataset", "stream_csv_dataset", "chunk_stream"]
