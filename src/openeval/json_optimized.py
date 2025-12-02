"""
High-Performance JSON Operations

Optimized JSON serialization/deserialization with streaming support,
custom encoders, and memory-efficient processing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union, Iterator

# Try to import orjson for better performance
try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False
    orjson = None


class OptimizedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder with optimizations for common types."""

    def default(self, o: Any) -> Any:
        # Handle numpy types efficiently
        if hasattr(o, "item"):  # numpy scalar
            return o.item()
        elif hasattr(o, "tolist"):  # numpy array
            return o.tolist()

        # Handle dataclasses
        if hasattr(o, "__dataclass_fields__"):
            return {field.name: getattr(o, field.name) for field in o.__dataclass_fields__.values()}

        # Handle sets
        if isinstance(o, set):
            return list(o)

        # Handle bytes
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")

        return super().default(o)


def dumps_fast(obj: Any, ensure_ascii: bool = False, separators: tuple = (",", ":")) -> str:
    """Fast JSON serialization with optimizations."""
    if HAS_ORJSON and orjson is not None:
        # Use orjson if available (fastest)
        return orjson.dumps(obj).decode("utf-8")
    else:
        # Fallback to standard json with optimizations
        return json.dumps(
            obj, cls=OptimizedJSONEncoder, ensure_ascii=ensure_ascii, separators=separators
        )


def loads_fast(s: Union[str, bytes]) -> Any:
    """Fast JSON deserialization."""
    if HAS_ORJSON and orjson is not None:
        return orjson.loads(s)
    else:
        return json.loads(s)


def dump_file_fast(obj: Any, file_path: str, ensure_ascii: bool = False) -> None:
    """Fast JSON file writing."""
    if HAS_ORJSON and orjson is not None:
        with open(file_path, "wb") as f:
            f.write(orjson.dumps(obj))
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, cls=OptimizedJSONEncoder, ensure_ascii=ensure_ascii)


def load_file_fast(file_path: str) -> Any:
    """Fast JSON file loading."""
    if HAS_ORJSON and orjson is not None:
        with open(file_path, "rb") as f:
            return orjson.loads(f.read())
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


def stream_json_array(file_path: str, chunk_size: int = 8192) -> Iterator[Dict[str, Any]]:
    """Stream JSON array items one by one for memory efficiency."""
    with open(file_path, "r", encoding="utf-8", buffering=chunk_size) as f:
        decoder = json.JSONDecoder()
        buffer = ""
        bracket_count = 0
        in_string = False
        escape_next = False
        current_object = ""

        for chunk in iter(lambda: f.read(chunk_size), ""):
            buffer += chunk

            i = 0
            while i < len(buffer):
                char = buffer[i]

                if escape_next:
                    escape_next = False
                    current_object += char
                    i += 1
                    continue

                if char == "\\":
                    escape_next = True
                    current_object += char
                    i += 1
                    continue

                if char == '"':
                    in_string = not in_string
                    current_object += char
                    i += 1
                    continue

                if in_string:
                    current_object += char
                    i += 1
                    continue

                if char == "{":
                    bracket_count += 1
                    current_object += char
                elif char == "}":
                    bracket_count -= 1
                    current_object += char

                    if bracket_count == 0:
                        # Complete object found
                        try:
                            obj = decoder.decode(current_object.strip())
                            yield obj
                        except json.JSONDecodeError:
                            pass  # Skip malformed objects
                        current_object = ""
                elif bracket_count > 0:
                    current_object += char

                i += 1

            # Keep unprocessed part
            if bracket_count > 0:
                buffer = current_object
                current_object = ""
            else:
                buffer = ""


def stream_jsonl(file_path: str, chunk_size: int = 8192) -> Iterator[Dict[str, Any]]:
    """Stream JSONL file line by line for memory efficiency."""
    with open(file_path, "r", encoding="utf-8", buffering=chunk_size) as f:
        buffer = ""

        for chunk in iter(lambda: f.read(chunk_size), ""):
            buffer += chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if line:
                    try:
                        yield loads_fast(line)
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

        # Process final line
        if buffer.strip():
            try:
                yield loads_fast(buffer.strip())
            except json.JSONDecodeError:
                pass


def write_jsonl_streaming(objects: Iterator[Dict[str, Any]], file_path: str) -> int:
    """Write objects to JSONL file in streaming fashion."""
    count = 0

    with open(file_path, "w", encoding="utf-8", buffering=8192) as f:
        for obj in objects:
            f.write(dumps_fast(obj))
            f.write("\n")
            count += 1

            # Periodic flush for large files
            if count % 1000 == 0:
                f.flush()

    return count


def batch_process_json(
    input_path: str, output_path: str, transform_func, batch_size: int = 100, is_jsonl: bool = True
) -> int:
    """Process JSON file in batches with transformation function."""
    processed_count = 0

    # Choose appropriate reader
    if is_jsonl:
        reader = stream_jsonl(input_path)
    else:
        reader = stream_json_array(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        batch = []

        for item in reader:
            batch.append(item)

            if len(batch) >= batch_size:
                # Process batch
                transformed = [transform_func(item) for item in batch]

                # Write batch
                for transformed_item in transformed:
                    if transformed_item is not None:
                        f.write(dumps_fast(transformed_item))
                        f.write("\n")
                        processed_count += 1

                batch = []
                f.flush()

        # Process final batch
        if batch:
            transformed = [transform_func(item) for item in batch]
            for transformed_item in transformed:
                if transformed_item is not None:
                    f.write(dumps_fast(transformed_item))
                    f.write("\n")
                    processed_count += 1

    return processed_count


def merge_json_files(input_paths: List[str], output_path: str, is_jsonl: bool = True) -> int:
    """Merge multiple JSON files efficiently."""
    total_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for input_path in input_paths:
            if is_jsonl:
                reader = stream_jsonl(input_path)
            else:
                reader = stream_json_array(input_path)

            for item in reader:
                out_f.write(dumps_fast(item))
                out_f.write("\n")
                total_count += 1

                # Periodic flush
                if total_count % 1000 == 0:
                    out_f.flush()

    return total_count


def validate_json_file(file_path: str, is_jsonl: bool = True) -> Dict[str, Any]:
    """Validate JSON file and return statistics."""
    stats = {"valid_items": 0, "invalid_items": 0, "total_size_bytes": 0, "errors": []}

    try:
        if is_jsonl:
            reader = stream_jsonl(file_path)
        else:
            reader = stream_json_array(file_path)

        for item in reader:
            stats["valid_items"] += 1

            # Estimate size
            item_json = dumps_fast(item)
            stats["total_size_bytes"] += len(item_json.encode("utf-8"))

    except Exception as e:
        stats["errors"].append(str(e))
        stats["invalid_items"] += 1

    # Get file size
    import os

    if os.path.exists(file_path):
        stats["file_size_bytes"] = os.path.getsize(file_path)

    return stats


__all__ = [
    "OptimizedJSONEncoder",
    "dumps_fast",
    "loads_fast",
    "dump_file_fast",
    "load_file_fast",
    "stream_json_array",
    "stream_jsonl",
    "write_jsonl_streaming",
    "batch_process_json",
    "merge_json_files",
    "validate_json_file",
    "HAS_ORJSON",
]
