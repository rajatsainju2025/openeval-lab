#!/usr/bin/env python3
"""
Advanced Dataset Loader for OpenEval Lab

This script provides comprehensive dataset loading capabilities for various formats
including JSON, CSV, Parquet, Hugging Face datasets, and custom formats.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
from abc import ABC, abstractmethod
import gzip
import bz2
import lzma

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    pa = None  # type: ignore
    pq = None  # type: ignore
    HAS_PYARROW = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    load_dataset = None  # type: ignore
    HAS_DATASETS = False


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self, path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """Load dataset from path."""
        pass

    @abstractmethod
    def save(self, data: List[Dict[str, Any]], path: Union[str, Path], **kwargs) -> None:
        """Save dataset to path."""
        pass

    def supports_format(self, path: Union[str, Path]) -> bool:
        """Check if loader supports the given file format."""
        return False


class JSONLoader(DatasetLoader):
    """Loader for JSON datasets."""

    def supports_format(self, path: Union[str, Path]) -> bool:
        path_str = str(path).lower()
        return path_str.endswith(('.json', '.json.gz', '.json.bz2', '.json.xz'))

    def load(self, path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """Load JSON dataset."""
        path = Path(path)

        # Open file with appropriate compression
        if path.suffix == '.gz':
            opener = gzip.open
        elif path.suffix == '.bz2':
            opener = bz2.open
        elif path.suffix == '.xz':
            opener = lzma.open
        else:
            opener = open

        with opener(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            return [data]

    def save(self, data: List[Dict[str, Any]], path: Union[str, Path], **kwargs) -> None:
        """Save data as JSON."""
        path = Path(path)

        # Determine compression
        if path.suffix == '.gz':
            opener = gzip.open
        elif path.suffix == '.bz2':
            opener = bz2.open
        elif path.suffix == '.xz':
            opener = lzma.open
        else:
            opener = open

        with opener(path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class JSONLinesLoader(DatasetLoader):
    """Loader for JSON Lines datasets."""

    def supports_format(self, path: Union[str, Path]) -> bool:
        path_str = str(path).lower()
        return path_str.endswith(('.jsonl', '.jsonlines', '.jl'))

    def load(self, path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """Load JSON Lines dataset."""
        path = Path(path)
        data = []

        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue

        return data

    def save(self, data: List[Dict[str, Any]], path: Union[str, Path], **kwargs) -> None:
        """Save data as JSON Lines."""
        path = Path(path)

        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


class CSVLoader(DatasetLoader):
    """Loader for CSV datasets."""

    def supports_format(self, path: Union[str, Path]) -> bool:
        if not HAS_PANDAS:
            return False
        path_str = str(path).lower()
        return path_str.endswith(('.csv', '.tsv', '.txt'))

    def load(self, path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """Load CSV dataset."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for CSV loading")

        path = Path(path)

        # Determine separator
        if path.suffix == '.tsv':
            sep = '\t'
        else:
            sep = kwargs.get('sep', ',')

        df = pd.read_csv(path, sep=sep, **kwargs)
        return df.to_dict('records')

    def save(self, data: List[Dict[str, Any]], path: Union[str, Path], **kwargs) -> None:
        """Save data as CSV."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for CSV saving")

        df = pd.DataFrame(data)
        path = Path(path)

        # Determine separator
        if path.suffix == '.tsv':
            sep = '\t'
        else:
            sep = kwargs.get('sep', ',')

        df.to_csv(path, sep=sep, index=False, **kwargs)


class ParquetLoader(DatasetLoader):
    """Loader for Parquet datasets."""

    def supports_format(self, path: Union[str, Path]) -> bool:
        if not HAS_PYARROW:
            return False
        path_str = str(path).lower()
        return path_str.endswith('.parquet')

    def load(self, path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """Load Parquet dataset."""
        if not HAS_PYARROW:
            raise ImportError("pyarrow required for Parquet loading")

        table = pq.read_table(path, **kwargs)
        df = table.to_pandas()
        return df.to_dict('records')

    def save(self, data: List[Dict[str, Any]], path: Union[str, Path], **kwargs) -> None:
        """Save data as Parquet."""
        if not HAS_PYARROW:
            raise ImportError("pyarrow required for Parquet saving")

        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, **kwargs)


class HuggingFaceLoader(DatasetLoader):
    """Loader for Hugging Face datasets."""

    def supports_format(self, path: Union[str, Path]) -> bool:
        if not HAS_DATASETS:
            return False
        path_str = str(path)
        # Support dataset names like "glue", "squad", etc.
        return '/' not in path_str and '.' not in path_str

    def load(self, path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """Load Hugging Face dataset."""
        if not HAS_DATASETS:
            raise ImportError("datasets library required for Hugging Face dataset loading")

        dataset_name = str(path)
        split = kwargs.get('split', 'train')

        dataset = load_dataset(dataset_name, **kwargs)
        if isinstance(dataset, dict):
            dataset = dataset[split]

        return [dict(item) for item in dataset]

    def save(self, data: List[Dict[str, Any]], path: Union[str, Path], **kwargs) -> None:
        """Save data (not typically used for HF datasets)."""
        raise NotImplementedError("Saving to Hugging Face Hub not implemented")


class DatasetLoaderRegistry:
    """Registry for dataset loaders."""

    def __init__(self):
        self.loaders: List[DatasetLoader] = [
            JSONLoader(),
            JSONLinesLoader(),
            CSVLoader(),
            ParquetLoader(),
            HuggingFaceLoader(),
        ]

    def get_loader(self, path: Union[str, Path]) -> Optional[DatasetLoader]:
        """Get appropriate loader for the given path."""
        for loader in self.loaders:
            if loader.supports_format(path):
                return loader
        return None

    def load_dataset(
        self,
        path: Union[str, Path],
        loader_type: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load dataset using appropriate loader.

        Args:
            path: Path to dataset file or Hugging Face dataset name
            loader_type: Force specific loader type
            **kwargs: Additional arguments for the loader

        Returns:
            List of dataset examples
        """
        if loader_type:
            # Find loader by type
            loader_map = {
                'json': JSONLoader(),
                'jsonl': JSONLinesLoader(),
                'csv': CSVLoader(),
                'parquet': ParquetLoader(),
                'huggingface': HuggingFaceLoader(),
            }
            loader = loader_map.get(loader_type.lower())
            if not loader:
                raise ValueError(f"Unknown loader type: {loader_type}")
        else:
            # Auto-detect loader
            loader = self.get_loader(path)
            if not loader:
                raise ValueError(f"No suitable loader found for {path}")

        return loader.load(path, **kwargs)

    def save_dataset(
        self,
        data: List[Dict[str, Any]],
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Save dataset using appropriate loader.

        Args:
            data: Dataset to save
            path: Output path
            format: Force output format
            **kwargs: Additional arguments for the loader
        """
        if format:
            loader_map = {
                'json': JSONLoader(),
                'jsonl': JSONLinesLoader(),
                'csv': CSVLoader(),
                'parquet': ParquetLoader(),
            }
            loader = loader_map.get(format.lower())
            if not loader:
                raise ValueError(f"Unknown format: {format}")
        else:
            # Auto-detect from extension
            loader = self.get_loader(path)
            if not loader:
                # Default to JSON
                loader = JSONLoader()

        loader.save(data, path, **kwargs)


def load_dataset_auto(
    path: Union[str, Path],
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Automatically load dataset with format detection.

    Args:
        path: Path to dataset
        **kwargs: Additional arguments

    Returns:
        Loaded dataset
    """
    registry = DatasetLoaderRegistry()
    return registry.load_dataset(path, **kwargs)


def save_dataset_auto(
    data: List[Dict[str, Any]],
    path: Union[str, Path],
    **kwargs
) -> None:
    """
    Automatically save dataset with format detection.

    Args:
        data: Dataset to save
        path: Output path
        **kwargs: Additional arguments
    """
    registry = DatasetLoaderRegistry()
    registry.save_dataset(data, path, **kwargs)


def main():
    """Main entry point for dataset loading."""
    import argparse

    parser = argparse.ArgumentParser(description="Load and convert datasets")
    parser.add_argument("input", help="Input dataset path")
    parser.add_argument("-o", "--output", help="Output path (for conversion)")
    parser.add_argument("-f", "--format", help="Output format")
    parser.add_argument("-l", "--loader", help="Force specific loader")
    parser.add_argument("--list-loaders", action="store_true", help="List available loaders")

    args = parser.parse_args()

    registry = DatasetLoaderRegistry()

    if args.list_loaders:
        print("Available loaders:")
        for loader in registry.loaders:
            print(f"  - {loader.__class__.__name__}")
        return

    try:
        # Load dataset
        print(f"Loading dataset from {args.input}...")
        data = registry.load_dataset(args.input, loader_type=args.loader)

        print(f"Loaded {len(data)} examples")

        if len(data) > 0:
            print(f"Sample keys: {list(data[0].keys())}")

        # Save if output specified
        if args.output:
            print(f"Saving to {args.output}...")
            registry.save_dataset(data, args.output, format=args.format)
            print("Dataset saved successfully")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
