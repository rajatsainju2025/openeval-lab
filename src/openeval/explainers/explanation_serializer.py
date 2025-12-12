"""Serialization and deserialization utilities for explanations.

This module provides tools for serializing explanations to various formats
and deserializing them back, enabling storage, transmission, and caching.
"""

import base64
import gzip
import hashlib
import json
import pickle
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, BinaryIO, Dict, List, Optional, Type, TypeVar, Union

from .types import CodeElement, CodeElementType, ExplanationResult, ExplainLevel


T = TypeVar("T")


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class SerializationFormat(str, Enum):
    """Supported serialization formats."""

    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    YAML = "yaml"
    BINARY = "binary"  # Custom compact binary format
    PROTOBUF = "protobuf"


class CompressionType(str, Enum):
    """Compression types."""

    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"


class EncodingType(str, Enum):
    """Text encoding types."""

    UTF8 = "utf-8"
    ASCII = "ascii"
    LATIN1 = "latin-1"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SerializationMetadata:
    """Metadata about serialized data."""

    format: SerializationFormat
    compression: CompressionType
    encoding: EncodingType
    version: str = "1.0"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    checksum: Optional[str] = None
    original_size: int = 0
    compressed_size: int = 0
    item_count: int = 0
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SerializedData:
    """Container for serialized data."""

    data: bytes
    metadata: SerializationMetadata

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.metadata.original_size == 0:
            return 0.0
        return 1 - (self.metadata.compressed_size / self.metadata.original_size)


@dataclass
class SerializerConfig:
    """Configuration for serialization."""

    format: SerializationFormat = SerializationFormat.JSON
    compression: CompressionType = CompressionType.NONE
    encoding: EncodingType = EncodingType.UTF8
    include_metadata: bool = True
    pretty_print: bool = False
    compression_level: int = 6  # 1-9 for gzip
    include_checksum: bool = True
    max_depth: int = 100
    custom_options: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Serializers
# =============================================================================


class Serializer(ABC):
    """Abstract base class for serializers."""

    @property
    @abstractmethod
    def format(self) -> SerializationFormat:
        """Get the serialization format."""
        pass

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes.

        Args:
            obj: Object to serialize.

        Returns:
            Serialized bytes.
        """
        pass

    @abstractmethod
    def deserialize(self, data: bytes, target_type: Optional[Type[T]] = None) -> Any:
        """Deserialize bytes to an object.

        Args:
            data: Serialized data.
            target_type: Optional target type hint.

        Returns:
            Deserialized object.
        """
        pass


class JSONSerializer(Serializer):
    """JSON serializer."""

    def __init__(self, pretty: bool = False, encoding: str = "utf-8"):
        """Initialize JSON serializer.

        Args:
            pretty: Whether to pretty-print JSON.
            encoding: Text encoding.
        """
        self.pretty = pretty
        self.encoding = encoding

    @property
    def format(self) -> SerializationFormat:
        return SerializationFormat.JSON

    def serialize(self, obj: Any) -> bytes:
        """Serialize to JSON bytes."""
        indent = 2 if self.pretty else None

        def default_handler(o):
            if hasattr(o, "to_dict"):
                return o.to_dict()
            elif hasattr(o, "__dict__"):
                return o.__dict__
            elif isinstance(o, Enum):
                return o.value
            elif isinstance(o, datetime):
                return o.isoformat()
            elif isinstance(o, bytes):
                return base64.b64encode(o).decode("ascii")
            elif isinstance(o, set):
                return list(o)
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

        json_str = json.dumps(obj, indent=indent, default=default_handler)
        return json_str.encode(self.encoding)

    def deserialize(self, data: bytes, target_type: Optional[Type[T]] = None) -> Any:
        """Deserialize JSON bytes."""
        json_str = data.decode(self.encoding)
        result = json.loads(json_str)

        if target_type and hasattr(target_type, "from_dict"):
            return target_type.from_dict(result)
        return result


class PickleSerializer(Serializer):
    """Pickle serializer for Python objects."""

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        """Initialize pickle serializer.

        Args:
            protocol: Pickle protocol version.
        """
        self.protocol = protocol

    @property
    def format(self) -> SerializationFormat:
        return SerializationFormat.PICKLE

    def serialize(self, obj: Any) -> bytes:
        """Serialize to pickle bytes."""
        return pickle.dumps(obj, protocol=self.protocol)

    def deserialize(self, data: bytes, target_type: Optional[Type[T]] = None) -> Any:
        """Deserialize pickle bytes."""
        return pickle.loads(data)


class BinarySerializer(Serializer):
    """Custom compact binary serializer."""

    # Magic bytes for format identification
    MAGIC = b"EXPL"
    VERSION = 1

    @property
    def format(self) -> SerializationFormat:
        return SerializationFormat.BINARY

    def serialize(self, obj: Any) -> bytes:
        """Serialize to compact binary format."""
        buffer = BytesIO()

        # Write header
        buffer.write(self.MAGIC)
        buffer.write(struct.pack("B", self.VERSION))

        # Serialize the object
        self._write_value(buffer, obj)

        return buffer.getvalue()

    def deserialize(self, data: bytes, target_type: Optional[Type[T]] = None) -> Any:
        """Deserialize from binary format."""
        buffer = BytesIO(data)

        # Read and verify header
        magic = buffer.read(4)
        if magic != self.MAGIC:
            raise ValueError("Invalid binary format magic bytes")

        version = struct.unpack("B", buffer.read(1))[0]
        if version > self.VERSION:
            raise ValueError(f"Unsupported binary format version: {version}")

        return self._read_value(buffer)

    def _write_value(self, buffer: BinaryIO, value: Any) -> None:
        """Write a value to the buffer."""
        if value is None:
            buffer.write(b"\x00")  # Type: None
        elif isinstance(value, bool):
            buffer.write(b"\x01")  # Type: Bool
            buffer.write(b"\x01" if value else b"\x00")
        elif isinstance(value, int):
            buffer.write(b"\x02")  # Type: Int
            encoded = value.to_bytes(8, "little", signed=True)
            buffer.write(encoded)
        elif isinstance(value, float):
            buffer.write(b"\x03")  # Type: Float
            buffer.write(struct.pack("d", value))
        elif isinstance(value, str):
            buffer.write(b"\x04")  # Type: String
            encoded = value.encode("utf-8")
            buffer.write(struct.pack("I", len(encoded)))
            buffer.write(encoded)
        elif isinstance(value, bytes):
            buffer.write(b"\x05")  # Type: Bytes
            buffer.write(struct.pack("I", len(value)))
            buffer.write(value)
        elif isinstance(value, list):
            buffer.write(b"\x06")  # Type: List
            buffer.write(struct.pack("I", len(value)))
            for item in value:
                self._write_value(buffer, item)
        elif isinstance(value, dict):
            buffer.write(b"\x07")  # Type: Dict
            buffer.write(struct.pack("I", len(value)))
            for k, v in value.items():
                self._write_value(buffer, k)
                self._write_value(buffer, v)
        elif isinstance(value, Enum):
            buffer.write(b"\x08")  # Type: Enum
            self._write_value(buffer, value.value)
        elif hasattr(value, "to_dict"):
            buffer.write(b"\x09")  # Type: Object with to_dict
            self._write_value(buffer, value.to_dict())
        elif hasattr(value, "__dict__"):
            buffer.write(b"\x09")  # Type: Object
            self._write_value(buffer, value.__dict__)
        else:
            raise ValueError(f"Cannot serialize type: {type(value)}")

    def _read_value(self, buffer: BinaryIO) -> Any:
        """Read a value from the buffer."""
        type_byte = buffer.read(1)
        if not type_byte:
            raise ValueError("Unexpected end of data")

        type_code = type_byte[0]

        if type_code == 0:  # None
            return None
        elif type_code == 1:  # Bool
            return buffer.read(1)[0] != 0
        elif type_code == 2:  # Int
            return int.from_bytes(buffer.read(8), "little", signed=True)
        elif type_code == 3:  # Float
            return struct.unpack("d", buffer.read(8))[0]
        elif type_code == 4:  # String
            length = struct.unpack("I", buffer.read(4))[0]
            return buffer.read(length).decode("utf-8")
        elif type_code == 5:  # Bytes
            length = struct.unpack("I", buffer.read(4))[0]
            return buffer.read(length)
        elif type_code == 6:  # List
            length = struct.unpack("I", buffer.read(4))[0]
            return [self._read_value(buffer) for _ in range(length)]
        elif type_code == 7:  # Dict
            length = struct.unpack("I", buffer.read(4))[0]
            result = {}
            for _ in range(length):
                key = self._read_value(buffer)
                value = self._read_value(buffer)
                result[key] = value
            return result
        elif type_code == 8:  # Enum value
            return self._read_value(buffer)
        elif type_code == 9:  # Object
            return self._read_value(buffer)
        else:
            raise ValueError(f"Unknown type code: {type_code}")


# =============================================================================
# Compression
# =============================================================================


class Compressor(ABC):
    """Abstract base class for compressors."""

    @property
    @abstractmethod
    def compression_type(self) -> CompressionType:
        """Get the compression type."""
        pass

    @abstractmethod
    def compress(self, data: bytes, level: int = 6) -> bytes:
        """Compress data.

        Args:
            data: Data to compress.
            level: Compression level (1-9).

        Returns:
            Compressed bytes.
        """
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data.

        Args:
            data: Compressed data.

        Returns:
            Decompressed bytes.
        """
        pass


class NoCompressor(Compressor):
    """No-op compressor."""

    @property
    def compression_type(self) -> CompressionType:
        return CompressionType.NONE

    def compress(self, data: bytes, level: int = 6) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data


class GzipCompressor(Compressor):
    """Gzip compressor."""

    @property
    def compression_type(self) -> CompressionType:
        return CompressionType.GZIP

    def compress(self, data: bytes, level: int = 6) -> bytes:
        return gzip.compress(data, compresslevel=level)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class ZlibCompressor(Compressor):
    """Zlib compressor."""

    @property
    def compression_type(self) -> CompressionType:
        return CompressionType.ZLIB

    def compress(self, data: bytes, level: int = 6) -> bytes:
        import zlib

        return zlib.compress(data, level)

    def decompress(self, data: bytes) -> bytes:
        import zlib

        return zlib.decompress(data)


# =============================================================================
# Explanation Serializer
# =============================================================================


class ExplanationSerializer:
    """Serializes and deserializes ExplanationResult objects."""

    def __init__(self, config: Optional[SerializerConfig] = None):
        """Initialize serializer.

        Args:
            config: Optional serialization configuration.
        """
        self.config = config or SerializerConfig()
        self._serializers: Dict[SerializationFormat, Serializer] = {
            SerializationFormat.JSON: JSONSerializer(
                pretty=self.config.pretty_print, encoding=self.config.encoding.value
            ),
            SerializationFormat.PICKLE: PickleSerializer(),
            SerializationFormat.BINARY: BinarySerializer(),
        }
        self._compressors: Dict[CompressionType, Compressor] = {
            CompressionType.NONE: NoCompressor(),
            CompressionType.GZIP: GzipCompressor(),
            CompressionType.ZLIB: ZlibCompressor(),
        }

    def serialize(
        self,
        explanation: ExplanationResult,
        format: Optional[SerializationFormat] = None,
        compression: Optional[CompressionType] = None,
    ) -> SerializedData:
        """Serialize an explanation.

        Args:
            explanation: Explanation to serialize.
            format: Optional format override.
            compression: Optional compression override.

        Returns:
            SerializedData with bytes and metadata.
        """
        fmt = format or self.config.format
        comp = compression or self.config.compression

        # Convert to serializable dict
        data_dict = self._to_serializable(explanation)

        # Serialize
        serializer = self._serializers.get(fmt)
        if not serializer:
            raise ValueError(f"Unsupported format: {fmt}")

        raw_bytes = serializer.serialize(data_dict)
        original_size = len(raw_bytes)

        # Compress
        compressor = self._compressors.get(comp, NoCompressor())
        compressed_bytes = compressor.compress(raw_bytes, self.config.compression_level)

        # Calculate checksum
        checksum = None
        if self.config.include_checksum:
            checksum = hashlib.sha256(compressed_bytes).hexdigest()

        metadata = SerializationMetadata(
            format=fmt,
            compression=comp,
            encoding=self.config.encoding,
            original_size=original_size,
            compressed_size=len(compressed_bytes),
            item_count=1,
            checksum=checksum,
        )

        return SerializedData(data=compressed_bytes, metadata=metadata)

    def deserialize(
        self,
        data: Union[bytes, SerializedData],
        format: Optional[SerializationFormat] = None,
        compression: Optional[CompressionType] = None,
    ) -> ExplanationResult:
        """Deserialize an explanation.

        Args:
            data: Serialized data (bytes or SerializedData).
            format: Format hint (required if data is bytes).
            compression: Compression hint (required if data is bytes).

        Returns:
            Deserialized ExplanationResult.
        """
        if isinstance(data, SerializedData):
            raw_bytes = data.data
            fmt = data.metadata.format
            comp = data.metadata.compression

            # Verify checksum
            if data.metadata.checksum:
                checksum = hashlib.sha256(raw_bytes).hexdigest()
                if checksum != data.metadata.checksum:
                    raise ValueError("Checksum mismatch - data may be corrupted")
        else:
            raw_bytes = data
            fmt = format or self.config.format
            comp = compression or self.config.compression

        # Decompress
        compressor = self._compressors.get(comp, NoCompressor())
        decompressed = compressor.decompress(raw_bytes)

        # Deserialize
        serializer = self._serializers.get(fmt)
        if not serializer:
            raise ValueError(f"Unsupported format: {fmt}")

        data_dict = serializer.deserialize(decompressed)
        return self._from_serializable(data_dict)

    def serialize_batch(self, explanations: List[ExplanationResult]) -> SerializedData:
        """Serialize multiple explanations.

        Args:
            explanations: List of explanations.

        Returns:
            SerializedData containing all explanations.
        """
        data_list = [self._to_serializable(e) for e in explanations]

        serializer = self._serializers.get(self.config.format)
        if not serializer:
            raise ValueError(f"Unsupported format: {self.config.format}")

        raw_bytes = serializer.serialize(data_list)
        original_size = len(raw_bytes)

        compressor = self._compressors.get(self.config.compression, NoCompressor())
        compressed_bytes = compressor.compress(raw_bytes, self.config.compression_level)

        checksum = None
        if self.config.include_checksum:
            checksum = hashlib.sha256(compressed_bytes).hexdigest()

        metadata = SerializationMetadata(
            format=self.config.format,
            compression=self.config.compression,
            encoding=self.config.encoding,
            original_size=original_size,
            compressed_size=len(compressed_bytes),
            item_count=len(explanations),
            checksum=checksum,
        )

        return SerializedData(data=compressed_bytes, metadata=metadata)

    def deserialize_batch(self, data: Union[bytes, SerializedData]) -> List[ExplanationResult]:
        """Deserialize multiple explanations.

        Args:
            data: Serialized batch data.

        Returns:
            List of ExplanationResult objects.
        """
        if isinstance(data, SerializedData):
            raw_bytes = data.data
            fmt = data.metadata.format
            comp = data.metadata.compression
        else:
            raw_bytes = data
            fmt = self.config.format
            comp = self.config.compression

        compressor = self._compressors.get(comp, NoCompressor())
        decompressed = compressor.decompress(raw_bytes)

        serializer = self._serializers.get(fmt)
        if not serializer:
            raise ValueError(f"Unsupported format: {fmt}")

        data_list = serializer.deserialize(decompressed)
        return [self._from_serializable(d) for d in data_list]

    def _to_serializable(self, explanation: ExplanationResult) -> Dict[str, Any]:
        """Convert ExplanationResult to serializable dict."""
        return {
            "element": {
                "type": explanation.element.type.value,
                "name": explanation.element.name,
                "source_code": explanation.element.source_code,
                "line_start": explanation.element.line_start,
                "line_end": explanation.element.line_end,
                "docstring": explanation.element.docstring,
                "metadata": explanation.element.metadata,
            },
            "explanation": explanation.explanation,
            "level": explanation.level.value,
            "confidence": explanation.confidence,
            "analysis_metadata": explanation.analysis_metadata,
            "timestamp": explanation.timestamp,
            "model_used": explanation.model_used,
        }

    def _from_serializable(self, data: Dict[str, Any]) -> ExplanationResult:
        """Convert dict back to ExplanationResult."""
        element_data = data["element"]
        element = CodeElement(
            type=CodeElementType(element_data["type"]),
            name=element_data["name"],
            source_code=element_data["source_code"],
            line_start=element_data.get("line_start", 0),
            line_end=element_data.get("line_end", 0),
            docstring=element_data.get("docstring"),
            metadata=element_data.get("metadata", {}),
        )

        return ExplanationResult(
            element=element,
            explanation=data["explanation"],
            level=ExplainLevel(data["level"]),
            confidence=data.get("confidence", 0.0),
            analysis_metadata=data.get("analysis_metadata", {}),
            timestamp=data.get("timestamp", ""),
            model_used=data.get("model_used", ""),
        )


# =============================================================================
# File I/O
# =============================================================================


class FileSerializer:
    """File-based serialization utilities."""

    def __init__(self, serializer: Optional[ExplanationSerializer] = None):
        """Initialize file serializer.

        Args:
            serializer: Optional base serializer.
        """
        self.serializer = serializer or ExplanationSerializer()

    def save(
        self,
        explanation: ExplanationResult,
        file_path: str,
        include_metadata: bool = True,
    ) -> None:
        """Save explanation to file.

        Args:
            explanation: Explanation to save.
            file_path: Output file path.
            include_metadata: Whether to include metadata.
        """
        serialized = self.serializer.serialize(explanation)

        with open(file_path, "wb") as f:
            if include_metadata:
                # Write metadata length and metadata
                metadata_bytes = json.dumps(
                    {
                        "format": serialized.metadata.format.value,
                        "compression": serialized.metadata.compression.value,
                        "encoding": serialized.metadata.encoding.value,
                        "version": serialized.metadata.version,
                        "checksum": serialized.metadata.checksum,
                        "original_size": serialized.metadata.original_size,
                        "compressed_size": serialized.metadata.compressed_size,
                    }
                ).encode("utf-8")
                f.write(struct.pack("I", len(metadata_bytes)))
                f.write(metadata_bytes)

            f.write(serialized.data)

    def load(self, file_path: str) -> ExplanationResult:
        """Load explanation from file.

        Args:
            file_path: Input file path.

        Returns:
            Loaded ExplanationResult.
        """
        with open(file_path, "rb") as f:
            # Try to read metadata
            try:
                metadata_length = struct.unpack("I", f.read(4))[0]
                if metadata_length < 1000:  # Reasonable metadata size
                    metadata_bytes = f.read(metadata_length)
                    metadata_dict = json.loads(metadata_bytes.decode("utf-8"))
                    format_type = SerializationFormat(metadata_dict["format"])
                    compression = CompressionType(metadata_dict["compression"])
                    data = f.read()
                else:
                    # No metadata header, rewind and read all
                    f.seek(0)
                    data = f.read()
                    format_type = self.serializer.config.format
                    compression = self.serializer.config.compression
            except Exception:
                f.seek(0)
                data = f.read()
                format_type = self.serializer.config.format
                compression = self.serializer.config.compression

        return self.serializer.deserialize(data, format=format_type, compression=compression)

    def save_batch(
        self,
        explanations: List[ExplanationResult],
        file_path: str,
    ) -> None:
        """Save multiple explanations to file.

        Args:
            explanations: Explanations to save.
            file_path: Output file path.
        """
        serialized = self.serializer.serialize_batch(explanations)

        with open(file_path, "wb") as f:
            metadata_bytes = json.dumps(
                {
                    "format": serialized.metadata.format.value,
                    "compression": serialized.metadata.compression.value,
                    "encoding": serialized.metadata.encoding.value,
                    "item_count": serialized.metadata.item_count,
                    "checksum": serialized.metadata.checksum,
                }
            ).encode("utf-8")
            f.write(struct.pack("I", len(metadata_bytes)))
            f.write(metadata_bytes)
            f.write(serialized.data)

    def load_batch(self, file_path: str) -> List[ExplanationResult]:
        """Load multiple explanations from file.

        Args:
            file_path: Input file path.

        Returns:
            List of ExplanationResult objects.
        """
        with open(file_path, "rb") as f:
            metadata_length = struct.unpack("I", f.read(4))[0]
            metadata_bytes = f.read(metadata_length)
            metadata_dict = json.loads(metadata_bytes.decode("utf-8"))

            serialized = SerializedData(
                data=f.read(),
                metadata=SerializationMetadata(
                    format=SerializationFormat(metadata_dict["format"]),
                    compression=CompressionType(metadata_dict["compression"]),
                    encoding=EncodingType(metadata_dict.get("encoding", "utf-8")),
                    item_count=metadata_dict.get("item_count", 0),
                    checksum=metadata_dict.get("checksum"),
                ),
            )

        return self.serializer.deserialize_batch(serialized)


# =============================================================================
# Stream Serialization
# =============================================================================


class StreamSerializer:
    """Streaming serialization for large datasets."""

    def __init__(
        self,
        serializer: Optional[ExplanationSerializer] = None,
        chunk_size: int = 100,
    ):
        """Initialize stream serializer.

        Args:
            serializer: Base serializer.
            chunk_size: Items per chunk.
        """
        self.serializer = serializer or ExplanationSerializer()
        self.chunk_size = chunk_size

    def write_stream(
        self,
        explanations: List[ExplanationResult],
        output: BinaryIO,
    ) -> int:
        """Write explanations to a stream.

        Args:
            explanations: Explanations to write.
            output: Output stream.

        Returns:
            Number of items written.
        """
        count = 0

        # Write header
        output.write(b"STRM")  # Magic
        output.write(struct.pack("I", len(explanations)))  # Total count

        # Write in chunks
        for i in range(0, len(explanations), self.chunk_size):
            chunk = explanations[i : i + self.chunk_size]
            serialized = self.serializer.serialize_batch(chunk)

            # Write chunk size and data
            output.write(struct.pack("I", len(serialized.data)))
            output.write(serialized.data)
            count += len(chunk)

        return count

    def read_stream(self, input_stream: BinaryIO) -> List[ExplanationResult]:
        """Read explanations from a stream.

        Args:
            input_stream: Input stream.

        Returns:
            List of ExplanationResult objects.
        """
        # Read header
        magic = input_stream.read(4)
        if magic != b"STRM":
            raise ValueError("Invalid stream format")

        total_count = struct.unpack("I", input_stream.read(4))[0]
        results = []

        while len(results) < total_count:
            # Read chunk
            chunk_size_bytes = input_stream.read(4)
            if not chunk_size_bytes:
                break

            chunk_size = struct.unpack("I", chunk_size_bytes)[0]
            chunk_data = input_stream.read(chunk_size)

            chunk_explanations = self.serializer.deserialize_batch(chunk_data)
            results.extend(chunk_explanations)

        return results


# =============================================================================
# Global Instance Management
# =============================================================================


_global_serializer: Optional[ExplanationSerializer] = None


def get_serializer() -> ExplanationSerializer:
    """Get the global serializer instance."""
    global _global_serializer
    if _global_serializer is None:
        _global_serializer = ExplanationSerializer()
    return _global_serializer


def reset_serializer() -> None:
    """Reset the global serializer."""
    global _global_serializer
    _global_serializer = None


def create_serializer(config: Optional[SerializerConfig] = None) -> ExplanationSerializer:
    """Create a new serializer with optional config."""
    return ExplanationSerializer(config=config)


def serialize_explanation(
    explanation: ExplanationResult,
    format: SerializationFormat = SerializationFormat.JSON,
) -> bytes:
    """Convenience function to serialize an explanation."""
    return get_serializer().serialize(explanation, format=format).data


def deserialize_explanation(
    data: bytes,
    format: SerializationFormat = SerializationFormat.JSON,
) -> ExplanationResult:
    """Convenience function to deserialize an explanation."""
    return get_serializer().deserialize(data, format=format)


def save_explanation(explanation: ExplanationResult, file_path: str) -> None:
    """Convenience function to save an explanation to file."""
    FileSerializer().save(explanation, file_path)


def load_explanation(file_path: str) -> ExplanationResult:
    """Convenience function to load an explanation from file."""
    return FileSerializer().load(file_path)
