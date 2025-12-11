"""Streaming explainer for real-time explanation generation.

This module provides streaming explanation generation that yields
explanation chunks in real-time, suitable for long explanations
and interactive UIs.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
)

from .types import CodeElement, ExplainLevel


class StreamEventType(Enum):
    """Types of streaming events."""

    START = "start"
    CHUNK = "chunk"
    PROGRESS = "progress"
    METADATA = "metadata"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StreamChunk:
    """A chunk of streamed explanation content."""

    content: str
    event_type: StreamEventType = StreamEventType.CHUNK
    index: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        """Check if this is the final chunk."""
        return self.event_type == StreamEventType.COMPLETE


@dataclass
class StreamProgress:
    """Progress information for streaming."""

    current: int
    total: int
    percentage: float
    estimated_remaining_ms: Optional[float] = None
    message: str = ""

    @classmethod
    def from_fraction(cls, current: int, total: int, message: str = "") -> "StreamProgress":
        """Create progress from current/total fraction."""
        percentage = (current / total * 100) if total > 0 else 0.0
        return cls(
            current=current,
            total=total,
            percentage=percentage,
            message=message,
        )


@dataclass
class StreamResult:
    """Complete result from streaming explanation."""

    content: str
    chunks: List[StreamChunk] = field(default_factory=list)
    total_time_ms: float = 0.0
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Estimate tokens per second."""
        if self.total_time_ms <= 0:
            return 0.0
        # Rough estimate: ~4 chars per token
        estimated_tokens = len(self.content) / 4
        return estimated_tokens / (self.total_time_ms / 1000)


class StreamingExplainer(ABC):
    """Abstract base class for streaming explainers."""

    @abstractmethod
    def stream(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> Iterator[StreamChunk]:
        """Stream explanation chunks synchronously.

        Args:
            element: Code element to explain.
            level: Explanation detail level.

        Yields:
            StreamChunk objects with explanation content.
        """
        pass

    async def stream_async(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream explanation chunks asynchronously.

        Args:
            element: Code element to explain.
            level: Explanation detail level.

        Yields:
            StreamChunk objects with explanation content.
        """
        # Default implementation wraps sync stream
        for chunk in self.stream(element, level):
            yield chunk

    def collect(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> StreamResult:
        """Collect all streamed chunks into a complete result.

        Args:
            element: Code element to explain.
            level: Explanation detail level.

        Returns:
            StreamResult with complete explanation.
        """
        start_time = time.perf_counter()
        chunks = []
        content_parts = []

        for chunk in self.stream(element, level):
            chunks.append(chunk)
            if chunk.event_type == StreamEventType.CHUNK:
                content_parts.append(chunk.content)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return StreamResult(
            content="".join(content_parts),
            chunks=chunks,
            total_time_ms=elapsed_ms,
            chunk_count=len(chunks),
        )

    async def collect_async(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> StreamResult:
        """Collect all streamed chunks asynchronously.

        Args:
            element: Code element to explain.
            level: Explanation detail level.

        Returns:
            StreamResult with complete explanation.
        """
        start_time = time.perf_counter()
        chunks = []
        content_parts = []

        async for chunk in self.stream_async(element, level):
            chunks.append(chunk)
            if chunk.event_type == StreamEventType.CHUNK:
                content_parts.append(chunk.content)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return StreamResult(
            content="".join(content_parts),
            chunks=chunks,
            total_time_ms=elapsed_ms,
            chunk_count=len(chunks),
        )


class ChunkedStreamingExplainer(StreamingExplainer):
    """Streaming explainer that chunks pre-generated content.

    Useful for simulating streaming from non-streaming sources.
    """

    def __init__(
        self,
        explainer_func: Callable[[CodeElement, ExplainLevel], str],
        chunk_size: int = 50,
        delay_ms: float = 10.0,
    ) -> None:
        """Initialize chunked streaming explainer.

        Args:
            explainer_func: Function that generates complete explanation.
            chunk_size: Size of each chunk in characters.
            delay_ms: Delay between chunks in milliseconds.
        """
        self._explainer_func = explainer_func
        self._chunk_size = chunk_size
        self._delay_ms = delay_ms

    def stream(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> Iterator[StreamChunk]:
        """Stream explanation in chunks."""
        # Generate full explanation
        full_explanation = self._explainer_func(element, level)

        # Yield start event
        yield StreamChunk(
            content="",
            event_type=StreamEventType.START,
            index=0,
            metadata={"total_length": len(full_explanation)},
        )

        # Chunk and yield
        total_chunks = (len(full_explanation) + self._chunk_size - 1) // self._chunk_size
        for i in range(0, len(full_explanation), self._chunk_size):
            chunk_index = i // self._chunk_size
            chunk_content = full_explanation[i : i + self._chunk_size]

            # Yield progress
            yield StreamChunk(
                content="",
                event_type=StreamEventType.PROGRESS,
                index=chunk_index,
                metadata={
                    "progress": StreamProgress.from_fraction(
                        chunk_index + 1,
                        total_chunks,
                        f"Generating chunk {chunk_index + 1}/{total_chunks}",
                    ).__dict__
                },
            )

            # Yield content chunk
            yield StreamChunk(
                content=chunk_content,
                event_type=StreamEventType.CHUNK,
                index=chunk_index,
            )

            # Simulate delay
            if self._delay_ms > 0:
                time.sleep(self._delay_ms / 1000)

        # Yield complete event
        yield StreamChunk(
            content="",
            event_type=StreamEventType.COMPLETE,
            index=total_chunks,
            metadata={"total_chunks": total_chunks},
        )

    async def stream_async(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream explanation in chunks asynchronously."""
        # Generate full explanation
        full_explanation = self._explainer_func(element, level)

        # Yield start event
        yield StreamChunk(
            content="",
            event_type=StreamEventType.START,
            index=0,
            metadata={"total_length": len(full_explanation)},
        )

        # Chunk and yield
        total_chunks = (len(full_explanation) + self._chunk_size - 1) // self._chunk_size
        for i in range(0, len(full_explanation), self._chunk_size):
            chunk_index = i // self._chunk_size
            chunk_content = full_explanation[i : i + self._chunk_size]

            # Yield progress
            yield StreamChunk(
                content="",
                event_type=StreamEventType.PROGRESS,
                index=chunk_index,
                metadata={
                    "progress": StreamProgress.from_fraction(
                        chunk_index + 1,
                        total_chunks,
                        f"Generating chunk {chunk_index + 1}/{total_chunks}",
                    ).__dict__
                },
            )

            # Yield content chunk
            yield StreamChunk(
                content=chunk_content,
                event_type=StreamEventType.CHUNK,
                index=chunk_index,
            )

            # Simulate delay
            if self._delay_ms > 0:
                await asyncio.sleep(self._delay_ms / 1000)

        # Yield complete event
        yield StreamChunk(
            content="",
            event_type=StreamEventType.COMPLETE,
            index=total_chunks,
            metadata={"total_chunks": total_chunks},
        )


class WordStreamingExplainer(StreamingExplainer):
    """Streaming explainer that streams word by word.

    Provides more natural streaming for text content.
    """

    def __init__(
        self,
        explainer_func: Callable[[CodeElement, ExplainLevel], str],
        delay_ms: float = 20.0,
        include_whitespace: bool = True,
    ) -> None:
        """Initialize word streaming explainer.

        Args:
            explainer_func: Function that generates complete explanation.
            delay_ms: Delay between words in milliseconds.
            include_whitespace: Include whitespace in chunks.
        """
        self._explainer_func = explainer_func
        self._delay_ms = delay_ms
        self._include_whitespace = include_whitespace

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words and whitespace."""
        tokens = []
        current = ""

        for char in text:
            if char.isspace():
                if current:
                    tokens.append(current)
                    current = ""
                if self._include_whitespace:
                    tokens.append(char)
            else:
                current += char

        if current:
            tokens.append(current)

        return tokens

    def stream(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> Iterator[StreamChunk]:
        """Stream explanation word by word."""
        full_explanation = self._explainer_func(element, level)
        tokens = self._tokenize(full_explanation)

        yield StreamChunk(
            content="",
            event_type=StreamEventType.START,
            index=0,
            metadata={"total_tokens": len(tokens)},
        )

        for i, token in enumerate(tokens):
            yield StreamChunk(
                content=token,
                event_type=StreamEventType.CHUNK,
                index=i,
            )

            if self._delay_ms > 0 and not token.isspace():
                time.sleep(self._delay_ms / 1000)

        yield StreamChunk(
            content="",
            event_type=StreamEventType.COMPLETE,
            index=len(tokens),
            metadata={"total_tokens": len(tokens)},
        )

    async def stream_async(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream explanation word by word asynchronously."""
        full_explanation = self._explainer_func(element, level)
        tokens = self._tokenize(full_explanation)

        yield StreamChunk(
            content="",
            event_type=StreamEventType.START,
            index=0,
            metadata={"total_tokens": len(tokens)},
        )

        for i, token in enumerate(tokens):
            yield StreamChunk(
                content=token,
                event_type=StreamEventType.CHUNK,
                index=i,
            )

            if self._delay_ms > 0 and not token.isspace():
                await asyncio.sleep(self._delay_ms / 1000)

        yield StreamChunk(
            content="",
            event_type=StreamEventType.COMPLETE,
            index=len(tokens),
            metadata={"total_tokens": len(tokens)},
        )


class SectionStreamingExplainer(StreamingExplainer):
    """Streaming explainer that streams section by section.

    Streams explanation in logical sections (e.g., paragraphs).
    """

    def __init__(
        self,
        explainer_func: Callable[[CodeElement, ExplainLevel], str],
        section_delimiter: str = "\n\n",
        delay_ms: float = 100.0,
    ) -> None:
        """Initialize section streaming explainer.

        Args:
            explainer_func: Function that generates complete explanation.
            section_delimiter: Delimiter to split sections.
            delay_ms: Delay between sections in milliseconds.
        """
        self._explainer_func = explainer_func
        self._section_delimiter = section_delimiter
        self._delay_ms = delay_ms

    def stream(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> Iterator[StreamChunk]:
        """Stream explanation section by section."""
        full_explanation = self._explainer_func(element, level)
        sections = full_explanation.split(self._section_delimiter)

        yield StreamChunk(
            content="",
            event_type=StreamEventType.START,
            index=0,
            metadata={"total_sections": len(sections)},
        )

        for i, section in enumerate(sections):
            # Add delimiter back except for first section
            content = section if i == 0 else self._section_delimiter + section

            yield StreamChunk(
                content=content,
                event_type=StreamEventType.CHUNK,
                index=i,
                metadata={"section": i + 1, "total": len(sections)},
            )

            if self._delay_ms > 0 and i < len(sections) - 1:
                time.sleep(self._delay_ms / 1000)

        yield StreamChunk(
            content="",
            event_type=StreamEventType.COMPLETE,
            index=len(sections),
            metadata={"total_sections": len(sections)},
        )

    async def stream_async(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream explanation section by section asynchronously."""
        full_explanation = self._explainer_func(element, level)
        sections = full_explanation.split(self._section_delimiter)

        yield StreamChunk(
            content="",
            event_type=StreamEventType.START,
            index=0,
            metadata={"total_sections": len(sections)},
        )

        for i, section in enumerate(sections):
            content = section if i == 0 else self._section_delimiter + section

            yield StreamChunk(
                content=content,
                event_type=StreamEventType.CHUNK,
                index=i,
                metadata={"section": i + 1, "total": len(sections)},
            )

            if self._delay_ms > 0 and i < len(sections) - 1:
                await asyncio.sleep(self._delay_ms / 1000)

        yield StreamChunk(
            content="",
            event_type=StreamEventType.COMPLETE,
            index=len(sections),
            metadata={"total_sections": len(sections)},
        )


class StreamMultiplexer:
    """Multiplex multiple streaming explainers.

    Combines output from multiple explainers into a single stream.
    """

    def __init__(self, explainers: List[StreamingExplainer]) -> None:
        """Initialize multiplexer.

        Args:
            explainers: List of streaming explainers.
        """
        self._explainers = explainers

    def stream_all(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> Iterator[tuple[int, StreamChunk]]:
        """Stream from all explainers, yielding (explainer_index, chunk).

        Args:
            element: Code element to explain.
            level: Explanation detail level.

        Yields:
            Tuples of (explainer_index, StreamChunk).
        """
        for i, explainer in enumerate(self._explainers):
            for chunk in explainer.stream(element, level):
                yield (i, chunk)

    async def stream_all_async(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> AsyncIterator[tuple[int, StreamChunk]]:
        """Stream from all explainers asynchronously.

        Args:
            element: Code element to explain.
            level: Explanation detail level.

        Yields:
            Tuples of (explainer_index, StreamChunk).
        """
        for i, explainer in enumerate(self._explainers):
            async for chunk in explainer.stream_async(element, level):
                yield (i, chunk)


def stream_to_callback(
    explainer: StreamingExplainer,
    element: CodeElement,
    callback: Callable[[StreamChunk], None],
    level: ExplainLevel = ExplainLevel.SUMMARY,
) -> StreamResult:
    """Stream explanation and call callback for each chunk.

    Args:
        explainer: Streaming explainer to use.
        element: Code element to explain.
        callback: Function to call with each chunk.
        level: Explanation detail level.

    Returns:
        Complete StreamResult.
    """
    start_time = time.perf_counter()
    chunks = []
    content_parts = []

    for chunk in explainer.stream(element, level):
        callback(chunk)
        chunks.append(chunk)
        if chunk.event_type == StreamEventType.CHUNK:
            content_parts.append(chunk.content)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return StreamResult(
        content="".join(content_parts),
        chunks=chunks,
        total_time_ms=elapsed_ms,
        chunk_count=len(chunks),
    )


async def stream_to_callback_async(
    explainer: StreamingExplainer,
    element: CodeElement,
    callback: Callable[[StreamChunk], None],
    level: ExplainLevel = ExplainLevel.SUMMARY,
) -> StreamResult:
    """Stream explanation asynchronously and call callback for each chunk.

    Args:
        explainer: Streaming explainer to use.
        element: Code element to explain.
        callback: Function to call with each chunk.
        level: Explanation detail level.

    Returns:
        Complete StreamResult.
    """
    start_time = time.perf_counter()
    chunks = []
    content_parts = []

    async for chunk in explainer.stream_async(element, level):
        callback(chunk)
        chunks.append(chunk)
        if chunk.event_type == StreamEventType.CHUNK:
            content_parts.append(chunk.content)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return StreamResult(
        content="".join(content_parts),
        chunks=chunks,
        total_time_ms=elapsed_ms,
        chunk_count=len(chunks),
    )


def create_streaming_explainer(
    explainer_func: Callable[[CodeElement, ExplainLevel], str],
    mode: str = "chunked",
    **kwargs: Any,
) -> StreamingExplainer:
    """Create a streaming explainer from a regular explainer function.

    Args:
        explainer_func: Function that generates complete explanation.
        mode: Streaming mode ("chunked", "word", "section").
        **kwargs: Additional arguments for the streaming explainer.

    Returns:
        StreamingExplainer instance.
    """
    if mode == "word":
        return WordStreamingExplainer(explainer_func, **kwargs)
    elif mode == "section":
        return SectionStreamingExplainer(explainer_func, **kwargs)
    else:
        return ChunkedStreamingExplainer(explainer_func, **kwargs)
