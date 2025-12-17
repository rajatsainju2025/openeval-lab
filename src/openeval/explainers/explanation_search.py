"""Full-text search for code explanations.

This module provides search functionality for indexing and querying
code explanations with support for ranking and filtering.

Example:
    >>> from openeval.explainers import SearchEngine, index_explanation
    >>> engine = SearchEngine()
    >>> engine.index("doc1", "This is a function that sorts arrays")
    >>> results = engine.search("sort arrays")
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class TokenizerType(Enum):
    """Types of tokenizers."""

    WHITESPACE = "whitespace"
    WORD = "word"
    NGRAM = "ngram"
    STEMMING = "stemming"


class SearchMode(Enum):
    """Search modes."""

    ALL = "all"  # All terms must match
    ANY = "any"  # Any term can match
    PHRASE = "phrase"  # Exact phrase match
    FUZZY = "fuzzy"  # Approximate match


@dataclass
class Token:
    """A tokenized term."""

    term: str
    position: int
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexedDocument:
    """An indexed document."""

    doc_id: str
    content: str
    title: str = ""
    tokens: list[Token] = field(default_factory=list)
    term_frequencies: dict[str, int] = field(default_factory=dict)
    indexed_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    """A search result hit."""

    doc_id: str
    score: float
    title: str = ""
    snippet: str = ""
    highlights: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResults:
    """Search results container."""

    hits: list[SearchHit]
    total_hits: int
    query: str
    query_time: float
    max_score: float = 0.0
    facets: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """A search query with options."""

    query_text: str
    mode: SearchMode = SearchMode.ANY
    fields: list[str] | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    offset: int = 0
    highlight: bool = True
    min_score: float = 0.0


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> list[Token]:
        """Tokenize text into tokens."""
        pass


class WhitespaceTokenizer(Tokenizer):
    """Simple whitespace-based tokenizer."""

    def tokenize(self, text: str) -> list[Token]:
        """Tokenize by whitespace."""
        tokens = []
        position = 0
        offset = 0

        for part in text.split():
            start = text.find(part, offset)
            end = start + len(part)

            tokens.append(
                Token(
                    term=part.lower(),
                    position=position,
                    start_offset=start,
                    end_offset=end,
                )
            )

            position += 1
            offset = end

        return tokens


class WordTokenizer(Tokenizer):
    """Word-based tokenizer with punctuation handling."""

    def __init__(self, min_length: int = 2) -> None:
        """Initialize with minimum token length."""
        self.min_length = min_length
        self._pattern = re.compile(r"\b\w+\b")

    def tokenize(self, text: str) -> list[Token]:
        """Tokenize into words."""
        tokens = []
        position = 0

        for match in self._pattern.finditer(text):
            term = match.group().lower()
            if len(term) >= self.min_length:
                tokens.append(
                    Token(
                        term=term,
                        position=position,
                        start_offset=match.start(),
                        end_offset=match.end(),
                    )
                )
                position += 1

        return tokens


class NgramTokenizer(Tokenizer):
    """N-gram tokenizer for partial matching."""

    def __init__(self, min_gram: int = 2, max_gram: int = 4) -> None:
        """Initialize with n-gram sizes."""
        self.min_gram = min_gram
        self.max_gram = max_gram

    def tokenize(self, text: str) -> list[Token]:
        """Tokenize into n-grams."""
        tokens = []
        text_lower = text.lower()
        position = 0

        # First, get word tokens
        words = re.findall(r"\b\w+\b", text_lower)

        for word in words:
            for gram_size in range(self.min_gram, self.max_gram + 1):
                for i in range(len(word) - gram_size + 1):
                    ngram = word[i : i + gram_size]
                    tokens.append(
                        Token(
                            term=ngram,
                            position=position,
                            start_offset=i,
                            end_offset=i + gram_size,
                            metadata={"is_ngram": True, "gram_size": gram_size},
                        )
                    )
                    position += 1

        return tokens


class StopwordFilter:
    """Filter for removing stop words."""

    DEFAULT_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }

    def __init__(self, stopwords: set[str] | None = None) -> None:
        """Initialize with optional custom stopwords."""
        self.stopwords = stopwords or self.DEFAULT_STOPWORDS

    def filter(self, tokens: list[Token]) -> list[Token]:
        """Filter out stop words."""
        return [t for t in tokens if t.term not in self.stopwords]


class Stemmer:
    """Simple suffix-stripping stemmer."""

    SUFFIXES = ["ing", "ed", "es", "s", "ly", "er", "tion", "ness"]

    def stem(self, term: str) -> str:
        """Stem a term."""
        for suffix in sorted(self.SUFFIXES, key=len, reverse=True):
            if term.endswith(suffix) and len(term) - len(suffix) >= 3:
                return term[: -len(suffix)]
        return term


class SearchIndex:
    """Inverted index for full-text search."""

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        use_stemming: bool = True,
        use_stopwords: bool = True,
    ) -> None:
        """Initialize the index."""
        self.tokenizer = tokenizer or WordTokenizer()
        self.stemmer = Stemmer() if use_stemming else None
        self.stopword_filter = StopwordFilter() if use_stopwords else None

        self._documents: dict[str, IndexedDocument] = {}
        self._inverted_index: dict[str, set[str]] = {}
        self._term_doc_freq: dict[str, int] = {}
        self._doc_lengths: dict[str, int] = {}

    def add_document(
        self,
        doc_id: str,
        content: str,
        title: str = "",
        **metadata: Any,
    ) -> IndexedDocument:
        """Add a document to the index."""
        # Tokenize
        tokens = self.tokenizer.tokenize(content)

        # Filter stopwords
        if self.stopword_filter:
            tokens = self.stopword_filter.filter(tokens)

        # Apply stemming
        if self.stemmer:
            for token in tokens:
                token.term = self.stemmer.stem(token.term)

        # Calculate term frequencies
        term_frequencies: dict[str, int] = Counter(t.term for t in tokens)

        # Create document
        doc = IndexedDocument(
            doc_id=doc_id,
            content=content,
            title=title,
            tokens=tokens,
            term_frequencies=term_frequencies,
            metadata=metadata,
        )

        # Store document
        self._documents[doc_id] = doc
        self._doc_lengths[doc_id] = len(tokens)

        # Update inverted index
        for term in term_frequencies:
            if term not in self._inverted_index:
                self._inverted_index[term] = set()
            self._inverted_index[term].add(doc_id)
            self._term_doc_freq[term] = len(self._inverted_index[term])

        return doc

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if doc_id not in self._documents:
            return False

        doc = self._documents[doc_id]

        # Update inverted index
        for term in doc.term_frequencies:
            if term in self._inverted_index:
                self._inverted_index[term].discard(doc_id)
                if not self._inverted_index[term]:
                    del self._inverted_index[term]
                else:
                    self._term_doc_freq[term] = len(self._inverted_index[term])

        # Remove document
        del self._documents[doc_id]
        del self._doc_lengths[doc_id]

        return True

    def get_document(self, doc_id: str) -> IndexedDocument | None:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    def search(
        self,
        query: str | SearchQuery,
        limit: int = 10,
        offset: int = 0,
    ) -> list[tuple[str, float]]:
        """Search the index."""
        if isinstance(query, str):
            query = SearchQuery(query_text=query, limit=limit, offset=offset)

        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query.query_text)

        if self.stopword_filter:
            query_tokens = self.stopword_filter.filter(query_tokens)

        if self.stemmer:
            for token in query_tokens:
                token.term = self.stemmer.stem(token.term)

        query_terms = [t.term for t in query_tokens]

        if not query_terms:
            return []

        # Find candidate documents
        candidates: set[str] = set()
        for term in query_terms:
            if term in self._inverted_index:
                if query.mode == SearchMode.ALL:
                    if not candidates:
                        candidates = self._inverted_index[term].copy()
                    else:
                        candidates &= self._inverted_index[term]
                else:
                    candidates |= self._inverted_index[term]

        # Score documents using BM25
        scores: list[tuple[str, float]] = []
        for doc_id in candidates:
            score = self._bm25_score(doc_id, query_terms)
            if score >= query.min_score:
                scores.append((doc_id, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Apply pagination
        return scores[query.offset : query.offset + query.limit]

    def _bm25_score(
        self,
        doc_id: str,
        query_terms: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """Calculate BM25 score for a document."""
        doc = self._documents.get(doc_id)
        if not doc:
            return 0.0

        num_docs = len(self._documents)
        avg_doc_len = sum(self._doc_lengths.values()) / num_docs if num_docs > 0 else 1
        doc_len = self._doc_lengths.get(doc_id, 0)

        score = 0.0
        for term in query_terms:
            if term not in self._inverted_index:
                continue

            # Term frequency in document
            term_freq = doc.term_frequencies.get(term, 0)

            # Document frequency
            doc_freq = self._term_doc_freq.get(term, 0)

            # IDF
            idf = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

            # BM25 term score
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf * (numerator / denominator)

        return score

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "num_documents": len(self._documents),
            "num_terms": len(self._inverted_index),
            "total_tokens": sum(self._doc_lengths.values()),
            "avg_doc_length": (
                sum(self._doc_lengths.values()) / len(self._documents) if self._documents else 0
            ),
        }


class SearchEngine:
    """Main search engine class."""

    def __init__(
        self,
        index: SearchIndex | None = None,
    ) -> None:
        """Initialize the search engine."""
        self.index = index or SearchIndex()
        self._analyzers: dict[str, Callable[[str], str]] = {}

    def index_document(
        self,
        doc_id: str,
        content: str,
        title: str = "",
        **metadata: Any,
    ) -> IndexedDocument:
        """Index a document.

        Args:
            doc_id: Document identifier.
            content: Document content.
            title: Document title.
            **metadata: Additional metadata.

        Returns:
            The indexed document.
        """
        # Apply analyzers
        analyzed_content = content
        for analyzer in self._analyzers.values():
            analyzed_content = analyzer(analyzed_content)

        return self.index.add_document(
            doc_id=doc_id,
            content=content,
            title=title,
            **metadata,
        )

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        return self.index.remove_document(doc_id)

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.ANY,
        limit: int = 10,
        offset: int = 0,
        highlight: bool = True,
        **filters: Any,
    ) -> SearchResults:
        """Search for documents.

        Args:
            query: Search query text.
            mode: Search mode.
            limit: Maximum results.
            offset: Result offset.
            highlight: Whether to highlight matches.
            **filters: Additional filters.

        Returns:
            SearchResults with hits.
        """
        import time

        start = time.time()

        search_query = SearchQuery(
            query_text=query,
            mode=mode,
            limit=limit,
            offset=offset,
            highlight=highlight,
            filters=filters,
        )

        # Get scored documents
        scored_docs = self.index.search(search_query)

        # Build hits
        hits = []
        max_score = 0.0

        for doc_id, score in scored_docs:
            doc = self.index.get_document(doc_id)
            if not doc:
                continue

            max_score = max(max_score, score)

            hit = SearchHit(
                doc_id=doc_id,
                score=score,
                title=doc.title,
                snippet=self._create_snippet(doc.content, query),
                highlights=self._get_highlights(doc.content, query) if highlight else [],
                matched_terms=self._get_matched_terms(doc, query),
                metadata=doc.metadata,
            )
            hits.append(hit)

        query_time = time.time() - start

        return SearchResults(
            hits=hits,
            total_hits=len(scored_docs),
            query=query,
            query_time=query_time,
            max_score=max_score,
        )

    def _create_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """Create a snippet from content."""
        query_terms = query.lower().split()

        # Find the first occurrence of any query term
        content_lower = content.lower()
        first_pos = len(content)

        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and pos < first_pos:
                first_pos = pos

        if first_pos == len(content):
            first_pos = 0

        # Extract snippet around the position
        start = max(0, first_pos - 50)
        end = min(len(content), first_pos + max_length)

        snippet = content[start:end]

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def _get_highlights(self, content: str, query: str) -> list[str]:
        """Get highlighted matches from content."""
        highlights = []
        query_terms = query.lower().split()

        for term in query_terms:
            pattern = re.compile(
                r".{0,30}" + re.escape(term) + r".{0,30}",
                re.IGNORECASE,
            )
            for match in pattern.finditer(content):
                highlight = match.group()
                highlights.append(highlight)

        return highlights[:5]

    def _get_matched_terms(self, doc: IndexedDocument, query: str) -> list[str]:
        """Get terms that matched in the document."""
        query_tokens = self.index.tokenizer.tokenize(query)
        query_terms = {t.term for t in query_tokens}

        matched = []
        for term in doc.term_frequencies:
            if term in query_terms:
                matched.append(term)

        return matched

    def add_analyzer(self, name: str, analyzer: Callable[[str], str]) -> None:
        """Add a content analyzer."""
        self._analyzers[name] = analyzer

    def suggest(self, prefix: str, limit: int = 10) -> list[str]:
        """Get search suggestions.

        Args:
            prefix: Term prefix.
            limit: Maximum suggestions.

        Returns:
            List of suggested terms.
        """
        prefix_lower = prefix.lower()
        suggestions = []

        for term in self.index._inverted_index:
            if term.startswith(prefix_lower):
                suggestions.append(term)

        # Sort by document frequency
        suggestions.sort(
            key=lambda t: self.index._term_doc_freq.get(t, 0),
            reverse=True,
        )

        return suggestions[:limit]

    def get_similar(self, doc_id: str, limit: int = 5) -> list[SearchHit]:
        """Get similar documents.

        Args:
            doc_id: Reference document ID.
            limit: Maximum results.

        Returns:
            List of similar documents.
        """
        doc = self.index.get_document(doc_id)
        if not doc:
            return []

        # Use top terms from document as query
        top_terms = sorted(
            doc.term_frequencies.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        query = " ".join(term for term, _ in top_terms)
        results = self.search(query, limit=limit + 1)

        # Filter out the reference document
        return [hit for hit in results.hits if hit.doc_id != doc_id][:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get search engine statistics."""
        return self.index.get_stats()


# Global instance
_search_engine: SearchEngine | None = None


def get_search_engine() -> SearchEngine:
    """Get the global search engine."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
    return _search_engine


def reset_search_engine() -> None:
    """Reset the global search engine."""
    global _search_engine
    _search_engine = None


def create_search_engine(
    tokenizer: Tokenizer | None = None,
    use_stemming: bool = True,
) -> SearchEngine:
    """Create a new search engine.

    Args:
        tokenizer: Custom tokenizer.
        use_stemming: Whether to use stemming.

    Returns:
        New SearchEngine instance.
    """
    index = SearchIndex(tokenizer=tokenizer, use_stemming=use_stemming)
    return SearchEngine(index=index)


def index_explanation(doc_id: str, content: str, **metadata: Any) -> IndexedDocument:
    """Index an explanation.

    Args:
        doc_id: Document identifier.
        content: Explanation content.
        **metadata: Additional metadata.

    Returns:
        The indexed document.
    """
    return get_search_engine().index_document(doc_id, content, **metadata)


def search_explanations(query: str, limit: int = 10, **kwargs: Any) -> SearchResults:
    """Search explanations.

    Args:
        query: Search query.
        limit: Maximum results.
        **kwargs: Additional options.

    Returns:
        SearchResults with hits.
    """
    return get_search_engine().search(query, limit=limit, **kwargs)


def create_word_tokenizer(min_length: int = 2) -> WordTokenizer:
    """Create a word tokenizer.

    Args:
        min_length: Minimum token length.

    Returns:
        WordTokenizer instance.
    """
    return WordTokenizer(min_length=min_length)


def create_ngram_tokenizer(min_gram: int = 2, max_gram: int = 4) -> NgramTokenizer:
    """Create an n-gram tokenizer.

    Args:
        min_gram: Minimum n-gram size.
        max_gram: Maximum n-gram size.

    Returns:
        NgramTokenizer instance.
    """
    return NgramTokenizer(min_gram=min_gram, max_gram=max_gram)
