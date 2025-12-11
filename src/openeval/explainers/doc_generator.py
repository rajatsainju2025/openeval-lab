"""Documentation generator from code explanations.

This module provides utilities for generating various documentation formats
from code explanations, including Markdown, HTML, and docstrings.
"""

from __future__ import annotations

import html
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, TextIO


from .types import CodeElement, CodeElementType, ExplanationResult


class DocFormat(Enum):
    """Documentation output format."""

    MARKDOWN = auto()
    HTML = auto()
    RST = auto()  # reStructuredText
    DOCSTRING = auto()
    JSON = auto()
    PLAINTEXT = auto()


class DocSection(Enum):
    """Documentation section types."""

    OVERVIEW = auto()
    PARAMETERS = auto()
    RETURNS = auto()
    RAISES = auto()
    EXAMPLES = auto()
    NOTES = auto()
    SEE_ALSO = auto()
    ATTRIBUTES = auto()
    METHODS = auto()


@dataclass
class DocConfig:
    """Configuration for documentation generation."""

    format: DocFormat = DocFormat.MARKDOWN
    include_source: bool = False
    include_metadata: bool = True
    include_timestamp: bool = True
    include_toc: bool = True  # Table of contents
    max_depth: int = 3  # Max heading depth
    code_language: str = "python"
    custom_css: str = ""
    custom_header: str = ""
    custom_footer: str = ""


@dataclass
class DocElement:
    """A documentation element."""

    title: str
    content: str
    section: DocSection | None = None
    level: int = 1  # Heading level
    element_type: CodeElementType | None = None
    children: list["DocElement"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedDoc:
    """Generated documentation output."""

    content: str
    format: DocFormat
    element_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save documentation to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.content)


class DocFormatter(ABC):
    """Abstract base for documentation formatters."""

    @abstractmethod
    def format_heading(self, text: str, level: int) -> str:
        """Format a heading."""
        ...

    @abstractmethod
    def format_paragraph(self, text: str) -> str:
        """Format a paragraph."""
        ...

    @abstractmethod
    def format_code_block(self, code: str, language: str = "") -> str:
        """Format a code block."""
        ...

    @abstractmethod
    def format_list(self, items: list[str], ordered: bool = False) -> str:
        """Format a list."""
        ...

    @abstractmethod
    def format_table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Format a table."""
        ...

    @abstractmethod
    def format_link(self, text: str, url: str) -> str:
        """Format a link."""
        ...

    @abstractmethod
    def format_emphasis(self, text: str, strong: bool = False) -> str:
        """Format emphasized text."""
        ...


class MarkdownFormatter(DocFormatter):
    """Markdown documentation formatter."""

    def format_heading(self, text: str, level: int) -> str:
        level = min(max(1, level), 6)
        return f"{'#' * level} {text}\n\n"

    def format_paragraph(self, text: str) -> str:
        return f"{text}\n\n"

    def format_code_block(self, code: str, language: str = "") -> str:
        return f"```{language}\n{code}\n```\n\n"

    def format_list(self, items: list[str], ordered: bool = False) -> str:
        result = []
        for i, item in enumerate(items, 1):
            prefix = f"{i}." if ordered else "-"
            result.append(f"{prefix} {item}")
        return "\n".join(result) + "\n\n"

    def format_table(self, headers: list[str], rows: list[list[str]]) -> str:
        result = ["| " + " | ".join(headers) + " |"]
        result.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            result.append("| " + " | ".join(row) + " |")
        return "\n".join(result) + "\n\n"

    def format_link(self, text: str, url: str) -> str:
        return f"[{text}]({url})"

    def format_emphasis(self, text: str, strong: bool = False) -> str:
        marker = "**" if strong else "*"
        return f"{marker}{text}{marker}"


class HTMLFormatter(DocFormatter):
    """HTML documentation formatter."""

    def format_heading(self, text: str, level: int) -> str:
        level = min(max(1, level), 6)
        return f"<h{level}>{html.escape(text)}</h{level}>\n"

    def format_paragraph(self, text: str) -> str:
        return f"<p>{html.escape(text)}</p>\n"

    def format_code_block(self, code: str, language: str = "") -> str:
        lang_class = f' class="language-{language}"' if language else ""
        return f"<pre><code{lang_class}>{html.escape(code)}</code></pre>\n"

    def format_list(self, items: list[str], ordered: bool = False) -> str:
        tag = "ol" if ordered else "ul"
        item_html = "\n".join(f"  <li>{html.escape(item)}</li>" for item in items)
        return f"<{tag}>\n{item_html}\n</{tag}>\n"

    def format_table(self, headers: list[str], rows: list[list[str]]) -> str:
        result = ["<table>", "  <thead>", "    <tr>"]
        for h in headers:
            result.append(f"      <th>{html.escape(h)}</th>")
        result.extend(["    </tr>", "  </thead>", "  <tbody>"])
        for row in rows:
            result.append("    <tr>")
            for cell in row:
                result.append(f"      <td>{html.escape(cell)}</td>")
            result.append("    </tr>")
        result.extend(["  </tbody>", "</table>"])
        return "\n".join(result) + "\n"

    def format_link(self, text: str, url: str) -> str:
        return f'<a href="{html.escape(url)}">{html.escape(text)}</a>'

    def format_emphasis(self, text: str, strong: bool = False) -> str:
        tag = "strong" if strong else "em"
        return f"<{tag}>{html.escape(text)}</{tag}>"


class RSTFormatter(DocFormatter):
    """reStructuredText documentation formatter."""

    def format_heading(self, text: str, level: int) -> str:
        chars = ["=", "-", "~", "^", '"']
        char = chars[min(level - 1, len(chars) - 1)]
        underline = char * len(text)
        if level == 1:
            return f"{underline}\n{text}\n{underline}\n\n"
        return f"{text}\n{underline}\n\n"

    def format_paragraph(self, text: str) -> str:
        return f"{text}\n\n"

    def format_code_block(self, code: str, language: str = "") -> str:
        directive = f".. code-block:: {language}\n\n" if language else ".. code::\n\n"
        indented = "\n".join(f"    {line}" for line in code.split("\n"))
        return f"{directive}{indented}\n\n"

    def format_list(self, items: list[str], ordered: bool = False) -> str:
        result = []
        for i, item in enumerate(items, 1):
            prefix = f"{i}." if ordered else "*"
            result.append(f"{prefix} {item}")
        return "\n".join(result) + "\n\n"

    def format_table(self, headers: list[str], rows: list[list[str]]) -> str:
        # Simple RST table
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        result = [sep]

        # Headers
        header_row = (
            "|" + "|".join(f" {h.ljust(col_widths[i])} " for i, h in enumerate(headers)) + "|"
        )
        result.append(header_row)
        result.append("+" + "+".join("=" * (w + 2) for w in col_widths) + "+")

        # Rows
        for row in rows:
            row_str = (
                "|" + "|".join(f" {cell.ljust(col_widths[i])} " for i, cell in enumerate(row)) + "|"
            )
            result.append(row_str)
            result.append(sep)

        return "\n".join(result) + "\n\n"

    def format_link(self, text: str, url: str) -> str:
        return f"`{text} <{url}>`_"

    def format_emphasis(self, text: str, strong: bool = False) -> str:
        marker = "**" if strong else "*"
        return f"{marker}{text}{marker}"


class DocstringFormatter(DocFormatter):
    """Docstring format (Google style)."""

    def format_heading(self, text: str, level: int) -> str:
        if level == 1:
            return f"{text}\n\n"
        return f"{text}:\n"

    def format_paragraph(self, text: str) -> str:
        # Wrap at 80 chars
        words = text.split()
        lines = []
        current_line = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > 76:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
            else:
                current_line.append(word)
                current_len += len(word) + 1
        if current_line:
            lines.append(" ".join(current_line))
        return "\n".join(f"    {line}" for line in lines) + "\n\n"

    def format_code_block(self, code: str, language: str = "") -> str:
        return "    >>> " + code.replace("\n", "\n    >>> ") + "\n\n"

    def format_list(self, items: list[str], ordered: bool = False) -> str:
        result = []
        for item in items:
            result.append(f"    - {item}")
        return "\n".join(result) + "\n\n"

    def format_table(self, headers: list[str], rows: list[list[str]]) -> str:
        # Tables not well supported in docstrings
        return self.format_list([f"{h}: {', '.join(r)}" for h, r in zip(headers, rows)])

    def format_link(self, text: str, url: str) -> str:
        return f"{text} ({url})"

    def format_emphasis(self, text: str, strong: bool = False) -> str:
        return text.upper() if strong else text


class DocGenerator:
    """Main documentation generator."""

    def __init__(self, config: DocConfig | None = None):
        """Initialize generator with configuration."""
        self.config = config or DocConfig()
        self._formatter = self._get_formatter()

    def _get_formatter(self) -> DocFormatter:
        """Get formatter for configured format."""
        formatters = {
            DocFormat.MARKDOWN: MarkdownFormatter,
            DocFormat.HTML: HTMLFormatter,
            DocFormat.RST: RSTFormatter,
            DocFormat.DOCSTRING: DocstringFormatter,
        }
        return formatters.get(self.config.format, MarkdownFormatter)()

    def generate(
        self,
        results: list[ExplanationResult],
        title: str = "Code Documentation",
    ) -> GeneratedDoc:
        """Generate documentation from explanation results."""
        content = []

        # Add custom header
        if self.config.custom_header:
            content.append(self.config.custom_header)

        # Add title
        content.append(self._formatter.format_heading(title, 1))

        # Add metadata
        if self.config.include_metadata:
            content.append(self._generate_metadata(results))

        # Add table of contents
        if self.config.include_toc and len(results) > 1:
            content.append(self._generate_toc(results))

        # Generate content for each result
        for result in results:
            content.append(self._generate_element_doc(result))

        # Add custom footer
        if self.config.custom_footer:
            content.append(self.config.custom_footer)

        # Wrap HTML if needed
        final_content = "".join(content)
        if self.config.format == DocFormat.HTML:
            final_content = self._wrap_html(final_content, title)

        return GeneratedDoc(
            content=final_content,
            format=self.config.format,
            element_count=len(results),
            metadata={
                "title": title,
                "config": {
                    "include_source": self.config.include_source,
                    "include_metadata": self.config.include_metadata,
                },
            },
        )

    def generate_single(
        self,
        result: ExplanationResult,
    ) -> GeneratedDoc:
        """Generate documentation for a single result."""
        return self.generate([result], title=f"Documentation: {result.element.name}")

    def _generate_metadata(self, results: list[ExplanationResult]) -> str:
        """Generate metadata section."""
        content = []
        content.append(self._formatter.format_heading("Documentation Info", 2))

        rows = [
            ["Generated", datetime.utcnow().isoformat()],
            ["Total Elements", str(len(results))],
        ]

        # Count by type
        type_counts: dict[str, int] = {}
        for r in results:
            type_name = r.element.type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        for type_name, count in type_counts.items():
            rows.append([type_name.title(), str(count)])

        content.append(self._formatter.format_table(["Property", "Value"], rows))

        return "".join(content)

    def _generate_toc(self, results: list[ExplanationResult]) -> str:
        """Generate table of contents."""
        content = []
        content.append(self._formatter.format_heading("Table of Contents", 2))

        items = []
        for result in results:
            element = result.element
            anchor = self._make_anchor(element.name)
            items.append(self._formatter.format_link(element.name, f"#{anchor}"))

        content.append(self._formatter.format_list(items))

        return "".join(content)

    def _generate_element_doc(self, result: ExplanationResult) -> str:
        """Generate documentation for a single element."""
        content = []
        element = result.element

        # Element heading
        heading = f"{element.type.name.title()}: {element.name}"
        content.append(self._formatter.format_heading(heading, 2))

        # Explanation
        content.append(self._formatter.format_paragraph(result.explanation))

        # Source code
        if self.config.include_source and element.source_code:
            content.append(self._formatter.format_heading("Source Code", 3))
            content.append(
                self._formatter.format_code_block(
                    element.source_code,
                    self.config.code_language,
                )
            )

        # Existing docstring
        if element.docstring:
            content.append(self._formatter.format_heading("Original Documentation", 3))
            content.append(self._formatter.format_paragraph(element.docstring))

        # Metadata
        if self.config.include_metadata and element.metadata:
            content.append(self._formatter.format_heading("Metadata", 3))
            items = [f"{k}: {v}" for k, v in element.metadata.items()]
            content.append(self._formatter.format_list(items))

        return "".join(content)

    def _make_anchor(self, text: str) -> str:
        """Create an anchor from text."""
        return re.sub(r"[^\w\-]", "-", text.lower())

    def _wrap_html(self, content: str, title: str) -> str:
        """Wrap content in HTML document."""
        css = (
            self.config.custom_css
            or """
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 800px; margin: 0 auto; padding: 20px; }
        pre { background: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto; }
        code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f6f8fa; }
        """
        )
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>{css}</style>
</head>
<body>
{content}
</body>
</html>"""


class DocstringGenerator:
    """Generate docstrings from explanations."""

    def __init__(self, style: str = "google"):
        """Initialize with docstring style."""
        self.style = style  # google, numpy, sphinx

    def generate(self, result: ExplanationResult) -> str:
        """Generate a docstring from an explanation."""
        element = result.element
        explanation = result.explanation

        if self.style == "google":
            return self._generate_google(element, explanation)
        elif self.style == "numpy":
            return self._generate_numpy(element, explanation)
        elif self.style == "sphinx":
            return self._generate_sphinx(element, explanation)
        else:
            return self._generate_simple(explanation)

    def _generate_google(self, element: CodeElement, explanation: str) -> str:
        """Generate Google-style docstring."""
        lines = [f'"""{self._get_summary(explanation)}']

        # Extract parameters from code
        if element.type == CodeElementType.FUNCTION:
            params = self._extract_params(element.source_code or "")
            if params:
                lines.append("")
                lines.append("Args:")
                for param, desc in params.items():
                    lines.append(f"    {param}: {desc}")

            # Check for return statement
            if element.source_code and "return" in element.source_code:
                lines.append("")
                lines.append("Returns:")
                lines.append("    The computed result.")

        lines.append('"""')
        return "\n".join(lines)

    def _generate_numpy(self, element: CodeElement, explanation: str) -> str:
        """Generate NumPy-style docstring."""
        lines = [f'"""{self._get_summary(explanation)}']

        if element.type == CodeElementType.FUNCTION:
            params = self._extract_params(element.source_code or "")
            if params:
                lines.append("")
                lines.append("Parameters")
                lines.append("----------")
                for param, desc in params.items():
                    lines.append(f"{param} : type")
                    lines.append(f"    {desc}")

            if element.source_code and "return" in element.source_code:
                lines.append("")
                lines.append("Returns")
                lines.append("-------")
                lines.append("type")
                lines.append("    Description of return value.")

        lines.append('"""')
        return "\n".join(lines)

    def _generate_sphinx(self, element: CodeElement, explanation: str) -> str:
        """Generate Sphinx-style docstring."""
        lines = [f'"""{self._get_summary(explanation)}']

        if element.type == CodeElementType.FUNCTION:
            params = self._extract_params(element.source_code or "")
            if params:
                lines.append("")
                for param, desc in params.items():
                    lines.append(f":param {param}: {desc}")
                    lines.append(f":type {param}: type")

            if element.source_code and "return" in element.source_code:
                lines.append(":returns: Description of return value.")
                lines.append(":rtype: type")

        lines.append('"""')
        return "\n".join(lines)

    def _generate_simple(self, explanation: str) -> str:
        """Generate simple docstring."""
        summary = self._get_summary(explanation)
        return f'"""{summary}"""'

    def _get_summary(self, explanation: str) -> str:
        """Extract first sentence as summary."""
        sentences = re.split(r"[.!?]", explanation)
        if sentences:
            return sentences[0].strip() + "."
        return explanation[:100].strip() + "..."

    def _extract_params(self, source_code: str) -> dict[str, str]:
        """Extract parameter names from function signature."""
        # Simple regex to find function parameters
        match = re.search(r"def\s+\w+\s*\(([^)]*)\)", source_code)
        if not match:
            return {}

        params_str = match.group(1)
        params = {}
        for param in params_str.split(","):
            param = param.strip()
            if param and param not in ("self", "cls"):
                # Remove type hints and defaults
                param_name = param.split(":")[0].split("=")[0].strip()
                if param_name:
                    params[param_name] = "Description needed."
        return params


class DocWriter:
    """Write documentation to files."""

    def __init__(self, output_dir: str | Path):
        """Initialize with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, doc: GeneratedDoc, filename: str) -> Path:
        """Write documentation to a file."""
        extension = self._get_extension(doc.format)
        filepath = self.output_dir / f"{filename}.{extension}"
        filepath.write_text(doc.content)
        return filepath

    def write_stream(self, doc: GeneratedDoc, stream: TextIO) -> None:
        """Write documentation to a stream."""
        stream.write(doc.content)

    def _get_extension(self, format: DocFormat) -> str:
        """Get file extension for format."""
        extensions = {
            DocFormat.MARKDOWN: "md",
            DocFormat.HTML: "html",
            DocFormat.RST: "rst",
            DocFormat.DOCSTRING: "py",
            DocFormat.JSON: "json",
            DocFormat.PLAINTEXT: "txt",
        }
        return extensions.get(format, "txt")


# Convenience functions
def generate_markdown(
    results: list[ExplanationResult],
    title: str = "Code Documentation",
    include_source: bool = False,
) -> str:
    """Generate Markdown documentation."""
    config = DocConfig(
        format=DocFormat.MARKDOWN,
        include_source=include_source,
    )
    generator = DocGenerator(config)
    return generator.generate(results, title).content


def generate_html(
    results: list[ExplanationResult],
    title: str = "Code Documentation",
    include_source: bool = False,
) -> str:
    """Generate HTML documentation."""
    config = DocConfig(
        format=DocFormat.HTML,
        include_source=include_source,
    )
    generator = DocGenerator(config)
    return generator.generate(results, title).content


def generate_docstring(
    result: ExplanationResult,
    style: str = "google",
) -> str:
    """Generate a docstring from an explanation."""
    generator = DocstringGenerator(style)
    return generator.generate(result)


def create_doc_generator(
    format: DocFormat = DocFormat.MARKDOWN,
    **kwargs: Any,
) -> DocGenerator:
    """Create a documentation generator."""
    config = DocConfig(format=format, **kwargs)
    return DocGenerator(config)


__all__ = [
    # Enums
    "DocFormat",
    "DocSection",
    # Data classes
    "DocConfig",
    "DocElement",
    "GeneratedDoc",
    # Formatters
    "DocFormatter",
    "MarkdownFormatter",
    "HTMLFormatter",
    "RSTFormatter",
    "DocstringFormatter",
    # Generators
    "DocGenerator",
    "DocstringGenerator",
    # Writer
    "DocWriter",
    # Functions
    "generate_markdown",
    "generate_html",
    "generate_docstring",
    "create_doc_generator",
]
