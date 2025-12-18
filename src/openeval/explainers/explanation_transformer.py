"""Explanation transformer module for converting between different formats.

This module provides comprehensive transformation capabilities for code explanations,
supporting multiple output formats including markdown, HTML, JSON, plain text,
and structured formats for different use cases.
"""

from __future__ import annotations

import html
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class TransformFormat(Enum):
    """Supported output formats for explanations."""

    MARKDOWN = auto()
    HTML = auto()
    JSON = auto()
    PLAIN_TEXT = auto()
    RST = auto()
    ASCIIDOC = auto()
    LATEX = auto()
    XML = auto()
    YAML = auto()
    STRUCTURED = auto()


class TransformStyle(Enum):
    """Transformation style options."""

    MINIMAL = auto()
    STANDARD = auto()
    VERBOSE = auto()
    COMPACT = auto()
    TECHNICAL = auto()
    EDUCATIONAL = auto()


@dataclass
class TransformOptions:
    """Options for controlling transformation behavior."""

    style: TransformStyle = TransformStyle.STANDARD
    include_metadata: bool = True
    include_code_blocks: bool = True
    include_examples: bool = True
    include_cross_references: bool = True
    max_line_length: int = 80
    indent_size: int = 4
    escape_special_chars: bool = True
    preserve_formatting: bool = True
    add_table_of_contents: bool = False
    add_syntax_highlighting: bool = True
    language_hint: str = "python"
    custom_css: str | None = None
    custom_template: str | None = None


@dataclass
class ExplanationSection:
    """A section within an explanation."""

    title: str
    content: str
    level: int = 1
    section_type: str = "text"
    code_block: str | None = None
    language: str = "python"
    metadata: dict[str, Any] = field(default_factory=dict)
    subsections: list[ExplanationSection] = field(default_factory=list)


@dataclass
class ExplanationData:
    """Structured explanation data for transformation."""

    title: str
    summary: str
    sections: list[ExplanationSection] = field(default_factory=list)
    code_snippet: str | None = None
    language: str = "python"
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    references: list[dict[str, str]] = field(default_factory=list)
    created_at: str | None = None
    version: str = "1.0"


@dataclass
class TransformResult:
    """Result of a transformation operation."""

    output: str
    format: TransformFormat
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FormatTransformer(ABC):
    """Abstract base class for format transformers."""

    @property
    @abstractmethod
    def format_type(self) -> TransformFormat:
        """Get the output format type."""
        pass

    @abstractmethod
    def transform(self, data: ExplanationData, options: TransformOptions) -> TransformResult:
        """Transform explanation data to the target format."""
        pass

    @abstractmethod
    def validate(self, output: str) -> tuple[bool, list[str]]:
        """Validate the transformed output."""
        pass


class MarkdownTransformer(FormatTransformer):
    """Transform explanations to Markdown format."""

    @property
    def format_type(self) -> TransformFormat:
        return TransformFormat.MARKDOWN

    def transform(self, data: ExplanationData, options: TransformOptions) -> TransformResult:
        """Transform to Markdown."""
        lines: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # Title
            lines.append(f"# {data.title}\n")

            # Metadata
            if options.include_metadata and data.metadata:
                lines.append("---")
                for key, value in data.metadata.items():
                    lines.append(f"{key}: {value}")
                lines.append("---\n")

            # Table of contents
            if options.add_table_of_contents and data.sections:
                lines.append("## Table of Contents\n")
                for i, section in enumerate(data.sections, 1):
                    anchor = section.title.lower().replace(" ", "-")
                    lines.append(f"{i}. [{section.title}](#{anchor})")
                lines.append("")

            # Summary
            lines.append(f"{data.summary}\n")

            # Code snippet
            if options.include_code_blocks and data.code_snippet:
                lines.append(f"```{data.language}")
                lines.append(data.code_snippet)
                lines.append("```\n")

            # Sections
            for section in data.sections:
                lines.extend(self._transform_section(section, options))

            # Tags
            if data.tags:
                lines.append("## Tags\n")
                lines.append(", ".join(f"`{tag}`" for tag in data.tags))
                lines.append("")

            # References
            if options.include_cross_references and data.references:
                lines.append("## References\n")
                for ref in data.references:
                    lines.append(f"- [{ref.get('title', 'Reference')}]({ref.get('url', '#')})")
                lines.append("")

            output = "\n".join(lines)
            return TransformResult(
                output=output,
                format=TransformFormat.MARKDOWN,
                success=True,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"Transformation error: {e!s}")
            return TransformResult(
                output="",
                format=TransformFormat.MARKDOWN,
                success=False,
                errors=errors,
                warnings=warnings,
            )

    def _transform_section(
        self, section: ExplanationSection, options: TransformOptions
    ) -> list[str]:
        """Transform a single section."""
        lines: list[str] = []
        prefix = "#" * (section.level + 1)
        lines.append(f"{prefix} {section.title}\n")
        lines.append(f"{section.content}\n")

        if options.include_code_blocks and section.code_block:
            lines.append(f"```{section.language}")
            lines.append(section.code_block)
            lines.append("```\n")

        for subsection in section.subsections:
            lines.extend(self._transform_section(subsection, options))

        return lines

    def validate(self, output: str) -> tuple[bool, list[str]]:
        """Validate Markdown output."""
        errors: list[str] = []

        # Check for unclosed code blocks
        code_block_count = output.count("```")
        if code_block_count % 2 != 0:
            errors.append("Unclosed code block detected")

        # Check for empty headers
        if re.search(r"^#+\s*$", output, re.MULTILINE):
            errors.append("Empty header detected")

        return len(errors) == 0, errors


class HTMLTransformer(FormatTransformer):
    """Transform explanations to HTML format."""

    @property
    def format_type(self) -> TransformFormat:
        return TransformFormat.HTML

    def transform(self, data: ExplanationData, options: TransformOptions) -> TransformResult:
        """Transform to HTML."""
        errors: list[str] = []
        warnings: list[str] = []

        try:
            parts: list[str] = []

            # HTML header
            parts.append("<!DOCTYPE html>")
            parts.append('<html lang="en">')
            parts.append("<head>")
            parts.append(f"<title>{html.escape(data.title)}</title>")
            parts.append('<meta charset="UTF-8">')
            parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')

            # CSS
            if options.custom_css:
                parts.append(f"<style>{options.custom_css}</style>")
            else:
                parts.append(self._default_css())

            # Syntax highlighting CSS
            if options.add_syntax_highlighting:
                parts.append(self._syntax_highlight_css())

            parts.append("</head>")
            parts.append("<body>")
            parts.append('<article class="explanation">')

            # Title
            parts.append(f"<h1>{html.escape(data.title)}</h1>")

            # Metadata
            if options.include_metadata and data.metadata:
                parts.append('<div class="metadata">')
                for key, value in data.metadata.items():
                    parts.append(
                        f'<span class="meta-item"><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</span>'
                    )
                parts.append("</div>")

            # Summary
            parts.append(f'<div class="summary"><p>{html.escape(data.summary)}</p></div>')

            # Code snippet
            if options.include_code_blocks and data.code_snippet:
                parts.append('<div class="code-block">')
                parts.append(
                    f'<pre><code class="language-{data.language}">{html.escape(data.code_snippet)}</code></pre>'
                )
                parts.append("</div>")

            # Sections
            for section in data.sections:
                parts.extend(self._transform_section(section, options))

            # Tags
            if data.tags:
                parts.append('<div class="tags">')
                parts.append("<h2>Tags</h2>")
                parts.append('<ul class="tag-list">')
                for tag in data.tags:
                    parts.append(f'<li class="tag">{html.escape(tag)}</li>')
                parts.append("</ul>")
                parts.append("</div>")

            # References
            if options.include_cross_references and data.references:
                parts.append('<div class="references">')
                parts.append("<h2>References</h2>")
                parts.append("<ul>")
                for ref in data.references:
                    title = html.escape(ref.get("title", "Reference"))
                    url = html.escape(ref.get("url", "#"))
                    parts.append(f'<li><a href="{url}">{title}</a></li>')
                parts.append("</ul>")
                parts.append("</div>")

            parts.append("</article>")
            parts.append("</body>")
            parts.append("</html>")

            output = "\n".join(parts)
            return TransformResult(
                output=output,
                format=TransformFormat.HTML,
                success=True,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"Transformation error: {e!s}")
            return TransformResult(
                output="",
                format=TransformFormat.HTML,
                success=False,
                errors=errors,
                warnings=warnings,
            )

    def _transform_section(
        self, section: ExplanationSection, options: TransformOptions
    ) -> list[str]:
        """Transform a single section to HTML."""
        parts: list[str] = []
        level = min(section.level + 1, 6)
        parts.append(f'<section class="section-level-{level}">')
        parts.append(f"<h{level}>{html.escape(section.title)}</h{level}>")
        parts.append(f"<p>{html.escape(section.content)}</p>")

        if options.include_code_blocks and section.code_block:
            parts.append('<div class="code-block">')
            parts.append(
                f'<pre><code class="language-{section.language}">{html.escape(section.code_block)}</code></pre>'
            )
            parts.append("</div>")

        for subsection in section.subsections:
            parts.extend(self._transform_section(subsection, options))

        parts.append("</section>")
        return parts

    def _default_css(self) -> str:
        """Get default CSS styles."""
        return """<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
.explanation { line-height: 1.6; }
h1, h2, h3, h4, h5, h6 { margin-top: 1.5em; margin-bottom: 0.5em; }
.metadata { background: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
.meta-item { display: block; margin: 5px 0; }
.summary { font-size: 1.1em; color: #333; margin-bottom: 20px; }
.code-block { background: #f8f8f8; border: 1px solid #ddd; border-radius: 5px; padding: 15px; overflow-x: auto; }
pre { margin: 0; }
code { font-family: 'SF Mono', Monaco, 'Courier New', monospace; }
.tags { margin-top: 30px; }
.tag-list { list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 10px; }
.tag { background: #e0e0e0; padding: 5px 10px; border-radius: 15px; }
.references { margin-top: 30px; }
</style>"""

    def _syntax_highlight_css(self) -> str:
        """Get syntax highlighting CSS."""
        return """<style>
.keyword { color: #0000ff; }
.string { color: #a31515; }
.comment { color: #008000; }
.number { color: #098658; }
.function { color: #795e26; }
</style>"""

    def validate(self, output: str) -> tuple[bool, list[str]]:
        """Validate HTML output."""
        errors: list[str] = []

        # Check for basic structure
        if "<html" not in output.lower():
            errors.append("Missing <html> tag")
        if "<body" not in output.lower():
            errors.append("Missing <body> tag")

        return len(errors) == 0, errors


class JSONTransformer(FormatTransformer):
    """Transform explanations to JSON format."""

    @property
    def format_type(self) -> TransformFormat:
        return TransformFormat.JSON

    def transform(self, data: ExplanationData, options: TransformOptions) -> TransformResult:
        """Transform to JSON."""
        errors: list[str] = []
        warnings: list[str] = []

        try:
            result: dict[str, Any] = {
                "title": data.title,
                "summary": data.summary,
                "version": data.version,
            }

            if options.include_metadata:
                result["metadata"] = data.metadata
                result["created_at"] = data.created_at
                result["tags"] = data.tags

            if options.include_code_blocks and data.code_snippet:
                result["code"] = {
                    "snippet": data.code_snippet,
                    "language": data.language,
                }

            result["sections"] = [
                self._section_to_dict(section, options) for section in data.sections
            ]

            if options.include_cross_references:
                result["references"] = data.references

            indent = options.indent_size if options.preserve_formatting else None
            output = json.dumps(result, indent=indent, ensure_ascii=False)

            return TransformResult(
                output=output,
                format=TransformFormat.JSON,
                success=True,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"Transformation error: {e!s}")
            return TransformResult(
                output="",
                format=TransformFormat.JSON,
                success=False,
                errors=errors,
                warnings=warnings,
            )

    def _section_to_dict(
        self, section: ExplanationSection, options: TransformOptions
    ) -> dict[str, Any]:
        """Convert a section to dictionary."""
        result: dict[str, Any] = {
            "title": section.title,
            "content": section.content,
            "level": section.level,
            "type": section.section_type,
        }

        if options.include_code_blocks and section.code_block:
            result["code_block"] = section.code_block
            result["language"] = section.language

        if options.include_metadata and section.metadata:
            result["metadata"] = section.metadata

        if section.subsections:
            result["subsections"] = [
                self._section_to_dict(sub, options) for sub in section.subsections
            ]

        return result

    def validate(self, output: str) -> tuple[bool, list[str]]:
        """Validate JSON output."""
        errors: list[str] = []

        try:
            json.loads(output)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")

        return len(errors) == 0, errors


class PlainTextTransformer(FormatTransformer):
    """Transform explanations to plain text format."""

    @property
    def format_type(self) -> TransformFormat:
        return TransformFormat.PLAIN_TEXT

    def transform(self, data: ExplanationData, options: TransformOptions) -> TransformResult:
        """Transform to plain text."""
        lines: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # Title
            lines.append(data.title.upper())
            lines.append("=" * len(data.title))
            lines.append("")

            # Summary
            lines.append(data.summary)
            lines.append("")

            # Code snippet
            if options.include_code_blocks and data.code_snippet:
                lines.append("CODE:")
                lines.append("-" * 40)
                lines.append(data.code_snippet)
                lines.append("-" * 40)
                lines.append("")

            # Sections
            for section in data.sections:
                lines.extend(self._transform_section(section, options))

            # Tags
            if data.tags:
                lines.append("TAGS: " + ", ".join(data.tags))
                lines.append("")

            output = "\n".join(lines)

            # Apply line wrapping
            if options.max_line_length > 0:
                output = self._wrap_lines(output, options.max_line_length)

            return TransformResult(
                output=output,
                format=TransformFormat.PLAIN_TEXT,
                success=True,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"Transformation error: {e!s}")
            return TransformResult(
                output="",
                format=TransformFormat.PLAIN_TEXT,
                success=False,
                errors=errors,
                warnings=warnings,
            )

    def _transform_section(
        self, section: ExplanationSection, options: TransformOptions
    ) -> list[str]:
        """Transform a section to plain text."""
        lines: list[str] = []
        indent = "  " * (section.level - 1)

        lines.append(f"{indent}{section.title}")
        lines.append(f"{indent}{'-' * len(section.title)}")
        lines.append(f"{indent}{section.content}")
        lines.append("")

        if options.include_code_blocks and section.code_block:
            lines.append(f"{indent}Code:")
            for code_line in section.code_block.split("\n"):
                lines.append(f"{indent}  {code_line}")
            lines.append("")

        for subsection in section.subsections:
            lines.extend(self._transform_section(subsection, options))

        return lines

    def _wrap_lines(self, text: str, max_length: int) -> str:
        """Wrap lines to max length."""
        import textwrap

        lines = text.split("\n")
        wrapped_lines: list[str] = []

        for line in lines:
            if len(line) > max_length and not line.startswith(" "):
                wrapped = textwrap.fill(line, width=max_length)
                wrapped_lines.append(wrapped)
            else:
                wrapped_lines.append(line)

        return "\n".join(wrapped_lines)

    def validate(self, output: str) -> tuple[bool, list[str]]:
        """Validate plain text output."""
        # Plain text is always valid
        return True, []


class RSTTransformer(FormatTransformer):
    """Transform explanations to reStructuredText format."""

    @property
    def format_type(self) -> TransformFormat:
        return TransformFormat.RST

    def transform(self, data: ExplanationData, options: TransformOptions) -> TransformResult:
        """Transform to RST."""
        lines: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # Title
            lines.append("=" * len(data.title))
            lines.append(data.title)
            lines.append("=" * len(data.title))
            lines.append("")

            # Metadata
            if options.include_metadata and data.metadata:
                lines.append("::")
                for key, value in data.metadata.items():
                    lines.append(f"   :{key}: {value}")
                lines.append("")

            # Summary
            lines.append(data.summary)
            lines.append("")

            # Code snippet
            if options.include_code_blocks and data.code_snippet:
                lines.append(f".. code-block:: {data.language}")
                lines.append("")
                for code_line in data.code_snippet.split("\n"):
                    lines.append(f"   {code_line}")
                lines.append("")

            # Sections
            for section in data.sections:
                lines.extend(self._transform_section(section, options))

            output = "\n".join(lines)
            return TransformResult(
                output=output,
                format=TransformFormat.RST,
                success=True,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"Transformation error: {e!s}")
            return TransformResult(
                output="",
                format=TransformFormat.RST,
                success=False,
                errors=errors,
                warnings=warnings,
            )

    def _transform_section(
        self, section: ExplanationSection, options: TransformOptions
    ) -> list[str]:
        """Transform a section to RST."""
        lines: list[str] = []
        underlines = ["=", "-", "~", "^", '"']
        underline = underlines[min(section.level - 1, len(underlines) - 1)]

        lines.append(section.title)
        lines.append(underline * len(section.title))
        lines.append("")
        lines.append(section.content)
        lines.append("")

        if options.include_code_blocks and section.code_block:
            lines.append(f".. code-block:: {section.language}")
            lines.append("")
            for code_line in section.code_block.split("\n"):
                lines.append(f"   {code_line}")
            lines.append("")

        for subsection in section.subsections:
            lines.extend(self._transform_section(subsection, options))

        return lines

    def validate(self, output: str) -> tuple[bool, list[str]]:
        """Validate RST output."""
        errors: list[str] = []
        # Basic RST validation
        lines = output.split("\n")

        for i, line in enumerate(lines):
            # Check for malformed directives
            if line.startswith(".. ") and "::" not in line:
                errors.append(f"Possible malformed directive at line {i + 1}")

        return len(errors) == 0, errors


class ExplanationTransformer:
    """Main transformer class for converting explanations between formats."""

    def __init__(self) -> None:
        """Initialize the transformer with default transformers."""
        self._transformers: dict[TransformFormat, FormatTransformer] = {}
        self._custom_transformers: dict[str, Callable[[ExplanationData, TransformOptions], str]] = (
            {}
        )

        # Register default transformers
        self.register_transformer(MarkdownTransformer())
        self.register_transformer(HTMLTransformer())
        self.register_transformer(JSONTransformer())
        self.register_transformer(PlainTextTransformer())
        self.register_transformer(RSTTransformer())

    def register_transformer(self, transformer: FormatTransformer) -> None:
        """Register a format transformer."""
        self._transformers[transformer.format_type] = transformer

    def register_custom_transformer(
        self, name: str, transformer: Callable[[ExplanationData, TransformOptions], str]
    ) -> None:
        """Register a custom transformer function."""
        self._custom_transformers[name] = transformer

    def transform(
        self,
        data: ExplanationData,
        target_format: TransformFormat,
        options: TransformOptions | None = None,
    ) -> TransformResult:
        """Transform explanation data to the target format."""
        if options is None:
            options = TransformOptions()

        transformer = self._transformers.get(target_format)
        if not transformer:
            return TransformResult(
                output="",
                format=target_format,
                success=False,
                errors=[f"No transformer registered for format: {target_format}"],
            )

        return transformer.transform(data, options)

    def transform_custom(
        self,
        data: ExplanationData,
        transformer_name: str,
        options: TransformOptions | None = None,
    ) -> TransformResult:
        """Transform using a custom transformer."""
        if options is None:
            options = TransformOptions()

        transformer = self._custom_transformers.get(transformer_name)
        if not transformer:
            return TransformResult(
                output="",
                format=TransformFormat.STRUCTURED,
                success=False,
                errors=[f"Custom transformer not found: {transformer_name}"],
            )

        try:
            output = transformer(data, options)
            return TransformResult(
                output=output,
                format=TransformFormat.STRUCTURED,
                success=True,
            )
        except Exception as e:
            return TransformResult(
                output="",
                format=TransformFormat.STRUCTURED,
                success=False,
                errors=[f"Custom transformation error: {e!s}"],
            )

    def validate(self, output: str, format_type: TransformFormat) -> tuple[bool, list[str]]:
        """Validate output for a specific format."""
        transformer = self._transformers.get(format_type)
        if not transformer:
            return False, [f"No validator for format: {format_type}"]
        return transformer.validate(output)

    def convert(
        self,
        source: str,
        source_format: TransformFormat,
        target_format: TransformFormat,
        options: TransformOptions | None = None,
    ) -> TransformResult:
        """Convert between formats (parses source, then transforms)."""
        data = self.parse(source, source_format)
        if data is None:
            return TransformResult(
                output="",
                format=target_format,
                success=False,
                errors=["Failed to parse source"],
            )
        return self.transform(data, target_format, options)

    def parse(self, source: str, format_type: TransformFormat) -> ExplanationData | None:
        """Parse source content into ExplanationData."""
        if format_type == TransformFormat.JSON:
            return self._parse_json(source)
        elif format_type == TransformFormat.MARKDOWN:
            return self._parse_markdown(source)
        return None

    def _parse_json(self, source: str) -> ExplanationData | None:
        """Parse JSON source into ExplanationData."""
        try:
            data = json.loads(source)
            sections = []
            for section_data in data.get("sections", []):
                section = ExplanationSection(
                    title=section_data.get("title", ""),
                    content=section_data.get("content", ""),
                    level=section_data.get("level", 1),
                    section_type=section_data.get("type", "text"),
                    code_block=section_data.get("code_block"),
                    language=section_data.get("language", "python"),
                )
                sections.append(section)

            return ExplanationData(
                title=data.get("title", ""),
                summary=data.get("summary", ""),
                sections=sections,
                code_snippet=(
                    data.get("code", {}).get("snippet")
                    if isinstance(data.get("code"), dict)
                    else None
                ),
                language=(
                    data.get("code", {}).get("language", "python")
                    if isinstance(data.get("code"), dict)
                    else "python"
                ),
                metadata=data.get("metadata", {}),
                tags=data.get("tags", []),
                references=data.get("references", []),
                version=data.get("version", "1.0"),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _parse_markdown(self, source: str) -> ExplanationData | None:
        """Parse Markdown source into ExplanationData."""
        try:
            lines = source.split("\n")
            title = ""
            summary_lines: list[str] = []
            sections: list[ExplanationSection] = []
            current_section: ExplanationSection | None = None
            in_code_block = False
            code_lines: list[str] = []

            for line in lines:
                # Handle code blocks
                if line.startswith("```"):
                    if in_code_block:
                        if current_section:
                            current_section.code_block = "\n".join(code_lines)
                        code_lines = []
                        in_code_block = False
                    else:
                        in_code_block = True
                        current_code_language = line[3:].strip() or "python"
                        if current_section:
                            current_section.language = current_code_language
                    continue

                if in_code_block:
                    code_lines.append(line)
                    continue

                # Handle headers
                if line.startswith("# "):
                    if not title:
                        title = line[2:].strip()
                    continue

                if line.startswith("## "):
                    if current_section:
                        sections.append(current_section)
                    current_section = ExplanationSection(
                        title=line[3:].strip(),
                        content="",
                        level=2,
                    )
                    continue

                if line.startswith("### "):
                    if current_section:
                        sections.append(current_section)
                    current_section = ExplanationSection(
                        title=line[4:].strip(),
                        content="",
                        level=3,
                    )
                    continue

                # Content
                if current_section:
                    current_section.content += line + "\n"
                elif title and not sections:
                    summary_lines.append(line)

            if current_section:
                sections.append(current_section)

            return ExplanationData(
                title=title,
                summary="\n".join(summary_lines).strip(),
                sections=sections,
            )
        except Exception:
            return None

    def supported_formats(self) -> list[TransformFormat]:
        """Get list of supported output formats."""
        return list(self._transformers.keys())


# Convenience functions
def create_transformer() -> ExplanationTransformer:
    """Create a new explanation transformer."""
    return ExplanationTransformer()


def transform_explanation(
    data: ExplanationData,
    target_format: TransformFormat,
    options: TransformOptions | None = None,
) -> TransformResult:
    """Transform explanation data to target format."""
    transformer = ExplanationTransformer()
    return transformer.transform(data, target_format, options)


def create_explanation_data(
    title: str,
    summary: str,
    code_snippet: str | None = None,
    language: str = "python",
    **kwargs: Any,
) -> ExplanationData:
    """Create explanation data from basic inputs."""
    return ExplanationData(
        title=title,
        summary=summary,
        code_snippet=code_snippet,
        language=language,
        **kwargs,
    )


def create_section(
    title: str,
    content: str,
    level: int = 1,
    code_block: str | None = None,
    language: str = "python",
) -> ExplanationSection:
    """Create an explanation section."""
    return ExplanationSection(
        title=title,
        content=content,
        level=level,
        code_block=code_block,
        language=language,
    )


def to_markdown(data: ExplanationData, options: TransformOptions | None = None) -> str:
    """Convert explanation data to Markdown."""
    result = transform_explanation(data, TransformFormat.MARKDOWN, options)
    return result.output if result.success else ""


def to_html(data: ExplanationData, options: TransformOptions | None = None) -> str:
    """Convert explanation data to HTML."""
    result = transform_explanation(data, TransformFormat.HTML, options)
    return result.output if result.success else ""


def to_json(data: ExplanationData, options: TransformOptions | None = None) -> str:
    """Convert explanation data to JSON."""
    result = transform_explanation(data, TransformFormat.JSON, options)
    return result.output if result.success else ""


def to_plain_text(data: ExplanationData, options: TransformOptions | None = None) -> str:
    """Convert explanation data to plain text."""
    result = transform_explanation(data, TransformFormat.PLAIN_TEXT, options)
    return result.output if result.success else ""


def to_rst(data: ExplanationData, options: TransformOptions | None = None) -> str:
    """Convert explanation data to reStructuredText."""
    result = transform_explanation(data, TransformFormat.RST, options)
    return result.output if result.success else ""
