"""
Explanation Exporter for multiple output formats.

This module provides tools for exporting code explanations to various
formats including PDF, HTML, Markdown, JSON, and more.
"""

from __future__ import annotations

import html
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


class ExportFormat(Enum):
    """Supported export formats."""

    HTML = auto()
    MARKDOWN = auto()
    JSON = auto()
    PLAIN_TEXT = auto()
    RST = auto()  # reStructuredText
    ASCIIDOC = auto()
    LATEX = auto()


class ExportStyle(Enum):
    """Visual styles for exports."""

    MINIMAL = auto()
    STANDARD = auto()
    DETAILED = auto()
    ACADEMIC = auto()
    PRESENTATION = auto()


@dataclass
class ExportOptions:
    """Options for export customization."""

    format: ExportFormat = ExportFormat.HTML
    style: ExportStyle = ExportStyle.STANDARD
    include_code: bool = True
    include_metrics: bool = True
    include_diagrams: bool = False
    include_toc: bool = True
    syntax_highlighting: bool = True
    dark_mode: bool = False
    custom_css: Optional[str] = None
    page_width: str = "800px"
    font_family: str = "system-ui, sans-serif"
    code_font: str = "monospace"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format.name,
            "style": self.style.name,
            "include_code": self.include_code,
            "include_metrics": self.include_metrics,
            "include_diagrams": self.include_diagrams,
            "include_toc": self.include_toc,
            "syntax_highlighting": self.syntax_highlighting,
            "dark_mode": self.dark_mode,
            "custom_css": self.custom_css,
            "page_width": self.page_width,
            "font_family": self.font_family,
            "code_font": self.code_font,
        }


@dataclass
class ExportSection:
    """A section in an export document."""

    id: str
    title: str
    content: str
    level: int = 1
    code_blocks: List[Dict[str, str]] = field(default_factory=list)
    subsections: List["ExportSection"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "code_blocks": self.code_blocks,
            "subsections": [s.to_dict() for s in self.subsections],
            "metadata": self.metadata,
        }


@dataclass
class ExportDocument:
    """A document to be exported."""

    id: str
    title: str
    description: str
    author: str
    sections: List[ExportSection]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "author": self.author,
            "sections": [s.to_dict() for s in self.sections],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ExportResult:
    """Result of an export operation."""

    id: str
    format: ExportFormat
    content: str
    file_path: Optional[str] = None
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "format": self.format.name,
            "content_length": len(self.content),
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]) -> str:
        """Save the export to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.content, encoding="utf-8")
        self.file_path = str(path)
        self.size_bytes = path.stat().st_size
        return str(path)


class ExportRenderer(ABC):
    """Base class for export renderers."""

    @property
    @abstractmethod
    def format(self) -> ExportFormat:
        """Return the format this renderer produces."""
        pass

    @abstractmethod
    def render(self, document: ExportDocument, options: ExportOptions) -> str:
        """Render the document to the target format."""
        pass


class HTMLRenderer(ExportRenderer):
    """Render documents to HTML."""

    @property
    def format(self) -> ExportFormat:
        return ExportFormat.HTML

    def render(self, document: ExportDocument, options: ExportOptions) -> str:
        """Render to HTML."""
        css = self._get_css(options)
        toc = self._generate_toc(document.sections) if options.include_toc else ""

        sections_html = "\n".join(self._render_section(s, options) for s in document.sections)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(document.title)}</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{html.escape(document.title)}</h1>
            <p class="description">{html.escape(document.description)}</p>
            <p class="meta">By {html.escape(document.author)} · {document.created_at.strftime('%Y-%m-%d')}</p>
        </header>
        {toc}
        <main>
            {sections_html}
        </main>
        <footer>
            <p>Generated by OpenEval Code Explainers</p>
        </footer>
    </div>
</body>
</html>"""

    def _get_css(self, options: ExportOptions) -> str:
        """Generate CSS for the document."""
        bg_color = "#1a1a2e" if options.dark_mode else "#ffffff"
        text_color = "#eaeaea" if options.dark_mode else "#333333"
        code_bg = "#0f0f23" if options.dark_mode else "#f5f5f5"
        border_color = "#333" if options.dark_mode else "#ddd"

        custom = options.custom_css or ""

        return f"""
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: {options.font_family};
            line-height: 1.6;
            color: {text_color};
            background-color: {bg_color};
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: {options.page_width};
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid {border_color};
        }}
        h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        h2 {{ font-size: 1.5rem; margin-top: 2rem; }}
        h3 {{ font-size: 1.25rem; margin-top: 1.5rem; }}
        .description {{ font-size: 1.1rem; opacity: 0.8; }}
        .meta {{ font-size: 0.9rem; opacity: 0.6; }}
        .toc {{
            background: {code_bg};
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }}
        .toc h2 {{ margin-top: 0; font-size: 1.2rem; }}
        .toc ul {{ margin: 0; padding-left: 1.5rem; }}
        .toc li {{ margin: 0.25rem 0; }}
        .toc a {{ color: inherit; text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
        section {{ margin-bottom: 2rem; }}
        pre {{
            background: {code_bg};
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-family: {options.code_font};
        }}
        code {{
            font-family: {options.code_font};
            background: {code_bg};
            padding: 0.2em 0.4em;
            border-radius: 3px;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid {border_color};
            font-size: 0.85rem;
            opacity: 0.6;
            text-align: center;
        }}
        {custom}
        """

    def _generate_toc(self, sections: List[ExportSection]) -> str:
        """Generate table of contents."""
        items = []
        for section in sections:
            items.append(f'<li><a href="#{section.id}">{html.escape(section.title)}</a>')
            if section.subsections:
                sub_items = "".join(
                    f'<li><a href="#{s.id}">{html.escape(s.title)}</a></li>'
                    for s in section.subsections
                )
                items.append(f"<ul>{sub_items}</ul>")
            items.append("</li>")

        return f"""
        <nav class="toc">
            <h2>Table of Contents</h2>
            <ul>
                {"".join(items)}
            </ul>
        </nav>
        """

    def _render_section(self, section: ExportSection, options: ExportOptions) -> str:
        """Render a section to HTML."""
        heading_tag = f"h{min(section.level + 1, 6)}"
        code_blocks_html = ""

        if options.include_code and section.code_blocks:
            code_blocks_html = "\n".join(
                f'<pre><code class="language-{cb.get("language", "text")}">'
                f'{html.escape(cb.get("code", ""))}</code></pre>'
                for cb in section.code_blocks
            )

        subsections_html = "\n".join(self._render_section(s, options) for s in section.subsections)

        content_html = self._markdown_to_html(section.content)

        return f"""
        <section id="{section.id}">
            <{heading_tag}>{html.escape(section.title)}</{heading_tag}>
            <div class="content">{content_html}</div>
            {code_blocks_html}
            {subsections_html}
        </section>
        """

    def _markdown_to_html(self, text: str) -> str:
        """Simple markdown to HTML conversion."""
        # Bold
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        # Italic
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        # Code
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        # Paragraphs
        paragraphs = text.split("\n\n")
        text = "".join(f"<p>{p}</p>" for p in paragraphs if p.strip())
        return text


class MarkdownRenderer(ExportRenderer):
    """Render documents to Markdown."""

    @property
    def format(self) -> ExportFormat:
        return ExportFormat.MARKDOWN

    def render(self, document: ExportDocument, options: ExportOptions) -> str:
        """Render to Markdown."""
        lines = []

        # Title and metadata
        lines.append(f"# {document.title}")
        lines.append("")
        lines.append(f"> {document.description}")
        lines.append("")
        lines.append(f"*By {document.author} · {document.created_at.strftime('%Y-%m-%d')}*")
        lines.append("")

        # Table of contents
        if options.include_toc:
            lines.append("## Table of Contents")
            lines.append("")
            for section in document.sections:
                anchor = section.id.lower().replace(" ", "-")
                lines.append(f"- [{section.title}](#{anchor})")
                for sub in section.subsections:
                    sub_anchor = sub.id.lower().replace(" ", "-")
                    lines.append(f"  - [{sub.title}](#{sub_anchor})")
            lines.append("")

        # Sections
        for section in document.sections:
            lines.extend(self._render_section(section, options))

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by OpenEval Code Explainers*")

        return "\n".join(lines)

    def _render_section(self, section: ExportSection, options: ExportOptions) -> List[str]:
        """Render a section to Markdown lines."""
        lines = []
        heading = "#" * (section.level + 1)
        lines.append(f"{heading} {section.title}")
        lines.append("")
        lines.append(section.content)
        lines.append("")

        # Code blocks
        if options.include_code:
            for cb in section.code_blocks:
                lang = cb.get("language", "")
                code = cb.get("code", "")
                lines.append(f"```{lang}")
                lines.append(code)
                lines.append("```")
                lines.append("")

        # Subsections
        for sub in section.subsections:
            lines.extend(self._render_section(sub, options))

        return lines


class JSONRenderer(ExportRenderer):
    """Render documents to JSON."""

    @property
    def format(self) -> ExportFormat:
        return ExportFormat.JSON

    def render(self, document: ExportDocument, options: ExportOptions) -> str:
        """Render to JSON."""
        data = document.to_dict()
        data["export_options"] = options.to_dict()
        return json.dumps(data, indent=2, ensure_ascii=False)


class PlainTextRenderer(ExportRenderer):
    """Render documents to plain text."""

    @property
    def format(self) -> ExportFormat:
        return ExportFormat.PLAIN_TEXT

    def render(self, document: ExportDocument, options: ExportOptions) -> str:
        """Render to plain text."""
        lines = []
        width = 80

        # Title
        lines.append("=" * width)
        lines.append(document.title.center(width))
        lines.append("=" * width)
        lines.append("")
        lines.append(document.description)
        lines.append("")
        lines.append(f"By {document.author} - {document.created_at.strftime('%Y-%m-%d')}")
        lines.append("")
        lines.append("-" * width)
        lines.append("")

        # Sections
        for section in document.sections:
            lines.extend(self._render_section(section, options))

        # Footer
        lines.append("-" * width)
        lines.append("Generated by OpenEval Code Explainers".center(width))

        return "\n".join(lines)

    def _render_section(self, section: ExportSection, options: ExportOptions) -> List[str]:
        """Render a section to plain text lines."""
        lines = []

        # Heading with underline
        lines.append(section.title)
        underline_char = "=" if section.level == 1 else "-"
        lines.append(underline_char * len(section.title))
        lines.append("")
        lines.append(section.content)
        lines.append("")

        # Code blocks
        if options.include_code:
            for cb in section.code_blocks:
                lines.append("    " + "-" * 40)
                for line in cb.get("code", "").split("\n"):
                    lines.append("    " + line)
                lines.append("    " + "-" * 40)
                lines.append("")

        # Subsections
        for sub in section.subsections:
            lines.extend(self._render_section(sub, options))

        return lines


class LaTeXRenderer(ExportRenderer):
    """Render documents to LaTeX."""

    @property
    def format(self) -> ExportFormat:
        return ExportFormat.LATEX

    def render(self, document: ExportDocument, options: ExportOptions) -> str:
        """Render to LaTeX."""
        sections = "\n".join(self._render_section(s, options) for s in document.sections)

        return rf"""\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{listings}}
\usepackage{{hyperref}}
\usepackage{{geometry}}
\geometry{{margin=1in}}

\title{{{self._escape_latex(document.title)}}}
\author{{{self._escape_latex(document.author)}}}
\date{{{document.created_at.strftime('%Y-%m-%d')}}}

\begin{{document}}
\maketitle

\begin{{abstract}}
{self._escape_latex(document.description)}
\end{{abstract}}

{"\\tableofcontents\\newpage" if options.include_toc else ""}

{sections}

\end{{document}}
"""

    def _render_section(self, section: ExportSection, options: ExportOptions) -> str:
        """Render a section to LaTeX."""
        level_cmds = ["section", "subsection", "subsubsection", "paragraph"]
        cmd = level_cmds[min(section.level - 1, len(level_cmds) - 1)]

        code_blocks = ""
        if options.include_code and section.code_blocks:
            code_blocks = "\n".join(
                f"\\begin{{lstlisting}}[language={cb.get('language', 'text')}]\n"
                f"{cb.get('code', '')}\n"
                f"\\end{{lstlisting}}"
                for cb in section.code_blocks
            )

        subsections = "\n".join(self._render_section(s, options) for s in section.subsections)

        return rf"""
\{cmd}{{{self._escape_latex(section.title)}}}

{self._escape_latex(section.content)}

{code_blocks}

{subsections}
"""

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = [
            ("\\", "\\textbackslash{}"),
            ("&", "\\&"),
            ("%", "\\%"),
            ("$", "\\$"),
            ("#", "\\#"),
            ("_", "\\_"),
            ("{", "\\{"),
            ("}", "\\}"),
            ("~", "\\textasciitilde{}"),
            ("^", "\\textasciicircum{}"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text


class ExplanationExporter:
    """Main class for exporting explanations."""

    def __init__(self, renderers: Optional[Dict[ExportFormat, ExportRenderer]] = None):
        """Initialize the exporter."""
        self.renderers = renderers or self._get_default_renderers()

    def _get_default_renderers(self) -> Dict[ExportFormat, ExportRenderer]:
        """Get default renderers."""
        return {
            ExportFormat.HTML: HTMLRenderer(),
            ExportFormat.MARKDOWN: MarkdownRenderer(),
            ExportFormat.JSON: JSONRenderer(),
            ExportFormat.PLAIN_TEXT: PlainTextRenderer(),
            ExportFormat.LATEX: LaTeXRenderer(),
        }

    def export(
        self,
        document: ExportDocument,
        options: Optional[ExportOptions] = None,
    ) -> ExportResult:
        """Export a document."""
        options = options or ExportOptions()

        if options.format not in self.renderers:
            raise ValueError(f"Unsupported format: {options.format.name}")

        renderer = self.renderers[options.format]
        content = renderer.render(document, options)

        return ExportResult(
            id=str(uuid4()),
            format=options.format,
            content=content,
            size_bytes=len(content.encode("utf-8")),
        )

    def export_to_file(
        self,
        document: ExportDocument,
        path: Union[str, Path],
        options: Optional[ExportOptions] = None,
    ) -> ExportResult:
        """Export a document to a file."""
        result = self.export(document, options)
        result.save(path)
        return result

    def batch_export(
        self,
        document: ExportDocument,
        formats: List[ExportFormat],
        output_dir: Union[str, Path],
        base_name: Optional[str] = None,
    ) -> List[ExportResult]:
        """Export a document to multiple formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = base_name or document.id

        results = []
        format_extensions = {
            ExportFormat.HTML: ".html",
            ExportFormat.MARKDOWN: ".md",
            ExportFormat.JSON: ".json",
            ExportFormat.PLAIN_TEXT: ".txt",
            ExportFormat.LATEX: ".tex",
            ExportFormat.RST: ".rst",
            ExportFormat.ASCIIDOC: ".adoc",
        }

        for fmt in formats:
            ext = format_extensions.get(fmt, ".txt")
            path = output_dir / f"{base_name}{ext}"
            options = ExportOptions(format=fmt)
            result = self.export_to_file(document, path, options)
            results.append(result)

        return results


# Convenience functions
def create_export_document(
    title: str,
    description: str,
    sections: List[ExportSection],
    author: str = "OpenEval",
    **kwargs,
) -> ExportDocument:
    """Create an export document."""
    return ExportDocument(
        id=str(uuid4()),
        title=title,
        description=description,
        author=author,
        sections=sections,
        **kwargs,
    )


def create_export_section(
    title: str,
    content: str,
    level: int = 1,
    code_blocks: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> ExportSection:
    """Create an export section."""
    return ExportSection(
        id=str(uuid4()),
        title=title,
        content=content,
        level=level,
        code_blocks=code_blocks or [],
        **kwargs,
    )


def export_to_html(
    document: ExportDocument,
    options: Optional[ExportOptions] = None,
) -> str:
    """Export a document to HTML."""
    exporter = ExplanationExporter()
    opts = options or ExportOptions(format=ExportFormat.HTML)
    opts.format = ExportFormat.HTML
    result = exporter.export(document, opts)
    return result.content


def export_to_markdown(
    document: ExportDocument,
    options: Optional[ExportOptions] = None,
) -> str:
    """Export a document to Markdown."""
    exporter = ExplanationExporter()
    opts = options or ExportOptions(format=ExportFormat.MARKDOWN)
    opts.format = ExportFormat.MARKDOWN
    result = exporter.export(document, opts)
    return result.content


def export_to_json(document: ExportDocument) -> str:
    """Export a document to JSON."""
    exporter = ExplanationExporter()
    result = exporter.export(document, ExportOptions(format=ExportFormat.JSON))
    return result.content


def export_to_text(
    document: ExportDocument,
    options: Optional[ExportOptions] = None,
) -> str:
    """Export a document to plain text."""
    exporter = ExplanationExporter()
    opts = options or ExportOptions(format=ExportFormat.PLAIN_TEXT)
    opts.format = ExportFormat.PLAIN_TEXT
    result = exporter.export(document, opts)
    return result.content


def export_to_latex(
    document: ExportDocument,
    options: Optional[ExportOptions] = None,
) -> str:
    """Export a document to LaTeX."""
    exporter = ExplanationExporter()
    opts = options or ExportOptions(format=ExportFormat.LATEX)
    opts.format = ExportFormat.LATEX
    result = exporter.export(document, opts)
    return result.content


def export_to_file(
    document: ExportDocument,
    path: Union[str, Path],
    format: Optional[ExportFormat] = None,
    **kwargs,
) -> ExportResult:
    """Export a document to a file."""
    exporter = ExplanationExporter()

    # Infer format from extension if not specified
    if format is None:
        ext = Path(path).suffix.lower()
        format_map = {
            ".html": ExportFormat.HTML,
            ".md": ExportFormat.MARKDOWN,
            ".json": ExportFormat.JSON,
            ".txt": ExportFormat.PLAIN_TEXT,
            ".tex": ExportFormat.LATEX,
        }
        format = format_map.get(ext, ExportFormat.HTML)

    options = ExportOptions(format=format, **kwargs)
    return exporter.export_to_file(document, path, options)
