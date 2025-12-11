"""Integration adapters for external systems.

This module provides adapters for integrating code explanations with
external systems like IDEs, CI/CD pipelines, and other tools.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable


from .types import ExplanationResult


class IntegrationType(Enum):
    """Type of integration."""

    IDE = auto()  # IDE integration (VS Code, PyCharm, etc.)
    CI_CD = auto()  # CI/CD pipeline integration
    API = auto()  # REST API integration
    WEBHOOK = auto()  # Webhook integration
    FILE = auto()  # File-based integration
    CUSTOM = auto()  # Custom integration


class OutputFormat(Enum):
    """Output format for integrations."""

    JSON = auto()
    SARIF = auto()  # Static Analysis Results Interchange Format
    JUNIT = auto()  # JUnit XML format
    CHECKSTYLE = auto()  # Checkstyle XML format
    GITHUB = auto()  # GitHub annotations format
    GITLAB = auto()  # GitLab code quality format
    WEBHOOK = auto()  # Webhook payload format
    PLAIN = auto()  # Plain text


@dataclass
class AdapterConfig:
    """Configuration for integration adapters."""

    output_format: OutputFormat = OutputFormat.JSON
    output_path: str | None = None
    include_metadata: bool = True
    include_source: bool = False
    verbose: bool = False
    custom_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationResult:
    """Result of an integration operation."""

    success: bool
    message: str = ""
    output: str = ""
    output_path: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class IntegrationAdapter(ABC):
    """Abstract base class for integration adapters."""

    def __init__(self, config: AdapterConfig | None = None):
        """Initialize adapter with configuration."""
        self.config = config or AdapterConfig()

    @abstractmethod
    def export(self, results: list[ExplanationResult]) -> IntegrationResult:
        """Export explanation results."""
        ...

    @abstractmethod
    def get_format(self) -> OutputFormat:
        """Get the output format."""
        ...


class JSONAdapter(IntegrationAdapter):
    """JSON export adapter."""

    def get_format(self) -> OutputFormat:
        return OutputFormat.JSON

    def export(self, results: list[ExplanationResult]) -> IntegrationResult:
        """Export results as JSON."""
        data = {
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(results),
            "explanations": [self._result_to_dict(r) for r in results],
        }

        output = json.dumps(data, indent=2)

        if self.config.output_path:
            Path(self.config.output_path).write_text(output)

        return IntegrationResult(
            success=True,
            message=f"Exported {len(results)} explanations to JSON",
            output=output,
            output_path=self.config.output_path,
        )

    def _result_to_dict(self, result: ExplanationResult) -> dict[str, Any]:
        """Convert result to dictionary."""
        data = {
            "element": {
                "name": result.element.name,
                "type": result.element.type.name,
                "line_start": result.element.line_start,
                "line_end": result.element.line_end,
            },
            "explanation": result.explanation,
            "level": result.level.name,
            "confidence": result.confidence,
        }

        if self.config.include_metadata:
            data["metadata"] = result.analysis_metadata
            data["model"] = result.model_used
            data["timestamp"] = result.timestamp

        if self.config.include_source:
            data["element"]["source"] = result.element.source_code

        return data


class SARIFAdapter(IntegrationAdapter):
    """SARIF (Static Analysis Results Interchange Format) adapter.

    Compatible with GitHub Advanced Security and other SARIF-compatible tools.
    """

    def get_format(self) -> OutputFormat:
        return OutputFormat.SARIF

    def export(self, results: list[ExplanationResult]) -> IntegrationResult:
        """Export results in SARIF format."""
        sarif = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "OpenEval Code Explainer",
                            "version": "1.0.0",
                            "informationUri": "https://github.com/openeval/openeval-lab",
                            "rules": self._get_rules(results),
                        }
                    },
                    "results": [self._result_to_sarif(r, i) for i, r in enumerate(results)],
                }
            ],
        }

        output = json.dumps(sarif, indent=2)

        if self.config.output_path:
            Path(self.config.output_path).write_text(output)

        return IntegrationResult(
            success=True,
            message=f"Exported {len(results)} explanations to SARIF",
            output=output,
            output_path=self.config.output_path,
        )

    def _get_rules(self, results: list[ExplanationResult]) -> list[dict]:
        """Generate SARIF rules from results."""
        rules = []
        seen_types = set()

        for result in results:
            type_name = result.element.type.name
            if type_name not in seen_types:
                seen_types.add(type_name)
                rules.append(
                    {
                        "id": f"EXPLAIN-{type_name}",
                        "name": f"Code Explanation: {type_name}",
                        "shortDescription": {
                            "text": f"Explanation for {type_name.lower()} elements"
                        },
                    }
                )

        return rules

    def _result_to_sarif(self, result: ExplanationResult, index: int) -> dict:
        """Convert result to SARIF result object."""
        return {
            "ruleId": f"EXPLAIN-{result.element.type.name}",
            "ruleIndex": index,
            "level": "note",
            "message": {"text": result.explanation[:500]},  # SARIF message length limit
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": result.element.metadata.get("file_path", "unknown")
                        },
                        "region": {
                            "startLine": result.element.line_start or 1,
                            "endLine": result.element.line_end or 1,
                        },
                    }
                }
            ],
            "properties": {
                "confidence": result.confidence,
                "level": result.level.name,
                "element_name": result.element.name,
            },
        }


class GitHubAdapter(IntegrationAdapter):
    """GitHub Actions integration adapter.

    Outputs annotations and workflow commands for GitHub Actions.
    """

    def get_format(self) -> OutputFormat:
        return OutputFormat.GITHUB

    def export(self, results: list[ExplanationResult]) -> IntegrationResult:
        """Export results as GitHub annotations."""
        lines = []

        for result in results:
            # GitHub workflow command format
            file_path = result.element.metadata.get("file_path", "")
            line = result.element.line_start or 1
            end_line = result.element.line_end or line

            # Escape message for GitHub
            message = result.explanation.replace("\n", "%0A").replace("\r", "%0D")

            # Use notice level for explanations
            annotation = (
                f"::notice file={file_path},line={line},endLine={end_line},"
                f"title=Code Explanation: {result.element.name}::{message[:500]}"
            )
            lines.append(annotation)

        output = "\n".join(lines)

        if self.config.output_path:
            Path(self.config.output_path).write_text(output)

        return IntegrationResult(
            success=True,
            message=f"Generated {len(results)} GitHub annotations",
            output=output,
            output_path=self.config.output_path,
        )


class GitLabAdapter(IntegrationAdapter):
    """GitLab Code Quality integration adapter.

    Outputs in GitLab Code Quality report format.
    """

    def get_format(self) -> OutputFormat:
        return OutputFormat.GITLAB

    def export(self, results: list[ExplanationResult]) -> IntegrationResult:
        """Export results in GitLab Code Quality format."""
        issues = []

        for result in results:
            file_path = result.element.metadata.get("file_path", "unknown")
            issue = {
                "description": result.explanation[:500],
                "fingerprint": f"{file_path}:{result.element.name}:{result.element.line_start}",
                "severity": "info",
                "location": {
                    "path": file_path,
                    "lines": {
                        "begin": result.element.line_start or 1,
                        "end": result.element.line_end or result.element.line_start or 1,
                    },
                },
            }
            issues.append(issue)

        output = json.dumps(issues, indent=2)

        if self.config.output_path:
            Path(self.config.output_path).write_text(output)

        return IntegrationResult(
            success=True,
            message=f"Generated {len(results)} GitLab code quality issues",
            output=output,
            output_path=self.config.output_path,
        )


class JUnitAdapter(IntegrationAdapter):
    """JUnit XML format adapter for test framework integration."""

    def get_format(self) -> OutputFormat:
        return OutputFormat.JUNIT

    def export(self, results: list[ExplanationResult]) -> IntegrationResult:
        """Export results in JUnit XML format."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="Code Explanations" tests="{len(results)}" '
            f'failures="0" errors="0" time="0">',
        ]

        for result in results:
            test_name = f"explain_{result.element.type.name}_{result.element.name}"
            lines.append(f'  <testcase name="{test_name}" classname="openeval.explainers">')
            lines.append("    <system-out><![CDATA[")
            lines.append(f"Element: {result.element.name}")
            lines.append(f"Type: {result.element.type.name}")
            lines.append(f"Confidence: {result.confidence:.2f}")
            lines.append("")
            lines.append("Explanation:")
            lines.append(result.explanation)
            lines.append("    ]]></system-out>")
            lines.append("  </testcase>")

        lines.append("</testsuite>")

        output = "\n".join(lines)

        if self.config.output_path:
            Path(self.config.output_path).write_text(output)

        return IntegrationResult(
            success=True,
            message=f"Generated JUnit XML with {len(results)} test cases",
            output=output,
            output_path=self.config.output_path,
        )


class VSCodeAdapter(IntegrationAdapter):
    """VS Code integration adapter.

    Generates output suitable for VS Code extensions and decorations.
    """

    def get_format(self) -> OutputFormat:
        return OutputFormat.JSON

    def export(self, results: list[ExplanationResult]) -> IntegrationResult:
        """Export results for VS Code integration."""
        decorations = []

        for result in results:
            file_path = result.element.metadata.get("file_path", "")
            decoration = {
                "uri": f"file://{file_path}",
                "range": {
                    "start": {"line": (result.element.line_start or 1) - 1, "character": 0},
                    "end": {
                        "line": (result.element.line_end or result.element.line_start or 1) - 1,
                        "character": 0,
                    },
                },
                "hoverMessage": result.explanation,
                "contentText": f"📖 {result.element.name}: {result.explanation[:100]}...",
                "color": self._get_confidence_color(result.confidence),
                "element": {
                    "name": result.element.name,
                    "type": result.element.type.name,
                },
                "metadata": {
                    "confidence": result.confidence,
                    "level": result.level.name,
                    "model": result.model_used,
                },
            }
            decorations.append(decoration)

        output = json.dumps({"decorations": decorations}, indent=2)

        if self.config.output_path:
            Path(self.config.output_path).write_text(output)

        return IntegrationResult(
            success=True,
            message=f"Generated {len(results)} VS Code decorations",
            output=output,
            output_path=self.config.output_path,
        )

    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level."""
        if confidence >= 0.8:
            return "#4CAF50"  # Green
        elif confidence >= 0.6:
            return "#FFC107"  # Yellow
        else:
            return "#FF5722"  # Orange


class WebhookAdapter(IntegrationAdapter):
    """Webhook integration adapter for sending results to external services."""

    def __init__(
        self,
        webhook_url: str,
        config: AdapterConfig | None = None,
        headers: dict[str, str] | None = None,
    ):
        """Initialize with webhook URL."""
        super().__init__(config)
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}

    def get_format(self) -> OutputFormat:
        return OutputFormat.WEBHOOK

    def export(self, results: list[ExplanationResult]) -> IntegrationResult:
        """Export results via webhook (simulated - actual HTTP call would require requests)."""
        payload = {
            "event": "code_explanations",
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(results),
            "explanations": [
                {
                    "element": result.element.name,
                    "type": result.element.type.name,
                    "explanation": result.explanation,
                    "confidence": result.confidence,
                }
                for result in results
            ],
        }

        output = json.dumps(payload, indent=2)

        # Note: Actual HTTP call would require an HTTP library
        return IntegrationResult(
            success=True,
            message=f"Prepared webhook payload for {len(results)} explanations",
            output=output,
            metadata={
                "webhook_url": self.webhook_url,
                "headers": self.headers,
            },
        )


class FileSystemAdapter(IntegrationAdapter):
    """File system adapter for saving explanations as files."""

    def __init__(
        self,
        output_dir: str | Path,
        config: AdapterConfig | None = None,
    ):
        """Initialize with output directory."""
        super().__init__(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_format(self) -> OutputFormat:
        return OutputFormat.PLAIN

    def export(self, results: list[ExplanationResult]) -> IntegrationResult:
        """Export results as individual files."""
        files_created = []

        for result in results:
            # Create safe filename
            filename = f"{result.element.type.name.lower()}_{result.element.name}.md"
            filename = filename.replace("/", "_").replace("\\", "_")
            filepath = self.output_dir / filename

            content = self._format_explanation(result)
            filepath.write_text(content)
            files_created.append(str(filepath))

        return IntegrationResult(
            success=True,
            message=f"Created {len(files_created)} explanation files",
            output="\n".join(files_created),
            output_path=str(self.output_dir),
            metadata={"files": files_created},
        )

    def _format_explanation(self, result: ExplanationResult) -> str:
        """Format explanation as Markdown."""
        lines = [
            f"# {result.element.name}",
            "",
            f"**Type:** {result.element.type.name}",
            f"**Confidence:** {result.confidence:.2f}",
            f"**Level:** {result.level.name}",
            "",
            "## Explanation",
            "",
            result.explanation,
        ]

        if self.config.include_source and result.element.source_code:
            lines.extend(
                [
                    "",
                    "## Source Code",
                    "",
                    "```python",
                    result.element.source_code,
                    "```",
                ]
            )

        return "\n".join(lines)


class AdapterRegistry:
    """Registry for integration adapters."""

    def __init__(self):
        """Initialize registry with default adapters."""
        self._adapters: dict[str, type[IntegrationAdapter]] = {
            "json": JSONAdapter,
            "sarif": SARIFAdapter,
            "github": GitHubAdapter,
            "gitlab": GitLabAdapter,
            "junit": JUnitAdapter,
            "vscode": VSCodeAdapter,
        }

    def register(self, name: str, adapter_class: type[IntegrationAdapter]) -> "AdapterRegistry":
        """Register a custom adapter."""
        self._adapters[name] = adapter_class
        return self

    def get(self, name: str) -> type[IntegrationAdapter] | None:
        """Get an adapter class by name."""
        return self._adapters.get(name)

    def create(self, name: str, config: AdapterConfig | None = None) -> IntegrationAdapter | None:
        """Create an adapter instance."""
        adapter_class = self.get(name)
        if adapter_class:
            return adapter_class(config)
        return None

    def list_adapters(self) -> list[str]:
        """List all registered adapters."""
        return list(self._adapters.keys())


class IntegrationManager:
    """Manager for handling multiple integrations."""

    def __init__(self):
        """Initialize manager."""
        self.registry = AdapterRegistry()
        self._callbacks: list[Callable[[list[ExplanationResult]], None]] = []

    def add_callback(
        self, callback: Callable[[list[ExplanationResult]], None]
    ) -> "IntegrationManager":
        """Add a callback for when results are exported."""
        self._callbacks.append(callback)
        return self

    def export(
        self,
        results: list[ExplanationResult],
        adapter_name: str,
        config: AdapterConfig | None = None,
    ) -> IntegrationResult:
        """Export results using specified adapter."""
        adapter = self.registry.create(adapter_name, config)
        if not adapter:
            return IntegrationResult(
                success=False,
                message=f"Unknown adapter: {adapter_name}",
            )

        result = adapter.export(results)

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(results)
            except Exception:
                pass

        return result

    def export_multiple(
        self,
        results: list[ExplanationResult],
        adapters: list[tuple[str, AdapterConfig | None]],
    ) -> list[IntegrationResult]:
        """Export to multiple adapters."""
        return [self.export(results, name, config) for name, config in adapters]


# Convenience functions
def export_to_json(
    results: list[ExplanationResult],
    output_path: str | None = None,
) -> IntegrationResult:
    """Export results to JSON."""
    config = AdapterConfig(output_path=output_path)
    adapter = JSONAdapter(config)
    return adapter.export(results)


def export_to_sarif(
    results: list[ExplanationResult],
    output_path: str | None = None,
) -> IntegrationResult:
    """Export results to SARIF format."""
    config = AdapterConfig(output_path=output_path)
    adapter = SARIFAdapter(config)
    return adapter.export(results)


def export_to_github(
    results: list[ExplanationResult],
) -> IntegrationResult:
    """Export results as GitHub annotations."""
    adapter = GitHubAdapter()
    return adapter.export(results)


def export_to_gitlab(
    results: list[ExplanationResult],
    output_path: str | None = None,
) -> IntegrationResult:
    """Export results to GitLab Code Quality format."""
    config = AdapterConfig(output_path=output_path)
    adapter = GitLabAdapter(config)
    return adapter.export(results)


def create_adapter(
    name: str,
    output_path: str | None = None,
    **options: Any,
) -> IntegrationAdapter | None:
    """Create an integration adapter by name."""
    config = AdapterConfig(output_path=output_path, custom_options=options)
    registry = AdapterRegistry()
    return registry.create(name, config)


# Singleton manager
_default_manager: IntegrationManager | None = None


def get_integration_manager() -> IntegrationManager:
    """Get or create the default integration manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = IntegrationManager()
    return _default_manager


def reset_integration_manager() -> None:
    """Reset the default manager."""
    global _default_manager
    _default_manager = None


__all__ = [
    # Enums
    "IntegrationType",
    "OutputFormat",
    # Config and result
    "AdapterConfig",
    "IntegrationResult",
    # Base class
    "IntegrationAdapter",
    # Adapters
    "JSONAdapter",
    "SARIFAdapter",
    "GitHubAdapter",
    "GitLabAdapter",
    "JUnitAdapter",
    "VSCodeAdapter",
    "WebhookAdapter",
    "FileSystemAdapter",
    # Registry and manager
    "AdapterRegistry",
    "IntegrationManager",
    # Functions
    "export_to_json",
    "export_to_sarif",
    "export_to_github",
    "export_to_gitlab",
    "create_adapter",
    "get_integration_manager",
    "reset_integration_manager",
]
