"""
Enhanced error handling with context and recovery suggestions.

Provides structured error messages with:
- Clear problem descriptions
- Context about what was happening
- Actionable recovery suggestions
- Links to relevant documentation
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ErrorContext:
    """Structured error information with recovery suggestions."""

    error_type: str
    message: str
    context: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    docs_link: Optional[str] = None

    def __str__(self) -> str:
        """Format error with context and suggestions."""
        lines = [f"❌ {self.error_type}: {self.message}"]

        if self.context:
            lines.append("\n📋 Context:")
            for key, value in self.context.items():
                lines.append(f"  • {key}: {value}")

        if self.suggestions:
            lines.append("\n💡 Recovery suggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")

        if self.docs_link:
            lines.append(f"\n📖 Learn more: {self.docs_link}")

        return "\n".join(lines)


class ErrorContextFactory:
    """Factory for creating contextualized errors."""

    ERROR_TEMPLATES = {
        "spec_not_found": {
            "message": "Spec file not found",
            "suggestions": [
                "Check if the file path is correct",
                "Use 'pwd' to verify your current directory",
                "Try using an absolute path instead of relative path",
            ],
            "docs_link": "https://docs.openeval.org/configuration",
        },
        "spec_invalid": {
            "message": "Spec file is invalid JSON/YAML",
            "suggestions": [
                "Validate JSON/YAML syntax using an online validator",
                "Check for missing quotes or commas",
                "Run 'openeval validate <spec>' for detailed validation",
            ],
            "docs_link": "https://docs.openeval.org/spec-format",
        },
        "cache_corrupted": {
            "message": "Cache is corrupted",
            "suggestions": [
                "Clear cache with 'rm -rf ~/.openeval/cache'",
                "Run evaluation again (cache will be rebuilt)",
                "Check disk space to ensure adequate storage",
            ],
            "docs_link": "https://docs.openeval.org/caching",
        },
        "adapter_not_found": {
            "message": "Adapter not found in registry",
            "suggestions": [
                "List available adapters with 'openeval adapters list'",
                "Check adapter name spelling",
                "Install missing dependencies if needed",
            ],
            "docs_link": "https://docs.openeval.org/adapters",
        },
        "dataset_not_found": {
            "message": "Dataset not found",
            "suggestions": [
                "List available datasets with 'openeval datasets list'",
                "Check if dataset is downloaded with 'openeval datasets download'",
                "Verify dataset path or URL",
            ],
            "docs_link": "https://docs.openeval.org/datasets",
        },
        "metric_not_found": {
            "message": "Metric not found in registry",
            "suggestions": [
                "List available metrics with 'openeval metrics list'",
                "Check metric name spelling",
                "Install metric dependencies if needed",
            ],
            "docs_link": "https://docs.openeval.org/metrics",
        },
        "timeout": {
            "message": "Operation timed out",
            "suggestions": [
                "Increase timeout value with --timeout flag",
                "Check network connectivity",
                "Try running with fewer samples with --max-samples",
                "Check if external services (APIs) are responding",
            ],
            "docs_link": "https://docs.openeval.org/timeouts",
        },
        "rate_limit": {
            "message": "Rate limit exceeded",
            "suggestions": [
                "Reduce batch size with --batch-size flag",
                "Add delay between requests with --request-delay",
                "Check API quotas and limits",
                "Contact API provider for higher limits",
            ],
            "docs_link": "https://docs.openeval.org/rate-limiting",
        },
        "insufficient_resources": {
            "message": "Insufficient system resources",
            "suggestions": [
                "Close other applications to free memory",
                "Enable streaming with --stream flag",
                "Reduce batch size with --batch-size",
                "Split dataset into smaller chunks",
            ],
            "docs_link": "https://docs.openeval.org/resources",
        },
    }

    @staticmethod
    def create(
        error_type: str,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> ErrorContext:
        """Create an error context with optional template lookup."""
        template = ErrorContextFactory.ERROR_TEMPLATES.get(error_type, {})

        return ErrorContext(
            error_type=error_type,
            message=message or template.get("message", "Unknown error"),
            context=context,
            suggestions=suggestions or template.get("suggestions"),
            docs_link=template.get("docs_link"),
        )

    @staticmethod
    def add_template(error_type: str, template: Dict[str, Any]) -> None:
        """Register custom error template."""
        ErrorContextFactory.ERROR_TEMPLATES[error_type] = template
