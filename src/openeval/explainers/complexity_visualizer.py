"""Complexity visualizer module for visual representations of code complexity.

This module provides tools for generating visual representations of code
complexity metrics, including ASCII art, SVG graphs, and structured reports.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class VisualizationType(Enum):
    """Types of complexity visualizations."""

    ASCII_BAR = auto()
    ASCII_HEATMAP = auto()
    ASCII_TREE = auto()
    TEXT_REPORT = auto()
    SPARKLINE = auto()
    SVG_BAR = auto()
    SVG_PIE = auto()
    SVG_TREE = auto()
    HTML_REPORT = auto()


class VisualizerColorScheme(Enum):
    """Color schemes for visualizations."""

    DEFAULT = auto()
    TRAFFIC_LIGHT = auto()
    GRADIENT = auto()
    MONOCHROME = auto()
    HIGH_CONTRAST = auto()


class ComplexityGrade(Enum):
    """Complexity level classification."""

    TRIVIAL = "trivial"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


@dataclass
class ComplexityMetric:
    """A single complexity metric."""

    name: str
    value: float
    max_value: float = 100.0
    unit: str = ""
    description: str = ""
    threshold_low: float = 10.0
    threshold_moderate: float = 20.0
    threshold_high: float = 30.0
    threshold_critical: float = 50.0

    @property
    def normalized_value(self) -> float:
        """Get normalized value between 0 and 1."""
        return min(self.value / self.max_value, 1.0) if self.max_value > 0 else 0.0

    @property
    def percentage(self) -> float:
        """Get value as percentage of max."""
        return self.normalized_value * 100

    @property
    def level(self) -> ComplexityGrade:
        """Classify complexity level based on thresholds."""
        if self.value <= self.threshold_low / 2:
            return ComplexityGrade.TRIVIAL
        elif self.value <= self.threshold_low:
            return ComplexityGrade.LOW
        elif self.value <= self.threshold_moderate:
            return ComplexityGrade.MODERATE
        elif self.value <= self.threshold_high:
            return ComplexityGrade.HIGH
        elif self.value <= self.threshold_critical:
            return ComplexityGrade.VERY_HIGH
        return ComplexityGrade.CRITICAL


@dataclass
class FunctionComplexity:
    """Complexity metrics for a function."""

    name: str
    line_start: int
    line_end: int
    cyclomatic_complexity: float = 1.0
    cognitive_complexity: float = 0.0
    nesting_depth: int = 0
    lines_of_code: int = 0
    parameter_count: int = 0
    return_points: int = 1
    halstead_volume: float = 0.0
    maintainability_index: float = 100.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Calculate overall complexity score (0-100, lower is better)."""
        # Weighted average of normalized metrics
        cc_score = min(self.cyclomatic_complexity / 30, 1.0) * 30
        cog_score = min(self.cognitive_complexity / 25, 1.0) * 25
        nest_score = min(self.nesting_depth / 5, 1.0) * 15
        loc_score = min(self.lines_of_code / 100, 1.0) * 15
        param_score = min(self.parameter_count / 7, 1.0) * 10
        mi_score = max(0, (100 - self.maintainability_index)) * 0.05

        return cc_score + cog_score + nest_score + loc_score + param_score + mi_score


@dataclass
class ModuleComplexity:
    """Complexity metrics for a module."""

    name: str
    file_path: str
    functions: list[FunctionComplexity] = field(default_factory=list)
    total_lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    code_lines: int = 0
    import_count: int = 0
    class_count: int = 0
    function_count: int = 0
    global_variables: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def average_function_complexity(self) -> float:
        """Calculate average cyclomatic complexity of functions."""
        if not self.functions:
            return 0.0
        return sum(f.cyclomatic_complexity for f in self.functions) / len(self.functions)

    @property
    def max_function_complexity(self) -> float:
        """Get maximum function complexity."""
        if not self.functions:
            return 0.0
        return max(f.cyclomatic_complexity for f in self.functions)

    @property
    def overall_score(self) -> float:
        """Calculate overall module complexity score."""
        if not self.functions:
            return 0.0
        return sum(f.overall_score for f in self.functions) / len(self.functions)


@dataclass
class VisualizationOptions:
    """Options for controlling visualization output."""

    width: int = 60
    height: int = 20
    show_legend: bool = True
    show_values: bool = True
    show_labels: bool = True
    color_scheme: VisualizerColorScheme = VisualizerColorScheme.DEFAULT
    bar_character: str = "█"
    empty_character: str = "░"
    decimal_places: int = 1
    title: str | None = None
    description: str | None = None
    include_summary: bool = True


class ComplexityVisualizer(ABC):
    """Abstract base class for complexity visualizers."""

    @property
    @abstractmethod
    def visualization_type(self) -> VisualizationType:
        """Get the visualization type."""
        pass

    @abstractmethod
    def visualize_metric(self, metric: ComplexityMetric, options: VisualizationOptions) -> str:
        """Visualize a single complexity metric."""
        pass

    @abstractmethod
    def visualize_function(self, func: FunctionComplexity, options: VisualizationOptions) -> str:
        """Visualize function complexity."""
        pass

    @abstractmethod
    def visualize_module(self, module: ModuleComplexity, options: VisualizationOptions) -> str:
        """Visualize module complexity."""
        pass


class ASCIIBarVisualizer(ComplexityVisualizer):
    """ASCII bar chart visualizer."""

    @property
    def visualization_type(self) -> VisualizationType:
        return VisualizationType.ASCII_BAR

    def visualize_metric(self, metric: ComplexityMetric, options: VisualizationOptions) -> str:
        """Visualize a metric as an ASCII bar."""
        bar_width = options.width - 20  # Leave space for label and value
        filled = int(metric.normalized_value * bar_width)
        empty = bar_width - filled

        bar = options.bar_character * filled + options.empty_character * empty
        level_indicator = self._level_indicator(metric.level)

        label = metric.name[:15].ljust(15)
        value = f"{metric.value:.{options.decimal_places}f}"

        if options.show_values:
            return f"{label} [{bar}] {value} {level_indicator}"
        return f"{label} [{bar}] {level_indicator}"

    def visualize_function(self, func: FunctionComplexity, options: VisualizationOptions) -> str:
        """Visualize function complexity as ASCII bars."""
        lines: list[str] = []

        if options.title:
            lines.append(f"╔{'═' * (options.width - 2)}╗")
            lines.append(f"║ {options.title.center(options.width - 4)} ║")
            lines.append(f"╠{'═' * (options.width - 2)}╣")
        else:
            lines.append(f"╔{'═' * (options.width - 2)}╗")
            lines.append(
                f"║ Function: {func.name[:options.width - 14].ljust(options.width - 14)} ║"
            )
            lines.append(f"╠{'═' * (options.width - 2)}╣")

        # Create metrics
        metrics = [
            ComplexityMetric(
                "Cyclomatic",
                func.cyclomatic_complexity,
                30,
                threshold_low=5,
                threshold_moderate=10,
                threshold_high=20,
                threshold_critical=30,
            ),
            ComplexityMetric(
                "Cognitive",
                func.cognitive_complexity,
                25,
                threshold_low=5,
                threshold_moderate=10,
                threshold_high=15,
                threshold_critical=25,
            ),
            ComplexityMetric(
                "Nesting",
                func.nesting_depth,
                5,
                threshold_low=2,
                threshold_moderate=3,
                threshold_high=4,
                threshold_critical=5,
            ),
            ComplexityMetric(
                "LOC",
                func.lines_of_code,
                100,
                threshold_low=20,
                threshold_moderate=40,
                threshold_high=60,
                threshold_critical=100,
            ),
            ComplexityMetric(
                "Parameters",
                func.parameter_count,
                7,
                threshold_low=3,
                threshold_moderate=4,
                threshold_high=5,
                threshold_critical=7,
            ),
            ComplexityMetric(
                "Maintainability",
                func.maintainability_index,
                100,
                threshold_low=80,
                threshold_moderate=65,
                threshold_high=50,
                threshold_critical=20,
            ),
        ]

        inner_options = VisualizationOptions(
            width=options.width - 4,
            show_values=options.show_values,
            bar_character=options.bar_character,
            empty_character=options.empty_character,
            decimal_places=options.decimal_places,
        )

        for metric in metrics:
            bar_line = self.visualize_metric(metric, inner_options)
            lines.append(f"║ {bar_line.ljust(options.width - 4)} ║")

        lines.append(f"╠{'═' * (options.width - 2)}╣")
        score = f"Overall Score: {func.overall_score:.1f}/100"
        lines.append(f"║ {score.center(options.width - 4)} ║")
        lines.append(f"╚{'═' * (options.width - 2)}╝")

        return "\n".join(lines)

    def visualize_module(self, module: ModuleComplexity, options: VisualizationOptions) -> str:
        """Visualize module complexity."""
        lines: list[str] = []
        header = f"Module: {module.name}"

        lines.append("=" * options.width)
        lines.append(header.center(options.width))
        lines.append("=" * options.width)
        lines.append("")

        # Module statistics
        lines.append("Module Statistics:")
        lines.append(f"  Total Lines: {module.total_lines}")
        lines.append(f"  Code Lines: {module.code_lines}")
        lines.append(f"  Comment Lines: {module.comment_lines}")
        lines.append(f"  Functions: {module.function_count}")
        lines.append(f"  Classes: {module.class_count}")
        lines.append(f"  Avg Complexity: {module.average_function_complexity:.1f}")
        lines.append(f"  Max Complexity: {module.max_function_complexity:.1f}")
        lines.append("")

        # Function complexity distribution
        if module.functions:
            lines.append("Function Complexity Distribution:")
            lines.append("-" * options.width)

            sorted_funcs = sorted(
                module.functions, key=lambda f: f.cyclomatic_complexity, reverse=True
            )

            inner_options = VisualizationOptions(
                width=options.width - 4,
                show_values=True,
                bar_character=options.bar_character,
                empty_character=options.empty_character,
                decimal_places=0,
            )

            for func in sorted_funcs[:10]:  # Top 10 most complex
                metric = ComplexityMetric(
                    name=func.name[:20],
                    value=func.cyclomatic_complexity,
                    max_value=30,
                )
                lines.append(self.visualize_metric(metric, inner_options))

        return "\n".join(lines)

    def _level_indicator(self, level: ComplexityGrade) -> str:
        """Get level indicator symbol."""
        indicators = {
            ComplexityGrade.TRIVIAL: "●",
            ComplexityGrade.LOW: "●",
            ComplexityGrade.MODERATE: "◐",
            ComplexityGrade.HIGH: "◑",
            ComplexityGrade.VERY_HIGH: "○",
            ComplexityGrade.CRITICAL: "✗",
        }
        return indicators.get(level, "●")


class ASCIIHeatmapVisualizer(ComplexityVisualizer):
    """ASCII heatmap visualizer for complexity data."""

    HEAT_CHARS = " ░▒▓█"

    @property
    def visualization_type(self) -> VisualizationType:
        return VisualizationType.ASCII_HEATMAP

    def visualize_metric(self, metric: ComplexityMetric, options: VisualizationOptions) -> str:
        """Visualize a metric as a heat cell."""
        heat_index = int(metric.normalized_value * (len(self.HEAT_CHARS) - 1))
        char = self.HEAT_CHARS[heat_index]
        return f"{metric.name}: {char * 3} ({metric.value:.1f})"

    def visualize_function(self, func: FunctionComplexity, options: VisualizationOptions) -> str:
        """Visualize function as a heatmap."""
        metrics = [
            ("CC", func.cyclomatic_complexity / 30),
            ("Cog", func.cognitive_complexity / 25),
            ("Nest", func.nesting_depth / 5),
            ("LOC", func.lines_of_code / 100),
            ("Param", func.parameter_count / 7),
        ]

        lines: list[str] = []
        lines.append(f"Function: {func.name}")
        lines.append("-" * (len(func.name) + 10))

        heat_row = ""
        for name, normalized in metrics:
            heat_index = min(int(normalized * (len(self.HEAT_CHARS) - 1)), len(self.HEAT_CHARS) - 1)
            heat_row += self.HEAT_CHARS[heat_index] * 3 + " "

        lines.append(heat_row)
        lines.append("  ".join(m[0] for m in metrics))

        return "\n".join(lines)

    def visualize_module(self, module: ModuleComplexity, options: VisualizationOptions) -> str:
        """Visualize module as heatmap grid."""
        lines: list[str] = []
        lines.append(f"Module Complexity Heatmap: {module.name}")
        lines.append("=" * options.width)
        lines.append("")

        if not module.functions:
            lines.append("No functions to display")
            return "\n".join(lines)

        # Header
        lines.append("Function          CC  Cog Nest LOC Param")
        lines.append("-" * 45)

        for func in module.functions:
            name = func.name[:15].ljust(15)
            metrics = [
                func.cyclomatic_complexity / 30,
                func.cognitive_complexity / 25,
                func.nesting_depth / 5,
                func.lines_of_code / 100,
                func.parameter_count / 7,
            ]

            heat_cells = ""
            for val in metrics:
                heat_index = min(int(val * (len(self.HEAT_CHARS) - 1)), len(self.HEAT_CHARS) - 1)
                heat_cells += self.HEAT_CHARS[heat_index] * 3 + " "

            lines.append(f"{name} {heat_cells}")

        if options.show_legend:
            lines.append("")
            lines.append(
                "Legend: " + " ".join(f"{c}={i * 25}%" for i, c in enumerate(self.HEAT_CHARS))
            )

        return "\n".join(lines)


class SparklineVisualizer(ComplexityVisualizer):
    """Sparkline visualizer for complexity trends."""

    SPARK_CHARS = "▁▂▃▄▅▆▇█"

    @property
    def visualization_type(self) -> VisualizationType:
        return VisualizationType.SPARKLINE

    def visualize_metric(self, metric: ComplexityMetric, options: VisualizationOptions) -> str:
        """Visualize a metric as a sparkline character."""
        spark_index = int(metric.normalized_value * (len(self.SPARK_CHARS) - 1))
        return self.SPARK_CHARS[spark_index]

    def visualize_function(self, func: FunctionComplexity, options: VisualizationOptions) -> str:
        """Create sparkline for function metrics."""
        metrics = [
            func.cyclomatic_complexity / 30,
            func.cognitive_complexity / 25,
            func.nesting_depth / 5,
            func.lines_of_code / 100,
            func.parameter_count / 7,
        ]

        sparkline = ""
        for val in metrics:
            spark_index = min(int(val * (len(self.SPARK_CHARS) - 1)), len(self.SPARK_CHARS) - 1)
            sparkline += self.SPARK_CHARS[spark_index]

        return f"{func.name}: {sparkline}"

    def visualize_module(self, module: ModuleComplexity, options: VisualizationOptions) -> str:
        """Create sparklines for all functions in module."""
        lines: list[str] = []
        lines.append(f"Complexity Sparklines: {module.name}")
        lines.append("-" * 40)

        for func in module.functions:
            lines.append(self.visualize_function(func, options))

        return "\n".join(lines)


class TextReportVisualizer(ComplexityVisualizer):
    """Text-based report visualizer."""

    @property
    def visualization_type(self) -> VisualizationType:
        return VisualizationType.TEXT_REPORT

    def visualize_metric(self, metric: ComplexityMetric, options: VisualizationOptions) -> str:
        """Generate text report for a metric."""
        level = metric.level.value.replace("_", " ").title()
        return f"- {metric.name}: {metric.value:.{options.decimal_places}f} ({level})"

    def visualize_function(self, func: FunctionComplexity, options: VisualizationOptions) -> str:
        """Generate detailed text report for function."""
        lines: list[str] = []

        lines.append(f"FUNCTION COMPLEXITY REPORT: {func.name}")
        lines.append("=" * 60)
        lines.append(f"Location: Lines {func.line_start}-{func.line_end}")
        lines.append("")

        lines.append("METRICS:")
        lines.append("-" * 30)
        lines.append(f"  Cyclomatic Complexity:  {func.cyclomatic_complexity}")
        lines.append(f"  Cognitive Complexity:   {func.cognitive_complexity}")
        lines.append(f"  Nesting Depth:          {func.nesting_depth}")
        lines.append(f"  Lines of Code:          {func.lines_of_code}")
        lines.append(f"  Parameter Count:        {func.parameter_count}")
        lines.append(f"  Return Points:          {func.return_points}")
        lines.append(f"  Halstead Volume:        {func.halstead_volume:.1f}")
        lines.append(f"  Maintainability Index:  {func.maintainability_index:.1f}")
        lines.append("")

        lines.append("ASSESSMENT:")
        lines.append("-" * 30)
        score = func.overall_score
        if score <= 20:
            assessment = "Excellent - Simple and maintainable"
        elif score <= 40:
            assessment = "Good - Acceptable complexity"
        elif score <= 60:
            assessment = "Fair - Consider refactoring"
        elif score <= 80:
            assessment = "Poor - Needs refactoring"
        else:
            assessment = "Critical - Immediate attention required"

        lines.append(f"  Overall Score: {score:.1f}/100")
        lines.append(f"  Assessment: {assessment}")

        if options.include_summary:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            lines.append("-" * 30)
            lines.extend(self._generate_recommendations(func))

        return "\n".join(lines)

    def visualize_module(self, module: ModuleComplexity, options: VisualizationOptions) -> str:
        """Generate detailed text report for module."""
        lines: list[str] = []

        lines.append("=" * 70)
        lines.append(f"MODULE COMPLEXITY REPORT: {module.name}")
        lines.append("=" * 70)
        lines.append(f"File: {module.file_path}")
        lines.append("")

        lines.append("OVERVIEW:")
        lines.append("-" * 40)
        lines.append(f"  Total Lines:        {module.total_lines}")
        lines.append(f"  Code Lines:         {module.code_lines}")
        lines.append(f"  Comment Lines:      {module.comment_lines}")
        lines.append(f"  Blank Lines:        {module.blank_lines}")
        lines.append(f"  Classes:            {module.class_count}")
        lines.append(f"  Functions:          {module.function_count}")
        lines.append(f"  Imports:            {module.import_count}")
        lines.append(f"  Global Variables:   {module.global_variables}")
        lines.append("")

        lines.append("COMPLEXITY SUMMARY:")
        lines.append("-" * 40)
        lines.append(f"  Average Complexity: {module.average_function_complexity:.1f}")
        lines.append(f"  Maximum Complexity: {module.max_function_complexity:.1f}")
        lines.append(f"  Overall Score:      {module.overall_score:.1f}/100")
        lines.append("")

        if module.functions:
            lines.append("FUNCTION BREAKDOWN:")
            lines.append("-" * 40)

            # Sort by complexity
            sorted_funcs = sorted(
                module.functions, key=lambda f: f.cyclomatic_complexity, reverse=True
            )

            for func in sorted_funcs:
                cc_level = self._get_complexity_level(func.cyclomatic_complexity)
                lines.append(
                    f"  {func.name[:30].ljust(30)} CC={func.cyclomatic_complexity:>3.0f} [{cc_level}]"
                )

        return "\n".join(lines)

    def _generate_recommendations(self, func: FunctionComplexity) -> list[str]:
        """Generate recommendations based on metrics."""
        recs: list[str] = []

        if func.cyclomatic_complexity > 10:
            recs.append("  - Consider breaking down complex conditionals")

        if func.nesting_depth > 3:
            recs.append("  - Reduce nesting by using guard clauses or early returns")

        if func.lines_of_code > 50:
            recs.append("  - Extract parts of the function into smaller helper functions")

        if func.parameter_count > 4:
            recs.append("  - Consider using a parameter object or builder pattern")

        if func.return_points > 3:
            recs.append("  - Consider consolidating return statements")

        if func.maintainability_index < 65:
            recs.append("  - Add documentation and simplify logic")

        if not recs:
            recs.append("  - No immediate improvements needed")

        return recs

    def _get_complexity_level(self, cc: float) -> str:
        """Get complexity level string."""
        if cc <= 5:
            return "Simple"
        elif cc <= 10:
            return "Moderate"
        elif cc <= 20:
            return "Complex"
        elif cc <= 30:
            return "Very Complex"
        return "Critical"


class SVGBarVisualizer(ComplexityVisualizer):
    """SVG bar chart visualizer."""

    @property
    def visualization_type(self) -> VisualizationType:
        return VisualizationType.SVG_BAR

    def visualize_metric(self, metric: ComplexityMetric, options: VisualizationOptions) -> str:
        """Generate SVG bar for a metric."""
        width = int(metric.normalized_value * (options.width - 100))
        color = self._get_level_color(metric.level)

        return f"""<g>
  <text x="5" y="15">{metric.name}</text>
  <rect x="100" y="5" width="{width}" height="15" fill="{color}"/>
  <text x="{105 + width}" y="15">{metric.value:.1f}</text>
</g>"""

    def visualize_function(self, func: FunctionComplexity, options: VisualizationOptions) -> str:
        """Generate SVG visualization for function."""
        metrics = [
            ComplexityMetric("Cyclomatic", func.cyclomatic_complexity, 30),
            ComplexityMetric("Cognitive", func.cognitive_complexity, 25),
            ComplexityMetric("Nesting", func.nesting_depth, 5),
            ComplexityMetric("LOC", func.lines_of_code, 100),
            ComplexityMetric("Params", func.parameter_count, 7),
        ]

        height = 40 + len(metrics) * 30
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{options.width}" height="{height}">'
        ]
        svg_parts.append(f'<text x="10" y="25" font-weight="bold">{func.name}</text>')

        for i, metric in enumerate(metrics):
            y_offset = 40 + i * 30
            width = int(metric.normalized_value * (options.width - 150))
            color = self._get_level_color(metric.level)

            svg_parts.append(f'<text x="10" y="{y_offset + 15}">{metric.name}</text>')
            svg_parts.append(
                f'<rect x="100" y="{y_offset}" width="{width}" height="20" fill="{color}" rx="3"/>'
            )
            svg_parts.append(
                f'<text x="{110 + width}" y="{y_offset + 15}">{metric.value:.1f}</text>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def visualize_module(self, module: ModuleComplexity, options: VisualizationOptions) -> str:
        """Generate SVG visualization for module."""
        func_count = min(len(module.functions), 15)
        height = 80 + func_count * 25

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{options.width}" height="{height}">'
        ]
        svg_parts.append(
            f'<text x="10" y="25" font-weight="bold" font-size="16">{module.name}</text>'
        )
        svg_parts.append(
            f'<text x="10" y="50">Avg Complexity: {module.average_function_complexity:.1f}</text>'
        )

        sorted_funcs = sorted(
            module.functions, key=lambda f: f.cyclomatic_complexity, reverse=True
        )[:func_count]

        for i, func in enumerate(sorted_funcs):
            y_offset = 70 + i * 25
            normalized = min(func.cyclomatic_complexity / 30, 1.0)
            width = int(normalized * (options.width - 200))
            metric = ComplexityMetric("", func.cyclomatic_complexity, 30)
            color = self._get_level_color(metric.level)

            name = func.name[:20]
            svg_parts.append(f'<text x="10" y="{y_offset + 12}">{name}</text>')
            svg_parts.append(
                f'<rect x="150" y="{y_offset}" width="{width}" height="18" fill="{color}" rx="2"/>'
            )
            svg_parts.append(
                f'<text x="{160 + width}" y="{y_offset + 12}">{func.cyclomatic_complexity:.0f}</text>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _get_level_color(self, level: ComplexityGrade) -> str:
        """Get color for complexity level."""
        colors = {
            ComplexityGrade.TRIVIAL: "#4CAF50",
            ComplexityGrade.LOW: "#8BC34A",
            ComplexityGrade.MODERATE: "#FFC107",
            ComplexityGrade.HIGH: "#FF9800",
            ComplexityGrade.VERY_HIGH: "#FF5722",
            ComplexityGrade.CRITICAL: "#F44336",
        }
        return colors.get(level, "#9E9E9E")


class SVGPieVisualizer(ComplexityVisualizer):
    """SVG pie chart visualizer for complexity distribution."""

    @property
    def visualization_type(self) -> VisualizationType:
        return VisualizationType.SVG_PIE

    def visualize_metric(self, metric: ComplexityMetric, options: VisualizationOptions) -> str:
        """Generate SVG arc for a metric (partial pie)."""
        # Single metric visualization as a gauge
        radius = 50
        # Note: angle could be used for more complex gauge visualizations
        # angle = metric.normalized_value * 270 - 135  # -135 to 135 degrees

        return f"""<svg width="120" height="120">
  <circle cx="60" cy="60" r="{radius}" fill="none" stroke="#e0e0e0" stroke-width="10"/>
  <circle cx="60" cy="60" r="{radius}" fill="none" stroke="{self._get_color(metric.normalized_value)}"
          stroke-width="10" stroke-dasharray="{metric.normalized_value * 314} 314"
          transform="rotate(-90 60 60)"/>
  <text x="60" y="65" text-anchor="middle" font-size="14">{metric.value:.1f}</text>
</svg>"""

    def visualize_function(self, func: FunctionComplexity, options: VisualizationOptions) -> str:
        """Generate pie chart showing metric distribution."""
        metrics = [
            ("Cyclomatic", func.cyclomatic_complexity / 30),
            ("Cognitive", func.cognitive_complexity / 25),
            ("Nesting", func.nesting_depth / 5),
            ("LOC", func.lines_of_code / 100),
            ("Params", func.parameter_count / 7),
        ]

        total = sum(m[1] for m in metrics)
        if total == 0:
            total = 1

        svg_parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200">']
        svg_parts.append(
            f'<text x="150" y="20" text-anchor="middle" font-weight="bold">{func.name}</text>'
        )

        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63"]
        cx, cy, r = 100, 110, 60
        start_angle = 0

        for i, (name, value) in enumerate(metrics):
            angle = (value / total) * 360
            end_angle = start_angle + angle

            # Calculate arc path
            large_arc = 1 if angle > 180 else 0
            start_rad = math.radians(start_angle - 90)
            end_rad = math.radians(end_angle - 90)

            x1 = cx + r * math.cos(start_rad)
            y1 = cy + r * math.sin(start_rad)
            x2 = cx + r * math.cos(end_rad)
            y2 = cy + r * math.sin(end_rad)

            path = f"M {cx},{cy} L {x1},{y1} A {r},{r} 0 {large_arc},1 {x2},{y2} Z"
            svg_parts.append(f'<path d="{path}" fill="{colors[i % len(colors)]}"/>')

            start_angle = end_angle

        # Legend
        for i, (name, _) in enumerate(metrics):
            y = 40 + i * 20
            svg_parts.append(f'<rect x="200" y="{y}" width="12" height="12" fill="{colors[i]}"/>')
            svg_parts.append(f'<text x="220" y="{y + 10}" font-size="10">{name}</text>')

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def visualize_module(self, module: ModuleComplexity, options: VisualizationOptions) -> str:
        """Generate pie chart for module complexity distribution."""
        # Count functions by complexity level
        level_counts: dict[str, int] = {
            "Simple (1-5)": 0,
            "Moderate (6-10)": 0,
            "Complex (11-20)": 0,
            "Very Complex (21+)": 0,
        }

        for func in module.functions:
            cc = func.cyclomatic_complexity
            if cc <= 5:
                level_counts["Simple (1-5)"] += 1
            elif cc <= 10:
                level_counts["Moderate (6-10)"] += 1
            elif cc <= 20:
                level_counts["Complex (11-20)"] += 1
            else:
                level_counts["Very Complex (21+)"] += 1

        total = sum(level_counts.values())
        if total == 0:
            return "<svg><text>No functions</text></svg>"

        svg_parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="350" height="250">']
        svg_parts.append(
            f'<text x="175" y="20" text-anchor="middle" font-weight="bold">{module.name}</text>'
        )

        colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
        cx, cy, r = 120, 140, 80
        start_angle = 0

        for i, (label, count) in enumerate(level_counts.items()):
            if count == 0:
                continue

            angle = (count / total) * 360
            end_angle = start_angle + angle

            large_arc = 1 if angle > 180 else 0
            start_rad = math.radians(start_angle - 90)
            end_rad = math.radians(end_angle - 90)

            x1 = cx + r * math.cos(start_rad)
            y1 = cy + r * math.sin(start_rad)
            x2 = cx + r * math.cos(end_rad)
            y2 = cy + r * math.sin(end_rad)

            path = f"M {cx},{cy} L {x1},{y1} A {r},{r} 0 {large_arc},1 {x2},{y2} Z"
            svg_parts.append(f'<path d="{path}" fill="{colors[i]}"/>')

            start_angle = end_angle

        # Legend
        for i, (label, count) in enumerate(level_counts.items()):
            y = 40 + i * 25
            svg_parts.append(f'<rect x="230" y="{y}" width="15" height="15" fill="{colors[i]}"/>')
            svg_parts.append(f'<text x="250" y="{y + 12}" font-size="11">{label}: {count}</text>')

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _get_color(self, value: float) -> str:
        """Get color based on normalized value."""
        if value <= 0.25:
            return "#4CAF50"
        elif value <= 0.5:
            return "#FFC107"
        elif value <= 0.75:
            return "#FF9800"
        return "#F44336"


class ComplexityVisualizationManager:
    """Manager for complexity visualizations."""

    def __init__(self) -> None:
        """Initialize with default visualizers."""
        self._visualizers: dict[VisualizationType, ComplexityVisualizer] = {}

        # Register default visualizers
        self.register_visualizer(ASCIIBarVisualizer())
        self.register_visualizer(ASCIIHeatmapVisualizer())
        self.register_visualizer(SparklineVisualizer())
        self.register_visualizer(TextReportVisualizer())
        self.register_visualizer(SVGBarVisualizer())
        self.register_visualizer(SVGPieVisualizer())

    def register_visualizer(self, visualizer: ComplexityVisualizer) -> None:
        """Register a visualizer."""
        self._visualizers[visualizer.visualization_type] = visualizer

    def visualize(
        self,
        data: FunctionComplexity | ModuleComplexity | ComplexityMetric,
        viz_type: VisualizationType,
        options: VisualizationOptions | None = None,
    ) -> str:
        """Visualize complexity data."""
        if options is None:
            options = VisualizationOptions()

        visualizer = self._visualizers.get(viz_type)
        if not visualizer:
            return f"Visualizer not found for type: {viz_type}"

        if isinstance(data, FunctionComplexity):
            return visualizer.visualize_function(data, options)
        elif isinstance(data, ModuleComplexity):
            return visualizer.visualize_module(data, options)
        elif isinstance(data, ComplexityMetric):
            return visualizer.visualize_metric(data, options)

        return "Unsupported data type"

    def available_types(self) -> list[VisualizationType]:
        """Get available visualization types."""
        return list(self._visualizers.keys())


# Global instance
_visualization_manager: ComplexityVisualizationManager | None = None


def get_visualization_manager() -> ComplexityVisualizationManager:
    """Get or create global visualization manager."""
    global _visualization_manager
    if _visualization_manager is None:
        _visualization_manager = ComplexityVisualizationManager()
    return _visualization_manager


def reset_visualization_manager() -> None:
    """Reset global visualization manager."""
    global _visualization_manager
    _visualization_manager = None


# Convenience functions
def create_complexity_metric(
    name: str,
    value: float,
    max_value: float = 100.0,
    **kwargs: Any,
) -> ComplexityMetric:
    """Create a complexity metric."""
    return ComplexityMetric(name=name, value=value, max_value=max_value, **kwargs)


def create_function_complexity(
    name: str,
    line_start: int,
    line_end: int,
    cyclomatic_complexity: float = 1.0,
    **kwargs: Any,
) -> FunctionComplexity:
    """Create function complexity data."""
    return FunctionComplexity(
        name=name,
        line_start=line_start,
        line_end=line_end,
        cyclomatic_complexity=cyclomatic_complexity,
        **kwargs,
    )


def create_module_complexity(
    name: str,
    file_path: str,
    **kwargs: Any,
) -> ModuleComplexity:
    """Create module complexity data."""
    return ModuleComplexity(name=name, file_path=file_path, **kwargs)


def visualize_complexity(
    data: FunctionComplexity | ModuleComplexity | ComplexityMetric,
    viz_type: VisualizationType = VisualizationType.ASCII_BAR,
    options: VisualizationOptions | None = None,
) -> str:
    """Visualize complexity data using the global manager."""
    return get_visualization_manager().visualize(data, viz_type, options)


def visualize_as_ascii_bar(
    data: FunctionComplexity | ModuleComplexity,
    options: VisualizationOptions | None = None,
) -> str:
    """Visualize as ASCII bar chart."""
    return visualize_complexity(data, VisualizationType.ASCII_BAR, options)


def visualize_as_heatmap(
    data: FunctionComplexity | ModuleComplexity,
    options: VisualizationOptions | None = None,
) -> str:
    """Visualize as ASCII heatmap."""
    return visualize_complexity(data, VisualizationType.ASCII_HEATMAP, options)


def visualize_as_sparkline(
    data: FunctionComplexity | ModuleComplexity,
    options: VisualizationOptions | None = None,
) -> str:
    """Visualize as sparkline."""
    return visualize_complexity(data, VisualizationType.SPARKLINE, options)


def visualize_as_text_report(
    data: FunctionComplexity | ModuleComplexity,
    options: VisualizationOptions | None = None,
) -> str:
    """Visualize as text report."""
    return visualize_complexity(data, VisualizationType.TEXT_REPORT, options)


def visualize_as_svg_bar(
    data: FunctionComplexity | ModuleComplexity,
    options: VisualizationOptions | None = None,
) -> str:
    """Visualize as SVG bar chart."""
    return visualize_complexity(data, VisualizationType.SVG_BAR, options)


def visualize_as_svg_pie(
    data: FunctionComplexity | ModuleComplexity,
    options: VisualizationOptions | None = None,
) -> str:
    """Visualize as SVG pie chart."""
    return visualize_complexity(data, VisualizationType.SVG_PIE, options)
