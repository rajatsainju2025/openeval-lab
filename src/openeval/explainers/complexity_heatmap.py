"""Complexity heatmap for visualizing code complexity hotspots.

This module provides tools for generating visual representations of
code complexity, identifying hotspots, and tracking complexity trends.
"""

import ast
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .types import CodeElementType


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class ComplexityLevel(str, Enum):
    """Complexity severity levels."""

    TRIVIAL = "trivial"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of complexity metrics."""

    CYCLOMATIC = "cyclomatic"
    COGNITIVE = "cognitive"
    HALSTEAD = "halstead"
    LINES_OF_CODE = "lines_of_code"
    NESTING_DEPTH = "nesting_depth"
    PARAMETER_COUNT = "parameter_count"
    MAINTAINABILITY_INDEX = "maintainability_index"


class TrendDirection(str, Enum):
    """Direction of complexity trend."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ComplexityScore:
    """Complexity score for a code element."""

    element_name: str
    element_type: CodeElementType
    line_start: int
    line_end: int
    cyclomatic: int = 0
    cognitive: int = 0
    nesting_depth: int = 0
    lines_of_code: int = 0
    parameter_count: int = 0
    level: ComplexityLevel = ComplexityLevel.LOW
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "element_name": self.element_name,
            "element_type": self.element_type.value,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "cyclomatic": self.cyclomatic,
            "cognitive": self.cognitive,
            "nesting_depth": self.nesting_depth,
            "lines_of_code": self.lines_of_code,
            "parameter_count": self.parameter_count,
            "level": self.level.value,
            "score": self.score,
        }


@dataclass
class HeatmapCell:
    """A cell in the complexity heatmap."""

    line_start: int
    line_end: int
    score: float
    level: ComplexityLevel
    element_name: Optional[str] = None
    element_type: Optional[CodeElementType] = None
    color: str = ""
    tooltip: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "line_start": self.line_start,
            "line_end": self.line_end,
            "score": self.score,
            "level": self.level.value,
            "element_name": self.element_name,
            "color": self.color,
            "tooltip": self.tooltip,
        }


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""

    # Thresholds for complexity levels
    trivial_threshold: float = 5.0
    low_threshold: float = 10.0
    moderate_threshold: float = 20.0
    high_threshold: float = 30.0
    very_high_threshold: float = 50.0

    # Metric weights for combined score
    cyclomatic_weight: float = 0.3
    cognitive_weight: float = 0.3
    nesting_weight: float = 0.2
    loc_weight: float = 0.1
    parameter_weight: float = 0.1

    # Display options
    show_functions: bool = True
    show_classes: bool = True
    show_methods: bool = True
    color_scheme: str = "default"

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplexityHotspot:
    """A complexity hotspot in the code."""

    element_name: str
    element_type: CodeElementType
    line_start: int
    line_end: int
    score: float
    level: ComplexityLevel
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "element_name": self.element_name,
            "element_type": self.element_type.value,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "score": self.score,
            "level": self.level.value,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "priority": self.priority,
        }


@dataclass
class ComplexityTrend:
    """Complexity trend over time."""

    timestamps: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    direction: TrendDirection = TrendDirection.STABLE
    change_rate: float = 0.0
    prediction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamps": self.timestamps,
            "scores": self.scores,
            "direction": self.direction.value,
            "change_rate": self.change_rate,
            "prediction": self.prediction,
        }


@dataclass
class HeatmapReport:
    """Complete heatmap report."""

    cells: List[HeatmapCell] = field(default_factory=list)
    hotspots: List[ComplexityHotspot] = field(default_factory=list)
    scores: List[ComplexityScore] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def total_elements(self) -> int:
        """Total number of analyzed elements."""
        return len(self.scores)

    @property
    def hotspot_count(self) -> int:
        """Number of hotspots."""
        return len(self.hotspots)

    @property
    def average_score(self) -> float:
        """Average complexity score."""
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_elements": self.total_elements,
            "hotspot_count": self.hotspot_count,
            "average_score": self.average_score,
            "cells": [c.to_dict() for c in self.cells],
            "hotspots": [h.to_dict() for h in self.hotspots],
            "scores": [s.to_dict() for s in self.scores],
            "summary": self.summary,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Metric Calculators
# =============================================================================


class MetricCalculator(ABC):
    """Abstract base class for metric calculators."""

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Get the metric type."""
        pass

    @abstractmethod
    def calculate(self, node: ast.AST, code: str) -> float:
        """Calculate the metric value.

        Args:
            node: AST node to analyze.
            code: Original source code.

        Returns:
            Metric value.
        """
        pass


class CyclomaticComplexityCalculator(MetricCalculator):
    """Calculates cyclomatic complexity."""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.CYCLOMATIC

    def calculate(self, node: ast.AST, code: str) -> float:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points add complexity
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                if child.ifs:
                    complexity += len(child.ifs)

        return float(complexity)


class CognitiveComplexityCalculator(MetricCalculator):
    """Calculates cognitive complexity."""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.COGNITIVE

    def calculate(self, node: ast.AST, code: str) -> float:
        """Calculate cognitive complexity."""
        return self._calculate_cognitive(node, 0)

    def _calculate_cognitive(self, node: ast.AST, nesting: int) -> float:
        """Recursively calculate cognitive complexity."""
        complexity = 0.0

        if isinstance(node, (ast.If, ast.While, ast.For)):
            complexity += 1 + nesting
            nesting += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, ast.Try):
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1 + nesting
        elif isinstance(node, ast.Lambda):
            complexity += 1
        elif isinstance(node, ast.Break) or isinstance(node, ast.Continue):
            complexity += 1

        # Recursively process children
        for child in ast.iter_child_nodes(node):
            complexity += self._calculate_cognitive(child, nesting)

        return complexity


class NestingDepthCalculator(MetricCalculator):
    """Calculates maximum nesting depth."""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.NESTING_DEPTH

    def calculate(self, node: ast.AST, code: str) -> float:
        """Calculate maximum nesting depth."""
        return float(self._max_depth(node, 0))

    def _max_depth(self, node: ast.AST, current_depth: int) -> int:
        """Recursively find maximum depth."""
        max_depth = current_depth

        nesting_nodes = (
            ast.If,
            ast.While,
            ast.For,
            ast.With,
            ast.Try,
            ast.FunctionDef,
            ast.ClassDef,
        )

        for child in ast.iter_child_nodes(node):
            if isinstance(child, nesting_nodes):
                child_depth = self._max_depth(child, current_depth + 1)
            else:
                child_depth = self._max_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)

        return max_depth


class LinesOfCodeCalculator(MetricCalculator):
    """Calculates lines of code."""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.LINES_OF_CODE

    def calculate(self, node: ast.AST, code: str) -> float:
        """Calculate lines of code."""
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            return float(node.end_lineno - node.lineno + 1)
        return 0.0


class ParameterCountCalculator(MetricCalculator):
    """Calculates parameter count for functions."""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.PARAMETER_COUNT

    def calculate(self, node: ast.AST, code: str) -> float:
        """Calculate parameter count."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            count = len(args.args) + len(args.kwonlyargs)
            if args.vararg:
                count += 1
            if args.kwarg:
                count += 1
            return float(count)
        return 0.0


class MaintainabilityIndexCalculator(MetricCalculator):
    """Calculates maintainability index."""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.MAINTAINABILITY_INDEX

    def __init__(self):
        self._cyclomatic = CyclomaticComplexityCalculator()
        self._loc = LinesOfCodeCalculator()

    def calculate(self, node: ast.AST, code: str) -> float:
        """Calculate maintainability index.

        Uses the Visual Studio formula:
        MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(L)
        where V = Halstead Volume, G = Cyclomatic, L = LOC

        Simplified version without Halstead.
        """
        loc = max(1, self._loc.calculate(node, code))
        cyclomatic = max(1, self._cyclomatic.calculate(node, code))

        # Simplified MI calculation
        mi = 171 - 0.23 * cyclomatic - 16.2 * math.log(loc)
        mi = max(0, min(100, mi))  # Clamp to 0-100

        return mi


# =============================================================================
# Color Schemes
# =============================================================================


class ColorScheme(ABC):
    """Abstract base class for color schemes."""

    @abstractmethod
    def get_color(self, level: ComplexityLevel) -> str:
        """Get color for complexity level."""
        pass


class DefaultColorScheme(ColorScheme):
    """Default color scheme (green to red)."""

    def get_color(self, level: ComplexityLevel) -> str:
        """Get color for complexity level."""
        colors = {
            ComplexityLevel.TRIVIAL: "#00FF00",  # Green
            ComplexityLevel.LOW: "#7FFF00",  # Light green
            ComplexityLevel.MODERATE: "#FFFF00",  # Yellow
            ComplexityLevel.HIGH: "#FF7F00",  # Orange
            ComplexityLevel.VERY_HIGH: "#FF0000",  # Red
            ComplexityLevel.CRITICAL: "#8B0000",  # Dark red
        }
        return colors.get(level, "#FFFFFF")


class GrayscaleColorScheme(ColorScheme):
    """Grayscale color scheme."""

    def get_color(self, level: ComplexityLevel) -> str:
        """Get grayscale color for complexity level."""
        colors = {
            ComplexityLevel.TRIVIAL: "#FFFFFF",
            ComplexityLevel.LOW: "#DDDDDD",
            ComplexityLevel.MODERATE: "#AAAAAA",
            ComplexityLevel.HIGH: "#777777",
            ComplexityLevel.VERY_HIGH: "#444444",
            ComplexityLevel.CRITICAL: "#000000",
        }
        return colors.get(level, "#FFFFFF")


class BlueRedColorScheme(ColorScheme):
    """Blue to red color scheme."""

    def get_color(self, level: ComplexityLevel) -> str:
        """Get blue-red color for complexity level."""
        colors = {
            ComplexityLevel.TRIVIAL: "#0000FF",  # Blue
            ComplexityLevel.LOW: "#4444FF",
            ComplexityLevel.MODERATE: "#8888FF",
            ComplexityLevel.HIGH: "#FF8888",
            ComplexityLevel.VERY_HIGH: "#FF4444",
            ComplexityLevel.CRITICAL: "#FF0000",  # Red
        }
        return colors.get(level, "#FFFFFF")


# =============================================================================
# Main Complexity Heatmap
# =============================================================================


class ComplexityHeatmap:
    """Generates complexity heatmaps for code analysis."""

    def __init__(self, config: Optional[HeatmapConfig] = None):
        """Initialize complexity heatmap.

        Args:
            config: Optional configuration.
        """
        self.config = config or HeatmapConfig()
        self._calculators: Dict[MetricType, MetricCalculator] = {
            MetricType.CYCLOMATIC: CyclomaticComplexityCalculator(),
            MetricType.COGNITIVE: CognitiveComplexityCalculator(),
            MetricType.NESTING_DEPTH: NestingDepthCalculator(),
            MetricType.LINES_OF_CODE: LinesOfCodeCalculator(),
            MetricType.PARAMETER_COUNT: ParameterCountCalculator(),
            MetricType.MAINTAINABILITY_INDEX: MaintainabilityIndexCalculator(),
        }
        self._color_schemes: Dict[str, ColorScheme] = {
            "default": DefaultColorScheme(),
            "grayscale": GrayscaleColorScheme(),
            "blue_red": BlueRedColorScheme(),
        }
        self._history: List[Tuple[str, float]] = []

    def analyze(self, code: str) -> HeatmapReport:
        """Analyze code and generate heatmap report.

        Args:
            code: Source code to analyze.

        Returns:
            Heatmap report with all analysis results.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return HeatmapReport()

        scores = []
        cells = []

        # Analyze all functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self.config.show_functions or self.config.show_methods:
                    score = self._analyze_node(node, code, CodeElementType.FUNCTION)
                    scores.append(score)
                    cells.append(self._create_cell(score))

            elif isinstance(node, ast.ClassDef):
                if self.config.show_classes:
                    score = self._analyze_node(node, code, CodeElementType.CLASS)
                    scores.append(score)
                    cells.append(self._create_cell(score))

        # Find hotspots
        hotspots = self._find_hotspots(scores)

        # Generate summary
        summary = self._generate_summary(scores, hotspots)

        # Record for trend tracking
        avg_score = sum(s.score for s in scores) / len(scores) if scores else 0.0
        self._history.append((datetime.utcnow().isoformat(), avg_score))

        return HeatmapReport(
            cells=cells,
            hotspots=hotspots,
            scores=scores,
            summary=summary,
        )

    def get_trend(self) -> ComplexityTrend:
        """Get complexity trend from history.

        Returns:
            Complexity trend analysis.
        """
        if len(self._history) < 2:
            return ComplexityTrend(
                timestamps=[h[0] for h in self._history],
                scores=[h[1] for h in self._history],
                direction=TrendDirection.STABLE,
            )

        timestamps = [h[0] for h in self._history]
        scores = [h[1] for h in self._history]

        # Calculate trend direction
        recent_avg = sum(scores[-3:]) / min(3, len(scores))
        older_avg = sum(scores[:-3]) / max(1, len(scores) - 3) if len(scores) > 3 else scores[0]

        change_rate = (recent_avg - older_avg) / max(1, older_avg)

        if change_rate < -0.05:
            direction = TrendDirection.IMPROVING
        elif change_rate > 0.05:
            direction = TrendDirection.DEGRADING
        else:
            direction = TrendDirection.STABLE

        # Simple prediction (linear extrapolation)
        prediction = recent_avg + change_rate * recent_avg

        return ComplexityTrend(
            timestamps=timestamps,
            scores=scores,
            direction=direction,
            change_rate=change_rate,
            prediction=prediction,
        )

    def clear_history(self) -> None:
        """Clear complexity history."""
        self._history.clear()

    def _analyze_node(
        self, node: ast.AST, code: str, element_type: CodeElementType
    ) -> ComplexityScore:
        """Analyze a single AST node."""
        name = getattr(node, "name", "unknown")
        line_start = getattr(node, "lineno", 0)
        line_end = getattr(node, "end_lineno", line_start)

        # Calculate metrics
        cyclomatic = int(self._calculators[MetricType.CYCLOMATIC].calculate(node, code))
        cognitive = int(self._calculators[MetricType.COGNITIVE].calculate(node, code))
        nesting = int(self._calculators[MetricType.NESTING_DEPTH].calculate(node, code))
        loc = int(self._calculators[MetricType.LINES_OF_CODE].calculate(node, code))
        params = int(self._calculators[MetricType.PARAMETER_COUNT].calculate(node, code))

        # Calculate combined score
        score = (
            cyclomatic * self.config.cyclomatic_weight
            + cognitive * self.config.cognitive_weight
            + nesting * 5 * self.config.nesting_weight  # Scale nesting
            + loc * 0.1 * self.config.loc_weight  # Scale LOC
            + params * 2 * self.config.parameter_weight  # Scale params
        )

        level = self._get_complexity_level(score)

        return ComplexityScore(
            element_name=name,
            element_type=element_type,
            line_start=line_start,
            line_end=line_end,
            cyclomatic=cyclomatic,
            cognitive=cognitive,
            nesting_depth=nesting,
            lines_of_code=loc,
            parameter_count=params,
            level=level,
            score=score,
        )

    def _get_complexity_level(self, score: float) -> ComplexityLevel:
        """Get complexity level from score."""
        if score <= self.config.trivial_threshold:
            return ComplexityLevel.TRIVIAL
        elif score <= self.config.low_threshold:
            return ComplexityLevel.LOW
        elif score <= self.config.moderate_threshold:
            return ComplexityLevel.MODERATE
        elif score <= self.config.high_threshold:
            return ComplexityLevel.HIGH
        elif score <= self.config.very_high_threshold:
            return ComplexityLevel.VERY_HIGH
        else:
            return ComplexityLevel.CRITICAL

    def _create_cell(self, score: ComplexityScore) -> HeatmapCell:
        """Create heatmap cell from score."""
        color_scheme = self._color_schemes.get(
            self.config.color_scheme, self._color_schemes["default"]
        )

        tooltip = (
            f"{score.element_name}\n"
            f"Cyclomatic: {score.cyclomatic}\n"
            f"Cognitive: {score.cognitive}\n"
            f"Nesting: {score.nesting_depth}\n"
            f"LOC: {score.lines_of_code}"
        )

        return HeatmapCell(
            line_start=score.line_start,
            line_end=score.line_end,
            score=score.score,
            level=score.level,
            element_name=score.element_name,
            element_type=score.element_type,
            color=color_scheme.get_color(score.level),
            tooltip=tooltip,
        )

    def _find_hotspots(self, scores: List[ComplexityScore]) -> List[ComplexityHotspot]:
        """Find complexity hotspots."""
        hotspots = []

        # Threshold for hotspot (high or above)
        for score in scores:
            if score.level in (
                ComplexityLevel.HIGH,
                ComplexityLevel.VERY_HIGH,
                ComplexityLevel.CRITICAL,
            ):
                issues = []
                suggestions = []

                if score.cyclomatic > 10:
                    issues.append(f"High cyclomatic complexity ({score.cyclomatic})")
                    suggestions.append("Consider breaking into smaller functions")

                if score.cognitive > 15:
                    issues.append(f"High cognitive complexity ({score.cognitive})")
                    suggestions.append("Simplify control flow and reduce nesting")

                if score.nesting_depth > 4:
                    issues.append(f"Deep nesting ({score.nesting_depth} levels)")
                    suggestions.append("Extract nested logic into helper functions")

                if score.lines_of_code > 50:
                    issues.append(f"Long function ({score.lines_of_code} lines)")
                    suggestions.append("Split into smaller, focused functions")

                if score.parameter_count > 5:
                    issues.append(f"Too many parameters ({score.parameter_count})")
                    suggestions.append("Consider using a parameter object")

                priority = {
                    ComplexityLevel.HIGH: 3,
                    ComplexityLevel.VERY_HIGH: 2,
                    ComplexityLevel.CRITICAL: 1,
                }.get(score.level, 4)

                hotspots.append(
                    ComplexityHotspot(
                        element_name=score.element_name,
                        element_type=score.element_type,
                        line_start=score.line_start,
                        line_end=score.line_end,
                        score=score.score,
                        level=score.level,
                        issues=issues,
                        suggestions=suggestions,
                        priority=priority,
                    )
                )

        # Sort by priority
        hotspots.sort(key=lambda h: h.priority)

        return hotspots

    def _generate_summary(
        self, scores: List[ComplexityScore], hotspots: List[ComplexityHotspot]
    ) -> Dict[str, Any]:
        """Generate analysis summary."""
        if not scores:
            return {"status": "no_elements"}

        level_counts = {}
        for score in scores:
            level_counts[score.level.value] = level_counts.get(score.level.value, 0) + 1

        return {
            "total_elements": len(scores),
            "total_hotspots": len(hotspots),
            "average_score": sum(s.score for s in scores) / len(scores),
            "max_score": max(s.score for s in scores),
            "min_score": min(s.score for s in scores),
            "level_distribution": level_counts,
            "health_status": self._get_health_status(scores, hotspots),
        }

    def _get_health_status(
        self, scores: List[ComplexityScore], hotspots: List[ComplexityHotspot]
    ) -> str:
        """Get overall health status."""
        if not scores:
            return "unknown"

        critical_count = sum(1 for s in scores if s.level == ComplexityLevel.CRITICAL)
        very_high_count = sum(1 for s in scores if s.level == ComplexityLevel.VERY_HIGH)

        if critical_count > 0:
            return "critical"
        elif very_high_count > 2:
            return "warning"
        elif len(hotspots) > len(scores) * 0.3:
            return "needs_attention"
        else:
            return "healthy"


# =============================================================================
# Global Instance Management
# =============================================================================


_global_heatmap: Optional[ComplexityHeatmap] = None


def get_complexity_heatmap() -> ComplexityHeatmap:
    """Get the global complexity heatmap instance."""
    global _global_heatmap
    if _global_heatmap is None:
        _global_heatmap = ComplexityHeatmap()
    return _global_heatmap


def reset_complexity_heatmap() -> None:
    """Reset the global complexity heatmap."""
    global _global_heatmap
    _global_heatmap = None


def create_heatmap(config: Optional[HeatmapConfig] = None) -> ComplexityHeatmap:
    """Create a new complexity heatmap with optional config."""
    return ComplexityHeatmap(config=config)


def analyze_complexity(code: str) -> HeatmapReport:
    """Convenience function to analyze code complexity."""
    return get_complexity_heatmap().analyze(code)


def find_hotspots(code: str) -> List[ComplexityHotspot]:
    """Convenience function to find complexity hotspots."""
    report = analyze_complexity(code)
    return report.hotspots


def get_complexity_trend() -> ComplexityTrend:
    """Convenience function to get complexity trend."""
    return get_complexity_heatmap().get_trend()
