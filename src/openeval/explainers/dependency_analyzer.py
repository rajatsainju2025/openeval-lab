"""Dependency analyzer for code dependency graph analysis.

This module provides tools for analyzing code dependencies, detecting
circular dependencies, and assessing impact of changes.
"""

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import CodeElementType


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class DependencyType(str, Enum):
    """Types of dependencies."""

    IMPORT = "import"  # Import statement
    FROM_IMPORT = "from_import"  # From import
    CALL = "call"  # Function/method call
    INHERITANCE = "inheritance"  # Class inheritance
    ATTRIBUTE = "attribute"  # Attribute access
    TYPE_HINT = "type_hint"  # Type annotation
    PARAMETER = "parameter"  # Function parameter


class DependencyDirection(str, Enum):
    """Direction of dependency."""

    INCOMING = "incoming"  # Other depends on this
    OUTGOING = "outgoing"  # This depends on other
    BIDIRECTIONAL = "bidirectional"  # Mutual dependency


class ImpactLevel(str, Enum):
    """Impact level of a change."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CycleType(str, Enum):
    """Types of dependency cycles."""

    DIRECT = "direct"  # A -> B -> A
    INDIRECT = "indirect"  # A -> B -> C -> A
    SELF = "self"  # A -> A


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Dependency:
    """Represents a dependency between code elements."""

    source: str
    target: str
    type: DependencyType
    line_number: Optional[int] = None
    column: Optional[int] = None
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str]:
        """Get unique key for this dependency."""
        return (self.source, self.target)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "line_number": self.line_number,
            "column": self.column,
            "context": self.context,
            "metadata": self.metadata,
        }


@dataclass
class DependencyNode:
    """A node in the dependency graph."""

    name: str
    type: CodeElementType = CodeElementType.MODULE
    incoming: Set[str] = field(default_factory=set)
    outgoing: Set[str] = field(default_factory=set)
    dependencies: List[Dependency] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def in_degree(self) -> int:
        """Number of incoming dependencies."""
        return len(self.incoming)

    @property
    def out_degree(self) -> int:
        """Number of outgoing dependencies."""
        return len(self.outgoing)

    @property
    def total_degree(self) -> int:
        """Total number of connections."""
        return self.in_degree + self.out_degree

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "incoming": list(self.incoming),
            "outgoing": list(self.outgoing),
        }


@dataclass
class DependencyCycle:
    """Represents a dependency cycle."""

    nodes: List[str]
    type: CycleType
    length: int = 0
    severity: ImpactLevel = ImpactLevel.MEDIUM
    suggestion: str = ""

    def __post_init__(self):
        self.length = len(self.nodes)
        if self.length == 1:
            self.type = CycleType.SELF
        elif self.length == 2:
            self.type = CycleType.DIRECT
        else:
            self.type = CycleType.INDIRECT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": self.nodes,
            "type": self.type.value,
            "length": self.length,
            "severity": self.severity.value,
            "suggestion": self.suggestion,
        }


@dataclass
class ImpactAnalysis:
    """Result of impact analysis."""

    changed_node: str
    directly_affected: Set[str] = field(default_factory=set)
    transitively_affected: Set[str] = field(default_factory=set)
    impact_level: ImpactLevel = ImpactLevel.LOW
    affected_paths: List[List[str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def total_affected(self) -> int:
        """Total number of affected nodes."""
        return len(self.directly_affected) + len(self.transitively_affected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "changed_node": self.changed_node,
            "directly_affected": list(self.directly_affected),
            "transitively_affected": list(self.transitively_affected),
            "total_affected": self.total_affected,
            "impact_level": self.impact_level.value,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


@dataclass
class DependencyGraph:
    """Complete dependency graph."""

    nodes: Dict[str, DependencyNode] = field(default_factory=dict)
    edges: List[Dependency] = field(default_factory=list)
    cycles: List[DependencyCycle] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges."""
        return len(self.edges)

    @property
    def has_cycles(self) -> bool:
        """Whether graph has cycles."""
        return len(self.cycles) > 0

    def add_node(
        self, name: str, node_type: CodeElementType = CodeElementType.MODULE
    ) -> DependencyNode:
        """Add a node to the graph."""
        if name not in self.nodes:
            self.nodes[name] = DependencyNode(name=name, type=node_type)
        return self.nodes[name]

    def add_edge(self, dependency: Dependency) -> None:
        """Add an edge to the graph."""
        source_node = self.add_node(dependency.source)
        target_node = self.add_node(dependency.target)

        source_node.outgoing.add(dependency.target)
        source_node.dependencies.append(dependency)
        target_node.incoming.add(dependency.source)

        self.edges.append(dependency)

    def get_dependents(self, name: str) -> Set[str]:
        """Get nodes that depend on the given node."""
        if name in self.nodes:
            return self.nodes[name].incoming.copy()
        return set()

    def get_dependencies(self, name: str) -> Set[str]:
        """Get nodes that the given node depends on."""
        if name in self.nodes:
            return self.nodes[name].outgoing.copy()
        return set()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "has_cycles": self.has_cycles,
            "cycle_count": len(self.cycles),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "cycles": [c.to_dict() for c in self.cycles],
        }


@dataclass
class AnalyzerConfig:
    """Configuration for dependency analyzer."""

    include_builtins: bool = False
    include_type_hints: bool = True
    include_calls: bool = True
    max_depth: int = 10
    ignore_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Dependency Extractors
# =============================================================================


class DependencyExtractor(ABC):
    """Abstract base class for dependency extractors."""

    @abstractmethod
    def extract(self, code: str, module_name: str = "module") -> List[Dependency]:
        """Extract dependencies from code.

        Args:
            code: Source code to analyze.
            module_name: Name of the module being analyzed.

        Returns:
            List of dependencies found.
        """
        pass


class PythonImportExtractor(DependencyExtractor):
    """Extracts import dependencies from Python code."""

    def extract(self, code: str, module_name: str = "module") -> List[Dependency]:
        """Extract import dependencies."""
        dependencies = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return dependencies

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(
                        Dependency(
                            source=module_name,
                            target=alias.name,
                            type=DependencyType.IMPORT,
                            line_number=node.lineno,
                            column=node.col_offset,
                            context=f"import {alias.name}",
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    target = f"{module}.{alias.name}" if module else alias.name
                    dependencies.append(
                        Dependency(
                            source=module_name,
                            target=target,
                            type=DependencyType.FROM_IMPORT,
                            line_number=node.lineno,
                            column=node.col_offset,
                            context=f"from {module} import {alias.name}",
                        )
                    )

        return dependencies


class PythonCallExtractor(DependencyExtractor):
    """Extracts function call dependencies from Python code."""

    def extract(self, code: str, module_name: str = "module") -> List[Dependency]:
        """Extract call dependencies."""
        dependencies = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return dependencies

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node.func)
                if func_name:
                    dependencies.append(
                        Dependency(
                            source=module_name,
                            target=func_name,
                            type=DependencyType.CALL,
                            line_number=node.lineno,
                            column=node.col_offset,
                            context=f"call to {func_name}",
                        )
                    )

        return dependencies

    def _get_call_name(self, node: ast.expr) -> Optional[str]:
        """Get the name of a function call."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_call_name(node.value)
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        return None


class PythonInheritanceExtractor(DependencyExtractor):
    """Extracts class inheritance dependencies from Python code."""

    def extract(self, code: str, module_name: str = "module") -> List[Dependency]:
        """Extract inheritance dependencies."""
        dependencies = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return dependencies

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = f"{module_name}.{node.name}"
                for base in node.bases:
                    base_name = self._get_name(base)
                    if base_name:
                        dependencies.append(
                            Dependency(
                                source=class_name,
                                target=base_name,
                                type=DependencyType.INHERITANCE,
                                line_number=node.lineno,
                                column=node.col_offset,
                                context=f"class {node.name} inherits from {base_name}",
                            )
                        )

        return dependencies

    def _get_name(self, node: ast.expr) -> Optional[str]:
        """Get the name from an expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        return None


class PythonTypeHintExtractor(DependencyExtractor):
    """Extracts type hint dependencies from Python code."""

    def extract(self, code: str, module_name: str = "module") -> List[Dependency]:
        """Extract type hint dependencies."""
        dependencies = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return dependencies

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Return annotation
                if node.returns:
                    types = self._extract_types(node.returns)
                    for t in types:
                        dependencies.append(
                            Dependency(
                                source=f"{module_name}.{node.name}",
                                target=t,
                                type=DependencyType.TYPE_HINT,
                                line_number=node.lineno,
                                context=f"return type hint: {t}",
                            )
                        )

                # Argument annotations
                for arg in node.args.args:
                    if arg.annotation:
                        types = self._extract_types(arg.annotation)
                        for t in types:
                            dependencies.append(
                                Dependency(
                                    source=f"{module_name}.{node.name}",
                                    target=t,
                                    type=DependencyType.TYPE_HINT,
                                    line_number=node.lineno,
                                    context=f"parameter type hint: {t}",
                                )
                            )

            elif isinstance(node, ast.AnnAssign):
                if node.annotation:
                    types = self._extract_types(node.annotation)
                    for t in types:
                        dependencies.append(
                            Dependency(
                                source=module_name,
                                target=t,
                                type=DependencyType.TYPE_HINT,
                                line_number=node.lineno,
                                context=f"variable type hint: {t}",
                            )
                        )

        return dependencies

    def _extract_types(self, node: ast.expr) -> List[str]:
        """Extract type names from annotation."""
        types = []

        if isinstance(node, ast.Name):
            types.append(node.id)
        elif isinstance(node, ast.Attribute):
            name = self._get_full_name(node)
            if name:
                types.append(name)
        elif isinstance(node, ast.Subscript):
            # Handle generics like List[int], Dict[str, int]
            types.extend(self._extract_types(node.value))
            if isinstance(node.slice, ast.Tuple):
                for elt in node.slice.elts:
                    types.extend(self._extract_types(elt))
            else:
                types.extend(self._extract_types(node.slice))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Handle Union types with | syntax
            types.extend(self._extract_types(node.left))
            types.extend(self._extract_types(node.right))

        return types

    def _get_full_name(self, node: ast.expr) -> Optional[str]:
        """Get fully qualified name from attribute access."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_full_name(node.value)
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        return None


# =============================================================================
# Cycle Detection
# =============================================================================


class CycleDetector:
    """Detects cycles in dependency graphs."""

    def detect_cycles(self, graph: DependencyGraph) -> List[DependencyCycle]:
        """Detect all cycles in the graph.

        Args:
            graph: Dependency graph to analyze.

        Returns:
            List of detected cycles.
        """
        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            if node in graph.nodes:
                for neighbor in graph.nodes[node].outgoing:
                    if neighbor not in visited:
                        dfs(neighbor)
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor)
                        cycle_nodes = path[cycle_start:] + [neighbor]
                        cycles.append(
                            DependencyCycle(
                                nodes=cycle_nodes,
                                type=CycleType.INDIRECT,
                                suggestion=self._get_suggestion(cycle_nodes),
                            )
                        )

            path.pop()
            rec_stack.remove(node)

        for node in graph.nodes:
            if node not in visited:
                dfs(node)

        return cycles

    def _get_suggestion(self, cycle_nodes: List[str]) -> str:
        """Generate suggestion for breaking cycle."""
        if len(cycle_nodes) == 2:
            return f"Consider extracting common functionality from {cycle_nodes[0]} and {cycle_nodes[1]} into a shared module."
        else:
            return f"Consider introducing an interface or dependency injection to break the cycle involving {len(cycle_nodes) - 1} modules."


# =============================================================================
# Impact Analyzer
# =============================================================================


class ImpactAnalyzer:
    """Analyzes the impact of changes on dependencies."""

    def __init__(self, graph: DependencyGraph):
        """Initialize impact analyzer.

        Args:
            graph: Dependency graph to analyze.
        """
        self.graph = graph

    def analyze_impact(self, changed_node: str, max_depth: int = 10) -> ImpactAnalysis:
        """Analyze the impact of changing a node.

        Args:
            changed_node: Node that will be changed.
            max_depth: Maximum depth for transitive analysis.

        Returns:
            Impact analysis result.
        """
        directly_affected = self.graph.get_dependents(changed_node)

        transitively_affected: Set[str] = set()
        affected_paths: List[List[str]] = []

        # BFS for transitive dependencies
        queue = [(dep, [changed_node, dep]) for dep in directly_affected]
        visited: Set[str] = directly_affected.copy()
        depth = 0

        while queue and depth < max_depth:
            next_queue: List[Tuple[str, List[str]]] = []
            for node, path in queue:
                dependents = self.graph.get_dependents(node)
                for dep in dependents:
                    if dep not in visited and dep != changed_node:
                        visited.add(dep)
                        transitively_affected.add(dep)
                        new_path = path + [dep]
                        affected_paths.append(new_path)
                        next_queue.append((dep, new_path))
            queue = next_queue
            depth += 1

        # Determine impact level
        total = len(directly_affected) + len(transitively_affected)
        if total == 0:
            impact_level = ImpactLevel.NONE
        elif total <= 2:
            impact_level = ImpactLevel.LOW
        elif total <= 5:
            impact_level = ImpactLevel.MEDIUM
        elif total <= 10:
            impact_level = ImpactLevel.HIGH
        else:
            impact_level = ImpactLevel.CRITICAL

        # Generate recommendations
        recommendations = self._generate_recommendations(
            changed_node, directly_affected, transitively_affected
        )

        return ImpactAnalysis(
            changed_node=changed_node,
            directly_affected=directly_affected,
            transitively_affected=transitively_affected,
            impact_level=impact_level,
            affected_paths=affected_paths[:10],  # Limit paths
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        changed_node: str,
        directly: Set[str],
        transitively: Set[str],
    ) -> List[str]:
        """Generate recommendations based on impact."""
        recommendations = []

        if len(directly) > 5:
            recommendations.append(
                f"High coupling detected: {changed_node} has {len(directly)} direct dependents. "
                "Consider introducing an abstraction layer."
            )

        if len(transitively) > len(directly) * 2:
            recommendations.append(
                "Large transitive impact detected. Consider adding integration tests for affected modules."
            )

        if self.graph.has_cycles:
            recommendations.append(
                "Circular dependencies exist in the graph. Breaking these may reduce cascading impacts."
            )

        if not recommendations:
            recommendations.append("Impact is manageable. Standard testing should suffice.")

        return recommendations


# =============================================================================
# Main Dependency Analyzer
# =============================================================================


class DependencyAnalyzer:
    """Main dependency analyzer combining all extractors."""

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """Initialize dependency analyzer.

        Args:
            config: Optional configuration.
        """
        self.config = config or AnalyzerConfig()
        self._extractors: List[DependencyExtractor] = [
            PythonImportExtractor(),
            PythonInheritanceExtractor(),
        ]

        if self.config.include_calls:
            self._extractors.append(PythonCallExtractor())

        if self.config.include_type_hints:
            self._extractors.append(PythonTypeHintExtractor())

        self._cycle_detector = CycleDetector()

    def analyze(self, code: str, module_name: str = "module") -> DependencyGraph:
        """Analyze dependencies in code.

        Args:
            code: Source code to analyze.
            module_name: Name of the module.

        Returns:
            Dependency graph.
        """
        graph = DependencyGraph()
        graph.add_node(module_name)

        # Extract dependencies using all extractors
        for extractor in self._extractors:
            dependencies = extractor.extract(code, module_name)
            for dep in dependencies:
                if self._should_include(dep.target):
                    graph.add_edge(dep)

        # Detect cycles
        graph.cycles = self._cycle_detector.detect_cycles(graph)

        return graph

    def analyze_multiple(self, modules: Dict[str, str]) -> DependencyGraph:
        """Analyze dependencies across multiple modules.

        Args:
            modules: Dict mapping module names to source code.

        Returns:
            Combined dependency graph.
        """
        graph = DependencyGraph()

        for module_name, code in modules.items():
            module_graph = self.analyze(code, module_name)

            # Merge nodes and edges
            for name, node in module_graph.nodes.items():
                if name not in graph.nodes:
                    graph.nodes[name] = node
                else:
                    graph.nodes[name].incoming.update(node.incoming)
                    graph.nodes[name].outgoing.update(node.outgoing)
                    graph.nodes[name].dependencies.extend(node.dependencies)

            graph.edges.extend(module_graph.edges)

        # Re-detect cycles after merging
        graph.cycles = self._cycle_detector.detect_cycles(graph)

        return graph

    def get_impact_analysis(self, graph: DependencyGraph, changed_node: str) -> ImpactAnalysis:
        """Get impact analysis for a change.

        Args:
            graph: Dependency graph.
            changed_node: Node being changed.

        Returns:
            Impact analysis result.
        """
        analyzer = ImpactAnalyzer(graph)
        return analyzer.analyze_impact(changed_node, max_depth=self.config.max_depth)

    def find_unused_modules(self, graph: DependencyGraph) -> List[str]:
        """Find modules with no dependents.

        Args:
            graph: Dependency graph.

        Returns:
            List of unused module names.
        """
        unused = []
        for name, node in graph.nodes.items():
            if node.in_degree == 0 and node.out_degree > 0:
                unused.append(name)
        return unused

    def get_hub_modules(self, graph: DependencyGraph, threshold: int = 5) -> List[str]:
        """Find highly connected modules.

        Args:
            graph: Dependency graph.
            threshold: Minimum connection count.

        Returns:
            List of hub module names.
        """
        hubs = []
        for name, node in graph.nodes.items():
            if node.total_degree >= threshold:
                hubs.append(name)
        return sorted(hubs, key=lambda x: graph.nodes[x].total_degree, reverse=True)

    def _should_include(self, target: str) -> bool:
        """Check if a dependency target should be included."""
        # Filter builtins
        if not self.config.include_builtins:
            builtins = {
                "int",
                "str",
                "float",
                "bool",
                "list",
                "dict",
                "set",
                "tuple",
                "None",
                "True",
                "False",
                "print",
                "len",
                "range",
                "type",
                "object",
                "Exception",
                "BaseException",
            }
            if target in builtins:
                return False

        # Filter by patterns
        for pattern in self.config.ignore_patterns:
            if re.match(pattern, target):
                return False

        return True


# =============================================================================
# Global Instance Management
# =============================================================================


_global_analyzer: Optional[DependencyAnalyzer] = None


def get_dependency_analyzer() -> DependencyAnalyzer:
    """Get the global dependency analyzer instance."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = DependencyAnalyzer()
    return _global_analyzer


def reset_dependency_analyzer() -> None:
    """Reset the global dependency analyzer."""
    global _global_analyzer
    _global_analyzer = None


def create_dependency_analyzer(
    config: Optional[AnalyzerConfig] = None,
) -> DependencyAnalyzer:
    """Create a new dependency analyzer with optional config."""
    return DependencyAnalyzer(config=config)


def analyze_dependencies(code: str, module_name: str = "module") -> DependencyGraph:
    """Convenience function to analyze dependencies."""
    return get_dependency_analyzer().analyze(code, module_name)


def detect_cycles(code: str, module_name: str = "module") -> List[DependencyCycle]:
    """Convenience function to detect dependency cycles."""
    graph = analyze_dependencies(code, module_name)
    return graph.cycles


def analyze_impact(code: str, changed_element: str, module_name: str = "module") -> ImpactAnalysis:
    """Convenience function to analyze change impact."""
    analyzer = get_dependency_analyzer()
    graph = analyzer.analyze(code, module_name)
    return analyzer.get_impact_analysis(graph, changed_element)
