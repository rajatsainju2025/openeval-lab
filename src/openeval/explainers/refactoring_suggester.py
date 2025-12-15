"""
Refactoring Suggester for code explanations.

This module provides tools for detecting refactoring opportunities,
identifying code smells, and suggesting improvements with impact assessment.
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4


class RefactoringType(Enum):
    """Types of refactoring suggestions."""

    # Method-level refactorings
    EXTRACT_METHOD = auto()
    INLINE_METHOD = auto()
    RENAME_METHOD = auto()
    MOVE_METHOD = auto()
    PULL_UP_METHOD = auto()
    PUSH_DOWN_METHOD = auto()

    # Class-level refactorings
    EXTRACT_CLASS = auto()
    INLINE_CLASS = auto()
    EXTRACT_SUPERCLASS = auto()
    EXTRACT_INTERFACE = auto()
    COLLAPSE_HIERARCHY = auto()

    # Variable-level refactorings
    RENAME_VARIABLE = auto()
    EXTRACT_VARIABLE = auto()
    INLINE_VARIABLE = auto()
    SPLIT_VARIABLE = auto()

    # Simplification refactorings
    SIMPLIFY_CONDITIONAL = auto()
    DECOMPOSE_CONDITIONAL = auto()
    REPLACE_NESTED_CONDITIONAL = auto()
    CONSOLIDATE_CONDITIONAL = auto()

    # Code organization
    REMOVE_DEAD_CODE = auto()
    REMOVE_DUPLICATION = auto()
    INTRODUCE_PARAMETER_OBJECT = auto()
    REPLACE_MAGIC_NUMBER = auto()

    # Design improvement
    REPLACE_INHERITANCE_WITH_DELEGATION = auto()
    REPLACE_DELEGATION_WITH_INHERITANCE = auto()
    ENCAPSULATE_FIELD = auto()
    ENCAPSULATE_COLLECTION = auto()


class CodeSmellType(Enum):
    """Types of code smells."""

    # Bloaters
    LONG_METHOD = auto()
    LARGE_CLASS = auto()
    LONG_PARAMETER_LIST = auto()
    DATA_CLUMPS = auto()
    PRIMITIVE_OBSESSION = auto()

    # Object-Orientation Abusers
    SWITCH_STATEMENTS = auto()
    PARALLEL_INHERITANCE = auto()
    REFUSED_BEQUEST = auto()
    ALTERNATIVE_CLASSES = auto()

    # Change Preventers
    DIVERGENT_CHANGE = auto()
    SHOTGUN_SURGERY = auto()
    PARALLEL_MODIFICATION = auto()

    # Dispensables
    DEAD_CODE = auto()
    SPECULATIVE_GENERALITY = auto()
    LAZY_CLASS = auto()
    DATA_CLASS = auto()
    DUPLICATE_CODE = auto()

    # Couplers
    FEATURE_ENVY = auto()
    INAPPROPRIATE_INTIMACY = auto()
    MESSAGE_CHAINS = auto()
    MIDDLE_MAN = auto()


class Severity(Enum):
    """Severity levels for code smells and suggestions."""

    INFO = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class Impact(Enum):
    """Impact levels for refactoring suggestions."""

    MINIMAL = auto()  # Cosmetic changes
    LOW = auto()  # Small scope, low risk
    MEDIUM = auto()  # Moderate scope and risk
    HIGH = auto()  # Large scope, significant changes
    BREAKING = auto()  # API changes, requires updates


@dataclass
class Location:
    """Location in source code."""

    file: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line_start}-{self.line_end}"


@dataclass
class CodeSmell:
    """Represents a detected code smell."""

    id: str
    smell_type: CodeSmellType
    severity: Severity
    location: Location
    description: str
    code_snippet: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    suggested_refactorings: List[RefactoringType] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "smell_type": self.smell_type.name,
            "severity": self.severity.name,
            "location": str(self.location),
            "description": self.description,
            "code_snippet": self.code_snippet,
            "metrics": self.metrics,
            "suggested_refactorings": [r.name for r in self.suggested_refactorings],
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class RefactoringSuggestion:
    """Represents a refactoring suggestion."""

    id: str
    refactoring_type: RefactoringType
    severity: Severity
    impact: Impact
    location: Location
    title: str
    description: str
    rationale: str
    before_code: str
    after_code: str
    affected_files: List[str] = field(default_factory=list)
    related_smells: List[str] = field(default_factory=list)
    estimated_effort: str = "unknown"
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "refactoring_type": self.refactoring_type.name,
            "severity": self.severity.name,
            "impact": self.impact.name,
            "location": str(self.location),
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "before_code": self.before_code,
            "after_code": self.after_code,
            "affected_files": self.affected_files,
            "related_smells": self.related_smells,
            "estimated_effort": self.estimated_effort,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ImpactAssessment:
    """Assessment of refactoring impact."""

    suggestion_id: str
    impact_level: Impact
    affected_files: List[str]
    affected_tests: List[str]
    breaking_changes: List[str]
    dependencies_affected: List[str]
    estimated_time: str
    risk_factors: List[str]
    mitigation_strategies: List[str]
    rollback_plan: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggestion_id": self.suggestion_id,
            "impact_level": self.impact_level.name,
            "affected_files": self.affected_files,
            "affected_tests": self.affected_tests,
            "breaking_changes": self.breaking_changes,
            "dependencies_affected": self.dependencies_affected,
            "estimated_time": self.estimated_time,
            "risk_factors": self.risk_factors,
            "mitigation_strategies": self.mitigation_strategies,
            "rollback_plan": self.rollback_plan,
        }


@dataclass
class RefactoringReport:
    """Report containing all smells and suggestions."""

    file: str
    smells: List[CodeSmell]
    suggestions: List[RefactoringSuggestion]
    impact_assessments: Dict[str, ImpactAssessment]
    summary: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file": self.file,
            "smells": [s.to_dict() for s in self.smells],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "impact_assessments": {k: v.to_dict() for k, v in self.impact_assessments.items()},
            "summary": self.summary,
            "generated_at": self.generated_at.isoformat(),
        }


class SmellDetector(ABC):
    """Base class for code smell detectors."""

    @property
    @abstractmethod
    def smell_type(self) -> CodeSmellType:
        """Return the type of smell this detector finds."""
        pass

    @abstractmethod
    def detect(self, code: str, file: str = "<unknown>") -> List[CodeSmell]:
        """Detect code smells in the given code."""
        pass


class LongMethodDetector(SmellDetector):
    """Detects methods that are too long."""

    def __init__(
        self,
        max_lines: int = 30,
        max_statements: int = 20,
    ):
        self.max_lines = max_lines
        self.max_statements = max_statements

    @property
    def smell_type(self) -> CodeSmellType:
        return CodeSmellType.LONG_METHOD

    def detect(self, code: str, file: str = "<unknown>") -> List[CodeSmell]:
        """Detect long methods."""
        smells = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return smells

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                line_count = node.end_lineno - node.lineno + 1
                statement_count = sum(1 for child in ast.walk(node) if isinstance(child, ast.stmt))

                if line_count > self.max_lines or statement_count > self.max_statements:
                    severity = self._calculate_severity(line_count, statement_count)
                    code_snippet = "\n".join(
                        lines[node.lineno - 1 : min(node.end_lineno, node.lineno + 5)]
                    )

                    smells.append(
                        CodeSmell(
                            id=str(uuid4()),
                            smell_type=self.smell_type,
                            severity=severity,
                            location=Location(
                                file=file,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                                column_start=node.col_offset,
                            ),
                            description=(
                                f"Method '{node.name}' is too long "
                                f"({line_count} lines, {statement_count} statements)"
                            ),
                            code_snippet=code_snippet,
                            metrics={
                                "line_count": line_count,
                                "statement_count": statement_count,
                                "max_lines": self.max_lines,
                                "max_statements": self.max_statements,
                            },
                            suggested_refactorings=[
                                RefactoringType.EXTRACT_METHOD,
                                RefactoringType.DECOMPOSE_CONDITIONAL,
                            ],
                        )
                    )

        return smells

    def _calculate_severity(self, lines: int, statements: int) -> Severity:
        """Calculate severity based on method size."""
        if lines > self.max_lines * 3 or statements > self.max_statements * 3:
            return Severity.CRITICAL
        elif lines > self.max_lines * 2 or statements > self.max_statements * 2:
            return Severity.HIGH
        elif lines > self.max_lines * 1.5 or statements > self.max_statements * 1.5:
            return Severity.MEDIUM
        return Severity.LOW


class LongParameterListDetector(SmellDetector):
    """Detects functions with too many parameters."""

    def __init__(self, max_params: int = 5):
        self.max_params = max_params

    @property
    def smell_type(self) -> CodeSmellType:
        return CodeSmellType.LONG_PARAMETER_LIST

    def detect(self, code: str, file: str = "<unknown>") -> List[CodeSmell]:
        """Detect long parameter lists."""
        smells = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return smells

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = node.args
                total_params = (
                    len(params.args)
                    + len(params.posonlyargs)
                    + len(params.kwonlyargs)
                    + (1 if params.vararg else 0)
                    + (1 if params.kwarg else 0)
                )

                # Exclude 'self' and 'cls' from count
                if params.args and params.args[0].arg in ("self", "cls"):
                    total_params -= 1

                if total_params > self.max_params:
                    code_snippet = "\n".join(lines[node.lineno - 1 : node.lineno + 2])

                    smells.append(
                        CodeSmell(
                            id=str(uuid4()),
                            smell_type=self.smell_type,
                            severity=self._calculate_severity(total_params),
                            location=Location(
                                file=file,
                                line_start=node.lineno,
                                line_end=node.lineno,
                                column_start=node.col_offset,
                            ),
                            description=(
                                f"Function '{node.name}' has too many parameters "
                                f"({total_params} > {self.max_params})"
                            ),
                            code_snippet=code_snippet,
                            metrics={
                                "param_count": total_params,
                                "max_params": self.max_params,
                            },
                            suggested_refactorings=[
                                RefactoringType.INTRODUCE_PARAMETER_OBJECT,
                            ],
                        )
                    )

        return smells

    def _calculate_severity(self, params: int) -> Severity:
        """Calculate severity based on parameter count."""
        excess = params - self.max_params
        if excess >= 5:
            return Severity.HIGH
        elif excess >= 3:
            return Severity.MEDIUM
        return Severity.LOW


class DuplicateCodeDetector(SmellDetector):
    """Detects duplicate code patterns."""

    def __init__(self, min_lines: int = 5, similarity_threshold: float = 0.8):
        self.min_lines = min_lines
        self.similarity_threshold = similarity_threshold

    @property
    def smell_type(self) -> CodeSmellType:
        return CodeSmellType.DUPLICATE_CODE

    def detect(self, code: str, file: str = "<unknown>") -> List[CodeSmell]:
        """Detect duplicate code."""
        smells = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return smells

        lines = code.split("\n")
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_code = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                functions.append((node.name, node.lineno, node.end_lineno, func_code))

        # Compare functions for similarity
        for i, (name1, start1, end1, code1) in enumerate(functions):
            for name2, start2, end2, code2 in functions[i + 1 :]:
                similarity = self._calculate_similarity(code1, code2)

                if similarity >= self.similarity_threshold:
                    smells.append(
                        CodeSmell(
                            id=str(uuid4()),
                            smell_type=self.smell_type,
                            severity=Severity.MEDIUM,
                            location=Location(
                                file=file,
                                line_start=start1,
                                line_end=end1,
                            ),
                            description=(
                                f"Functions '{name1}' and '{name2}' have similar code "
                                f"({similarity:.0%} similarity)"
                            ),
                            code_snippet=code1[:200] + "...",
                            metrics={
                                "similarity": similarity,
                                "other_function": name2,
                                "other_location": f"{start2}-{end2}",
                            },
                            suggested_refactorings=[
                                RefactoringType.REMOVE_DUPLICATION,
                                RefactoringType.EXTRACT_METHOD,
                            ],
                        )
                    )

        return smells

    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets."""
        # Normalize code
        norm1 = self._normalize_code(code1)
        norm2 = self._normalize_code(code2)

        # Simple token-based similarity
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        # Remove comments
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        # Remove string literals
        code = re.sub(r"['\"].*?['\"]", "STR", code)
        # Normalize whitespace
        code = " ".join(code.split())
        return code


class DeadCodeDetector(SmellDetector):
    """Detects potentially dead code."""

    @property
    def smell_type(self) -> CodeSmellType:
        return CodeSmellType.DEAD_CODE

    def detect(self, code: str, file: str = "<unknown>") -> List[CodeSmell]:
        """Detect dead code."""
        smells = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return smells

        lines = code.split("\n")

        # Find unreachable code after return/raise
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._check_unreachable_code(node, lines, file, smells)

        # Find unused variables
        self._check_unused_variables(tree, lines, file, smells)

        return smells

    def _check_unreachable_code(
        self,
        func_node: ast.FunctionDef,
        lines: List[str],
        file: str,
        smells: List[CodeSmell],
    ) -> None:
        """Check for unreachable code after return/raise."""
        body = func_node.body

        for i, stmt in enumerate(body[:-1]):
            if isinstance(stmt, (ast.Return, ast.Raise)):
                # Code after return/raise is unreachable
                next_stmt = body[i + 1]
                code_snippet = "\n".join(lines[next_stmt.lineno - 1 : next_stmt.end_lineno])

                smells.append(
                    CodeSmell(
                        id=str(uuid4()),
                        smell_type=self.smell_type,
                        severity=Severity.MEDIUM,
                        location=Location(
                            file=file,
                            line_start=next_stmt.lineno,
                            line_end=next_stmt.end_lineno,
                        ),
                        description=(
                            f"Unreachable code after "
                            f"{'return' if isinstance(stmt, ast.Return) else 'raise'}"
                        ),
                        code_snippet=code_snippet,
                        suggested_refactorings=[RefactoringType.REMOVE_DEAD_CODE],
                    )
                )
                break

    def _check_unused_variables(
        self,
        tree: ast.AST,
        lines: List[str],
        file: str,
        smells: List[CodeSmell],
    ) -> None:
        """Check for unused variables."""
        # Simple heuristic: find assignments and check if name is used later
        assignments: Dict[str, ast.Assign] = {}
        used_names: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)

        # Find unused assignments (excluding _ and double underscore)
        for name, assign_node in assignments.items():
            if name not in used_names and not name.startswith("_"):
                code_snippet = lines[assign_node.lineno - 1]

                smells.append(
                    CodeSmell(
                        id=str(uuid4()),
                        smell_type=self.smell_type,
                        severity=Severity.LOW,
                        location=Location(
                            file=file,
                            line_start=assign_node.lineno,
                            line_end=assign_node.lineno,
                        ),
                        description=f"Variable '{name}' is assigned but never used",
                        code_snippet=code_snippet,
                        suggested_refactorings=[RefactoringType.REMOVE_DEAD_CODE],
                    )
                )


class NestedConditionalDetector(SmellDetector):
    """Detects deeply nested conditionals."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    @property
    def smell_type(self) -> CodeSmellType:
        return CodeSmellType.SWITCH_STATEMENTS  # Repurposed for deep nesting

    def detect(self, code: str, file: str = "<unknown>") -> List[CodeSmell]:
        """Detect deeply nested conditionals."""
        smells = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return smells

        lines = code.split("\n")

        def check_nesting(node: ast.AST, depth: int = 0, parent_if: ast.If = None):
            if isinstance(node, ast.If):
                depth += 1

                if depth > self.max_depth:
                    code_snippet = "\n".join(
                        lines[node.lineno - 1 : min(node.end_lineno, node.lineno + 5)]
                    )

                    smells.append(
                        CodeSmell(
                            id=str(uuid4()),
                            smell_type=self.smell_type,
                            severity=self._calculate_severity(depth),
                            location=Location(
                                file=file,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                            ),
                            description=(
                                f"Deeply nested conditional (depth {depth} > {self.max_depth})"
                            ),
                            code_snippet=code_snippet,
                            metrics={"depth": depth, "max_depth": self.max_depth},
                            suggested_refactorings=[
                                RefactoringType.SIMPLIFY_CONDITIONAL,
                                RefactoringType.REPLACE_NESTED_CONDITIONAL,
                                RefactoringType.EXTRACT_METHOD,
                            ],
                        )
                    )

                parent_if = node

            for child in ast.iter_child_nodes(node):
                if isinstance(node, ast.If):
                    check_nesting(child, depth, node)
                else:
                    check_nesting(child, depth, parent_if)

        check_nesting(tree)
        return smells

    def _calculate_severity(self, depth: int) -> Severity:
        """Calculate severity based on nesting depth."""
        if depth > self.max_depth + 3:
            return Severity.CRITICAL
        elif depth > self.max_depth + 2:
            return Severity.HIGH
        elif depth > self.max_depth + 1:
            return Severity.MEDIUM
        return Severity.LOW


class LargeClassDetector(SmellDetector):
    """Detects classes that are too large."""

    def __init__(self, max_methods: int = 20, max_attributes: int = 15):
        self.max_methods = max_methods
        self.max_attributes = max_attributes

    @property
    def smell_type(self) -> CodeSmellType:
        return CodeSmellType.LARGE_CLASS

    def detect(self, code: str, file: str = "<unknown>") -> List[CodeSmell]:
        """Detect large classes."""
        smells = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return smells

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                attributes = self._count_attributes(node)

                if len(methods) > self.max_methods or attributes > self.max_attributes:
                    code_snippet = "\n".join(
                        lines[node.lineno - 1 : min(node.end_lineno, node.lineno + 10)]
                    )

                    smells.append(
                        CodeSmell(
                            id=str(uuid4()),
                            smell_type=self.smell_type,
                            severity=self._calculate_severity(len(methods), attributes),
                            location=Location(
                                file=file,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                            ),
                            description=(
                                f"Class '{node.name}' is too large "
                                f"({len(methods)} methods, {attributes} attributes)"
                            ),
                            code_snippet=code_snippet,
                            metrics={
                                "method_count": len(methods),
                                "attribute_count": attributes,
                                "max_methods": self.max_methods,
                                "max_attributes": self.max_attributes,
                            },
                            suggested_refactorings=[
                                RefactoringType.EXTRACT_CLASS,
                                RefactoringType.EXTRACT_SUPERCLASS,
                            ],
                        )
                    )

        return smells

    def _count_attributes(self, class_node: ast.ClassDef) -> int:
        """Count class attributes."""
        attributes = set()

        for node in ast.walk(class_node):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    attributes.add(node.attr)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        attributes.add(target.id)

        return len(attributes)

    def _calculate_severity(self, methods: int, attributes: int) -> Severity:
        """Calculate severity based on class size."""
        method_excess = methods - self.max_methods
        attr_excess = attributes - self.max_attributes

        if method_excess > 20 or attr_excess > 10:
            return Severity.CRITICAL
        elif method_excess > 10 or attr_excess > 5:
            return Severity.HIGH
        elif method_excess > 5 or attr_excess > 3:
            return Severity.MEDIUM
        return Severity.LOW


class RefactoringSuggester:
    """Suggests refactorings based on detected code smells."""

    def __init__(self, detectors: Optional[List[SmellDetector]] = None):
        """Initialize the suggester with detectors."""
        self.detectors = detectors or self._get_default_detectors()
        self.suggestion_generators: Dict[
            CodeSmellType, Callable[[CodeSmell, str], RefactoringSuggestion]
        ] = {
            CodeSmellType.LONG_METHOD: self._suggest_for_long_method,
            CodeSmellType.LONG_PARAMETER_LIST: self._suggest_for_long_params,
            CodeSmellType.DUPLICATE_CODE: self._suggest_for_duplication,
            CodeSmellType.DEAD_CODE: self._suggest_for_dead_code,
            CodeSmellType.SWITCH_STATEMENTS: self._suggest_for_nested_conditional,
            CodeSmellType.LARGE_CLASS: self._suggest_for_large_class,
        }

    def _get_default_detectors(self) -> List[SmellDetector]:
        """Get default set of detectors."""
        return [
            LongMethodDetector(),
            LongParameterListDetector(),
            DuplicateCodeDetector(),
            DeadCodeDetector(),
            NestedConditionalDetector(),
            LargeClassDetector(),
        ]

    def analyze(self, code: str, file: str = "<unknown>") -> RefactoringReport:
        """Analyze code and generate refactoring report."""
        # Detect all smells
        all_smells = []
        for detector in self.detectors:
            smells = detector.detect(code, file)
            all_smells.extend(smells)

        # Generate suggestions for each smell
        suggestions = []
        for smell in all_smells:
            if smell.smell_type in self.suggestion_generators:
                suggestion = self.suggestion_generators[smell.smell_type](smell, code)
                if suggestion:
                    suggestions.append(suggestion)

        # Generate impact assessments
        impact_assessments = {}
        for suggestion in suggestions:
            assessment = self._assess_impact(suggestion, code, file)
            impact_assessments[suggestion.id] = assessment

        # Generate summary
        summary = self._generate_summary(all_smells, suggestions)

        return RefactoringReport(
            file=file,
            smells=all_smells,
            suggestions=suggestions,
            impact_assessments=impact_assessments,
            summary=summary,
        )

    def _suggest_for_long_method(
        self, smell: CodeSmell, code: str
    ) -> Optional[RefactoringSuggestion]:
        """Generate suggestion for long method."""
        return RefactoringSuggestion(
            id=str(uuid4()),
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            severity=smell.severity,
            impact=Impact.MEDIUM,
            location=smell.location,
            title="Extract method to reduce complexity",
            description=(
                f"The method is too long with {smell.metrics.get('line_count', 'N/A')} lines. "
                "Consider extracting logical blocks into separate helper methods."
            ),
            rationale=(
                "Long methods are hard to understand, test, and maintain. "
                "Breaking them into smaller, focused methods improves readability "
                "and enables better code reuse."
            ),
            before_code=smell.code_snippet,
            after_code=(
                "# Suggested structure:\n"
                "def original_method(self, ...):\n"
                "    result = self._step_one(...)\n"
                "    result = self._step_two(result, ...)\n"
                "    return self._finalize(result)\n\n"
                "def _step_one(self, ...) -> ...:\n"
                "    # First logical block\n"
                "    pass"
            ),
            related_smells=[smell.id],
            estimated_effort="30 minutes - 2 hours",
            confidence=0.8,
        )

    def _suggest_for_long_params(
        self, smell: CodeSmell, code: str
    ) -> Optional[RefactoringSuggestion]:
        """Generate suggestion for long parameter list."""
        return RefactoringSuggestion(
            id=str(uuid4()),
            refactoring_type=RefactoringType.INTRODUCE_PARAMETER_OBJECT,
            severity=smell.severity,
            impact=Impact.MEDIUM,
            location=smell.location,
            title="Introduce parameter object",
            description=(
                f"The function has {smell.metrics.get('param_count', 'N/A')} parameters. "
                "Consider grouping related parameters into a data class or named tuple."
            ),
            rationale=(
                "Long parameter lists make functions hard to call and understand. "
                "Grouping related parameters into objects improves clarity and allows "
                "for easier extension without changing the function signature."
            ),
            before_code=smell.code_snippet,
            after_code=(
                "@dataclass\n"
                "class FunctionParams:\n"
                "    param1: type1\n"
                "    param2: type2\n"
                "    # ... other parameters\n\n"
                "def original_function(self, params: FunctionParams):\n"
                "    # Use params.param1, params.param2, etc."
            ),
            related_smells=[smell.id],
            estimated_effort="1-2 hours",
            confidence=0.75,
        )

    def _suggest_for_duplication(
        self, smell: CodeSmell, code: str
    ) -> Optional[RefactoringSuggestion]:
        """Generate suggestion for duplicate code."""
        return RefactoringSuggestion(
            id=str(uuid4()),
            refactoring_type=RefactoringType.REMOVE_DUPLICATION,
            severity=smell.severity,
            impact=Impact.MEDIUM,
            location=smell.location,
            title="Remove code duplication",
            description=(
                f"Similar code detected ({smell.metrics.get('similarity', 0):.0%} similarity). "
                "Extract the common logic into a shared method."
            ),
            rationale=(
                "Duplicate code increases maintenance burden and risk of inconsistent changes. "
                "Extracting common logic ensures changes only need to be made in one place."
            ),
            before_code=smell.code_snippet,
            after_code=(
                "# Extract common logic:\n"
                "def _common_logic(self, ...):\n"
                "    # Shared implementation\n"
                "    pass\n\n"
                "def method_a(self, ...):\n"
                "    # Specific setup\n"
                "    return self._common_logic(...)\n\n"
                "def method_b(self, ...):\n"
                "    # Specific setup\n"
                "    return self._common_logic(...)"
            ),
            related_smells=[smell.id],
            estimated_effort="1-3 hours",
            confidence=0.7,
        )

    def _suggest_for_dead_code(
        self, smell: CodeSmell, code: str
    ) -> Optional[RefactoringSuggestion]:
        """Generate suggestion for dead code."""
        return RefactoringSuggestion(
            id=str(uuid4()),
            refactoring_type=RefactoringType.REMOVE_DEAD_CODE,
            severity=smell.severity,
            impact=Impact.LOW,
            location=smell.location,
            title="Remove dead code",
            description=smell.description,
            rationale=(
                "Dead code adds noise and confusion. "
                "It can mislead developers and may mask actual bugs. "
                "Removing it improves code clarity."
            ),
            before_code=smell.code_snippet,
            after_code="# Simply remove the unreachable/unused code",
            related_smells=[smell.id],
            estimated_effort="5-15 minutes",
            confidence=0.9,
        )

    def _suggest_for_nested_conditional(
        self, smell: CodeSmell, code: str
    ) -> Optional[RefactoringSuggestion]:
        """Generate suggestion for nested conditionals."""
        return RefactoringSuggestion(
            id=str(uuid4()),
            refactoring_type=RefactoringType.SIMPLIFY_CONDITIONAL,
            severity=smell.severity,
            impact=Impact.MEDIUM,
            location=smell.location,
            title="Simplify nested conditionals",
            description=(
                f"Deeply nested conditional (depth {smell.metrics.get('depth', 'N/A')}). "
                "Consider using guard clauses or extracting conditions."
            ),
            rationale=(
                "Deeply nested conditionals are hard to follow and test. "
                "Guard clauses (early returns) and extracted methods improve readability."
            ),
            before_code=smell.code_snippet,
            after_code=(
                "# Using guard clauses:\n"
                "def method(self, ...):\n"
                "    if not condition1:\n"
                "        return early_result1\n"
                "    if not condition2:\n"
                "        return early_result2\n"
                "    # Main logic here\n"
                "    return main_result"
            ),
            related_smells=[smell.id],
            estimated_effort="30 minutes - 1 hour",
            confidence=0.75,
        )

    def _suggest_for_large_class(
        self, smell: CodeSmell, code: str
    ) -> Optional[RefactoringSuggestion]:
        """Generate suggestion for large class."""
        return RefactoringSuggestion(
            id=str(uuid4()),
            refactoring_type=RefactoringType.EXTRACT_CLASS,
            severity=smell.severity,
            impact=Impact.HIGH,
            location=smell.location,
            title="Extract class to reduce responsibility",
            description=(
                f"Class has {smell.metrics.get('method_count', 'N/A')} methods and "
                f"{smell.metrics.get('attribute_count', 'N/A')} attributes. "
                "Consider splitting into smaller, focused classes."
            ),
            rationale=(
                "Large classes often violate the Single Responsibility Principle. "
                "Extracting cohesive groups of methods and attributes into separate "
                "classes improves maintainability and testability."
            ),
            before_code=smell.code_snippet,
            after_code=(
                "# Split by responsibility:\n"
                "class MainClass:\n"
                "    def __init__(self):\n"
                "        self.helper_a = HelperA()\n"
                "        self.helper_b = HelperB()\n\n"
                "class HelperA:\n"
                "    # Methods for responsibility A\n"
                "    pass\n\n"
                "class HelperB:\n"
                "    # Methods for responsibility B\n"
                "    pass"
            ),
            affected_files=["Current file", "New files for extracted classes"],
            related_smells=[smell.id],
            estimated_effort="2-4 hours",
            confidence=0.7,
        )

    def _assess_impact(
        self,
        suggestion: RefactoringSuggestion,
        code: str,
        file: str,
    ) -> ImpactAssessment:
        """Assess the impact of a refactoring suggestion."""
        return ImpactAssessment(
            suggestion_id=suggestion.id,
            impact_level=suggestion.impact,
            affected_files=[file] + suggestion.affected_files,
            affected_tests=[f"test_{file}"],
            breaking_changes=self._identify_breaking_changes(suggestion),
            dependencies_affected=[],
            estimated_time=suggestion.estimated_effort,
            risk_factors=self._identify_risks(suggestion),
            mitigation_strategies=self._get_mitigation_strategies(suggestion),
            rollback_plan="Revert the commit if tests fail or issues are detected",
        )

    def _identify_breaking_changes(self, suggestion: RefactoringSuggestion) -> List[str]:
        """Identify potential breaking changes."""
        breaking = []

        if suggestion.refactoring_type in (
            RefactoringType.RENAME_METHOD,
            RefactoringType.MOVE_METHOD,
            RefactoringType.EXTRACT_CLASS,
        ):
            breaking.append("API signature changes may affect callers")

        if suggestion.refactoring_type == RefactoringType.INTRODUCE_PARAMETER_OBJECT:
            breaking.append("Function signature change requires updating all callers")

        return breaking

    def _identify_risks(self, suggestion: RefactoringSuggestion) -> List[str]:
        """Identify risk factors."""
        risks = []

        if suggestion.impact in (Impact.HIGH, Impact.BREAKING):
            risks.append("Large scope increases risk of introducing bugs")

        if suggestion.confidence < 0.7:
            risks.append("Lower confidence in automated suggestion")

        return risks or ["Low risk refactoring"]

    def _get_mitigation_strategies(self, suggestion: RefactoringSuggestion) -> List[str]:
        """Get mitigation strategies for the refactoring."""
        strategies = [
            "Ensure comprehensive test coverage before refactoring",
            "Perform refactoring in small, incremental steps",
            "Use IDE refactoring tools when available",
            "Code review the changes",
        ]

        if suggestion.impact in (Impact.HIGH, Impact.BREAKING):
            strategies.append("Consider feature flag for gradual rollout")

        return strategies

    def _generate_summary(
        self,
        smells: List[CodeSmell],
        suggestions: List[RefactoringSuggestion],
    ) -> Dict[str, Any]:
        """Generate summary of analysis."""
        smell_counts = {}
        for smell in smells:
            key = smell.smell_type.name
            smell_counts[key] = smell_counts.get(key, 0) + 1

        severity_counts = {}
        for smell in smells:
            key = smell.severity.name
            severity_counts[key] = severity_counts.get(key, 0) + 1

        refactoring_counts = {}
        for suggestion in suggestions:
            key = suggestion.refactoring_type.name
            refactoring_counts[key] = refactoring_counts.get(key, 0) + 1

        return {
            "total_smells": len(smells),
            "total_suggestions": len(suggestions),
            "smell_breakdown": smell_counts,
            "severity_breakdown": severity_counts,
            "refactoring_breakdown": refactoring_counts,
            "estimated_total_effort": self._estimate_total_effort(suggestions),
        }

    def _estimate_total_effort(self, suggestions: List[RefactoringSuggestion]) -> str:
        """Estimate total effort for all suggestions."""
        # Simple heuristic based on impact levels
        effort_map = {
            Impact.MINIMAL: 0.25,
            Impact.LOW: 0.5,
            Impact.MEDIUM: 1.5,
            Impact.HIGH: 3,
            Impact.BREAKING: 5,
        }

        total_hours = sum(effort_map.get(s.impact, 1) for s in suggestions)

        if total_hours < 1:
            return "Less than 1 hour"
        elif total_hours < 8:
            return f"{total_hours:.1f} hours"
        else:
            days = total_hours / 8
            return f"{days:.1f} days"


# Convenience functions
def analyze_code(code: str, file: str = "<unknown>") -> RefactoringReport:
    """Analyze code for refactoring opportunities."""
    suggester = RefactoringSuggester()
    return suggester.analyze(code, file)


def detect_smells(
    code: str,
    file: str = "<unknown>",
    smell_types: Optional[List[CodeSmellType]] = None,
) -> List[CodeSmell]:
    """Detect specific code smells."""
    suggester = RefactoringSuggester()

    all_smells = []
    for detector in suggester.detectors:
        if smell_types is None or detector.smell_type in smell_types:
            smells = detector.detect(code, file)
            all_smells.extend(smells)

    return all_smells


def suggest_refactorings(code: str, file: str = "<unknown>") -> List[RefactoringSuggestion]:
    """Get refactoring suggestions for code."""
    report = analyze_code(code, file)
    return report.suggestions


def get_impact_assessment(
    suggestion: RefactoringSuggestion,
    code: str,
    file: str = "<unknown>",
) -> ImpactAssessment:
    """Get impact assessment for a refactoring suggestion."""
    suggester = RefactoringSuggester()
    return suggester._assess_impact(suggestion, code, file)


def create_long_method_detector(
    max_lines: int = 30, max_statements: int = 20
) -> LongMethodDetector:
    """Create a long method detector with custom thresholds."""
    return LongMethodDetector(max_lines, max_statements)


def create_parameter_detector(max_params: int = 5) -> LongParameterListDetector:
    """Create a parameter list detector with custom threshold."""
    return LongParameterListDetector(max_params)


def create_duplicate_detector(
    min_lines: int = 5, similarity_threshold: float = 0.8
) -> DuplicateCodeDetector:
    """Create a duplicate code detector with custom settings."""
    return DuplicateCodeDetector(min_lines, similarity_threshold)


def create_nesting_detector(max_depth: int = 3) -> NestedConditionalDetector:
    """Create a nesting detector with custom depth threshold."""
    return NestedConditionalDetector(max_depth)


def create_class_size_detector(
    max_methods: int = 20, max_attributes: int = 15
) -> LargeClassDetector:
    """Create a class size detector with custom thresholds."""
    return LargeClassDetector(max_methods, max_attributes)
