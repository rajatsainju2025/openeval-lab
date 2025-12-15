"""
Pattern Detector for code analysis.

This module provides tools for recognizing design patterns,
detecting anti-patterns, and providing pattern recommendations.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from uuid import uuid4


class PatternCategory(Enum):
    """Categories of design patterns."""

    CREATIONAL = auto()
    STRUCTURAL = auto()
    BEHAVIORAL = auto()
    CONCURRENCY = auto()
    ARCHITECTURAL = auto()
    PYTHONIC = auto()


class PatternType(Enum):
    """Types of design patterns."""

    # Creational patterns
    SINGLETON = auto()
    FACTORY_METHOD = auto()
    ABSTRACT_FACTORY = auto()
    BUILDER = auto()
    PROTOTYPE = auto()

    # Structural patterns
    ADAPTER = auto()
    BRIDGE = auto()
    COMPOSITE = auto()
    DECORATOR = auto()
    FACADE = auto()
    FLYWEIGHT = auto()
    PROXY = auto()

    # Behavioral patterns
    CHAIN_OF_RESPONSIBILITY = auto()
    COMMAND = auto()
    INTERPRETER = auto()
    ITERATOR = auto()
    MEDIATOR = auto()
    MEMENTO = auto()
    OBSERVER = auto()
    STATE = auto()
    STRATEGY = auto()
    TEMPLATE_METHOD = auto()
    VISITOR = auto()

    # Pythonic patterns
    CONTEXT_MANAGER = auto()
    DESCRIPTOR = auto()
    METACLASS = auto()
    MIXIN = auto()
    BORG = auto()


class AntiPatternType(Enum):
    """Types of anti-patterns."""

    # Code smells
    GOD_CLASS = auto()
    GOD_METHOD = auto()
    SPAGHETTI_CODE = auto()
    COPY_PASTE = auto()
    DEAD_CODE = auto()

    # Design anti-patterns
    CIRCULAR_DEPENDENCY = auto()
    BLOB = auto()
    POLTERGEIST = auto()
    GOLDEN_HAMMER = auto()
    LAVA_FLOW = auto()

    # Architecture anti-patterns
    BIG_BALL_OF_MUD = auto()
    VENDOR_LOCK_IN = auto()
    REINVENTING_THE_WHEEL = auto()

    # Python-specific
    MUTABLE_DEFAULT = auto()
    BARE_EXCEPT = auto()
    STAR_IMPORT = auto()
    GLOBAL_STATE = auto()


class Confidence(Enum):
    """Confidence levels for pattern detection."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CERTAIN = auto()


@dataclass
class PatternLocation:
    """Location of a detected pattern."""

    file: str
    class_name: Optional[str] = None
    method_name: Optional[str] = None
    line_start: int = 0
    line_end: int = 0

    def __str__(self) -> str:
        parts = [self.file]
        if self.class_name:
            parts.append(self.class_name)
        if self.method_name:
            parts.append(self.method_name)
        return ":".join(parts)


@dataclass
class PatternMatch:
    """Represents a detected pattern match."""

    id: str
    pattern_type: PatternType
    category: PatternCategory
    confidence: Confidence
    location: PatternLocation
    description: str
    evidence: List[str]
    code_snippet: str
    participants: Dict[str, str] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.name,
            "category": self.category.name,
            "confidence": self.confidence.name,
            "location": str(self.location),
            "description": self.description,
            "evidence": self.evidence,
            "code_snippet": (
                self.code_snippet[:300] + "..."
                if len(self.code_snippet) > 300
                else self.code_snippet
            ),
            "participants": self.participants,
            "recommendations": self.recommendations,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class AntiPatternMatch:
    """Represents a detected anti-pattern."""

    id: str
    anti_pattern_type: AntiPatternType
    severity: str  # low, medium, high, critical
    location: PatternLocation
    description: str
    evidence: List[str]
    code_snippet: str
    impact: str
    remediation: str
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "anti_pattern_type": self.anti_pattern_type.name,
            "severity": self.severity,
            "location": str(self.location),
            "description": self.description,
            "evidence": self.evidence,
            "code_snippet": (
                self.code_snippet[:300] + "..."
                if len(self.code_snippet) > 300
                else self.code_snippet
            ),
            "impact": self.impact,
            "remediation": self.remediation,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class PatternRecommendation:
    """A recommendation for using a pattern."""

    id: str
    pattern_type: PatternType
    context: str
    rationale: str
    example_code: str
    benefits: List[str]
    tradeoffs: List[str]
    confidence: Confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.name,
            "context": self.context,
            "rationale": self.rationale,
            "example_code": self.example_code,
            "benefits": self.benefits,
            "tradeoffs": self.tradeoffs,
            "confidence": self.confidence.name,
        }


@dataclass
class PatternReport:
    """Report of pattern analysis."""

    file: str
    patterns: List[PatternMatch]
    anti_patterns: List[AntiPatternMatch]
    recommendations: List[PatternRecommendation]
    summary: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file": self.file,
            "patterns": [p.to_dict() for p in self.patterns],
            "anti_patterns": [ap.to_dict() for ap in self.anti_patterns],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "summary": self.summary,
            "generated_at": self.generated_at.isoformat(),
        }


class PatternDetectorBase(ABC):
    """Base class for pattern detectors."""

    @property
    @abstractmethod
    def pattern_type(self) -> PatternType:
        """Return the type of pattern this detector finds."""
        pass

    @property
    @abstractmethod
    def category(self) -> PatternCategory:
        """Return the category of pattern."""
        pass

    @abstractmethod
    def detect(self, code: str, file: str = "<unknown>") -> List[PatternMatch]:
        """Detect patterns in the given code."""
        pass


class AntiPatternDetectorBase(ABC):
    """Base class for anti-pattern detectors."""

    @property
    @abstractmethod
    def anti_pattern_type(self) -> AntiPatternType:
        """Return the type of anti-pattern this detector finds."""
        pass

    @abstractmethod
    def detect(self, code: str, file: str = "<unknown>") -> List[AntiPatternMatch]:
        """Detect anti-patterns in the given code."""
        pass


class SingletonDetector(PatternDetectorBase):
    """Detect singleton pattern implementations."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.SINGLETON

    @property
    def category(self) -> PatternCategory:
        return PatternCategory.CREATIONAL

    def detect(self, code: str, file: str = "<unknown>") -> List[PatternMatch]:
        """Detect singleton patterns."""
        matches = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return matches

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                evidence = []
                confidence = Confidence.LOW

                # Check for _instance class attribute
                has_instance_attr = False
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                if target.id in ("_instance", "__instance", "instance"):
                                    has_instance_attr = True
                                    evidence.append(f"Has instance attribute: {target.id}")

                # Check for __new__ method
                has_new_method = False
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__new__":
                        has_new_method = True
                        evidence.append("Implements __new__ method")

                # Check for get_instance method
                has_get_instance = False
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in ("get_instance", "getInstance", "instance"):
                            has_get_instance = True
                            evidence.append(f"Has instance method: {item.name}")

                # Calculate confidence
                if has_instance_attr and has_new_method:
                    confidence = Confidence.HIGH
                elif has_instance_attr and has_get_instance:
                    confidence = Confidence.HIGH
                elif has_instance_attr or has_new_method or has_get_instance:
                    confidence = Confidence.MEDIUM

                if evidence:
                    code_snippet = "\n".join(
                        lines[node.lineno - 1 : min(node.end_lineno, node.lineno + 15)]
                    )

                    matches.append(
                        PatternMatch(
                            id=str(uuid4()),
                            pattern_type=self.pattern_type,
                            category=self.category,
                            confidence=confidence,
                            location=PatternLocation(
                                file=file,
                                class_name=node.name,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                            ),
                            description=(f"Singleton pattern detected in class '{node.name}'"),
                            evidence=evidence,
                            code_snippet=code_snippet,
                            participants={"Singleton": node.name},
                            recommendations=[
                                "Consider using module-level instance for simpler implementation",
                                "Ensure thread-safety if used in concurrent code",
                            ],
                        )
                    )

        return matches


class FactoryMethodDetector(PatternDetectorBase):
    """Detect factory method pattern implementations."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.FACTORY_METHOD

    @property
    def category(self) -> PatternCategory:
        return PatternCategory.CREATIONAL

    def detect(self, code: str, file: str = "<unknown>") -> List[PatternMatch]:
        """Detect factory method patterns."""
        matches = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return matches

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                factory_methods = []

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Check for factory method indicators
                        name_lower = item.name.lower()
                        if any(
                            keyword in name_lower
                            for keyword in ("create", "make", "build", "factory", "new")
                        ):
                            # Check if it returns an object
                            has_return = False
                            for subnode in ast.walk(item):
                                if isinstance(subnode, ast.Return) and subnode.value:
                                    has_return = True
                                    break

                            if has_return:
                                factory_methods.append(item.name)

                if factory_methods:
                    code_snippet = "\n".join(
                        lines[node.lineno - 1 : min(node.end_lineno, node.lineno + 20)]
                    )

                    matches.append(
                        PatternMatch(
                            id=str(uuid4()),
                            pattern_type=self.pattern_type,
                            category=self.category,
                            confidence=Confidence.MEDIUM,
                            location=PatternLocation(
                                file=file,
                                class_name=node.name,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                            ),
                            description=(f"Factory method pattern detected in class '{node.name}'"),
                            evidence=[f"Factory method: {m}" for m in factory_methods],
                            code_snippet=code_snippet,
                            participants={
                                "Creator": node.name,
                                "Factory Methods": ", ".join(factory_methods),
                            },
                            recommendations=[
                                "Consider using abstract base class for formal pattern",
                                "Document what types are created by each factory method",
                            ],
                        )
                    )

        return matches


class DecoratorPatternDetector(PatternDetectorBase):
    """Detect decorator pattern implementations."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.DECORATOR

    @property
    def category(self) -> PatternCategory:
        return PatternCategory.STRUCTURAL

    def detect(self, code: str, file: str = "<unknown>") -> List[PatternMatch]:
        """Detect decorator patterns (structural pattern, not Python decorators)."""
        matches = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return matches

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                evidence = []

                # Check for wrapped/component attribute in __init__
                has_wrapped = False
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        for subnode in ast.walk(item):
                            if isinstance(subnode, ast.Attribute):
                                attr_name = subnode.attr.lower()
                                if any(
                                    keyword in attr_name
                                    for keyword in (
                                        "wrapped",
                                        "component",
                                        "decorated",
                                        "inner",
                                    )
                                ):
                                    has_wrapped = True
                                    evidence.append(f"Wraps component in attribute: {subnode.attr}")

                # Check for method delegation
                delegation_count = 0
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        for subnode in ast.walk(item):
                            if isinstance(subnode, ast.Call):
                                if isinstance(subnode.func, ast.Attribute):
                                    if subnode.func.attr == item.name:
                                        delegation_count += 1

                if delegation_count >= 2:
                    evidence.append(f"Delegates {delegation_count} methods to wrapped")

                if has_wrapped and evidence:
                    code_snippet = "\n".join(
                        lines[node.lineno - 1 : min(node.end_lineno, node.lineno + 20)]
                    )

                    matches.append(
                        PatternMatch(
                            id=str(uuid4()),
                            pattern_type=self.pattern_type,
                            category=self.category,
                            confidence=(
                                Confidence.MEDIUM if delegation_count >= 2 else Confidence.LOW
                            ),
                            location=PatternLocation(
                                file=file,
                                class_name=node.name,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                            ),
                            description=(f"Decorator pattern detected in class '{node.name}'"),
                            evidence=evidence,
                            code_snippet=code_snippet,
                            participants={"Decorator": node.name},
                            recommendations=[
                                "Ensure all component interface methods are delegated",
                                "Consider using composition over inheritance",
                            ],
                        )
                    )

        return matches


class StrategyDetector(PatternDetectorBase):
    """Detect strategy pattern implementations."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.STRATEGY

    @property
    def category(self) -> PatternCategory:
        return PatternCategory.BEHAVIORAL

    def detect(self, code: str, file: str = "<unknown>") -> List[PatternMatch]:
        """Detect strategy patterns."""
        matches = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return matches

        lines = code.split("\n")

        # Find abstract base classes that look like strategies
        potential_strategies = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for ABC base
                is_abstract = False
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in ("ABC", "Protocol"):
                        is_abstract = True
                    elif isinstance(base, ast.Attribute) and base.attr in (
                        "ABC",
                        "Protocol",
                    ):
                        is_abstract = True

                # Check for abstract methods
                abstract_methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        for decorator in item.decorator_list:
                            if isinstance(decorator, ast.Name):
                                if decorator.id == "abstractmethod":
                                    abstract_methods.append(item.name)

                name_lower = node.name.lower()
                if (
                    is_abstract
                    and abstract_methods
                    and any(
                        keyword in name_lower
                        for keyword in ("strategy", "policy", "handler", "algorithm")
                    )
                ):
                    potential_strategies[node.name] = {
                        "node": node,
                        "methods": abstract_methods,
                    }

        for strategy_name, info in potential_strategies.items():
            node = info["node"]
            code_snippet = "\n".join(
                lines[node.lineno - 1 : min(node.end_lineno, node.lineno + 15)]
            )

            matches.append(
                PatternMatch(
                    id=str(uuid4()),
                    pattern_type=self.pattern_type,
                    category=self.category,
                    confidence=Confidence.HIGH,
                    location=PatternLocation(
                        file=file,
                        class_name=strategy_name,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                    ),
                    description=f"Strategy pattern interface '{strategy_name}'",
                    evidence=[
                        "Abstract base class",
                        f"Abstract methods: {', '.join(info['methods'])}",
                    ],
                    code_snippet=code_snippet,
                    participants={"Strategy Interface": strategy_name},
                    recommendations=[
                        "Implement concrete strategies for different algorithms",
                        "Consider using Protocol for structural subtyping",
                    ],
                )
            )

        return matches


class ContextManagerDetector(PatternDetectorBase):
    """Detect context manager pattern implementations."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.CONTEXT_MANAGER

    @property
    def category(self) -> PatternCategory:
        return PatternCategory.PYTHONIC

    def detect(self, code: str, file: str = "<unknown>") -> List[PatternMatch]:
        """Detect context manager patterns."""
        matches = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return matches

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_enter = False
                has_exit = False

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "__enter__":
                            has_enter = True
                        elif item.name == "__exit__":
                            has_exit = True

                if has_enter and has_exit:
                    code_snippet = "\n".join(
                        lines[node.lineno - 1 : min(node.end_lineno, node.lineno + 20)]
                    )

                    matches.append(
                        PatternMatch(
                            id=str(uuid4()),
                            pattern_type=self.pattern_type,
                            category=self.category,
                            confidence=Confidence.CERTAIN,
                            location=PatternLocation(
                                file=file,
                                class_name=node.name,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                            ),
                            description=(f"Context manager implemented in class '{node.name}'"),
                            evidence=[
                                "Implements __enter__ method",
                                "Implements __exit__ method",
                            ],
                            code_snippet=code_snippet,
                            participants={"Context Manager": node.name},
                            recommendations=[
                                "Consider @contextmanager decorator for simpler cases",
                                "Handle exceptions properly in __exit__",
                            ],
                        )
                    )

        return matches


class MutableDefaultDetector(AntiPatternDetectorBase):
    """Detect mutable default argument anti-pattern."""

    @property
    def anti_pattern_type(self) -> AntiPatternType:
        return AntiPatternType.MUTABLE_DEFAULT

    def detect(self, code: str, file: str = "<unknown>") -> List[AntiPatternMatch]:
        """Detect mutable default arguments."""
        matches = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return matches

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is None:
                        continue

                    is_mutable = False
                    mutable_type = ""

                    if isinstance(default, ast.List):
                        is_mutable = True
                        mutable_type = "list"
                    elif isinstance(default, ast.Dict):
                        is_mutable = True
                        mutable_type = "dict"
                    elif isinstance(default, ast.Set):
                        is_mutable = True
                        mutable_type = "set"
                    elif isinstance(default, ast.Call):
                        if isinstance(default.func, ast.Name):
                            if default.func.id in ("list", "dict", "set"):
                                is_mutable = True
                                mutable_type = default.func.id

                    if is_mutable:
                        code_snippet = lines[node.lineno - 1]

                        matches.append(
                            AntiPatternMatch(
                                id=str(uuid4()),
                                anti_pattern_type=self.anti_pattern_type,
                                severity="medium",
                                location=PatternLocation(
                                    file=file,
                                    method_name=node.name,
                                    line_start=node.lineno,
                                    line_end=node.lineno,
                                ),
                                description=(
                                    f"Mutable default argument ({mutable_type}) in "
                                    f"function '{node.name}'"
                                ),
                                evidence=[
                                    f"Default value is a mutable {mutable_type}",
                                    "Mutable defaults are shared across calls",
                                ],
                                code_snippet=code_snippet,
                                impact=("Unexpected behavior when the default is modified"),
                                remediation=(
                                    f"Use None as default and create {mutable_type} "
                                    "inside function"
                                ),
                            )
                        )

        return matches


class BareExceptDetector(AntiPatternDetectorBase):
    """Detect bare except anti-pattern."""

    @property
    def anti_pattern_type(self) -> AntiPatternType:
        return AntiPatternType.BARE_EXCEPT

    def detect(self, code: str, file: str = "<unknown>") -> List[AntiPatternMatch]:
        """Detect bare except clauses."""
        matches = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return matches

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    code_snippet = "\n".join(lines[node.lineno - 1 : node.end_lineno])

                    matches.append(
                        AntiPatternMatch(
                            id=str(uuid4()),
                            anti_pattern_type=self.anti_pattern_type,
                            severity="high",
                            location=PatternLocation(
                                file=file,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                            ),
                            description="Bare except clause catches all exceptions",
                            evidence=[
                                "Except clause without exception type",
                                "Catches SystemExit, KeyboardInterrupt, etc.",
                            ],
                            code_snippet=code_snippet,
                            impact="May hide bugs and prevent proper error handling",
                            remediation="Use 'except Exception:' or specific exceptions",
                        )
                    )

        return matches


class StarImportDetector(AntiPatternDetectorBase):
    """Detect star import anti-pattern."""

    @property
    def anti_pattern_type(self) -> AntiPatternType:
        return AntiPatternType.STAR_IMPORT

    def detect(self, code: str, file: str = "<unknown>") -> List[AntiPatternMatch]:
        """Detect star imports."""
        matches = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return matches

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        code_snippet = lines[node.lineno - 1]

                        matches.append(
                            AntiPatternMatch(
                                id=str(uuid4()),
                                anti_pattern_type=self.anti_pattern_type,
                                severity="low",
                                location=PatternLocation(
                                    file=file,
                                    line_start=node.lineno,
                                    line_end=node.lineno,
                                ),
                                description=f"Star import from '{node.module}'",
                                evidence=[
                                    "Imports all names from module",
                                    "Namespace pollution",
                                ],
                                code_snippet=code_snippet,
                                impact="Makes it unclear where names come from",
                                remediation="Import specific names or use module prefix",
                            )
                        )

        return matches


class GodClassDetector(AntiPatternDetectorBase):
    """Detect god class anti-pattern."""

    def __init__(
        self,
        max_methods: int = 25,
        max_attributes: int = 20,
        max_lines: int = 500,
    ):
        self.max_methods = max_methods
        self.max_attributes = max_attributes
        self.max_lines = max_lines

    @property
    def anti_pattern_type(self) -> AntiPatternType:
        return AntiPatternType.GOD_CLASS

    def detect(self, code: str, file: str = "<unknown>") -> List[AntiPatternMatch]:
        """Detect god classes."""
        matches = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return matches

        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                line_count = node.end_lineno - node.lineno + 1

                # Count attributes
                attributes = set()
                for item in ast.walk(node):
                    if isinstance(item, ast.Attribute):
                        if isinstance(item.value, ast.Name) and item.value.id == "self":
                            attributes.add(item.attr)

                evidence = []
                is_god_class = False

                if len(methods) > self.max_methods:
                    evidence.append(f"Too many methods: {len(methods)} > {self.max_methods}")
                    is_god_class = True

                if len(attributes) > self.max_attributes:
                    evidence.append(
                        f"Too many attributes: {len(attributes)} > {self.max_attributes}"
                    )
                    is_god_class = True

                if line_count > self.max_lines:
                    evidence.append(f"Too many lines: {line_count} > {self.max_lines}")
                    is_god_class = True

                if is_god_class:
                    code_snippet = "\n".join(
                        lines[node.lineno - 1 : min(node.end_lineno, node.lineno + 10)]
                    )

                    matches.append(
                        AntiPatternMatch(
                            id=str(uuid4()),
                            anti_pattern_type=self.anti_pattern_type,
                            severity="high",
                            location=PatternLocation(
                                file=file,
                                class_name=node.name,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                            ),
                            description=f"God class detected: '{node.name}'",
                            evidence=evidence,
                            code_snippet=code_snippet,
                            impact=(
                                "Hard to understand, test, and maintain. "
                                "Violates Single Responsibility Principle."
                            ),
                            remediation=(
                                "Split into smaller, focused classes. "
                                "Extract cohesive groups of methods."
                            ),
                        )
                    )

        return matches


class PatternDetector:
    """Main class for detecting patterns and anti-patterns."""

    def __init__(
        self,
        pattern_detectors: Optional[List[PatternDetectorBase]] = None,
        anti_pattern_detectors: Optional[List[AntiPatternDetectorBase]] = None,
    ):
        """Initialize the pattern detector."""
        self.pattern_detectors = pattern_detectors or self._get_default_pattern_detectors()
        self.anti_pattern_detectors = (
            anti_pattern_detectors or self._get_default_anti_pattern_detectors()
        )

    def _get_default_pattern_detectors(self) -> List[PatternDetectorBase]:
        """Get default pattern detectors."""
        return [
            SingletonDetector(),
            FactoryMethodDetector(),
            DecoratorPatternDetector(),
            StrategyDetector(),
            ContextManagerDetector(),
        ]

    def _get_default_anti_pattern_detectors(self) -> List[AntiPatternDetectorBase]:
        """Get default anti-pattern detectors."""
        return [
            MutableDefaultDetector(),
            BareExceptDetector(),
            StarImportDetector(),
            GodClassDetector(),
        ]

    def analyze(self, code: str, file: str = "<unknown>") -> PatternReport:
        """Analyze code for patterns and anti-patterns."""
        # Detect patterns
        all_patterns = []
        for detector in self.pattern_detectors:
            patterns = detector.detect(code, file)
            all_patterns.extend(patterns)

        # Detect anti-patterns
        all_anti_patterns = []
        for detector in self.anti_pattern_detectors:
            anti_patterns = detector.detect(code, file)
            all_anti_patterns.extend(anti_patterns)

        # Generate recommendations
        recommendations = self._generate_recommendations(code, all_patterns, all_anti_patterns)

        # Generate summary
        summary = self._generate_summary(all_patterns, all_anti_patterns)

        return PatternReport(
            file=file,
            patterns=all_patterns,
            anti_patterns=all_anti_patterns,
            recommendations=recommendations,
            summary=summary,
        )

    def _generate_recommendations(
        self,
        code: str,
        patterns: List[PatternMatch],
        anti_patterns: List[AntiPatternMatch],
    ) -> List[PatternRecommendation]:
        """Generate pattern recommendations based on code analysis."""
        recommendations = []

        # Check for patterns that might benefit from other patterns
        pattern_types = {p.pattern_type for p in patterns}

        # If multiple creation points detected, suggest Factory
        if PatternType.FACTORY_METHOD not in pattern_types:
            if self._has_multiple_creation_points(code):
                recommendations.append(
                    PatternRecommendation(
                        id=str(uuid4()),
                        pattern_type=PatternType.FACTORY_METHOD,
                        context="Multiple object creation points detected",
                        rationale=("Centralizing object creation improves maintainability"),
                        example_code=(
                            "class ProductFactory:\n"
                            "    @staticmethod\n"
                            "    def create(product_type: str) -> Product:\n"
                            "        if product_type == 'A':\n"
                            "            return ProductA()\n"
                            "        return ProductB()"
                        ),
                        benefits=[
                            "Centralizes creation logic",
                            "Easier to add new product types",
                        ],
                        tradeoffs=["Additional abstraction layer"],
                        confidence=Confidence.MEDIUM,
                    )
                )

        # If god class detected, suggest Strategy pattern
        for ap in anti_patterns:
            if ap.anti_pattern_type == AntiPatternType.GOD_CLASS:
                recommendations.append(
                    PatternRecommendation(
                        id=str(uuid4()),
                        pattern_type=PatternType.STRATEGY,
                        context=f"God class detected: {ap.location.class_name}",
                        rationale="Extract algorithms into separate strategy classes",
                        example_code=(
                            "class AlgorithmStrategy(ABC):\n"
                            "    @abstractmethod\n"
                            "    def execute(self, data): pass\n\n"
                            "class Context:\n"
                            "    def __init__(self, strategy: AlgorithmStrategy):\n"
                            "        self.strategy = strategy"
                        ),
                        benefits=[
                            "Separates concerns",
                            "Algorithms can vary independently",
                        ],
                        tradeoffs=["More classes to manage"],
                        confidence=Confidence.HIGH,
                    )
                )

        return recommendations

    def _has_multiple_creation_points(self, code: str) -> bool:
        """Check if code has multiple object creation points."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        creation_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Heuristic: capitalized names are likely class instantiations
                    if node.func.id[0].isupper():
                        creation_count += 1

        return creation_count >= 5

    def _generate_summary(
        self,
        patterns: List[PatternMatch],
        anti_patterns: List[AntiPatternMatch],
    ) -> Dict[str, Any]:
        """Generate summary of pattern analysis."""
        pattern_counts = {}
        for pattern in patterns:
            key = pattern.pattern_type.name
            pattern_counts[key] = pattern_counts.get(key, 0) + 1

        anti_pattern_counts = {}
        for ap in anti_patterns:
            key = ap.anti_pattern_type.name
            anti_pattern_counts[key] = anti_pattern_counts.get(key, 0) + 1

        severity_counts = {}
        for ap in anti_patterns:
            severity_counts[ap.severity] = severity_counts.get(ap.severity, 0) + 1

        return {
            "total_patterns": len(patterns),
            "total_anti_patterns": len(anti_patterns),
            "pattern_breakdown": pattern_counts,
            "anti_pattern_breakdown": anti_pattern_counts,
            "severity_breakdown": severity_counts,
        }


# Convenience functions
def detect_patterns(code: str, file: str = "<unknown>") -> List[PatternMatch]:
    """Detect design patterns in code."""
    detector = PatternDetector()
    report = detector.analyze(code, file)
    return report.patterns


def detect_anti_patterns(code: str, file: str = "<unknown>") -> List[AntiPatternMatch]:
    """Detect anti-patterns in code."""
    detector = PatternDetector()
    report = detector.analyze(code, file)
    return report.anti_patterns


def analyze_patterns(code: str, file: str = "<unknown>") -> PatternReport:
    """Full pattern analysis of code."""
    detector = PatternDetector()
    return detector.analyze(code, file)


def get_pattern_recommendations(code: str, file: str = "<unknown>") -> List[PatternRecommendation]:
    """Get pattern recommendations for code."""
    detector = PatternDetector()
    report = detector.analyze(code, file)
    return report.recommendations


def create_singleton_detector() -> SingletonDetector:
    """Create a singleton pattern detector."""
    return SingletonDetector()


def create_factory_detector() -> FactoryMethodDetector:
    """Create a factory method pattern detector."""
    return FactoryMethodDetector()


def create_god_class_detector(
    max_methods: int = 25,
    max_attributes: int = 20,
    max_lines: int = 500,
) -> GodClassDetector:
    """Create a god class detector with custom thresholds."""
    return GodClassDetector(max_methods, max_attributes, max_lines)
