"""Explanation validator for verifying explanation accuracy.

This module provides tools to validate that code explanations accurately
describe the code's behavior, catch potential hallucinations, and ensure
explanations are grounded in the actual code semantics.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import CodeElement, CodeElementType, ExplanationResult


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class ValidationLevel(str, Enum):
    """Strictness level for validation."""

    LENIENT = "lenient"
    STANDARD = "standard"
    STRICT = "strict"


class ValidationStatus(str, Enum):
    """Status of a validation check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class IssueType(str, Enum):
    """Types of validation issues."""

    HALLUCINATION = "hallucination"
    MISSING_INFO = "missing_info"
    INCORRECT_CLAIM = "incorrect_claim"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    INCONSISTENCY = "inconsistency"
    AMBIGUITY = "ambiguity"
    OUTDATED = "outdated"
    TERMINOLOGY = "terminology"


class IssueSeverity(str, Enum):
    """Severity of validation issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ValidationIssue:
    """A single validation issue found in an explanation."""

    type: IssueType
    severity: IssueSeverity
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    evidence: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    status: ValidationStatus
    message: str
    duration_ms: float = 0.0
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "issues": [i.to_dict() for i in self.issues],
            "metadata": self.metadata,
        }


@dataclass
class ValidationResult:
    """Complete result of validating an explanation."""

    is_valid: bool
    overall_status: ValidationStatus
    confidence: float  # 0.0 to 1.0
    checks: List[ValidationCheck] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: str = ""
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_check(self, check: ValidationCheck) -> None:
        """Add a validation check result."""
        self.checks.append(check)
        self.issues.extend(check.issues)

    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get all critical severity issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    def get_issues_by_type(self, issue_type: IssueType) -> List[ValidationIssue]:
        """Get issues of a specific type."""
        return [i for i in self.issues if i.type == issue_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "overall_status": self.overall_status.value,
            "confidence": self.confidence,
            "checks": [c.to_dict() for c in self.checks],
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ValidatorConfig:
    """Configuration for explanation validators."""

    level: ValidationLevel = ValidationLevel.STANDARD
    fail_on_warning: bool = False
    check_hallucinations: bool = True
    check_completeness: bool = True
    check_consistency: bool = True
    check_terminology: bool = True
    min_confidence: float = 0.7
    max_issues: int = 100
    timeout_ms: float = 10000.0
    custom_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Abstract Base Class
# =============================================================================


class ValidationRule(ABC):
    """Abstract base class for validation rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this rule."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this rule checks."""
        pass

    @abstractmethod
    def validate(
        self,
        explanation: ExplanationResult,
        element: CodeElement,
        config: ValidatorConfig,
    ) -> ValidationCheck:
        """Validate an explanation against this rule.

        Args:
            explanation: The explanation to validate.
            element: The code element being explained.
            config: Validator configuration.

        Returns:
            ValidationCheck with results.
        """
        pass


# =============================================================================
# Built-in Validation Rules
# =============================================================================


class HallucinationDetector(ValidationRule):
    """Detects potential hallucinations in explanations."""

    @property
    def name(self) -> str:
        return "hallucination_detector"

    @property
    def description(self) -> str:
        return "Checks for claims not supported by the code"

    def validate(
        self,
        explanation: ExplanationResult,
        element: CodeElement,
        config: ValidatorConfig,
    ) -> ValidationCheck:
        """Check for hallucinated content."""
        import time

        start = time.time()
        issues = []
        code = element.source_code.lower()
        text = explanation.explanation.lower()

        # Check for mentioned function/method names not in code
        mentioned_funcs = self._extract_mentioned_functions(text)
        code_funcs = self._extract_code_identifiers(element.source_code)

        for func in mentioned_funcs:
            if func not in code_funcs and func not in code:
                issues.append(
                    ValidationIssue(
                        type=IssueType.HALLUCINATION,
                        severity=IssueSeverity.HIGH,
                        message=f"Mentioned '{func}' not found in code",
                        suggestion="Verify this identifier exists in the code",
                        evidence=f"'{func}' not in code identifiers",
                    )
                )

        # Check for mentioned types/classes not in code
        mentioned_types = self._extract_mentioned_types(text)
        for type_name in mentioned_types:
            if type_name.lower() not in code and len(type_name) > 3:
                # Check if it's a common Python type
                common_types = {
                    "list",
                    "dict",
                    "set",
                    "tuple",
                    "string",
                    "str",
                    "int",
                    "float",
                    "bool",
                    "none",
                    "array",
                    "object",
                    "function",
                    "class",
                    "module",
                    "iterator",
                    "generator",
                }
                if type_name.lower() not in common_types:
                    issues.append(
                        ValidationIssue(
                            type=IssueType.HALLUCINATION,
                            severity=IssueSeverity.MEDIUM,
                            message=f"Mentioned type '{type_name}' not evident in code",
                            suggestion="Check if this type is actually used",
                        )
                    )

        # Check for specific number claims
        numbers_in_text = re.findall(r"\b(\d+)\b", text)
        for num in numbers_in_text:
            if int(num) > 1 and num not in element.source_code:
                # Could be a hallucinated count or value
                issues.append(
                    ValidationIssue(
                        type=IssueType.UNSUPPORTED_CLAIM,
                        severity=IssueSeverity.LOW,
                        message=f"Number '{num}' mentioned but not found in code",
                        suggestion="Verify numeric claims are accurate",
                    )
                )

        status = ValidationStatus.PASSED
        if any(i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH] for i in issues):
            status = ValidationStatus.FAILED
        elif issues:
            status = ValidationStatus.WARNING

        return ValidationCheck(
            name=self.name,
            status=status,
            message=f"Found {len(issues)} potential hallucinations",
            duration_ms=(time.time() - start) * 1000,
            issues=issues,
        )

    def _extract_mentioned_functions(self, text: str) -> Set[str]:
        """Extract function names mentioned in text."""
        # Look for patterns like "the X function" or "calls X"
        patterns = [
            r"(?:the\s+)?(\w+)\s+function",
            r"(?:calls?|invoke[sd]?|run[s]?)\s+(\w+)",
            r"(\w+)\s+method",
            r"(\w+)\(\)",
        ]
        funcs = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            funcs.update(matches)
        return funcs

    def _extract_mentioned_types(self, text: str) -> Set[str]:
        """Extract type names mentioned in text."""
        patterns = [
            r"(\w+)\s+(?:class|type|object)",
            r"(?:a|an|the)\s+(\w+)\s+instance",
            r"returns?\s+(?:a|an)?\s*(\w+)",
        ]
        types = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            types.update(matches)
        return types

    def _extract_code_identifiers(self, code: str) -> Set[str]:
        """Extract all identifiers from code."""
        # Match word characters that look like identifiers
        identifiers = set(re.findall(r"\b([a-zA-Z_]\w*)\b", code))
        return identifiers


class CompletenessChecker(ValidationRule):
    """Checks if explanation covers all key aspects of the code."""

    @property
    def name(self) -> str:
        return "completeness_checker"

    @property
    def description(self) -> str:
        return "Checks if explanation covers key code aspects"

    def validate(
        self,
        explanation: ExplanationResult,
        element: CodeElement,
        config: ValidatorConfig,
    ) -> ValidationCheck:
        """Check explanation completeness."""
        import time

        start = time.time()
        issues = []
        text = explanation.explanation.lower()
        code = element.source_code

        # Check if parameters are mentioned (for functions)
        if element.type == CodeElementType.FUNCTION:
            params = self._extract_parameters(code)
            mentioned_params = sum(1 for p in params if p.lower() in text)
            if params and mentioned_params < len(params) * 0.5:
                issues.append(
                    ValidationIssue(
                        type=IssueType.MISSING_INFO,
                        severity=IssueSeverity.MEDIUM,
                        message=f"Only {mentioned_params}/{len(params)} parameters mentioned",
                        suggestion="Consider explaining all parameters",
                        evidence=f"Missing: {[p for p in params if p.lower() not in text]}",
                    )
                )

        # Check if return value is mentioned (for functions)
        if element.type == CodeElementType.FUNCTION:
            has_return = "return" in code and "return" not in code.split("\n")[0]
            return_mentioned = any(
                word in text for word in ["return", "returns", "output", "result"]
            )
            if has_return and not return_mentioned:
                issues.append(
                    ValidationIssue(
                        type=IssueType.MISSING_INFO,
                        severity=IssueSeverity.MEDIUM,
                        message="Function return value not explained",
                        suggestion="Describe what the function returns",
                    )
                )

        # Check if class attributes are mentioned (for classes)
        if element.type == CodeElementType.CLASS:
            attributes = self._extract_class_attributes(code)
            if attributes and not any(a.lower() in text for a in attributes[:3]):
                issues.append(
                    ValidationIssue(
                        type=IssueType.MISSING_INFO,
                        severity=IssueSeverity.LOW,
                        message="Key class attributes not mentioned",
                        suggestion="Consider explaining main attributes",
                        evidence=f"Attributes: {attributes[:5]}",
                    )
                )

        # Check if exceptions/errors are mentioned
        if "raise" in code or "except" in code:
            error_mentioned = any(
                word in text for word in ["error", "exception", "raise", "throw", "fail"]
            )
            if not error_mentioned:
                issues.append(
                    ValidationIssue(
                        type=IssueType.MISSING_INFO,
                        severity=IssueSeverity.LOW,
                        message="Error handling not mentioned",
                        suggestion="Explain exception handling behavior",
                    )
                )

        status = ValidationStatus.PASSED
        if any(i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH] for i in issues):
            status = ValidationStatus.FAILED
        elif issues:
            status = ValidationStatus.WARNING

        return ValidationCheck(
            name=self.name,
            status=status,
            message=f"Found {len(issues)} completeness issues",
            duration_ms=(time.time() - start) * 1000,
            issues=issues,
        )

    def _extract_parameters(self, code: str) -> List[str]:
        """Extract parameter names from function definition."""
        match = re.search(r"def\s+\w+\s*\((.*?)\)", code, re.DOTALL)
        if not match:
            return []

        param_str = match.group(1)
        # Remove type annotations and defaults
        params = []
        for part in param_str.split(","):
            part = part.strip()
            if not part or part == "self" or part == "cls":
                continue
            # Extract just the parameter name
            name = re.match(r"(\*{0,2}\w+)", part)
            if name:
                params.append(name.group(1).lstrip("*"))
        return params

    def _extract_class_attributes(self, code: str) -> List[str]:
        """Extract class attribute names."""
        # Look for self.attribute = ... patterns
        attributes = re.findall(r"self\.(\w+)\s*=", code)
        return list(dict.fromkeys(attributes))  # Remove duplicates, preserve order


class ConsistencyChecker(ValidationRule):
    """Checks for internal consistency in explanations."""

    @property
    def name(self) -> str:
        return "consistency_checker"

    @property
    def description(self) -> str:
        return "Checks for contradictions in the explanation"

    def validate(
        self,
        explanation: ExplanationResult,
        element: CodeElement,
        config: ValidatorConfig,
    ) -> ValidationCheck:
        """Check for inconsistencies."""
        import time

        start = time.time()
        issues = []
        text = explanation.explanation.lower()

        # Check for contradictory statements
        contradictions = [
            (r"always\s+\w+", r"never\s+\w+"),
            (r"must\s+\w+", r"(?:cannot|can't|won't)\s+\w+"),
            (r"returns\s+(?:a\s+)?(\w+)", r"returns\s+(?:a\s+)?(?!\\1)(\w+)"),
        ]

        for pos_pattern, neg_pattern in contradictions:
            pos_matches = re.findall(pos_pattern, text)
            neg_matches = re.findall(neg_pattern, text)
            if pos_matches and neg_matches:
                # Check if they refer to the same thing
                for pos in pos_matches:
                    for neg in neg_matches:
                        if self._similar_context(pos, neg, text):
                            issues.append(
                                ValidationIssue(
                                    type=IssueType.INCONSISTENCY,
                                    severity=IssueSeverity.MEDIUM,
                                    message="Potentially contradictory statements",
                                    evidence=f"'{pos}' vs '{neg}'",
                                )
                            )

        # Check element type consistency
        type_words = {
            CodeElementType.FUNCTION: ["function", "method", "def", "callable"],
            CodeElementType.CLASS: ["class", "type", "object"],
            CodeElementType.MODULE: ["module", "file", "package"],
        }

        expected_words = type_words.get(element.type, [])
        wrong_type_words = []
        for other_type, words in type_words.items():
            if other_type != element.type:
                for word in words:
                    if word in text and word not in expected_words:
                        wrong_type_words.append(word)

        if wrong_type_words and not any(w in text for w in expected_words):
            issues.append(
                ValidationIssue(
                    type=IssueType.INCONSISTENCY,
                    severity=IssueSeverity.MEDIUM,
                    message="Explanation refers to wrong element type",
                    suggestion=f"Element is a {element.type.value}, not {wrong_type_words[0]}",
                    evidence=f"Uses: {wrong_type_words}",
                )
            )

        status = ValidationStatus.PASSED
        if any(i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH] for i in issues):
            status = ValidationStatus.FAILED
        elif issues:
            status = ValidationStatus.WARNING

        return ValidationCheck(
            name=self.name,
            status=status,
            message=f"Found {len(issues)} consistency issues",
            duration_ms=(time.time() - start) * 1000,
            issues=issues,
        )

    def _similar_context(self, phrase1: str, phrase2: str, text: str) -> bool:
        """Check if two phrases appear in similar context."""
        # Simple proximity check - within 100 characters
        pos1 = text.find(phrase1)
        pos2 = text.find(phrase2)
        if pos1 >= 0 and pos2 >= 0:
            return abs(pos1 - pos2) < 100
        return False


class TerminologyChecker(ValidationRule):
    """Checks for correct terminology usage."""

    @property
    def name(self) -> str:
        return "terminology_checker"

    @property
    def description(self) -> str:
        return "Checks for correct technical terminology"

    # Common terminology mistakes
    CORRECTIONS = {
        "method": {"applies_to": [CodeElementType.CLASS], "not_for": [CodeElementType.FUNCTION]},
        "function": {"applies_to": [CodeElementType.FUNCTION], "not_for": [CodeElementType.CLASS]},
        "arguements": {"correct": "arguments"},
        "paramaters": {"correct": "parameters"},
        "inheritence": {"correct": "inheritance"},
        "instanciate": {"correct": "instantiate"},
        "syncronous": {"correct": "synchronous"},
        "asyncronous": {"correct": "asynchronous"},
    }

    def validate(
        self,
        explanation: ExplanationResult,
        element: CodeElement,
        config: ValidatorConfig,
    ) -> ValidationCheck:
        """Check terminology usage."""
        import time

        start = time.time()
        issues = []
        text = explanation.explanation.lower()

        # Check for common misspellings
        for term, info in self.CORRECTIONS.items():
            if "correct" in info and term in text:
                issues.append(
                    ValidationIssue(
                        type=IssueType.TERMINOLOGY,
                        severity=IssueSeverity.LOW,
                        message=f"Misspelling: '{term}'",
                        suggestion=f"Use '{info['correct']}' instead",
                    )
                )

        # Check for incorrect Python terminology
        python_terms = {
            "array": ("Python uses 'list' or 'array' from numpy", IssueSeverity.INFO),
            "null": ("Python uses 'None', not 'null'", IssueSeverity.LOW),
            "undefined": ("Python doesn't have 'undefined', use 'None'", IssueSeverity.LOW),
        }

        for term, (suggestion, severity) in python_terms.items():
            if term in text:
                issues.append(
                    ValidationIssue(
                        type=IssueType.TERMINOLOGY,
                        severity=severity,
                        message=f"Non-Pythonic term: '{term}'",
                        suggestion=suggestion,
                    )
                )

        status = ValidationStatus.PASSED
        if any(i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH] for i in issues):
            status = ValidationStatus.FAILED
        elif issues:
            status = ValidationStatus.WARNING

        return ValidationCheck(
            name=self.name,
            status=status,
            message=f"Found {len(issues)} terminology issues",
            duration_ms=(time.time() - start) * 1000,
            issues=issues,
        )


class CodeReferenceChecker(ValidationRule):
    """Checks that code references in explanation are accurate."""

    @property
    def name(self) -> str:
        return "code_reference_checker"

    @property
    def description(self) -> str:
        return "Verifies code snippets and references are accurate"

    def validate(
        self,
        explanation: ExplanationResult,
        element: CodeElement,
        config: ValidatorConfig,
    ) -> ValidationCheck:
        """Check code references."""
        import time

        start = time.time()
        issues = []
        text = explanation.explanation
        code = element.source_code

        # Extract code snippets from explanation (backtick blocks)
        code_refs = re.findall(r"`([^`]+)`", text)

        for ref in code_refs:
            # Skip common non-code references
            if len(ref) < 2 or ref in ["True", "False", "None"]:
                continue

            # Check if the reference exists in the actual code
            if ref not in code:
                # It might be a modified version, check for similarity
                if not self._is_similar_to_code(ref, code):
                    issues.append(
                        ValidationIssue(
                            type=IssueType.INCORRECT_CLAIM,
                            severity=IssueSeverity.MEDIUM,
                            message=f"Code reference `{ref}` not found in actual code",
                            suggestion="Verify this code snippet is accurate",
                        )
                    )

        # Check quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        for q in quoted:
            if len(q) > 10 and q not in code and not self._is_common_phrase(q):
                # Might be a made-up example
                issues.append(
                    ValidationIssue(
                        type=IssueType.UNSUPPORTED_CLAIM,
                        severity=IssueSeverity.LOW,
                        message=f"Quoted text '{q[:30]}...' not found in code",
                        suggestion="Ensure examples match actual code",
                    )
                )

        status = ValidationStatus.PASSED
        if any(i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH] for i in issues):
            status = ValidationStatus.FAILED
        elif issues:
            status = ValidationStatus.WARNING

        return ValidationCheck(
            name=self.name,
            status=status,
            message=f"Found {len(issues)} code reference issues",
            duration_ms=(time.time() - start) * 1000,
            issues=issues,
        )

    def _is_similar_to_code(self, ref: str, code: str) -> bool:
        """Check if reference is similar to something in the code."""
        # Remove whitespace and compare
        ref_clean = re.sub(r"\s+", "", ref)
        code_clean = re.sub(r"\s+", "", code)
        return ref_clean in code_clean

    def _is_common_phrase(self, phrase: str) -> bool:
        """Check if phrase is a common explanation phrase."""
        common = [
            "for example",
            "in this case",
            "this means",
            "when called",
            "if the",
        ]
        return any(c in phrase.lower() for c in common)


# =============================================================================
# Main Validator Class
# =============================================================================


class ExplanationValidator:
    """Main validator class that orchestrates validation rules."""

    def __init__(
        self,
        config: Optional[ValidatorConfig] = None,
        rules: Optional[List[ValidationRule]] = None,
    ):
        """Initialize validator.

        Args:
            config: Validator configuration.
            rules: List of validation rules to apply.
        """
        self.config = config or ValidatorConfig()
        self.rules: List[ValidationRule] = rules or self._default_rules()

    def _default_rules(self) -> List[ValidationRule]:
        """Get default validation rules based on config."""
        rules = []

        if self.config.check_hallucinations:
            rules.append(HallucinationDetector())

        if self.config.check_completeness:
            rules.append(CompletenessChecker())

        if self.config.check_consistency:
            rules.append(ConsistencyChecker())

        if self.config.check_terminology:
            rules.append(TerminologyChecker())

        rules.append(CodeReferenceChecker())

        return rules

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(i)
                return True
        return False

    def validate(self, explanation: ExplanationResult, element: CodeElement) -> ValidationResult:
        """Validate an explanation.

        Args:
            explanation: The explanation to validate.
            element: The code element being explained.

        Returns:
            ValidationResult with all check results.
        """
        import time

        start = time.time()
        result = ValidationResult(
            is_valid=True,
            overall_status=ValidationStatus.PASSED,
            confidence=1.0,
        )

        passed_count = 0
        failed_count = 0
        warning_count = 0

        for rule in self.rules:
            try:
                check = rule.validate(explanation, element, self.config)
                result.add_check(check)

                if check.status == ValidationStatus.PASSED:
                    passed_count += 1
                elif check.status == ValidationStatus.FAILED:
                    failed_count += 1
                elif check.status == ValidationStatus.WARNING:
                    warning_count += 1

            except Exception as e:
                result.add_check(
                    ValidationCheck(
                        name=rule.name,
                        status=ValidationStatus.SKIPPED,
                        message=f"Rule failed with error: {str(e)}",
                    )
                )

        # Determine overall status
        if failed_count > 0:
            result.overall_status = ValidationStatus.FAILED
            result.is_valid = False
        elif warning_count > 0:
            result.overall_status = ValidationStatus.WARNING
            if self.config.fail_on_warning:
                result.is_valid = False

        # Calculate confidence
        total_checks = passed_count + failed_count + warning_count
        if total_checks > 0:
            result.confidence = passed_count / total_checks

        # Generate summary
        result.summary = self._generate_summary(result)
        result.duration_ms = (time.time() - start) * 1000

        return result

    def validate_batch(
        self, explanations: List[Tuple[ExplanationResult, CodeElement]]
    ) -> List[ValidationResult]:
        """Validate multiple explanations.

        Args:
            explanations: List of (explanation, element) tuples.

        Returns:
            List of ValidationResult objects.
        """
        return [self.validate(exp, elem) for exp, elem in explanations]

    def _generate_summary(self, result: ValidationResult) -> str:
        """Generate a human-readable summary."""
        total = len(result.checks)
        passed = sum(1 for c in result.checks if c.status == ValidationStatus.PASSED)
        failed = sum(1 for c in result.checks if c.status == ValidationStatus.FAILED)
        warnings = sum(1 for c in result.checks if c.status == ValidationStatus.WARNING)

        critical = len(result.get_critical_issues())

        parts = [f"Validation: {passed}/{total} checks passed"]
        if failed:
            parts.append(f"{failed} failed")
        if warnings:
            parts.append(f"{warnings} warnings")
        if critical:
            parts.append(f"{critical} critical issues")

        return ", ".join(parts)


# =============================================================================
# Batch Validator
# =============================================================================


class BatchValidator:
    """Validates multiple explanations with aggregated reporting."""

    def __init__(self, validator: Optional[ExplanationValidator] = None):
        """Initialize batch validator."""
        self.validator = validator or ExplanationValidator()
        self.results: List[ValidationResult] = []

    def validate_all(
        self, explanations: List[Tuple[ExplanationResult, CodeElement]]
    ) -> Dict[str, Any]:
        """Validate all explanations and return aggregate report."""
        self.results = []

        for explanation, element in explanations:
            result = self.validator.validate(explanation, element)
            self.results.append(result)

        return self._generate_report()

    def _generate_report(self) -> Dict[str, Any]:
        """Generate aggregate validation report."""
        total = len(self.results)
        if total == 0:
            return {"total": 0, "valid": 0, "invalid": 0}

        valid = sum(1 for r in self.results if r.is_valid)
        invalid = total - valid

        all_issues = []
        for result in self.results:
            all_issues.extend(result.issues)

        issue_counts = {}
        for issue in all_issues:
            key = issue.type.value
            issue_counts[key] = issue_counts.get(key, 0) + 1

        severity_counts = {}
        for issue in all_issues:
            key = issue.severity.value
            severity_counts[key] = severity_counts.get(key, 0) + 1

        avg_confidence = sum(r.confidence for r in self.results) / total

        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "validation_rate": valid / total,
            "average_confidence": avg_confidence,
            "total_issues": len(all_issues),
            "issues_by_type": issue_counts,
            "issues_by_severity": severity_counts,
        }


# =============================================================================
# Global Instance Management
# =============================================================================


_global_validator: Optional[ExplanationValidator] = None


def get_validator() -> ExplanationValidator:
    """Get the global validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = ExplanationValidator()
    return _global_validator


def reset_validator() -> None:
    """Reset the global validator."""
    global _global_validator
    _global_validator = None


def validate_explanation(explanation: ExplanationResult, element: CodeElement) -> ValidationResult:
    """Convenience function to validate using global validator."""
    return get_validator().validate(explanation, element)


def create_validator(config: Optional[ValidatorConfig] = None) -> ExplanationValidator:
    """Create a new validator with optional config."""
    return ExplanationValidator(config=config)
