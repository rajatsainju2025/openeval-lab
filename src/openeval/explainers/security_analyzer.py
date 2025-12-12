"""Security-focused code analysis for explanations.

This module provides tools for analyzing code from a security perspective,
identifying vulnerabilities, and generating security-aware explanations.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .types import CodeElement, CodeElementType


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class VulnerabilityType(str, Enum):
    """Types of security vulnerabilities."""

    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    HARDCODED_SECRETS = "hardcoded_secrets"
    WEAK_CRYPTO = "weak_crypto"
    INSECURE_RANDOM = "insecure_random"
    UNSAFE_EVAL = "unsafe_eval"
    UNSAFE_EXEC = "unsafe_exec"
    SSRF = "ssrf"
    XXE = "xxe"
    OPEN_REDIRECT = "open_redirect"
    BUFFER_OVERFLOW = "buffer_overflow"
    RACE_CONDITION = "race_condition"
    INFORMATION_DISCLOSURE = "information_disclosure"
    DENIAL_OF_SERVICE = "denial_of_service"
    BROKEN_AUTH = "broken_auth"
    SENSITIVE_DATA = "sensitive_data"
    INSECURE_CONFIG = "insecure_config"


class Severity(str, Enum):
    """Severity levels for findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Confidence(str, Enum):
    """Confidence levels for findings."""

    CERTAIN = "certain"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RemediationDifficulty(str, Enum):
    """Difficulty of remediation."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    COMPLEX = "complex"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SecurityFinding:
    """A security finding/vulnerability."""

    id: str
    type: VulnerabilityType
    title: str
    description: str
    severity: Severity
    confidence: Confidence
    line_number: Optional[int] = None
    line_end: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    remediation: Optional[str] = None
    remediation_difficulty: RemediationDifficulty = RemediationDifficulty.MODERATE
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def risk_score(self) -> float:
        """Calculate risk score (0-10)."""
        severity_scores = {
            Severity.CRITICAL: 10,
            Severity.HIGH: 8,
            Severity.MEDIUM: 5,
            Severity.LOW: 2,
            Severity.INFO: 0.5,
        }
        confidence_multipliers = {
            Confidence.CERTAIN: 1.0,
            Confidence.HIGH: 0.9,
            Confidence.MEDIUM: 0.7,
            Confidence.LOW: 0.4,
        }
        base = severity_scores.get(self.severity, 5)
        multiplier = confidence_multipliers.get(self.confidence, 0.7)
        return base * multiplier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "line_number": self.line_number,
            "line_end": self.line_end,
            "column": self.column,
            "code_snippet": self.code_snippet,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "remediation": self.remediation,
            "remediation_difficulty": self.remediation_difficulty.value,
            "references": self.references,
            "risk_score": self.risk_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class SecurityReport:
    """A comprehensive security report."""

    element_name: str
    findings: List[SecurityFinding] = field(default_factory=list)
    summary: str = ""
    overall_risk: Severity = Severity.INFO
    total_risk_score: float = 0.0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    recommendations: List[str] = field(default_factory=list)
    analyzed_lines: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "element_name": self.element_name,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "overall_risk": self.overall_risk.value,
            "total_risk_score": self.total_risk_score,
            "counts": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
            "recommendations": self.recommendations,
            "analyzed_lines": self.analyzed_lines,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class SecurityRule:
    """A security detection rule."""

    id: str
    name: str
    type: VulnerabilityType
    severity: Severity
    pattern: Optional[str] = None  # Regex pattern
    ast_pattern: Optional[str] = None  # AST node type
    description: str = ""
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    remediation: str = ""
    references: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class AnalyzerConfig:
    """Configuration for security analyzer."""

    enabled_rules: Optional[Set[str]] = None  # None = all rules
    disabled_rules: Set[str] = field(default_factory=set)
    min_severity: Severity = Severity.INFO
    min_confidence: Confidence = Confidence.LOW
    include_info: bool = True
    max_findings: int = 100
    analyze_imports: bool = True
    analyze_strings: bool = True
    analyze_calls: bool = True
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Security Detectors
# =============================================================================


class SecurityDetector(ABC):
    """Abstract base class for security detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get detector name."""
        pass

    @property
    @abstractmethod
    def vulnerability_type(self) -> VulnerabilityType:
        """Get the type of vulnerability this detector finds."""
        pass

    @abstractmethod
    def detect(
        self, code: str, element: CodeElement, config: AnalyzerConfig
    ) -> List[SecurityFinding]:
        """Detect vulnerabilities in code.

        Args:
            code: Source code to analyze.
            element: Code element being analyzed.
            config: Analyzer configuration.

        Returns:
            List of SecurityFinding objects.
        """
        pass


class SQLInjectionDetector(SecurityDetector):
    """Detects potential SQL injection vulnerabilities."""

    @property
    def name(self) -> str:
        return "SQL Injection Detector"

    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.SQL_INJECTION

    def detect(
        self, code: str, element: CodeElement, config: AnalyzerConfig
    ) -> List[SecurityFinding]:
        """Detect SQL injection patterns."""
        findings = []
        lines = code.split("\n")

        # Patterns for SQL injection
        patterns = [
            (r'execute\s*\(\s*["\'].*%s', "String formatting in SQL execute"),
            (r'execute\s*\(\s*["\'].*\+', "String concatenation in SQL execute"),
            (r'execute\s*\(\s*f["\']', "F-string in SQL execute"),
            (r"cursor\.execute\s*\(\s*[^,]+\+", "String concatenation in cursor.execute"),
            (r"\.format\s*\([^)]*\)\s*\)", "String format in SQL query"),
            (r'raw\s*\(\s*["\'].*%', "Raw SQL with string formatting"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, description in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(
                        SecurityFinding(
                            id=f"sqli_{i}_{len(findings)}",
                            type=VulnerabilityType.SQL_INJECTION,
                            title="Potential SQL Injection",
                            description=f"{description}. User input may be directly interpolated into SQL query.",
                            severity=Severity.HIGH,
                            confidence=Confidence.MEDIUM,
                            line_number=i,
                            code_snippet=line.strip(),
                            cwe_id="CWE-89",
                            owasp_category="A03:2021 - Injection",
                            remediation="Use parameterized queries or prepared statements instead of string formatting.",
                        )
                    )

        return findings


class CommandInjectionDetector(SecurityDetector):
    """Detects potential command injection vulnerabilities."""

    @property
    def name(self) -> str:
        return "Command Injection Detector"

    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.COMMAND_INJECTION

    def detect(
        self, code: str, element: CodeElement, config: AnalyzerConfig
    ) -> List[SecurityFinding]:
        """Detect command injection patterns."""
        findings = []
        lines = code.split("\n")

        # Dangerous functions
        dangerous_funcs = [
            (r"os\.system\s*\(", "os.system", Severity.HIGH),
            (r"os\.popen\s*\(", "os.popen", Severity.HIGH),
            (
                r"subprocess\.call\s*\([^)]*shell\s*=\s*True",
                "subprocess with shell=True",
                Severity.HIGH,
            ),
            (
                r"subprocess\.run\s*\([^)]*shell\s*=\s*True",
                "subprocess.run with shell=True",
                Severity.HIGH,
            ),
            (
                r"subprocess\.Popen\s*\([^)]*shell\s*=\s*True",
                "Popen with shell=True",
                Severity.HIGH,
            ),
            (r"commands\.getoutput\s*\(", "commands.getoutput", Severity.HIGH),
            (r"commands\.getstatusoutput\s*\(", "commands.getstatusoutput", Severity.HIGH),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, func_name, severity in dangerous_funcs:
                if re.search(pattern, line):
                    findings.append(
                        SecurityFinding(
                            id=f"cmdi_{i}_{len(findings)}",
                            type=VulnerabilityType.COMMAND_INJECTION,
                            title=f"Dangerous Function: {func_name}",
                            description=f"Use of {func_name} can lead to command injection if user input is included.",
                            severity=severity,
                            confidence=Confidence.MEDIUM,
                            line_number=i,
                            code_snippet=line.strip(),
                            cwe_id="CWE-78",
                            owasp_category="A03:2021 - Injection",
                            remediation="Use subprocess with shell=False and pass arguments as a list.",
                        )
                    )

        return findings


class HardcodedSecretsDetector(SecurityDetector):
    """Detects hardcoded secrets and credentials."""

    @property
    def name(self) -> str:
        return "Hardcoded Secrets Detector"

    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.HARDCODED_SECRETS

    def detect(
        self, code: str, element: CodeElement, config: AnalyzerConfig
    ) -> List[SecurityFinding]:
        """Detect hardcoded secrets."""
        findings = []
        lines = code.split("\n")

        # Secret patterns
        patterns = [
            (
                r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']+["\']',
                "Hardcoded password",
                Severity.HIGH,
            ),
            (
                r'(?:api_key|apikey|api_token)\s*=\s*["\'][^"\']+["\']',
                "Hardcoded API key",
                Severity.HIGH,
            ),
            (r'(?:secret|secret_key)\s*=\s*["\'][^"\']+["\']', "Hardcoded secret", Severity.HIGH),
            (
                r'(?:aws_access_key|aws_secret)\s*=\s*["\'][^"\']+["\']',
                "Hardcoded AWS credentials",
                Severity.CRITICAL,
            ),
            (
                r'(?:private_key|priv_key)\s*=\s*["\'][^"\']+["\']',
                "Hardcoded private key",
                Severity.CRITICAL,
            ),
            (
                r'(?:token|auth_token)\s*=\s*["\'][A-Za-z0-9_-]{20,}["\']',
                "Hardcoded token",
                Severity.HIGH,
            ),
            (r"Bearer\s+[A-Za-z0-9_-]{20,}", "Hardcoded bearer token", Severity.HIGH),
            (r"Basic\s+[A-Za-z0-9+/=]{20,}", "Hardcoded basic auth", Severity.HIGH),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, description, severity in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(
                        SecurityFinding(
                            id=f"secret_{i}_{len(findings)}",
                            type=VulnerabilityType.HARDCODED_SECRETS,
                            title=description,
                            description=f"{description} detected. Credentials should not be hardcoded in source code.",
                            severity=severity,
                            confidence=Confidence.HIGH,
                            line_number=i,
                            code_snippet=self._redact_secret(line.strip()),
                            cwe_id="CWE-798",
                            owasp_category="A07:2021 - Identification and Authentication Failures",
                            remediation="Use environment variables, secrets manager, or configuration files (not in version control).",
                        )
                    )

        return findings

    def _redact_secret(self, line: str) -> str:
        """Redact secrets in code snippet."""
        # Replace quoted strings that look like secrets
        return re.sub(r'(["\'])[^"\']{8,}(["\'])', r"\1****REDACTED****\2", line)


class UnsafeEvalDetector(SecurityDetector):
    """Detects unsafe use of eval and exec."""

    @property
    def name(self) -> str:
        return "Unsafe Eval/Exec Detector"

    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.UNSAFE_EVAL

    def detect(
        self, code: str, element: CodeElement, config: AnalyzerConfig
    ) -> List[SecurityFinding]:
        """Detect unsafe eval/exec usage."""
        findings = []
        lines = code.split("\n")

        patterns = [
            (r"\beval\s*\(", "eval()", VulnerabilityType.UNSAFE_EVAL, Severity.CRITICAL),
            (r"\bexec\s*\(", "exec()", VulnerabilityType.UNSAFE_EXEC, Severity.CRITICAL),
            (r"\bcompile\s*\(", "compile()", VulnerabilityType.UNSAFE_EVAL, Severity.HIGH),
            (r"__import__\s*\(", "__import__()", VulnerabilityType.UNSAFE_EVAL, Severity.MEDIUM),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, func_name, vuln_type, severity in patterns:
                if re.search(pattern, line):
                    findings.append(
                        SecurityFinding(
                            id=f"eval_{i}_{len(findings)}",
                            type=vuln_type,
                            title=f"Dangerous Function: {func_name}",
                            description=f"Use of {func_name} can execute arbitrary code if user input is passed.",
                            severity=severity,
                            confidence=Confidence.HIGH,
                            line_number=i,
                            code_snippet=line.strip(),
                            cwe_id="CWE-95",
                            owasp_category="A03:2021 - Injection",
                            remediation=f"Avoid using {func_name}. Use safer alternatives like ast.literal_eval for data parsing.",
                        )
                    )

        return findings


class PathTraversalDetector(SecurityDetector):
    """Detects potential path traversal vulnerabilities."""

    @property
    def name(self) -> str:
        return "Path Traversal Detector"

    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.PATH_TRAVERSAL

    def detect(
        self, code: str, element: CodeElement, config: AnalyzerConfig
    ) -> List[SecurityFinding]:
        """Detect path traversal patterns."""
        findings = []
        lines = code.split("\n")

        patterns = [
            (r"open\s*\([^)]*\+[^)]*\)", "File open with concatenation"),
            (r"open\s*\(.*format\s*\(", "File open with format string"),
            (r"open\s*\(\s*f['\"]", "File open with f-string"),
            (r"os\.path\.join\s*\([^)]*user", "os.path.join with user input"),
            (r"Path\s*\([^)]*\+", "pathlib Path with concatenation"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, description in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(
                        SecurityFinding(
                            id=f"path_{i}_{len(findings)}",
                            type=VulnerabilityType.PATH_TRAVERSAL,
                            title="Potential Path Traversal",
                            description=f"{description}. User input in file paths may allow access to unauthorized files.",
                            severity=Severity.HIGH,
                            confidence=Confidence.MEDIUM,
                            line_number=i,
                            code_snippet=line.strip(),
                            cwe_id="CWE-22",
                            owasp_category="A01:2021 - Broken Access Control",
                            remediation="Validate and sanitize file paths. Use os.path.basename() to strip directory components.",
                        )
                    )

        return findings


class WeakCryptoDetector(SecurityDetector):
    """Detects use of weak cryptographic algorithms."""

    @property
    def name(self) -> str:
        return "Weak Cryptography Detector"

    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.WEAK_CRYPTO

    def detect(
        self, code: str, element: CodeElement, config: AnalyzerConfig
    ) -> List[SecurityFinding]:
        """Detect weak cryptography usage."""
        findings = []
        lines = code.split("\n")

        weak_patterns = [
            (r"\bmd5\b", "MD5", "MD5 is cryptographically broken", Severity.MEDIUM),
            (r"\bsha1\b", "SHA1", "SHA1 is considered weak", Severity.MEDIUM),
            (r"\bDES\b", "DES", "DES is considered weak", Severity.HIGH),
            (r"\bRC4\b", "RC4", "RC4 is considered weak", Severity.HIGH),
            (r"ECB\s*mode", "ECB mode", "ECB mode is insecure for most uses", Severity.MEDIUM),
            (r"random\.random\s*\(", "random.random", "Not cryptographically secure", Severity.LOW),
            (
                r"random\.randint\s*\(",
                "random.randint",
                "Not cryptographically secure",
                Severity.LOW,
            ),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, algo, description, severity in weak_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(
                        SecurityFinding(
                            id=f"crypto_{i}_{len(findings)}",
                            type=VulnerabilityType.WEAK_CRYPTO,
                            title=f"Weak Cryptography: {algo}",
                            description=description,
                            severity=severity,
                            confidence=Confidence.HIGH,
                            line_number=i,
                            code_snippet=line.strip(),
                            cwe_id="CWE-327",
                            owasp_category="A02:2021 - Cryptographic Failures",
                            remediation="Use strong algorithms: SHA-256+, AES-256, use secrets module for random values.",
                        )
                    )

        return findings


class InsecureDeserializationDetector(SecurityDetector):
    """Detects insecure deserialization patterns."""

    @property
    def name(self) -> str:
        return "Insecure Deserialization Detector"

    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.INSECURE_DESERIALIZATION

    def detect(
        self, code: str, element: CodeElement, config: AnalyzerConfig
    ) -> List[SecurityFinding]:
        """Detect insecure deserialization."""
        findings = []
        lines = code.split("\n")

        patterns = [
            (r"pickle\.load\s*\(", "pickle.load", Severity.HIGH),
            (r"pickle\.loads\s*\(", "pickle.loads", Severity.HIGH),
            (r"cPickle\.load", "cPickle.load", Severity.HIGH),
            (r"marshal\.load", "marshal.load", Severity.HIGH),
            (r"yaml\.load\s*\([^)]*\)", "yaml.load without safe_load", Severity.HIGH),
            (r"yaml\.unsafe_load", "yaml.unsafe_load", Severity.CRITICAL),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, func_name, severity in patterns:
                if re.search(pattern, line):
                    # Check for safe alternatives
                    if "yaml.load" in line and "Loader=yaml.SafeLoader" in line:
                        continue

                    findings.append(
                        SecurityFinding(
                            id=f"deser_{i}_{len(findings)}",
                            type=VulnerabilityType.INSECURE_DESERIALIZATION,
                            title=f"Insecure Deserialization: {func_name}",
                            description=f"{func_name} can execute arbitrary code when deserializing untrusted data.",
                            severity=severity,
                            confidence=Confidence.HIGH,
                            line_number=i,
                            code_snippet=line.strip(),
                            cwe_id="CWE-502",
                            owasp_category="A08:2021 - Software and Data Integrity Failures",
                            remediation="Use safe alternatives like json.load, yaml.safe_load, or verify data source.",
                        )
                    )

        return findings


class XSSDetector(SecurityDetector):
    """Detects potential XSS vulnerabilities."""

    @property
    def name(self) -> str:
        return "XSS Detector"

    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.XSS

    def detect(
        self, code: str, element: CodeElement, config: AnalyzerConfig
    ) -> List[SecurityFinding]:
        """Detect XSS patterns."""
        findings = []
        lines = code.split("\n")

        patterns = [
            (r"innerHTML\s*=", "innerHTML assignment", Severity.HIGH),
            (r"\.html\s*\(", "jQuery .html() method", Severity.HIGH),
            (r"document\.write\s*\(", "document.write", Severity.HIGH),
            (r"outerHTML\s*=", "outerHTML assignment", Severity.HIGH),
            (r"\|safe\b", "Django safe filter", Severity.MEDIUM),
            (r"mark_safe\s*\(", "Django mark_safe", Severity.MEDIUM),
            (r"Markup\s*\(", "Flask Markup", Severity.MEDIUM),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, description, severity in patterns:
                if re.search(pattern, line):
                    findings.append(
                        SecurityFinding(
                            id=f"xss_{i}_{len(findings)}",
                            type=VulnerabilityType.XSS,
                            title=f"Potential XSS: {description}",
                            description=f"Use of {description} may lead to XSS if user input is included.",
                            severity=severity,
                            confidence=Confidence.MEDIUM,
                            line_number=i,
                            code_snippet=line.strip(),
                            cwe_id="CWE-79",
                            owasp_category="A03:2021 - Injection",
                            remediation="Escape user input before rendering. Use textContent instead of innerHTML.",
                        )
                    )

        return findings


# =============================================================================
# Security Analyzer
# =============================================================================


class SecurityAnalyzer:
    """Analyzes code for security vulnerabilities."""

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """Initialize security analyzer.

        Args:
            config: Optional analyzer configuration.
        """
        self.config = config or AnalyzerConfig()
        self._detectors: List[SecurityDetector] = [
            SQLInjectionDetector(),
            CommandInjectionDetector(),
            HardcodedSecretsDetector(),
            UnsafeEvalDetector(),
            PathTraversalDetector(),
            WeakCryptoDetector(),
            InsecureDeserializationDetector(),
            XSSDetector(),
        ]

    def analyze(self, element: CodeElement) -> SecurityReport:
        """Analyze a code element for security issues.

        Args:
            element: Code element to analyze.

        Returns:
            SecurityReport with findings.
        """
        all_findings = []

        for detector in self._detectors:
            if self._is_detector_enabled(detector):
                findings = detector.detect(element.source_code, element, self.config)
                all_findings.extend(findings)

        # Filter by severity and confidence
        filtered_findings = [f for f in all_findings if self._meets_threshold(f)]

        # Limit findings
        filtered_findings = filtered_findings[: self.config.max_findings]

        # Generate report
        return self._create_report(element, filtered_findings)

    def analyze_code(self, code: str, name: str = "code") -> SecurityReport:
        """Analyze raw code string.

        Args:
            code: Source code to analyze.
            name: Name for the code element.

        Returns:
            SecurityReport with findings.
        """
        element = CodeElement(
            type=CodeElementType.MODULE,
            name=name,
            source_code=code,
            line_start=1,
            line_end=len(code.split("\n")),
        )
        return self.analyze(element)

    def add_detector(self, detector: SecurityDetector) -> None:
        """Add a custom detector.

        Args:
            detector: Detector to add.
        """
        self._detectors.append(detector)

    def _is_detector_enabled(self, detector: SecurityDetector) -> bool:
        """Check if a detector is enabled."""
        if self.config.enabled_rules is not None:
            if detector.name not in self.config.enabled_rules:
                return False
        if detector.name in self.config.disabled_rules:
            return False
        return True

    def _meets_threshold(self, finding: SecurityFinding) -> bool:
        """Check if finding meets severity/confidence thresholds."""
        severity_order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        confidence_order = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH, Confidence.CERTAIN]

        min_sev_idx = severity_order.index(self.config.min_severity)
        min_conf_idx = confidence_order.index(self.config.min_confidence)

        finding_sev_idx = severity_order.index(finding.severity)
        finding_conf_idx = confidence_order.index(finding.confidence)

        return finding_sev_idx >= min_sev_idx and finding_conf_idx >= min_conf_idx

    def _create_report(
        self, element: CodeElement, findings: List[SecurityFinding]
    ) -> SecurityReport:
        """Create a security report from findings."""
        # Count by severity
        critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        high = sum(1 for f in findings if f.severity == Severity.HIGH)
        medium = sum(1 for f in findings if f.severity == Severity.MEDIUM)
        low = sum(1 for f in findings if f.severity == Severity.LOW)

        # Determine overall risk
        if critical > 0:
            overall = Severity.CRITICAL
        elif high > 0:
            overall = Severity.HIGH
        elif medium > 0:
            overall = Severity.MEDIUM
        elif low > 0:
            overall = Severity.LOW
        else:
            overall = Severity.INFO

        # Calculate total risk score
        total_risk = sum(f.risk_score for f in findings)

        # Generate recommendations
        recommendations = self._generate_recommendations(findings)

        # Generate summary
        summary = self._generate_summary(element.name, findings, overall)

        return SecurityReport(
            element_name=element.name,
            findings=findings,
            summary=summary,
            overall_risk=overall,
            total_risk_score=total_risk,
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            recommendations=recommendations,
            analyzed_lines=len(element.source_code.split("\n")),
        )

    def _generate_summary(
        self, name: str, findings: List[SecurityFinding], overall: Severity
    ) -> str:
        """Generate a summary of findings."""
        if not findings:
            return f"No security issues found in {name}."

        vuln_types = set(f.type.value for f in findings)
        return (
            f"Found {len(findings)} security issue(s) in {name}. "
            f"Overall risk: {overall.value}. "
            f"Vulnerability types: {', '.join(vuln_types)}."
        )

    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []

        # Group by vulnerability type
        by_type: Dict[VulnerabilityType, List[SecurityFinding]] = {}
        for f in findings:
            if f.type not in by_type:
                by_type[f.type] = []
            by_type[f.type].append(f)

        # Priority recommendations
        if VulnerabilityType.HARDCODED_SECRETS in by_type:
            recommendations.append(
                "URGENT: Remove hardcoded secrets and use environment variables or a secrets manager."
            )

        if VulnerabilityType.SQL_INJECTION in by_type:
            recommendations.append(
                "Use parameterized queries for all database operations to prevent SQL injection."
            )

        if VulnerabilityType.COMMAND_INJECTION in by_type:
            recommendations.append(
                "Avoid shell=True in subprocess calls. Use list arguments instead."
            )

        if VulnerabilityType.UNSAFE_EVAL in by_type or VulnerabilityType.UNSAFE_EXEC in by_type:
            recommendations.append("Remove all uses of eval() and exec(). Use safer alternatives.")

        if VulnerabilityType.INSECURE_DESERIALIZATION in by_type:
            recommendations.append("Replace pickle with JSON or other safe serialization formats.")

        if not recommendations:
            recommendations.append("Review all findings and address based on severity.")

        return recommendations


# =============================================================================
# Security-Aware Explanation Generator
# =============================================================================


class SecurityExplainer:
    """Generates security-focused explanations."""

    def __init__(self, analyzer: Optional[SecurityAnalyzer] = None):
        """Initialize security explainer.

        Args:
            analyzer: Optional security analyzer.
        """
        self.analyzer = analyzer or SecurityAnalyzer()

    def explain_security(self, element: CodeElement) -> str:
        """Generate a security-focused explanation.

        Args:
            element: Code element to explain.

        Returns:
            Security-focused explanation text.
        """
        report = self.analyzer.analyze(element)

        sections = [f"## Security Analysis: {element.name}\n"]

        # Summary
        sections.append(f"### Summary\n{report.summary}\n")

        # Findings
        if report.findings:
            sections.append("### Security Issues Found\n")
            for i, finding in enumerate(report.findings, 1):
                sections.append(
                    f"{i}. **{finding.title}** ({finding.severity.value})\n"
                    f"   - Line: {finding.line_number or 'N/A'}\n"
                    f"   - {finding.description}\n"
                    f"   - Remediation: {finding.remediation}\n"
                )
        else:
            sections.append("### ✓ No security issues detected\n")

        # Recommendations
        if report.recommendations:
            sections.append("### Recommendations\n")
            for rec in report.recommendations:
                sections.append(f"- {rec}\n")

        return "\n".join(sections)


# =============================================================================
# Global Instance Management
# =============================================================================


_global_analyzer: Optional[SecurityAnalyzer] = None


def get_security_analyzer() -> SecurityAnalyzer:
    """Get the global security analyzer instance."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = SecurityAnalyzer()
    return _global_analyzer


def reset_security_analyzer() -> None:
    """Reset the global security analyzer."""
    global _global_analyzer
    _global_analyzer = None


def analyze_security(element: CodeElement) -> SecurityReport:
    """Convenience function to analyze security of a code element."""
    return get_security_analyzer().analyze(element)


def analyze_code_security(code: str, name: str = "code") -> SecurityReport:
    """Convenience function to analyze security of code string."""
    return get_security_analyzer().analyze_code(code, name)


def create_security_analyzer(
    config: Optional[AnalyzerConfig] = None,
) -> SecurityAnalyzer:
    """Create a new security analyzer."""
    return SecurityAnalyzer(config=config)
