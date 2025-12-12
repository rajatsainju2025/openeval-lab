"""Multi-language support for code explanation.

This module provides language detection, multi-language parsing, and
language-specific explanation capabilities for various programming languages.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import CodeElement, CodeElementType


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    SQL = "sql"
    SHELL = "shell"
    HTML = "html"
    CSS = "css"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class LanguageFamily(str, Enum):
    """Language families for grouping similar languages."""

    C_LIKE = "c_like"
    DYNAMIC = "dynamic"
    JVM = "jvm"
    SCRIPTING = "scripting"
    MARKUP = "markup"
    DATA = "data"
    SYSTEMS = "systems"
    FUNCTIONAL = "functional"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LanguageFeatures:
    """Features and characteristics of a programming language."""

    name: str
    language: Language
    family: LanguageFamily
    file_extensions: List[str]
    line_comment: str
    block_comment_start: Optional[str] = None
    block_comment_end: Optional[str] = None
    string_delimiters: List[str] = field(default_factory=lambda: ['"', "'"])
    keywords: List[str] = field(default_factory=list)
    typed: bool = False
    object_oriented: bool = True
    functional: bool = False
    async_support: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Result of language detection."""

    language: Language
    confidence: float  # 0.0 to 1.0
    features_matched: List[str] = field(default_factory=list)
    alternative_languages: List[Tuple[Language, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedElement:
    """A parsed code element with language-specific info."""

    element: CodeElement
    language: Language
    language_specific: Dict[str, Any] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    complexity: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LanguageConfig:
    """Configuration for language-specific processing."""

    enabled_languages: Set[Language] = field(default_factory=lambda: {lang for lang in Language})
    default_language: Language = Language.PYTHON
    detect_from_extension: bool = True
    detect_from_content: bool = True
    confidence_threshold: float = 0.6
    fallback_language: Language = Language.UNKNOWN
    custom_patterns: Dict[Language, List[str]] = field(default_factory=dict)


# =============================================================================
# Language Definitions
# =============================================================================


LANGUAGE_FEATURES: Dict[Language, LanguageFeatures] = {
    Language.PYTHON: LanguageFeatures(
        name="Python",
        language=Language.PYTHON,
        family=LanguageFamily.DYNAMIC,
        file_extensions=[".py", ".pyw", ".pyx"],
        line_comment="#",
        block_comment_start='"""',
        block_comment_end='"""',
        keywords=[
            "def",
            "class",
            "import",
            "from",
            "if",
            "elif",
            "else",
            "for",
            "while",
            "try",
            "except",
            "finally",
            "with",
            "async",
            "await",
            "yield",
            "return",
            "lambda",
            "pass",
            "raise",
            "assert",
        ],
        typed=False,
        object_oriented=True,
        functional=True,
        async_support=True,
    ),
    Language.JAVASCRIPT: LanguageFeatures(
        name="JavaScript",
        language=Language.JAVASCRIPT,
        family=LanguageFamily.DYNAMIC,
        file_extensions=[".js", ".mjs", ".cjs"],
        line_comment="//",
        block_comment_start="/*",
        block_comment_end="*/",
        keywords=[
            "function",
            "const",
            "let",
            "var",
            "class",
            "import",
            "export",
            "if",
            "else",
            "for",
            "while",
            "try",
            "catch",
            "finally",
            "async",
            "await",
            "return",
            "throw",
            "new",
            "this",
        ],
        typed=False,
        object_oriented=True,
        functional=True,
        async_support=True,
    ),
    Language.TYPESCRIPT: LanguageFeatures(
        name="TypeScript",
        language=Language.TYPESCRIPT,
        family=LanguageFamily.DYNAMIC,
        file_extensions=[".ts", ".tsx"],
        line_comment="//",
        block_comment_start="/*",
        block_comment_end="*/",
        keywords=[
            "function",
            "const",
            "let",
            "var",
            "class",
            "import",
            "export",
            "interface",
            "type",
            "enum",
            "if",
            "else",
            "for",
            "while",
            "async",
            "await",
            "return",
            "private",
            "public",
            "protected",
        ],
        typed=True,
        object_oriented=True,
        functional=True,
        async_support=True,
    ),
    Language.JAVA: LanguageFeatures(
        name="Java",
        language=Language.JAVA,
        family=LanguageFamily.JVM,
        file_extensions=[".java"],
        line_comment="//",
        block_comment_start="/*",
        block_comment_end="*/",
        keywords=[
            "public",
            "private",
            "protected",
            "class",
            "interface",
            "extends",
            "implements",
            "static",
            "final",
            "void",
            "return",
            "if",
            "else",
            "for",
            "while",
            "try",
            "catch",
            "throw",
            "throws",
            "new",
            "import",
            "package",
        ],
        typed=True,
        object_oriented=True,
        functional=True,
        async_support=False,
    ),
    Language.CSHARP: LanguageFeatures(
        name="C#",
        language=Language.CSHARP,
        family=LanguageFamily.C_LIKE,
        file_extensions=[".cs"],
        line_comment="//",
        block_comment_start="/*",
        block_comment_end="*/",
        keywords=[
            "public",
            "private",
            "protected",
            "class",
            "interface",
            "struct",
            "namespace",
            "using",
            "static",
            "void",
            "return",
            "if",
            "else",
            "for",
            "foreach",
            "while",
            "async",
            "await",
            "var",
            "new",
        ],
        typed=True,
        object_oriented=True,
        functional=True,
        async_support=True,
    ),
    Language.CPP: LanguageFeatures(
        name="C++",
        language=Language.CPP,
        family=LanguageFamily.C_LIKE,
        file_extensions=[".cpp", ".cc", ".cxx", ".hpp", ".h"],
        line_comment="//",
        block_comment_start="/*",
        block_comment_end="*/",
        keywords=[
            "class",
            "struct",
            "public",
            "private",
            "protected",
            "virtual",
            "override",
            "template",
            "namespace",
            "using",
            "include",
            "define",
            "if",
            "else",
            "for",
            "while",
            "return",
            "new",
            "delete",
            "const",
        ],
        typed=True,
        object_oriented=True,
        functional=True,
        async_support=False,
    ),
    Language.GO: LanguageFeatures(
        name="Go",
        language=Language.GO,
        family=LanguageFamily.SYSTEMS,
        file_extensions=[".go"],
        line_comment="//",
        block_comment_start="/*",
        block_comment_end="*/",
        keywords=[
            "func",
            "package",
            "import",
            "type",
            "struct",
            "interface",
            "if",
            "else",
            "for",
            "range",
            "switch",
            "case",
            "return",
            "defer",
            "go",
            "chan",
            "select",
            "var",
            "const",
        ],
        typed=True,
        object_oriented=False,
        functional=True,
        async_support=True,
    ),
    Language.RUST: LanguageFeatures(
        name="Rust",
        language=Language.RUST,
        family=LanguageFamily.SYSTEMS,
        file_extensions=[".rs"],
        line_comment="//",
        block_comment_start="/*",
        block_comment_end="*/",
        keywords=[
            "fn",
            "let",
            "mut",
            "const",
            "struct",
            "enum",
            "impl",
            "trait",
            "pub",
            "mod",
            "use",
            "if",
            "else",
            "match",
            "for",
            "while",
            "loop",
            "return",
            "async",
            "await",
        ],
        typed=True,
        object_oriented=True,
        functional=True,
        async_support=True,
    ),
    Language.RUBY: LanguageFeatures(
        name="Ruby",
        language=Language.RUBY,
        family=LanguageFamily.DYNAMIC,
        file_extensions=[".rb", ".rake"],
        line_comment="#",
        block_comment_start="=begin",
        block_comment_end="=end",
        keywords=[
            "def",
            "class",
            "module",
            "if",
            "elsif",
            "else",
            "unless",
            "case",
            "when",
            "while",
            "until",
            "for",
            "do",
            "end",
            "return",
            "yield",
            "require",
            "include",
            "attr_accessor",
        ],
        typed=False,
        object_oriented=True,
        functional=True,
        async_support=False,
    ),
    Language.SQL: LanguageFeatures(
        name="SQL",
        language=Language.SQL,
        family=LanguageFamily.DATA,
        file_extensions=[".sql"],
        line_comment="--",
        block_comment_start="/*",
        block_comment_end="*/",
        keywords=[
            "SELECT",
            "FROM",
            "WHERE",
            "JOIN",
            "LEFT",
            "RIGHT",
            "INNER",
            "OUTER",
            "ON",
            "AND",
            "OR",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "ALTER",
            "DROP",
            "TABLE",
            "INDEX",
            "VIEW",
        ],
        typed=True,
        object_oriented=False,
        functional=False,
        async_support=False,
    ),
    Language.SHELL: LanguageFeatures(
        name="Shell",
        language=Language.SHELL,
        family=LanguageFamily.SCRIPTING,
        file_extensions=[".sh", ".bash", ".zsh"],
        line_comment="#",
        keywords=[
            "if",
            "then",
            "else",
            "elif",
            "fi",
            "for",
            "while",
            "do",
            "done",
            "case",
            "esac",
            "function",
            "return",
            "exit",
            "export",
            "source",
            "echo",
            "read",
        ],
        typed=False,
        object_oriented=False,
        functional=False,
        async_support=False,
    ),
}


# Extension to language mapping
EXTENSION_MAP: Dict[str, Language] = {}
for lang, features in LANGUAGE_FEATURES.items():
    for ext in features.file_extensions:
        EXTENSION_MAP[ext] = lang


# =============================================================================
# Language Detector
# =============================================================================


class LanguageDetector:
    """Detects programming language from code or file information."""

    def __init__(self, config: Optional[LanguageConfig] = None):
        """Initialize detector."""
        self.config = config or LanguageConfig()

    def detect(
        self,
        code: str,
        file_path: Optional[str] = None,
        hints: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """Detect language from code and/or file path.

        Args:
            code: Source code to analyze.
            file_path: Optional file path for extension-based detection.
            hints: Optional hints about the language.

        Returns:
            DetectionResult with detected language and confidence.
        """
        candidates: Dict[Language, float] = {}

        # Check file extension first
        if file_path and self.config.detect_from_extension:
            ext_lang = self._detect_from_extension(file_path)
            if ext_lang != Language.UNKNOWN:
                candidates[ext_lang] = candidates.get(ext_lang, 0) + 0.5

        # Analyze code content
        if code and self.config.detect_from_content:
            content_results = self._detect_from_content(code)
            for lang, score in content_results.items():
                candidates[lang] = candidates.get(lang, 0) + score

        # Apply hints
        if hints and "language" in hints:
            hint_lang = self._parse_language_hint(hints["language"])
            if hint_lang:
                candidates[hint_lang] = candidates.get(hint_lang, 0) + 0.3

        # Determine best match
        if not candidates:
            return DetectionResult(
                language=self.config.fallback_language,
                confidence=0.0,
                features_matched=[],
            )

        # Sort by score
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        best_lang, best_score = sorted_candidates[0]

        # Normalize confidence
        max_possible = 1.5  # extension + content + hints
        confidence = min(best_score / max_possible, 1.0)

        return DetectionResult(
            language=best_lang,
            confidence=confidence,
            features_matched=self._get_matched_features(code, best_lang),
            alternative_languages=sorted_candidates[1:4],
        )

    def _detect_from_extension(self, file_path: str) -> Language:
        """Detect language from file extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in EXTENSION_MAP:
            return EXTENSION_MAP[ext]

        # Check for compound extensions
        if len(path.suffixes) > 1:
            compound = "".join(path.suffixes[-2:]).lower()
            if compound in EXTENSION_MAP:
                return EXTENSION_MAP[compound]

        return Language.UNKNOWN

    def _detect_from_content(self, code: str) -> Dict[Language, float]:
        """Detect language from code content."""
        scores: Dict[Language, float] = {}

        for lang, features in LANGUAGE_FEATURES.items():
            if lang not in self.config.enabled_languages:
                continue

            score = self._score_language(code, features)
            if score > 0:
                scores[lang] = score

        return scores

    def _score_language(self, code: str, features: LanguageFeatures) -> float:
        """Calculate a score for how well code matches a language."""
        score = 0.0
        code_lower = code.lower()

        # Check for language-specific patterns
        patterns = self._get_language_patterns(features.language)
        for pattern in patterns:
            if re.search(pattern, code, re.MULTILINE):
                score += 0.2

        # Check for keywords
        keyword_count = 0
        for keyword in features.keywords[:15]:  # Check top 15 keywords
            pattern = rf"\b{re.escape(keyword)}\b"
            matches = len(re.findall(pattern, code_lower))
            if matches > 0:
                keyword_count += 1
                score += min(matches * 0.02, 0.1)

        # Bonus for multiple keywords found
        if keyword_count >= 5:
            score += 0.2
        elif keyword_count >= 3:
            score += 0.1

        # Check comment style
        if features.line_comment in code:
            score += 0.05
        if features.block_comment_start and features.block_comment_start in code:
            score += 0.05

        return min(score, 0.8)  # Cap content score

    def _get_language_patterns(self, lang: Language) -> List[str]:
        """Get regex patterns specific to a language."""
        patterns: Dict[Language, List[str]] = {
            Language.PYTHON: [
                r"^def\s+\w+\s*\(.*\)\s*:",
                r"^class\s+\w+.*:",
                r"^import\s+\w+",
                r"^from\s+\w+\s+import",
                r"if\s+__name__\s*==\s*['\"]__main__['\"]",
            ],
            Language.JAVASCRIPT: [
                r"^(?:const|let|var)\s+\w+\s*=",
                r"^function\s+\w+\s*\(",
                r"=>\s*{",
                r"^export\s+(?:default\s+)?(?:function|class|const)",
                r"require\s*\(['\"]",
            ],
            Language.TYPESCRIPT: [
                r":\s*(?:string|number|boolean|any|void)\b",
                r"^interface\s+\w+",
                r"^type\s+\w+\s*=",
                r"<[A-Z]\w*>",
                r"^export\s+(?:default\s+)?(?:function|class|const|interface|type)",
            ],
            Language.JAVA: [
                r"^public\s+class\s+\w+",
                r"^package\s+[\w.]+;",
                r"public\s+static\s+void\s+main",
                r"@Override",
                r"System\.out\.println",
            ],
            Language.GO: [
                r"^package\s+\w+",
                r"^func\s+\w+\s*\(",
                r"^func\s+\(.*\)\s+\w+",
                r":=",
                r"go\s+\w+\(",
            ],
            Language.RUST: [
                r"^fn\s+\w+\s*\(",
                r"^impl\s+\w+",
                r"^struct\s+\w+",
                r"let\s+mut\s+",
                r"->.*\{",
            ],
            Language.SQL: [
                r"^\s*SELECT\s+.*FROM",
                r"^\s*INSERT\s+INTO",
                r"^\s*UPDATE\s+\w+\s+SET",
                r"^\s*CREATE\s+TABLE",
                r"^\s*ALTER\s+TABLE",
            ],
        }

        return patterns.get(lang, [])

    def _get_matched_features(self, code: str, lang: Language) -> List[str]:
        """Get list of features that matched for detection."""
        matched = []
        features = LANGUAGE_FEATURES.get(lang)
        if not features:
            return matched

        code_lower = code.lower()

        # Check keywords
        for keyword in features.keywords[:10]:
            if re.search(rf"\b{re.escape(keyword)}\b", code_lower):
                matched.append(f"keyword:{keyword}")

        # Check patterns
        for i, pattern in enumerate(self._get_language_patterns(lang)[:5]):
            if re.search(pattern, code, re.MULTILINE):
                matched.append(f"pattern:{i}")

        return matched

    def _parse_language_hint(self, hint: str) -> Optional[Language]:
        """Parse a language hint string."""
        hint_lower = hint.lower().strip()

        # Direct match
        for lang in Language:
            if lang.value == hint_lower or lang.name.lower() == hint_lower:
                return lang

        # Common aliases
        aliases = {
            "py": Language.PYTHON,
            "python3": Language.PYTHON,
            "js": Language.JAVASCRIPT,
            "es6": Language.JAVASCRIPT,
            "ts": Language.TYPESCRIPT,
            "c++": Language.CPP,
            "golang": Language.GO,
            "bash": Language.SHELL,
            "sh": Language.SHELL,
        }

        return aliases.get(hint_lower)


# =============================================================================
# Language-Specific Parser
# =============================================================================


class LanguageParser(ABC):
    """Abstract base class for language-specific parsing."""

    @property
    @abstractmethod
    def language(self) -> Language:
        """Return the language this parser handles."""
        pass

    @abstractmethod
    def parse(self, code: str) -> List[ParsedElement]:
        """Parse code into elements.

        Args:
            code: Source code to parse.

        Returns:
            List of ParsedElement objects.
        """
        pass

    @abstractmethod
    def extract_functions(self, code: str) -> List[CodeElement]:
        """Extract functions/methods from code."""
        pass

    @abstractmethod
    def extract_classes(self, code: str) -> List[CodeElement]:
        """Extract classes/types from code."""
        pass


class PythonParser(LanguageParser):
    """Parser for Python code."""

    @property
    def language(self) -> Language:
        return Language.PYTHON

    def parse(self, code: str) -> List[ParsedElement]:
        """Parse Python code."""
        elements = []
        elements.extend(
            ParsedElement(element=e, language=self.language) for e in self.extract_functions(code)
        )
        elements.extend(
            ParsedElement(element=e, language=self.language) for e in self.extract_classes(code)
        )
        return elements

    def extract_functions(self, code: str) -> List[CodeElement]:
        """Extract functions from Python code."""
        functions = []
        lines = code.split("\n")

        # Pattern for function definitions
        pattern = re.compile(r"^\s*(async\s+)?def\s+(\w+)\s*\((.*?)\)\s*(?:->.*?)?:")

        i = 0
        while i < len(lines):
            match = pattern.match(lines[i])
            if match:
                is_async = match.group(1) is not None
                name = match.group(2)
                params = match.group(3)
                start_line = i + 1

                # Find the end of the function
                base_indent = len(lines[i]) - len(lines[i].lstrip())
                end_line = start_line
                j = i + 1
                while j < len(lines):
                    line = lines[j]
                    if line.strip() and not line.strip().startswith("#"):
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent <= base_indent:
                            break
                    end_line = j + 1
                    j += 1

                # Extract docstring
                docstring = self._extract_docstring(lines[i + 1 : end_line])

                func_code = "\n".join(lines[i:end_line])
                functions.append(
                    CodeElement(
                        type=CodeElementType.FUNCTION,
                        name=name,
                        source_code=func_code,
                        line_start=start_line,
                        line_end=end_line,
                        docstring=docstring,
                        metadata={
                            "async": is_async,
                            "parameters": params,
                        },
                    )
                )
                i = end_line
            else:
                i += 1

        return functions

    def extract_classes(self, code: str) -> List[CodeElement]:
        """Extract classes from Python code."""
        classes = []
        lines = code.split("\n")

        pattern = re.compile(r"^\s*class\s+(\w+)(?:\s*\((.*?)\))?\s*:")

        i = 0
        while i < len(lines):
            match = pattern.match(lines[i])
            if match:
                name = match.group(1)
                bases = match.group(2) or ""
                start_line = i + 1

                # Find end of class
                base_indent = len(lines[i]) - len(lines[i].lstrip())
                end_line = start_line
                j = i + 1
                while j < len(lines):
                    line = lines[j]
                    if line.strip() and not line.strip().startswith("#"):
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent <= base_indent:
                            break
                    end_line = j + 1
                    j += 1

                docstring = self._extract_docstring(lines[i + 1 : end_line])

                class_code = "\n".join(lines[i:end_line])
                classes.append(
                    CodeElement(
                        type=CodeElementType.CLASS,
                        name=name,
                        source_code=class_code,
                        line_start=start_line,
                        line_end=end_line,
                        docstring=docstring,
                        metadata={
                            "bases": bases.split(",") if bases else [],
                        },
                    )
                )
                i = end_line
            else:
                i += 1

        return classes

    def _extract_docstring(self, lines: List[str]) -> Optional[str]:
        """Extract docstring from lines."""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Single line docstring
                if (stripped.count('"""') >= 2) or (stripped.count("'''") >= 2):
                    return stripped.strip("\"'").strip()
                # Multi-line docstring start
                delimiter = '"""' if '"""' in stripped else "'''"
                docstring_lines = [stripped.lstrip(delimiter)]
                for next_line in lines[lines.index(line) + 1 :]:
                    if delimiter in next_line:
                        docstring_lines.append(next_line.split(delimiter)[0])
                        return "\n".join(docstring_lines).strip()
                    docstring_lines.append(next_line.strip())
                return None
            elif stripped and not stripped.startswith("#"):
                return None
        return None


class JavaScriptParser(LanguageParser):
    """Parser for JavaScript/TypeScript code."""

    @property
    def language(self) -> Language:
        return Language.JAVASCRIPT

    def parse(self, code: str) -> List[ParsedElement]:
        """Parse JavaScript code."""
        elements = []
        elements.extend(
            ParsedElement(element=e, language=self.language) for e in self.extract_functions(code)
        )
        elements.extend(
            ParsedElement(element=e, language=self.language) for e in self.extract_classes(code)
        )
        return elements

    def extract_functions(self, code: str) -> List[CodeElement]:
        """Extract functions from JavaScript code."""
        functions = []
        lines = code.split("\n")

        # Patterns for function definitions
        patterns = [
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\((.*?)\)",
            r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\((.*?)\)\s*=>",
            r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function\s*\((.*?)\)",
        ]

        for pattern in patterns:
            regex = re.compile(pattern, re.MULTILINE)
            for match in regex.finditer(code):
                name = match.group(1)
                params = match.group(2)
                start_pos = match.start()

                # Find line number
                start_line = code[:start_pos].count("\n") + 1

                # Find end of function (simplified)
                end_line = self._find_block_end(lines, start_line - 1)

                func_code = "\n".join(lines[start_line - 1 : end_line])
                functions.append(
                    CodeElement(
                        type=CodeElementType.FUNCTION,
                        name=name,
                        source_code=func_code,
                        line_start=start_line,
                        line_end=end_line,
                        metadata={"parameters": params},
                    )
                )

        return functions

    def extract_classes(self, code: str) -> List[CodeElement]:
        """Extract classes from JavaScript code."""
        classes = []
        lines = code.split("\n")

        pattern = re.compile(
            r"^\s*(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{", re.MULTILINE
        )

        for match in pattern.finditer(code):
            name = match.group(1)
            base = match.group(2)
            start_pos = match.start()
            start_line = code[:start_pos].count("\n") + 1

            end_line = self._find_block_end(lines, start_line - 1)

            class_code = "\n".join(lines[start_line - 1 : end_line])
            classes.append(
                CodeElement(
                    type=CodeElementType.CLASS,
                    name=name,
                    source_code=class_code,
                    line_start=start_line,
                    line_end=end_line,
                    metadata={"extends": base},
                )
            )

        return classes

    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a block (matching braces)."""
        brace_count = 0
        started = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == "{":
                    brace_count += 1
                    started = True
                elif char == "}":
                    brace_count -= 1

            if started and brace_count == 0:
                return i + 1

        return len(lines)


# =============================================================================
# Multi-Language Explainer
# =============================================================================


class MultiLanguageExplainer:
    """Explainer that handles multiple programming languages."""

    def __init__(
        self,
        config: Optional[LanguageConfig] = None,
        parsers: Optional[Dict[Language, LanguageParser]] = None,
    ):
        """Initialize multi-language explainer."""
        self.config = config or LanguageConfig()
        self.detector = LanguageDetector(self.config)
        self.parsers: Dict[Language, LanguageParser] = parsers or {
            Language.PYTHON: PythonParser(),
            Language.JAVASCRIPT: JavaScriptParser(),
        }

    def detect_language(self, code: str, file_path: Optional[str] = None) -> DetectionResult:
        """Detect the language of code."""
        return self.detector.detect(code, file_path)

    def parse_code(self, code: str, language: Optional[Language] = None) -> List[ParsedElement]:
        """Parse code into elements.

        Args:
            code: Source code to parse.
            language: Optional language override.

        Returns:
            List of parsed elements.
        """
        if language is None:
            detection = self.detect_language(code)
            language = detection.language

        parser = self.parsers.get(language)
        if parser:
            return parser.parse(code)

        # Fallback: return whole code as module
        return [
            ParsedElement(
                element=CodeElement(
                    type=CodeElementType.MODULE,
                    name="code",
                    source_code=code,
                    line_start=1,
                    line_end=len(code.split("\n")),
                ),
                language=language,
            )
        ]

    def get_language_info(self, language: Language) -> Optional[LanguageFeatures]:
        """Get feature information for a language."""
        return LANGUAGE_FEATURES.get(language)

    def register_parser(self, language: Language, parser: LanguageParser) -> None:
        """Register a custom parser for a language."""
        self.parsers[language] = parser


# =============================================================================
# Global Instance Management
# =============================================================================


_global_detector: Optional[LanguageDetector] = None
_global_explainer: Optional[MultiLanguageExplainer] = None


def get_language_detector() -> LanguageDetector:
    """Get the global language detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = LanguageDetector()
    return _global_detector


def get_multi_language_explainer() -> MultiLanguageExplainer:
    """Get the global multi-language explainer instance."""
    global _global_explainer
    if _global_explainer is None:
        _global_explainer = MultiLanguageExplainer()
    return _global_explainer


def reset_language_detector() -> None:
    """Reset the global language detector."""
    global _global_detector
    _global_detector = None


def reset_multi_language_explainer() -> None:
    """Reset the global multi-language explainer."""
    global _global_explainer
    _global_explainer = None


def detect_language(code: str, file_path: Optional[str] = None) -> DetectionResult:
    """Convenience function to detect language."""
    return get_language_detector().detect(code, file_path)


def get_language_features(language: Language) -> Optional[LanguageFeatures]:
    """Get features for a language."""
    return LANGUAGE_FEATURES.get(language)
