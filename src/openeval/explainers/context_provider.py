"""Context providers for enriching code explanations.

This module provides various context providers that gather relevant information
to enrich code explanations with additional context from files, projects, git
history, and other sources.
"""

import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import CodeElement


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class ContextType(str, Enum):
    """Types of context that can be provided."""

    FILE = "file"
    PROJECT = "project"
    GIT = "git"
    DOCUMENTATION = "documentation"
    DEPENDENCY = "dependency"
    TEST = "test"
    USAGE = "usage"
    SEMANTIC = "semantic"


class ContextPriority(str, Enum):
    """Priority levels for context information."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class ContextScope(str, Enum):
    """Scope of context information."""

    LOCAL = "local"  # Same file
    MODULE = "module"  # Same module/package
    PROJECT = "project"  # Whole project
    EXTERNAL = "external"  # External dependencies


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ContextItem:
    """A single piece of context information."""

    type: ContextType
    key: str
    value: Any
    priority: ContextPriority = ContextPriority.MEDIUM
    scope: ContextScope = ContextScope.LOCAL
    source: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "key": self.key,
            "value": self.value,
            "priority": self.priority.value,
            "scope": self.scope.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ContextResult:
    """Result of context gathering."""

    items: List[ContextItem] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, item: ContextItem) -> None:
        """Add a context item."""
        self.items.append(item)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def get_by_type(self, context_type: ContextType) -> List[ContextItem]:
        """Get items by type."""
        return [item for item in self.items if item.type == context_type]

    def get_by_priority(self, priority: ContextPriority) -> List[ContextItem]:
        """Get items by priority."""
        return [item for item in self.items if item.priority == priority]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "items": [item.to_dict() for item in self.items],
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class FileContext:
    """Context from a single file."""

    path: str
    name: str
    extension: str
    size_bytes: int
    lines: int
    imports: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    last_modified: Optional[str] = None


@dataclass
class GitContext:
    """Context from git history."""

    commits: List[Dict[str, Any]] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    blame_info: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    branch: Optional[str] = None
    remote_url: Optional[str] = None
    last_commit_hash: Optional[str] = None
    file_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProjectContext:
    """Context from project structure."""

    name: str
    root_path: str
    language: str
    framework: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)


@dataclass
class ProviderConfig:
    """Configuration for context providers."""

    enabled: bool = True
    max_depth: int = 3
    max_items: int = 100
    timeout_ms: float = 5000.0
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(
        default_factory=lambda: ["__pycache__", ".git", "node_modules", ".venv", "venv"]
    )
    priority_threshold: ContextPriority = ContextPriority.LOW
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Abstract Base Class
# =============================================================================


class ContextProvider(ABC):
    """Abstract base class for context providers."""

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize provider with optional config."""
        self.config = config or ProviderConfig()

    @property
    @abstractmethod
    def context_type(self) -> ContextType:
        """Return the type of context this provider supplies."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this provider."""
        pass

    @abstractmethod
    def gather(self, element: CodeElement, file_path: Optional[str] = None) -> ContextResult:
        """Gather context for a code element.

        Args:
            element: The code element to gather context for.
            file_path: Optional path to the file containing the element.

        Returns:
            ContextResult with gathered context items.
        """
        pass

    def is_enabled(self) -> bool:
        """Check if provider is enabled."""
        return self.config.enabled

    def should_include(self, path: str) -> bool:
        """Check if path should be included based on config patterns."""
        for pattern in self.config.exclude_patterns:
            if pattern in path:
                return False
        if not self.config.include_patterns:
            return True
        return any(pattern in path for pattern in self.config.include_patterns)


# =============================================================================
# File Context Provider
# =============================================================================


class FileContextProvider(ContextProvider):
    """Provides context from file structure and contents."""

    @property
    def context_type(self) -> ContextType:
        return ContextType.FILE

    @property
    def name(self) -> str:
        return "file_context"

    def gather(self, element: CodeElement, file_path: Optional[str] = None) -> ContextResult:
        """Gather file-level context."""
        import time

        start = time.time()
        result = ContextResult()

        if not file_path:
            result.add_error("No file path provided")
            return result

        path = Path(file_path)
        if not path.exists():
            result.add_error(f"File not found: {file_path}")
            return result

        try:
            # Basic file info
            stat = path.stat()
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")

            file_ctx = FileContext(
                path=str(path.absolute()),
                name=path.name,
                extension=path.suffix,
                size_bytes=stat.st_size,
                lines=len(lines),
                last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            )

            # Extract imports (Python-specific)
            if path.suffix == ".py":
                file_ctx.imports = self._extract_python_imports(content)
                file_ctx.functions = self._extract_python_functions(content)
                file_ctx.classes = self._extract_python_classes(content)
                file_ctx.docstring = self._extract_module_docstring(content)

            result.add(
                ContextItem(
                    type=ContextType.FILE,
                    key="file_info",
                    value=file_ctx.__dict__,
                    priority=ContextPriority.HIGH,
                    scope=ContextScope.LOCAL,
                    source=file_path,
                )
            )

            # Find related files (same directory, similar names)
            related = self._find_related_files(path)
            if related:
                result.add(
                    ContextItem(
                        type=ContextType.FILE,
                        key="related_files",
                        value=related,
                        priority=ContextPriority.MEDIUM,
                        scope=ContextScope.MODULE,
                        source=str(path.parent),
                    )
                )

            # Extract surrounding code context
            if element.line_start > 0:
                context_lines = self._get_surrounding_context(
                    lines, element.line_start, element.line_end
                )
                result.add(
                    ContextItem(
                        type=ContextType.FILE,
                        key="surrounding_code",
                        value=context_lines,
                        priority=ContextPriority.HIGH,
                        scope=ContextScope.LOCAL,
                        source=file_path,
                        metadata={"line_start": element.line_start},
                    )
                )

        except Exception as e:
            result.add_error(f"Error gathering file context: {str(e)}")

        result.duration_ms = (time.time() - start) * 1000
        return result

    def _extract_python_imports(self, content: str) -> List[str]:
        """Extract import statements from Python code."""
        imports = []
        import_pattern = re.compile(r"^(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s]+)", re.MULTILINE)
        for match in import_pattern.finditer(content):
            if match.group(1):
                imports.append(f"from {match.group(1)} import {match.group(2).strip()}")
            else:
                imports.append(f"import {match.group(2).strip()}")
        return imports

    def _extract_python_functions(self, content: str) -> List[str]:
        """Extract function names from Python code."""
        pattern = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.MULTILINE)
        return [match.group(1) for match in pattern.finditer(content)]

    def _extract_python_classes(self, content: str) -> List[str]:
        """Extract class names from Python code."""
        pattern = re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)
        return [match.group(1) for match in pattern.finditer(content)]

    def _extract_module_docstring(self, content: str) -> Optional[str]:
        """Extract module docstring from Python code."""
        pattern = re.compile(r'^(?:"""|\'\'\')(.*?)(?:"""|\'\'\'))', re.DOTALL)
        match = pattern.match(content.lstrip())
        return match.group(1).strip() if match else None

    def _find_related_files(self, path: Path) -> List[str]:
        """Find files related to the given file."""
        related = []
        parent = path.parent
        stem = path.stem

        # Look for test file
        test_patterns = [f"test_{stem}.py", f"{stem}_test.py", f"tests/test_{stem}.py"]
        for pattern in test_patterns:
            test_path = parent / pattern
            if test_path.exists():
                related.append(str(test_path))

        # Look for __init__.py
        init_path = parent / "__init__.py"
        if init_path.exists() and init_path != path:
            related.append(str(init_path))

        # Look for files with similar names
        for sibling in parent.iterdir():
            if sibling.is_file() and sibling != path:
                if sibling.stem.startswith(stem) or stem.startswith(sibling.stem):
                    related.append(str(sibling))
                    if len(related) >= 5:
                        break

        return related[:10]

    def _get_surrounding_context(
        self, lines: List[str], start: int, end: int, context_lines: int = 10
    ) -> Dict[str, Any]:
        """Get surrounding code context."""
        before_start = max(0, start - context_lines - 1)
        after_end = min(len(lines), end + context_lines)

        return {
            "before": "\n".join(lines[before_start : start - 1]),
            "after": "\n".join(lines[end:after_end]),
            "before_line_start": before_start + 1,
            "after_line_end": after_end,
        }


# =============================================================================
# Git Context Provider
# =============================================================================


class GitContextProvider(ContextProvider):
    """Provides context from git history."""

    @property
    def context_type(self) -> ContextType:
        return ContextType.GIT

    @property
    def name(self) -> str:
        return "git_context"

    def gather(self, element: CodeElement, file_path: Optional[str] = None) -> ContextResult:
        """Gather git-related context."""
        import time

        start = time.time()
        result = ContextResult()

        if not file_path:
            result.add_error("No file path provided for git context")
            return result

        path = Path(file_path)
        if not path.exists():
            result.add_error(f"File not found: {file_path}")
            return result

        try:
            git_root = self._find_git_root(path)
            if not git_root:
                result.add_error("Not in a git repository")
                return result

            git_ctx = GitContext()

            # Get current branch
            git_ctx.branch = self._run_git_command(git_root, ["rev-parse", "--abbrev-ref", "HEAD"])

            # Get remote URL
            git_ctx.remote_url = self._run_git_command(
                git_root, ["config", "--get", "remote.origin.url"]
            )

            # Get recent commits for this file
            commits = self._get_file_commits(git_root, path, limit=5)
            git_ctx.commits = commits

            # Get unique authors
            git_ctx.authors = list({c.get("author", "") for c in commits if c})

            # Get last commit hash
            if commits:
                git_ctx.last_commit_hash = commits[0].get("hash")

            # Get blame info for the element's lines
            if element.line_start > 0:
                blame = self._get_blame_info(git_root, path, element.line_start, element.line_end)
                git_ctx.blame_info = blame

            result.add(
                ContextItem(
                    type=ContextType.GIT,
                    key="git_info",
                    value=git_ctx.__dict__,
                    priority=ContextPriority.MEDIUM,
                    scope=ContextScope.PROJECT,
                    source=str(git_root),
                )
            )

            # Add individual high-priority items
            if git_ctx.commits:
                result.add(
                    ContextItem(
                        type=ContextType.GIT,
                        key="recent_commits",
                        value=git_ctx.commits,
                        priority=ContextPriority.HIGH,
                        scope=ContextScope.LOCAL,
                        source=file_path,
                    )
                )

        except Exception as e:
            result.add_error(f"Error gathering git context: {str(e)}")

        result.duration_ms = (time.time() - start) * 1000
        return result

    def _find_git_root(self, path: Path) -> Optional[Path]:
        """Find the root of the git repository."""
        current = path if path.is_dir() else path.parent
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        return None

    def _run_git_command(
        self, git_root: Path, args: List[str], timeout: float = 5.0
    ) -> Optional[str]:
        """Run a git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=git_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _get_file_commits(
        self, git_root: Path, file_path: Path, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent commits for a file."""
        format_str = "%H|%an|%ae|%at|%s"
        output = self._run_git_command(
            git_root,
            [
                "log",
                f"-{limit}",
                f"--format={format_str}",
                "--",
                str(file_path.relative_to(git_root)),
            ],
        )

        if not output:
            return []

        commits = []
        for line in output.split("\n"):
            if "|" in line:
                parts = line.split("|", 4)
                if len(parts) == 5:
                    commits.append(
                        {
                            "hash": parts[0][:8],
                            "author": parts[1],
                            "email": parts[2],
                            "timestamp": parts[3],
                            "message": parts[4],
                        }
                    )
        return commits

    def _get_blame_info(
        self, git_root: Path, file_path: Path, start_line: int, end_line: int
    ) -> Dict[int, Dict[str, Any]]:
        """Get git blame info for specific lines."""
        output = self._run_git_command(
            git_root,
            [
                "blame",
                "-L",
                f"{start_line},{end_line}",
                "--porcelain",
                "--",
                str(file_path.relative_to(git_root)),
            ],
        )

        if not output:
            return {}

        blame = {}
        current_hash = None
        current_line = start_line

        for line in output.split("\n"):
            if len(line) >= 40 and not line.startswith("\t"):
                parts = line.split()
                if len(parts) >= 3:
                    current_hash = parts[0][:8]
            elif line.startswith("author "):
                if current_hash:
                    blame[current_line] = {
                        "hash": current_hash,
                        "author": line[7:],
                    }
                    current_line += 1

        return blame


# =============================================================================
# Project Context Provider
# =============================================================================


class ProjectContextProvider(ContextProvider):
    """Provides context from project structure and configuration."""

    @property
    def context_type(self) -> ContextType:
        return ContextType.PROJECT

    @property
    def name(self) -> str:
        return "project_context"

    def gather(self, element: CodeElement, file_path: Optional[str] = None) -> ContextResult:
        """Gather project-level context."""
        import time

        start = time.time()
        result = ContextResult()

        if not file_path:
            result.add_error("No file path provided")
            return result

        path = Path(file_path)
        project_root = self._find_project_root(path)

        if not project_root:
            result.add_error("Could not determine project root")
            return result

        try:
            proj_ctx = ProjectContext(
                name=project_root.name,
                root_path=str(project_root),
                language=self._detect_language(project_root),
            )

            # Parse project configuration
            config_data = self._parse_project_config(project_root)
            if config_data:
                proj_ctx.dependencies = config_data.get("dependencies", {})
                proj_ctx.dev_dependencies = config_data.get("dev_dependencies", {})
                proj_ctx.framework = config_data.get("framework")

            # Find entry points
            proj_ctx.entry_points = self._find_entry_points(project_root)

            # Find test files
            proj_ctx.test_files = self._find_test_files(project_root)

            result.add(
                ContextItem(
                    type=ContextType.PROJECT,
                    key="project_info",
                    value=proj_ctx.__dict__,
                    priority=ContextPriority.HIGH,
                    scope=ContextScope.PROJECT,
                    source=str(project_root),
                )
            )

            # Add dependencies as separate item
            if proj_ctx.dependencies:
                result.add(
                    ContextItem(
                        type=ContextType.DEPENDENCY,
                        key="project_dependencies",
                        value=proj_ctx.dependencies,
                        priority=ContextPriority.MEDIUM,
                        scope=ContextScope.PROJECT,
                        source=str(project_root),
                    )
                )

        except Exception as e:
            result.add_error(f"Error gathering project context: {str(e)}")

        result.duration_ms = (time.time() - start) * 1000
        return result

    def _find_project_root(self, path: Path) -> Optional[Path]:
        """Find the project root directory."""
        markers = [
            "pyproject.toml",
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
            ".git",
        ]

        current = path if path.is_dir() else path.parent
        while current != current.parent:
            for marker in markers:
                if (current / marker).exists():
                    return current
            current = current.parent
        return None

    def _detect_language(self, root: Path) -> str:
        """Detect the primary language of the project."""
        if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
            return "python"
        if (root / "package.json").exists():
            return "javascript"
        if (root / "Cargo.toml").exists():
            return "rust"
        if (root / "go.mod").exists():
            return "go"
        if (root / "pom.xml").exists():
            return "java"
        return "unknown"

    def _parse_project_config(self, root: Path) -> Dict[str, Any]:
        """Parse project configuration files."""
        config = {}

        # Try pyproject.toml
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomli

                with open(pyproject, "rb") as f:
                    data = tomli.load(f)
                    deps = {}
                    dev_deps = {}

                    # Poetry format
                    if "tool" in data and "poetry" in data["tool"]:
                        poetry = data["tool"]["poetry"]
                        deps = poetry.get("dependencies", {})
                        dev_deps = poetry.get("dev-dependencies", {})

                    # PEP 621 format
                    if "project" in data:
                        project = data["project"]
                        if "dependencies" in project:
                            for dep in project["dependencies"]:
                                name = dep.split(">=")[0].split("==")[0].split("<")[0]
                                deps[name] = dep

                    config["dependencies"] = {k: str(v) for k, v in deps.items()}
                    config["dev_dependencies"] = {k: str(v) for k, v in dev_deps.items()}

            except ImportError:
                pass
            except Exception:
                pass

        # Try package.json
        package_json = root / "package.json"
        if package_json.exists():
            try:
                import json

                with open(package_json) as f:
                    data = json.load(f)
                    config["dependencies"] = data.get("dependencies", {})
                    config["dev_dependencies"] = data.get("devDependencies", {})

                    # Detect framework
                    all_deps = {**config["dependencies"], **config["dev_dependencies"]}
                    if "react" in all_deps:
                        config["framework"] = "react"
                    elif "vue" in all_deps:
                        config["framework"] = "vue"
                    elif "angular" in all_deps:
                        config["framework"] = "angular"
                    elif "express" in all_deps:
                        config["framework"] = "express"

            except Exception:
                pass

        return config

    def _find_entry_points(self, root: Path) -> List[str]:
        """Find project entry points."""
        entry_points = []
        candidates = [
            "main.py",
            "__main__.py",
            "app.py",
            "index.py",
            "src/main.py",
            "src/__main__.py",
            "index.js",
            "src/index.js",
            "main.go",
            "cmd/main.go",
        ]

        for candidate in candidates:
            if (root / candidate).exists():
                entry_points.append(candidate)

        return entry_points

    def _find_test_files(self, root: Path) -> List[str]:
        """Find test files in the project."""
        test_files = []
        test_dirs = ["tests", "test", "spec", "__tests__"]

        for test_dir in test_dirs:
            test_path = root / test_dir
            if test_path.exists():
                for test_file in test_path.rglob("test_*.py"):
                    test_files.append(str(test_file.relative_to(root)))
                for test_file in test_path.rglob("*_test.py"):
                    test_files.append(str(test_file.relative_to(root)))
                for test_file in test_path.rglob("*.test.js"):
                    test_files.append(str(test_file.relative_to(root)))
                for test_file in test_path.rglob("*.spec.js"):
                    test_files.append(str(test_file.relative_to(root)))

        return test_files[:20]  # Limit to 20 test files


# =============================================================================
# Documentation Context Provider
# =============================================================================


class DocumentationContextProvider(ContextProvider):
    """Provides context from documentation and comments."""

    @property
    def context_type(self) -> ContextType:
        return ContextType.DOCUMENTATION

    @property
    def name(self) -> str:
        return "documentation_context"

    def gather(self, element: CodeElement, file_path: Optional[str] = None) -> ContextResult:
        """Gather documentation context."""
        import time

        start = time.time()
        result = ContextResult()

        # Add element's own docstring
        if element.docstring:
            result.add(
                ContextItem(
                    type=ContextType.DOCUMENTATION,
                    key="element_docstring",
                    value=element.docstring,
                    priority=ContextPriority.CRITICAL,
                    scope=ContextScope.LOCAL,
                    source=file_path,
                )
            )

        if file_path:
            path = Path(file_path)

            # Look for README in same directory
            readme_files = ["README.md", "README.rst", "README.txt", "README"]
            for readme in readme_files:
                readme_path = path.parent / readme
                if readme_path.exists():
                    try:
                        content = readme_path.read_text(encoding="utf-8")[:2000]
                        result.add(
                            ContextItem(
                                type=ContextType.DOCUMENTATION,
                                key="readme",
                                value=content,
                                priority=ContextPriority.MEDIUM,
                                scope=ContextScope.MODULE,
                                source=str(readme_path),
                            )
                        )
                        break
                    except Exception:
                        pass

            # Extract inline comments near the element
            if path.exists() and element.line_start > 0:
                try:
                    lines = path.read_text(encoding="utf-8").split("\n")
                    comments = self._extract_nearby_comments(
                        lines, element.line_start, element.line_end
                    )
                    if comments:
                        result.add(
                            ContextItem(
                                type=ContextType.DOCUMENTATION,
                                key="inline_comments",
                                value=comments,
                                priority=ContextPriority.HIGH,
                                scope=ContextScope.LOCAL,
                                source=file_path,
                            )
                        )
                except Exception:
                    pass

        result.duration_ms = (time.time() - start) * 1000
        return result

    def _extract_nearby_comments(
        self, lines: List[str], start: int, end: int
    ) -> List[Dict[str, Any]]:
        """Extract comments near the element."""
        comments = []

        # Look for comments before the element
        for i in range(max(0, start - 10), start - 1):
            line = lines[i].strip()
            if line.startswith("#") or line.startswith("//"):
                comments.append({"line": i + 1, "text": line, "position": "before"})

        # Look for inline comments within the element
        for i in range(start - 1, min(len(lines), end)):
            line = lines[i]
            # Check for inline comment
            if "#" in line:
                parts = line.split("#", 1)
                if len(parts) > 1 and not line.strip().startswith("#"):
                    comments.append(
                        {
                            "line": i + 1,
                            "text": "#" + parts[1].strip(),
                            "position": "inline",
                        }
                    )

        return comments


# =============================================================================
# Usage Context Provider
# =============================================================================


class UsageContextProvider(ContextProvider):
    """Provides context about how code elements are used."""

    @property
    def context_type(self) -> ContextType:
        return ContextType.USAGE

    @property
    def name(self) -> str:
        return "usage_context"

    def gather(self, element: CodeElement, file_path: Optional[str] = None) -> ContextResult:
        """Gather usage context."""
        import time

        start = time.time()
        result = ContextResult()

        if not file_path:
            result.add_error("No file path provided")
            return result

        path = Path(file_path)
        if not path.exists():
            return result

        try:
            # Find usages in the same file
            content = path.read_text(encoding="utf-8")
            same_file_usages = self._find_usages_in_content(
                content, element.name, exclude_definition=True
            )

            if same_file_usages:
                result.add(
                    ContextItem(
                        type=ContextType.USAGE,
                        key="same_file_usages",
                        value=same_file_usages,
                        priority=ContextPriority.HIGH,
                        scope=ContextScope.LOCAL,
                        source=file_path,
                    )
                )

            # Find usages in sibling files
            project_root = self._find_project_root(path)
            if project_root:
                other_usages = self._find_usages_in_project(
                    project_root, element.name, exclude_file=path
                )
                if other_usages:
                    result.add(
                        ContextItem(
                            type=ContextType.USAGE,
                            key="project_usages",
                            value=other_usages,
                            priority=ContextPriority.MEDIUM,
                            scope=ContextScope.PROJECT,
                            source=str(project_root),
                        )
                    )

        except Exception as e:
            result.add_error(f"Error gathering usage context: {str(e)}")

        result.duration_ms = (time.time() - start) * 1000
        return result

    def _find_usages_in_content(
        self, content: str, name: str, exclude_definition: bool = True
    ) -> List[Dict[str, Any]]:
        """Find usages of a name in content."""
        usages = []
        lines = content.split("\n")

        # Pattern to match the name as a whole word
        pattern = re.compile(rf"\b{re.escape(name)}\b")

        for i, line in enumerate(lines):
            if pattern.search(line):
                # Skip definition lines
                if exclude_definition:
                    stripped = line.strip()
                    if stripped.startswith(f"def {name}") or stripped.startswith(f"class {name}"):
                        continue

                usages.append(
                    {
                        "line": i + 1,
                        "text": line.strip(),
                        "context": self._get_usage_context(line, name),
                    }
                )

        return usages[:20]  # Limit results

    def _find_usages_in_project(
        self, root: Path, name: str, exclude_file: Path
    ) -> List[Dict[str, Any]]:
        """Find usages in project files."""
        usages = []

        python_files = list(root.rglob("*.py"))
        for py_file in python_files[:50]:  # Limit files to scan
            if py_file == exclude_file:
                continue
            if not self.should_include(str(py_file)):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                if name in content:
                    file_usages = self._find_usages_in_content(content, name)
                    for usage in file_usages[:3]:  # Limit per file
                        usage["file"] = str(py_file.relative_to(root))
                        usages.append(usage)
            except Exception:
                continue

            if len(usages) >= 10:
                break

        return usages

    def _get_usage_context(self, line: str, name: str) -> str:
        """Determine the context of a usage."""
        stripped = line.strip()
        if f"{name}(" in stripped:
            return "function_call"
        if f"{name}." in stripped:
            return "attribute_access"
        if f"= {name}" in stripped:
            return "assignment"
        if f"import {name}" in stripped or "from" in stripped:
            return "import"
        return "reference"

    def _find_project_root(self, path: Path) -> Optional[Path]:
        """Find project root."""
        markers = ["pyproject.toml", "setup.py", "package.json", ".git"]
        current = path.parent
        while current != current.parent:
            for marker in markers:
                if (current / marker).exists():
                    return current
            current = current.parent
        return None


# =============================================================================
# Context Manager
# =============================================================================


class ContextManager:
    """Manages multiple context providers and aggregates results."""

    def __init__(self, providers: Optional[List[ContextProvider]] = None):
        """Initialize context manager."""
        self.providers: List[ContextProvider] = providers or []
        self._default_providers: List[ContextProvider] = [
            FileContextProvider(),
            GitContextProvider(),
            ProjectContextProvider(),
            DocumentationContextProvider(),
            UsageContextProvider(),
        ]

    def add_provider(self, provider: ContextProvider) -> None:
        """Add a context provider."""
        self.providers.append(provider)

    def remove_provider(self, name: str) -> bool:
        """Remove a provider by name."""
        for i, provider in enumerate(self.providers):
            if provider.name == name:
                self.providers.pop(i)
                return True
        return False

    def get_providers(self) -> List[ContextProvider]:
        """Get all active providers."""
        return self.providers if self.providers else self._default_providers

    def gather_context(
        self,
        element: CodeElement,
        file_path: Optional[str] = None,
        provider_names: Optional[List[str]] = None,
    ) -> ContextResult:
        """Gather context from all enabled providers.

        Args:
            element: Code element to gather context for.
            file_path: Path to the file containing the element.
            provider_names: Optional list of provider names to use.

        Returns:
            Aggregated ContextResult from all providers.
        """
        import time

        start = time.time()
        result = ContextResult()

        providers = self.get_providers()
        if provider_names:
            providers = [p for p in providers if p.name in provider_names]

        for provider in providers:
            if not provider.is_enabled():
                continue

            try:
                provider_result = provider.gather(element, file_path)
                result.items.extend(provider_result.items)
                result.errors.extend(provider_result.errors)
                result.metadata[provider.name] = {
                    "duration_ms": provider_result.duration_ms,
                    "item_count": len(provider_result.items),
                }
            except Exception as e:
                result.add_error(f"Provider {provider.name} failed: {str(e)}")

        result.duration_ms = (time.time() - start) * 1000
        return result

    def gather_context_filtered(
        self,
        element: CodeElement,
        file_path: Optional[str] = None,
        min_priority: ContextPriority = ContextPriority.LOW,
    ) -> ContextResult:
        """Gather context filtered by priority."""
        result = self.gather_context(element, file_path)

        priority_order = [
            ContextPriority.CRITICAL,
            ContextPriority.HIGH,
            ContextPriority.MEDIUM,
            ContextPriority.LOW,
            ContextPriority.OPTIONAL,
        ]
        min_idx = priority_order.index(min_priority)
        allowed_priorities = set(priority_order[: min_idx + 1])

        result.items = [item for item in result.items if item.priority in allowed_priorities]
        return result


# =============================================================================
# Global Instance Management
# =============================================================================


_global_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global _global_context_manager
    if _global_context_manager is None:
        _global_context_manager = ContextManager()
    return _global_context_manager


def reset_context_manager() -> None:
    """Reset the global context manager."""
    global _global_context_manager
    _global_context_manager = None


def gather_context(
    element: CodeElement,
    file_path: Optional[str] = None,
    provider_names: Optional[List[str]] = None,
) -> ContextResult:
    """Convenience function to gather context using global manager."""
    return get_context_manager().gather_context(element, file_path, provider_names)


def create_context_provider(provider_type: ContextType) -> ContextProvider:
    """Create a context provider by type."""
    providers = {
        ContextType.FILE: FileContextProvider,
        ContextType.GIT: GitContextProvider,
        ContextType.PROJECT: ProjectContextProvider,
        ContextType.DOCUMENTATION: DocumentationContextProvider,
        ContextType.USAGE: UsageContextProvider,
    }

    provider_class = providers.get(provider_type)
    if not provider_class:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return provider_class()
