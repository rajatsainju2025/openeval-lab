"""
Documentation Generator and API Reference System for OpenEval Lab

This module provides automated documentation generation, API reference creation,
and comprehensive documentation management for the OpenEval Lab project.
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import importlib
import pkgutil
import ast
import textwrap

from .enhanced_logging import get_logger

logger = get_logger(__name__)


class DocumentationFormat(Enum):
    """Supported documentation formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    RST = "rst"
    JSON = "json"


@dataclass
class APIEndpoint:
    """Represents an API endpoint or function."""
    name: str
    module: str
    signature: str
    docstring: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    category: str = "function"
    is_class: bool = False
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "module": self.module,
            "signature": self.signature,
            "docstring": self.docstring,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "examples": self.examples,
            "category": self.category,
            "is_class": self.is_class,
            "is_method": self.is_method,
            "decorators": self.decorators
        }


@dataclass
class ModuleDocumentation:
    """Documentation for a Python module."""
    name: str
    path: Path
    docstring: Optional[str] = None
    functions: List[APIEndpoint] = field(default_factory=list)
    classes: List[APIEndpoint] = field(default_factory=list)
    submodules: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": str(self.path),
            "docstring": self.docstring,
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "submodules": self.submodules,
            "dependencies": self.dependencies
        }


@dataclass
class DocumentationProject:
    """Complete documentation project."""
    title: str
    version: str
    modules: List[ModuleDocumentation] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    format: DocumentationFormat = DocumentationFormat.MARKDOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "version": self.version,
            "generated_at": self.generated_at.isoformat(),
            "format": self.format.value,
            "modules": [m.to_dict() for m in self.modules]
        }


class DocumentationGenerator:
    """
    Automated documentation generator for Python codebases.
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.source_dir = self.project_root / "src"
        self.docs_dir = self.project_root / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def generate_api_reference(
        self,
        package_name: str = "openeval",
        output_format: DocumentationFormat = DocumentationFormat.MARKDOWN
    ) -> Path:
        """
        Generate comprehensive API reference documentation.

        Args:
            package_name: Name of the package to document
            output_format: Output format for documentation

        Returns:
            Path to generated documentation
        """
        logger.info(f"Generating API reference for package: {package_name}")

        # Discover all modules in the package
        modules = self._discover_modules(package_name)

        # Generate documentation for each module
        project = DocumentationProject(
            title=f"{package_name} API Reference",
            version=self._get_package_version(),
            format=output_format
        )

        for module_name in modules:
            try:
                module_doc = self._document_module(module_name)
                if module_doc:
                    project.modules.append(module_doc)
            except Exception as e:
                logger.warning(f"Failed to document module {module_name}: {e}")

        # Generate output file
        if output_format == DocumentationFormat.MARKDOWN:
            content = self._generate_markdown_api_ref(project)
            file_path = self.docs_dir / "api_reference.md"
        elif output_format == DocumentationFormat.HTML:
            content = self._generate_html_api_ref(project)
            file_path = self.docs_dir / "api_reference.html"
        elif output_format == DocumentationFormat.JSON:
            content = json.dumps(project.to_dict(), indent=2)
            file_path = self.docs_dir / "api_reference.json"
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Generated API reference: {file_path}")
        return file_path

    def generate_usage_examples(
        self,
        examples_dir: Optional[Path] = None,
        output_format: DocumentationFormat = DocumentationFormat.MARKDOWN
    ) -> Path:
        """
        Generate usage examples documentation.

        Args:
            examples_dir: Directory containing example files
            output_format: Output format

        Returns:
            Path to generated documentation
        """
        examples_dir = examples_dir or self.project_root / "examples"

        if not examples_dir.exists():
            logger.warning(f"Examples directory not found: {examples_dir}")
            return self._create_empty_examples_doc(output_format)

        examples = []
        for example_file in examples_dir.glob("*.py"):
            try:
                example_doc = self._document_example(example_file)
                if example_doc:
                    examples.append(example_doc)
            except Exception as e:
                logger.warning(f"Failed to document example {example_file}: {e}")

        # Generate documentation
        if output_format == DocumentationFormat.MARKDOWN:
            content = self._generate_markdown_examples(examples)
            file_path = self.docs_dir / "usage_examples.md"
        elif output_format == DocumentationFormat.HTML:
            content = self._generate_html_examples(examples)
            file_path = self.docs_dir / "usage_examples.html"
        else:
            raise ValueError(f"Unsupported format for examples: {output_format}")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Generated usage examples: {file_path}")
        return file_path

    def generate_tutorial(
        self,
        tutorial_name: str,
        steps: List[Dict[str, Any]],
        output_format: DocumentationFormat = DocumentationFormat.MARKDOWN
    ) -> Path:
        """
        Generate a tutorial document.

        Args:
            tutorial_name: Name of the tutorial
            steps: List of tutorial steps with 'title', 'content', and optional 'code'
            output_format: Output format

        Returns:
            Path to generated tutorial
        """
        if output_format == DocumentationFormat.MARKDOWN:
            content = self._generate_markdown_tutorial(tutorial_name, steps)
            file_path = self.docs_dir / f"tutorial_{tutorial_name.lower().replace(' ', '_')}.md"
        elif output_format == DocumentationFormat.HTML:
            content = self._generate_html_tutorial(tutorial_name, steps)
            file_path = self.docs_dir / f"tutorial_{tutorial_name.lower().replace(' ', '_')}.html"
        else:
            raise ValueError(f"Unsupported format for tutorial: {output_format}")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Generated tutorial: {file_path}")
        return file_path

    def _discover_modules(self, package_name: str) -> List[str]:
        """Discover all modules in a package."""
        modules = []

        try:
            package = importlib.import_module(package_name)
            package_path = getattr(package, '__path__', [])

            for importer, modname, ispkg in pkgutil.walk_packages(
                package_path, package_name + "."
            ):
                modules.append(modname)

            # Add the root package
            modules.insert(0, package_name)

        except Exception as e:
            logger.error(f"Failed to discover modules in {package_name}: {e}")

        return modules

    def _document_module(self, module_name: str) -> Optional[ModuleDocumentation]:
        """Generate documentation for a single module."""
        try:
            module = importlib.import_module(module_name)
            module_file = getattr(module, '__file__', None)

            if not module_file:
                return None

            module_path = Path(module_file)
            doc = ModuleDocumentation(
                name=module_name,
                path=module_path,
                docstring=inspect.getdoc(module)
            )

            # Get all members
            members = inspect.getmembers(module)

            for name, obj in members:
                if name.startswith('_'):
                    continue  # Skip private members

                try:
                    if inspect.isclass(obj):
                        class_doc = self._document_class(name, obj, module_name)
                        if class_doc:
                            doc.classes.append(class_doc)
                    elif inspect.isfunction(obj):
                        func_doc = self._document_function(name, obj, module_name)
                        if func_doc:
                            doc.functions.append(func_doc)
                except Exception as e:
                    logger.debug(f"Failed to document {name} in {module_name}: {e}")

            return doc

        except Exception as e:
            logger.warning(f"Failed to document module {module_name}: {e}")
            return None

    def _document_function(
        self,
        name: str,
        func: Callable,
        module_name: str
    ) -> Optional[APIEndpoint]:
        """Document a function."""
        try:
            sig = inspect.signature(func)
            docstring = inspect.getdoc(func)

            # Parse parameters
            parameters = []
            for param_name, param in sig.parameters.items():
                param_info = {
                    "name": param_name,
                    "kind": str(param.kind),
                    "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                    "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None
                }
                parameters.append(param_info)

            # Get return type
            return_type = str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None

            # Get decorators (simplified)
            decorators = []
            if hasattr(func, '__wrapped__'):
                decorators.append("wrapped")

            return APIEndpoint(
                name=name,
                module=module_name,
                signature=f"{name}{sig}",
                docstring=docstring,
                parameters=parameters,
                return_type=return_type,
                decorators=decorators
            )

        except Exception as e:
            logger.debug(f"Failed to document function {name}: {e}")
            return None

    def _document_class(self, name: str, cls: type, module_name: str) -> Optional[APIEndpoint]:
        """Document a class."""
        try:
            sig = f"class {name}"
            if hasattr(cls, '__init__'):
                init_sig = inspect.signature(cls.__init__)
                sig += f"{init_sig}"

            docstring = inspect.getdoc(cls)

            # Get methods
            methods = []
            for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if not method_name.startswith('_'):
                    method_doc = self._document_function(method_name, method, f"{module_name}.{name}")
                    if method_doc:
                        method_doc.is_method = True
                        methods.append(method_doc)

            return APIEndpoint(
                name=name,
                module=module_name,
                signature=sig,
                docstring=docstring,
                category="class",
                is_class=True
            )

        except Exception as e:
            logger.debug(f"Failed to document class {name}: {e}")
            return None

    def _document_example(self, example_file: Path) -> Optional[Dict[str, Any]]:
        """Document a usage example."""
        try:
            with open(example_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract docstring or title from comments
            lines = content.split('\n')
            title = example_file.stem.replace('_', ' ').title()
            description = ""

            # Look for docstring at the top
            if lines and lines[0].startswith('"""'):
                # Multi-line docstring
                docstring_lines = []
                for line in lines[1:]:
                    if line.strip().endswith('"""'):
                        break
                    docstring_lines.append(line)
                description = '\n'.join(docstring_lines).strip()
            elif lines and lines[0].startswith('#'):
                # Single line comment
                description = lines[0][1:].strip()

            return {
                "title": title,
                "filename": example_file.name,
                "description": description,
                "code": content,
                "path": str(example_file)
            }

        except Exception as e:
            logger.warning(f"Failed to document example {example_file}: {e}")
            return None

    def _get_package_version(self) -> str:
        """Get the package version."""
        try:
            # Try to get version from pyproject.toml
            pyproject_file = self.project_root / "pyproject.toml"
            if pyproject_file.exists():
                import tomllib
                with open(pyproject_file, 'rb') as f:
                    data = tomllib.load(f)
                return data.get('tool', {}).get('poetry', {}).get('version', 'unknown')

            # Try __init__.py
            init_file = self.source_dir / "openeval" / "__init__.py"
            if init_file.exists():
                with open(init_file, 'r') as f:
                    content = f.read()
                    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                    if version_match:
                        return version_match.group(1)

        except Exception:
            pass

        return "unknown"

    def _generate_markdown_api_ref(self, project: DocumentationProject) -> str:
        """Generate Markdown API reference."""
        md = f"""# {project.title}

**Version:** {project.version}  
**Generated:** {project.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

"""

        for module in project.modules:
            md += f"## Module: {module.name}\n\n"

            if module.docstring:
                md += f"{module.docstring}\n\n"

            # Functions
            if module.functions:
                md += "### Functions\n\n"
                for func in module.functions:
                    md += f"#### `{func.signature}`\n\n"
                    if func.docstring:
                        md += f"{func.docstring}\n\n"
                    if func.parameters:
                        md += "**Parameters:**\n\n"
                        for param in func.parameters:
                            default = f" = {param['default']}" if param['default'] else ""
                            annotation = f": {param['annotation']}" if param['annotation'] else ""
                            md += f"- `{param['name']}{annotation}{default}`\n"
                        md += "\n"
                    if func.return_type:
                        md += f"**Returns:** {func.return_type}\n\n"

            # Classes
            if module.classes:
                md += "### Classes\n\n"
                for cls in module.classes:
                    md += f"#### `{cls.signature}`\n\n"
                    if cls.docstring:
                        md += f"{cls.docstring}\n\n"

            md += "---\n\n"

        return md

    def _generate_html_api_ref(self, project: DocumentationProject) -> str:
        """Generate HTML API reference."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{project.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .module {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .module-header {{ background: #e9ecef; padding: 10px; border-radius: 5px 5px 0 0; }}
        .function {{ margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px; }}
        .signature {{ font-family: monospace; font-weight: bold; }}
        .docstring {{ margin: 10px 0; }}
        .parameters {{ margin: 10px 0; }}
        .parameter {{ margin: 5px 0; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{project.title}</h1>
        <p><strong>Version:</strong> {project.version}</p>
        <p><strong>Generated:</strong> {project.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

        for module in project.modules:
            html += f"""
    <div class="module">
        <div class="module-header">
            <h2>Module: {module.name}</h2>
        </div>
        <div style="padding: 20px;">
"""

            if module.docstring:
                html += f"<p>{module.docstring}</p>"

            # Functions
            if module.functions:
                html += "<h3>Functions</h3>"
                for func in module.functions:
                    html += f"""
            <div class="function">
                <div class="signature">{func.signature}</div>
"""
                    if func.docstring:
                        html += f'<div class="docstring">{func.docstring}</div>'
                    if func.parameters:
                        html += '<div class="parameters"><strong>Parameters:</strong><br>'
                        for param in func.parameters:
                            default = f" = {param['default']}" if param['default'] else ""
                            annotation = f": {param['annotation']}" if param['annotation'] else ""
                            html += f'<div class="parameter">{param["name"]}{annotation}{default}</div>'
                        html += '</div>'
                    if func.return_type:
                        html += f'<div><strong>Returns:</strong> {func.return_type}</div>'

                    html += "</div>"

            # Classes
            if module.classes:
                html += "<h3>Classes</h3>"
                for cls in module.classes:
                    html += f"""
            <div class="function">
                <div class="signature">{cls.signature}</div>
"""
                    if cls.docstring:
                        html += f'<div class="docstring">{cls.docstring}</div>'
                    html += "</div>"

            html += """
        </div>
    </div>
"""

        html += """
</body>
</html>"""

        return html

    def _generate_markdown_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Generate Markdown usage examples."""
        md = "# Usage Examples\n\n"

        for example in examples:
            md += f"## {example['title']}\n\n"

            if example['description']:
                md += f"{example['description']}\n\n"

            md += f"**File:** `{example['filename']}`\n\n"

            # Format code with syntax highlighting
            code = textwrap.dedent(example['code']).strip()
            md += f"```python\n{code}\n```\n\n"

            md += "---\n\n"

        return md

    def _generate_html_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Generate HTML usage examples."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenEval Usage Examples</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .example { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
        .example-header { background: #e9ecef; padding: 10px; border-radius: 5px 5px 0 0; }
        .code { background: #f8f9fa; padding: 15px; border-radius: 0 0 5px 5px; font-family: monospace; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>OpenEval Usage Examples</h1>
    </div>
"""

        for example in examples:
            html += f"""
    <div class="example">
        <div class="example-header">
            <h2>{example['title']}</h2>
            <p><strong>File:</strong> {example['filename']}</p>
"""

            if example['description']:
                html += f"<p>{example['description']}</p>"

            html += f"""
        </div>
        <div class="code">{example['code'].replace('<', '&lt;').replace('>', '&gt;')}</div>
    </div>
"""

        html += """
</body>
</html>"""

        return html

    def _generate_markdown_tutorial(self, tutorial_name: str, steps: List[Dict[str, Any]]) -> str:
        """Generate Markdown tutorial."""
        md = f"# {tutorial_name}\n\n"

        for i, step in enumerate(steps, 1):
            md += f"## Step {i}: {step['title']}\n\n"

            if 'content' in step:
                md += f"{step['content']}\n\n"

            if 'code' in step:
                code = textwrap.dedent(step['code']).strip()
                md += f"```python\n{code}\n```\n\n"

        return md

    def _generate_html_tutorial(self, tutorial_name: str, steps: List[Dict[str, Any]]) -> str:
        """Generate HTML tutorial."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{tutorial_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .step {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .step-header {{ background: #e9ecef; padding: 10px; border-radius: 5px 5px 0 0; }}
        .content {{ padding: 15px; }}
        .code {{ background: #f8f9fa; padding: 15px; border-radius: 3px; font-family: monospace; overflow-x: auto; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{tutorial_name}</h1>
    </div>
"""

        for i, step in enumerate(steps, 1):
            html += f"""
    <div class="step">
        <div class="step-header">
            <h2>Step {i}: {step['title']}</h2>
        </div>
        <div class="content">
"""

            if 'content' in step:
                html += f"<p>{step['content']}</p>"

            if 'code' in step:
                code = step['code'].replace('<', '&lt;').replace('>', '&gt;')
                html += f'<div class="code">{code}</div>'

            html += """
        </div>
    </div>
"""

        html += """
</body>
</html>"""

        return html

    def _create_empty_examples_doc(self, output_format: DocumentationFormat) -> Path:
        """Create an empty examples documentation file."""
        if output_format == DocumentationFormat.MARKDOWN:
            content = "# Usage Examples\n\nNo examples found.\n"
            file_path = self.docs_dir / "usage_examples.md"
        else:
            content = "<!DOCTYPE html><html><head><title>Usage Examples</title></head><body><h1>Usage Examples</h1><p>No examples found.</p></body></html>"
            file_path = self.docs_dir / "usage_examples.html"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return file_path


def generate_api_docs(
    package_name: str = "openeval",
    output_format: DocumentationFormat = DocumentationFormat.MARKDOWN,
    project_root: Optional[Path] = None
) -> Path:
    """
    Generate API documentation for a package.

    Args:
        package_name: Name of the package
        output_format: Output format
        project_root: Project root directory

    Returns:
        Path to generated documentation
    """
    generator = DocumentationGenerator(project_root)
    return generator.generate_api_reference(package_name, output_format)


def generate_examples_docs(
    examples_dir: Optional[Path] = None,
    output_format: DocumentationFormat = DocumentationFormat.MARKDOWN,
    project_root: Optional[Path] = None
) -> Path:
    """
    Generate usage examples documentation.

    Args:
        examples_dir: Directory containing examples
        output_format: Output format
        project_root: Project root directory

    Returns:
        Path to generated documentation
    """
    generator = DocumentationGenerator(project_root)
    return generator.generate_usage_examples(examples_dir, output_format)


def create_tutorial(
    tutorial_name: str,
    steps: List[Dict[str, Any]],
    output_format: DocumentationFormat = DocumentationFormat.MARKDOWN,
    project_root: Optional[Path] = None
) -> Path:
    """
    Create a tutorial document.

    Args:
        tutorial_name: Name of the tutorial
        steps: Tutorial steps
        output_format: Output format
        project_root: Project root directory

    Returns:
        Path to generated tutorial
    """
    generator = DocumentationGenerator(project_root)
    return generator.generate_tutorial(tutorial_name, steps, output_format)