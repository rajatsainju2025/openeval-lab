#!/usr/bin/env python3
"""
API Documentation Generator for OpenEval Lab

This script automatically generates comprehensive API documentation
from the codebase, including modules, classes, functions, and usage examples.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import re


@dataclass
class APIEndpoint:
    """Represents an API endpoint."""

    path: str
    method: str
    description: str
    parameters: List[Dict[str, Any]]
    responses: Dict[str, Any]
    examples: List[Dict[str, Any]]


@dataclass
class APIModule:
    """Represents a module in the API."""

    name: str
    path: str
    description: str
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    submodules: List["APIModule"]


@dataclass
class APIClass:
    """Represents a class in the API."""

    name: str
    description: str
    methods: List[Dict[str, Any]]
    attributes: List[Dict[str, Any]]
    inheritance: List[str]


@dataclass
class APIFunction:
    """Represents a function in the API."""

    name: str
    signature: str
    description: str
    parameters: List[Dict[str, Any]]
    returns: Dict[str, Any]
    examples: List[str]
    raises: List[Dict[str, Any]]


class APIDocumentationGenerator:
    """Generates comprehensive API documentation."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_root = project_root / "src" / "openeval"
        self.docs_output = project_root / "docs" / "api"
        self.docs_output.mkdir(parents=True, exist_ok=True)
        # Cache for AST parsing to avoid reparsing files
        self._ast_cache: Dict[str, ast.AST] = {}
        self._content_cache: Dict[str, str] = {}
        # Cache for string representations of AST nodes to avoid repeated str() calls
        self._node_str_cache: Dict[int, str] = {}

    def generate_documentation(self) -> None:
        """Generate complete API documentation."""
        print("🔍 Analyzing codebase for API documentation...")

        # Generate module documentation
        modules = self._analyze_modules()

        # Generate web API documentation
        web_endpoints = self._analyze_web_endpoints()

        # Generate CLI documentation
        cli_commands = self._analyze_cli_commands()

        # Create documentation files
        self._create_module_docs(modules)
        self._create_web_api_docs(web_endpoints)
        self._create_cli_docs(cli_commands)
        self._create_index()

        print("✅ API documentation generated successfully!")

    def _analyze_modules(self) -> List[APIModule]:
        """Analyze Python modules for documentation."""
        modules = []

        # Walk through all Python files
        for py_file in self.src_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
                continue

            try:
                module_info = self._analyze_module(py_file)
                if module_info:
                    modules.append(module_info)
            except Exception as e:
                print(f"Warning: Failed to analyze {py_file}: {e}")

        return modules

    def _analyze_module(self, file_path: Path) -> Optional[APIModule]:
        """Analyze a single Python module."""
        try:
            file_path_str = str(file_path)

            # Check cache first
            if file_path_str in self._ast_cache:
                tree = self._ast_cache[file_path_str]
                content = self._content_cache[file_path_str]
            else:
                # Read and parse the file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content, filename=file_path_str)

                # Cache the results
                self._ast_cache[file_path_str] = tree  # type: ignore
                self._content_cache[file_path_str] = content

            # Extract module docstring
            module_doc = ast.get_docstring(tree) or ""  # type: ignore

            # Extract classes and functions
            classes = []
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, content)
                    if class_info:
                        classes.append(class_info)
                elif isinstance(node, ast.FunctionDef) and not any(
                    isinstance(parent, ast.ClassDef) for parent in self._get_parents(node, tree)
                ):
                    func_info = self._analyze_function(node, content)
                    if func_info:
                        functions.append(func_info)

            # Create module info
            rel_path = file_path.relative_to(self.src_root)
            module_name = str(rel_path).replace("/", ".").replace(".py", "")

            return APIModule(
                name=module_name,
                path=str(rel_path),
                description=module_doc,
                classes=classes,
                functions=functions,
                submodules=[],
            )

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def _analyze_class(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Analyze a class definition."""
        class_doc = ast.get_docstring(node) or ""

        methods = []
        attributes = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item, content)
                if method_info:
                    methods.append(method_info)
            elif isinstance(item, ast.AnnAssign):
                # Simple attribute analysis
                attr_name = (
                    item.target.id if isinstance(item.target, ast.Name) else str(item.target)
                )
                attributes.append(
                    {
                        "name": attr_name,
                        "type": (
                            self._get_type_annotation(item.annotation) if item.annotation else "Any"
                        ),
                    }
                )

        return {
            "name": node.name,
            "description": class_doc,
            "methods": methods,
            "attributes": attributes,
            "bases": [self._get_name(base) for base in node.bases],
        }

    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analyze a function definition."""
        func_doc = ast.get_docstring(node) or ""

        # Extract signature
        signature = f"{node.name}({', '.join(arg.arg for arg in node.args.args)})"

        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "type": self._get_type_annotation(arg.annotation) if arg.annotation else "Any",
                "default": None,  # Would need more complex analysis
            }
            parameters.append(param_info)

        # Extract return type
        returns = {"type": self._get_type_annotation(node.returns) if node.returns else "Any"}

        return {
            "name": node.name,
            "signature": signature,
            "description": func_doc,
            "parameters": parameters,
            "returns": returns,
            "examples": self._extract_examples(func_doc),
        }

    def _get_type_annotation(self, node: ast.AST) -> str:
        """Convert AST type annotation to string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            base = self._get_type_annotation(node.value)
            if isinstance(node.slice, ast.Tuple):
                args = [self._get_type_annotation(arg) for arg in node.slice.elts]
                return f"{base}[{', '.join(args)}]"
            else:
                arg = self._get_type_annotation(node.slice)
                return f"{base}[{arg}]"
        elif isinstance(node, ast.Str):
            return repr(node.s)
        else:
            return "Any"

    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node."""
        node_id = id(node)
        if node_id in self._node_str_cache:
            return self._node_str_cache[node_id]

        if isinstance(node, ast.Name):
            result = node.id
        elif isinstance(node, ast.Attribute):
            result = f"{self._get_name(node.value)}.{node.attr}"
        else:
            result = str(node)

        self._node_str_cache[node_id] = result
        return result

    def _get_parents(self, node: ast.AST, tree: ast.AST) -> List[ast.AST]:
        """Get parent nodes for a given node."""
        parents = []
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                if child == node:
                    parents.append(parent)
        return parents

    def _extract_examples(self, docstring: str) -> List[str]:
        """Extract code examples from docstring."""
        examples = []
        lines = docstring.split("\n")
        in_example = False
        current_example = []

        for line in lines:
            if line.strip().startswith(">>>") or line.strip().startswith("..."):
                in_example = True
                current_example.append(line)
            elif in_example and line.strip():
                current_example.append(line)
            elif in_example and not line.strip():
                if current_example:
                    examples.append("\n".join(current_example))
                    current_example = []
                in_example = False

        if current_example:
            examples.append("\n".join(current_example))

        return examples

    def _analyze_web_endpoints(self) -> List[APIEndpoint]:
        """Analyze web API endpoints."""
        endpoints = []

        # Look for FastAPI route definitions
        web_files = list(self.src_root.glob("web/*.py"))
        for web_file in web_files:
            try:
                with open(web_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Simple regex-based endpoint extraction
                # This could be enhanced with AST analysis
                route_patterns = [
                    r'@app\.(get|post|put|delete)\(["\']([^"\']+)["\']',
                    r'@router\.(get|post|put|delete)\(["\']([^"\']+)["\']',
                ]

                for pattern in route_patterns:
                    matches = re.findall(pattern, content)
                    for method, path in matches:
                        endpoints.append(
                            APIEndpoint(
                                path=path,
                                method=method.upper(),
                                description=f"{method.upper()} {path}",
                                parameters=[],
                                responses={},
                                examples=[],
                            )
                        )

            except Exception as e:
                print(f"Warning: Failed to analyze web endpoints in {web_file}: {e}")

        return endpoints

    def _analyze_cli_commands(self) -> List[Dict[str, Any]]:
        """Analyze CLI commands."""
        commands = []

        try:
            cli_file = self.src_root / "cli.py"
            if cli_file.exists():
                with open(cli_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract Typer commands
                command_pattern = r'@app\.command\(["\']([^"\']+)["\']'
                matches = re.findall(command_pattern, content)

                for command in matches:
                    commands.append(
                        {
                            "name": command,
                            "description": f"CLI command: {command}",
                            "usage": f"openeval {command}",
                        }
                    )

        except Exception as e:
            print(f"Warning: Failed to analyze CLI commands: {e}")

        return commands

    def _create_module_docs(self, modules: List[APIModule]) -> None:
        """Create module documentation files."""
        for module in modules:
            output_file = self.docs_output / f"{module.name.replace('.', '_')}.md"

            content = f"# {module.name}\n\n"
            content += f"**Path:** `{module.path}`\n\n"

            if module.description:
                content += f"{module.description}\n\n"

            # Classes
            if module.classes:
                content += "## Classes\n\n"
                for cls in module.classes:
                    content += f"### {cls['name']}\n\n"
                    if cls["description"]:
                        content += f"{cls['description']}\n\n"

                    if cls["bases"]:
                        content += f"**Inherits from:** {', '.join(cls['bases'])}\n\n"

                    if cls["methods"]:
                        content += "#### Methods\n\n"
                        for method in cls["methods"]:
                            content += f"- `{method['signature']}`\n"
                            if method["description"]:
                                content += f"  {method['description'].split('.')[0]}\n"
                        content += "\n"

            # Functions
            if module.functions:
                content += "## Functions\n\n"
                for func in module.functions:
                    content += f"### {func['name']}\n\n"
                    content += f"```python\n{func['signature']}\n```\n\n"
                    if func["description"]:
                        content += f"{func['description']}\n\n"

                    if func["parameters"]:
                        content += "#### Parameters\n\n"
                        for param in func["parameters"]:
                            content += (
                                f"- `{param['name']}` ({param['type']}): Parameter description\n"
                            )
                        content += "\n"

                    if func["examples"]:
                        content += "#### Examples\n\n"
                        for example in func["examples"][:2]:  # Limit examples
                            content += f"```python\n{example}\n```\n\n"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)

    def _create_web_api_docs(self, endpoints: List[APIEndpoint]) -> None:
        """Create web API documentation."""
        output_file = self.docs_output / "web_api.md"

        content = "# Web API Reference\n\n"
        content += "This section documents the REST API endpoints available in OpenEval Lab.\n\n"

        if endpoints:
            content += "## Endpoints\n\n"
            for endpoint in endpoints:
                content += f"### {endpoint.method} {endpoint.path}\n\n"
                content += f"{endpoint.description}\n\n"
        else:
            content += "No endpoints documented yet.\n\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

    def _create_cli_docs(self, commands: List[Dict[str, Any]]) -> None:
        """Create CLI documentation."""
        output_file = self.docs_output / "cli_reference.md"

        content = "# CLI Reference\n\n"
        content += "Command-line interface commands for OpenEval Lab.\n\n"

        if commands:
            content += "## Commands\n\n"
            for cmd in commands:
                content += f"### {cmd['name']}\n\n"
                content += f"{cmd['description']}\n\n"
                content += f"**Usage:** `{cmd['usage']}`\n\n"
        else:
            content += "No commands documented yet.\n\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

    def _create_index(self) -> None:
        """Create API documentation index."""
        index_file = self.docs_output / "index.md"

        content = "# API Documentation\n\n"
        content += "Welcome to the OpenEval Lab API documentation.\n\n"
        content += "This documentation is automatically generated from the codebase.\n\n"

        content += "## 📚 Documentation Sections\n\n"
        content += "- [Web API Reference](web_api.md) - REST API endpoints\n"
        content += "- [CLI Reference](cli_reference.md) - Command-line interface\n"
        content += "- [Core Module](core.md) - Core functionality\n"
        content += "- [Metrics](metrics.md) - Evaluation metrics\n"
        content += "- [Adapters](adapters.md) - Model adapters\n"
        content += "- [Tasks](tasks.md) - Evaluation tasks\n\n"

        content += "## 🔧 Development\n\n"
        content += "To regenerate this documentation:\n\n"
        content += "```bash\n"
        content += "python scripts/generate_api_docs.py\n"
        content += "```\n\n"

        content += "## 📖 Usage Examples\n\n"
        content += "```python\n"
        content += "# Basic evaluation\n"
        content += "from openeval import evaluate\n"
        content += "results = evaluate('config.json')\n"
        content += "```\n\n"

        with open(index_file, "w", encoding="utf-8") as f:
            f.write(content)


def main():
    """Main entry point for API documentation generation."""
    project_root = Path(__file__).parent.parent
    generator = APIDocumentationGenerator(project_root)
    generator.generate_documentation()


if __name__ == "__main__":
    main()
