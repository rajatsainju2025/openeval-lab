"""API documentation generator for OpenEval Lab."""

import json
import inspect
from typing import Any, Dict, List, Optional, Type, get_type_hints
from dataclasses import dataclass
from datetime import datetime

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class APIEndpoint:
    """Represents an API endpoint."""

    path: str
    method: str
    description: str
    parameters: Dict[str, Any]
    responses: Dict[str, Any]
    examples: List[Dict[str, Any]]
    tags: List[str]


@dataclass
class APISchema:
    """Represents an API schema."""

    name: str
    type: str
    properties: Dict[str, Any]
    required: List[str]
    examples: List[Any]


class APIDocumentationGenerator:
    """Generates comprehensive API documentation."""

    def __init__(self, title: str = "OpenEval Lab API", version: str = "1.0.0"):
        self.title = title
        self.version = version
        self.endpoints: List[APIEndpoint] = []
        self.schemas: Dict[str, APISchema] = {}
        self.info = {
            "title": title,
            "version": version,
            "description": "Enterprise-grade evaluation framework API",
            "contact": {
                "name": "OpenEval Lab Team",
                "url": "https://github.com/rajatsainju2025/openeval-lab",
            },
            "license": {"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        }

    def add_endpoint(self, endpoint: APIEndpoint):
        """Add an API endpoint."""
        self.endpoints.append(endpoint)

    def add_schema(self, schema: APISchema):
        """Add an API schema."""
        self.schemas[schema.name] = schema

    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification."""
        spec = {
            "openapi": "3.0.3",
            "info": self.info,
            "servers": [
                {"url": "https://api.openeval-lab.com/v1", "description": "Production server"},
                {"url": "http://localhost:8000/v1", "description": "Development server"},
            ],
            "paths": {},
            "components": {"schemas": {}},
            "tags": [],
        }

        # Group endpoints by path
        paths = {}
        tags = set()

        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}

            paths[endpoint.path][endpoint.method.lower()] = {
                "summary": endpoint.description,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "parameters": self._convert_parameters(endpoint.parameters),
                "responses": self._convert_responses(endpoint.responses),
                "requestBody": (
                    self._convert_request_body(endpoint)
                    if endpoint.method.upper() in ["POST", "PUT", "PATCH"]
                    else None
                ),
            }

            # Remove None requestBody
            if paths[endpoint.path][endpoint.method.lower()]["requestBody"] is None:
                del paths[endpoint.path][endpoint.method.lower()]["requestBody"]

            # Add examples
            if endpoint.examples:
                paths[endpoint.path][endpoint.method.lower()]["examples"] = endpoint.examples

            tags.update(endpoint.tags)

        spec["paths"] = paths
        spec["tags"] = [
            {"name": tag, "description": f"Operations related to {tag}"} for tag in sorted(tags)
        ]

        # Add schemas
        for name, schema in self.schemas.items():
            spec["components"]["schemas"][name] = {
                "type": schema.type,
                "properties": schema.properties,
                "required": schema.required,
                "example": schema.examples[0] if schema.examples else None,
            }

        return spec

    def _convert_parameters(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert parameter dict to OpenAPI format."""
        openapi_params = []

        for name, param_info in parameters.items():
            param = {
                "name": name,
                "in": param_info.get("in", "query"),
                "description": param_info.get("description", ""),
                "required": param_info.get("required", False),
                "schema": {"type": param_info.get("type", "string")},
            }

            if "enum" in param_info:
                param["schema"]["enum"] = param_info["enum"]

            openapi_params.append(param)

        return openapi_params

    def _convert_responses(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Convert response dict to OpenAPI format."""
        openapi_responses = {}

        for status, response_info in responses.items():
            openapi_responses[status] = {
                "description": response_info.get("description", ""),
                "content": {},
            }

            if "schema" in response_info:
                openapi_responses[status]["content"] = {
                    "application/json": {
                        "schema": {"$ref": f"#/components/schemas/{response_info['schema']}"}
                    }
                }

        return openapi_responses

    def _convert_request_body(self, endpoint: APIEndpoint) -> Optional[Dict[str, Any]]:
        """Convert request body to OpenAPI format."""
        # Look for body parameters
        body_params = [p for p in endpoint.parameters.values() if p.get("in") == "body"]
        if not body_params:
            return None

        return {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": f"#/components/schemas/{body_params[0].get('schema', 'GenericObject')}"
                    },
                    "example": endpoint.examples[0].get("request") if endpoint.examples else None,
                }
            },
        }

    def generate_markdown_docs(self) -> str:
        """Generate Markdown documentation."""
        docs = [f"# {self.title}\n"]
        docs.append(f"**Version:** {self.version}")
        docs.append(f"**Generated:** {datetime.now().isoformat()}\n")

        docs.append("## Overview\n")
        docs.append(
            "Enterprise-grade evaluation framework API for comprehensive model evaluation.\n"
        )

        # Group endpoints by tags
        tagged_endpoints = {}
        for endpoint in self.endpoints:
            for tag in endpoint.tags:
                if tag not in tagged_endpoints:
                    tagged_endpoints[tag] = []
                tagged_endpoints[tag].append(endpoint)

        for tag, endpoints in tagged_endpoints.items():
            docs.append(f"## {tag.title()}\n")

            for endpoint in endpoints:
                docs.append(f"### {endpoint.method.upper()} {endpoint.path}\n")
                docs.append(f"{endpoint.description}\n")

                if endpoint.parameters:
                    docs.append("**Parameters:**\n")
                    for name, param in endpoint.parameters.items():
                        required = " (required)" if param.get("required") else ""
                        docs.append(f"- `{name}`: {param.get('description', '')}{required}")
                    docs.append("")

                if endpoint.responses:
                    docs.append("**Responses:**\n")
                    for status, response in endpoint.responses.items():
                        docs.append(f"- `{status}`: {response.get('description', '')}")
                    docs.append("")

                if endpoint.examples:
                    docs.append("**Examples:**\n")
                    for example in endpoint.examples:
                        if "request" in example:
                            docs.append("Request:")
                            docs.append("```json")
                            docs.append(json.dumps(example["request"], indent=2))
                            docs.append("```")
                        if "response" in example:
                            docs.append("Response:")
                            docs.append("```json")
                            docs.append(json.dumps(example["response"], indent=2))
                            docs.append("```")
                    docs.append("")

        # Add schemas section
        if self.schemas:
            docs.append("## Schemas\n")
            for name, schema in self.schemas.items():
                docs.append(f"### {name}\n")
                docs.append(f"**Type:** {schema.type}\n")

                if schema.properties:
                    docs.append("**Properties:**\n")
                    for prop_name, prop_info in schema.properties.items():
                        required = " (required)" if prop_name in schema.required else ""
                        prop_type = prop_info.get("type", "any")
                        description = prop_info.get("description", "")
                        docs.append(f"- `{prop_name}` ({prop_type}){required}: {description}")
                    docs.append("")

                if schema.examples:
                    docs.append("**Example:**\n")
                    docs.append("```json")
                    docs.append(json.dumps(schema.examples[0], indent=2))
                    docs.append("```\n")

        return "\n".join(docs)

    def save_openapi_spec(self, file_path: str):
        """Save OpenAPI specification to file."""
        spec = self.generate_openapi_spec()
        with open(file_path, "w") as f:
            json.dump(spec, f, indent=2)
        logger.info(f"OpenAPI spec saved to {file_path}")

    def save_markdown_docs(self, file_path: str):
        """Save Markdown documentation to file."""
        docs = self.generate_markdown_docs()
        with open(file_path, "w") as f:
            f.write(docs)
        logger.info(f"Markdown docs saved to {file_path}")

    @classmethod
    def from_module(
        cls, module, title: Optional[str] = None, version: Optional[str] = None
    ) -> "APIDocumentationGenerator":
        """Create documentation generator from a Python module."""
        generator = cls(title or "API Documentation", version or "1.0.0")

        # Inspect module for classes and functions
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, "__doc__"):
                generator._add_class_schema(obj)
            elif inspect.isfunction(obj) and hasattr(obj, "__doc__"):
                generator._add_function_endpoint(obj)

        return generator

    def _add_class_schema(self, cls: Type):
        """Add schema from class."""
        if not hasattr(cls, "__annotations__"):
            return

        properties = {}
        required = []

        for attr_name, attr_type in get_type_hints(cls).items():
            prop_info = {
                "type": self._get_json_type(attr_type),
                "description": getattr(cls, f"{attr_name}_description", ""),
            }
            properties[attr_name] = prop_info

            # Check if field is required (no default)
            if not hasattr(cls, attr_name):
                required.append(attr_name)

        schema = APISchema(
            name=cls.__name__, type="object", properties=properties, required=required, examples=[]
        )

        self.add_schema(schema)

    def _add_function_endpoint(self, func):
        """Add endpoint from function."""
        # This is a simplified implementation
        # In practice, you'd need decorators or other metadata
        pass

    def _get_json_type(self, python_type) -> str:
        """Convert Python type to JSON schema type."""
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        return type_mapping.get(python_type, "string")


# Example usage and predefined endpoints
def create_evaluation_api_docs() -> APIDocumentationGenerator:
    """Create API documentation for evaluation endpoints."""
    generator = APIDocumentationGenerator("OpenEval Lab API", "1.0.0")

    # Add evaluation schema
    eval_schema = APISchema(
        name="EvaluationRequest",
        type="object",
        properties={
            "name": {"type": "string", "description": "Name of the evaluation"},
            "task": {"type": "string", "description": "Task type (qa, code, etc.)"},
            "dataset": {"type": "string", "description": "Dataset name"},
            "model": {"type": "string", "description": "Model to evaluate"},
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Metrics to compute",
            },
        },
        required=["name", "task", "dataset"],
        examples=[
            {
                "name": "sample_eval",
                "task": "qa",
                "dataset": "squad",
                "model": "gpt-3.5-turbo",
                "metrics": ["accuracy", "f1"],
            }
        ],
    )
    generator.add_schema(eval_schema)

    # Add evaluation endpoint
    eval_endpoint = APIEndpoint(
        path="/evaluations",
        method="POST",
        description="Create a new evaluation",
        parameters={
            "requestBody": {
                "in": "body",
                "schema": "EvaluationRequest",
                "required": True,
                "description": "Evaluation configuration",
            }
        },
        responses={
            "201": {
                "description": "Evaluation created successfully",
                "schema": "EvaluationResponse",
            },
            "400": {"description": "Invalid request data"},
        },
        examples=[
            {
                "request": {
                    "name": "my_evaluation",
                    "task": "qa",
                    "dataset": "squad",
                    "model": "gpt-4",
                    "metrics": ["exact_match", "f1_score"],
                },
                "response": {
                    "id": "eval_123",
                    "status": "running",
                    "created_at": "2024-01-01T00:00:00Z",
                },
            }
        ],
        tags=["evaluations"],
    )
    generator.add_endpoint(eval_endpoint)

    return generator
