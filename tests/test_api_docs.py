"""Tests for API documentation generator."""

import json
import tempfile
from pathlib import Path
from openeval.api_docs import (
    APIDocumentationGenerator,
    APIEndpoint,
    APISchema,
    create_evaluation_api_docs,
)


class TestAPIDocumentationGenerator:
    """Test the API documentation generator."""

    def test_init(self):
        """Test initialization."""
        generator = APIDocumentationGenerator("Test API", "2.0.0")
        assert generator.title == "Test API"
        assert generator.version == "2.0.0"
        assert len(generator.endpoints) == 0
        assert len(generator.schemas) == 0

    def test_add_endpoint(self):
        """Test adding endpoints."""
        generator = APIDocumentationGenerator()
        endpoint = APIEndpoint(
            path="/test",
            method="GET",
            description="Test endpoint",
            parameters={},
            responses={},
            examples=[],
            tags=["test"],
        )
        generator.add_endpoint(endpoint)
        assert len(generator.endpoints) == 1

    def test_add_schema(self):
        """Test adding schemas."""
        generator = APIDocumentationGenerator()
        schema = APISchema(
            name="TestSchema",
            type="object",
            properties={"name": {"type": "string"}},
            required=["name"],
            examples=[{"name": "test"}],
        )
        generator.add_schema(schema)
        assert "TestSchema" in generator.schemas

    def test_generate_openapi_spec(self):
        """Test OpenAPI spec generation."""
        generator = APIDocumentationGenerator("Test API", "1.0.0")

        # Add a schema
        schema = APISchema(
            name="TestResponse",
            type="object",
            properties={"message": {"type": "string"}},
            required=["message"],
            examples=[{"message": "Hello World"}],
        )
        generator.add_schema(schema)

        # Add an endpoint
        endpoint = APIEndpoint(
            path="/hello",
            method="GET",
            description="Say hello",
            parameters={},
            responses={"200": {"description": "Success", "schema": "TestResponse"}},
            examples=[],
            tags=["greeting"],
        )
        generator.add_endpoint(endpoint)

        spec = generator.generate_openapi_spec()

        assert spec["openapi"] == "3.0.3"
        assert spec["info"]["title"] == "Test API"
        assert spec["info"]["version"] == "1.0.0"
        assert "/hello" in spec["paths"]
        assert "get" in spec["paths"]["/hello"]
        assert "TestResponse" in spec["components"]["schemas"]

    def test_generate_markdown_docs(self):
        """Test Markdown documentation generation."""
        generator = APIDocumentationGenerator("Test API", "1.0.0")

        endpoint = APIEndpoint(
            path="/test",
            method="POST",
            description="Test endpoint",
            parameters={
                "name": {
                    "in": "query",
                    "description": "Name parameter",
                    "required": True,
                    "type": "string",
                }
            },
            responses={"200": {"description": "Success"}, "400": {"description": "Bad Request"}},
            examples=[{"request": {"name": "test"}, "response": {"status": "ok"}}],
            tags=["test"],
        )
        generator.add_endpoint(endpoint)

        docs = generator.generate_markdown_docs()

        assert "# Test API" in docs
        assert "POST /test" in docs
        assert "Test endpoint" in docs
        assert "name" in docs
        assert "Success" in docs
        assert "Bad Request" in docs

    def test_save_openapi_spec(self):
        """Test saving OpenAPI spec to file."""
        generator = APIDocumentationGenerator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            generator.save_openapi_spec(temp_path)

            # Verify file was created and contains valid JSON
            with open(temp_path, "r") as f:
                data = json.load(f)
                assert "openapi" in data
                assert "info" in data
        finally:
            Path(temp_path).unlink()

    def test_save_markdown_docs(self):
        """Test saving Markdown docs to file."""
        generator = APIDocumentationGenerator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = f.name

        try:
            generator.save_markdown_docs(temp_path)

            # Verify file was created and contains content
            with open(temp_path, "r") as f:
                content = f.read()
                assert "# OpenEval Lab API" in content
        finally:
            Path(temp_path).unlink()

    def test_convert_parameters(self):
        """Test parameter conversion."""
        generator = APIDocumentationGenerator()

        parameters = {
            "limit": {
                "in": "query",
                "description": "Maximum number of results",
                "required": False,
                "type": "integer",
            },
            "status": {
                "in": "query",
                "description": "Filter by status",
                "required": False,
                "type": "string",
                "enum": ["active", "inactive"],
            },
        }

        openapi_params = generator._convert_parameters(parameters)

        assert len(openapi_params) == 2
        limit_param = next(p for p in openapi_params if p["name"] == "limit")
        assert limit_param["schema"]["type"] == "integer"
        assert not limit_param["required"]

        status_param = next(p for p in openapi_params if p["name"] == "status")
        assert "enum" in status_param["schema"]

    def test_convert_responses(self):
        """Test response conversion."""
        generator = APIDocumentationGenerator()

        responses = {
            "200": {"description": "Success", "schema": "SuccessResponse"},
            "404": {"description": "Not found"},
        }

        openapi_responses = generator._convert_responses(responses)

        assert "200" in openapi_responses
        assert "404" in openapi_responses
        assert openapi_responses["200"]["description"] == "Success"
        assert "application/json" in openapi_responses["200"]["content"]

    def test_get_json_type(self):
        """Test JSON type conversion."""
        generator = APIDocumentationGenerator()

        assert generator._get_json_type(str) == "string"
        assert generator._get_json_type(int) == "integer"
        assert generator._get_json_type(float) == "number"
        assert generator._get_json_type(bool) == "boolean"
        assert generator._get_json_type(list) == "array"
        assert generator._get_json_type(dict) == "object"


class TestCreateEvaluationAPIDocs:
    """Test the evaluation API docs factory."""

    def test_create_evaluation_api_docs(self):
        """Test creating evaluation API documentation."""
        generator = create_evaluation_api_docs()

        assert generator.title == "OpenEval Lab API"
        assert generator.version == "1.0.0"
        assert len(generator.endpoints) > 0
        assert len(generator.schemas) > 0

        # Check that evaluation schema exists
        assert "EvaluationRequest" in generator.schemas

        # Check that evaluation endpoint exists
        endpoint_paths = [e.path for e in generator.endpoints]
        assert "/evaluations" in endpoint_paths

    def test_evaluation_schema_structure(self):
        """Test the structure of the evaluation schema."""
        generator = create_evaluation_api_docs()
        schema = generator.schemas["EvaluationRequest"]

        assert schema.type == "object"
        assert "name" in schema.properties
        assert "task" in schema.properties
        assert "dataset" in schema.properties
        assert "name" in schema.required
        assert "task" in schema.required
        assert "dataset" in schema.required

    def test_evaluation_endpoint_structure(self):
        """Test the structure of the evaluation endpoint."""
        generator = create_evaluation_api_docs()
        endpoint = next(e for e in generator.endpoints if e.path == "/evaluations")

        assert endpoint.method == "POST"
        assert endpoint.description == "Create a new evaluation"
        assert "evaluations" in endpoint.tags
        assert len(endpoint.examples) > 0
