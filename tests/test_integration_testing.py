"""Tests for integration testing framework."""

import pytest
import time
from openeval.integration_testing import (
    MockService,
    IntegrationTestScenario,
    MockServiceRegistry,
    IntegrationTestRunner,
    HTTPClientMock,
    create_evaluation_integration_tests,
    HAS_FASTAPI,
)


class TestMockService:
    """Test the MockService class."""

    def test_init(self):
        """Test initialization."""
        service = MockService(name="test_service", base_url="http://test.com")
        assert service.name == "test_service"
        assert service.base_url == "http://test.com"
        assert service.delay == 0.0
        assert service.error_rate == 0.0

    def test_add_endpoint(self):
        """Test adding endpoints."""
        service = MockService(name="test", base_url="http://test.com")
        service.add_endpoint("/api/test", "GET", {"result": "ok"}, 200)

        assert "/api/test" in service.endpoints
        assert "GET" in service.endpoints["/api/test"]
        assert service.endpoints["/api/test"]["GET"]["response"] == {"result": "ok"}
        assert service.endpoints["/api/test"]["GET"]["status_code"] == 200

    def test_get_response_success(self):
        """Test getting response for existing endpoint."""
        service = MockService(name="test", base_url="http://test.com")
        service.add_endpoint("/api/test", "GET", {"result": "ok"}, 200)

        response = service.get_response("/api/test", "GET")
        assert response["status_code"] == 200
        assert response["response"] == {"result": "ok"}

    def test_get_response_not_found(self):
        """Test getting response for non-existent endpoint."""
        service = MockService(name="test", base_url="http://test.com")

        response = service.get_response("/api/missing", "GET")
        assert response["status_code"] == 404
        assert response["response"] == {"error": "Not Found"}

    def test_get_response_with_delay(self):
        """Test response with delay."""
        service = MockService(name="test", base_url="http://test.com", delay=0.1)
        service.add_endpoint("/api/test", "GET", {"result": "ok"})

        start_time = time.time()
        response = service.get_response("/api/test", "GET")
        duration = time.time() - start_time

        assert response["status_code"] == 200
        assert duration >= 0.1

    def test_get_response_with_error_rate(self):
        """Test response with error rate."""
        service = MockService(name="test", base_url="http://test.com", error_rate=1.0)
        service.add_endpoint("/api/test", "GET", {"result": "ok"})

        response = service.get_response("/api/test", "GET")
        assert response["status_code"] == 500
        assert response["response"] == {"error": "Internal Server Error"}


class TestMockServiceRegistry:
    """Test the MockServiceRegistry class."""

    def test_register_and_get(self):
        """Test registering and getting services."""
        registry = MockServiceRegistry()
        service = MockService(name="test", base_url="http://test.com")

        registry.register(service)
        retrieved = registry.get_service("test")

        assert retrieved is service

    def test_get_nonexistent_service(self):
        """Test getting a non-existent service."""
        registry = MockServiceRegistry()
        assert registry.get_service("missing") is None

    def test_start_all_services(self):
        """Test starting all services."""
        registry = MockServiceRegistry()
        service1 = MockService(name="service1", base_url="http://service1.com")
        service2 = MockService(name="service2", base_url="http://service2.com")

        registry.register(service1)
        registry.register(service2)

        # Should not raise any exceptions
        registry.start_all()
        registry.stop_all()


class TestIntegrationTestRunner:
    """Test the IntegrationTestRunner class."""

    def test_add_scenario(self):
        """Test adding scenarios."""
        runner = IntegrationTestRunner()

        def dummy_test():
            pass

        scenario = IntegrationTestScenario(
            name="test_scenario", description="Test scenario", services=[], test_function=dummy_test
        )

        runner.add_scenario(scenario)
        assert len(runner.scenarios) == 1

    def test_run_scenario_success(self):
        """Test running a successful scenario."""
        runner = IntegrationTestRunner()

        def success_test():
            time.sleep(0.01)  # Simulate some work

        scenario = IntegrationTestScenario(
            name="success_test",
            description="Successful test",
            services=[],
            test_function=success_test,
            timeout=1.0,
        )

        result = runner.run_scenario(scenario)

        assert result["scenario"] == "success_test"
        assert result["status"] == "passed"
        assert result["duration"] is not None
        assert result["error"] is None

    def test_run_scenario_failure(self):
        """Test running a failing scenario."""
        runner = IntegrationTestRunner()

        def failing_test():
            raise ValueError("Test failure")

        scenario = IntegrationTestScenario(
            name="failing_test", description="Failing test", services=[], test_function=failing_test
        )

        result = runner.run_scenario(scenario)

        assert result["status"] == "failed"
        assert "Test failure" in result["error"]

    def test_run_scenario_timeout(self):
        """Test running a scenario that times out."""
        runner = IntegrationTestRunner()

        def slow_test():
            time.sleep(0.2)  # Longer than timeout

        scenario = IntegrationTestScenario(
            name="slow_test",
            description="Slow test",
            services=[],
            test_function=slow_test,
            timeout=0.1,
        )

        result = runner.run_scenario(scenario)

        assert result["status"] == "timeout"
        assert "timed out" in result["error"]

    def test_run_all_scenarios(self):
        """Test running all scenarios."""
        runner = IntegrationTestRunner()

        def test1():
            pass

        def test2():
            pass

        scenario1 = IntegrationTestScenario(
            name="test1", description="Test 1", services=[], test_function=test1, tags=["tag1"]
        )

        scenario2 = IntegrationTestScenario(
            name="test2", description="Test 2", services=[], test_function=test2, tags=["tag2"]
        )

        runner.add_scenario(scenario1)
        runner.add_scenario(scenario2)

        # Run all
        results = runner.run_all_scenarios()
        assert len(results) == 2

        # Run filtered by tags
        results = runner.run_all_scenarios(tags=["tag1"])
        assert len(results) == 1
        assert results[0]["scenario"] == "test1"

    def test_get_summary(self):
        """Test getting test summary."""
        runner = IntegrationTestRunner()

        # Simulate some results
        runner.results = [
            {"status": "passed"},
            {"status": "failed"},
            {"status": "passed"},
            {"status": "timeout"},
        ]

        summary = runner.get_summary()
        assert summary["total"] == 4
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["timeout"] == 1
        assert summary["success_rate"] == 0.5


class TestHTTPClientMock:
    """Test the HTTPClientMock class."""

    @pytest.mark.asyncio
    async def test_get_request(self):
        """Test GET request."""
        registry = MockServiceRegistry()
        service = MockService(name="test", base_url="http://test.com")
        service.add_endpoint("/api/data", "GET", {"data": "test"}, 200)
        registry.register(service)

        client = HTTPClientMock(registry)
        response = await client.get("http://test.com/api/data")

        assert response["status_code"] == 200
        assert response["json"]() == {"data": "test"}

    @pytest.mark.asyncio
    async def test_post_request(self):
        """Test POST request."""
        registry = MockServiceRegistry()
        service = MockService(name="test", base_url="http://test.com")
        service.add_endpoint("/api/create", "POST", {"id": 123}, 201)
        registry.register(service)

        client = HTTPClientMock(registry)
        response = await client.post("http://test.com/api/create")

        assert response["status_code"] == 201
        assert response["json"]() == {"id": 123}

    @pytest.mark.asyncio
    async def test_service_not_found(self):
        """Test request to non-existent service."""
        registry = MockServiceRegistry()
        client = HTTPClientMock(registry)

        response = await client.get("http://unknown.com/api/test")

        assert response["status_code"] == 404
        assert response["json"]() == {"error": "Service not found"}


class TestCreateEvaluationIntegrationTests:
    """Test the predefined evaluation integration tests."""

    def test_create_evaluation_integration_tests(self):
        """Test creating evaluation integration tests."""
        scenarios = create_evaluation_integration_tests()

        assert len(scenarios) == 3

        # Check scenario names
        names = [s.name for s in scenarios]
        assert "basic_evaluation_workflow" in names
        assert "evaluation_error_handling" in names
        assert "evaluation_performance" in names

    def test_scenario_structure(self):
        """Test the structure of created scenarios."""
        scenarios = create_evaluation_integration_tests()

        for scenario in scenarios:
            assert isinstance(scenario, IntegrationTestScenario)
            assert scenario.name
            assert scenario.description
            assert callable(scenario.test_function)
            assert isinstance(scenario.services, list)
            assert isinstance(scenario.tags, list)

    def test_basic_evaluation_scenario(self):
        """Test the basic evaluation scenario."""
        scenarios = create_evaluation_integration_tests()
        scenario = next(s for s in scenarios if s.name == "basic_evaluation_workflow")

        assert len(scenario.services) == 2
        service_names = [s.name for s in scenario.services]
        assert "dataset_service" in service_names
        assert "model_service" in service_names
        assert "evaluation" in scenario.tags
        assert "basic" in scenario.tags


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestFastAPIMockService:
    """Test the FastAPI mock service."""

    def test_fastapi_service_creation(self):
        """Test creating a FastAPI mock service."""
        from openeval.integration_testing import FastAPIMockService

        service = FastAPIMockService(
            name="fastapi_test",
            base_url="http://test.com",
            endpoints={"/test": {"GET": {"response": {"message": "ok"}, "status_code": 200}}},
        )

        assert service.name == "fastapi_test"
        assert hasattr(service, "app")
        assert hasattr(service, "client")
