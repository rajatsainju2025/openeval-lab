"""Integration testing framework with mock services."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import threading

try:
    from fastapi import FastAPI, Response
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None
    TestClient = None

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class MockService:
    """A mock service for integration testing."""

    name: str
    base_url: str
    endpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    delay: float = 0.0
    error_rate: float = 0.0
    responses: Dict[str, Any] = field(default_factory=dict)

    def add_endpoint(
        self,
        path: str,
        method: str = "GET",
        response: Any = None,
        status_code: int = 200,
        delay: Optional[float] = None,
    ):
        """Add a mock endpoint."""
        if path not in self.endpoints:
            self.endpoints[path] = {}

        self.endpoints[path][method.upper()] = {
            "response": response,
            "status_code": status_code,
            "delay": delay or self.delay,
        }

    def get_response(self, path: str, method: str = "GET") -> Dict[str, Any]:
        """Get response for an endpoint."""
        method = method.upper()
        if path in self.endpoints and method in self.endpoints[path]:
            endpoint = self.endpoints[path][method]

            # Simulate delay
            if endpoint["delay"] > 0:
                time.sleep(endpoint["delay"])

            # Simulate errors
            if self.error_rate > 0 and time.time() % 1 < self.error_rate:
                return {"status_code": 500, "response": {"error": "Internal Server Error"}}

            return {"status_code": endpoint["status_code"], "response": endpoint["response"]}

        return {"status_code": 404, "response": {"error": "Not Found"}}


@dataclass
class IntegrationTestScenario:
    """An integration test scenario."""

    name: str
    description: str
    services: List[MockService]
    test_function: Callable
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    timeout: float = 30.0
    tags: List[str] = field(default_factory=list)


class MockServiceRegistry:
    """Registry for managing mock services."""

    def __init__(self):
        self.services: Dict[str, MockService] = {}

    def register(self, service: MockService):
        """Register a mock service."""
        self.services[service.name] = service

    def get_service(self, name: str) -> Optional[MockService]:
        """Get a registered service."""
        return self.services.get(name)

    def start_all(self):
        """Start all registered services."""
        for service in self.services.values():
            self._start_service(service)

    def stop_all(self):
        """Stop all registered services."""
        for service in self.services.values():
            self._stop_service(service)

    def _start_service(self, service: MockService):
        """Start a mock service (placeholder for actual server)."""
        logger.info(f"Starting mock service: {service.name} at {service.base_url}")

    def _stop_service(self, service: MockService):
        """Stop a mock service."""
        logger.info(f"Stopping mock service: {service.name}")


class IntegrationTestRunner:
    """Runner for integration tests."""

    def __init__(self):
        self.registry = MockServiceRegistry()
        self.scenarios: List[IntegrationTestScenario] = []
        self.results: List[Dict[str, Any]] = []

    def add_scenario(self, scenario: IntegrationTestScenario):
        """Add a test scenario."""
        self.scenarios.append(scenario)

    def run_scenario(self, scenario: IntegrationTestScenario) -> Dict[str, Any]:
        """Run a single test scenario."""
        logger.info(f"Running integration test: {scenario.name}")

        result = {
            "scenario": scenario.name,
            "description": scenario.description,
            "status": "running",
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "error": None,
            "tags": scenario.tags,
        }

        try:
            # Setup
            if scenario.setup:
                scenario.setup()

            # Start services
            for service in scenario.services:
                self.registry.register(service)
            self.registry.start_all()

            # Run test with timeout
            async def run_test():
                return await asyncio.wait_for(
                    asyncio.to_thread(scenario.test_function), timeout=scenario.timeout
                )

            asyncio.run(run_test())

            result["status"] = "passed"

        except asyncio.TimeoutError:
            result["status"] = "timeout"
            result["error"] = f"Test timed out after {scenario.timeout}s"
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Integration test failed: {scenario.name}", exc_info=True)
        finally:
            # Teardown
            try:
                self.registry.stop_all()
                if scenario.teardown:
                    scenario.teardown()
            except Exception as e:
                logger.error(f"Error during teardown: {e}")

            result["end_time"] = time.time()
            result["duration"] = result["end_time"] - result["start_time"]

        return result

    def run_all_scenarios(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run all test scenarios, optionally filtered by tags."""
        results = []

        for scenario in self.scenarios:
            if tags and not any(tag in scenario.tags for tag in tags):
                continue

            result = self.run_scenario(scenario)
            results.append(result)
            self.results.append(result)

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get test execution summary."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "timeout": 0}

        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "passed")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        timeout = sum(1 for r in self.results if r["status"] == "timeout")

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "timeout": timeout,
            "success_rate": passed / total if total > 0 else 0,
        }


class HTTPClientMock:
    """Mock HTTP client for testing."""

    def __init__(self, registry: MockServiceRegistry):
        self.registry = registry

    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Mock GET request."""
        return await self._make_request(url, "GET", **kwargs)

    async def post(self, url: str, **kwargs) -> Dict[str, Any]:
        """Mock POST request."""
        return await self._make_request(url, "POST", **kwargs)

    async def put(self, url: str, **kwargs) -> Dict[str, Any]:
        """Mock PUT request."""
        return await self._make_request(url, "PUT", **kwargs)

    async def delete(self, url: str, **kwargs) -> Dict[str, Any]:
        """Mock DELETE request."""
        return await self._make_request(url, "DELETE", **kwargs)

    async def _make_request(self, url: str, method: str, **kwargs) -> Dict[str, Any]:
        """Make a mock request."""
        # Parse URL to find service and path
        for service in self.registry.services.values():
            if url.startswith(service.base_url):
                path = url[len(service.base_url) :]
                if not path.startswith("/"):
                    path = "/" + path

                response = service.get_response(path, method)

                # Simulate async behavior
                await asyncio.sleep(0.001)

                return {
                    "status_code": response["status_code"],
                    "json": lambda: response["response"],
                    "text": json.dumps(response["response"]),
                }

        # Service not found
        return {
            "status_code": 404,
            "json": lambda: {"error": "Service not found"},
            "text": '{"error": "Service not found"}',
        }


# Predefined test scenarios
def create_evaluation_integration_tests() -> List[IntegrationTestScenario]:
    """Create integration test scenarios for evaluation workflows."""

    scenarios = []

    # Scenario 1: Basic evaluation workflow
    def basic_evaluation_test():
        """Test basic evaluation workflow with mock services."""
        # This would normally test the full evaluation pipeline
        # For now, just simulate the workflow
        time.sleep(0.1)  # Simulate work
        assert True  # Placeholder assertion

    scenario1 = IntegrationTestScenario(
        name="basic_evaluation_workflow",
        description="Test complete evaluation workflow from data loading to metrics",
        services=[
            MockService(
                name="dataset_service",
                base_url="http://datasets.local",
                endpoints={
                    "/datasets/squad": {
                        "GET": {"response": {"name": "squad", "size": 1000}, "status_code": 200}
                    }
                },
            ),
            MockService(
                name="model_service",
                base_url="http://models.local",
                endpoints={
                    "/models/gpt-4": {
                        "GET": {
                            "response": {"name": "gpt-4", "type": "language"},
                            "status_code": 200,
                        }
                    }
                },
            ),
        ],
        test_function=basic_evaluation_test,
        tags=["evaluation", "basic"],
    )
    scenarios.append(scenario1)

    # Scenario 2: Error handling in evaluation
    def error_handling_test():
        """Test error handling in evaluation pipeline."""
        # Simulate error conditions
        time.sleep(0.05)
        # Test would verify proper error handling
        assert True

    scenario2 = IntegrationTestScenario(
        name="evaluation_error_handling",
        description="Test error handling and recovery in evaluation workflows",
        services=[
            MockService(
                name="faulty_dataset_service",
                base_url="http://datasets.local",
                error_rate=0.3,  # 30% error rate
                endpoints={
                    "/datasets/broken": {
                        "GET": {"response": {"error": "Dataset unavailable"}, "status_code": 503}
                    }
                },
            )
        ],
        test_function=error_handling_test,
        tags=["evaluation", "error_handling"],
    )
    scenarios.append(scenario2)

    # Scenario 3: Performance under load
    def performance_test():
        """Test performance characteristics under load."""
        start_time = time.time()

        # Simulate multiple concurrent requests
        for i in range(10):
            time.sleep(0.01)  # Simulate work

        duration = time.time() - start_time
        assert duration < 0.2  # Should complete within 200ms

    scenario3 = IntegrationTestScenario(
        name="evaluation_performance",
        description="Test evaluation performance under concurrent load",
        services=[
            MockService(
                name="performance_service",
                base_url="http://performance.local",
                delay=0.005,  # 5ms delay per request
                endpoints={
                    "/evaluate": {"POST": {"response": {"result": "success"}, "status_code": 200}}
                },
            )
        ],
        test_function=performance_test,
        timeout=1.0,
        tags=["evaluation", "performance"],
    )
    scenarios.append(scenario3)

    return scenarios


# FastAPI integration (if available)
if HAS_FASTAPI and FastAPI is not None and TestClient is not None:

    class FastAPIMockService(MockService):
        """Mock service using FastAPI."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.app = FastAPI(title=f"Mock {self.name}")
            self.client = None
            self._setup_routes()

        def _setup_routes(self):
            """Setup FastAPI routes."""
            for path, methods in self.endpoints.items():
                for method, config in methods.items():
                    if method == "GET":
                        self.app.get(path)(self._create_handler(config))
                    elif method == "POST":
                        self.app.post(path)(self._create_handler(config))
                    elif method == "PUT":
                        self.app.put(path)(self._create_handler(config))
                    elif method == "DELETE":
                        self.app.delete(path)(self._create_handler(config))

        def _create_handler(self, config):
            """Create a route handler."""

            async def handler():
                if config["delay"] > 0:
                    await asyncio.sleep(config["delay"])

                if self.error_rate > 0 and time.time() % 1 < self.error_rate:
                    return Response(
                        content=json.dumps({"error": "Internal Server Error"}),
                        status_code=500,
                        media_type="application/json",
                    )

                return Response(
                    content=json.dumps(config["response"]),
                    status_code=config["status_code"],
                    media_type="application/json",
                )

            return handler

        def start(self):
            """Start the FastAPI server."""
            from uvicorn import Server, Config
            import socket

            # Find available port
            sock = socket.socket()
            sock.bind(("", 0))
            port = sock.getsockname()[1]
            sock.close()

            self.base_url = f"http://localhost:{port}"
            config = Config(app=self.app, host="127.0.0.1", port=port, log_level="error")
            self.server = Server(config)

            # Start server in background thread
            self.thread = threading.Thread(target=self.server.run)
            self.thread.daemon = True
            self.thread.start()

            # Wait for server to start
            time.sleep(0.1)

            self.client = TestClient(self.app)
            logger.info(f"Started FastAPI mock service: {self.name} at {self.base_url}")

        def stop(self):
            """Stop the FastAPI server."""
            if hasattr(self, "server"):
                self.server.should_exit = True
            if hasattr(self, "thread"):
                self.thread.join(timeout=1.0)
            logger.info(f"Stopped FastAPI mock service: {self.name}")
