"""Tests for distributed processing engine."""

import pytest
from openeval.engine.distributed import (
    DistributedEngine,
    WorkerTask,
    LoadBalancer,
    ClusterManager,
)


class TestDistributedEngine:
    """Test distributed processing engine."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initialization."""
        engine = DistributedEngine(num_workers=2)
        assert engine.num_workers == 2
        assert not engine.running
        assert len(engine.workers) == 0

    @pytest.mark.asyncio
    async def test_engine_start_stop(self):
        """Test engine start and stop."""
        engine = DistributedEngine(num_workers=2)
        await engine.start()
        assert engine.running
        assert len(engine.workers) == 2

        await engine.stop()
        assert not engine.running

    @pytest.mark.asyncio
    async def test_task_submission(self):
        """Test task submission and processing."""
        engine = DistributedEngine(num_workers=1)
        await engine.start()

        task = WorkerTask(task_id="test-1", data="test data")
        task_id = await engine.submit_task(task)
        assert task_id == "test-1"

        # Wait for result
        result = await engine.get_result("test-1", timeout=5.0)
        assert result is not None
        assert result.task_id == "test-1"
        assert result.result == "Processed: test data"
        assert result.worker_id == "worker-0"
        assert result.error is None
        assert result.processing_time > 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_task_error_handling(self):
        """Test error handling in task processing."""
        engine = DistributedEngine(num_workers=1)

        # Mock error in processing
        async def error_process(task):
            raise ValueError("Test error")

        engine._process_task = error_process
        await engine.start()

        task = WorkerTask(task_id="error-test", data="error data")
        await engine.submit_task(task)

        result = await engine.get_result("error-test", timeout=5.0)
        assert result is not None
        assert result.task_id == "error-test"
        assert result.result is None
        assert result.error == "Test error"

        await engine.stop()


class TestLoadBalancer:
    """Test load balancer."""

    @pytest.mark.asyncio
    async def test_load_balancer_round_robin(self):
        """Test round-robin load balancing."""
        engine1 = DistributedEngine(num_workers=1)
        engine2 = DistributedEngine(num_workers=1)
        balancer = LoadBalancer([engine1, engine2])

        await engine1.start()
        await engine2.start()

        # Submit multiple tasks
        tasks = []
        for i in range(4):
            task = WorkerTask(task_id=f"task-{i}", data=f"data-{i}")
            task_id = await balancer.submit_task(task)
            tasks.append(task_id)

        # Verify tasks were distributed
        results = []
        for task_id in tasks:
            result = await balancer.get_result(task_id)
            assert result is not None
            results.append(result.worker_id)

        # Should alternate between workers
        assert "worker-0" in results
        assert "worker-1" in results

        await engine1.stop()
        await engine2.stop()


class TestClusterManager:
    """Test cluster manager."""

    def test_cluster_initialization(self):
        """Test cluster manager initialization."""
        manager = ClusterManager()
        assert len(manager.nodes) == 0
        assert len(manager.engines) == 0

    def test_add_node(self):
        """Test adding nodes to cluster."""
        manager = ClusterManager()
        manager.add_node("localhost", 8080)
        manager.add_node("localhost", 8081)

        assert len(manager.nodes) == 2
        assert manager.nodes[0] == {"host": "localhost", "port": 8080}
        assert manager.nodes[1] == {"host": "localhost", "port": 8081}

    def test_create_engines(self):
        """Test creating engines for cluster."""
        manager = ClusterManager()
        manager.add_node("localhost", 8080)
        manager.add_node("localhost", 8081)

        balancer = manager.create_engines()
        assert isinstance(balancer, LoadBalancer)
        assert len(balancer.engines) == 2
