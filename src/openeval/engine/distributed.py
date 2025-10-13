"""Distributed processing and horizontal scaling."""

from __future__ import annotations

import asyncio
import multiprocessing
from typing import List, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class WorkerTask:
    """Task for distributed processing."""

    task_id: str
    data: Any
    priority: int = 1


@dataclass
class WorkerResult:
    """Result from distributed processing."""

    task_id: str
    result: Any
    worker_id: str
    processing_time: float
    error: Optional[str] = None


class DistributedEngine:
    """Distributed processing engine for horizontal scaling."""

    def __init__(self, num_workers: int = multiprocessing.cpu_count()):
        self.num_workers = num_workers
        self.workers = []
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.running = False

    async def start(self):
        """Start the distributed engine."""
        self.running = True
        self.workers = [asyncio.create_task(self._worker_loop(i)) for i in range(self.num_workers)]

    async def stop(self):
        """Stop the distributed engine."""
        self.running = False
        # Send stop signals
        for _ in range(self.num_workers):
            await self.task_queue.put(None)

        await asyncio.gather(*self.workers, return_exceptions=True)

    async def submit_task(self, task: WorkerTask) -> str:
        """Submit a task for processing."""
        await self.task_queue.put(task)
        return task.task_id

    async def get_result(self, task_id: str, timeout: float = 30.0) -> Optional[WorkerResult]:
        """Get result for a task."""
        try:
            while True:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=timeout)
                if result.task_id == task_id:
                    return result
                # Put back if not our task
                await self.result_queue.put(result)
        except asyncio.TimeoutError:
            return None

    async def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        worker_name = f"worker-{worker_id}"
        print(f"Starting {worker_name}")

        while self.running:
            try:
                task = await self.task_queue.get()
                if task is None:  # Stop signal
                    break

                start_time = time.time()
                try:
                    # Process task (placeholder)
                    result_data = await self._process_task(task)
                    processing_time = time.time() - start_time

                    result = WorkerResult(
                        task_id=task.task_id,
                        result=result_data,
                        worker_id=worker_name,
                        processing_time=processing_time,
                    )
                    await self.result_queue.put(result)

                except Exception as e:
                    processing_time = time.time() - start_time
                    result = WorkerResult(
                        task_id=task.task_id,
                        result=None,
                        worker_id=worker_name,
                        processing_time=processing_time,
                        error=str(e),
                    )
                    await self.result_queue.put(result)

            except Exception as e:
                print(f"Worker {worker_name} error: {e}")

        print(f"Stopping {worker_name}")

    async def _process_task(self, task: WorkerTask) -> Any:
        """Process a single task."""
        # Placeholder processing logic
        await asyncio.sleep(0.1)  # Simulate work
        return f"Processed: {task.data}"


class LoadBalancer:
    """Load balancer for distributed tasks."""

    def __init__(self, engines: List[DistributedEngine]):
        self.engines = engines
        self.current_engine = 0

    async def submit_task(self, task: WorkerTask) -> str:
        """Submit task using round-robin load balancing."""
        engine = self.engines[self.current_engine]
        self.current_engine = (self.current_engine + 1) % len(self.engines)
        return await engine.submit_task(task)

    async def get_result(self, task_id: str) -> Optional[WorkerResult]:
        """Get result from any engine."""
        for engine in self.engines:
            result = await engine.get_result(task_id, timeout=1.0)
            if result:
                return result
        return None


class ClusterManager:
    """Manager for distributed clusters."""

    def __init__(self):
        self.nodes = []
        self.engines = []

    def add_node(self, host: str, port: int):
        """Add a cluster node."""
        self.nodes.append({"host": host, "port": port})

    def create_engines(self) -> LoadBalancer:
        """Create engines for all nodes."""
        # For now, create local engines
        engines = [DistributedEngine() for _ in self.nodes]
        return LoadBalancer(engines)


__all__ = ["WorkerTask", "WorkerResult", "DistributedEngine", "LoadBalancer", "ClusterManager"]
