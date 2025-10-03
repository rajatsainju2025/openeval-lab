"""
Distributed Processing Module for OpenEval

This module provides distributed processing capabilities for large-scale evaluations,
enabling parallel execution across multiple nodes/workers with load balancing,
fault tolerance, and result aggregation.
"""

import asyncio
import logging
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from queue import Queue

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
    logger.warning("psutil not available, system monitoring disabled")


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    node_id: str
    host: str
    port: int
    capacity: int = 4  # Max concurrent tasks
    active_tasks: int = 0
    status: str = "idle"  # idle, busy, offline
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)  # e.g., ["gpu", "cpu", "memory_intensive"]


@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 1
    timeout: Optional[float] = None
    retries: int = 3
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class LoadBalancer:
    """Load balancer for distributing tasks across worker nodes."""

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.nodes: Dict[str, WorkerNode] = {}
        self.task_queue: Queue = Queue()
        self._lock = threading.Lock()

    def register_node(self, node: WorkerNode) -> None:
        """Register a new worker node."""
        with self._lock:
            self.nodes[node.node_id] = node
            logger.info(f"Registered worker node: {node.node_id}")

    def unregister_node(self, node_id: str) -> None:
        """Unregister a worker node."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Unregistered worker node: {node_id}")

    def get_available_node(self, task_requirements: Optional[List[str]] = None) -> Optional[WorkerNode]:
        """Get the best available node for a task based on current load and requirements."""
        with self._lock:
            available_nodes = [
                node for node in self.nodes.values()
                if node.status != "offline" and node.active_tasks < node.capacity
            ]

            if not available_nodes:
                return None

            # Filter by requirements if specified
            if task_requirements:
                available_nodes = [
                    node for node in available_nodes
                    if all(req in node.capabilities for req in task_requirements)
                ]

            if not available_nodes:
                return None

            # Apply load balancing strategy
            if self.strategy == "round_robin":
                return min(available_nodes, key=lambda n: n.active_tasks)
            elif self.strategy == "least_loaded":
                return min(available_nodes, key=lambda n: n.active_tasks / n.capacity)
            elif self.strategy == "most_capable":
                return max(available_nodes, key=lambda n: n.capacity - n.active_tasks)
            else:
                return available_nodes[0]

    def update_node_load(self, node_id: str, active_tasks: int) -> None:
        """Update the load of a specific node."""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].active_tasks = active_tasks
                self.nodes[node_id].last_heartbeat = time.time()


class DistributedProcessor:
    """Main distributed processing coordinator."""

    def __init__(self, max_workers: Optional[int] = None, enable_gpu: bool = False):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.enable_gpu = enable_gpu
        self.load_balancer = LoadBalancer(strategy="least_loaded")
        self.tasks: Dict[str, DistributedTask] = {}
        self.results: Dict[str, Any] = {}
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self._shutdown = False
        self._task_lock = threading.Lock()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()

    def _monitor_system(self) -> None:
        """Monitor system resources and adjust worker allocation."""
        while not self._shutdown:
            try:
                if HAS_PSUTIL and psutil:
                    # Monitor CPU and memory usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent

                    # Adjust worker allocation based on system load
                    if cpu_percent > 80 or memory_percent > 85:
                        # High load - reduce active workers
                        self._adjust_worker_allocation(0.7)
                    elif cpu_percent < 30 and memory_percent < 50:
                        # Low load - can increase workers
                        self._adjust_worker_allocation(1.2)
                else:
                    # Fallback without psutil
                    time.sleep(30)

                time.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(30)

    def _adjust_worker_allocation(self, factor: float) -> None:
        """Adjust the number of active workers based on system load."""
        # Note: ProcessPoolExecutor doesn't support dynamic worker adjustment
        # This is a placeholder for future implementation with custom worker pool
        new_max_workers = int(self.max_workers * factor)
        new_max_workers = max(1, min(new_max_workers, multiprocessing.cpu_count() * 2))

        logger.info(f"Worker allocation adjustment recommended: {self.max_workers} -> {new_max_workers}")
        # In a real implementation, you would need to shutdown and recreate the executor

    async def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed processing."""
        with self._task_lock:
            self.tasks[task.task_id] = task

        # Find available node
        node = self.load_balancer.get_available_node()
        if node:
            task.assigned_node = node.node_id
            node.active_tasks += 1
            task.status = "running"

            # Submit to executor
            future = self.executor.submit(self._execute_task, task)
            future.add_done_callback(lambda f: self._task_completed(f, task.task_id))

        return task.task_id

    def _execute_task(self, task: DistributedTask) -> Any:
        """Execute a task in a worker process."""
        start_time = time.time()

        try:
            # Generic task execution - can be extended for specific task types
            if task.task_type == "evaluation":
                # Use a generic evaluation function
                result = self._execute_evaluation_task(task.payload)
            elif task.task_type == "metric_computation":
                result = self._execute_metric_task(task.payload)
            elif task.task_type == "data_processing":
                result = self._execute_data_processing_task(task.payload)
            else:
                # Generic task execution
                result = self._execute_generic_task(task.payload)

            task.execution_time = time.time() - start_time
            return result

        except Exception as e:
            task.error = str(e)
            task.execution_time = time.time() - start_time
            raise

    def _execute_evaluation_task(self, payload: Dict[str, Any]) -> Any:
        """Execute an evaluation task."""
        # Placeholder for evaluation logic
        # In a real implementation, this would import and use the evaluation engine
        logger.info(f"Executing evaluation task with payload: {payload}")
        return {"status": "completed", "result": "evaluation_result"}

    def _execute_metric_task(self, payload: Dict[str, Any]) -> Any:
        """Execute a metric computation task."""
        # Placeholder for metric computation
        logger.info(f"Executing metric task with payload: {payload}")
        return {"status": "completed", "metrics": {}}

    def _execute_data_processing_task(self, payload: Dict[str, Any]) -> Any:
        """Execute a data processing task."""
        # Placeholder for data processing
        logger.info(f"Executing data processing task with payload: {payload}")
        return {"status": "completed", "processed_data": []}

    def _execute_generic_task(self, payload: Dict[str, Any]) -> Any:
        """Execute a generic task."""
        logger.info(f"Executing generic task with payload: {payload}")
        return {"status": "completed", "result": payload}

    def _task_completed(self, future, task_id: str) -> None:
        """Handle task completion."""
        with self._task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]

                try:
                    result = future.result()
                    task.result = result
                    task.status = "completed"
                    self.results[task_id] = result

                    # Update node load
                    if task.assigned_node:
                        node = self.load_balancer.nodes.get(task.assigned_node)
                        if node:
                            node.active_tasks = max(0, node.active_tasks - 1)

                    logger.info(f"Task {task_id} completed successfully")

                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)

                    # Retry logic
                    if task.retries > 0:
                        task.retries -= 1
                        task.status = "pending"
                        logger.warning(f"Task {task_id} failed, retrying ({task.retries} retries left): {e}")
                        # Re-submit task
                        asyncio.create_task(self.submit_task(task))
                    else:
                        logger.error(f"Task {task_id} failed permanently: {e}")

    async def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """Get the status of a task."""
        with self._task_lock:
            return self.tasks.get(task_id)

    async def get_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a completed task."""
        return self.results.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self._task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status in ["pending", "running"]:
                    task.status = "cancelled"
                    return True
        return False

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        with self._task_lock:
            total_tasks = len(self.tasks)
            completed_tasks = sum(1 for t in self.tasks.values() if t.status == "completed")
            failed_tasks = sum(1 for t in self.tasks.values() if t.status == "failed")
            running_tasks = sum(1 for t in self.tasks.values() if t.status == "running")

        system_load = {}
        if HAS_PSUTIL and psutil:
            try:
                system_load = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "active_processes": len(psutil.pids())
                }
            except Exception as e:
                logger.warning(f"Failed to get system stats: {e}")
                system_load = {"error": "Failed to get system stats"}
        else:
            system_load = {"note": "psutil not available"}

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "success_rate": (completed_tasks / total_tasks) if total_tasks > 0 else 0,
            "active_nodes": len([n for n in self.load_balancer.nodes.values() if n.status != "offline"]),
            "system_load": system_load
        }

    def shutdown(self) -> None:
        """Shutdown the distributed processor."""
        self._shutdown = True
        self.executor.shutdown(wait=True)
        logger.info("Distributed processor shutdown complete")


class FaultToleranceManager:
    """Manages fault tolerance and recovery for distributed tasks."""

    def __init__(self, processor: DistributedProcessor):
        self.processor = processor
        self.failed_nodes: Dict[str, int] = {}
        self.recovery_strategies = {
            "retry": self._retry_failed_task,
            "reassign": self._reassign_to_healthy_node,
            "checkpoint": self._checkpoint_and_restart
        }

    def handle_node_failure(self, node_id: str) -> None:
        """Handle failure of a worker node."""
        logger.warning(f"Node failure detected: {node_id}")

        # Mark node as offline
        if node_id in self.processor.load_balancer.nodes:
            self.processor.load_balancer.nodes[node_id].status = "offline"

        # Count failures
        self.failed_nodes[node_id] = self.failed_nodes.get(node_id, 0) + 1

        # Reassign running tasks from failed node
        failed_tasks = [
            task for task in self.processor.tasks.values()
            if task.assigned_node == node_id and task.status == "running"
        ]

        for task in failed_tasks:
            logger.info(f"Reassigning failed task {task.task_id} from node {node_id}")
            task.assigned_node = None
            task.status = "pending"
            # Re-submit task
            asyncio.create_task(self.processor.submit_task(task))

    def _retry_failed_task(self, task: DistributedTask) -> None:
        """Retry a failed task."""
        if task.retries > 0:
            task.retries -= 1
            task.status = "pending"
            task.error = None
            asyncio.create_task(self.processor.submit_task(task))

    def _reassign_to_healthy_node(self, task: DistributedTask) -> None:
        """Reassign task to a healthy node."""
        task.assigned_node = None
        task.status = "pending"
        asyncio.create_task(self.processor.submit_task(task))

    def _checkpoint_and_restart(self, task: DistributedTask) -> None:
        """Checkpoint task state and restart from checkpoint."""
        # This would require task-specific checkpointing logic
        logger.info(f"Checkpointing task {task.task_id} for restart")
        # Implementation would depend on task type and checkpoint requirements
        self._reassign_to_healthy_node(task)