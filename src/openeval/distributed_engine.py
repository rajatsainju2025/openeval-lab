"""Distributed evaluation engine with load balancing and fault tolerance.

This module provides a distributed evaluation system that can scale across
multiple workers with automatic load balancing, fault tolerance, and failover.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from threading import Lock, Event
from typing import Any, Dict, List, Optional

from .core import Task, Dataset, Adapter, Metric
from .logging import get_logger

logger = get_logger(__name__)


class WorkerStatus(Enum):
    """Worker status enumeration."""

    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    OFFLINE = "offline"


@dataclass
class EvaluationJob:
    """Represents a single evaluation job."""

    job_id: str
    task: Task
    adapter: Adapter
    dataset_slice: List[Any]
    metrics: List[Metric]
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class Worker:
    """Represents a worker node in the distributed system."""

    worker_id: str
    status: WorkerStatus = WorkerStatus.IDLE
    current_job: Optional[str] = None
    last_heartbeat: Optional[float] = None
    total_jobs: int = 0
    failed_jobs: int = 0
    avg_job_time: float = 0.0
    capabilities: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = time.time()
        if self.capabilities is None:
            self.capabilities = {}


class LoadBalancer:
    """Intelligent load balancer for distributing evaluation jobs."""

    def __init__(self):
        self._lock = Lock()
        self._workers: Dict[str, Worker] = {}
        self._job_queue = Queue()
        self._completed_jobs: Dict[str, Any] = {}
        self._failed_jobs: Dict[str, EvaluationJob] = {}

    def register_worker(self, worker_id: str, capabilities: Optional[Dict] = None) -> Worker:
        """Register a new worker with the load balancer."""
        with self._lock:
            worker = Worker(worker_id=worker_id, capabilities=capabilities or {})
            self._workers[worker_id] = worker
            logger.info(f"Registered worker {worker_id} with capabilities: {capabilities}")
            return worker

    def unregister_worker(self, worker_id: str):
        """Unregister a worker from the load balancer."""
        with self._lock:
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker.status = WorkerStatus.OFFLINE

                # Reassign any current job
                if worker.current_job:
                    self._reassign_job(worker.current_job)

                del self._workers[worker_id]
                logger.info(f"Unregistered worker {worker_id}")

    def submit_job(self, job: EvaluationJob):
        """Submit a job to the evaluation queue."""
        self._job_queue.put(job)
        logger.info(f"Submitted job {job.job_id} to queue")

    def get_next_job(self, worker_id: str) -> Optional[EvaluationJob]:
        """Get the next job for a specific worker."""
        with self._lock:
            if worker_id not in self._workers:
                return None

            worker = self._workers[worker_id]
            if worker.status != WorkerStatus.IDLE:
                return None

            # Try to get a job from the queue
            try:
                job = self._job_queue.get_nowait()

                # Check if worker has required capabilities
                if self._worker_can_handle_job(worker, job):
                    worker.status = WorkerStatus.BUSY
                    worker.current_job = job.job_id
                    job.started_at = time.time()
                    return job
                else:
                    # Put job back in queue for other workers
                    self._job_queue.put(job)
                    return None

            except:
                return None

    def complete_job(self, worker_id: str, job_id: str, result: Any):
        """Mark a job as completed."""
        with self._lock:
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker.status = WorkerStatus.IDLE
                worker.current_job = None
                worker.total_jobs += 1
                worker.last_heartbeat = time.time()

                self._completed_jobs[job_id] = {
                    "result": result,
                    "worker_id": worker_id,
                    "completed_at": time.time(),
                }

                logger.info(f"Job {job_id} completed by worker {worker_id}")

    def fail_job(self, worker_id: str, job_id: str, error: Exception):
        """Mark a job as failed and handle retry logic."""
        with self._lock:
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker.status = WorkerStatus.IDLE
                worker.current_job = None
                worker.failed_jobs += 1
                worker.last_heartbeat = time.time()

                # Handle job retry logic
                if job_id in self._failed_jobs:
                    job = self._failed_jobs[job_id]
                    job.retry_count += 1

                    if job.retry_count < job.max_retries:
                        # Retry the job
                        logger.warning(
                            f"Retrying job {job_id} (attempt {job.retry_count}/{job.max_retries})"
                        )
                        self._job_queue.put(job)
                    else:
                        logger.error(
                            f"Job {job_id} failed permanently after {job.max_retries} attempts"
                        )

                logger.warning(f"Job {job_id} failed on worker {worker_id}: {error}")

    def get_worker_stats(self) -> Dict[str, Dict]:
        """Get statistics for all workers."""
        with self._lock:
            stats = {}
            for worker_id, worker in self._workers.items():
                stats[worker_id] = {
                    "status": worker.status.value,
                    "total_jobs": worker.total_jobs,
                    "failed_jobs": worker.failed_jobs,
                    "success_rate": (worker.total_jobs - worker.failed_jobs)
                    / max(worker.total_jobs, 1),
                    "avg_job_time": worker.avg_job_time,
                    "last_heartbeat": worker.last_heartbeat,
                }
            return stats

    def _worker_can_handle_job(self, worker: Worker, job: EvaluationJob) -> bool:
        """Check if a worker has the capabilities to handle a specific job."""
        # Simple capability matching - can be extended
        required_capabilities = getattr(job.task, "required_capabilities", {})

        if not worker.capabilities:
            return len(required_capabilities) == 0

        for capability, requirement in required_capabilities.items():
            if capability not in worker.capabilities:
                return False
            if not self._meets_requirement(worker.capabilities[capability], requirement):
                return False

        return True

    def _meets_requirement(self, capability_value: Any, requirement: Any) -> bool:
        """Check if a capability value meets a requirement."""
        if isinstance(requirement, str):
            return capability_value == requirement
        elif isinstance(requirement, (int, float)):
            return capability_value >= requirement
        elif isinstance(requirement, list):
            return capability_value in requirement
        return True

    def _reassign_job(self, job_id: str):
        """Reassign a failed job to the queue."""
        if job_id in self._failed_jobs:
            job = self._failed_jobs[job_id]
            job.retry_count += 1
            if job.retry_count < job.max_retries:
                self._job_queue.put(job)


class DistributedEvaluationEngine:
    """Main distributed evaluation engine."""

    def __init__(self, max_workers: int = 4, heartbeat_interval: float = 30.0):
        self.max_workers = max_workers
        self.heartbeat_interval = heartbeat_interval
        self.load_balancer = LoadBalancer()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.shutdown_event = Event()
        self._active_futures = []

    def register_worker(self, capabilities: Optional[Dict] = None) -> str:
        """Register a new worker and return its ID."""
        worker_id = str(uuid.uuid4())
        self.load_balancer.register_worker(worker_id, capabilities)
        return worker_id

    def evaluate_distributed(
        self,
        task: Task,
        dataset: Dataset,
        adapters: List[Adapter],
        metrics: List[Metric],
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """Perform distributed evaluation across multiple workers."""

        # Create evaluation jobs by batching dataset
        jobs = self._create_evaluation_jobs(task, dataset, adapters, metrics, batch_size)

        # Submit jobs to load balancer
        for job in jobs:
            self.load_balancer.submit_job(job)

        # Start worker threads
        futures = []
        for i in range(min(self.max_workers, len(jobs))):
            worker_id = self.register_worker({"type": "evaluation_worker"})
            future = self.executor.submit(self._worker_loop, worker_id)
            futures.append(future)
            self._active_futures.append(future)

        # Wait for all jobs to complete
        try:
            # Monitor progress
            total_jobs = len(jobs)
            completed = 0

            while completed < total_jobs and not self.shutdown_event.is_set():
                completed = len(self.load_balancer._completed_jobs)
                failed = len(self.load_balancer._failed_jobs)

                logger.info(f"Progress: {completed}/{total_jobs} completed, {failed} failed")
                time.sleep(1.0)

            # Collect results
            results = self._collect_results()

        finally:
            # Signal shutdown and cleanup
            self.shutdown_event.set()
            for future in futures:
                future.cancel()

        return results

    def _create_evaluation_jobs(
        self,
        task: Task,
        dataset: Dataset,
        adapters: List[Adapter],
        metrics: List[Metric],
        batch_size: int,
    ) -> List[EvaluationJob]:
        """Create evaluation jobs by batching the dataset."""
        jobs = []

        # Convert dataset to list for batching
        examples = list(dataset)

        # Create batches
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]

            for adapter in adapters:
                job_id = f"{uuid.uuid4()}"
                job = EvaluationJob(
                    job_id=job_id, task=task, adapter=adapter, dataset_slice=batch, metrics=metrics
                )
                jobs.append(job)

        return jobs

    def _worker_loop(self, worker_id: str):
        """Main worker loop for processing evaluation jobs."""
        logger.info(f"Worker {worker_id} started")

        while not self.shutdown_event.is_set():
            try:
                # Get next job
                job = self.load_balancer.get_next_job(worker_id)

                if job is None:
                    # No job available, wait a bit
                    time.sleep(0.1)
                    continue

                # Execute the job
                try:
                    result = self._execute_job(job)
                    self.load_balancer.complete_job(worker_id, job.job_id, result)

                except Exception as e:
                    logger.error(f"Job {job.job_id} failed: {e}")
                    self.load_balancer.fail_job(worker_id, job.job_id, e)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                break

        logger.info(f"Worker {worker_id} shutting down")

    def _execute_job(self, job: EvaluationJob) -> Dict[str, Any]:
        """Execute a single evaluation job."""
        results = []

        for example in job.dataset_slice:
            # Get model prediction
            prediction = job.adapter.generate(example.input)

            # Compute metrics
            metric_results = {}
            for metric in job.metrics:
                score = metric.compute(prediction, example.reference)
                metric_results[metric.name] = score

            results.append(
                {
                    "input": example.input,
                    "prediction": prediction,
                    "reference": example.reference,
                    "metrics": metric_results,
                }
            )

        execution_time = 0.0
        if job.started_at is not None:
            execution_time = time.time() - job.started_at

        return {
            "job_id": job.job_id,
            "adapter": str(job.adapter),
            "results": results,
            "execution_time": execution_time,
        }

    def _collect_results(self) -> Dict[str, Any]:
        """Collect and aggregate results from all completed jobs."""
        completed_jobs = self.load_balancer._completed_jobs

        # Group results by adapter
        adapter_results = {}

        for job_id, job_data in completed_jobs.items():
            result = job_data["result"]
            adapter_name = result["adapter"]

            if adapter_name not in adapter_results:
                adapter_results[adapter_name] = []

            adapter_results[adapter_name].extend(result["results"])

        # Aggregate metrics
        aggregated = {}
        for adapter_name, results in adapter_results.items():
            # Compute aggregate metrics
            metric_sums = {}
            total_examples = len(results)

            for result in results:
                for metric_name, score in result["metrics"].items():
                    if metric_name not in metric_sums:
                        metric_sums[metric_name] = 0
                    metric_sums[metric_name] += score

            # Calculate averages
            metric_averages = {name: total / total_examples for name, total in metric_sums.items()}

            aggregated[adapter_name] = {
                "total_examples": total_examples,
                "metrics": metric_averages,
                "raw_results": results,
            }

        return {
            "aggregated_results": aggregated,
            "worker_stats": self.load_balancer.get_worker_stats(),
            "total_jobs": len(completed_jobs),
            "failed_jobs": len(self.load_balancer._failed_jobs),
        }

    def shutdown(self):
        """Gracefully shutdown the distributed engine."""
        logger.info("Shutting down distributed evaluation engine")
        self.shutdown_event.set()
        self.executor.shutdown(wait=True)

        # Cancel any remaining futures
        for future in self._active_futures:
            future.cancel()

        logger.info("Distributed evaluation engine shutdown complete")


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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    capabilities: List[str] = field(
        default_factory=list
    )  # e.g., ["gpu", "cpu", "memory_intensive"]


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

    def get_available_node(
        self, task_requirements: Optional[List[str]] = None
    ) -> Optional[WorkerNode]:
        """Get the best available node for a task based on current load and requirements."""
        with self._lock:
            available_nodes = [
                node
                for node in self.nodes.values()
                if node.status != "offline" and node.active_tasks < node.capacity
            ]

            if not available_nodes:
                return None

            # Filter by requirements if specified
            if task_requirements:
                available_nodes = [
                    node
                    for node in available_nodes
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

        logger.info(
            f"Worker allocation adjustment recommended: {self.max_workers} -> {new_max_workers}"
        )
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
                        logger.warning(
                            f"Task {task_id} failed, retrying ({task.retries} retries left): {e}"
                        )
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
                    "active_processes": len(psutil.pids()),
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
            "active_nodes": len(
                [n for n in self.load_balancer.nodes.values() if n.status != "offline"]
            ),
            "system_load": system_load,
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
            "checkpoint": self._checkpoint_and_restart,
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
            task
            for task in self.processor.tasks.values()
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
