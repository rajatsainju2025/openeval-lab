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
from .enhanced_logging import get_logger

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
