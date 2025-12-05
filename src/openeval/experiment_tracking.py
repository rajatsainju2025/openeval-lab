"""Experiment tracking integration for MLflow and Weights & Biases.

This module provides unified experiment tracking capabilities with support for
MLflow and Weights & Biases (wandb), enabling automatic logging of metrics,
parameters, and artifacts during evaluation runs.
"""

from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime

from .logging import get_logger

logger = get_logger(__name__)


# Optional imports for tracking backends
try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking."""

    backend: str = "none"  # none, mlflow, wandb, or both
    project_name: str = "openeval"
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    auto_log_params: bool = True
    auto_log_metrics: bool = True
    auto_log_artifacts: bool = True
    mlflow_tracking_uri: Optional[str] = None
    wandb_entity: Optional[str] = None


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking backends."""

    @abstractmethod
    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Start a new tracking run."""
        pass

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current tracking run."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        pass

    @abstractmethod
    def log_artifact(
        self, local_path: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """Log an artifact file."""
        pass

    @abstractmethod
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags for the run."""
        pass


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker."""

    def __init__(self, config: TrackingConfig):
        """Initialize MLflow tracker.

        Args:
            config: Tracking configuration
        """
        if not HAS_MLFLOW:
            raise ImportError("MLflow is not installed. Install it with: pip install mlflow")

        self.config = config
        self.active_run = None

        # Set tracking URI if provided
        if config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)

        # Set experiment
        if config.experiment_name:
            mlflow.set_experiment(config.experiment_name)

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Start MLflow run."""
        run_name = run_name or self.config.run_name
        combined_tags = {**self.config.tags, **(tags or {})}

        self.active_run = mlflow.start_run(run_name=run_name, tags=combined_tags)
        logger.info(f"Started MLflow run: {self.active_run.info.run_id}")

    def end_run(self, status: str = "FINISHED") -> None:
        """End MLflow run."""
        if self.active_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.active_run.info.run_id}")
            self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self, local_path: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """Log artifact to MLflow."""
        mlflow.log_artifact(str(local_path), artifact_path)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags in MLflow."""
        mlflow.set_tags(tags)

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""

    def __init__(self, config: TrackingConfig):
        """Initialize Wandb tracker.

        Args:
            config: Tracking configuration
        """
        if not HAS_WANDB:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")

        self.config = config
        self.run = None

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Start wandb run."""
        run_name = run_name or self.config.run_name
        combined_tags = {**self.config.tags, **(tags or {})}

        # Convert tags dict to list format for wandb
        tag_list = [f"{k}:{v}" for k, v in combined_tags.items()]

        self.run = wandb.init(
            project=self.config.project_name,
            name=run_name,
            entity=self.config.wandb_entity,
            tags=tag_list,
            reinit=True,
        )
        logger.info(f"Started wandb run: {self.run.id}")

    def end_run(self, status: str = "FINISHED") -> None:
        """End wandb run."""
        if self.run:
            self.run.finish(exit_code=0 if status == "FINISHED" else 1)
            logger.info(f"Ended wandb run: {self.run.id}")
            self.run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to wandb."""
        if self.run:
            self.run.config.update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_artifact(
        self, local_path: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """Log artifact to wandb."""
        if self.run:
            artifact = wandb.Artifact(
                name=Path(local_path).stem,
                type="dataset" if "data" in str(local_path) else "result",
            )
            artifact.add_file(str(local_path))
            self.run.log_artifact(artifact)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags in wandb."""
        if self.run:
            tag_list = [f"{k}:{v}" for k, v in tags.items()]
            self.run.tags = tag_list


class MultiTracker(ExperimentTracker):
    """Tracker that logs to multiple backends simultaneously."""

    def __init__(self, trackers: List[ExperimentTracker]):
        """Initialize multi-tracker.

        Args:
            trackers: List of tracker instances to use
        """
        self.trackers = trackers

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Start run on all trackers."""
        for tracker in self.trackers:
            try:
                tracker.start_run(run_name, tags)
            except Exception as e:
                logger.warning(f"Failed to start run on {tracker.__class__.__name__}: {e}")

    def end_run(self, status: str = "FINISHED") -> None:
        """End run on all trackers."""
        for tracker in self.trackers:
            try:
                tracker.end_run(status)
            except Exception as e:
                logger.warning(f"Failed to end run on {tracker.__class__.__name__}: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_params(params)
            except Exception as e:
                logger.warning(f"Failed to log params on {tracker.__class__.__name__}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                logger.warning(f"Failed to log metrics on {tracker.__class__.__name__}: {e}")

    def log_artifact(
        self, local_path: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """Log artifact to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_artifact(local_path, artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log artifact on {tracker.__class__.__name__}: {e}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on all trackers."""
        for tracker in self.trackers:
            try:
                tracker.set_tags(tags)
            except Exception as e:
                logger.warning(f"Failed to set tags on {tracker.__class__.__name__}: {e}")


class NoOpTracker(ExperimentTracker):
    """No-op tracker that does nothing (for when tracking is disabled)."""

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """No-op start run."""
        pass

    def end_run(self, status: str = "FINISHED") -> None:
        """No-op end run."""
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        """No-op log params."""
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """No-op log metrics."""
        pass

    def log_artifact(
        self, local_path: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """No-op log artifact."""
        pass

    def set_tags(self, tags: Dict[str, str]) -> None:
        """No-op set tags."""
        pass


def create_tracker(config: TrackingConfig) -> ExperimentTracker:
    """Create experiment tracker based on configuration.

    Args:
        config: Tracking configuration

    Returns:
        Initialized experiment tracker

    Raises:
        ValueError: If backend is unsupported
        ImportError: If required backend library is not installed
    """
    backend = config.backend.lower()

    if backend == "none":
        return NoOpTracker()

    elif backend == "mlflow":
        return MLflowTracker(config)

    elif backend == "wandb":
        return WandbTracker(config)

    elif backend == "both":
        trackers = []
        if HAS_MLFLOW:
            trackers.append(MLflowTracker(config))
        else:
            logger.warning("MLflow not installed, skipping MLflow tracking")

        if HAS_WANDB:
            trackers.append(WandbTracker(config))
        else:
            logger.warning("wandb not installed, skipping wandb tracking")

        if not trackers:
            logger.warning("No tracking backends available, using no-op tracker")
            return NoOpTracker()

        return MultiTracker(trackers)

    else:
        raise ValueError(
            f"Unsupported tracking backend: {backend}. "
            f"Supported backends: none, mlflow, wandb, both"
        )


class TrackingCallback:
    """Callback for automatic experiment tracking during evaluations."""

    def __init__(self, tracker: ExperimentTracker, config: TrackingConfig):
        """Initialize tracking callback.

        Args:
            tracker: Experiment tracker instance
            config: Tracking configuration
        """
        self.tracker = tracker
        self.config = config
        self.run_started = False

    def on_evaluation_start(
        self, eval_config: Dict[str, Any], run_name: Optional[str] = None
    ) -> None:
        """Called when evaluation starts.

        Args:
            eval_config: Evaluation configuration
            run_name: Optional run name
        """
        if not self.run_started:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = run_name or f"eval_{timestamp}"

            tags = {
                "task": eval_config.get("task", "unknown"),
                "dataset": eval_config.get("dataset", {}).get("type", "unknown"),
                "adapter": eval_config.get("adapter", {}).get("type", "unknown"),
            }

            self.tracker.start_run(run_name=run_name, tags=tags)
            self.run_started = True

            # Log configuration as parameters
            if self.config.auto_log_params:
                self.tracker.log_params(eval_config)

    def on_evaluation_end(self, metrics: Dict[str, float], status: str = "FINISHED") -> None:
        """Called when evaluation ends.

        Args:
            metrics: Final evaluation metrics
            status: Run status (FINISHED, FAILED, etc.)
        """
        if self.run_started:
            # Log final metrics
            if self.config.auto_log_metrics and metrics:
                self.tracker.log_metrics(metrics)

            self.tracker.end_run(status=status)
            self.run_started = False

    def on_batch_complete(self, batch_idx: int, batch_metrics: Dict[str, float]) -> None:
        """Called when a batch completes.

        Args:
            batch_idx: Batch index
            batch_metrics: Metrics for this batch
        """
        if self.run_started and self.config.auto_log_metrics:
            self.tracker.log_metrics(batch_metrics, step=batch_idx)

    def log_result_file(self, result_path: Union[str, Path]) -> None:
        """Log evaluation result file as artifact.

        Args:
            result_path: Path to result file
        """
        if self.run_started and self.config.auto_log_artifacts:
            self.tracker.log_artifact(result_path, artifact_path="results")


def auto_track(func: Callable, config: TrackingConfig, run_name: Optional[str] = None) -> Callable:
    """Decorator to automatically track function execution.

    Args:
        func: Function to wrap
        config: Tracking configuration
        run_name: Optional run name

    Returns:
        Wrapped function with tracking
    """

    def wrapper(*args, **kwargs):
        tracker = create_tracker(config)
        callback = TrackingCallback(tracker, config)

        # Start tracking
        eval_config = kwargs.get("config", {})
        callback.on_evaluation_start(eval_config, run_name)

        try:
            # Execute function
            result = func(*args, **kwargs)

            # End tracking with success
            metrics = result if isinstance(result, dict) else {}
            callback.on_evaluation_end(metrics, status="FINISHED")

            return result

        except Exception as e:
            # End tracking with failure
            callback.on_evaluation_end({}, status="FAILED")
            raise e

    return wrapper
