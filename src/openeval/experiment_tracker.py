"""
Experiment Tracking and Management System for OpenEval Lab

This module provides comprehensive experiment tracking, versioning, and management
capabilities for evaluation experiments, ensuring reproducibility and organization.
"""

from __future__ import annotations

import json
import uuid
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import shutil
import tempfile

from .enhanced_logging import get_logger

logger = get_logger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentPriority(Enum):
    """Priority levels for experiments."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    task: str
    model: Dict[str, Any]
    dataset: Dict[str, Any]
    metrics: List[str]
    evaluation: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task": self.task,
            "model": self.model,
            "dataset": self.dataset,
            "metrics": self.metrics,
            "evaluation": self.evaluation,
            "hyperparameters": self.hyperparameters,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentResult:
    """Result of an experiment."""
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # filename -> path
    logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    duration: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "logs": self.logs,
            "error_message": self.error_message,
            "duration": self.duration
        }


@dataclass
class Experiment:
    """Represents an evaluation experiment."""
    id: str
    name: str
    description: Optional[str] = None
    config: ExperimentConfig = field(default_factory=lambda: ExperimentConfig("", {}, {}, []))
    status: ExperimentStatus = ExperimentStatus.CREATED
    priority: ExperimentPriority = ExperimentPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[ExperimentResult] = None
    parent_experiment_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_finished(self) -> bool:
        """Check if experiment is finished."""
        return self.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result.to_dict() if self.result else None,
            "parent_experiment_id": self.parent_experiment_id,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Experiment:
        """Create from dictionary."""
        # Convert nested objects
        config = ExperimentConfig.from_dict(data["config"])
        status = ExperimentStatus(data["status"])
        priority = ExperimentPriority(data["priority"])

        # Parse timestamps
        created_at = datetime.fromisoformat(data["created_at"])
        started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None

        # Parse result if present
        result = None
        if data.get("result"):
            result = ExperimentResult(**data["result"])

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            config=config,
            status=status,
            priority=priority,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            result=result,
            parent_experiment_id=data.get("parent_experiment_id"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


class ExperimentTracker:
    """
    Comprehensive experiment tracking and management system.
    """

    def __init__(self, experiments_dir: Optional[Path] = None):
        self.experiments_dir = experiments_dir or Path("experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.metadata_dir = self.experiments_dir / "metadata"
        self.artifacts_dir = self.experiments_dir / "artifacts"
        self.logs_dir = self.experiments_dir / "logs"

        for dir_path in [self.metadata_dir, self.artifacts_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._experiment_cache: Dict[str, Experiment] = {}

    def create_experiment(
        self,
        name: str,
        config: ExperimentConfig,
        description: Optional[str] = None,
        priority: ExperimentPriority = ExperimentPriority.MEDIUM,
        tags: Optional[List[str]] = None,
        parent_experiment_id: Optional[str] = None
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            config: Experiment configuration
            description: Optional description
            priority: Experiment priority
            tags: Optional tags
            parent_experiment_id: ID of parent experiment

        Returns:
            Created Experiment object
        """
        experiment_id = str(uuid.uuid4())

        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            config=config,
            priority=priority,
            tags=tags or [],
            parent_experiment_id=parent_experiment_id
        )

        # Save to disk
        self._save_experiment(experiment)

        # Cache
        self._experiment_cache[experiment_id] = experiment

        logger.info(f"Created experiment '{name}' with ID {experiment_id}")
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        # Check cache first
        if experiment_id in self._experiment_cache:
            return self._experiment_cache[experiment_id]

        # Load from disk
        metadata_file = self.metadata_dir / f"{experiment_id}.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                experiment = Experiment.from_dict(data)
                self._experiment_cache[experiment_id] = experiment
                return experiment
            except Exception as e:
                logger.error(f"Failed to load experiment {experiment_id}: {e}")

        return None

    def update_experiment_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        result: Optional[ExperimentResult] = None
    ) -> bool:
        """
        Update experiment status.

        Args:
            experiment_id: Experiment ID
            status: New status
            result: Experiment result (if completed)

        Returns:
            True if update was successful
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False

        old_status = experiment.status
        experiment.status = status

        # Update timestamps
        if status == ExperimentStatus.RUNNING and not experiment.started_at:
            experiment.started_at = datetime.now()
        elif status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]:
            experiment.completed_at = datetime.now()
            if result:
                experiment.result = result

        # Save changes
        self._save_experiment(experiment)

        logger.info(f"Updated experiment {experiment_id} status: {old_status.value} -> {status.value}")
        return True

    def log_experiment_message(self, experiment_id: str, message: str, level: str = "INFO") -> None:
        """
        Log a message for an experiment.

        Args:
            experiment_id: Experiment ID
            message: Log message
            level: Log level
        """
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"

        # Save to log file
        log_file = self.logs_dir / f"{experiment_id}.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

        # Also add to experiment result if it exists
        experiment = self.get_experiment(experiment_id)
        if experiment and experiment.result:
            experiment.result.logs.append(log_entry)
            self._save_experiment(experiment)

    def add_experiment_artifact(
        self,
        experiment_id: str,
        artifact_name: str,
        artifact_path: Union[str, Path]
    ) -> bool:
        """
        Add an artifact to an experiment.

        Args:
            artifact_name: Name of the artifact
            artifact_path: Path to the artifact file

        Returns:
            True if artifact was added successfully
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment or not experiment.result:
            return False

        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            return False

        # Copy artifact to experiments directory
        artifact_dest = self.artifacts_dir / experiment_id / artifact_name
        artifact_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_path, artifact_dest)

        # Update experiment
        experiment.result.artifacts[artifact_name] = str(artifact_dest)
        self._save_experiment(experiment)

        logger.info(f"Added artifact '{artifact_name}' to experiment {experiment_id}")
        return True

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Experiment]:
        """
        List experiments with optional filtering.

        Args:
            status: Filter by status
            tags: Filter by tags (experiments must have all specified tags)
            limit: Maximum number of experiments to return

        Returns:
            List of matching experiments
        """
        experiments = []

        # Load all experiments
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                experiment = Experiment.from_dict(data)

                # Apply filters
                if status and experiment.status != status:
                    continue
                if tags and not all(tag in experiment.tags for tag in tags):
                    continue

                experiments.append(experiment)

            except Exception as e:
                logger.warning(f"Failed to load experiment from {metadata_file}: {e}")

        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x.created_at, reverse=True)

        if limit:
            experiments = experiments[:limit]

        return experiments

    def search_experiments(
        self,
        query: str,
        fields: Optional[List[str]] = None
    ) -> List[Experiment]:
        """
        Search experiments by text query.

        Args:
            query: Search query (case-insensitive)
            fields: Fields to search in (if None, search in name, description, tags)

        Returns:
            List of matching experiments
        """
        if fields is None:
            fields = ["name", "description", "tags"]

        query_lower = query.lower()
        matching_experiments = []

        for experiment in self.list_experiments():
            match = False

            for field in fields:
                if field == "name" and query_lower in experiment.name.lower():
                    match = True
                    break
                elif field == "description" and experiment.description and query_lower in experiment.description.lower():
                    match = True
                    break
                elif field == "tags" and any(query_lower in tag.lower() for tag in experiment.tags):
                    match = True
                    break

            if match:
                matching_experiments.append(experiment)

        return matching_experiments

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment and all its artifacts.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deletion was successful
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False

        # Remove files
        metadata_file = self.metadata_dir / f"{experiment_id}.json"
        log_file = self.logs_dir / f"{experiment_id}.log"
        artifacts_dir = self.artifacts_dir / experiment_id

        for file_path in [metadata_file, log_file]:
            if file_path.exists():
                file_path.unlink()

        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)

        # Remove from cache
        self._experiment_cache.pop(experiment_id, None)

        logger.info(f"Deleted experiment {experiment_id}")
        return True

    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        return {
            "id": experiment.id,
            "name": experiment.name,
            "status": experiment.status.value,
            "priority": experiment.priority.value,
            "created_at": experiment.created_at.isoformat(),
            "duration": experiment.duration,
            "has_result": experiment.result is not None,
            "metrics_count": len(experiment.result.metrics) if experiment.result else 0,
            "artifacts_count": len(experiment.result.artifacts) if experiment.result else 0,
            "tags": experiment.tags
        }

    def export_experiments(
        self,
        experiment_ids: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        format: str = "json"
    ) -> Path:
        """
        Export experiments to a file.

        Args:
            experiment_ids: Specific experiment IDs to export (if None, export all)
            output_path: Output path (if None, auto-generate)
            format: Export format ('json', 'csv')

        Returns:
            Path to exported file
        """
        if experiment_ids:
            experiments = [self.get_experiment(eid) for eid in experiment_ids]
            experiments = [e for e in experiments if e is not None]
        else:
            experiments = self.list_experiments()

        if not output_path:
            timestamp = int(datetime.now().timestamp())
            output_path = self.experiments_dir / f"experiments_export_{timestamp}.{format}"

        if format == "json":
            data = [exp.to_dict() for exp in experiments]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            try:
                import pandas as pd
                # Convert to flat structure for CSV
                rows = []
                for exp in experiments:
                    row = {
                        "id": exp.id,
                        "name": exp.name,
                        "description": exp.description,
                        "status": exp.status.value,
                        "priority": exp.priority.value,
                        "created_at": exp.created_at.isoformat(),
                        "duration": exp.duration,
                        "tags": ",".join(exp.tags)
                    }

                    # Add metrics if available
                    if exp.result:
                        for metric_name, metric_value in exp.result.metrics.items():
                            row[f"metric_{metric_name}"] = metric_value

                    rows.append(row)

                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)

            except ImportError:
                raise ImportError("pandas required for CSV export")

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported {len(experiments)} experiments to {output_path}")
        return output_path

    def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to disk."""
        metadata_file = self.metadata_dir / f"{experiment.id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(experiment.to_dict(), f, indent=2, ensure_ascii=False)

    def get_experiment_lineage(self, experiment_id: str) -> List[Experiment]:
        """
        Get the lineage of an experiment (experiment and its ancestors).

        Args:
            experiment_id: Experiment ID

        Returns:
            List of experiments in lineage order (oldest first)
        """
        lineage = []
        current_id = experiment_id

        while current_id:
            experiment = self.get_experiment(current_id)
            if not experiment:
                break

            lineage.insert(0, experiment)  # Insert at beginning to maintain order
            current_id = experiment.parent_experiment_id

        return lineage

    def clone_experiment(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        config_changes: Optional[Dict[str, Any]] = None
    ) -> Optional[Experiment]:
        """
        Clone an existing experiment with optional modifications.

        Args:
            experiment_id: ID of experiment to clone
            name: New name for cloned experiment
            config_changes: Changes to apply to configuration

        Returns:
            Cloned experiment or None if original not found
        """
        original = self.get_experiment(experiment_id)
        if not original:
            return None

        # Create new config
        new_config = ExperimentConfig.from_dict(original.config.to_dict())
        if config_changes:
            # Apply changes (simplified - in practice you'd need more sophisticated merging)
            for key, value in config_changes.items():
                if hasattr(new_config, key):
                    setattr(new_config, key, value)

        # Create cloned experiment
        clone_name = name or f"{original.name} (clone)"
        clone = self.create_experiment(
            name=clone_name,
            config=new_config,
            description=f"Clone of experiment {experiment_id}",
            priority=original.priority,
            tags=original.tags.copy(),
            parent_experiment_id=experiment_id
        )

        logger.info(f"Cloned experiment {experiment_id} to {clone.id}")
        return clone


def create_experiment_tracker(experiments_dir: Optional[Path] = None) -> ExperimentTracker:
    """Create an experiment tracker instance."""
    return ExperimentTracker(experiments_dir)


def quick_experiment(
    name: str,
    task: str,
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    metrics: List[str],
    experiments_dir: Optional[Path] = None
) -> Experiment:
    """
    Create a quick experiment with minimal configuration.

    Args:
        name: Experiment name
        task: Evaluation task
        model_config: Model configuration
        dataset_config: Dataset configuration
        metrics: List of metrics
        experiments_dir: Experiments directory

    Returns:
        Created experiment
    """
    config = ExperimentConfig(
        task=task,
        model=model_config,
        dataset=dataset_config,
        metrics=metrics
    )

    tracker = ExperimentTracker(experiments_dir)
    return tracker.create_experiment(name, config)