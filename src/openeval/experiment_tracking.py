"""Experiment tracking and reproducibility system for OpenEval Lab."""

import json
import time
import hashlib
import platform
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict

from .logging import get_logger


@dataclass
class ExperimentEnvironment:
    """Environment information for experiment reproducibility."""
    
    python_version: str
    platform: str
    hostname: str
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    installed_packages: Dict[str, str] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ExperimentParameters:
    """Experiment configuration and parameters."""
    
    spec_file: str
    dataset_name: str
    adapter_name: str
    task_name: str
    metric_names: List[str]
    random_seed: Optional[int] = None
    concurrency: int = 1
    cache_enabled: bool = True
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentMetrics:
    """Aggregated metrics from experiment runs."""
    
    primary_score: float
    all_scores: Dict[str, float]
    runtime_seconds: float
    samples_processed: int
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    throughput_samples_per_sec: float = 0.0
    
    def __post_init__(self):
        if self.runtime_seconds > 0 and self.samples_processed > 0:
            self.throughput_samples_per_sec = self.samples_processed / self.runtime_seconds


@dataclass
class ExperimentRun:
    """Complete experiment run record."""
    
    experiment_id: str
    run_id: str
    name: str
    description: str
    parameters: ExperimentParameters
    environment: ExperimentEnvironment
    metrics: ExperimentMetrics
    artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_name -> file_path
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    status: str = "completed"  # running, completed, failed
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ExperimentTracker:
    """Comprehensive experiment tracking and reproducibility system."""
    
    def __init__(self, experiments_dir: Optional[Path] = None):
        """Initialize experiment tracker."""
        self.logger = get_logger()
        self.experiments_dir = experiments_dir or Path("experiments")
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Current experiment context
        self.current_experiment: Optional[ExperimentRun] = None
        self.start_time: Optional[float] = None
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a new experiment and return its ID."""
        experiment_id = self._generate_experiment_id(name)
        run_id = self._generate_run_id()
        
        # Capture environment
        environment = self._capture_environment()
        
        # Initialize experiment run
        self.current_experiment = ExperimentRun(
            experiment_id=experiment_id,
            run_id=run_id,
            name=name,
            description=description,
            parameters=ExperimentParameters(
                spec_file="",
                dataset_name="",
                adapter_name="",
                task_name="",
                metric_names=[]
            ),
            environment=environment,
            metrics=ExperimentMetrics(
                primary_score=0.0,
                all_scores={},
                runtime_seconds=0.0,
                samples_processed=0
            ),
            tags=tags or [],
            status="running"
        )
        
        self.start_time = time.time()
        self.logger.info(f"Created experiment: {experiment_id} (run: {run_id})")
        
        return experiment_id
    
    def log_parameters(self, **kwargs) -> None:
        """Log experiment parameters."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call create_experiment() first.")
        
        # Update known parameters
        params = self.current_experiment.parameters
        
        if "spec_file" in kwargs:
            params.spec_file = kwargs["spec_file"]
        if "dataset_name" in kwargs:
            params.dataset_name = kwargs["dataset_name"]
        if "adapter_name" in kwargs:
            params.adapter_name = kwargs["adapter_name"]
        if "task_name" in kwargs:
            params.task_name = kwargs["task_name"]
        if "metric_names" in kwargs:
            params.metric_names = kwargs["metric_names"]
        if "random_seed" in kwargs:
            params.random_seed = kwargs["random_seed"]
        if "concurrency" in kwargs:
            params.concurrency = kwargs["concurrency"]
        if "cache_enabled" in kwargs:
            params.cache_enabled = kwargs["cache_enabled"]
        
        # Store additional parameters
        for key, value in kwargs.items():
            if key not in ["spec_file", "dataset_name", "adapter_name", "task_name", 
                          "metric_names", "random_seed", "concurrency", "cache_enabled"]:
                params.additional_params[key] = value
        
        self.logger.debug(f"Logged parameters: {kwargs}")
    
    def log_metrics(self, **kwargs) -> None:
        """Log experiment metrics."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call create_experiment() first.")
        
        metrics = self.current_experiment.metrics
        
        # Update known metrics
        if "primary_score" in kwargs:
            metrics.primary_score = kwargs["primary_score"]
        if "samples_processed" in kwargs:
            metrics.samples_processed = kwargs["samples_processed"]
        if "cache_hits" in kwargs:
            metrics.cache_hits = kwargs["cache_hits"]
        if "cache_misses" in kwargs:
            metrics.cache_misses = kwargs["cache_misses"]
        if "error_count" in kwargs:
            metrics.error_count = kwargs["error_count"]
        
        # Store all scores
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                metrics.all_scores[key] = float(value)
        
        self.logger.debug(f"Logged metrics: {kwargs}")
    
    def log_artifact(self, name: str, file_path: Union[str, Path]) -> None:
        """Log an experiment artifact."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call create_experiment() first.")
        
        file_path = str(file_path)
        self.current_experiment.artifacts[name] = file_path
        self.logger.debug(f"Logged artifact '{name}': {file_path}")
    
    def add_tags(self, *tags: str) -> None:
        """Add tags to the current experiment."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call create_experiment() first.")
        
        for tag in tags:
            if tag not in self.current_experiment.tags:
                self.current_experiment.tags.append(tag)
        
        self.logger.debug(f"Added tags: {tags}")
    
    def set_notes(self, notes: str) -> None:
        """Set experiment notes."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call create_experiment() first.")
        
        self.current_experiment.notes = notes
        self.logger.debug("Updated experiment notes")
    
    def log_evaluation_result(self, result: Any) -> None:
        """Log an evaluation result to the current experiment."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call create_experiment() first.")
        
        # Extract metrics from evaluation result
        primary_score = 0.0
        all_scores = {}
        
        if hasattr(result, 'metrics') and result.metrics:
            for metric_name, score in result.metrics.items():
                all_scores[metric_name] = score
                if not primary_score:  # Use first metric as primary
                    primary_score = score
        
        # Log metrics
        self.log_metrics(
            primary_score=primary_score,
            samples_processed=len(result.predictions) if hasattr(result, 'predictions') else 0,
            **all_scores
        )
        
        # Log manifest as artifact if available
        if hasattr(result, 'manifest') and result.manifest:
            manifest_path = self.experiments_dir / f"{self.current_experiment.run_id}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(result.manifest, f, indent=2)
            self.log_artifact("manifest", manifest_path)
    
    def finish_experiment(self, status: str = "completed") -> ExperimentRun:
        """Finish the current experiment and save it."""
        if not self.current_experiment:
            raise ValueError("No active experiment to finish.")
        
        # Calculate runtime
        if self.start_time:
            runtime = time.time() - self.start_time
            self.current_experiment.metrics.runtime_seconds = runtime
            
            # Recalculate throughput
            if self.current_experiment.metrics.samples_processed > 0:
                throughput = self.current_experiment.metrics.samples_processed / runtime
                self.current_experiment.metrics.throughput_samples_per_sec = throughput
        
        # Update status and timestamp
        self.current_experiment.status = status
        self.current_experiment.updated_at = datetime.utcnow().isoformat()
        
        # Save experiment
        self._save_experiment(self.current_experiment)
        
        experiment = self.current_experiment
        self.current_experiment = None
        self.start_time = None
        
        self.logger.info(f"Finished experiment: {experiment.experiment_id} (status: {status})")
        
        return experiment
    
    def load_experiment(self, experiment_id: str, run_id: Optional[str] = None) -> ExperimentRun:
        """Load an experiment by ID."""
        experiment_dir = self.experiments_dir / experiment_id
        
        if not experiment_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # If run_id not specified, load the latest run
        if run_id is None:
            run_files = list(experiment_dir.glob("run_*.json"))
            if not run_files:
                raise ValueError(f"No runs found for experiment {experiment_id}")
            
            # Sort by creation time (filename contains timestamp)
            run_files.sort(key=lambda x: x.name)
            run_file = run_files[-1]
        else:
            run_file = experiment_dir / f"run_{run_id}.json"
            if not run_file.exists():
                raise ValueError(f"Run {run_id} not found for experiment {experiment_id}")
        
        with open(run_file) as f:
            data = json.load(f)
        
        # Reconstruct experiment run
        return self._dict_to_experiment_run(data)
    
    def list_experiments(self, limit: Optional[int] = None) -> List[ExperimentRun]:
        """List all experiments, optionally limited to most recent."""
        experiments = []
        
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                try:
                    # Load the latest run for each experiment
                    experiment = self.load_experiment(exp_dir.name)
                    experiments.append(experiment)
                except Exception as e:
                    self.logger.warning(f"Failed to load experiment {exp_dir.name}: {e}")
        
        # Sort by creation time (most recent first)
        experiments.sort(key=lambda x: x.created_at, reverse=True)
        
        if limit:
            experiments = experiments[:limit]
        
        return experiments
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        experiments = []
        
        for exp_id in experiment_ids:
            try:
                experiment = self.load_experiment(exp_id)
                experiments.append(experiment)
            except Exception as e:
                self.logger.warning(f"Failed to load experiment {exp_id}: {e}")
        
        if not experiments:
            return {"error": "No valid experiments found"}
        
        # Compare metrics
        comparison = {
            "experiments": [
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "primary_score": exp.metrics.primary_score,
                    "runtime_seconds": exp.metrics.runtime_seconds,
                    "throughput": exp.metrics.throughput_samples_per_sec,
                    "all_scores": exp.metrics.all_scores
                }
                for exp in experiments
            ],
            "best_primary_score": max(exp.metrics.primary_score for exp in experiments),
            "best_runtime": min(exp.metrics.runtime_seconds for exp in experiments if exp.metrics.runtime_seconds > 0),
            "best_throughput": max(exp.metrics.throughput_samples_per_sec for exp in experiments)
        }
        
        # Find best performing experiment for each metric
        all_metric_names = set()
        for exp in experiments:
            all_metric_names.update(exp.metrics.all_scores.keys())
        
        best_by_metric = {}
        for metric in all_metric_names:
            scores = [
                (exp.experiment_id, exp.metrics.all_scores.get(metric, 0))
                for exp in experiments
                if metric in exp.metrics.all_scores
            ]
            if scores:
                best_exp_id, best_score = max(scores, key=lambda x: x[1])
                best_by_metric[metric] = {"experiment_id": best_exp_id, "score": best_score}
        
        comparison["best_by_metric"] = best_by_metric
        
        return comparison
    
    def _generate_experiment_id(self, name: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{timestamp}-{name_hash}"
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        return timestamp
    
    def _capture_environment(self) -> ExperimentEnvironment:
        """Capture current environment information."""
        import sys
        import os
        
        env = ExperimentEnvironment(
            python_version=sys.version,
            platform=platform.platform(),
            hostname=platform.node()
        )
        
        # Capture git information if available
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            env.git_commit = git_commit
            
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            env.git_branch = git_branch
            
            # Check if working directory is dirty
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            env.git_dirty = bool(git_status)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Git not available or not a git repository
        
        # Capture key environment variables
        important_env_vars = [
            "PYTHONPATH", "PATH", "CUDA_VISIBLE_DEVICES", 
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY"
        ]
        
        for var in important_env_vars:
            value = os.environ.get(var)
            if value:
                # Mask API keys
                if "API_KEY" in var:
                    value = "***masked***"
                env.environment_variables[var] = value
        
        return env
    
    def _save_experiment(self, experiment: ExperimentRun) -> None:
        """Save experiment to disk."""
        experiment_dir = self.experiments_dir / experiment.experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Save run data
        run_file = experiment_dir / f"run_{experiment.run_id}.json"
        
        with open(run_file, 'w') as f:
            json.dump(asdict(experiment), f, indent=2)
        
        # Update experiment index
        index_file = self.experiments_dir / "index.json"
        
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
        else:
            index = {"experiments": []}
        
        # Update or add experiment entry
        exp_entry = {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "latest_run_id": experiment.run_id,
            "created_at": experiment.created_at,
            "updated_at": experiment.updated_at,
            "status": experiment.status
        }
        
        # Remove existing entry if present
        index["experiments"] = [
            e for e in index["experiments"] 
            if e["experiment_id"] != experiment.experiment_id
        ]
        
        # Add updated entry
        index["experiments"].append(exp_entry)
        
        # Sort by update time
        index["experiments"].sort(key=lambda x: x["updated_at"], reverse=True)
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _dict_to_experiment_run(self, data: Dict[str, Any]) -> ExperimentRun:
        """Convert dictionary back to ExperimentRun object."""
        # Reconstruct nested objects
        data["parameters"] = ExperimentParameters(**data["parameters"])
        data["environment"] = ExperimentEnvironment(**data["environment"])
        data["metrics"] = ExperimentMetrics(**data["metrics"])
        
        return ExperimentRun(**data)
    
    def export_experiments(self, output_file: Path, experiment_ids: Optional[List[str]] = None) -> None:
        """Export experiments to a file for sharing."""
        if experiment_ids:
            experiments = [self.load_experiment(exp_id) for exp_id in experiment_ids]
        else:
            experiments = self.list_experiments()
        
        # Convert to serializable format
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "experiments": [asdict(exp) for exp in experiments]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(experiments)} experiments to {output_file}")


# Global experiment tracker instance
experiment_tracker = ExperimentTracker()


def track_experiment(name: str, description: str = "", tags: Optional[List[str]] = None):
    """Decorator for tracking experiments."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create experiment
            exp_id = experiment_tracker.create_experiment(name, description, tags)
            
            try:
                # Log function parameters
                experiment_tracker.log_parameters(**kwargs)
                
                # Run function
                result = func(*args, **kwargs)
                
                # Log result if it's an EvaluationResult
                if hasattr(result, 'metrics'):
                    experiment_tracker.log_evaluation_result(result)
                
                # Finish successfully
                experiment_tracker.finish_experiment("completed")
                
                return result
                
            except Exception as e:
                # Finish with error
                experiment_tracker.finish_experiment("failed")
                raise e
        
        return wrapper
    return decorator
