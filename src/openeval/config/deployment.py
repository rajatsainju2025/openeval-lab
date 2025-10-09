"""Production deployment and scaling features."""

from __future__ import annotations

from typing import Dict, Any, Optional
from pathlib import Path
import yaml


class DeploymentConfig:
    """Configuration for production deployment."""

    def __init__(
        self,
        environment: str = "development",
        replicas: int = 1,
        resources: Optional[Dict[str, Any]] = None,
        ingress: Optional[Dict[str, Any]] = None,
        monitoring: Optional[Dict[str, Any]] = None,
    ):
        self.environment = environment
        self.replicas = replicas
        self.resources = resources or {}
        self.ingress = ingress or {}
        self.monitoring = monitoring or {}

    @classmethod
    def from_yaml(cls, path: Path) -> DeploymentConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_kubernetes_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"openeval-{self.environment}",
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {"matchLabels": {"app": "openeval"}},
                "template": {
                    "metadata": {"labels": {"app": "openeval"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "openeval",
                                "image": "openeval:latest",
                                "resources": self.resources,
                                "ports": [{"containerPort": 8000}],
                                "env": [{"name": "ENVIRONMENT", "value": self.environment}],
                            }
                        ]
                    },
                },
            },
        }


class Scaler:
    """Auto-scaling logic for production deployments."""

    def __init__(self, min_replicas: int = 1, max_replicas: int = 10):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas

    def scale_based_on_load(self, cpu_usage: float, queue_length: int) -> int:
        """Determine optimal replica count based on load."""
        if cpu_usage > 80 or queue_length > 100:
            self.current_replicas = min(self.current_replicas + 1, self.max_replicas)
        elif cpu_usage < 30 and queue_length < 10:
            self.current_replicas = max(self.current_replicas - 1, self.min_replicas)

        return self.current_replicas


class LoadBalancer:
    """Load balancing configuration."""

    def __init__(self, algorithm: str = "round_robin"):
        self.algorithm = algorithm
        self.backends = []

    def add_backend(self, host: str, port: int, weight: int = 1):
        """Add backend server."""
        self.backends.append({"host": host, "port": port, "weight": weight, "healthy": True})

    def get_next_backend(self) -> Optional[Dict[str, Any]]:
        """Get next backend using load balancing algorithm."""
        healthy_backends = [b for b in self.backends if b["healthy"]]
        if not healthy_backends:
            return None

        if self.algorithm == "round_robin":
            # Simple round robin
            backend = healthy_backends[0]
            self.backends.append(self.backends.pop(0))
            return backend

        return healthy_backends[0]


__all__ = ["DeploymentConfig", "Scaler", "LoadBalancer"]
