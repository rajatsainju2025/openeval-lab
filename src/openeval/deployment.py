"""Deployment automation and CI/CD scripts."""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import yaml
import shutil
from datetime import datetime

from .logging import get_logger

logger = get_logger(__name__)


class DeploymentConfig:
    """Configuration for deployment."""

    def __init__(
        self,
        environment: str = "production",
        version: Optional[str] = None,
        docker_registry: Optional[str] = None,
        kubernetes_namespace: str = "default",
        helm_chart_path: Optional[str] = None,
        terraform_path: Optional[str] = None,
    ):
        self.environment = environment
        self.version = version or self._get_version()
        self.docker_registry = docker_registry
        self.kubernetes_namespace = kubernetes_namespace
        self.helm_chart_path = helm_chart_path
        self.terraform_path = terraform_path

    def _get_version(self) -> str:
        """Get version from git tags or package."""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        # Fallback to timestamp-based version
        return f"v{datetime.now().strftime('%Y%m%d.%H%M%S')}"


class DockerManager:
    """Docker image management for deployment."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.image_name = f"openeval-lab:{self.config.version}"

    def build_image(
        self, dockerfile_path: str = "Dockerfile", build_args: Optional[Dict[str, str]] = None
    ) -> bool:
        """Build Docker image."""
        cmd = ["docker", "build", "-t", self.image_name]

        if build_args:
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

        cmd.append(".")

        logger.info(f"Building Docker image: {self.image_name}")
        cwd_path = Path(dockerfile_path).parent if dockerfile_path != "Dockerfile" else None
        result = self._run_command(cmd, cwd=cwd_path)

        if result and self.config.docker_registry:
            tagged_image = f"{self.config.docker_registry}/{self.image_name}"
            self._run_command(["docker", "tag", self.image_name, tagged_image])
            self._run_command(["docker", "push", tagged_image])
            logger.info(f"Pushed image to registry: {tagged_image}")

        return result

    def run_container(
        self, ports: Optional[Dict[str, str]] = None, env_vars: Optional[Dict[str, str]] = None
    ) -> bool:
        """Run Docker container for testing."""
        cmd = ["docker", "run", "--rm"]

        if ports:
            for host_port, container_port in ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])

        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.append(self.image_name)

        logger.info(f"Running Docker container: {self.image_name}")
        return self._run_command(cmd)

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> bool:
        """Run shell command."""
        try:
            result = subprocess.run(
                cmd, cwd=cwd or Path.cwd(), capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                logger.info(f"Command succeeded: {' '.join(cmd)}")
                return True
            else:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Error output: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return False
        except Exception as e:
            logger.error(f"Command error: {e}")
            return False


class KubernetesManager:
    """Kubernetes deployment management."""

    def __init__(self, config: DeploymentConfig):
        self.config = config

    def deploy_with_helm(self, values_file: Optional[str] = None) -> bool:
        """Deploy using Helm."""
        if not self.config.helm_chart_path:
            logger.error("Helm chart path not configured")
            return False

        cmd = ["helm", "upgrade", "--install", "openeval-lab", self.config.helm_chart_path]

        if self.config.kubernetes_namespace != "default":
            cmd.extend(["--namespace", self.config.kubernetes_namespace])

        if values_file:
            cmd.extend(["-f", values_file])

        cmd.extend(["--set", f"image.tag={self.config.version}"])

        logger.info(f"Deploying with Helm to namespace: {self.config.kubernetes_namespace}")
        return self._run_command(cmd)

    def deploy_with_kubectl(self, manifest_path: str) -> bool:
        """Deploy using kubectl."""
        cmd = ["kubectl", "apply", "-f", manifest_path]

        if self.config.kubernetes_namespace != "default":
            cmd.extend(["--namespace", self.config.kubernetes_namespace])

        logger.info(f"Deploying with kubectl to namespace: {self.config.kubernetes_namespace}")
        return self._run_command(cmd)

    def check_deployment_status(self, deployment_name: str = "openeval-lab") -> Dict[str, Any]:
        """Check deployment status."""
        cmd = ["kubectl", "get", "deployment", deployment_name, "-o", "json"]

        if self.config.kubernetes_namespace != "default":
            cmd.extend(["--namespace", self.config.kubernetes_namespace])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return {
                    "ready": data["status"]["readyReplicas"] == data["status"]["replicas"],
                    "available": data["status"]["availableReplicas"] == data["status"]["replicas"],
                    "replicas": data["status"]["replicas"],
                }
        except Exception as e:
            logger.error(f"Failed to check deployment status: {e}")

        return {"ready": False, "available": False, "replicas": 0}

    def _run_command(self, cmd: List[str]) -> bool:
        """Run shell command."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info(f"Command succeeded: {' '.join(cmd)}")
                return True
            else:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Error output: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return False
        except Exception as e:
            logger.error(f"Command error: {e}")
            return False


class TerraformManager:
    """Terraform infrastructure management."""

    def __init__(self, config: DeploymentConfig):
        self.config = config

    def init(self) -> bool:
        """Initialize Terraform."""
        if not self.config.terraform_path:
            logger.error("Terraform path not configured")
            return False

        cmd = ["terraform", "init"]
        logger.info("Initializing Terraform")
        return self._run_command(cmd, cwd=Path(self.config.terraform_path))

    def plan(self, variables: Optional[Dict[str, str]] = None) -> bool:
        """Plan Terraform changes."""
        cmd = ["terraform", "plan"]

        if variables:
            for key, value in variables.items():
                cmd.extend(["-var", f"{key}={value}"])

        logger.info("Planning Terraform changes")
        if self.config.terraform_path:
            return self._run_command(cmd, cwd=Path(self.config.terraform_path))
        else:
            return self._run_command(cmd)

    def apply(self, variables: Optional[Dict[str, str]] = None) -> bool:
        """Apply Terraform changes."""
        cmd = ["terraform", "apply", "-auto-approve"]

        if variables:
            for key, value in variables.items():
                cmd.extend(["-var", f"{key}={value}"])

        logger.info("Applying Terraform changes")
        if self.config.terraform_path:
            return self._run_command(cmd, cwd=Path(self.config.terraform_path))
        else:
            return self._run_command(cmd)

    def destroy(self, variables: Optional[Dict[str, str]] = None) -> bool:
        """Destroy Terraform resources."""
        cmd = ["terraform", "destroy", "-auto-approve"]

        if variables:
            for key, value in variables.items():
                cmd.extend(["-var", f"{key}={value}"])

        logger.info("Destroying Terraform resources")
        if self.config.terraform_path:
            return self._run_command(cmd, cwd=Path(self.config.terraform_path))
        else:
            return self._run_command(cmd)

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> bool:
        """Run shell command."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or Path.cwd(),
                capture_output=True,
                text=True,
                timeout=600,  # Terraform can take longer
            )
            if result.returncode == 0:
                logger.info(f"Command succeeded: {' '.join(cmd)}")
                return True
            else:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Error output: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return False
        except Exception as e:
            logger.error(f"Command error: {e}")
            return False


class CICDPipeline:
    """CI/CD pipeline management."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.docker_manager = DockerManager(config)
        self.k8s_manager = KubernetesManager(config)
        self.tf_manager = TerraformManager(config)

    def run_pipeline(self, stages: Optional[List[str]] = None) -> bool:
        """Run the complete CI/CD pipeline."""
        if stages is None:
            stages = ["lint", "test", "build", "deploy"]

        results = {}

        for stage in stages:
            logger.info(f"Running pipeline stage: {stage}")
            if stage == "lint":
                results[stage] = self._run_lint()
            elif stage == "test":
                results[stage] = self._run_tests()
            elif stage == "build":
                results[stage] = self._run_build()
            elif stage == "deploy":
                results[stage] = self._run_deploy()
            else:
                logger.warning(f"Unknown pipeline stage: {stage}")
                results[stage] = False

            if not results[stage]:
                logger.error(f"Pipeline failed at stage: {stage}")
                return False

        logger.info("CI/CD pipeline completed successfully")
        return True

    def _run_lint(self) -> bool:
        """Run linting checks."""
        logger.info("Running lint checks")

        # Run black
        if not self._run_command(["black", "--check", "--diff", "src/"]):
            return False

        # Run ruff
        if not self._run_command(["ruff", "check", "src/"]):
            return False

        # Run mypy (if available)
        try:
            if not self._run_command(["mypy", "src/"]):
                return False
        except FileNotFoundError:
            logger.warning("mypy not available, skipping type checking")

        return True

    def _run_tests(self) -> bool:
        """Run test suite."""
        logger.info("Running test suite")

        # Run pytest
        cmd = ["python", "-m", "pytest", "--cov=src", "--cov-report=xml", "--cov-report=term"]

        # Add coverage threshold for CI
        if self.config.environment == "ci":
            cmd.extend(["--cov-fail-under=80"])

        return self._run_command(cmd)

    def _run_build(self) -> bool:
        """Build application."""
        logger.info("Building application")

        # Build Docker image
        build_args = {"ENVIRONMENT": self.config.environment, "VERSION": self.config.version}

        return self.docker_manager.build_image(build_args=build_args)

    def _run_deploy(self) -> bool:
        """Deploy application."""
        logger.info("Deploying application")

        if self.config.terraform_path:
            # Deploy infrastructure first
            if not self.tf_manager.init():
                return False
            if not self.tf_manager.plan():
                return False
            if not self.tf_manager.apply():
                return False

        # Deploy application
        if self.config.helm_chart_path:
            values_file = f"deploy/{self.config.environment}-values.yaml"
            if Path(values_file).exists():
                return self.k8s_manager.deploy_with_helm(values_file)
            else:
                return self.k8s_manager.deploy_with_helm()
        else:
            # Fallback to direct kubectl
            manifest_file = f"deploy/{self.config.environment}-deployment.yaml"
            if Path(manifest_file).exists():
                return self.k8s_manager.deploy_with_kubectl(manifest_file)
            else:
                logger.error("No deployment configuration found")
                return False

    def _run_command(self, cmd: List[str]) -> bool:
        """Run shell command."""
        try:
            result = subprocess.run(
                cmd, cwd=Path.cwd(), capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                logger.info(f"Command succeeded: {' '.join(cmd)}")
                return True
            else:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Error output: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return False
        except Exception as e:
            logger.error(f"Command error: {e}")
            return False


# Utility functions
def create_deployment_package(source_dir: str = ".", output_file: Optional[str] = None) -> str:
    """Create deployment package."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file is None:
        output_file = f"openeval-lab-deployment-{timestamp}.tar.gz"

    source_path = Path(source_dir)
    if Path(output_file).is_absolute():
        output_path = Path(output_file)
    else:
        output_path = Path.cwd() / output_file

    # Create temporary directory for package
    temp_dir = Path(f"temp_deployment_{timestamp}")
    temp_dir.mkdir()

    try:
        # Copy source files (excluding unnecessary files)
        excludes = [".git", "__pycache__", "*.pyc", ".pytest_cache", "temp_*"]

        for item in source_path.rglob("*"):
            if item.is_file():
                # Check if file should be excluded
                should_exclude = False
                for exclude in excludes:
                    if exclude.startswith("*."):
                        if item.name.endswith(exclude[1:]):
                            should_exclude = True
                            break
                    elif exclude in str(item):
                        should_exclude = True
                        break

                if not should_exclude:
                    # Copy file to temp directory
                    rel_path = item.relative_to(source_path)
                    dest_path = temp_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)

        # Create tar.gz archive
        shutil.make_archive(str(output_path.with_suffix("")), "gztar", temp_dir)

        logger.info(f"Created deployment package: {output_file}")
        return str(output_path)

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def generate_dockerfile(base_image: str = "python:3.12-slim", port: int = 8000) -> str:
    """Generate Dockerfile for the application."""
    dockerfile = f"""FROM {base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{port}/health || exit 1

# Run application
CMD ["python", "-m", "src.openeval.main"]
"""

    return dockerfile


def generate_kubernetes_manifests(
    namespace: str = "default", image: str = "openeval-lab:latest", replicas: int = 3
) -> Dict[str, str]:
    """Generate Kubernetes manifests."""
    manifests = {}

    # Deployment
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "openeval-lab", "namespace": namespace},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": "openeval-lab"}},
            "template": {
                "metadata": {"labels": {"app": "openeval-lab"}},
                "spec": {
                    "containers": [
                        {
                            "name": "openeval-lab",
                            "image": image,
                            "ports": [{"containerPort": 8000}],
                            "env": [{"name": "ENVIRONMENT", "value": "production"}],
                            "resources": {
                                "requests": {"memory": "512Mi", "cpu": "250m"},
                                "limits": {"memory": "1Gi", "cpu": "500m"},
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                            },
                        }
                    ]
                },
            },
        },
    }

    # Service
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "openeval-lab-service", "namespace": namespace},
        "spec": {
            "selector": {"app": "openeval-lab"},
            "ports": [{"port": 80, "targetPort": 8000}],
            "type": "ClusterIP",
        },
    }

    manifests["deployment.yaml"] = yaml.dump(deployment, default_flow_style=False)
    manifests["service.yaml"] = yaml.dump(service, default_flow_style=False)

    return manifests


def setup_ci_cd_pipeline() -> Dict[str, Any]:
    """Setup CI/CD pipeline configuration."""
    # GitHub Actions workflow
    github_actions = {
        "name": "CI/CD Pipeline",
        "on": {"push": {"branches": ["main", "develop"]}, "pull_request": {"branches": ["main"]}},
        "jobs": {
            "lint": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {"uses": "actions/checkout@v3"},
                    {"uses": "actions/setup-python@v4", "with": {"python-version": "3.12"}},
                    {"run": "pip install black ruff"},
                    {"run": "black --check src/"},
                    {"run": "ruff check src/"},
                ],
            },
            "test": {
                "runs-on": "ubuntu-latest",
                "needs": "lint",
                "steps": [
                    {"uses": "actions/checkout@v3"},
                    {"uses": "actions/setup-python@v4", "with": {"python-version": "3.12"}},
                    {"run": "pip install -r requirements.txt"},
                    {"run": "pip install pytest pytest-cov"},
                    {"run": "pytest --cov=src --cov-report=xml"},
                    {"uses": "codecov/codecov-action@v3"},
                ],
            },
            "build-and-deploy": {
                "runs-on": "ubuntu-latest",
                "needs": "test",
                "if": "github.ref == 'refs/heads/main'",
                "steps": [
                    {"uses": "actions/checkout@v3"},
                    {
                        "name": "Login to Docker Hub",
                        "uses": "docker/login-action@v2",
                        "with": {
                            "username": "${{ secrets.DOCKER_USERNAME }}",
                            "password": "${{ secrets.DOCKER_PASSWORD }}",
                        },
                    },
                    {
                        "name": "Build and push Docker image",
                        "run": "docker build -t openeval-lab . && docker push openeval-lab",
                    },
                    {"name": "Deploy to production", "run": "echo 'Deploy logic here'"},
                ],
            },
        },
    }

    return {"github_actions": github_actions}
