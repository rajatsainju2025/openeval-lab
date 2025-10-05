"""Tests for deployment automation and CI/CD scripts."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from openeval.deployment import (
    DeploymentConfig,
    DockerManager,
    KubernetesManager,
    TerraformManager,
    CICDPipeline,
    create_deployment_package,
    generate_dockerfile,
    generate_kubernetes_manifests,
    setup_ci_cd_pipeline,
)


class TestDeploymentConfig:
    """Test DeploymentConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = DeploymentConfig()
        assert config.environment == "production"
        assert config.version is not None
        assert config.docker_registry is None
        assert config.kubernetes_namespace == "default"
        assert config.helm_chart_path is None
        assert config.terraform_path is None

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        config = DeploymentConfig(
            environment="staging",
            version="v1.0.0",
            docker_registry="registry.example.com",
            kubernetes_namespace="staging",
            helm_chart_path="./charts",
            terraform_path="./terraform",
        )
        assert config.environment == "staging"
        assert config.version == "v1.0.0"
        assert config.docker_registry == "registry.example.com"
        assert config.kubernetes_namespace == "staging"
        assert config.helm_chart_path == "./charts"
        assert config.terraform_path == "./terraform"

    @patch("subprocess.run")
    def test_get_version_from_git(self, mock_run):
        """Test version retrieval from git tags."""
        mock_run.return_value = Mock(returncode=0, stdout="v1.2.3\n")
        config = DeploymentConfig()
        assert config.version == "v1.2.3"


class TestDockerManager:
    """Test DockerManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = DeploymentConfig(version="v1.0.0")
        self.docker_manager = DockerManager(self.config)

    @patch("subprocess.run")
    def test_build_image_success(self, mock_run):
        """Test successful Docker image build."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.docker_manager.build_image()
        assert result is True
        assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_build_image_with_registry(self, mock_run):
        """Test Docker image build with registry push."""
        self.config.docker_registry = "registry.example.com"
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.docker_manager.build_image()
        assert result is True
        assert mock_run.call_count == 3  # build, tag, push

    @patch("subprocess.run")
    def test_build_image_failure(self, mock_run):
        """Test Docker image build failure."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Build failed")
        result = self.docker_manager.build_image()
        assert result is False

    @patch("subprocess.run")
    def test_run_container(self, mock_run):
        """Test running Docker container."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.docker_manager.run_container(ports={"8080": "8000"}, env_vars={"ENV": "test"})
        assert result is True


class TestKubernetesManager:
    """Test KubernetesManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = DeploymentConfig(kubernetes_namespace="test-ns", helm_chart_path="./charts")
        self.k8s_manager = KubernetesManager(self.config)

    @patch("subprocess.run")
    def test_deploy_with_helm_success(self, mock_run):
        """Test successful Helm deployment."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.k8s_manager.deploy_with_helm()
        assert result is True

    @patch("subprocess.run")
    def test_deploy_with_helm_no_chart_path(self, mock_run):
        """Test Helm deployment without chart path."""
        self.config.helm_chart_path = None
        result = self.k8s_manager.deploy_with_helm()
        assert result is False
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_deploy_with_kubectl(self, mock_run):
        """Test kubectl deployment."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.k8s_manager.deploy_with_kubectl("deployment.yaml")
        assert result is True

    @patch("subprocess.run")
    def test_check_deployment_status_ready(self, mock_run):
        """Test deployment status check when ready."""
        mock_result = Mock(
            returncode=0,
            stdout='{"status": {"readyReplicas": 3, "replicas": 3, "availableReplicas": 3}}',
        )
        mock_run.return_value = mock_result
        status = self.k8s_manager.check_deployment_status()
        assert status["ready"] is True
        assert status["available"] is True
        assert status["replicas"] == 3


class TestTerraformManager:
    """Test TerraformManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = DeploymentConfig(terraform_path="./terraform")
        self.tf_manager = TerraformManager(self.config)

    @patch("subprocess.run")
    def test_init_success(self, mock_run):
        """Test successful Terraform init."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.tf_manager.init()
        assert result is True

    @patch("subprocess.run")
    def test_plan_success(self, mock_run):
        """Test successful Terraform plan."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.tf_manager.plan()
        assert result is True

    @patch("subprocess.run")
    def test_apply_success(self, mock_run):
        """Test successful Terraform apply."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.tf_manager.apply()
        assert result is True

    @patch("subprocess.run")
    def test_destroy_success(self, mock_run):
        """Test successful Terraform destroy."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.tf_manager.destroy()
        assert result is True

    def test_no_terraform_path(self):
        """Test operations without terraform path configured."""
        self.config.terraform_path = None
        tf_manager = TerraformManager(self.config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            result = tf_manager.plan()
            assert result is True


class TestCICDPipeline:
    """Test CI/CD Pipeline class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = DeploymentConfig()
        self.pipeline = CICDPipeline(self.config)

    @patch("subprocess.run")
    def test_run_pipeline_all_stages(self, mock_run):
        """Test running complete pipeline."""
        # Set up deployment configuration
        self.config.helm_chart_path = "./charts"
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.pipeline.run_pipeline()
        assert result is True

    @patch("subprocess.run")
    def test_run_pipeline_custom_stages(self, mock_run):
        """Test running pipeline with custom stages."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.pipeline.run_pipeline(["lint", "test"])
        assert result is True

    @patch("subprocess.run")
    def test_run_lint_success(self, mock_run):
        """Test lint stage success."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.pipeline._run_lint()
        assert result is True

    @patch("subprocess.run")
    def test_run_tests_success(self, mock_run):
        """Test test stage success."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.pipeline._run_tests()
        assert result is True

    @patch("subprocess.run")
    def test_run_build_success(self, mock_run):
        """Test build stage success."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.pipeline._run_build()
        assert result is True

    @patch("subprocess.run")
    def test_run_deploy_success(self, mock_run):
        """Test deploy stage success."""
        # Set up helm chart path for the test
        self.config.helm_chart_path = "./charts"
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        result = self.pipeline._run_deploy()
        assert result is True


class TestUtilityFunctions:
    """Test utility functions."""

    @patch("shutil.make_archive")
    def test_create_deployment_package(self, mock_make_archive):
        """Test deployment package creation."""
        mock_make_archive.return_value = None  # make_archive returns None

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files and directory structure
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")

            src_dir = Path(temp_dir) / "src"
            src_dir.mkdir()
            (src_dir / "main.py").write_text("print('hello')")

            # Create package (will be created in current working directory)
            package_path = create_deployment_package(temp_dir)

            # Verify make_archive was called
            mock_make_archive.assert_called_once()
            assert package_path.endswith(".tar.gz")

    def test_generate_dockerfile(self):
        """Test Dockerfile generation."""
        dockerfile = generate_dockerfile()
        assert "FROM python:3.12-slim" in dockerfile
        assert "EXPOSE 8000" in dockerfile
        assert "HEALTHCHECK" in dockerfile

    def test_generate_kubernetes_manifests(self):
        """Test Kubernetes manifests generation."""
        manifests = generate_kubernetes_manifests(namespace="test", replicas=2)
        assert "deployment.yaml" in manifests
        assert "service.yaml" in manifests

        deployment = manifests["deployment.yaml"]
        assert "test" in deployment
        assert "replicas: 2" in deployment

    def test_setup_ci_cd_pipeline(self):
        """Test CI/CD pipeline setup."""
        config = setup_ci_cd_pipeline()
        assert "github_actions" in config
        assert "jobs" in config["github_actions"]
        assert "lint" in config["github_actions"]["jobs"]
        assert "test" in config["github_actions"]["jobs"]
        assert "build-and-deploy" in config["github_actions"]["jobs"]


class TestIntegration:
    """Integration tests for deployment components."""

    def test_full_deployment_workflow(self):
        """Test complete deployment workflow."""
        config = DeploymentConfig(environment="staging", version="v1.0.0-test")

        # Test Docker manager
        docker_manager = DockerManager(config)
        assert docker_manager.image_name == "openeval-lab:v1.0.0-test"

        # Test Kubernetes manager
        k8s_manager = KubernetesManager(config)
        assert k8s_manager.config.kubernetes_namespace == "default"

        # Test Terraform manager
        tf_manager = TerraformManager(config)
        assert tf_manager.config.terraform_path is None

        # Test CI/CD pipeline
        pipeline = CICDPipeline(config)
        assert pipeline.config.environment == "staging"

    @patch("subprocess.run")
    def test_pipeline_failure_handling(self, mock_run):
        """Test pipeline failure handling."""
        config = DeploymentConfig()
        pipeline = CICDPipeline(config)

        # Mock failure in lint stage
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Lint failed")

        result = pipeline.run_pipeline(["lint"])
        assert result is False
