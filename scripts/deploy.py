#!/usr/bin/env python3
"""Deployment script for OpenEval Lab."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openeval.deployment import (
    DeploymentConfig,
    DockerManager,
    TerraformManager,
    CICDPipeline,
    create_deployment_package,
    generate_dockerfile,
    generate_kubernetes_manifests,
)


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy OpenEval Lab")
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment environment",
    )
    parser.add_argument(
        "--action",
        choices=["build", "deploy", "infrastructure", "package", "generate"],
        required=True,
        help="Deployment action",
    )
    parser.add_argument("--version", help="Application version")
    parser.add_argument("--docker-registry", help="Docker registry URL")
    parser.add_argument("--kubernetes-namespace", default="default", help="Kubernetes namespace")
    parser.add_argument("--helm-chart", help="Helm chart path")
    parser.add_argument("--terraform-path", help="Terraform configuration path")

    args = parser.parse_args()

    # Create deployment configuration
    config = DeploymentConfig(
        environment=args.environment,
        version=args.version,
        docker_registry=args.docker_registry,
        kubernetes_namespace=args.kubernetes_namespace,
        helm_chart_path=args.helm_chart,
        terraform_path=args.terraform_path,
    )

    if args.action == "build":
        # Build Docker image
        docker_manager = DockerManager(config)
        success = docker_manager.build_image()
        if success:
            print("✅ Docker image built successfully")
        else:
            print("❌ Docker image build failed")
            sys.exit(1)

    elif args.action == "deploy":
        # Deploy application
        pipeline = CICDPipeline(config)
        success = pipeline.run_pipeline(["deploy"])
        if success:
            print("✅ Deployment completed successfully")
        else:
            print("❌ Deployment failed")
            sys.exit(1)

    elif args.action == "infrastructure":
        # Manage infrastructure with Terraform
        tf_manager = TerraformManager(config)

        # Initialize Terraform
        if not tf_manager.init():
            print("❌ Terraform init failed")
            sys.exit(1)

        # Plan changes
        if not tf_manager.plan():
            print("❌ Terraform plan failed")
            sys.exit(1)

        # Apply changes
        if tf_manager.apply():
            print("✅ Infrastructure deployed successfully")
        else:
            print("❌ Infrastructure deployment failed")
            sys.exit(1)

    elif args.action == "package":
        # Create deployment package
        package_path = create_deployment_package()
        print(f"✅ Deployment package created: {package_path}")

    elif args.action == "generate":
        # Generate deployment configurations
        dockerfile = generate_dockerfile()
        dockerfile_path = Path("Dockerfile.generated")
        dockerfile_path.write_text(dockerfile)
        print(f"✅ Dockerfile generated: {dockerfile_path}")

        manifests = generate_kubernetes_manifests(namespace=config.kubernetes_namespace)
        for name, content in manifests.items():
            manifest_path = Path(f"k8s-{name}")
            manifest_path.write_text(content)
            print(f"✅ Kubernetes manifest generated: {manifest_path}")


if __name__ == "__main__":
    main()
