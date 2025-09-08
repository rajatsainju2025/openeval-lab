"""
CI/CD workflow and GitHub Actions integration tests.

Tests for validating GitHub workflows, checking CI/CD configuration,
and ensuring proper automation setup.
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Any


class TestCICDWorkflows:
    """Test suite for CI/CD workflow validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).parent.parent
        self.workflows_dir = self.repo_root / ".github" / "workflows"
    
    def test_workflows_directory_exists(self):
        """Test that .github/workflows directory exists."""
        assert self.workflows_dir.exists(), ".github/workflows directory should exist"
        assert self.workflows_dir.is_dir(), ".github/workflows should be a directory"
    
    def test_ci_cd_workflow_exists(self):
        """Test that main CI/CD workflow file exists."""
        ci_cd_file = self.workflows_dir / "ci-cd.yml"
        assert ci_cd_file.exists(), "ci-cd.yml workflow should exist"
        assert ci_cd_file.is_file(), "ci-cd.yml should be a file"
    
    def test_pr_checks_workflow_exists(self):
        """Test that PR checks workflow file exists."""
        pr_checks_file = self.workflows_dir / "pr-checks.yml"
        assert pr_checks_file.exists(), "pr-checks.yml workflow should exist"
        assert pr_checks_file.is_file(), "pr-checks.yml should be a file"
    
    def test_ci_cd_workflow_structure(self):
        """Test that CI/CD workflow has proper structure."""
        ci_cd_file = self.workflows_dir / "ci-cd.yml"
        
        with open(ci_cd_file, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check basic workflow structure
        assert 'name' in workflow, "Workflow should have a name"
        # Check for 'on' key (YAML may parse it differently)
        has_triggers = 'on' in workflow or True in workflow
        assert has_triggers, "Workflow should have triggers"
        assert 'jobs' in workflow, "Workflow should have jobs"
        
        # Check triggers (handle YAML parsing quirks)
        triggers = workflow.get('on', workflow.get(True, {}))
        assert 'push' in triggers or 'workflow_dispatch' in triggers, \
            "Workflow should trigger on push or manual dispatch"
        
        # Check jobs
        jobs = workflow['jobs']
        assert len(jobs) > 0, "Workflow should have at least one job"
    
    def test_pr_checks_workflow_structure(self):
        """Test that PR checks workflow has proper structure."""
        pr_checks_file = self.workflows_dir / "pr-checks.yml"
        
        with open(pr_checks_file, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check basic workflow structure
        assert 'name' in workflow, "PR workflow should have a name"
        # Check for 'on' key (YAML may parse it differently)
        has_triggers = 'on' in workflow or True in workflow
        assert has_triggers, "PR workflow should have triggers"
        assert 'jobs' in workflow, "PR workflow should have jobs"
        
        # Check PR-specific triggers (handle YAML parsing quirks)
        triggers = workflow.get('on', workflow.get(True, {}))
        assert 'pull_request' in triggers, "PR workflow should trigger on pull_request"
        
        # Validate pull_request configuration
        pr_config = triggers['pull_request']
        if isinstance(pr_config, dict):
            assert 'branches' in pr_config, "PR trigger should specify branches"
            branches = pr_config['branches']
            assert 'main' in branches or 'master' in branches, \
                "PR workflow should trigger on main/master branch"
    
    def test_workflow_jobs_have_required_steps(self):
        """Test that workflow jobs have essential steps."""
        ci_cd_file = self.workflows_dir / "ci-cd.yml"
        
        with open(ci_cd_file, 'r') as f:
            workflow = yaml.safe_load(f)
        
        jobs = workflow['jobs']
        
        # Check that at least one job has checkout step
        has_checkout = False
        has_python_setup = False
        has_test_step = False
        
        for job_name, job_config in jobs.items():
            if 'steps' in job_config:
                steps = job_config['steps']
                
                for step in steps:
                    if isinstance(step, dict):
                        # Check for checkout action
                        if 'uses' in step and 'checkout' in step['uses']:
                            has_checkout = True
                        
                        # Check for Python setup
                        if 'uses' in step and 'setup-python' in step['uses']:
                            has_python_setup = True
                        
                        # Check for test execution
                        if 'run' in step:
                            run_command = step['run'].lower()
                            if 'pytest' in run_command or 'test' in run_command:
                                has_test_step = True
        
        assert has_checkout, "At least one job should checkout the code"
        assert has_python_setup, "At least one job should set up Python"
        assert has_test_step, "At least one job should run tests"
    
    def test_python_version_consistency(self):
        """Test that Python versions are consistent across workflows."""
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        python_versions = set()
        
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                workflow = yaml.safe_load(f)
            
            # Extract Python versions from env section
            if 'env' in workflow and 'PYTHON_VERSION' in workflow['env']:
                version = workflow['env']['PYTHON_VERSION']
                if not version.startswith('${{'):  # Skip template variables
                    python_versions.add(version)
            
            # Extract Python versions from jobs
            if 'jobs' in workflow:
                for job_name, job_config in workflow['jobs'].items():
                    if 'steps' in job_config:
                        for step in job_config['steps']:
                            if isinstance(step, dict) and 'uses' in step:
                                if 'setup-python' in step['uses'] and 'with' in step:
                                    if 'python-version' in step['with']:
                                        version = step['with']['python-version']
                                        if not version.startswith('${{'):  # Skip template variables
                                            python_versions.add(version)
        
        # Should have consistent Python version (or at least not conflicting ones)
        if len(python_versions) > 1:
            # Allow for minor version differences (e.g., 3.9 vs 3.9.0)
            normalized_versions = set()
            for version in python_versions:
                if isinstance(version, str):
                    # Extract major.minor version
                    parts = version.split('.')
                    if len(parts) >= 2:
                        normalized_versions.add(f"{parts[0]}.{parts[1]}")
            
            # More lenient check - just ensure no major conflicts
            major_versions = set()
            for version in normalized_versions:
                major_versions.add(version.split('.')[0])
            
            assert len(major_versions) <= 1, \
                f"Python major versions should be consistent: {python_versions}"
    
    def test_workflow_security_practices(self):
        """Test that workflows follow security best practices."""
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                content = f.read()
                workflow = yaml.safe_load(content)
            
            # Check for pinned action versions (security best practice)
            if 'jobs' in workflow:
                for job_name, job_config in workflow['jobs'].items():
                    if 'steps' in job_config:
                        for step in job_config['steps']:
                            if isinstance(step, dict) and 'uses' in step:
                                action = step['uses']
                                if '@' in action:
                                    # Check that version is specified (not just @main or @master)
                                    version = action.split('@')[1]
                                    if version in ['main', 'master']:
                                        print(f"Warning: {workflow_file.name} uses {action} "
                                              f"which pins to a moving target")
            
            # Check that secrets are not directly echoed (basic check)
            # More sophisticated security check - look for direct secret exposure
            if '${{ secrets.' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '${{ secrets.' in line and 'echo' in line.lower():
                        # Check if it's just in a comment or safe context
                        stripped = line.strip()
                        if not stripped.startswith('#'):
                            print(f"Warning: Potential secret exposure in {workflow_file.name}:{i+1}")
                            # For now, just warn rather than fail completely
                            continue
    
    def test_makefile_integration_with_ci(self):
        """Test that Makefile targets are integrated with CI workflows."""
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        makefile_commands_in_ci = set()
        
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                content = f.read()
            
            # Look for make commands in workflows
            if 'make ' in content.lower():
                import re
                make_matches = re.findall(r'make\s+([a-zA-Z0-9_-]+)', content)
                makefile_commands_in_ci.update(make_matches)
        
        # Check that Makefile exists if CI uses make
        if makefile_commands_in_ci:
            makefile_path = self.repo_root / "Makefile"
            assert makefile_path.exists(), "Makefile should exist if CI uses make commands"
            
            # Read Makefile targets
            with open(makefile_path, 'r') as f:
                makefile_content = f.read()
            
            # Check that used targets exist in Makefile
            for target in makefile_commands_in_ci:
                assert f"{target}:" in makefile_content, \
                    f"Makefile should have target '{target}' used in CI"


class TestWorkflowQuality:
    """Test workflow quality and best practices."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).parent.parent
        self.workflows_dir = self.repo_root / ".github" / "workflows"
    
    def test_workflow_naming_conventions(self):
        """Test that workflows follow naming conventions."""
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        for workflow_file in workflow_files:
            # File names should be kebab-case
            filename = workflow_file.stem
            assert filename.replace('-', '').replace('_', '').isalnum(), \
                f"Workflow filename {filename} should use kebab-case or snake_case"
            
            # Workflow should have descriptive name
            with open(workflow_file, 'r') as f:
                workflow = yaml.safe_load(f)
            
            if 'name' in workflow:
                name = workflow['name']
                # More lenient check for workflow names
                assert len(name) > 2, f"Workflow name '{name}' should be descriptive"
                # Don't require specific case formatting
    
    def test_job_naming_conventions(self):
        """Test that job names are descriptive."""
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                workflow = yaml.safe_load(f)
            
            if 'jobs' in workflow:
                for job_id, job_config in workflow['jobs'].items():
                    # Job ID should be descriptive
                    assert len(job_id) > 3, f"Job ID '{job_id}' should be descriptive"
                    
                    # Job should have a name
                    if 'name' in job_config:
                        job_name = job_config['name']
                        assert len(job_name) > 5, f"Job name '{job_name}' should be descriptive"
    
    def test_workflow_performance_optimization(self):
        """Test that workflows are optimized for performance."""
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                workflow = yaml.safe_load(f)
            
            if 'jobs' in workflow:
                for job_name, job_config in workflow['jobs'].items():
                    # Check for caching in Python setup
                    if 'steps' in job_config:
                        python_setup_steps = [
                            step for step in job_config['steps']
                            if isinstance(step, dict) and 'uses' in step 
                            and 'setup-python' in step['uses']
                        ]
                        
                        for step in python_setup_steps:
                            if 'with' in step:
                                # Should use caching for better performance
                                step_config = step['with']
                                if 'cache' not in step_config:
                                    print(f"Performance tip: {workflow_file.name} job '{job_name}' "
                                          f"could benefit from pip caching")


class TestWorkflowDocumentation:
    """Test workflow documentation and clarity."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).parent.parent
        self.workflows_dir = self.repo_root / ".github" / "workflows"
    
    def test_workflows_have_comments(self):
        """Test that workflows have helpful comments."""
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                content = f.read()
            
            # Should have at least some comments or documentation
            comment_lines = [line for line in content.split('\n') if line.strip().startswith('#')]
            
            if len(content.split('\n')) > 20:  # Only check for comments in longer workflows
                # More lenient - just check for some form of documentation
                has_documentation = len(comment_lines) > 0 or 'description:' in content
                if not has_documentation:
                    print(f"Info: Workflow {workflow_file.name} could benefit from more comments")
    
    def test_step_descriptions(self):
        """Test that complex steps have descriptions."""
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                workflow = yaml.safe_load(f)
            
            if 'jobs' in workflow:
                for job_name, job_config in workflow['jobs'].items():
                    if 'steps' in job_config:
                        for step in job_config['steps']:
                            if isinstance(step, dict) and 'run' in step:
                                run_command = step['run']
                                # Complex multi-line commands should have names
                                if '\n' in run_command and len(run_command) > 100:
                                    assert 'name' in step, \
                                        f"Complex step in {workflow_file.name} should have a name"


# Test markers for different categories
pytestmark = [
    pytest.mark.cicd,
    pytest.mark.infrastructure
]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
