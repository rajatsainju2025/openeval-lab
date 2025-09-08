#!/usr/bin/env python3
"""
OpenEval Lab Project Health Dashboard

Analyzes project health metrics, code quality, documentation coverage,
and development progress. Provides actionable insights for maintainers.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import os
import re


class ProjectHealthDashboard:
    """Comprehensive project health analysis and reporting."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize dashboard with project root directory."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.repo_root = self.project_root
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.docs_dir = self.project_root / "docs"
        
        # Initialize metrics storage
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "project_info": {},
            "code_quality": {},
            "test_coverage": {},
            "documentation": {},
            "git_health": {},
            "dependencies": {},
            "performance": {},
            "recommendations": []
        }
    
    def run_command(self, command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run shell command safely."""
        try:
            return subprocess.run(
                command, 
                capture_output=capture_output, 
                text=True, 
                cwd=self.project_root,
                timeout=30
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return subprocess.CompletedProcess(command, 1, "", str(e))
    
    def analyze_project_info(self) -> Dict[str, Any]:
        """Analyze basic project information."""
        info = {
            "name": "OpenEval Lab",
            "root_path": str(self.project_root),
            "python_files": 0,
            "test_files": 0,
            "doc_files": 0,
            "config_files": 0,
            "total_lines": 0
        }
        
        # Count files and lines
        if self.src_dir.exists():
            py_files = list(self.src_dir.rglob("*.py"))
            info["python_files"] = len(py_files)
            
            for py_file in py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        info["total_lines"] += len(f.readlines())
                except (UnicodeDecodeError, OSError):
                    pass
        
        if self.tests_dir.exists():
            info["test_files"] = len(list(self.tests_dir.rglob("test_*.py")))
        
        if self.docs_dir.exists():
            doc_files = list(self.docs_dir.rglob("*.md")) + list(self.project_root.glob("*.md"))
            info["doc_files"] = len(doc_files)
        
        # Count config files
        config_patterns = ["*.yml", "*.yaml", "*.json", "*.toml", "*.cfg", "*.ini"]
        for pattern in config_patterns:
            info["config_files"] += len(list(self.project_root.glob(pattern)))
            if (self.project_root / ".github").exists():
                info["config_files"] += len(list((self.project_root / ".github").rglob(pattern)))
        
        self.metrics["project_info"] = info
        return info
    
    def analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality metrics."""
        quality = {
            "style_issues": 0,
            "type_issues": 0,
            "complexity_issues": 0,
            "security_issues": 0,
            "quality_score": 0.0
        }
        
        # Check if black is installed and run formatting check
        black_result = self.run_command(["python", "-m", "black", "--check", "--diff", "src/", "tests/"])
        if black_result.returncode == 0:
            quality["formatting"] = "✅ Clean"
        else:
            quality["formatting"] = "⚠️ Issues found"
            quality["style_issues"] = len(black_result.stdout.split('\n')) if black_result.stdout else 1
        
        # Check flake8 if available
        flake8_result = self.run_command(["python", "-m", "flake8", "src/", "tests/", "--count", "--statistics"])
        if flake8_result.returncode == 0:
            quality["linting"] = "✅ Clean"
        else:
            quality["linting"] = "⚠️ Issues found"
            # Parse flake8 output for issue count
            if flake8_result.stdout:
                lines = flake8_result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip().isdigit():
                        quality["style_issues"] += int(line.strip())
        
        # Calculate quality score
        total_files = self.metrics.get("project_info", {}).get("python_files", 1)
        quality["quality_score"] = max(0.0, 100.0 - (quality["style_issues"] / total_files * 10))
        
        self.metrics["code_quality"] = quality
        return quality
    
    def analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage and test health."""
        coverage = {
            "test_files": 0,
            "test_functions": 0,
            "coverage_percentage": 0.0,
            "test_status": "unknown"
        }
        
        if self.tests_dir.exists():
            test_files = list(self.tests_dir.rglob("test_*.py"))
            coverage["test_files"] = len(test_files)
            
            # Count test functions
            test_function_count = 0
            for test_file in test_files:
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        test_function_count += len(re.findall(r'def test_\w+', content))
                except (UnicodeDecodeError, OSError):
                    pass
            
            coverage["test_functions"] = test_function_count
        
        # Run pytest to check test status
        pytest_result = self.run_command(["python", "-m", "pytest", "--collect-only", "-q"])
        if pytest_result.returncode == 0:
            coverage["test_status"] = "✅ Tests found and collectible"
            # Try to extract test count from output
            if pytest_result.stdout:
                lines = pytest_result.stdout.split('\n')
                for line in lines:
                    if "collected" in line:
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            coverage["collected_tests"] = int(numbers[0])
        else:
            coverage["test_status"] = "⚠️ Test collection issues"
        
        # Estimate coverage based on test-to-code ratio
        python_files = self.metrics.get("project_info", {}).get("python_files", 1)
        test_files = coverage["test_files"]
        coverage["coverage_percentage"] = min(100.0, (test_files / python_files) * 100 * 0.8)
        
        self.metrics["test_coverage"] = coverage
        return coverage
    
    def analyze_documentation(self) -> Dict[str, Any]:
        """Analyze documentation coverage and quality."""
        docs = {
            "readme_exists": False,
            "changelog_exists": False,
            "contributing_exists": False,
            "docs_directory": False,
            "api_docs": False,
            "doc_coverage_score": 0.0
        }
        
        # Check for essential documentation files
        docs["readme_exists"] = (self.project_root / "README.md").exists()
        docs["changelog_exists"] = (self.project_root / "CHANGELOG.md").exists()
        docs["contributing_exists"] = (self.project_root / "CONTRIBUTING.md").exists()
        docs["docs_directory"] = self.docs_dir.exists() and any(self.docs_dir.iterdir())
        
        # Check for API documentation in Python files
        if self.src_dir.exists():
            docstring_count = 0
            total_functions = 0
            
            for py_file in self.src_dir.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Count functions and classes
                        functions = re.findall(r'def \w+\(', content)
                        classes = re.findall(r'class \w+\(?', content)
                        total_functions += len(functions) + len(classes)
                        
                        # Count docstrings (simple heuristic)
                        docstrings = re.findall(r'"""[\s\S]*?"""', content)
                        docstring_count += len(docstrings)
                        
                except (UnicodeDecodeError, OSError):
                    pass
            
            if total_functions > 0:
                docs["api_docs"] = f"{docstring_count}/{total_functions} documented"
                api_coverage = (docstring_count / total_functions) * 100
            else:
                api_coverage = 0
        else:
            api_coverage = 0
        
        # Calculate documentation coverage score
        essential_docs = sum([
            docs["readme_exists"],
            docs["changelog_exists"], 
            docs["contributing_exists"],
            docs["docs_directory"]
        ])
        
        docs["doc_coverage_score"] = (essential_docs / 4 * 50) + (api_coverage * 0.5)
        
        self.metrics["documentation"] = docs
        return docs
    
    def analyze_git_health(self) -> Dict[str, Any]:
        """Analyze git repository health."""
        git = {
            "is_git_repo": False,
            "recent_commits": 0,
            "branches": [],
            "uncommitted_changes": False,
            "commit_frequency": "unknown"
        }
        
        # Check if this is a git repository
        git_result = self.run_command(["git", "status", "--porcelain"])
        if git_result.returncode == 0:
            git["is_git_repo"] = True
            git["uncommitted_changes"] = bool(git_result.stdout.strip())
            
            # Get recent commit count (last 30 days)
            since_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            commit_result = self.run_command(["git", "rev-list", "--count", f"--since={since_date}", "HEAD"])
            if commit_result.returncode == 0 and commit_result.stdout.strip().isdigit():
                git["recent_commits"] = int(commit_result.stdout.strip())
                
                # Determine commit frequency
                if git["recent_commits"] > 20:
                    git["commit_frequency"] = "🔥 Very Active"
                elif git["recent_commits"] > 10:
                    git["commit_frequency"] = "✅ Active"
                elif git["recent_commits"] > 3:
                    git["commit_frequency"] = "⚠️ Moderate"
                else:
                    git["commit_frequency"] = "❌ Low Activity"
            
            # Get branch information
            branch_result = self.run_command(["git", "branch", "-a"])
            if branch_result.returncode == 0:
                branches = [line.strip().replace('* ', '').replace('remotes/origin/', '') 
                           for line in branch_result.stdout.split('\n') 
                           if line.strip() and not line.strip().startswith('HEAD')]
                git["branches"] = list(set(branches))[:10]  # Limit to 10 branches
        
        self.metrics["git_health"] = git
        return git
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies and security."""
        deps = {
            "requirements_files": [],
            "dependency_count": 0,
            "security_issues": 0,
            "outdated_packages": 0
        }
        
        # Check for requirements files
        req_files = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile", "poetry.lock"]
        for req_file in req_files:
            if (self.project_root / req_file).exists():
                deps["requirements_files"].append(req_file)
        
        # Try to get pip list
        pip_result = self.run_command(["pip", "list", "--format=json"])
        if pip_result.returncode == 0:
            try:
                packages = json.loads(pip_result.stdout)
                deps["dependency_count"] = len(packages)
            except json.JSONDecodeError:
                pass
        
        # Check for security issues with safety (if available)
        safety_result = self.run_command(["python", "-m", "safety", "check", "--json"])
        if safety_result.returncode == 0:
            try:
                safety_data = json.loads(safety_result.stdout)
                deps["security_issues"] = len(safety_data)
            except json.JSONDecodeError:
                deps["security_issues"] = 0
        
        self.metrics["dependencies"] = deps
        return deps
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Code quality recommendations
        code_quality = self.metrics.get("code_quality", {})
        if code_quality.get("quality_score", 100) < 80:
            recommendations.append("🔧 Consider running 'make format' and 'make lint' to improve code quality")
        
        # Test coverage recommendations
        test_coverage = self.metrics.get("test_coverage", {})
        if test_coverage.get("coverage_percentage", 0) < 60:
            recommendations.append("🧪 Add more tests to improve coverage (target: >80%)")
        
        # Documentation recommendations
        docs = self.metrics.get("documentation", {})
        if not docs.get("changelog_exists", False):
            recommendations.append("📝 Create CHANGELOG.md to track project changes")
        if docs.get("doc_coverage_score", 0) < 70:
            recommendations.append("📚 Improve documentation coverage with more docstrings and guides")
        
        # Git health recommendations
        git = self.metrics.get("git_health", {})
        if git.get("uncommitted_changes", False):
            recommendations.append("🔄 Commit uncommitted changes")
        if git.get("recent_commits", 0) < 5:
            recommendations.append("⏰ Consider more frequent commits for better development tracking")
        
        # Security recommendations
        deps = self.metrics.get("dependencies", {})
        if deps.get("security_issues", 0) > 0:
            recommendations.append("🔒 Address security vulnerabilities in dependencies")
        
        # Performance recommendations
        project_info = self.metrics.get("project_info", {})
        if project_info.get("python_files", 0) > 50 and test_coverage.get("test_files", 0) < 10:
            recommendations.append("🚀 Consider adding integration tests for better coverage")
        
        self.metrics["recommendations"] = recommendations
        return recommendations
    
    def calculate_overall_health_score(self) -> float:
        """Calculate overall project health score (0-100)."""
        scores = []
        
        # Code quality score (weight: 25%)
        code_quality = self.metrics.get("code_quality", {})
        scores.append(code_quality.get("quality_score", 50) * 0.25)
        
        # Test coverage score (weight: 25%)
        test_coverage = self.metrics.get("test_coverage", {})
        scores.append(test_coverage.get("coverage_percentage", 0) * 0.25)
        
        # Documentation score (weight: 20%)
        docs = self.metrics.get("documentation", {})
        scores.append(docs.get("doc_coverage_score", 0) * 0.20)
        
        # Git activity score (weight: 15%)
        git = self.metrics.get("git_health", {})
        git_score = min(100, git.get("recent_commits", 0) * 5)  # 20 commits = 100%
        scores.append(git_score * 0.15)
        
        # Security score (weight: 15%)
        deps = self.metrics.get("dependencies", {})
        security_issues = deps.get("security_issues", 0)
        security_score = max(0, 100 - security_issues * 20)
        scores.append(security_score * 0.15)
        
        return sum(scores)
    
    def print_dashboard(self):
        """Print formatted dashboard to console."""
        print("🚀 OpenEval Lab - Project Health Dashboard")
        print("=" * 50)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Overall health score
        health_score = self.calculate_overall_health_score()
        if health_score >= 80:
            health_emoji = "🟢"
        elif health_score >= 60:
            health_emoji = "🟡"
        else:
            health_emoji = "🔴"
        
        print(f"\n{health_emoji} Overall Health Score: {health_score:.1f}/100")
        
        # Project info
        print(f"\n📊 Project Overview:")
        info = self.metrics.get("project_info", {})
        print(f"  • Python files: {info.get('python_files', 0)}")
        print(f"  • Total lines: {info.get('total_lines', 0):,}")
        print(f"  • Test files: {info.get('test_files', 0)}")
        print(f"  • Documentation files: {info.get('doc_files', 0)}")
        print(f"  • Configuration files: {info.get('config_files', 0)}")
        
        # Code quality
        print(f"\n🔧 Code Quality:")
        quality = self.metrics.get("code_quality", {})
        print(f"  • Quality Score: {quality.get('quality_score', 0):.1f}/100")
        print(f"  • Formatting: {quality.get('formatting', 'Unknown')}")
        print(f"  • Linting: {quality.get('linting', 'Unknown')}")
        
        # Test coverage
        print(f"\n🧪 Test Coverage:")
        coverage = self.metrics.get("test_coverage", {})
        print(f"  • Coverage: {coverage.get('coverage_percentage', 0):.1f}%")
        print(f"  • Test functions: {coverage.get('test_functions', 0)}")
        print(f"  • Status: {coverage.get('test_status', 'Unknown')}")
        
        # Documentation
        print(f"\n📚 Documentation:")
        docs = self.metrics.get("documentation", {})
        print(f"  • Coverage Score: {docs.get('doc_coverage_score', 0):.1f}/100")
        print(f"  • README: {'✅' if docs.get('readme_exists') else '❌'}")
        print(f"  • CHANGELOG: {'✅' if docs.get('changelog_exists') else '❌'}")
        print(f"  • Contributing Guide: {'✅' if docs.get('contributing_exists') else '❌'}")
        print(f"  • Docs Directory: {'✅' if docs.get('docs_directory') else '❌'}")
        
        # Git health
        print(f"\n🔄 Git Health:")
        git = self.metrics.get("git_health", {})
        print(f"  • Recent Commits (30d): {git.get('recent_commits', 0)}")
        print(f"  • Activity Level: {git.get('commit_frequency', 'Unknown')}")
        print(f"  • Uncommitted Changes: {'⚠️ Yes' if git.get('uncommitted_changes') else '✅ Clean'}")
        
        # Dependencies
        print(f"\n📦 Dependencies:")
        deps = self.metrics.get("dependencies", {})
        print(f"  • Total Packages: {deps.get('dependency_count', 0)}")
        print(f"  • Security Issues: {deps.get('security_issues', 0)}")
        print(f"  • Requirements Files: {', '.join(deps.get('requirements_files', []))}")
        
        # Recommendations
        recommendations = self.metrics.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations[:5]:  # Show top 5
                print(f"  • {rec}")
        
        print(f"\n🎯 Next Steps:")
        print(f"  • Run 'make test' to validate current state")
        print(f"  • Check 'make validate' for configuration issues")
        print(f"  • Review docs/ for latest documentation")
        print(f"  • Visit GitHub Actions for CI/CD status")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete project health analysis."""
        print("🔍 Analyzing project health...")
        
        self.analyze_project_info()
        self.analyze_code_quality()
        self.analyze_test_coverage()
        self.analyze_documentation()
        self.analyze_git_health()
        self.analyze_dependencies()
        self.generate_recommendations()
        
        return self.metrics
    
    def save_report(self, output_file: Optional[Path] = None):
        """Save detailed report to JSON file."""
        if not output_file:
            output_file = self.project_root / "project-health-report.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved to: {output_file}")


def main():
    """Main entry point for the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenEval Lab Project Health Dashboard")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--save", metavar="FILE", help="Save detailed report to file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    try:
        dashboard = ProjectHealthDashboard()
        metrics = dashboard.run_analysis()
        
        if args.json:
            print(json.dumps(metrics, indent=2, default=str))
        elif not args.quiet:
            dashboard.print_dashboard()
        
        if args.save:
            dashboard.save_report(Path(args.save))
        elif not args.json and not args.quiet:
            # Auto-save report
            dashboard.save_report()
        
        # Exit with appropriate code based on health
        health_score = dashboard.calculate_overall_health_score()
        if health_score < 50:
            sys.exit(1)  # Critical issues
        elif health_score < 70:
            sys.exit(2)  # Warning level
        else:
            sys.exit(0)  # Healthy
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
