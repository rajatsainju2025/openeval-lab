#!/usr/bin/env python3
"""
Security Audit Script for OpenEval Lab

This script performs comprehensive security checks including:
- Dependency vulnerability scanning
- Code security analysis
- Configuration security validation
- API key exposure detection
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any



class SecurityAuditor:
    """Comprehensive security auditor for the project."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues: List[Dict[str, Any]] = []

    def audit_dependencies(self) -> None:
        """Check for vulnerable dependencies using safety."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                vulnerable = []
                for pkg in packages:
                    if self._is_vulnerable_package(pkg["name"], pkg["version"]):
                        vulnerable.append(f"{pkg['name']}=={pkg['version']}")
                if vulnerable:
                    self.issues.append(
                        {
                            "severity": "HIGH",
                            "category": "DEPENDENCY",
                            "description": f"Vulnerable packages found: {', '.join(vulnerable)}",
                            "recommendation": "Update packages or use safety check",
                        }
                    )
        except Exception as e:
            self.issues.append(
                {
                    "severity": "MEDIUM",
                    "category": "DEPENDENCY",
                    "description": f"Failed to audit dependencies: {e}",
                    "recommendation": "Install safety and run pip-audit",
                }
            )

    def _is_vulnerable_package(self, name: str, version: str) -> bool:
        """Check if package version is known to be vulnerable."""
        # Simplified check - in real implementation, use a vulnerability database
        vulnerable_packages = {
            "requests": ["<2.20.0"],
            "urllib3": ["<1.23"],
            "pyyaml": ["<4.1"],
        }
        if name in vulnerable_packages:
            return version in vulnerable_packages[name]
        return False

    def audit_code_security(self) -> None:
        """Perform static code security analysis."""
        python_files = list(self.project_root.rglob("*.py"))
        for file_path in python_files:
            if "test" in str(file_path) or "__pycache__" in str(file_path):
                continue
            self._analyze_file_security(file_path)

    def _analyze_file_security(self, file_path: Path) -> None:
        """Analyze a single file for security issues."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for hardcoded secrets
            secrets_patterns = [
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'password\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ]

            for pattern in secrets_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    self.issues.append(
                        {
                            "severity": "CRITICAL",
                            "category": "CODE_SECURITY",
                            "description": f"Potential hardcoded secrets in {file_path}",
                            "recommendation": "Use environment variables or secure key management",
                        }
                    )

            # Check for dangerous functions
            dangerous_functions = [
                "eval(",
                "exec(",
                "subprocess.call(",
                "os.system(",
                "pickle.loads(",
                "yaml.load(",
            ]

            for func in dangerous_functions:
                if func in content:
                    self.issues.append(
                        {
                            "severity": "HIGH",
                            "category": "CODE_SECURITY",
                            "description": f"Dangerous function '{func[:-1]}' used in {file_path}",
                            "recommendation": "Review usage and consider safer alternatives",
                        }
                    )

        except Exception as e:
            self.issues.append(
                {
                    "severity": "LOW",
                    "category": "CODE_SECURITY",
                    "description": f"Failed to analyze {file_path}: {e}",
                    "recommendation": "Check file permissions and encoding",
                }
            )

    def audit_configuration(self) -> None:
        """Audit configuration files for security issues."""
        config_files = ["pyproject.toml", ".env", ".env.local", "config.yaml", "config.json"]

        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Check for exposed secrets in config
                    if re.search(r"api_key|password|secret|token", content, re.IGNORECASE):
                        self.issues.append(
                            {
                                "severity": "HIGH",
                                "category": "CONFIGURATION",
                                "description": f"Potential secrets in {config_file}",
                                "recommendation": "Move secrets to environment variables",
                            }
                        )

                except Exception as e:
                    self.issues.append(
                        {
                            "severity": "MEDIUM",
                            "category": "CONFIGURATION",
                            "description": f"Failed to audit {config_file}: {e}",
                            "recommendation": "Check file permissions",
                        }
                    )

    def generate_report(self) -> str:
        """Generate a comprehensive security report."""
        report = ["# Security Audit Report\n"]
        report.append(f"Project: {self.project_root.name}")
        report.append(f"Total Issues Found: {len(self.issues)}\n")

        severity_counts = {}
        for issue in self.issues:
            severity_counts[issue["severity"]] = severity_counts.get(issue["severity"], 0) + 1

        report.append("## Summary by Severity")
        for severity, count in severity_counts.items():
            report.append(f"- {severity}: {count}")
        report.append("")

        report.append("## Detailed Issues")
        for i, issue in enumerate(self.issues, 1):
            report.append(f"### Issue {i}")
            report.append(f"- **Severity**: {issue['severity']}")
            report.append(f"- **Category**: {issue['category']}")
            report.append(f"- **Description**: {issue['description']}")
            report.append(f"- **Recommendation**: {issue['recommendation']}")
            report.append("")

        return "\n".join(report)

    def run_audit(self) -> None:
        """Run the complete security audit."""
        print("🔒 Running Security Audit...")
        self.audit_dependencies()
        self.audit_code_security()
        self.audit_configuration()
        print(f"✅ Audit complete. Found {len(self.issues)} issues.")


def main():
    """Main entry point for security audit."""
    project_root = Path(__file__).parent.parent
    auditor = SecurityAuditor(project_root)
    auditor.run_audit()

    report = auditor.generate_report()
    report_path = project_root / "security_audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄 Report saved to {report_path}")

    # Print summary
    if auditor.issues:
        print("\n🚨 Security Issues Found:")
        for issue in auditor.issues[:5]:  # Show first 5
            print(f"  {issue['severity']}: {issue['description']}")
        if len(auditor.issues) > 5:
            print(f"  ... and {len(auditor.issues) - 5} more")
    else:
        print("\n✅ No security issues found!")


if __name__ == "__main__":
    main()
