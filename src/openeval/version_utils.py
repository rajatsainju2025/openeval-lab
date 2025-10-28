"""Version management and release utilities for OpenEval Lab."""

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def get_current_version() -> str:
    """Get current version from pyproject.toml.

    Returns:
        Current version string (e.g., "0.1.0")
    """
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        return "0.0.0"

    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    return "0.0.0"


def bump_version(current: str, part: str = "patch") -> str:
    """Bump version number.

    Args:
        current: Current version string (e.g., "0.1.0")
        part: Which part to bump: "major", "minor", or "patch"

    Returns:
        New version string

    Examples:
        >>> bump_version("0.1.0", "patch")
        "0.1.1"
        >>> bump_version("0.1.0", "minor")
        "0.2.0"
        >>> bump_version("0.1.0", "major")
        "1.0.0"
    """
    parts = current.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current}")

    major, minor, patch = map(int, parts)

    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    elif part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid part: {part}. Use 'major', 'minor', or 'patch'")


def update_version_in_file(new_version: str) -> None:
    """Update version in pyproject.toml.

    Args:
        new_version: New version string
    """
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        console.print("[red]Error: pyproject.toml not found[/red]")
        return

    content = pyproject_path.read_text()
    updated = re.sub(r'version\s*=\s*"[^"]+"', f'version = "{new_version}"', content)
    pyproject_path.write_text(updated)
    console.print(f"[green]✓[/green] Updated version to {new_version} in pyproject.toml")


def get_git_commits(since_tag: Optional[str] = None) -> List[Dict[str, str]]:
    """Get git commits since a specific tag or all commits.

    Args:
        since_tag: Git tag to start from (e.g., "v0.1.0"). If None, gets all commits.

    Returns:
        List of commit dictionaries with keys: hash, type, scope, message, body
    """
    try:
        if since_tag:
            # Get commits since tag
            cmd = ["git", "log", f"{since_tag}..HEAD", "--pretty=format:%H|||%s|||%b"]
        else:
            # Get recent commits (last 50)
            cmd = ["git", "log", "-50", "--pretty=format:%H|||%s|||%b"]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commits = []

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("|||")
            if len(parts) < 2:
                continue

            commit_hash = parts[0]
            subject = parts[1]
            body = parts[2] if len(parts) > 2 else ""

            # Parse conventional commit format
            match = re.match(
                r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)(\([^)]+\))?: (.+)$",
                subject,
            )
            if match:
                commit_type = match.group(1)
                scope = match.group(2).strip("()") if match.group(2) else ""
                message = match.group(3)
            else:
                commit_type = "other"
                scope = ""
                message = subject

            commits.append(
                {
                    "hash": commit_hash[:7],
                    "type": commit_type,
                    "scope": scope,
                    "message": message,
                    "body": body,
                }
            )

        return commits
    except subprocess.CalledProcessError:
        console.print("[yellow]Warning: Could not get git history[/yellow]")
        return []


def generate_changelog(
    commits: List[Dict[str, str]], version: str, date: Optional[str] = None
) -> str:
    """Generate CHANGELOG.md content from commits.

    Args:
        commits: List of commit dictionaries
        version: Version number for this release
        date: Release date (defaults to today)

    Returns:
        Formatted changelog content
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # Group commits by type
    grouped: Dict[str, List[Dict[str, str]]] = {
        "feat": [],
        "fix": [],
        "docs": [],
        "perf": [],
        "refactor": [],
        "test": [],
        "chore": [],
        "other": [],
    }

    for commit in commits:
        commit_type = commit["type"]
        if commit_type in grouped:
            grouped[commit_type].append(commit)
        else:
            grouped["other"].append(commit)

    # Generate changelog sections
    sections = []

    type_headers = {
        "feat": "✨ Features",
        "fix": "🐛 Bug Fixes",
        "docs": "📚 Documentation",
        "perf": "⚡ Performance",
        "refactor": "♻️  Refactoring",
        "test": "✅ Tests",
        "chore": "🔧 Chores",
        "other": "📦 Other",
    }

    for commit_type, header in type_headers.items():
        if grouped[commit_type]:
            sections.append(f"### {header}\n")
            for commit in grouped[commit_type]:
                scope_str = f"**{commit['scope']}**: " if commit["scope"] else ""
                sections.append(f"- {scope_str}{commit['message']} ({commit['hash']})")
            sections.append("")

    changelog = f"""## [{version}] - {date}

{chr(10).join(sections)}
"""
    return changelog


def prepend_to_changelog(new_content: str) -> None:
    """Prepend new content to CHANGELOG.md.

    Args:
        new_content: New changelog content to prepend
    """
    changelog_path = Path(__file__).parent.parent.parent / "CHANGELOG.md"

    if changelog_path.exists():
        existing = changelog_path.read_text()
    else:
        existing = """# Changelog

All notable changes to OpenEval Lab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""

    # Insert new content after the header
    lines = existing.split("\n")
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("## ["):
            header_end = i
            break

    if header_end == 0:
        # No existing releases, append after header
        header_end = len(lines)

    new_lines = lines[:header_end] + ["", new_content] + lines[header_end:]
    changelog_path.write_text("\n".join(new_lines))

    console.print("[green]✓[/green] Updated CHANGELOG.md")


def show_version_info() -> None:
    """Display current version and git information."""
    current = get_current_version()

    try:
        # Get latest tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True, check=True
        )
        latest_tag = result.stdout.strip()
    except subprocess.CalledProcessError:
        latest_tag = "No tags found"

    try:
        # Get commit count since tag
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"], capture_output=True, text=True, check=True
        )
        commit_count = result.stdout.strip()
    except subprocess.CalledProcessError:
        commit_count = "Unknown"

    table = Table(title="OpenEval Lab Version Info", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Current Version", current)
    table.add_row("Latest Git Tag", latest_tag)
    table.add_row("Total Commits", commit_count)

    console.print(table)


def create_release(version_part: str = "patch", dry_run: bool = False) -> Tuple[str, str]:
    """Create a new release with version bump and changelog.

    Args:
        version_part: Which version part to bump ("major", "minor", "patch")
        dry_run: If True, show what would happen without making changes

    Returns:
        Tuple of (new_version, changelog_content)
    """
    current = get_current_version()
    new_version = bump_version(current, version_part)

    console.print(
        Panel(
            f"[cyan]Current version:[/cyan] {current}\n"
            f"[green]New version:[/green] {new_version}",
            title="🚀 Release Preparation",
            border_style="blue",
        )
    )

    # Get commits since last tag
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True, check=True
        )
        since_tag = result.stdout.strip()
    except subprocess.CalledProcessError:
        since_tag = None

    commits = get_git_commits(since_tag)

    if not commits:
        console.print("[yellow]No new commits found for changelog[/yellow]")
        return new_version, ""

    console.print(f"\n[cyan]Found {len(commits)} commits since {since_tag or 'start'}[/cyan]\n")

    changelog = generate_changelog(commits, new_version)

    console.print("[bold]Generated Changelog:[/bold]")
    console.print(Panel(changelog, border_style="green"))

    if not dry_run:
        # Update version
        update_version_in_file(new_version)

        # Update changelog
        prepend_to_changelog(changelog)

        console.print("\n[green]✓[/green] Release preparation complete!")
        console.print("\n[yellow]Next steps:[/yellow]")
        console.print("1. Review the changes")
        console.print(f"2. Commit: git add -A && git commit -m 'chore: release v{new_version}'")
        console.print(f"3. Tag: git tag -a v{new_version} -m 'Release v{new_version}'")
        console.print("4. Push: git push origin main --tags")
    else:
        console.print("\n[yellow]Dry run - no changes made[/yellow]")

    return new_version, changelog
