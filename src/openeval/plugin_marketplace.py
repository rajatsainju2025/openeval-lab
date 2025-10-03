"""Plugin marketplace platform with discovery, validation, and community features.

This module provides a comprehensive plugin ecosystem including marketplace discovery,
security validation, version management, community features, and automated testing.
"""

import json
import shutil
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import logging

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import git
    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False

try:
    from packaging import version as pkg_version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False
    # Fallback version class
    class MockVersion:
        def __init__(self, version_str):
            self.version_str = version_str
        def __str__(self):
            return self.version_str
    
    class MockVersionModule:
        Version = MockVersion
        class InvalidVersion(Exception):
            pass
    
    pkg_version = MockVersionModule()

try:
    from .enhanced_logging import get_logger
    from .security import SecurityManager as RealSecurityManager
    from .observability import record_metric, log_event
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    
    class RealSecurityManager:
        def scan_file(self, *args, **kwargs):
            return {'threats_found': 0, 'security_score': 100.0}
    
    def record_metric(*args, **kwargs):
        pass
    def log_event(*args, **kwargs):
        pass

logger = get_logger(__name__)


class PluginStatus(Enum):
    """Plugin lifecycle status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review" 
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


class PluginType(Enum):
    """Plugin categories."""
    TASK = "task"
    DATASET = "dataset"
    ADAPTER = "adapter"
    METRIC = "metric"
    EVALUATOR = "evaluator"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    VISUALIZATION = "visualization"
    INTEGRATION = "integration"
    UTILITY = "utility"


class SecurityScanResult(Enum):
    """Security scan results."""
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass
class PluginVersion:
    """Plugin version information."""
    version: str
    changelog: str
    release_date: datetime
    download_url: str
    security_scan: SecurityScanResult = SecurityScanResult.UNKNOWN
    compatibility: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    file_hash: str = ""
    file_size: int = 0


@dataclass
class PluginAuthor:
    """Plugin author information."""
    username: str
    display_name: str
    email: str
    profile_url: str = ""
    verified: bool = False
    reputation_score: float = 0.0
    plugins_published: int = 0


@dataclass
class PluginReview:
    """Plugin review and rating."""
    reviewer: str
    rating: int  # 1-5 stars
    review_text: str
    created_at: datetime
    helpful_count: int = 0
    verified_download: bool = False


@dataclass
class PluginStats:
    """Plugin usage statistics."""
    total_downloads: int = 0
    weekly_downloads: int = 0
    rating_average: float = 0.0
    rating_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    first_published: datetime = field(default_factory=datetime.now)


@dataclass
class PluginManifest:
    """Plugin metadata and manifest."""
    name: str
    display_name: str
    description: str
    plugin_type: PluginType
    author: PluginAuthor
    current_version: str
    min_openeval_version: str
    homepage: str = ""
    documentation_url: str = ""
    source_url: str = ""
    license: str = "MIT"
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    versions: List[PluginVersion] = field(default_factory=list)
    status: PluginStatus = PluginStatus.DRAFT
    stats: PluginStats = field(default_factory=PluginStats)
    reviews: List[PluginReview] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PluginSearchResult:
    """Plugin search result."""
    plugin: PluginManifest
    relevance_score: float
    match_reasons: List[str] = field(default_factory=list)


class PluginValidator:
    """Validates plugin structure and security."""
    
    def __init__(self, security_manager: Optional[RealSecurityManager] = None):
        self.security_manager = security_manager or RealSecurityManager()
        
        # Required files for a valid plugin
        self.required_files = {
            'plugin.json',  # Plugin manifest
            '__init__.py'   # Python package
        }
        
        # Dangerous patterns to check for
        self.security_patterns = [
            'eval(',
            'exec(',
            'compile(',
            '__import__',
            'subprocess.',
            'os.system',
            'open(',
            'file(',
            'input(',
            'raw_input('
        ]
    
    async def validate_plugin(self, plugin_path: Path) -> Dict[str, Any]:
        """Comprehensive plugin validation."""
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'security_issues': [],
            'structure_valid': False,
            'manifest_valid': False,
            'code_quality_score': 0.0,
            'security_score': 0.0,
            'documentation_score': 0.0,
            'overall_score': 0.0
        }
        
        try:
            # 1. Structure validation
            structure_result = await self._validate_structure(plugin_path)
            validation_result.update(structure_result)
            
            # 2. Manifest validation
            if validation_result['structure_valid']:
                manifest_result = await self._validate_manifest(plugin_path)
                validation_result.update(manifest_result)
            
            # 3. Security scan
            security_result = await self._security_scan(plugin_path)
            validation_result.update(security_result)
            
            # 4. Code quality check
            quality_result = await self._check_code_quality(plugin_path)
            validation_result.update(quality_result)
            
            # 5. Documentation check
            docs_result = await self._check_documentation(plugin_path)
            validation_result.update(docs_result)
            
            # Calculate overall score
            validation_result['overall_score'] = self._calculate_overall_score(validation_result)
            
            # Determine if plugin is valid
            validation_result['valid'] = (
                validation_result['structure_valid'] and
                validation_result['manifest_valid'] and
                validation_result['security_score'] >= 80.0 and
                len(validation_result['errors']) == 0
            )
            
        except Exception as e:
            validation_result['errors'].append(f"Validation failed: {str(e)}")
            logger.error(f"Plugin validation error: {e}")
        
        return validation_result
    
    async def _validate_structure(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate plugin directory structure."""
        result = {
            'structure_valid': False,
            'structure_errors': []
        }
        
        if not plugin_path.exists():
            result['structure_errors'].append("Plugin path does not exist")
            return result
        
        if not plugin_path.is_dir():
            result['structure_errors'].append("Plugin path is not a directory")
            return result
        
        # Check for required files
        missing_files = []
        for required_file in self.required_files:
            if not (plugin_path / required_file).exists():
                missing_files.append(required_file)
        
        if missing_files:
            result['structure_errors'].append(f"Missing required files: {', '.join(missing_files)}")
        else:
            result['structure_valid'] = True
        
        return result
    
    async def _validate_manifest(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate plugin manifest file."""
        result = {
            'manifest_valid': False,
            'manifest_errors': [],
            'manifest': None
        }
        
        manifest_path = plugin_path / 'plugin.json'
        
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            # Required manifest fields
            required_fields = [
                'name', 'display_name', 'description', 'version',
                'plugin_type', 'author', 'min_openeval_version'
            ]
            
            missing_fields = []
            for field in required_fields:
                if field not in manifest_data:
                    missing_fields.append(field)
            
            if missing_fields:
                result['manifest_errors'].append(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Validate field formats
            if 'plugin_type' in manifest_data:
                try:
                    PluginType(manifest_data['plugin_type'])
                except ValueError:
                    result['manifest_errors'].append(f"Invalid plugin_type: {manifest_data['plugin_type']}")
            
            # Validate version format
            if 'version' in manifest_data and HAS_PACKAGING:
                try:
                    pkg_version.Version(manifest_data['version'])
                except pkg_version.InvalidVersion:
                    result['manifest_errors'].append(f"Invalid version format: {manifest_data['version']}")
            
            if not result['manifest_errors']:
                result['manifest_valid'] = True
                result['manifest'] = manifest_data
            
        except json.JSONDecodeError as e:
            result['manifest_errors'].append(f"Invalid JSON in manifest: {str(e)}")
        except Exception as e:
            result['manifest_errors'].append(f"Error reading manifest: {str(e)}")
        
        return result
    
    async def _security_scan(self, plugin_path: Path) -> Dict[str, Any]:
        """Perform security scan on plugin code."""
        result = {
            'security_score': 0.0,
            'security_issues': [],
            'security_warnings': []
        }
        
        security_score = 100.0
        
        # Scan Python files for dangerous patterns
        for py_file in plugin_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in self.security_patterns:
                    if pattern in content:
                        result['security_issues'].append(
                            f"Potentially dangerous pattern '{pattern}' found in {py_file.name}"
                        )
                        security_score -= 10.0
                
                # Check for suspicious imports
                dangerous_imports = ['subprocess', 'os', 'sys', '__builtin__', 'builtins']
                for imp in dangerous_imports:
                    if f'import {imp}' in content or f'from {imp}' in content:
                        result['security_warnings'].append(
                            f"Potentially dangerous import '{imp}' in {py_file.name}"
                        )
                        security_score -= 5.0
                
            except Exception as e:
                result['security_warnings'].append(f"Could not scan {py_file.name}: {str(e)}")
        
        # Use security manager for deep scan if available
        try:
            scan_result = self.security_manager.scan_file(str(plugin_path))
            if scan_result.get('threats_found', 0) > 0:
                security_score -= scan_result['threats_found'] * 20.0
                result['security_issues'].extend(scan_result.get('threats', []))
        except Exception as e:
            logger.warning(f"Security manager scan failed: {e}")
        
        result['security_score'] = max(0.0, security_score)
        return result
    
    async def _check_code_quality(self, plugin_path: Path) -> Dict[str, Any]:
        """Check code quality metrics."""
        result = {
            'code_quality_score': 0.0,
            'quality_issues': []
        }
        
        quality_score = 80.0  # Base score
        
        # Check for Python files
        python_files = list(plugin_path.rglob('*.py'))
        if not python_files:
            result['quality_issues'].append("No Python files found")
            return result
        
        # Basic quality checks
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                # Check for docstrings
                if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                    quality_score -= 5.0
                    result['quality_issues'].append(f"Missing module docstring in {py_file.name}")
                
                # Check for very long lines (basic check)
                long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 120]
                if long_lines:
                    quality_score -= min(len(long_lines), 10)
                    result['quality_issues'].append(f"Long lines (>120 chars) in {py_file.name}")
                
                # Check for TODO/FIXME comments
                todos = [i for i, line in enumerate(lines, 1) if 'TODO' in line or 'FIXME' in line]
                if len(todos) > 5:
                    quality_score -= 5.0
                    result['quality_issues'].append(f"Many TODO/FIXME comments in {py_file.name}")
                
            except Exception as e:
                result['quality_issues'].append(f"Could not analyze {py_file.name}: {str(e)}")
        
        result['code_quality_score'] = max(0.0, quality_score)
        return result
    
    async def _check_documentation(self, plugin_path: Path) -> Dict[str, Any]:
        """Check documentation quality."""
        result = {
            'documentation_score': 0.0,
            'documentation_issues': []
        }
        
        doc_score = 0.0
        
        # Check for README
        readme_files = ['README.md', 'README.rst', 'README.txt', 'readme.md']
        has_readme = any((plugin_path / readme).exists() for readme in readme_files)
        
        if has_readme:
            doc_score += 30.0
        else:
            result['documentation_issues'].append("No README file found")
        
        # Check for examples
        if (plugin_path / 'examples').exists():
            doc_score += 20.0
        else:
            result['documentation_issues'].append("No examples directory found")
        
        # Check for LICENSE
        license_files = ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'license.txt']
        has_license = any((plugin_path / license_f).exists() for license_f in license_files)
        
        if has_license:
            doc_score += 20.0
        else:
            result['documentation_issues'].append("No LICENSE file found")
        
        # Check for CHANGELOG
        changelog_files = ['CHANGELOG.md', 'CHANGELOG.rst', 'CHANGES.md', 'changelog.md']
        has_changelog = any((plugin_path / changelog).exists() for changelog in changelog_files)
        
        if has_changelog:
            doc_score += 15.0
        else:
            result['documentation_issues'].append("No CHANGELOG file found")
        
        # Check for tests
        if (plugin_path / 'tests').exists() or any(plugin_path.glob('test_*.py')):
            doc_score += 15.0
        else:
            result['documentation_issues'].append("No tests found")
        
        result['documentation_score'] = doc_score
        return result
    
    def _calculate_overall_score(self, validation_result: Dict[str, Any]) -> float:
        """Calculate overall plugin quality score."""
        weights = {
            'structure': 0.2,
            'security': 0.4,
            'code_quality': 0.2,
            'documentation': 0.2
        }
        
        scores = {
            'structure': 100.0 if validation_result.get('structure_valid', False) else 0.0,
            'security': validation_result.get('security_score', 0.0),
            'code_quality': validation_result.get('code_quality_score', 0.0),
            'documentation': validation_result.get('documentation_score', 0.0)
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        # Penalty for errors
        error_penalty = len(validation_result.get('errors', [])) * 10.0
        
        return max(0.0, overall_score - error_penalty)


class PluginRepository:
    """Manages plugin repository and marketplace."""
    
    def __init__(self, repo_path: Path, marketplace_url: Optional[str] = None):
        self.repo_path = repo_path
        self.marketplace_url = marketplace_url
        self.repo_path.mkdir(parents=True, exist_ok=True)
        
        self.validator = PluginValidator()
        self.plugins: Dict[str, PluginManifest] = {}
        
        # Load existing plugins
        self._load_plugins()
    
    def _load_plugins(self):
        """Load plugin manifests from repository."""
        for plugin_dir in self.repo_path.iterdir():
            if not plugin_dir.is_dir():
                continue
            
            manifest_path = plugin_dir / 'plugin.json'
            if not manifest_path.exists():
                continue
            
            try:
                with open(manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                
                # Convert to PluginManifest
                manifest = self._dict_to_manifest(manifest_data)
                self.plugins[manifest.name] = manifest
                
            except Exception as e:
                logger.error(f"Error loading plugin manifest from {plugin_dir}: {e}")
    
    def _dict_to_manifest(self, data: Dict[str, Any]) -> PluginManifest:
        """Convert dictionary to PluginManifest."""
        # Convert author data
        author_data = data.get('author', {})
        if isinstance(author_data, str):
            author = PluginAuthor(username=author_data, display_name=author_data, email="")
        else:
            author = PluginAuthor(**author_data)
        
        # Convert versions
        versions = []
        for version_data in data.get('versions', []):
            version_obj = PluginVersion(**version_data)
            versions.append(version_obj)
        
        # Convert reviews
        reviews = []
        for review_data in data.get('reviews', []):
            review = PluginReview(**review_data)
            reviews.append(review)
        
        # Convert stats
        stats_data = data.get('stats', {})
        stats = PluginStats(**stats_data)
        
        return PluginManifest(
            name=data['name'],
            display_name=data['display_name'],
            description=data['description'],
            plugin_type=PluginType(data['plugin_type']),
            author=author,
            current_version=data['current_version'],
            min_openeval_version=data['min_openeval_version'],
            homepage=data.get('homepage', ''),
            documentation_url=data.get('documentation_url', ''),
            source_url=data.get('source_url', ''),
            license=data.get('license', 'MIT'),
            keywords=data.get('keywords', []),
            categories=data.get('categories', []),
            versions=versions,
            status=PluginStatus(data.get('status', 'draft')),
            stats=stats,
            reviews=reviews,
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
        )
    
    async def submit_plugin(self, plugin_path: Path, author_info: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a new plugin for review."""
        result = {
            'success': False,
            'plugin_name': '',
            'validation_result': {},
            'message': ''
        }
        
        try:
            # Validate plugin
            validation_result = await self.validator.validate_plugin(plugin_path)
            result['validation_result'] = validation_result
            
            if not validation_result['valid']:
                result['message'] = "Plugin validation failed"
                return result
            
            # Load plugin manifest
            manifest_path = plugin_path / 'plugin.json'
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            plugin_name = manifest_data['name']
            result['plugin_name'] = plugin_name
            
            # Check if plugin already exists
            if plugin_name in self.plugins:
                result['message'] = "Plugin already exists. Use update_plugin instead."
                return result
            
            # Create plugin directory in repository
            plugin_repo_dir = self.repo_path / plugin_name
            if plugin_repo_dir.exists():
                shutil.rmtree(plugin_repo_dir)
            
            shutil.copytree(plugin_path, plugin_repo_dir)
            
            # Create plugin manifest with author info
            author = PluginAuthor(**author_info)
            manifest = PluginManifest(
                name=manifest_data['name'],
                display_name=manifest_data['display_name'],
                description=manifest_data['description'],
                plugin_type=PluginType(manifest_data['plugin_type']),
                author=author,
                current_version=manifest_data['version'],
                min_openeval_version=manifest_data['min_openeval_version'],
                homepage=manifest_data.get('homepage', ''),
                documentation_url=manifest_data.get('documentation_url', ''),
                source_url=manifest_data.get('source_url', ''),
                license=manifest_data.get('license', 'MIT'),
                keywords=manifest_data.get('keywords', []),
                categories=manifest_data.get('categories', []),
                status=PluginStatus.PENDING_REVIEW
            )
            
            # Add initial version
            version = PluginVersion(
                version=manifest_data['version'],
                changelog=manifest_data.get('changelog', 'Initial release'),
                release_date=datetime.now(),
                download_url='',  # Will be set when published
                security_scan=SecurityScanResult.SAFE if validation_result['security_score'] >= 80 else SecurityScanResult.WARNING,
                dependencies=manifest_data.get('dependencies', {})
            )
            manifest.versions.append(version)
            
            # Save manifest
            self.plugins[plugin_name] = manifest
            self._save_manifest(manifest)
            
            result['success'] = True
            result['message'] = f"Plugin '{plugin_name}' submitted successfully for review"
            
            # Record metrics
            record_metric("plugin_submitted", 1, "counter", {"plugin_type": manifest.plugin_type.value})
            log_event("info", f"Plugin submitted: {plugin_name}", 
                     validation_score=validation_result['overall_score'])
            
        except Exception as e:
            result['message'] = f"Error submitting plugin: {str(e)}"
            logger.error(f"Plugin submission error: {e}")
        
        return result
    
    def search_plugins(self, query: str, plugin_type: Optional[PluginType] = None, 
                      category: Optional[str] = None, limit: int = 20) -> List[PluginSearchResult]:
        """Search for plugins in the marketplace."""
        results = []
        
        query_lower = query.lower()
        
        for plugin in self.plugins.values():
            # Skip non-published plugins
            if plugin.status not in [PluginStatus.PUBLISHED, PluginStatus.APPROVED]:
                continue
            
            # Filter by type
            if plugin_type and plugin.plugin_type != plugin_type:
                continue
            
            # Filter by category
            if category and category not in plugin.categories:
                continue
            
            # Calculate relevance score
            relevance_score = 0.0
            match_reasons = []
            
            # Name match (highest weight)
            if query_lower in plugin.name.lower():
                relevance_score += 50.0
                match_reasons.append("name match")
            
            # Display name match
            if query_lower in plugin.display_name.lower():
                relevance_score += 40.0
                match_reasons.append("display name match")
            
            # Description match
            if query_lower in plugin.description.lower():
                relevance_score += 20.0
                match_reasons.append("description match")
            
            # Keywords match
            for keyword in plugin.keywords:
                if query_lower in keyword.lower():
                    relevance_score += 15.0
                    match_reasons.append("keyword match")
                    break
            
            # Author match
            if query_lower in plugin.author.display_name.lower():
                relevance_score += 10.0
                match_reasons.append("author match")
            
            # Boost by rating and downloads
            relevance_score += plugin.stats.rating_average * 2
            relevance_score += min(plugin.stats.total_downloads / 1000, 10)
            
            if relevance_score > 0:
                results.append(PluginSearchResult(
                    plugin=plugin,
                    relevance_score=relevance_score,
                    match_reasons=match_reasons
                ))
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]
    
    def get_plugin(self, name: str) -> Optional[PluginManifest]:
        """Get plugin by name."""
        return self.plugins.get(name)
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None, 
                    status: Optional[PluginStatus] = None) -> List[PluginManifest]:
        """List all plugins with optional filters."""
        plugins = list(self.plugins.values())
        
        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]
        
        if status:
            plugins = [p for p in plugins if p.status == status]
        
        return plugins
    
    async def install_plugin(self, name: str, version: Optional[str] = None,
                           install_path: Optional[Path] = None) -> Dict[str, Any]:
        """Install a plugin."""
        result = {
            'success': False,
            'message': '',
            'installed_path': None
        }
        
        plugin = self.get_plugin(name)
        if not plugin:
            result['message'] = f"Plugin '{name}' not found"
            return result
        
        if plugin.status != PluginStatus.PUBLISHED:
            result['message'] = f"Plugin '{name}' is not published"
            return result
        
        # Find version to install
        if version:
            plugin_version = next((v for v in plugin.versions if v.version == version), None)
            if not plugin_version:
                result['message'] = f"Version '{version}' not found for plugin '{name}'"
                return result
        else:
            # Use latest version
            plugin_version = max(plugin.versions, key=lambda v: v.release_date)
        
        try:
            # Install to default path if not specified
            if not install_path:
                install_path = Path.home() / '.openeval' / 'plugins' / name
            
            install_path.mkdir(parents=True, exist_ok=True)
            
            # Copy plugin files
            plugin_source = self.repo_path / name
            if plugin_source.exists():
                # Copy from local repository
                for item in plugin_source.iterdir():
                    if item.is_file():
                        shutil.copy2(item, install_path)
                    elif item.is_dir() and item.name != '__pycache__':
                        shutil.copytree(item, install_path / item.name, dirs_exist_ok=True)
            
            result['success'] = True
            result['message'] = f"Plugin '{name}' version '{plugin_version.version}' installed successfully"
            result['installed_path'] = str(install_path)
            
            # Update download stats
            plugin.stats.total_downloads += 1
            plugin.stats.weekly_downloads += 1
            self._save_manifest(plugin)
            
            # Record metrics
            record_metric("plugin_installed", 1, "counter", 
                         {"plugin": name, "version": plugin_version.version})
            
        except Exception as e:
            result['message'] = f"Installation failed: {str(e)}"
            logger.error(f"Plugin installation error: {e}")
        
        return result
    
    def add_review(self, plugin_name: str, reviewer: str, rating: int, 
                  review_text: str, verified_download: bool = False) -> bool:
        """Add a review for a plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return False
        
        if not (1 <= rating <= 5):
            return False
        
        review = PluginReview(
            reviewer=reviewer,
            rating=rating,
            review_text=review_text,
            created_at=datetime.now(),
            verified_download=verified_download
        )
        
        plugin.reviews.append(review)
        
        # Update rating statistics
        total_rating = sum(r.rating for r in plugin.reviews)
        plugin.stats.rating_average = total_rating / len(plugin.reviews)
        plugin.stats.rating_count = len(plugin.reviews)
        plugin.stats.last_updated = datetime.now()
        
        self._save_manifest(plugin)
        
        record_metric("plugin_review_added", 1, "counter", 
                     {"plugin": plugin_name, "rating": rating})
        
        return True
    
    def _save_manifest(self, manifest: PluginManifest):
        """Save plugin manifest to file."""
        plugin_dir = self.repo_path / manifest.name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = plugin_dir / 'manifest.json'
        
        # Convert to dictionary for JSON serialization
        manifest_dict = asdict(manifest)
        
        # Convert datetime objects to ISO format
        manifest_dict['created_at'] = manifest.created_at.isoformat()
        manifest_dict['updated_at'] = manifest.updated_at.isoformat()
        
        for version in manifest_dict['versions']:
            version['release_date'] = version['release_date'].isoformat() if isinstance(version['release_date'], datetime) else version['release_date']
        
        for review in manifest_dict['reviews']:
            review['created_at'] = review['created_at'].isoformat() if isinstance(review['created_at'], datetime) else review['created_at']
        
        # Convert enums to strings
        manifest_dict['plugin_type'] = manifest.plugin_type.value
        manifest_dict['status'] = manifest.status.value
        
        for version in manifest_dict['versions']:
            if 'security_scan' in version:
                version['security_scan'] = version['security_scan'].value if hasattr(version['security_scan'], 'value') else version['security_scan']
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_dict, f, indent=2)


class PluginManager:
    """Main plugin management interface."""
    
    def __init__(self, repo_path: Path = Path("~/.openeval/plugin_repo").expanduser()):
        self.repository = PluginRepository(repo_path)
        self.installed_plugins: Dict[str, Path] = {}
        
        # Discover installed plugins
        self._discover_installed_plugins()
    
    def _discover_installed_plugins(self):
        """Discover already installed plugins."""
        plugins_dir = Path.home() / '.openeval' / 'plugins'
        if not plugins_dir.exists():
            return
        
        for plugin_dir in plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue
            
            # Check if it has a valid plugin manifest
            manifest_file = plugin_dir / 'plugin.json'
            if manifest_file.exists():
                self.installed_plugins[plugin_dir.name] = plugin_dir
    
    async def submit_plugin(self, plugin_path: Path, author_info: Dict[str, Any]) -> Dict[str, Any]:
        """Submit plugin to marketplace."""
        return await self.repository.submit_plugin(plugin_path, author_info)
    
    def search(self, query: str, **kwargs) -> List[PluginSearchResult]:
        """Search plugins in marketplace."""
        return self.repository.search_plugins(query, **kwargs)
    
    async def install(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Install a plugin."""
        result = await self.repository.install_plugin(name, version)
        if result['success']:
            self.installed_plugins[name] = Path(result['installed_path'])
        return result
    
    def uninstall(self, name: str) -> bool:
        """Uninstall a plugin."""
        if name not in self.installed_plugins:
            return False
        
        try:
            shutil.rmtree(self.installed_plugins[name])
            del self.installed_plugins[name]
            return True
        except Exception as e:
            logger.error(f"Error uninstalling plugin {name}: {e}")
            return False
    
    def list_installed(self) -> List[str]:
        """List installed plugins."""
        return list(self.installed_plugins.keys())
    
    def add_review(self, plugin_name: str, reviewer: str, rating: int, 
                  review_text: str) -> bool:
        """Add review for a plugin."""
        verified = plugin_name in self.installed_plugins
        return self.repository.add_review(plugin_name, reviewer, rating, review_text, verified)


# Example usage and CLI integration
def create_example_plugin(output_path: Path):
    """Create an example plugin for testing."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create plugin.json
    manifest = {
        "name": "example-sentiment-analyzer",
        "display_name": "Sentiment Analysis Plugin",
        "description": "A sample plugin that performs sentiment analysis on text data",
        "version": "1.0.0",
        "plugin_type": "metric",
        "min_openeval_version": "1.0.0",
        "author": "openeval-community",
        "homepage": "https://github.com/openeval/example-plugins",
        "license": "MIT",
        "keywords": ["sentiment", "nlp", "text", "analysis"],
        "categories": ["nlp", "text-processing"],
        "dependencies": {
            "textblob": ">=0.17.1"
        },
        "changelog": "Initial release with basic sentiment analysis functionality"
    }
    
    with open(output_path / 'plugin.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create __init__.py
    init_content = '''"""Sentiment Analysis Plugin for OpenEval.

This plugin provides sentiment analysis capabilities for text evaluation tasks.
"""

from .sentiment_analyzer import SentimentAnalyzer

__version__ = "1.0.0"
__all__ = ["SentimentAnalyzer"]
'''
    
    with open(output_path / '__init__.py', 'w') as f:
        f.write(init_content)
    
    # Create main plugin file
    plugin_content = '''"""Sentiment analyzer implementation."""

from typing import Dict, Any, List
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False


class SentimentAnalyzer:
    """Sentiment analysis metric plugin."""
    
    def __init__(self):
        if not HAS_TEXTBLOB:
            raise ImportError("textblob is required for sentiment analysis")
    
    def evaluate(self, predictions: List[str], references: List[str] = None) -> Dict[str, Any]:
        """Evaluate sentiment of predictions."""
        sentiments = []
        polarities = []
        
        for text in predictions:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            # Classify sentiment
            if sentiment.polarity > 0.1:
                sent_label = "positive"
            elif sentiment.polarity < -0.1:
                sent_label = "negative"
            else:
                sent_label = "neutral"
            
            sentiments.append(sent_label)
            polarities.append(sentiment.polarity)
        
        # Calculate statistics
        avg_polarity = sum(polarities) / len(polarities)
        sentiment_dist = {
            "positive": sentiments.count("positive") / len(sentiments),
            "negative": sentiments.count("negative") / len(sentiments), 
            "neutral": sentiments.count("neutral") / len(sentiments)
        }
        
        return {
            "average_polarity": avg_polarity,
            "sentiment_distribution": sentiment_dist,
            "sentiments": sentiments,
            "polarities": polarities
        }
    
    @property
    def name(self) -> str:
        return "sentiment_analyzer"
    
    @property
    def description(self) -> str:
        return "Analyzes sentiment polarity and distribution in text"
'''
    
    with open(output_path / 'sentiment_analyzer.py', 'w') as f:
        f.write(plugin_content)
    
    # Create README.md
    readme_content = '''# Sentiment Analysis Plugin

A sample OpenEval plugin that performs sentiment analysis on text data using TextBlob.

## Features

- Sentiment polarity analysis (-1 to 1 scale)
- Sentiment classification (positive, negative, neutral)
- Distribution statistics
- Easy integration with OpenEval evaluation pipelines

## Installation

```bash
openeval plugin install example-sentiment-analyzer
```

## Usage

```python
from openeval.plugins import SentimentAnalyzer

analyzer = SentimentAnalyzer()
results = analyzer.evaluate(["I love this!", "This is terrible"])
print(results)
```

## Requirements

- textblob >= 0.17.1

## License

MIT License
'''
    
    with open(output_path / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"Example plugin created at: {output_path}")


# Plugin marketplace CLI commands (integration point)
if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create example plugin
        example_path = Path("/tmp/example_plugin")
        create_example_plugin(example_path)
        
        # Initialize plugin manager
        manager = PluginManager()
        
        # Submit plugin
        author_info = {
            "username": "testuser",
            "display_name": "Test User", 
            "email": "test@example.com",
            "verified": False
        }
        
        result = await manager.submit_plugin(example_path, author_info)
        print(f"Submission result: {result}")
        
        # Search plugins
        search_results = manager.search("sentiment")
        print(f"Found {len(search_results)} plugins")
        
        for result in search_results:
            print(f"- {result.plugin.display_name} ({result.relevance_score:.1f})")
    
    asyncio.run(main())