"""Comprehensive tests for advanced registry system with metadata and dependencies.

This module tests the enhanced registry functionality including metadata management,
dependency resolution, versioning, caching, validation, and discovery features.
"""

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

import pytest

from src.openeval.advanced_registry import (
    ComponentStatus,
    DependencyType,
    Dependency,
    ComponentMetadata,
    DependencyResolver,
    AdvancedRegistry,
    get_advanced_registry,
    register_component_metadata,
    example_registry_setup,
)


class TestDependency:
    """Test dependency specification and validation."""

    def test_dependency_creation(self):
        """Test basic dependency creation."""
        dep = Dependency(
            name="test-component",
            version_constraint=">=1.0.0",
            dependency_type=DependencyType.REQUIRED,
            description="Test dependency",
        )

        assert dep.name == "test-component"
        assert dep.version_constraint == ">=1.0.0"
        assert dep.dependency_type == DependencyType.REQUIRED
        assert dep.description == "Test dependency"

    def test_dependency_validation_empty_name(self):
        """Test dependency validation with empty name."""
        with pytest.raises(ValueError, match="Dependency name cannot be empty"):
            Dependency(name="")

    def test_dependency_validation_invalid_version(self):
        """Test dependency validation with invalid version constraint."""
        with pytest.raises(ValueError, match="Invalid version constraint"):
            Dependency(name="test", version_constraint="invalid-version")

    def test_dependency_matches_simple(self):
        """Test dependency version matching."""
        dep = Dependency(name="test", version_constraint=">=1.0.0")

        assert dep.matches("1.0.0") is True
        assert dep.matches("1.1.0") is True
        assert dep.matches("2.0.0") is True
        # Note: Without packaging library, this might fall back to simple comparison

    def test_dependency_matches_wildcard(self):
        """Test wildcard dependency matching."""
        dep = Dependency(name="test", version_constraint="*")

        assert dep.matches("1.0.0") is True
        assert dep.matches("2.0.0") is True
        assert dep.matches("0.1.0") is True

    def test_dependency_matches_exact(self):
        """Test exact version dependency matching."""
        dep = Dependency(name="test", version_constraint="1.0.0")

        assert dep.matches("1.0.0") is True
        assert dep.matches("1.1.0") is False

    def test_dependency_types(self):
        """Test different dependency types."""
        required = Dependency(name="req", dependency_type=DependencyType.REQUIRED)
        optional = Dependency(name="opt", dependency_type=DependencyType.OPTIONAL)
        conflicts = Dependency(name="conf", dependency_type=DependencyType.CONFLICTS)
        suggests = Dependency(name="sug", dependency_type=DependencyType.SUGGESTS)

        assert required.dependency_type == DependencyType.REQUIRED
        assert optional.dependency_type == DependencyType.OPTIONAL
        assert conflicts.dependency_type == DependencyType.CONFLICTS
        assert suggests.dependency_type == DependencyType.SUGGESTS


class TestComponentMetadata:
    """Test component metadata management."""

    def test_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = ComponentMetadata(
            name="test-component",
            display_name="Test Component",
            description="A test component",
            version="1.0.0",
            import_path="test.module.TestComponent",
        )

        assert metadata.name == "test-component"
        assert metadata.display_name == "Test Component"
        assert metadata.version == "1.0.0"
        assert metadata.status == ComponentStatus.STABLE
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.dependencies, list)

    def test_metadata_validation_empty_name(self):
        """Test metadata validation with empty name."""
        with pytest.raises(ValueError, match="Component name cannot be empty"):
            ComponentMetadata(
                name="",
                display_name="Test",
                description="Test",
                version="1.0.0",
                import_path="test.module",
            )

    def test_metadata_validation_empty_import_path(self):
        """Test metadata validation with empty import path."""
        with pytest.raises(ValueError, match="Import path cannot be empty"):
            ComponentMetadata(
                name="test",
                display_name="Test",
                description="Test",
                version="1.0.0",
                import_path="",
            )

    def test_metadata_add_dependency(self):
        """Test adding dependencies to metadata."""
        metadata = ComponentMetadata(
            name="test",
            display_name="Test",
            description="Test",
            version="1.0.0",
            import_path="test.module",
        )

        initial_updated = metadata.updated_at

        metadata.add_dependency(
            "numpy", ">=1.20.0", DependencyType.REQUIRED, "Numerical computation library"
        )

        assert len(metadata.dependencies) == 1
        dep = metadata.dependencies[0]
        assert dep.name == "numpy"
        assert dep.version_constraint == ">=1.20.0"
        assert dep.dependency_type == DependencyType.REQUIRED
        assert metadata.updated_at > initial_updated

    def test_metadata_get_dependencies_by_type(self):
        """Test filtering dependencies by type."""
        metadata = ComponentMetadata(
            name="test",
            display_name="Test",
            description="Test",
            version="1.0.0",
            import_path="test.module",
        )

        metadata.add_dependency("numpy", ">=1.20", DependencyType.REQUIRED)
        metadata.add_dependency("pandas", ">=1.3", DependencyType.OPTIONAL)
        metadata.add_dependency("old-lib", "*", DependencyType.CONFLICTS)

        required_deps = metadata.get_dependencies_by_type(DependencyType.REQUIRED)
        optional_deps = metadata.get_dependencies_by_type(DependencyType.OPTIONAL)
        conflicts = metadata.get_dependencies_by_type(DependencyType.CONFLICTS)

        assert len(required_deps) == 1
        assert len(optional_deps) == 1
        assert len(conflicts) == 1
        assert required_deps[0].name == "numpy"
        assert optional_deps[0].name == "pandas"
        assert conflicts[0].name == "old-lib"

    def test_metadata_compatibility_check(self):
        """Test compatibility checking between components."""
        comp1 = ComponentMetadata(
            name="comp1",
            display_name="Component 1",
            description="Test",
            version="1.0.0",
            import_path="test.comp1",
        )

        comp2 = ComponentMetadata(
            name="comp2",
            display_name="Component 2",
            description="Test",
            version="1.0.0",
            import_path="test.comp2",
        )

        # No conflicts initially
        assert comp1.is_compatible_with(comp2) is True
        assert comp2.is_compatible_with(comp1) is True

        # Add conflict
        comp1.add_dependency("comp2", "1.0.0", DependencyType.CONFLICTS)
        assert comp1.is_compatible_with(comp2) is False
        assert comp2.is_compatible_with(comp1) is False

    def test_metadata_serialization(self):
        """Test metadata serialization to/from dictionary."""
        metadata = ComponentMetadata(
            name="test",
            display_name="Test Component",
            description="A test component",
            version="1.0.0",
            import_path="test.module",
            status=ComponentStatus.EXPERIMENTAL,
            author="Test Author",
            category="testing",
            tags=["test", "mock"],
            keywords=["testing", "unit-test"],
        )

        metadata.add_dependency("numpy", ">=1.20", DependencyType.REQUIRED)

        # Serialize to dict
        data = metadata.to_dict()

        assert data["name"] == "test"
        assert data["status"] == "experimental"
        assert len(data["dependencies"]) == 1
        assert data["dependencies"][0]["name"] == "numpy"
        assert isinstance(data["created_at"], str)  # Should be ISO format

        # Deserialize from dict
        restored = ComponentMetadata.from_dict(data)

        assert restored.name == metadata.name
        assert restored.status == metadata.status
        assert len(restored.dependencies) == 1
        assert restored.dependencies[0].name == "numpy"


class TestDependencyResolver:
    """Test dependency resolution functionality."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry for dependency resolution testing."""
        registry = Mock()

        # Mock components
        comp_a = ComponentMetadata(
            name="comp_a",
            display_name="Component A",
            description="Test A",
            version="1.0.0",
            import_path="test.a",
        )
        comp_a.add_dependency("comp_b", ">=1.0", DependencyType.REQUIRED)

        comp_b = ComponentMetadata(
            name="comp_b",
            display_name="Component B",
            description="Test B",
            version="1.1.0",
            import_path="test.b",
        )
        comp_b.add_dependency("comp_c", ">=1.0", DependencyType.OPTIONAL)

        comp_c = ComponentMetadata(
            name="comp_c",
            display_name="Component C",
            description="Test C",
            version="1.0.0",
            import_path="test.c",
        )

        # Mock get_metadata method
        def mock_get_metadata(name):
            metadata_map = {"comp_a": comp_a, "comp_b": comp_b, "comp_c": comp_c}
            return metadata_map.get(name)

        registry.get_metadata = mock_get_metadata
        return registry

    def test_resolver_creation(self, mock_registry):
        """Test dependency resolver creation."""
        resolver = DependencyResolver(mock_registry)
        assert resolver.registry == mock_registry
        assert isinstance(resolver._resolution_cache, dict)

    def test_resolve_dependencies_simple(self, mock_registry):
        """Test simple dependency resolution."""
        resolver = DependencyResolver(mock_registry)

        resolved, unresolved = resolver.resolve_dependencies(["comp_b"])

        assert len(resolved) >= 1
        assert len(unresolved) == 0
        assert any(comp.name == "comp_b" for comp in resolved)

    def test_resolve_dependencies_with_chain(self, mock_registry):
        """Test dependency resolution with dependency chain."""
        resolver = DependencyResolver(mock_registry)

        resolved, unresolved = resolver.resolve_dependencies(["comp_a"])

        # Should resolve comp_a -> comp_b
        assert len(resolved) >= 2
        assert len(unresolved) == 0
        component_names = [comp.name for comp in resolved]
        assert "comp_a" in component_names
        assert "comp_b" in component_names

    def test_resolve_dependencies_with_optional(self, mock_registry):
        """Test dependency resolution including optional dependencies."""
        resolver = DependencyResolver(mock_registry)

        resolved, unresolved = resolver.resolve_dependencies(["comp_b"], resolve_optional=True)

        # Should resolve comp_b -> comp_c (optional)
        assert len(resolved) >= 2
        component_names = [comp.name for comp in resolved]
        assert "comp_b" in component_names
        assert "comp_c" in component_names

    def test_resolve_dependencies_missing(self, mock_registry):
        """Test dependency resolution with missing dependencies."""
        resolver = DependencyResolver(mock_registry)

        resolved, unresolved = resolver.resolve_dependencies(["nonexistent"])

        assert len(resolved) == 0
        assert "nonexistent" in unresolved

    def test_check_conflicts_no_conflicts(self, mock_registry):
        """Test conflict checking when no conflicts exist."""
        resolver = DependencyResolver(mock_registry)

        comp_b = mock_registry.get_metadata("comp_b")
        comp_c = mock_registry.get_metadata("comp_c")

        conflicts = resolver.check_conflicts([comp_b, comp_c])
        assert len(conflicts) == 0

    def test_check_conflicts_with_conflicts(self, mock_registry):
        """Test conflict checking when conflicts exist."""
        resolver = DependencyResolver(mock_registry)

        comp_b = mock_registry.get_metadata("comp_b")
        comp_c = mock_registry.get_metadata("comp_c")

        # Add conflict
        comp_b.add_dependency("comp_c", "1.0.0", DependencyType.CONFLICTS)

        conflicts = resolver.check_conflicts([comp_b, comp_c])
        assert len(conflicts) > 0
        assert (comp_b, comp_c) in conflicts or (comp_c, comp_b) in conflicts


class TestAdvancedRegistry:
    """Test advanced registry functionality."""

    @pytest.fixture
    def temp_registry(self):
        """Create temporary registry for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        registry = AdvancedRegistry(cache_dir=temp_dir, enable_discovery=False)
        yield registry
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_registry_creation(self, temp_registry):
        """Test registry creation and initialization."""
        assert isinstance(temp_registry._metadata, dict)
        assert isinstance(temp_registry._type_registries, dict)
        assert hasattr(temp_registry, "resolver")
        assert temp_registry.cache_dir.exists()

    def test_register_component(self, temp_registry):
        """Test component registration."""
        metadata = ComponentMetadata(
            name="test-comp",
            display_name="Test Component",
            description="A test component",
            version="1.0.0",
            import_path="test.module",
        )

        temp_registry.register_component(metadata, "task")

        assert "test-comp" in temp_registry._metadata
        assert "test-comp" in temp_registry._type_registries["task"]

        retrieved = temp_registry.get_metadata("test-comp")
        assert retrieved is not None
        assert retrieved.name == "test-comp"

    def test_unregister_component(self, temp_registry):
        """Test component unregistration."""
        metadata = ComponentMetadata(
            name="test-comp",
            display_name="Test Component",
            description="A test component",
            version="1.0.0",
            import_path="test.module",
        )

        temp_registry.register_component(metadata, "task")
        assert "test-comp" in temp_registry._metadata

        temp_registry.unregister_component("test-comp")
        assert "test-comp" not in temp_registry._metadata
        assert "test-comp" not in temp_registry._type_registries["task"]

    def test_list_components_no_filter(self, temp_registry):
        """Test listing all components without filters."""
        metadata1 = ComponentMetadata(
            name="comp1",
            display_name="Component 1",
            description="Test 1",
            version="1.0.0",
            import_path="test.comp1",
        )
        metadata2 = ComponentMetadata(
            name="comp2",
            display_name="Component 2",
            description="Test 2",
            version="1.0.0",
            import_path="test.comp2",
            status=ComponentStatus.EXPERIMENTAL,
        )

        temp_registry.register_component(metadata1)
        temp_registry.register_component(metadata2)

        components = temp_registry.list_components()
        assert len(components) == 2
        component_names = [comp.name for comp in components]
        assert "comp1" in component_names
        assert "comp2" in component_names

    def test_list_components_with_filters(self, temp_registry):
        """Test listing components with filters."""
        metadata1 = ComponentMetadata(
            name="stable-comp",
            display_name="Stable Component",
            description="Stable test",
            version="1.0.0",
            import_path="test.stable",
            status=ComponentStatus.STABLE,
            category="nlp",
            tags=["production"],
        )
        metadata2 = ComponentMetadata(
            name="exp-comp",
            display_name="Experimental Component",
            description="Experimental test",
            version="0.1.0",
            import_path="test.experimental",
            status=ComponentStatus.EXPERIMENTAL,
            category="vision",
            tags=["research"],
        )

        temp_registry.register_component(metadata1, "task")
        temp_registry.register_component(metadata2, "dataset")

        # Filter by type
        tasks = temp_registry.list_components("task")
        assert len(tasks) == 1
        assert tasks[0].name == "stable-comp"

        # Filter by status
        stable_components = temp_registry.list_components(status=ComponentStatus.STABLE)
        assert len(stable_components) == 1
        assert stable_components[0].name == "stable-comp"

        # Filter by category
        nlp_components = temp_registry.list_components(category="nlp")
        assert len(nlp_components) == 1
        assert nlp_components[0].name == "stable-comp"

        # Filter by tags
        production_components = temp_registry.list_components(tags=["production"])
        assert len(production_components) == 1
        assert production_components[0].name == "stable-comp"

    def test_search_components(self, temp_registry):
        """Test component search functionality."""
        metadata1 = ComponentMetadata(
            name="nlp-processor",
            display_name="NLP Text Processor",
            description="Natural language processing component for text analysis",
            version="1.0.0",
            import_path="nlp.processor",
            keywords=["nlp", "text", "processing"],
            tags=["natural-language"],
        )
        metadata2 = ComponentMetadata(
            name="image-classifier",
            display_name="Image Classification Model",
            description="Deep learning model for image classification tasks",
            version="1.0.0",
            import_path="vision.classifier",
            keywords=["vision", "classification", "deep-learning"],
            tags=["computer-vision"],
        )

        temp_registry.register_component(metadata1)
        temp_registry.register_component(metadata2)

        # Search by name
        nlp_results = temp_registry.search_components("nlp")
        assert len(nlp_results) == 1
        assert nlp_results[0].name == "nlp-processor"

        # Search by keyword
        text_results = temp_registry.search_components("text")
        assert len(text_results) == 1
        assert text_results[0].name == "nlp-processor"

        # Search by description
        classification_results = temp_registry.search_components("classification")
        assert len(classification_results) == 1
        assert classification_results[0].name == "image-classifier"

    def test_load_component_success(self, temp_registry):
        """Test successful component loading."""
        metadata = ComponentMetadata(
            name="mock-component",
            display_name="Mock Component",
            description="A mock component for testing",
            version="1.0.0",
            import_path="unittest.mock.Mock",  # Use real importable class
        )

        temp_registry.register_component(metadata)

        component_class = temp_registry.load_component("mock-component")
        assert component_class is not None

        # Verify caching
        assert "mock-component" in temp_registry._loaded_components
        assert temp_registry._access_counts.get("mock-component", 0) > 0

    def test_load_component_failure(self, temp_registry):
        """Test component loading failure."""
        metadata = ComponentMetadata(
            name="invalid-component",
            display_name="Invalid Component",
            description="A component that cannot be imported",
            version="1.0.0",
            import_path="nonexistent.module.Class",
        )

        temp_registry.register_component(metadata)

        component_class = temp_registry.load_component("invalid-component")
        assert component_class is None

    def test_get_component_stats(self, temp_registry):
        """Test component statistics retrieval."""
        # Add some components
        for i in range(3):
            metadata = ComponentMetadata(
                name=f"comp{i}",
                display_name=f"Component {i}",
                description=f"Test component {i}",
                version="1.0.0",
                import_path="unittest.mock.Mock",
                status=ComponentStatus.STABLE if i < 2 else ComponentStatus.EXPERIMENTAL,
            )
            temp_registry.register_component(metadata, "task" if i < 2 else "dataset")

        # Load one component to test caching stats
        temp_registry.load_component("comp0")

        stats = temp_registry.get_component_stats()

        assert stats["total_components"] == 3
        assert stats["loaded_components"] == 1
        assert stats["cache_hit_ratio"] > 0
        assert "components_by_status" in stats
        assert "components_by_type" in stats
        assert stats["components_by_status"]["stable"] == 2
        assert stats["components_by_status"]["experimental"] == 1
        assert stats["components_by_type"]["task"] == 2
        assert stats["components_by_type"]["dataset"] == 1

    def test_validate_registry(self, temp_registry):
        """Test registry validation."""
        # Add valid component
        valid_metadata = ComponentMetadata(
            name="valid-comp",
            display_name="Valid Component",
            description="A valid component",
            version="1.0.0",
            import_path="unittest.mock.Mock",
        )
        temp_registry.register_component(valid_metadata)

        # Add component with missing dependency
        dependent_metadata = ComponentMetadata(
            name="dependent-comp",
            display_name="Dependent Component",
            description="A component with dependencies",
            version="1.0.0",
            import_path="unittest.mock.Mock",
        )
        dependent_metadata.add_dependency("missing-dep", ">=1.0", DependencyType.REQUIRED)
        temp_registry.register_component(dependent_metadata)

        validation = temp_registry.validate_registry()

        assert "valid" in validation
        assert "component_count" in validation
        assert validation["component_count"] == 2
        assert len(validation["dependency_issues"]) > 0
        assert any("missing-dep" in issue for issue in validation["dependency_issues"])

    def test_export_import_registry(self, temp_registry):
        """Test registry export and import functionality."""
        # Add components
        metadata = ComponentMetadata(
            name="export-comp",
            display_name="Export Test Component",
            description="Component for export testing",
            version="1.0.0",
            import_path="test.export",
            author="Test Author",
            category="testing",
        )
        metadata.add_dependency("numpy", ">=1.20", DependencyType.REQUIRED)
        temp_registry.register_component(metadata)

        # Export registry
        export_file = temp_registry.cache_dir / "export_test.json"
        temp_registry.export_registry(export_file)

        assert export_file.exists()

        # Verify export content
        with open(export_file, "r") as f:
            export_data = json.load(f)

        assert "registry_version" in export_data
        assert "components" in export_data
        assert "export-comp" in export_data["components"]

        # Create new registry and import
        temp_dir2 = Path(tempfile.mkdtemp())
        try:
            new_registry = AdvancedRegistry(cache_dir=temp_dir2, enable_discovery=False)
            new_registry.import_registry(export_file)

            imported_metadata = new_registry.get_metadata("export-comp")
            assert imported_metadata is not None
            assert imported_metadata.name == "export-comp"
            assert imported_metadata.author == "Test Author"
            assert len(imported_metadata.dependencies) == 1
            assert imported_metadata.dependencies[0].name == "numpy"

        finally:
            shutil.rmtree(temp_dir2, ignore_errors=True)

    def test_clear_cache(self, temp_registry):
        """Test cache clearing functionality."""
        metadata = ComponentMetadata(
            name="cache-test",
            display_name="Cache Test",
            description="Component for cache testing",
            version="1.0.0",
            import_path="unittest.mock.Mock",
        )
        temp_registry.register_component(metadata)

        # Load component to populate caches
        temp_registry.load_component("cache-test")

        assert len(temp_registry._loaded_components) > 0
        assert len(temp_registry._access_counts) > 0

        temp_registry.clear_cache()

        assert len(temp_registry._loaded_components) == 0
        assert len(temp_registry._access_counts) == 0


class TestRegistryIntegration:
    """Test integration functionality and convenience functions."""

    def test_get_advanced_registry_singleton(self):
        """Test global registry singleton pattern."""
        registry1 = get_advanced_registry()
        registry2 = get_advanced_registry()

        assert registry1 is registry2

    def test_register_component_metadata_convenience(self):
        """Test convenience function for registering metadata."""
        metadata = register_component_metadata(
            name="convenience-test",
            display_name="Convenience Test Component",
            description="Testing convenience function",
            version="1.0.0",
            import_path="test.convenience",
            component_type="task",
            author="Test Author",
            category="testing",
        )

        assert isinstance(metadata, ComponentMetadata)
        assert metadata.name == "convenience-test"
        assert metadata.author == "Test Author"

        # Verify it was registered
        registry = get_advanced_registry()
        retrieved = registry.get_metadata("convenience-test")
        assert retrieved is not None
        assert retrieved.name == "convenience-test"

    def test_example_registry_setup(self):
        """Test example registry setup function."""
        registry = example_registry_setup()

        assert isinstance(registry, AdvancedRegistry)

        # Verify QA component was registered
        qa_metadata = registry.get_metadata("qa")
        assert qa_metadata is not None
        assert qa_metadata.display_name == "Question Answering Task"
        assert qa_metadata.version == "2.1.0"
        assert qa_metadata.status == ComponentStatus.STABLE
        assert len(qa_metadata.dependencies) >= 1


class TestRegistryPerformanceAndEdgeCases:
    """Test performance aspects and edge cases."""

    @pytest.fixture
    def large_registry(self):
        """Create registry with many components for performance testing."""
        temp_dir = Path(tempfile.mkdtemp())
        registry = AdvancedRegistry(cache_dir=temp_dir, enable_discovery=False)

        # Add many components
        for i in range(50):
            metadata = ComponentMetadata(
                name=f"perf-comp-{i:03d}",
                display_name=f"Performance Component {i}",
                description=f"Component for performance testing {i}",
                version=f"1.{i}.0",
                import_path="unittest.mock.Mock",
            )
            registry.register_component(metadata, "task")

        yield registry
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_large_registry_search_performance(self, large_registry):
        """Test search performance with many components."""
        import time

        start_time = time.time()
        results = large_registry.search_components("performance")
        end_time = time.time()

        # Should find all 50 components
        assert len(results) == 50

        # Search should be reasonably fast (adjust threshold as needed)
        search_time = end_time - start_time
        assert search_time < 1.0  # Should complete in under 1 second

    def test_component_loading_cache_efficiency(self, large_registry):
        """Test component loading cache efficiency."""
        # Load same component multiple times
        component_name = "perf-comp-000"

        # First load (should be slower)
        start_time = datetime.now()
        comp1 = large_registry.load_component(component_name)
        first_load_time = (datetime.now() - start_time).total_seconds()

        # Second load (should be faster due to caching)
        start_time = datetime.now()
        comp2 = large_registry.load_component(component_name)
        second_load_time = (datetime.now() - start_time).total_seconds()

        assert comp1 is comp2  # Same instance due to caching
        assert second_load_time < first_load_time  # Cached load should be faster

        # Verify access count tracking
        assert large_registry._access_counts[component_name] >= 2

    def test_dependency_resolution_caching(self):
        """Test dependency resolution caching."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            registry = AdvancedRegistry(cache_dir=temp_dir, enable_discovery=False)

            # Create dependency chain
            comp_c = ComponentMetadata(
                name="comp-c",
                display_name="Component C",
                description="Base component",
                version="1.0.0",
                import_path="test.c",
            )
            comp_b = ComponentMetadata(
                name="comp-b",
                display_name="Component B",
                description="Middle component",
                version="1.0.0",
                import_path="test.b",
            )
            comp_b.add_dependency("comp-c", ">=1.0", DependencyType.REQUIRED)

            comp_a = ComponentMetadata(
                name="comp-a",
                display_name="Component A",
                description="Top component",
                version="1.0.0",
                import_path="test.a",
            )
            comp_a.add_dependency("comp-b", ">=1.0", DependencyType.REQUIRED)

            registry.register_component(comp_c)
            registry.register_component(comp_b)
            registry.register_component(comp_a)

            # First resolution
            start_time = datetime.now()
            resolved1, unresolved1 = registry.resolver.resolve_dependencies(["comp-a"])
            first_resolution_time = (datetime.now() - start_time).total_seconds()

            # Second resolution (should be cached)
            start_time = datetime.now()
            resolved2, unresolved2 = registry.resolver.resolve_dependencies(["comp-a"])
            second_resolution_time = (datetime.now() - start_time).total_seconds()

            # Results should be identical
            assert len(resolved1) == len(resolved2)
            assert len(unresolved1) == len(unresolved2) == 0

            # Check that caching provides some benefit
            # Note: May not always be faster due to test overhead, so we just verify functionality
            assert len(registry.resolver._resolution_cache) > 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__])
