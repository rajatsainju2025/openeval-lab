"""Advanced registry system with metadata, versioning, and dependency resolution.

This module extends the basic registry with enterprise-grade features including
metadata management, semantic versioning, dependency resolution, validation,
caching, and discovery mechanisms.
"""

import json
import hashlib
import importlib
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, Callable, Protocol
from dataclasses import dataclass, field, asdict
from enum import Enum
import functools
import weakref
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from packaging import version
    HAS_PACKAGING = True
    Version = version.Version
except ImportError:
    HAS_PACKAGING = False
    # Fallback simple version implementation
    class Version:
        def __init__(self, version_str: str):
            self.version_str = version_str
            # Simple semantic version parsing
            parts = version_str.split('.')
            self.major = int(parts[0]) if len(parts) > 0 else 0
            self.minor = int(parts[1]) if len(parts) > 1 else 0
            self.patch = int(parts[2]) if len(parts) > 2 else 0
        
        def __str__(self):
            return self.version_str
        
        def __eq__(self, other):
            return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
        
        def __lt__(self, other):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        def __le__(self, other):
            return self == other or self < other
        
        def __gt__(self, other):
            return not self <= other
        
        def __ge__(self, other):
            return not self < other

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component lifecycle status."""
    EXPERIMENTAL = "experimental"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    OBSOLETE = "obsolete"


class DependencyType(Enum):
    """Types of component dependencies."""
    REQUIRED = "required"
    OPTIONAL = "optional" 
    CONFLICTS = "conflicts"
    SUGGESTS = "suggests"


@dataclass
class Dependency:
    """Component dependency specification."""
    name: str
    version_constraint: str = "*"
    dependency_type: DependencyType = DependencyType.REQUIRED
    description: str = ""
    
    def __post_init__(self):
        """Validate dependency specification."""
        if not self.name:
            raise ValueError("Dependency name cannot be empty")
        
        # Validate version constraint format
        if self.version_constraint and self.version_constraint != "*":
            try:
                # Simple validation - should support >=1.0.0, ~=1.0, etc.
                constraint = self.version_constraint.strip()
                if constraint.startswith(('>=', '<=', '==', '!=', '~=', '>')):
                    version_part = constraint[2:].strip()
                elif constraint.startswith(('<', '>')):
                    version_part = constraint[1:].strip()
                else:
                    version_part = constraint
                
                if version_part != "*":
                    if HAS_PACKAGING:
                        Version(version_part)
                    else:
                        # Simple validation
                        parts = version_part.split('.')
                        if not all(part.isdigit() for part in parts):
                            raise ValueError("Invalid version format")
            except Exception as e:
                raise ValueError(f"Invalid version constraint '{self.version_constraint}': {e}")
    
    def matches(self, target_version: str) -> bool:
        """Check if target version satisfies this dependency constraint."""
        if self.version_constraint == "*":
            return True
        
        try:
            if HAS_PACKAGING:
                target_ver = Version(target_version)
                constraint = self.version_constraint.strip()
                
                if constraint.startswith('>='):
                    min_ver = Version(constraint[2:].strip())
                    return target_ver >= min_ver
                elif constraint.startswith('<='):
                    max_ver = Version(constraint[2:].strip())
                    return target_ver <= max_ver
                elif constraint.startswith('=='):
                    exact_ver = Version(constraint[2:].strip())
                    return target_ver == exact_ver
                elif constraint.startswith('!='):
                    excl_ver = Version(constraint[2:].strip())
                    return target_ver != excl_ver
                elif constraint.startswith('~='):
                    # Compatible release
                    base_ver = Version(constraint[2:].strip())
                    return target_ver >= base_ver and target_ver.major == base_ver.major
                elif constraint.startswith('>'):
                    min_ver = Version(constraint[1:].strip())
                    return target_ver > min_ver
                elif constraint.startswith('<'):
                    max_ver = Version(constraint[1:].strip())
                    return target_ver < max_ver
                else:
                    # Exact match
                    exact_ver = Version(constraint)
                    return target_ver == exact_ver
            else:
                # Fallback simple matching
                return target_version == self.version_constraint or self.version_constraint == "*"
        except Exception:
            return False


@dataclass 
class ComponentMetadata:
    """Rich metadata for registry components."""
    name: str
    display_name: str
    description: str
    version: str
    import_path: str
    
    # Lifecycle information
    status: ComponentStatus = ComponentStatus.STABLE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Authorship and provenance
    author: str = ""
    maintainer: str = ""
    license: str = ""
    homepage: str = ""
    documentation_url: str = ""
    source_url: str = ""
    
    # Categorization
    category: str = ""
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Dependencies and compatibility
    dependencies: List[Dependency] = field(default_factory=list)
    python_requires: str = ">=3.8"
    platform_compatibility: List[str] = field(default_factory=lambda: ["any"])
    
    # Performance and resource information
    memory_requirements: Optional[str] = None
    compute_requirements: Optional[str] = None
    estimated_runtime: Optional[str] = None
    
    # Quality metrics
    test_coverage: Optional[float] = None
    performance_score: Optional[float] = None
    reliability_score: Optional[float] = None
    
    # Usage statistics
    download_count: int = 0
    usage_frequency: float = 0.0
    user_rating: Optional[float] = None
    
    # Custom attributes
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.name:
            raise ValueError("Component name cannot be empty")
        if not self.import_path:
            raise ValueError("Import path cannot be empty")
        
        # Validate version format
        try:
            if HAS_PACKAGING:
                Version(self.version)
            else:
                # Simple validation
                parts = self.version.split('.')
                if not all(part.isdigit() for part in parts):
                    raise ValueError("Invalid version format")
        except Exception as e:
            raise ValueError(f"Invalid version '{self.version}': {e}")
    
    def add_dependency(self, name: str, version_constraint: str = "*", 
                      dependency_type: DependencyType = DependencyType.REQUIRED,
                      description: str = ""):
        """Add a dependency to this component."""
        dep = Dependency(
            name=name,
            version_constraint=version_constraint,
            dependency_type=dependency_type,
            description=description
        )
        self.dependencies.append(dep)
        self.updated_at = datetime.now()
    
    def get_dependencies_by_type(self, dependency_type: DependencyType) -> List[Dependency]:
        """Get dependencies of a specific type."""
        return [dep for dep in self.dependencies if dep.dependency_type == dependency_type]
    
    def is_compatible_with(self, other: 'ComponentMetadata') -> bool:
        """Check if this component is compatible with another."""
        # Check for conflicts
        conflicts = self.get_dependencies_by_type(DependencyType.CONFLICTS)
        for conflict in conflicts:
            if conflict.name == other.name and conflict.matches(other.version):
                return False
        
        # Check if other conflicts with this
        other_conflicts = other.get_dependencies_by_type(DependencyType.CONFLICTS)
        for conflict in other_conflicts:
            if conflict.name == self.name and conflict.matches(self.version):
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        result = asdict(self)
        # Convert datetime objects to ISO format
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        # Convert enums to strings
        result['status'] = self.status.value
        # Convert dependencies
        result['dependencies'] = [
            {
                'name': dep.name,
                'version_constraint': dep.version_constraint,
                'dependency_type': dep.dependency_type.value,
                'description': dep.description
            }
            for dep in self.dependencies
        ]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentMetadata':
        """Create metadata from dictionary."""
        # Parse datetime fields
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Parse enum fields
        if 'status' in data:
            data['status'] = ComponentStatus(data['status'])
        
        # Parse dependencies
        if 'dependencies' in data:
            deps = []
            for dep_data in data['dependencies']:
                dep = Dependency(
                    name=dep_data['name'],
                    version_constraint=dep_data['version_constraint'],
                    dependency_type=DependencyType(dep_data['dependency_type']),
                    description=dep_data.get('description', '')
                )
                deps.append(dep)
            data['dependencies'] = deps
        
        return cls(**data)


class DependencyResolver:
    """Resolves component dependencies and checks for conflicts."""
    
    def __init__(self, registry: 'AdvancedRegistry'):
        self.registry = registry
        self._resolution_cache: Dict[frozenset, Optional[List[ComponentMetadata]]] = {}
    
    def resolve_dependencies(self, components: List[str], 
                           resolve_optional: bool = False) -> Tuple[List[ComponentMetadata], List[str]]:
        """
        Resolve dependencies for a set of components.
        
        Returns:
            Tuple of (resolved_components, unresolved_dependencies)
        """
        # Use cache key
        cache_key = frozenset(components + [str(resolve_optional)])
        if cache_key in self._resolution_cache:
            cached_result = self._resolution_cache[cache_key]
            if cached_result is not None:
                return cached_result, []
        
        resolved: Dict[str, ComponentMetadata] = {}
        unresolved: Set[str] = set()
        processing: Set[str] = set()
        
        def resolve_component(component_name: str) -> bool:
            """Recursively resolve a single component."""
            if component_name in resolved:
                return True
            if component_name in processing:
                # Circular dependency detected
                logger.warning(f"Circular dependency detected for {component_name}")
                return False
            
            processing.add(component_name)
            
            try:
                metadata = self.registry.get_metadata(component_name)
                if not metadata:
                    unresolved.add(component_name)
                    return False
                
                # Resolve dependencies
                for dep in metadata.dependencies:
                    if dep.dependency_type == DependencyType.REQUIRED:
                        if not resolve_component(dep.name):
                            unresolved.add(dep.name)
                            return False
                    elif dep.dependency_type == DependencyType.OPTIONAL and resolve_optional:
                        resolve_component(dep.name)  # Don't fail if optional dep fails
                
                resolved[component_name] = metadata
                return True
            
            finally:
                processing.discard(component_name)
        
        # Resolve each requested component
        for comp in components:
            resolve_component(comp)
        
        result_components = list(resolved.values())
        result_unresolved = list(unresolved)
        
        # Cache successful resolutions
        if not result_unresolved:
            self._resolution_cache[cache_key] = result_components
        
        return result_components, result_unresolved
    
    def check_conflicts(self, components: List[ComponentMetadata]) -> List[Tuple[ComponentMetadata, ComponentMetadata]]:
        """Check for conflicts between components."""
        conflicts = []
        
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                if not comp1.is_compatible_with(comp2):
                    conflicts.append((comp1, comp2))
        
        return conflicts
    
    def find_dependency_path(self, from_component: str, to_component: str) -> Optional[List[str]]:
        """Find dependency path between two components."""
        visited = set()
        path = []
        
        def dfs(current: str, target: str) -> bool:
            if current == target:
                return True
            if current in visited:
                return False
            
            visited.add(current)
            path.append(current)
            
            metadata = self.registry.get_metadata(current)
            if metadata:
                for dep in metadata.dependencies:
                    if dep.dependency_type in [DependencyType.REQUIRED, DependencyType.OPTIONAL]:
                        if dfs(dep.name, target):
                            return True
            
            path.pop()
            return False
        
        if dfs(from_component, to_component):
            return path + [to_component]
        return None


class AdvancedRegistry:
    """Advanced registry with metadata, versioning, and dependency resolution."""
    
    def __init__(self, cache_dir: Optional[Path] = None, enable_discovery: bool = True):
        self.cache_dir = cache_dir or Path.home() / '.openeval' / 'registry_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._metadata: Dict[str, ComponentMetadata] = {}
        self._loaded_components: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._lock = threading.RLock()
        self._discovery_enabled = enable_discovery
        
        # Component type registries
        self._type_registries: Dict[str, Dict[str, ComponentMetadata]] = {
            'task': {},
            'dataset': {},
            'adapter': {},
            'metric': {}
        }
        
        # Dependency resolver
        self.resolver = DependencyResolver(self)
        
        # Performance tracking
        self._load_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        
        # Load cached metadata
        self._load_cache()
        
        # Auto-discovery
        if enable_discovery:
            self._discover_components()
    
    def register_component(self, metadata: ComponentMetadata, component_type: str = ""):
        """Register a component with metadata."""
        with self._lock:
            self._metadata[metadata.name] = metadata
            
            # Add to type-specific registry if type is provided
            if component_type and component_type in self._type_registries:
                self._type_registries[component_type][metadata.name] = metadata
            
            # Save to cache
            self._save_to_cache(metadata)
            
            logger.info(f"Registered component '{metadata.name}' version {metadata.version}")
    
    def unregister_component(self, name: str):
        """Unregister a component."""
        with self._lock:
            if name in self._metadata:
                del self._metadata[name]
                
                # Remove from type registries
                for type_registry in self._type_registries.values():
                    if name in type_registry:
                        del type_registry[name]
                
                # Remove from cache
                cache_file = self.cache_dir / f"{name}.json"
                if cache_file.exists():
                    cache_file.unlink()
                
                logger.info(f"Unregistered component '{name}'")
    
    def get_metadata(self, name: str) -> Optional[ComponentMetadata]:
        """Get metadata for a component."""
        with self._lock:
            return self._metadata.get(name)
    
    def list_components(self, component_type: Optional[str] = None,
                       status: Optional[ComponentStatus] = None,
                       category: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> List[ComponentMetadata]:
        """List components with optional filtering."""
        with self._lock:
            if component_type and component_type in self._type_registries:
                candidates = list(self._type_registries[component_type].values())
            else:
                candidates = list(self._metadata.values())
            
            # Apply filters
            result = candidates
            
            if status:
                result = [comp for comp in result if comp.status == status]
            
            if category:
                result = [comp for comp in result if comp.category == category]
            
            if tags:
                result = [comp for comp in result 
                         if any(tag in comp.tags for tag in tags)]
            
            return sorted(result, key=lambda x: (x.status.value, x.name))
    
    def search_components(self, query: str, fuzzy: bool = True) -> List[ComponentMetadata]:
        """Search components by name, description, or keywords."""
        query_lower = query.lower()
        results = []
        
        with self._lock:
            for metadata in self._metadata.values():
                score = 0.0
                
                # Name match (highest weight)
                if query_lower in metadata.name.lower():
                    score += 10.0
                if query_lower in metadata.display_name.lower():
                    score += 8.0
                
                # Description match
                if query_lower in metadata.description.lower():
                    score += 5.0
                
                # Keywords and tags match
                for keyword in metadata.keywords:
                    if query_lower in keyword.lower():
                        score += 3.0
                
                for tag in metadata.tags:
                    if query_lower in tag.lower():
                        score += 2.0
                
                # Category match
                if query_lower in metadata.category.lower():
                    score += 1.0
                
                if score > 0:
                    results.append((metadata, score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        return [metadata for metadata, _ in results]
    
    def load_component(self, name: str, force_reload: bool = False) -> Optional[Type]:
        """Load a component class dynamically."""
        # Check cache first
        if not force_reload and name in self._loaded_components:
            self._access_counts[name] = self._access_counts.get(name, 0) + 1
            return self._loaded_components[name]
        
        metadata = self.get_metadata(name)
        if not metadata:
            return None
        
        start_time = datetime.now()
        
        try:
            # Dynamic import
            module_path, class_name = metadata.import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            
            # Cache the loaded component
            self._loaded_components[name] = component_class
            
            # Track performance
            load_time = (datetime.now() - start_time).total_seconds()
            self._load_times[name] = load_time
            self._access_counts[name] = self._access_counts.get(name, 0) + 1
            
            logger.debug(f"Loaded component '{name}' in {load_time:.3f}s")
            return component_class
            
        except Exception as e:
            logger.error(f"Failed to load component '{name}': {e}")
            return None
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get registry performance and usage statistics."""
        with self._lock:
            total_components = len(self._metadata)
            loaded_components = len(self._loaded_components)
            
            stats = {
                'total_components': total_components,
                'loaded_components': loaded_components,
                'cache_hit_ratio': loaded_components / max(total_components, 1),
                'average_load_time': sum(self._load_times.values()) / max(len(self._load_times), 1),
                'most_accessed': sorted(self._access_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                'components_by_status': {},
                'components_by_type': {}
            }
            
            # Status breakdown
            for metadata in self._metadata.values():
                status = metadata.status.value
                stats['components_by_status'][status] = stats['components_by_status'].get(status, 0) + 1
            
            # Type breakdown
            for comp_type, registry in self._type_registries.items():
                stats['components_by_type'][comp_type] = len(registry)
            
            return stats
    
    def validate_registry(self) -> Dict[str, Any]:
        """Validate registry integrity and component dependencies."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'component_count': len(self._metadata),
            'dependency_issues': [],
            'conflicts': []
        }
        
        with self._lock:
            # Check each component
            for name, metadata in self._metadata.items():
                try:
                    # Validate metadata
                    if not metadata.name or not metadata.import_path:
                        validation_results['errors'].append(f"Component '{name}' has invalid metadata")
                        validation_results['valid'] = False
                    
                    # Check if component can be imported
                    try:
                        self.load_component(name)
                    except Exception as e:
                        validation_results['warnings'].append(f"Component '{name}' import failed: {e}")
                    
                    # Check dependencies
                    for dep in metadata.dependencies:
                        if dep.dependency_type == DependencyType.REQUIRED:
                            if not self.get_metadata(dep.name):
                                validation_results['dependency_issues'].append(
                                    f"Component '{name}' requires missing dependency '{dep.name}'"
                                )
                    
                except Exception as e:
                    validation_results['errors'].append(f"Error validating component '{name}': {e}")
                    validation_results['valid'] = False
            
            # Check for conflicts
            all_components = list(self._metadata.values())
            conflicts = self.resolver.check_conflicts(all_components)
            if conflicts:
                validation_results['conflicts'] = [
                    f"Conflict between '{comp1.name}' and '{comp2.name}'"
                    for comp1, comp2 in conflicts
                ]
        
        return validation_results
    
    def export_registry(self, output_path: Path, format: str = "json"):
        """Export registry metadata to file."""
        with self._lock:
            if format == "json":
                data = {
                    'registry_version': '2.0',
                    'exported_at': datetime.now().isoformat(),
                    'components': {
                        name: metadata.to_dict()
                        for name, metadata in self._metadata.items()
                    }
                }
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def import_registry(self, input_path: Path, merge: bool = True):
        """Import registry metadata from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if not merge:
            self._metadata.clear()
            for type_registry in self._type_registries.values():
                type_registry.clear()
        
        components_data = data.get('components', {})
        for name, component_data in components_data.items():
            metadata = ComponentMetadata.from_dict(component_data)
            self.register_component(metadata)
    
    def _load_cache(self):
        """Load cached metadata from disk."""
        if not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                metadata = ComponentMetadata.from_dict(data)
                self._metadata[metadata.name] = metadata
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
    
    def _save_to_cache(self, metadata: ComponentMetadata):
        """Save component metadata to cache."""
        cache_file = self.cache_dir / f"{metadata.name}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache for '{metadata.name}': {e}")
    
    def _discover_components(self):
        """Auto-discover components from the codebase."""
        # This would scan for components with proper metadata
        # For now, we'll skip auto-discovery to avoid complexity
        pass
    
    def clear_cache(self):
        """Clear the registry cache."""
        with self._lock:
            self._loaded_components.clear()
            self._load_times.clear()
            self._access_counts.clear()
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")


# Global registry instance
_advanced_registry = None


def get_advanced_registry() -> AdvancedRegistry:
    """Get the global advanced registry instance."""
    global _advanced_registry
    if _advanced_registry is None:
        _advanced_registry = AdvancedRegistry()
    return _advanced_registry


def register_component_metadata(
    name: str,
    display_name: str,
    description: str,
    version: str,
    import_path: str,
    component_type: str = "",
    **kwargs
) -> ComponentMetadata:
    """Convenient function to register component metadata."""
    metadata = ComponentMetadata(
        name=name,
        display_name=display_name,
        description=description,
        version=version,
        import_path=import_path,
        **kwargs
    )
    
    registry = get_advanced_registry()
    registry.register_component(metadata, component_type)
    return metadata


# Example usage functions
def example_registry_setup():
    """Example of setting up advanced registry with metadata."""
    registry = get_advanced_registry()
    
    # Register components with rich metadata
    qa_metadata = ComponentMetadata(
        name="qa",
        display_name="Question Answering Task",
        description="Comprehensive question-answering evaluation with multiple metrics",
        version="2.1.0",
        import_path="openeval.tasks.qa.QATask",
        status=ComponentStatus.STABLE,
        author="OpenEval Team",
        category="nlp",
        tags=["question-answering", "nlp", "evaluation"],
        keywords=["qa", "question", "answer", "comprehension"],
        python_requires=">=3.8"
    )
    
    # Add dependencies
    qa_metadata.add_dependency("exact_match", ">=1.0.0", DependencyType.REQUIRED, "Core accuracy metric")
    qa_metadata.add_dependency("token_f1", ">=1.0.0", DependencyType.OPTIONAL, "Enhanced F1 scoring")
    
    registry.register_component(qa_metadata, "task")
    
    return registry


if __name__ == "__main__":
    # Example usage
    registry = example_registry_setup()
    
    # Demonstrate features
    print("Registry Stats:", registry.get_component_stats())
    print("Components:", [comp.name for comp in registry.list_components()])
    print("Validation:", registry.validate_registry())