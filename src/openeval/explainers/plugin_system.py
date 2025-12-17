"""Plugin system for extensible explainer architecture.

This module provides a comprehensive plugin system allowing third-party
extensions to add functionality to the code explainer framework.

Example:
    >>> from openeval.explainers import PluginManager, Plugin
    >>> manager = get_plugin_manager()
    >>> @manager.register_hook("pre_explain")
    ... def my_hook(code):
    ...     return preprocess(code)
    >>> manager.enable_plugin("my-plugin")
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class PluginStatus(Enum):
    """Status of a plugin."""

    DISCOVERED = "discovered"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    INCOMPATIBLE = "incompatible"


class PluginType(Enum):
    """Types of plugins."""

    EXPLAINER = "explainer"
    ANALYZER = "analyzer"
    FORMATTER = "formatter"
    VALIDATOR = "validator"
    STORAGE = "storage"
    INTEGRATION = "integration"
    UTILITY = "utility"


class HookPriority(Enum):
    """Priority levels for hooks."""

    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class PluginMetadata:
    """Metadata about a plugin."""

    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: list[str] = field(default_factory=list)
    requires_version: str = "0.0.0"
    tags: list[str] = field(default_factory=list)
    homepage: str = ""
    license: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookRegistration:
    """A registered hook."""

    hook_name: str
    callback: Callable
    priority: HookPriority
    plugin_name: str | None
    filter_func: Callable[[Any], bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginConfig:
    """Configuration for a plugin."""

    enabled: bool = True
    settings: dict[str, Any] = field(default_factory=dict)
    custom_hooks: list[str] = field(default_factory=list)


@dataclass
class PluginInfo:
    """Full information about a loaded plugin."""

    metadata: PluginMetadata
    status: PluginStatus
    config: PluginConfig
    instance: Plugin | None = None
    load_time: datetime | None = None
    error_message: str = ""
    path: Path | None = None


class Plugin(ABC):
    """Abstract base class for plugins."""

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass

    @abstractmethod
    def initialize(self, manager: PluginManager) -> None:
        """Initialize the plugin.

        Args:
            manager: The plugin manager.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass

    def get_config_schema(self) -> dict[str, Any]:
        """Get configuration schema for the plugin.

        Returns:
            JSON Schema for plugin configuration.
        """
        return {}

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate plugin configuration.

        Args:
            config: Configuration to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        return []


class ExplainerPlugin(Plugin):
    """Base class for explainer plugins."""

    @abstractmethod
    def explain(self, code: str, **kwargs: Any) -> str:
        """Generate an explanation for code.

        Args:
            code: The code to explain.
            **kwargs: Additional options.

        Returns:
            The explanation.
        """
        pass


class AnalyzerPlugin(Plugin):
    """Base class for analyzer plugins."""

    @abstractmethod
    def analyze(self, code: str, **kwargs: Any) -> dict[str, Any]:
        """Analyze code.

        Args:
            code: The code to analyze.
            **kwargs: Additional options.

        Returns:
            Analysis results.
        """
        pass


class FormatterPlugin(Plugin):
    """Base class for formatter plugins."""

    @abstractmethod
    def format(self, explanation: str, **kwargs: Any) -> str:
        """Format an explanation.

        Args:
            explanation: The explanation to format.
            **kwargs: Additional options.

        Returns:
            Formatted explanation.
        """
        pass


class HookManager:
    """Manages hooks for the plugin system."""

    def __init__(self) -> None:
        """Initialize the hook manager."""
        self._hooks: dict[str, list[HookRegistration]] = defaultdict(list)
        self._lock = threading.Lock()

    def register(
        self,
        hook_name: str,
        callback: Callable,
        priority: HookPriority = HookPriority.NORMAL,
        plugin_name: str | None = None,
        filter_func: Callable[[Any], bool] | None = None,
    ) -> HookRegistration:
        """Register a hook.

        Args:
            hook_name: Name of the hook point.
            callback: Function to call.
            priority: Hook priority.
            plugin_name: Name of registering plugin.
            filter_func: Optional filter function.

        Returns:
            The hook registration.
        """
        registration = HookRegistration(
            hook_name=hook_name,
            callback=callback,
            priority=priority,
            plugin_name=plugin_name,
            filter_func=filter_func,
        )

        with self._lock:
            self._hooks[hook_name].append(registration)
            # Sort by priority
            self._hooks[hook_name].sort(key=lambda h: h.priority.value)

        return registration

    def unregister(self, hook_name: str, callback: Callable) -> bool:
        """Unregister a hook.

        Args:
            hook_name: Hook name.
            callback: Callback to remove.

        Returns:
            True if removed.
        """
        with self._lock:
            original_len = len(self._hooks[hook_name])
            self._hooks[hook_name] = [h for h in self._hooks[hook_name] if h.callback != callback]
            return len(self._hooks[hook_name]) < original_len

    def unregister_plugin(self, plugin_name: str) -> int:
        """Unregister all hooks for a plugin.

        Args:
            plugin_name: Plugin name.

        Returns:
            Number of hooks removed.
        """
        count = 0
        with self._lock:
            for hook_name in self._hooks:
                original_len = len(self._hooks[hook_name])
                self._hooks[hook_name] = [
                    h for h in self._hooks[hook_name] if h.plugin_name != plugin_name
                ]
                count += original_len - len(self._hooks[hook_name])
        return count

    def execute(
        self,
        hook_name: str,
        *args: Any,
        context: Any = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute all hooks for a hook point.

        Args:
            hook_name: Hook name.
            *args: Positional arguments.
            context: Context for filtering.
            **kwargs: Keyword arguments.

        Returns:
            List of results from all hooks.
        """
        results = []

        with self._lock:
            hooks = self._hooks.get(hook_name, []).copy()

        for hook in hooks:
            # Apply filter if present
            if hook.filter_func and context is not None:
                if not hook.filter_func(context):
                    continue

            try:
                result = hook.callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "hook": hook.hook_name})

        return results

    def execute_filter(
        self,
        hook_name: str,
        value: T,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute hooks as a filter chain.

        Each hook receives the result of the previous hook.

        Args:
            hook_name: Hook name.
            value: Initial value to filter.
            *args: Additional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Filtered value.
        """
        with self._lock:
            hooks = self._hooks.get(hook_name, []).copy()

        for hook in hooks:
            try:
                value = hook.callback(value, *args, **kwargs)
            except Exception:
                pass

        return value

    def get_hooks(self, hook_name: str) -> list[HookRegistration]:
        """Get all hooks for a hook point.

        Args:
            hook_name: Hook name.

        Returns:
            List of hook registrations.
        """
        with self._lock:
            return self._hooks.get(hook_name, []).copy()

    def list_hook_points(self) -> list[str]:
        """List all registered hook points.

        Returns:
            List of hook names.
        """
        with self._lock:
            return list(self._hooks.keys())


class PluginLoader:
    """Loads plugins from various sources."""

    def __init__(self) -> None:
        """Initialize the loader."""
        self._search_paths: list[Path] = []

    def add_search_path(self, path: Path | str) -> None:
        """Add a search path for plugins.

        Args:
            path: Directory path to search.
        """
        path = Path(path)
        if path.is_dir() and path not in self._search_paths:
            self._search_paths.append(path)

    def discover(self) -> list[tuple[Path, PluginMetadata | None]]:
        """Discover plugins in search paths.

        Returns:
            List of (path, metadata) tuples.
        """
        discovered = []

        for search_path in self._search_paths:
            for item in search_path.iterdir():
                if item.is_file() and item.suffix == ".py":
                    metadata = self._extract_metadata(item)
                    discovered.append((item, metadata))
                elif item.is_dir() and (item / "__init__.py").exists():
                    metadata = self._extract_metadata(item / "__init__.py")
                    discovered.append((item, metadata))

        return discovered

    def _extract_metadata(self, path: Path) -> PluginMetadata | None:
        """Extract metadata from a plugin file.

        Args:
            path: Path to plugin file.

        Returns:
            PluginMetadata if found.
        """
        try:
            spec = importlib.util.spec_from_file_location("_temp_plugin", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for Plugin subclass
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, Plugin)
                        and obj is not Plugin
                        and not inspect.isabstract(obj)
                    ):
                        instance = obj()
                        return instance.metadata

        except Exception:
            pass

        return None

    def load(self, path: Path) -> Plugin | None:
        """Load a plugin from path.

        Args:
            path: Path to plugin.

        Returns:
            Plugin instance if loaded.
        """
        try:
            module_name = path.stem
            spec = importlib.util.spec_from_file_location(f"openeval_plugin_{module_name}", path)

            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find Plugin subclass
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, Plugin)
                        and obj is not Plugin
                        and not inspect.isabstract(obj)
                    ):
                        return obj()

        except Exception:
            pass

        return None


class PluginManager:
    """Main plugin manager."""

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        self._plugins: dict[str, PluginInfo] = {}
        self._hook_manager = HookManager()
        self._loader = PluginLoader()
        self._lock = threading.Lock()
        self._builtin_hooks = [
            "pre_explain",
            "post_explain",
            "pre_analyze",
            "post_analyze",
            "pre_format",
            "post_format",
            "on_error",
            "on_cache_hit",
            "on_cache_miss",
        ]

    @property
    def hooks(self) -> HookManager:
        """Get the hook manager."""
        return self._hook_manager

    def add_plugin_path(self, path: Path | str) -> None:
        """Add a path to search for plugins.

        Args:
            path: Directory path.
        """
        self._loader.add_search_path(path)

    def discover_plugins(self) -> list[PluginInfo]:
        """Discover available plugins.

        Returns:
            List of discovered plugins.
        """
        discovered = []

        for path, metadata in self._loader.discover():
            if metadata:
                info = PluginInfo(
                    metadata=metadata,
                    status=PluginStatus.DISCOVERED,
                    config=PluginConfig(),
                    path=path,
                )
                discovered.append(info)

                with self._lock:
                    if metadata.name not in self._plugins:
                        self._plugins[metadata.name] = info

        return discovered

    def load_plugin(self, name: str) -> PluginInfo | None:
        """Load a discovered plugin.

        Args:
            name: Plugin name.

        Returns:
            PluginInfo if loaded.
        """
        with self._lock:
            info = self._plugins.get(name)
            if not info:
                return None

            if info.status in [PluginStatus.LOADED, PluginStatus.ENABLED]:
                return info

            if not info.path:
                return None

        # Load the plugin
        try:
            plugin = self._loader.load(info.path)
            if plugin:
                info.instance = plugin
                info.status = PluginStatus.LOADED
                info.load_time = datetime.now()

                with self._lock:
                    self._plugins[name] = info

                return info
        except Exception as e:
            info.status = PluginStatus.ERROR
            info.error_message = str(e)

        return None

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin.

        Args:
            name: Plugin name.

        Returns:
            True if enabled.
        """
        info = self._plugins.get(name)
        if not info:
            return False

        if info.status == PluginStatus.DISCOVERED:
            self.load_plugin(name)
            info = self._plugins.get(name)
            if not info:
                return False

        if info.status != PluginStatus.LOADED:
            return False

        if info.instance:
            try:
                info.instance.initialize(self)
                info.status = PluginStatus.ENABLED
                info.config.enabled = True
                return True
            except Exception as e:
                info.status = PluginStatus.ERROR
                info.error_message = str(e)

        return False

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin.

        Args:
            name: Plugin name.

        Returns:
            True if disabled.
        """
        info = self._plugins.get(name)
        if not info or info.status != PluginStatus.ENABLED:
            return False

        if info.instance:
            try:
                info.instance.cleanup()
            except Exception:
                pass

        # Unregister hooks
        self._hook_manager.unregister_plugin(name)

        info.status = PluginStatus.DISABLED
        info.config.enabled = False
        return True

    def register_plugin(self, plugin: Plugin) -> PluginInfo:
        """Register a plugin directly.

        Args:
            plugin: Plugin instance.

        Returns:
            PluginInfo for the registered plugin.
        """
        metadata = plugin.metadata
        info = PluginInfo(
            metadata=metadata,
            status=PluginStatus.LOADED,
            config=PluginConfig(),
            instance=plugin,
            load_time=datetime.now(),
        )

        with self._lock:
            self._plugins[metadata.name] = info

        return info

    def register_hook(
        self,
        hook_name: str,
        priority: HookPriority = HookPriority.NORMAL,
        plugin_name: str | None = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator to register a hook.

        Args:
            hook_name: Hook point name.
            priority: Hook priority.
            plugin_name: Plugin name.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable) -> Callable:
            self._hook_manager.register(
                hook_name,
                func,
                priority,
                plugin_name,
            )
            return func

        return decorator

    def execute_hook(
        self,
        hook_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute a hook point.

        Args:
            hook_name: Hook name.
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            Results from hooks.
        """
        return self._hook_manager.execute(hook_name, *args, **kwargs)

    def filter_hook(
        self,
        hook_name: str,
        value: T,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute hooks as a filter.

        Args:
            hook_name: Hook name.
            value: Value to filter.
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            Filtered value.
        """
        return self._hook_manager.execute_filter(hook_name, value, *args, **kwargs)

    def get_plugin(self, name: str) -> PluginInfo | None:
        """Get plugin info.

        Args:
            name: Plugin name.

        Returns:
            PluginInfo if found.
        """
        return self._plugins.get(name)

    def list_plugins(
        self,
        status: PluginStatus | None = None,
        plugin_type: PluginType | None = None,
    ) -> list[PluginInfo]:
        """List plugins with optional filters.

        Args:
            status: Filter by status.
            plugin_type: Filter by type.

        Returns:
            List of matching plugins.
        """
        plugins = list(self._plugins.values())

        if status:
            plugins = [p for p in plugins if p.status == status]

        if plugin_type:
            plugins = [p for p in plugins if p.metadata.plugin_type == plugin_type]

        return plugins

    def get_enabled_plugins(self) -> list[PluginInfo]:
        """Get all enabled plugins.

        Returns:
            List of enabled plugins.
        """
        return self.list_plugins(status=PluginStatus.ENABLED)

    def configure_plugin(
        self,
        name: str,
        settings: dict[str, Any],
    ) -> bool:
        """Configure a plugin.

        Args:
            name: Plugin name.
            settings: Settings to apply.

        Returns:
            True if configured.
        """
        info = self._plugins.get(name)
        if not info:
            return False

        # Validate if possible
        if info.instance:
            errors = info.instance.validate_config(settings)
            if errors:
                return False

        info.config.settings.update(settings)
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get plugin system statistics.

        Returns:
            Statistics dictionary.
        """
        plugins = list(self._plugins.values())
        return {
            "total_plugins": len(plugins),
            "by_status": {
                status.value: sum(1 for p in plugins if p.status == status)
                for status in PluginStatus
            },
            "by_type": {
                ptype.value: sum(1 for p in plugins if p.metadata.plugin_type == ptype)
                for ptype in PluginType
            },
            "registered_hooks": len(self._hook_manager.list_hook_points()),
        }


# Global instance
_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager.

    Returns:
        The global PluginManager instance.
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def reset_plugin_manager() -> None:
    """Reset the global plugin manager."""
    global _plugin_manager
    _plugin_manager = None


def create_plugin_manager() -> PluginManager:
    """Create a new plugin manager.

    Returns:
        New PluginManager instance.
    """
    return PluginManager()


def register_hook(
    hook_name: str,
    priority: HookPriority = HookPriority.NORMAL,
) -> Callable[[Callable], Callable]:
    """Decorator to register a hook with the global manager.

    Args:
        hook_name: Hook point name.
        priority: Hook priority.

    Returns:
        Decorator function.
    """
    return get_plugin_manager().register_hook(hook_name, priority)


def execute_hook(hook_name: str, *args: Any, **kwargs: Any) -> list[Any]:
    """Execute hooks on the global manager.

    Args:
        hook_name: Hook name.
        *args: Arguments.
        **kwargs: Keyword arguments.

    Returns:
        Results from hooks.
    """
    return get_plugin_manager().execute_hook(hook_name, *args, **kwargs)


def load_plugin(name: str) -> PluginInfo | None:
    """Load a plugin in the global manager.

    Args:
        name: Plugin name.

    Returns:
        PluginInfo if loaded.
    """
    return get_plugin_manager().load_plugin(name)


def enable_plugin(name: str) -> bool:
    """Enable a plugin in the global manager.

    Args:
        name: Plugin name.

    Returns:
        True if enabled.
    """
    return get_plugin_manager().enable_plugin(name)
