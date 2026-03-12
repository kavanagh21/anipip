"""Plugin discovery and registration system."""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Type

from .plugin_base import BasePlugin
from .settings import PluginSettings


class PluginRegistry:
    """Registry for discovering and managing plugins.

    The registry scans the plugins directory and loads all plugin classes
    that inherit from BasePlugin.
    """

    def __init__(self):
        self._plugins: dict[str, Type[BasePlugin]] = {}
        self._plugins_by_category: dict[str, list[Type[BasePlugin]]] = {}
        self._settings: PluginSettings | None = None

    def set_settings(self, settings: PluginSettings) -> None:
        """Set the settings manager for applying plugin defaults."""
        self._settings = settings

    def discover_plugins(self, plugins_dir: Path) -> None:
        """Discover and register all plugins in the given directory.

        Args:
            plugins_dir: Path to the plugins directory
        """
        if not plugins_dir.exists():
            return

        # Walk through all Python files in the plugins directory
        for py_file in plugins_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            self._load_plugin_file(py_file)

    def _load_plugin_file(self, file_path: Path) -> None:
        """Load plugins from a single Python file.

        Args:
            file_path: Path to the Python file
        """
        module_name = f"plugins.{file_path.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find all BasePlugin subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BasePlugin)
                    and attr is not BasePlugin
                ):
                    self.register_plugin(attr)

        except Exception as e:
            print(f"Error loading plugin from {file_path}: {e}")

    def register_plugin(self, plugin_class: Type[BasePlugin]) -> None:
        """Register a plugin class.

        Args:
            plugin_class: The plugin class to register
        """
        name = plugin_class.name
        self._plugins[name] = plugin_class

        category = plugin_class.category
        if category not in self._plugins_by_category:
            self._plugins_by_category[category] = []
        if plugin_class not in self._plugins_by_category[category]:
            self._plugins_by_category[category].append(plugin_class)

    def get_plugin(self, name: str) -> Type[BasePlugin] | None:
        """Get a plugin class by name.

        Args:
            name: The plugin name

        Returns:
            The plugin class or None if not found
        """
        return self._plugins.get(name)

    def get_all_plugins(self) -> list[Type[BasePlugin]]:
        """Get all registered plugin classes.

        Returns:
            List of all plugin classes
        """
        return list(self._plugins.values())

    def get_plugins_by_category(self, category: str) -> list[Type[BasePlugin]]:
        """Get all plugins in a category.

        Args:
            category: The category name

        Returns:
            List of plugin classes in the category
        """
        return self._plugins_by_category.get(category, [])

    def get_categories(self) -> list[str]:
        """Get all plugin categories.

        Returns:
            List of category names
        """
        return list(self._plugins_by_category.keys())

    def create_instance(self, name: str) -> BasePlugin | None:
        """Create an instance of a plugin by name.

        Args:
            name: The plugin name

        Returns:
            A new plugin instance or None if not found
        """
        plugin_class = self.get_plugin(name)
        if plugin_class is None:
            return None
        instance = plugin_class()
        if self._settings:
            overrides = self._settings.get_plugin_defaults(name)
            if overrides:
                instance.set_all_parameters(overrides)
        return instance
