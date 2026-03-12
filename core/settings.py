"""Plugin default settings persistence."""

import json
from pathlib import Path
from typing import Any


class PluginSettings:
    """Manages persistent plugin default parameter overrides."""

    def __init__(self, config_dir: Path | None = None):
        if config_dir is None:
            config_dir = Path.home() / ".analysispipeline"
        self._config_dir = config_dir
        self._config_file = config_dir / "plugin_defaults.json"
        self._defaults: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load defaults from the config file."""
        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    self._defaults = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._defaults = {}
        else:
            self._defaults = {}

    def save(self) -> None:
        """Save defaults to the config file."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._config_file, "w") as f:
            json.dump(self._defaults, f, indent=2)

    def get_plugin_defaults(self, plugin_name: str) -> dict[str, Any]:
        """Get stored default overrides for a plugin."""
        return dict(self._defaults.get(plugin_name, {}))

    def set_plugin_defaults(self, plugin_name: str, params: dict[str, Any]) -> None:
        """Set default overrides for a plugin."""
        if params:
            self._defaults[plugin_name] = params
        elif plugin_name in self._defaults:
            del self._defaults[plugin_name]

    def get_all_defaults(self) -> dict[str, dict[str, Any]]:
        """Get all stored plugin defaults."""
        return dict(self._defaults)

    def clear_plugin_defaults(self, plugin_name: str) -> None:
        """Remove all stored defaults for a plugin."""
        self._defaults.pop(plugin_name, None)
