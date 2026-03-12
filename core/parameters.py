"""Parameter type definitions for plugin configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Parameter:
    """Base class for all parameter types."""

    name: str
    label: str
    default: Any = None

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a value for this parameter. Returns (is_valid, error_message)."""
        return True, ""


@dataclass
class IntParameter(Parameter):
    """Integer parameter with optional range constraints."""

    default: int = 0
    min_value: int = 0
    max_value: int = 100
    step: int = 1

    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, int):
            return False, f"{self.label} must be an integer"
        if value < self.min_value:
            return False, f"{self.label} must be at least {self.min_value}"
        if value > self.max_value:
            return False, f"{self.label} must be at most {self.max_value}"
        return True, ""


@dataclass
class FloatParameter(Parameter):
    """Float parameter with optional range constraints."""

    default: float = 0.0
    min_value: float = 0.0
    max_value: float = 1.0
    step: float = 0.1
    decimals: int = 2

    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, (int, float)):
            return False, f"{self.label} must be a number"
        if value < self.min_value:
            return False, f"{self.label} must be at least {self.min_value}"
        if value > self.max_value:
            return False, f"{self.label} must be at most {self.max_value}"
        return True, ""


@dataclass
class ChoiceParameter(Parameter):
    """Parameter with a fixed set of choices."""

    choices: list[str] = field(default_factory=list)
    default: str = ""

    def __post_init__(self):
        if self.choices and not self.default:
            self.default = self.choices[0]

    def validate(self, value: Any) -> tuple[bool, str]:
        if value not in self.choices:
            return False, f"{self.label} must be one of: {', '.join(self.choices)}"
        return True, ""


@dataclass
class BoolParameter(Parameter):
    """Boolean parameter."""

    default: bool = False
    group: str = ""

    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, bool):
            return False, f"{self.label} must be a boolean"
        return True, ""


@dataclass
class StringParameter(Parameter):
    """Free-form text parameter."""

    default: str = ""
    placeholder: str = ""

    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, str):
            return False, f"{self.label} must be a string"
        return True, ""


@dataclass
class FileParameter(Parameter):
    """File path parameter with optional filter."""

    default: str = ""
    filter: str = "All Files (*.*)"
    save_mode: bool = False  # True for save dialogs, False for open dialogs
    folder_mode: bool = False  # True for folder selection dialogs

    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, str):
            return False, f"{self.label} must be a file path"
        return True, ""


@dataclass
class ActionParameter(Parameter):
    """A button that triggers a plugin action when clicked.

    The ``callback`` names a method on the plugin class.  That method
    receives the node's current input data and returns a dict of
    parameter names to new values which are applied automatically.
    """

    callback: str = ""
    button_label: str = ""

    def __post_init__(self):
        if not self.button_label:
            self.button_label = self.label
