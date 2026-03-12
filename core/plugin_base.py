"""Base plugin class that all plugins must inherit from."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from .image_container import ImageContainer, ImageType
from .parameters import Parameter
from .pipeline_data import PipelineData
from .ports import Port, InputPort, OutputPort, PortDirection


class BasePlugin(ABC):
    """Abstract base class for all pipeline plugins.

    All plugins must inherit from this class and implement the process method.
    Plugins define their parameters as class attributes, which are used to
    auto-generate the UI for configuration.

    Plugins may optionally define ``ports`` to declare typed input/output slots.
    When ``ports`` is empty (the default), a single ImageContainer input and
    output is auto-generated for backwards compatibility with the linear
    pipeline model.
    """

    # Class attributes for registration (override in subclasses)
    name: str = "Base Plugin"
    category: str = "Uncategorized"
    description: str = "No description provided"
    help_text: str = ""
    icon: Optional[str] = None

    # Parameter definitions for auto-generated UI
    parameters: list[Parameter] = []

    # Port definitions — leave empty for legacy single-image-in/out behaviour
    ports: list[Port] = []

    # Image types this plugin can process natively (others are auto-iterated)
    accepted_image_types: set[ImageType] = {ImageType.SINGLE}

    def __init__(self):
        """Initialize the plugin with default parameter values."""
        self._param_values: dict[str, any] = {}
        for param in self.parameters:
            self._param_values[param.name] = param.default

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def get_parameter(self, name: str) -> any:
        """Get the current value of a parameter."""
        return self._param_values.get(name)

    def set_parameter(self, name: str, value: any) -> None:
        """Set the value of a parameter."""
        self._param_values[name] = value

    def get_all_parameters(self) -> dict[str, any]:
        """Get all parameter values as a dictionary."""
        return self._param_values.copy()

    def set_all_parameters(self, values: dict[str, any]) -> None:
        """Set multiple parameter values at once."""
        for name, value in values.items():
            if name in self._param_values:
                self._param_values[name] = value

    def validate_parameters(self) -> tuple[bool, list[str]]:
        """Validate all current parameter values.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        for param in self.parameters:
            value = self._param_values.get(param.name)
            is_valid, error = param.validate(value)
            if not is_valid:
                errors.append(error)
        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Port helpers
    # ------------------------------------------------------------------

    def get_ports(self) -> list[Port]:
        """Return the port list for this plugin.

        If no ports have been explicitly declared, returns a default pair of
        ImageContainer input/output ports so that legacy plugins work
        transparently in the DAG pipeline.
        """
        if self.ports:
            return list(self.ports)
        # Default: single image in → single image out
        return [
            InputPort("image_in", ImageContainer, label="Image In"),
            OutputPort("image_out", ImageContainer, label="Image Out"),
        ]

    def get_input_ports(self) -> list[Port]:
        """Return only the input ports."""
        return [p for p in self.get_ports() if p.direction == PortDirection.INPUT]

    def get_output_ports(self) -> list[Port]:
        """Return only the output ports."""
        return [p for p in self.get_ports() if p.direction == PortDirection.OUTPUT]

    # ------------------------------------------------------------------
    # Action parameters
    # ------------------------------------------------------------------

    def execute_action(self, action_name: str, inputs: dict[str, "PipelineData"]) -> dict[str, any]:
        """Execute a named action parameter callback.

        Returns a dict of {param_name: new_value} to apply, or empty dict.
        """
        from .parameters import ActionParameter
        param = next(
            (p for p in self.parameters
             if isinstance(p, ActionParameter) and p.name == action_name),
            None,
        )
        if param is None or not param.callback:
            return {}
        method = getattr(self, param.callback, None)
        if method is None:
            return {}
        return method(inputs) or {}

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    @abstractmethod
    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        """Process an image and return the result.

        Args:
            image: Input image container
            progress_callback: Callback function that accepts progress (0.0-1.0)

        Returns:
            Processed image container
        """
        pass

    def process_ports(
        self,
        inputs: dict[str, PipelineData],
        progress_callback: Callable[[float], None],
    ) -> dict[str, PipelineData]:
        """Process data through typed ports.

        The default implementation wraps the legacy ``process()`` method:
        it takes ``image_in`` from *inputs*, passes it to ``process()``,
        and returns the result as ``image_out``.

        Plugins that define custom ports should override this method.

        Args:
            inputs: Mapping of input port name to data object
            progress_callback: Progress callback (0.0-1.0)

        Returns:
            Mapping of output port name to produced data object
        """
        image = inputs.get("image_in")
        result = self.process(image, progress_callback)
        return {"image_out": result}

    # ------------------------------------------------------------------
    # Batch lifecycle hooks
    # ------------------------------------------------------------------

    def batch_initialize(self) -> None:
        """Called once before a batch run begins.

        Override to set up accumulators or open resources.
        """
        pass

    def batch_finalize(self) -> None:
        """Called once after a batch run completes.

        Override to flush accumulators, write summary files, etc.
        """
        pass

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        """Validate if input is suitable for this plugin.

        Override this method to add input validation.

        Args:
            image: Input image to validate

        Returns:
            Tuple of (is_valid, error message if invalid)
        """
        return True, ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
