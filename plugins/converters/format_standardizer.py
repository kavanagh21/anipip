"""Format standardizer plugin for converting images to consistent format."""

from typing import Callable

import numpy as np
from skimage import color, img_as_ubyte, img_as_uint, img_as_float32

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer
from core.parameters import ChoiceParameter


class FormatStandardizer(BasePlugin):
    """Convert images to a standardized format for downstream processing."""

    name = "Format Standardizer"
    category = "Converters"
    description = "Normalize images to consistent bit depth and color mode"
    help_text = (
        "Converts images to a consistent bit depth (8, 16, or 32-bit float) "
        "and colour mode (grayscale, RGB, or preserve). Useful for ensuring "
        "all images in a pipeline share the same format before downstream "
        "processing."
    )
    icon = None

    parameters = [
        ChoiceParameter(
            name="bit_depth",
            label="Target Bit Depth",
            choices=["8", "16", "32"],
            default="8",
        ),
        ChoiceParameter(
            name="color_mode",
            label="Color Mode",
            choices=["preserve", "grayscale", "rgb"],
            default="preserve",
        ),
    ]

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        """Standardize image format.

        Args:
            image: Input image container
            progress_callback: Progress callback function

        Returns:
            Standardized image container
        """
        progress_callback(0.1)

        data = image.data.copy()
        metadata = image.metadata.copy()

        # Convert color mode
        target_mode = self.get_parameter("color_mode")
        progress_callback(0.3)

        if target_mode == "grayscale":
            data = self._to_grayscale(data)
            metadata.color_space = "grayscale"
        elif target_mode == "rgb":
            data = self._to_rgb(data)
            metadata.color_space = "rgb"

        progress_callback(0.6)

        # Convert bit depth
        target_depth = int(self.get_parameter("bit_depth"))
        data = self._convert_bit_depth(data, target_depth)
        metadata.bit_depth = target_depth

        progress_callback(0.9)

        # Update history
        metadata.add_history(
            f"Standardized to {target_depth}-bit {target_mode if target_mode != 'preserve' else metadata.color_space}"
        )

        progress_callback(1.0)

        return ImageContainer(data=data, metadata=metadata)

    def _to_grayscale(self, data: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if data.ndim == 2:
            return data

        if data.shape[2] == 1:
            return data[:, :, 0]

        if data.shape[2] == 4:
            # RGBA - ignore alpha and convert RGB
            data = data[:, :, :3]

        if data.shape[2] == 3:
            # Ensure float for color conversion
            if data.dtype == np.uint8:
                data_float = data.astype(np.float64) / 255.0
            elif data.dtype == np.uint16:
                data_float = data.astype(np.float64) / 65535.0
            else:
                data_float = data.astype(np.float64)

            gray = color.rgb2gray(data_float)
            return gray

        return data

    def _to_rgb(self, data: np.ndarray) -> np.ndarray:
        """Convert image to RGB."""
        if data.ndim == 2:
            # Grayscale to RGB
            return np.stack([data, data, data], axis=-1)

        if data.shape[2] == 1:
            # Single channel to RGB
            return np.stack([data[:, :, 0]] * 3, axis=-1)

        if data.shape[2] == 4:
            # RGBA to RGB - simple drop of alpha
            return data[:, :, :3]

        if data.shape[2] == 3:
            return data

        return data

    def _convert_bit_depth(self, data: np.ndarray, target_depth: int) -> np.ndarray:
        """Convert image to target bit depth."""
        # First normalize to float if needed
        if data.dtype == np.uint8:
            data_float = data.astype(np.float64) / 255.0
        elif data.dtype == np.uint16:
            data_float = data.astype(np.float64) / 65535.0
        elif data.dtype in (np.float32, np.float64):
            data_float = data.astype(np.float64)
            # Clip to valid range
            data_float = np.clip(data_float, 0.0, 1.0)
        else:
            data_float = data.astype(np.float64)

        # Convert to target depth
        if target_depth == 8:
            return (data_float * 255).astype(np.uint8)
        elif target_depth == 16:
            return (data_float * 65535).astype(np.uint16)
        else:  # 32-bit float
            return data_float.astype(np.float32)

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        """Validate input image."""
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
