"""Grayscale conversion plugin."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer
from core.parameters import ChoiceParameter


class Grayscale(BasePlugin):
    """Convert a colour image to grayscale.

    Supports several conversion methods:
    * **Luminance** — perceptual weighting (0.2126 R + 0.7152 G + 0.0722 B)
    * **Average** — equal weighting of R, G, B
    * **Lightness** — (max + min) / 2

    For images that are already single-channel the data is passed through
    unchanged regardless of the chosen method.
    """

    name = "Grayscale"
    category = "Converters"
    description = "Convert a colour image to grayscale"
    help_text = (
        "Converts a colour image to single-channel grayscale. Luminance uses "
        "perceptual weighting (ITU-R BT.709), Average weights R/G/B equally, "
        "and Lightness takes (max + min) / 2. Already-grayscale images pass "
        "through unchanged."
    )
    icon = None

    parameters = [
        ChoiceParameter(
            name="method",
            label="Method",
            choices=["Luminance", "Average", "Lightness"],
            default="Luminance",
        ),
    ]

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        progress_callback(0.1)

        data = image.data
        metadata = image.metadata.copy()

        # Already grayscale — pass through
        if data.ndim == 2 or (data.ndim == 3 and data.shape[2] == 1):
            progress_callback(1.0)
            return ImageContainer(data=data.copy(), metadata=metadata)

        progress_callback(0.3)

        rgb = data[:, :, :3].astype(np.float64)
        method = self.get_parameter("method")

        if method == "Luminance":
            gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        elif method == "Average":
            gray = np.mean(rgb, axis=2)
        else:  # Lightness
            gray = (np.max(rgb, axis=2) + np.min(rgb, axis=2)) / 2.0

        progress_callback(0.8)

        # Preserve original dtype
        if data.dtype in (np.uint8, np.uint16):
            gray = np.clip(gray, 0, np.iinfo(data.dtype).max).astype(data.dtype)
        else:
            gray = gray.astype(data.dtype)

        metadata.color_space = "grayscale"
        metadata.add_history(f"Converted to grayscale ({method})")

        progress_callback(1.0)
        return ImageContainer(data=gray, metadata=metadata)

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
