"""Scale bar plugin — add a calibrated scale bar overlay."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.parameters import ChoiceParameter, FloatParameter


class ScaleBar(BasePlugin):
    """Add a calibrated scale bar to an image.

    Produces an RGB float32 image with the scale bar drawn in white.
    """

    name = "Scale Bar"
    category = "Converters"
    description = "Add calibrated scale bar overlay"
    help_text = (
        "Draws a calibrated scale bar on the image. Set the pixel size and "
        "desired bar length in micrometres. The bar is drawn in white and "
        "positioned in the chosen corner with automatic margin."
    )
    icon = None

    accepted_image_types = {ImageType.SINGLE}

    parameters = [
        FloatParameter(
            name="pixel_size_um",
            label="Pixel Size (um)",
            default=0.5,
            min_value=0.001,
            max_value=1000.0,
            step=0.1,
            decimals=3,
        ),
        FloatParameter(
            name="scale_bar_um",
            label="Scale Bar (um)",
            default=50.0,
            min_value=0.1,
            max_value=10000.0,
            step=10.0,
            decimals=1,
        ),
        ChoiceParameter(
            name="position",
            label="Position",
            choices=["bottom-right", "bottom-left", "top-right", "top-left"],
            default="bottom-right",
        ),
    ]

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        progress_callback(0.1)

        data = image.data.astype(np.float32, copy=True)
        pixel_size = float(self.get_parameter("pixel_size_um"))
        bar_um = float(self.get_parameter("scale_bar_um"))
        position = self.get_parameter("position")

        # Normalize to 0-1
        pmax = float(np.max(data))
        if pmax > 1.0:
            data = data / pmax

        # Convert grayscale to RGB
        if data.ndim == 2:
            data = np.stack([data, data, data], axis=-1)

        progress_callback(0.3)

        h, w = data.shape[:2]
        bar_px = int(round(bar_um / pixel_size))
        bar_h = max(4, int(round(h * 0.02)))
        margin = int(round(min(h, w) * 0.05))

        if "right" in position:
            x1, x2 = w - margin - bar_px, w - margin
        else:
            x1, x2 = margin, margin + bar_px

        if "bottom" in position:
            y1, y2 = h - margin - bar_h, h - margin
        else:
            y1, y2 = margin, margin + bar_h

        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        data[y1:y2, x1:x2, :] = 1.0

        metadata = image.metadata.copy()
        metadata.add_history(f"Scale Bar ({bar_um} um)")

        progress_callback(1.0)
        return ImageContainer(data=data, metadata=metadata)

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
