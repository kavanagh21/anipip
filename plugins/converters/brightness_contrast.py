"""Brightness & Contrast adjustment plugin (ImageJ-style display range)."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.parameters import ActionParameter, ChoiceParameter, FloatParameter


class BrightnessContrast(BasePlugin):
    """Adjust brightness and contrast via a linear display-range remap.

    Two modes:

    * **Relative** — ImageJ-style normalised sliders (brightness/contrast
      each -1..+1).  With both at 0 the image is unchanged.
    * **Manual** — set the display window min/max directly in the
      image's native units (e.g. 100-4000 for a 16-bit TIFF).
      Pixels at or below *Min* become black; pixels at or above *Max*
      become the full output value.

    Supports 8-bit, 16-bit, and float images.  Multi-channel images
    have each channel adjusted identically.
    """

    name = "Brightness/Contrast"
    category = "Converters"
    description = "Adjust brightness and contrast (ImageJ-style)"
    help_text = (
        "Adjusts the display range of the image. In Relative mode, sliders "
        "work like ImageJ \u2014 brightness shifts the window, contrast narrows "
        "it. In Manual mode, set explicit display min/max in native units. "
        "The Auto Levels button computes optimal min/max from the image data."
    )
    icon = None

    accepted_image_types = {ImageType.SINGLE, ImageType.Z_STACK, ImageType.TIMELAPSE}

    parameters = [
        ChoiceParameter(
            name="mode",
            label="Mode",
            choices=["Relative", "Manual"],
            default="Relative",
        ),
        # -- Relative mode --
        FloatParameter(
            name="brightness",
            label="Brightness",
            default=0.0,
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            decimals=3,
        ),
        FloatParameter(
            name="contrast",
            label="Contrast",
            default=0.0,
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            decimals=3,
        ),
        # -- Manual mode --
        FloatParameter(
            name="display_min",
            label="Display Min",
            default=0.0,
            min_value=0.0,
            max_value=65535.0,
            step=1.0,
            decimals=1,
        ),
        FloatParameter(
            name="display_max",
            label="Display Max",
            default=65535.0,
            min_value=0.0,
            max_value=65535.0,
            step=1.0,
            decimals=1,
        ),
        ActionParameter(
            name="auto_levels",
            label="Auto Levels",
            callback="_auto_levels",
            button_label="Auto Levels",
        ),
    ]

    def _auto_levels(self, inputs):
        """Compute display min/max from the actual image data."""
        image = inputs.get("image_in")
        if image is None:
            return {}
        data = image.data
        lo = float(np.percentile(data, 0.1))
        hi = float(np.percentile(data, 99.9))
        if hi <= lo:
            hi = lo + 1.0
        return {"mode": "Manual", "display_min": lo, "display_max": hi}

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        progress_callback(0.1)

        data = image.data
        metadata = image.metadata.copy()
        dtype = data.dtype

        # Determine the full intensity range for this dtype
        if dtype in (np.float32, np.float64):
            range_min, range_max = 0.0, 1.0
        elif dtype == np.uint8:
            range_min, range_max = 0.0, 255.0
        elif dtype == np.uint16:
            range_min, range_max = 0.0, 65535.0
        else:
            range_min, range_max = 0.0, float(np.iinfo(dtype).max)

        full_range = range_max - range_min
        mode = self.get_parameter("mode")

        if mode == "Manual":
            lo = float(self.get_parameter("display_min"))
            hi = float(self.get_parameter("display_max"))
            if hi <= lo:
                hi = lo + 1.0
            history = f"Brightness/Contrast (manual {lo:.1f}-{hi:.1f})"
        else:
            brightness = float(self.get_parameter("brightness"))
            contrast = float(self.get_parameter("contrast"))
            centre = 0.5 - brightness * 0.5
            half_width = 0.5 * max(1.0 - contrast, 0.0)
            lo = range_min + (centre - half_width) * full_range
            hi = range_min + (centre + half_width) * full_range
            history = (
                f"Brightness/Contrast adjusted "
                f"(brightness={brightness:+.3f}, contrast={contrast:+.3f})"
            )

        progress_callback(0.3)

        if hi <= lo:
            result = np.where(data.astype(np.float64) > lo, range_max, range_min)
        else:
            result = (data.astype(np.float64) - lo) / (hi - lo) * full_range
            result = np.clip(result, range_min, range_max)

        progress_callback(0.8)

        # Restore original dtype
        if np.issubdtype(dtype, np.integer):
            result = np.round(result).astype(dtype)
        else:
            result = result.astype(dtype)

        metadata.add_history(history)

        progress_callback(1.0)
        return ImageContainer(data=result, metadata=metadata)

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
