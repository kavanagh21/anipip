"""Brightness & Contrast adjustment plugin (data-range-aware)."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.parameters import ActionParameter, ChoiceParameter, FloatParameter


class BrightnessContrast(BasePlugin):
    """Adjust brightness and contrast via a linear display-range remap.

    Two modes:

    * **Relative** (default) — Auto-detects the actual data range using
      percentiles, then applies brightness/contrast adjustments relative
      to that range.  With both sliders at 0 the image is auto-levelled
      (stretched to fill the output range).  This is the most useful mode
      for scientific images where the data occupies a narrow band of the
      full bit depth.

    * **Manual** — Set the display window min/max directly in the
      image's native units.  Use *Fit to Data* to auto-populate the
      sliders from the actual image range, then fine-tune.

    Supports 8-bit, 16-bit, and float images.  Multi-channel images
    have each channel adjusted identically.
    """

    name = "Brightness/Contrast"
    category = "Converters"
    description = "Adjust brightness and contrast (data-range-aware)"
    help_text = (
        "Adjusts the display range of the image. In Relative mode, the "
        "actual data range is auto-detected and the sliders adjust relative "
        "to it \u2014 brightness shifts the window, contrast narrows or widens "
        "it. In Manual mode, set explicit display min/max. Use \"Fit to Data\" "
        "to lock the Manual sliders to the image's actual range."
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
            min_value=-100000.0,
            max_value=100000.0,
            step=0.1,
            decimals=2,
        ),
        FloatParameter(
            name="display_max",
            label="Display Max",
            default=65535.0,
            min_value=-100000.0,
            max_value=100000.0,
            step=0.1,
            decimals=2,
        ),
        ActionParameter(
            name="fit_to_data",
            label="Fit to Data",
            callback="_fit_to_data",
            button_label="Fit to Data",
        ),
        ActionParameter(
            name="auto_levels",
            label="Auto Levels (tight)",
            callback="_auto_levels",
            button_label="Auto Levels (tight)",
        ),
    ]

    def _fit_to_data(self, inputs):
        """Set display min/max to the image's actual data range (broad)."""
        image = inputs.get("image_in")
        if image is None:
            return {}
        data = image.data
        lo = float(np.percentile(data, 0.5))
        hi = float(np.percentile(data, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        return {"mode": "Manual", "display_min": lo, "display_max": hi}

    def _auto_levels(self, inputs):
        """Set display min/max using tighter percentiles for more contrast."""
        image = inputs.get("image_in")
        if image is None:
            return {}
        data = image.data
        lo = float(np.percentile(data, 1.0))
        hi = float(np.percentile(data, 99.0))
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

        # Determine the full output intensity range for this dtype
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
            history = f"Brightness/Contrast (manual {lo:.1f}\u2013{hi:.1f})"
        else:
            # Relative mode: compute window from actual data range
            data_lo, data_hi = self._data_range(data)
            data_range = data_hi - data_lo
            data_center = (data_lo + data_hi) / 2.0

            brightness = float(self.get_parameter("brightness"))
            contrast = float(self.get_parameter("contrast"))

            # Brightness shifts the window center (positive = brighter)
            center = data_center - brightness * data_range

            # Contrast narrows the window (positive = more contrast)
            width_factor = max(1.0 - contrast, 0.01)
            half_width = data_range * 0.5 * width_factor

            lo = center - half_width
            hi = center + half_width
            history = (
                f"Brightness/Contrast "
                f"(brightness={brightness:+.3f}, contrast={contrast:+.3f}, "
                f"window={lo:.1f}\u2013{hi:.1f})"
            )

        progress_callback(0.3)

        # Linear remap: [lo, hi] → [range_min, range_max]
        if hi <= lo:
            result = np.where(data.astype(np.float64) > lo, range_max, range_min)
        else:
            result = (data.astype(np.float64) - lo) / (hi - lo) * full_range + range_min
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

    @staticmethod
    def _data_range(data):
        """Compute the actual data range using robust percentiles."""
        lo = float(np.percentile(data, 0.5))
        hi = float(np.percentile(data, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
