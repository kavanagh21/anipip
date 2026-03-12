"""Normalize image plugin — rescale intensity to [0, 1]."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.parameters import ChoiceParameter, FloatParameter


class NormalizeImage(BasePlugin):
    """Normalize image intensity to the [0, 1] range.

    In *Percentile* mode the min/max are computed automatically from
    the non-zero pixel distribution.  In *Manual* mode explicit
    min/max values are used.
    """

    name = "Normalize Image"
    category = "Converters"
    description = "Percentile or manual min/max normalization to [0, 1]"
    help_text = (
        "Rescales image intensity to the 0\u20131 float range. In Percentile mode, "
        "the min/max are computed automatically from the non-zero pixel "
        "distribution. In Manual mode, you specify explicit min/max values. "
        "Useful as a preprocessing step before thresholding or mask creation."
    )
    icon = None

    accepted_image_types = {ImageType.SINGLE, ImageType.Z_STACK, ImageType.TIMELAPSE}

    parameters = [
        ChoiceParameter(
            name="method",
            label="Method",
            choices=["Percentile", "Manual"],
            default="Percentile",
        ),
        FloatParameter(
            name="min_percentile",
            label="Min Percentile",
            default=0.01,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            decimals=3,
        ),
        FloatParameter(
            name="max_percentile",
            label="Max Percentile",
            default=0.998,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            decimals=3,
        ),
        FloatParameter(
            name="min_value",
            label="Min Value",
            default=0.0,
            min_value=-1e6,
            max_value=1e6,
            step=1.0,
            decimals=2,
        ),
        FloatParameter(
            name="max_value",
            label="Max Value",
            default=65535.0,
            min_value=-1e6,
            max_value=1e6,
            step=1.0,
            decimals=2,
        ),
    ]

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        progress_callback(0.1)

        data = image.data
        method = self.get_parameter("method")

        if method == "Percentile":
            lo, hi = self._auto_range(
                data,
                float(self.get_parameter("min_percentile")),
                float(self.get_parameter("max_percentile")),
            )
        else:
            lo = float(self.get_parameter("min_value"))
            hi = float(self.get_parameter("max_value"))

        progress_callback(0.3)

        denom = hi - lo
        if denom <= 0:
            denom = 1.0
        result = np.clip((data.astype(np.float32) - lo) / denom, 0, 1)

        metadata = image.metadata.copy()
        metadata.add_history(f"Normalize ({method}, range={lo:.2f}-{hi:.2f})")

        progress_callback(1.0)
        return ImageContainer(data=result, metadata=metadata)

    @staticmethod
    def _auto_range(data, min_pct, max_pct):
        lo_p, hi_p = min(min_pct, max_pct), max(min_pct, max_pct)
        vals = data[data > 0]
        if vals.size > 100:
            lo = float(np.percentile(vals, lo_p * 100.0))
            hi = float(np.percentile(vals, hi_p * 100.0))
        else:
            lo = float(np.min(data))
            hi = float(np.max(data))
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
