"""Gaussian and box blur plugin."""

from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.parameters import FloatParameter, ChoiceParameter


class Blur(BasePlugin):
    """Apply a spatial blur/soften to an image.

    Supports Gaussian and Box (uniform) blur.  The filter is applied
    independently to each spatial plane so that z-stack slices and
    timelapse frames are blurred in 2-D without blending across the
    Z/T axis, and colour channels are kept separate.

    Works on single images, z-stacks, and timelapses of any bit depth.
    """

    name = "Blur"
    category = "Converters"
    description = "Gaussian or box blur / soften"
    help_text = (
        "Applies a spatial smoothing filter. Gaussian uses a bell-curve kernel "
        "controlled by the Radius parameter. Box uses a uniform-weight square "
        "kernel. Each Z-slice and colour channel is blurred independently."
    )
    icon = None

    accepted_image_types = {ImageType.SINGLE, ImageType.Z_STACK, ImageType.TIMELAPSE}

    parameters = [
        ChoiceParameter(
            name="method",
            label="Method",
            choices=["Gaussian", "Box"],
            default="Gaussian",
        ),
        FloatParameter(
            name="radius",
            label="Radius",
            default=1.0,
            min_value=0.1,
            max_value=50.0,
            step=0.1,
            decimals=1,
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
        dtype = data.dtype

        method = self.get_parameter("method")
        radius = float(self.get_parameter("radius"))

        # Work in float for precision
        if np.issubdtype(dtype, np.integer):
            work = data.astype(np.float64)
        else:
            work = data.astype(np.float64)

        progress_callback(0.2)

        # Build per-axis sigma/size so blur is spatial-only (H, W).
        # Shapes:
        #   SINGLE grayscale  (H, W)       → blur axes 0,1
        #   SINGLE colour     (H, W, C)    → blur axes 0,1  skip C
        #   STACK  grayscale  (Z, H, W)    → blur axes 1,2  skip Z
        #   STACK  colour     (Z, H, W, C) → blur axes 1,2  skip Z,C
        is_stack = image.image_type != ImageType.SINGLE

        if is_stack:
            if work.ndim == 3:
                # (Z, H, W) grayscale stack
                sigma = (0, radius, radius)
                size = (1, int(round(radius * 2)) | 1, int(round(radius * 2)) | 1)
            else:
                # (Z, H, W, C) colour stack
                sigma = (0, radius, radius, 0)
                size = (1, int(round(radius * 2)) | 1, int(round(radius * 2)) | 1, 1)
        else:
            if work.ndim == 2:
                # (H, W) grayscale single
                sigma = (radius, radius)
                size = (int(round(radius * 2)) | 1, int(round(radius * 2)) | 1)
            else:
                # (H, W, C) colour single
                sigma = (radius, radius, 0)
                size = (int(round(radius * 2)) | 1, int(round(radius * 2)) | 1, 1)

        progress_callback(0.3)

        if method == "Gaussian":
            result = gaussian_filter(work, sigma=sigma)
        else:
            result = uniform_filter(work, size=size)

        progress_callback(0.8)

        # Restore original dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            result = np.clip(result, info.min, info.max)
            result = np.round(result).astype(dtype)
        else:
            result = result.astype(dtype)

        metadata.add_history(f"Blur ({method}, radius={radius:.1f})")

        progress_callback(1.0)
        return ImageContainer(data=result, metadata=metadata)

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
