"""Background subtraction plugin."""

from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter, grey_opening, uniform_filter, zoom
from skimage.morphology import disk
from skimage.restoration import rolling_ball

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.parameters import ChoiceParameter, FloatParameter


class BackgroundSubtraction(BasePlugin):
    """Subtract background illumination from an image.

    Supports Rolling Ball, Gaussian, Percentile Baseline, and
    Percentile + Top-Hat methods.  For Z-stacks and timelapses each
    slice is processed independently.
    """

    name = "Background Subtraction"
    category = "Converters"
    description = "Remove background using rolling ball, Gaussian, percentile, or top-hat methods"
    help_text = (
        "Removes uneven background illumination. Rolling Ball (the ImageJ "
        "standard) rolls a ball under the intensity surface and subtracts the "
        "estimated background. Gaussian subtracts a heavily blurred copy of "
        "the image. Percentile subtracts a flat baseline. Top-Hat combines "
        "percentile removal with morphological opening to suppress broad "
        "structures."
    )
    icon = None

    accepted_image_types = {ImageType.SINGLE, ImageType.Z_STACK, ImageType.TIMELAPSE}

    parameters = [
        ChoiceParameter(
            name="method",
            label="Method",
            choices=["Rolling Ball", "Gaussian", "Percentile", "Percentile + Top-Hat"],
            default="Rolling Ball",
        ),
        FloatParameter(
            name="rolling_ball_radius",
            label="Rolling Ball Radius",
            default=50.0,
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            decimals=1,
        ),
        FloatParameter(
            name="gaussian_radius",
            label="Gaussian Radius",
            default=50.0,
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            decimals=1,
        ),
        FloatParameter(
            name="percentile",
            label="Percentile",
            default=50.0,
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            decimals=1,
        ),
        FloatParameter(
            name="tophat_radius",
            label="Top-Hat Radius",
            default=30.0,
            min_value=1.0,
            max_value=200.0,
            step=1.0,
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
        orig_dtype = data.dtype
        method = self.get_parameter("method")

        if image.image_type == ImageType.SINGLE:
            result = self._process_plane(data, method)
        else:
            n = data.shape[0]
            slices = []
            for i in range(n):
                slices.append(self._process_plane(data[i], method))
                progress_callback(0.1 + 0.8 * (i + 1) / n)
            result = np.stack(slices, axis=0)

        # Restore original dtype
        if np.issubdtype(orig_dtype, np.integer):
            result = np.clip(result, np.iinfo(orig_dtype).min, np.iinfo(orig_dtype).max)
            result = np.round(result).astype(orig_dtype)
        elif orig_dtype != result.dtype:
            result = result.astype(orig_dtype)

        metadata = image.metadata.copy()
        metadata.add_history(f"Background Subtraction ({method})")

        progress_callback(1.0)
        return ImageContainer(data=result, metadata=metadata)

    def _process_plane(self, plane, method):
        if plane.ndim == 2:
            return self._subtract_2d(plane, method)
        return np.stack(
            [self._subtract_2d(plane[..., c], method) for c in range(plane.shape[-1])],
            axis=-1,
        )

    def _subtract_2d(self, img, method):
        x = img.astype(np.float32, copy=True)

        if method == "Rolling Ball":
            radius = int(round(float(self.get_parameter("rolling_ball_radius"))))
            # ImageJ-style shrink/expand optimisation for large radii
            shrink = max(1, radius // 10)
            if shrink > 1:
                smoothed = uniform_filter(x, size=3)
                small = smoothed[::shrink, ::shrink]
                small_radius = max(1, radius // shrink)
                small_bg = rolling_ball(small, radius=small_radius)
                bg = zoom(small_bg,
                          (x.shape[0] / small_bg.shape[0],
                           x.shape[1] / small_bg.shape[1]),
                          order=1)
                # Handle potential off-by-one from zoom
                bg = bg[:x.shape[0], :x.shape[1]]
                if bg.shape[0] < x.shape[0] or bg.shape[1] < x.shape[1]:
                    bg = np.pad(bg,
                                ((0, x.shape[0] - bg.shape[0]),
                                 (0, x.shape[1] - bg.shape[1])),
                                mode='edge')
            else:
                bg = rolling_ball(x, radius=radius)
            y = x - bg

        elif method == "Gaussian":
            radius = float(self.get_parameter("gaussian_radius"))
            bg = gaussian_filter(x, sigma=radius / 2.0)
            y = x - bg

        elif method == "Percentile":
            pct = float(self.get_parameter("percentile"))
            floor = float(np.percentile(x, pct))
            y = x - floor

        else:  # Percentile + Top-Hat
            pct = float(self.get_parameter("percentile"))
            tophat_r = int(round(float(self.get_parameter("tophat_radius"))))
            floor = float(np.percentile(x, pct))
            y = x - floor
            y[y < 0] = 0
            if tophat_r >= 1:
                opened = grey_opening(y, footprint=disk(tophat_r))
                y = y - opened

        y[y < 0] = 0
        return y

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
