"""Z-Projection plugin — collapse a Z-stack into a single 2-D image."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.parameters import BoolParameter, ChoiceParameter, FloatParameter, IntParameter


def _blur_scores(stack: np.ndarray, patch_radius: int = 25) -> np.ndarray:
    """Score each slice by sharpness (std of centre patch). Higher = sharper."""
    n, h, w = stack.shape[0], stack.shape[1], stack.shape[2]
    r = max(1, patch_radius)
    r1, r2 = max(0, h // 2 - r), min(h, h // 2 + r)
    c1, c2 = max(0, w // 2 - r), min(w, w // 2 + r)
    return np.array(
        [float(np.std(stack[i, r1:r2, c1:c2].astype(np.float32))) for i in range(n)],
        dtype=np.float32,
    )


def _keep_indices(scores: np.ndarray, remove_frac: float, min_slices: int) -> np.ndarray:
    """Choose slice indices to keep after removing the blurriest fraction."""
    n = scores.size
    if remove_frac <= 0 or n <= min_slices:
        return np.arange(n, dtype=int)
    thresh = np.percentile(scores, remove_frac * 100.0)
    keep = np.where(scores > thresh)[0]
    if keep.size < min_slices:
        keep = np.argsort(scores)[-min_slices:]
    return np.sort(keep.astype(int))


class ZProjection(BasePlugin):
    """Collapse a Z-stack into a single 2-D image.

    Supports Maximum, Mean, Sum, and Top-K Mean projection methods.
    Optionally filters out blurry slices before projecting.
    """

    name = "Z-Projection"
    category = "Converters"
    description = "Collapse Z-stack to 2D via max, mean, sum, or top-k projection"
    help_text = (
        "Collapses a Z-stack into a single 2D image. Maximum projection shows "
        "the brightest value at each pixel across all slices. Mean and Sum "
        "average or sum all slices. Top-K Mean averages only the K brightest "
        "slices per pixel. Optionally filters out blurry slices before "
        "projecting."
    )
    icon = None

    accepted_image_types = {ImageType.Z_STACK}

    parameters = [
        ChoiceParameter(
            name="method",
            label="Method",
            choices=["Maximum", "Mean", "Sum", "Top-K Mean"],
            default="Maximum",
        ),
        IntParameter(
            name="k",
            label="K (for Top-K)",
            default=3,
            min_value=1,
            max_value=100,
        ),
        BoolParameter(
            name="blur_filter",
            label="Filter Blurry Slices",
            default=False,
        ),
        FloatParameter(
            name="blur_remove_percent",
            label="Blur Remove %",
            default=0.2,
            min_value=0.0,
            max_value=0.9,
            step=0.05,
            decimals=2,
        ),
        IntParameter(
            name="min_slices",
            label="Min Slices to Keep",
            default=3,
            min_value=1,
            max_value=100,
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
        k = int(self.get_parameter("k"))
        blur_filter = self.get_parameter("blur_filter")
        blur_pct = float(self.get_parameter("blur_remove_percent"))
        min_sl = int(self.get_parameter("min_slices"))

        if data.ndim == 4:
            channels = [
                self._project(data[..., c], method, k, blur_filter, blur_pct, min_sl)
                for c in range(data.shape[-1])
            ]
            result = np.stack(channels, axis=-1)
        else:
            result = self._project(data, method, k, blur_filter, blur_pct, min_sl)

        # Restore original dtype — projection math uses float32 internally
        if np.issubdtype(orig_dtype, np.integer):
            result = np.clip(result, np.iinfo(orig_dtype).min, np.iinfo(orig_dtype).max)
            result = np.round(result).astype(orig_dtype)
        elif orig_dtype != result.dtype:
            result = result.astype(orig_dtype)

        progress_callback(0.8)

        metadata = image.metadata.copy()
        metadata.image_type = ImageType.SINGLE
        metadata.num_slices = 1
        metadata.add_history(f"Z-Projection ({method})")

        progress_callback(1.0)
        return ImageContainer(data=result, metadata=metadata)

    def _project(self, stack, method, k, blur_filter, blur_pct, min_sl):
        if blur_filter:
            keep = _keep_indices(_blur_scores(stack), blur_pct, min_sl)
        else:
            keep = np.arange(stack.shape[0], dtype=int)

        sub = stack[keep].astype(np.float32)
        if sub.shape[0] == 0:
            return np.zeros(stack.shape[1:], dtype=np.float32)

        if method == "Maximum":
            return np.max(sub, axis=0)
        elif method == "Mean":
            return np.mean(sub, axis=0).astype(np.float32)
        elif method == "Sum":
            return np.sum(sub.astype(np.float64), axis=0).astype(np.float32)
        else:  # Top-K Mean
            kk = min(max(1, k), sub.shape[0])
            # Sort along Z-axis descending, take top-k per pixel, average
            sorted_desc = np.sort(sub, axis=0)[::-1]
            return np.mean(sorted_desc[:kk], axis=0).astype(np.float32)

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        if image.image_type != ImageType.Z_STACK:
            return False, "Z-Projection requires a Z-stack input"
        return True, ""
