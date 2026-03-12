"""Tissue slice filter plugin — keep Z-slices with sufficient tissue."""

from typing import Callable

import numpy as np
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter
from skimage.morphology import disk, remove_small_objects as _sk_remove_small
from skimage.filters import threshold_otsu

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.parameters import ChoiceParameter, FloatParameter, IntParameter


def _tissue_mask_2d(img: np.ndarray, method: str) -> np.ndarray:
    """Create a boolean tissue mask for a single 2-D slice."""
    x = img.astype(np.float32, copy=False)
    p1, p99 = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if (p99 - p1) < 25.0:
        return np.zeros_like(x, dtype=bool)

    xs = gaussian_filter(x, sigma=1.0)
    m = method.lower()

    if m == "otsu":
        samp = xs[::2, ::2].ravel()
        samp = samp[samp > p1]
        if samp.size < 200:
            return np.zeros_like(xs, dtype=bool)
        thr = float(threshold_otsu(samp))
    elif m == "percentile":
        thr = float(np.percentile(xs, 95))
        if thr <= p1:
            thr = p1 + 0.5 * (p99 - p1)
    else:  # MAD
        cutoff = float(np.percentile(xs, 70))
        bg = xs[xs <= cutoff]
        if bg.size < 500:
            bg = xs.ravel()
        med = float(np.median(bg))
        mad = float(np.median(np.abs(bg - med)))
        sig = 1.4826 * mad + 1e-6
        thr = min(med + 3.0 * sig, p99)

    mask = xs > thr
    mask = binary_closing(mask, structure=disk(2))
    mask = binary_opening(mask, structure=disk(1))
    mask = _sk_remove_small(mask, min_size=200)
    return mask


class TissueSliceFilter(BasePlugin):
    """Remove Z-slices with insufficient tissue coverage.

    Computes a tissue mask per slice and discards slices whose coverage
    falls below the threshold.  Falls back to keeping the best
    *min_keep_slices* if none exceed the threshold.
    """

    name = "Tissue Slice Filter"
    category = "Converters"
    description = "Keep Z-slices with sufficient tissue coverage"
    help_text = (
        "Removes Z-stack slices that contain too little tissue. Computes a "
        "tissue mask per slice using MAD, Otsu, or Percentile detection and "
        "discards slices whose coverage falls below the threshold. Falls back "
        "to keeping the best N slices if none pass."
    )
    icon = None

    accepted_image_types = {ImageType.Z_STACK}

    parameters = [
        FloatParameter(
            name="min_coverage_percent",
            label="Min Coverage %",
            default=5.0,
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            decimals=1,
        ),
        ChoiceParameter(
            name="method",
            label="Detection Method",
            choices=["MAD", "Otsu", "Percentile"],
            default="MAD",
        ),
        IntParameter(
            name="min_keep_slices",
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
        min_cov = float(self.get_parameter("min_coverage_percent"))
        method = self.get_parameter("method").lower()
        min_keep = int(self.get_parameter("min_keep_slices"))

        detect = data[..., 0] if data.ndim == 4 else data
        n = detect.shape[0]

        coverages = np.zeros(n, dtype=np.float32)
        for z in range(n):
            m = _tissue_mask_2d(detect[z], method)
            coverages[z] = float(np.sum(m)) / float(m.size) * 100.0
            progress_callback(0.1 + 0.7 * (z + 1) / n)

        keep = np.where(coverages >= min_cov)[0]
        if keep.size == 0:
            k = min(max(1, min_keep), n)
            keep = np.sort(np.argsort(coverages)[-k:])

        result = data[keep]

        metadata = image.metadata.copy()
        metadata.add_history(f"Tissue Filter (kept {len(keep)}/{n} slices)")

        progress_callback(1.0)
        return ImageContainer(data=result, metadata=metadata)

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        if image.image_type != ImageType.Z_STACK:
            return False, "Tissue Slice Filter requires a Z-stack input"
        return True, ""
