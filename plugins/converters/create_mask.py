"""Create mask plugin — statistical thresholding with morphological cleanup."""

from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing
from skimage.morphology import disk

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.parameters import FloatParameter, IntParameter


def _remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    out = np.zeros_like(mask, dtype=np.float32)
    for i, s in enumerate(sizes):
        if s >= min_size:
            out[lab == (i + 1)] = 1.0
    return out


class CreateMask(BasePlugin):
    """Create a binary mask using statistical thresholding.

    Uses background statistics (median + MAD) to determine a threshold,
    then applies morphological opening/closing cleanup.  For Z-stacks
    each slice is masked independently.

    Input should ideally be normalized to [0, 1] (run Normalize Image first).
    """

    name = "Create Mask"
    category = "Converters"
    description = "Statistical thresholding + morphological cleanup"
    help_text = (
        "Creates a binary mask by statistical thresholding. Estimates a "
        "threshold from the background distribution (median + SD multiplier "
        "\u00d7 robust sigma), then cleans up with morphological opening and "
        "closing. Small objects below Min Object Size are removed. Input "
        "should ideally be normalised to [0, 1] first."
    )
    icon = None

    accepted_image_types = {ImageType.SINGLE, ImageType.Z_STACK, ImageType.TIMELAPSE}

    parameters = [
        FloatParameter(
            name="sd_multiplier",
            label="SD Multiplier",
            default=4.0,
            min_value=0.5,
            max_value=20.0,
            step=0.5,
            decimals=1,
        ),
        IntParameter(
            name="min_object_size",
            label="Min Object Size (px)",
            default=200,
            min_value=0,
            max_value=100000,
            step=50,
        ),
        IntParameter(
            name="opening_radius",
            label="Opening Radius",
            default=1,
            min_value=0,
            max_value=20,
        ),
        IntParameter(
            name="closing_radius",
            label="Closing Radius",
            default=2,
            min_value=0,
            max_value=20,
        ),
    ]

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        progress_callback(0.1)

        data = image.data
        sd = float(self.get_parameter("sd_multiplier"))
        min_size = int(self.get_parameter("min_object_size"))
        opening_r = int(self.get_parameter("opening_radius"))
        closing_r = int(self.get_parameter("closing_radius"))

        if image.image_type == ImageType.SINGLE:
            result = self._mask_plane(data, sd, min_size, opening_r, closing_r)
        else:
            n = data.shape[0]
            slices = []
            for i in range(n):
                slices.append(self._mask_plane(data[i], sd, min_size, opening_r, closing_r))
                progress_callback(0.1 + 0.8 * (i + 1) / n)
            result = np.stack(slices, axis=0)

        metadata = image.metadata.copy()
        metadata.add_history(f"Create Mask (sd={sd:.1f})")

        progress_callback(1.0)
        return ImageContainer(data=result, metadata=metadata)

    def _mask_plane(self, plane, sd, min_size, opening_r, closing_r):
        gray = self._to_gray_2d(plane)

        # Auto-normalize to 0-1 if needed
        pmax = float(np.max(gray))
        if pmax > 1.0:
            gray = gray / pmax

        # Statistical threshold: median + sd * robust_sigma of background
        cutoff = float(np.percentile(gray, 70))
        bg = gray[gray <= cutoff]

        if bg.size < 100:
            thr = float(np.percentile(gray.ravel(), 95))
        else:
            med = float(np.median(bg))
            mad = float(np.median(np.abs(bg - med)))
            robust_sig = 1.4826 * mad
            thr = med + sd * robust_sig

        thr = max(0.01, min(0.99, thr))
        mask = (gray > thr).astype(np.float32)

        if min_size > 0:
            mask = _remove_small_objects(mask, min_size)

        # Morphological cleanup
        m = mask > 0
        if opening_r >= 1:
            m = binary_opening(m, structure=disk(opening_r))
        if closing_r >= 1:
            m = binary_closing(m, structure=disk(closing_r))
        return m.astype(np.float32)

    @staticmethod
    def _to_gray_2d(data):
        if data.ndim == 2:
            return data.astype(np.float32)
        return np.mean(data[..., :3].astype(np.float32), axis=-1)

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
