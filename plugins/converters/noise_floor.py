"""Noise floor plugin — zeros pixels below a robust noise estimate."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.parameters import FloatParameter


def _robust_sigma(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad + 1e-6


class NoiseFloor(BasePlugin):
    """Zero pixels below a robust noise estimate.

    Estimates noise from the lower-intensity portion of the image
    and zeros any pixel whose value falls below
    ``median + sigma_threshold * robust_sigma``.
    """

    name = "Noise Floor"
    category = "Converters"
    description = "Zero pixels below robust noise estimate"
    help_text = (
        "Zeros pixels that fall below a robust noise estimate. Noise is "
        "estimated from the lower-intensity portion of the image using the "
        "Median Absolute Deviation (MAD). Pixels below median + (sigma "
        "threshold \u00d7 robust sigma) are set to zero. Useful for cleaning "
        "up dim autofluorescence or camera noise before analysis."
    )
    icon = None

    accepted_image_types = {ImageType.SINGLE, ImageType.Z_STACK, ImageType.TIMELAPSE}

    parameters = [
        FloatParameter(
            name="sigma_threshold",
            label="Sigma Threshold",
            default=3.0,
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            decimals=1,
        ),
        FloatParameter(
            name="noise_estimate_percentile",
            label="Noise Estimate Percentile",
            default=70.0,
            min_value=10.0,
            max_value=99.0,
            step=5.0,
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
        sigma = float(self.get_parameter("sigma_threshold"))
        pct = float(self.get_parameter("noise_estimate_percentile"))

        if image.image_type == ImageType.SINGLE:
            result = self._process_plane(data, sigma, pct)
        else:
            n = data.shape[0]
            slices = []
            for i in range(n):
                slices.append(self._process_plane(data[i], sigma, pct))
                progress_callback(0.1 + 0.8 * (i + 1) / n)
            result = np.stack(slices, axis=0)

        # Restore original dtype
        if np.issubdtype(orig_dtype, np.integer):
            result = np.clip(result, np.iinfo(orig_dtype).min, np.iinfo(orig_dtype).max)
            result = np.round(result).astype(orig_dtype)
        elif orig_dtype != result.dtype:
            result = result.astype(orig_dtype)

        metadata = image.metadata.copy()
        metadata.add_history(f"Noise Floor (sigma={sigma:.1f})")

        progress_callback(1.0)
        return ImageContainer(data=result, metadata=metadata)

    def _process_plane(self, plane, sigma, pct):
        if plane.ndim == 2:
            return self._apply_2d(plane, sigma, pct)
        return np.stack(
            [self._apply_2d(plane[..., c], sigma, pct) for c in range(plane.shape[-1])],
            axis=-1,
        )

    @staticmethod
    def _apply_2d(img, sigma, pct):
        if sigma <= 0:
            return img.astype(np.float32, copy=False)
        y = img.astype(np.float32, copy=True)
        cutoff = float(np.percentile(y, pct))
        sample = y[y <= cutoff]
        if sample.size < 1000:
            return y
        sig = _robust_sigma(sample)
        thr = float(np.median(sample)) + sigma * sig
        y[y < max(0.0, thr)] = 0.0
        return y

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
