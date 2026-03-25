"""Noise analysis measurement plugin — report noise floor statistics."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.table_data import TableData
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import FloatParameter


def _robust_sigma(x: np.ndarray) -> float:
    """MAD-based robust standard deviation estimate."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad + 1e-6


class NoiseAnalysis(BasePlugin):
    """Measure the noise characteristics of an image.

    Estimates background noise using the Median Absolute Deviation (MAD)
    on the lower-intensity portion of the image, then reports noise
    floor statistics including robust sigma, SNR, the computed noise
    threshold, and the percentage of pixels that would be zeroed at
    that threshold.

    For Z-stacks, statistics are computed per-slice and both per-slice
    and aggregate (mean/std across slices) values are reported.

    The image is passed through unchanged.
    """

    name = "Noise Analysis"
    category = "Measurements"
    description = "Measure noise floor, robust sigma, and SNR"
    help_text = (
        "Computes noise statistics without modifying the image. Reports "
        "background median, robust sigma (MAD-based), signal-to-noise "
        "ratio, the noise floor threshold at the given sigma multiplier, "
        "and the percentage of pixels below that threshold. For Z-stacks, "
        "per-slice and aggregate statistics are reported."
    )
    icon = None

    ports: list[Port] = [
        InputPort("image_in", ImageContainer, label="Image In"),
        OutputPort("image_out", ImageContainer, label="Image Out"),
        OutputPort("measurements", TableData, label="Measurements"),
    ]

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
        FloatParameter(
            name="measurement_bg_percentile",
            label="Measurement BG Percentile",
            default=50.0,
            min_value=0.0,
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
        return image

    def process_ports(
        self,
        inputs: dict[str, PipelineData],
        progress_callback: Callable[[float], None],
    ) -> dict[str, PipelineData]:
        image: ImageContainer = inputs.get("image_in")
        if image is None:
            raise RuntimeError("No image provided to Noise Analysis")

        progress_callback(0.1)

        sigma_thr = float(self.get_parameter("sigma_threshold"))
        pct = float(self.get_parameter("noise_estimate_percentile"))
        meas_bg_pct = float(self.get_parameter("measurement_bg_percentile"))

        data = image.data

        # For stacks, analyse per-slice then aggregate
        if image.image_type != ImageType.SINGLE and data.ndim >= 3:
            slices = self._analyse_stack(data, sigma_thr, pct, meas_bg_pct, progress_callback)
        else:
            plane = data if data.ndim == 2 else self._to_gray(data)
            slices = [self._analyse_plane(plane, sigma_thr, pct, meas_bg_pct)]

        progress_callback(0.8)

        # Build output table
        table = TableData()
        source = image.metadata.source_path
        filename = str(source.name) if source else ""

        if len(slices) == 1:
            # Single image — one row
            row = {"filename": filename}
            row.update(slices[0])
            table.add_row(row)
        else:
            # Stack — aggregate row
            keys = [k for k in slices[0].keys() if isinstance(slices[0][k], (int, float))]
            row = {"filename": filename, "n_slices": len(slices)}
            for k in keys:
                vals = [s[k] for s in slices if isinstance(s.get(k), (int, float))]
                if vals:
                    row[f"mean_{k}"] = float(np.mean(vals))
                    row[f"std_{k}"] = float(np.std(vals))
                    row[f"min_{k}"] = float(np.min(vals))
                    row[f"max_{k}"] = float(np.max(vals))
            table.add_row(row)

        progress_callback(1.0)
        return {"image_out": image, "measurements": table}

    def _analyse_stack(self, data, sigma_thr, pct, meas_bg_pct, progress_callback):
        n = data.shape[0]
        results = []
        for i in range(n):
            plane = data[i]
            if plane.ndim > 2:
                plane = self._to_gray(plane)
            results.append(self._analyse_plane(plane, sigma_thr, pct, meas_bg_pct))
            progress_callback(0.1 + 0.7 * (i + 1) / n)
        return results

    @staticmethod
    def _analyse_plane(plane: np.ndarray, sigma_thr: float, pct: float, meas_bg_pct: float = 50.0) -> dict:
        """Compute noise statistics for a single 2D plane."""
        x = plane.astype(np.float32).ravel()

        # Background estimation from lower-intensity region
        cutoff = float(np.percentile(x, pct))
        bg = x[x <= cutoff]

        if bg.size < 100:
            bg = x

        bg_median = float(np.median(bg))
        noise_floor_sigma = _robust_sigma(bg)
        bg_mean = float(np.mean(bg))
        bg_std = float(np.std(bg))
        bg_mad = float(np.median(np.abs(bg - bg_median)))

        # Noise floor threshold at the configured sigma multiplier
        noise_threshold = bg_median + sigma_thr * noise_floor_sigma

        # Signal estimation
        signal_p95 = float(np.percentile(x, 95))
        signal_p99 = float(np.percentile(x, 99))
        signal_max = float(np.max(x))
        signal_mean = float(np.mean(x))

        # SNR (signal-to-noise ratio)
        snr = signal_p95 / noise_floor_sigma if noise_floor_sigma > 1e-6 else 0.0

        # Dynamic range
        p1 = float(np.percentile(x, 1))
        dynamic_range = signal_p99 - p1

        # Pixels below noise threshold
        below_count = int(np.sum(x < noise_threshold))
        below_percent = float(below_count) / float(x.size) * 100.0

        # Signal above threshold
        above = x[x >= noise_threshold]
        signal_above_count = int(above.size)
        signal_above_percent = float(signal_above_count) / float(x.size) * 100.0
        signal_above_mean = float(np.mean(above)) if above.size > 0 else 0.0

        # Non-zero pixel stats (useful after noise floor subtraction)
        nonzero = x[x > 0]
        nonzero_percent = float(nonzero.size) / float(x.size) * 100.0
        nonzero_mean = float(np.mean(nonzero)) if nonzero.size > 0 else 0.0

        # Signal-to-background ratio
        sbr = signal_above_mean / bg_mean if bg_mean > 1e-6 else 0.0

        # Measurement floor — the baseline percentile value that can be
        # subtracted post-hoc from intensity measurements to get net signal.
        # Matches julie pipeline's measurement_floor concept.
        measurement_floor = float(np.percentile(x, meas_bg_pct))

        return {
            "noise_floor_sigma": noise_floor_sigma,
            "bg_median": bg_median,
            "bg_mad": bg_mad,
            "bg_mean": bg_mean,
            "bg_std": bg_std,
            "noise_threshold": noise_threshold,
            "signal_mean": signal_mean,
            "signal_p95": signal_p95,
            "signal_p99": signal_p99,
            "signal_max": signal_max,
            "snr": snr,
            "sbr": sbr,
            "dynamic_range": dynamic_range,
            "below_threshold_percent": below_percent,
            "signal_above_percent": signal_above_percent,
            "signal_above_mean": signal_above_mean,
            "nonzero_percent": nonzero_percent,
            "nonzero_mean": nonzero_mean,
            "measurement_floor": measurement_floor,
        }

    @staticmethod
    def _to_gray(data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            return data
        if data.ndim == 3 and data.shape[-1] in (3, 4):
            return np.mean(data[..., :3].astype(np.float32), axis=-1)
        return data

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
