"""Intensity measurement plugin — computes per-image statistics."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer
from core.table_data import TableData
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import BoolParameter, ChoiceParameter, FloatParameter, StringParameter


def _robust_sigma(x: np.ndarray) -> float:
    """MAD-based robust standard deviation estimate."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad + 1e-6


class IntensityMeasurement(BasePlugin):
    """Compute intensity statistics for each image.

    Produces two outputs:
    * **image_out** — the input image passed through unchanged
    * **measurements** — a :class:`TableData` containing one row of statistics

    Optionally computes noise floor statistics (robust sigma, measurement
    floor) and provides noise-floor-adjusted versions of all measurements
    so that net signal above background can be reported directly.
    """

    name = "Intensity Measurement"
    category = "Measurements"
    description = "Compute mean, min, max, std, median, and sum intensity statistics"
    help_text = (
        "Computes per-image intensity statistics (mean, min, max, std, median, "
        "sum) and outputs them as a table row. Enable \"Report Noise Floor\" "
        "to include noise_floor_sigma and measurement_floor in the output. "
        "Enable \"Noise-Adjusted Values\" to also output net_ columns with "
        "the measurement floor subtracted. The image passes through unchanged."
    )
    icon = None

    # -- Ports ---------------------------------------------------------

    ports: list[Port] = [
        InputPort("image_in", ImageContainer, label="Image In"),
        OutputPort("image_out", ImageContainer, label="Image Out"),
        OutputPort("measurements", TableData, label="Measurements"),
    ]

    # -- Parameters ----------------------------------------------------

    parameters = [
        ChoiceParameter(
            name="channel_mode",
            label="Channel Mode (color only)",
            choices=["Per-Channel", "Luminance"],
            default="Per-Channel",
        ),
        BoolParameter(name="include_r", label="Red", default=True, group="Channels"),
        BoolParameter(name="include_g", label="Green", default=True, group="Channels"),
        BoolParameter(name="include_b", label="Blue", default=True, group="Channels"),
        BoolParameter(name="include_a", label="Alpha", default=False, group="Channels"),
        BoolParameter(name="measure_mean", label="Mean", default=True, group="Statistics"),
        BoolParameter(name="measure_min", label="Min", default=True, group="Statistics"),
        BoolParameter(name="measure_max", label="Max", default=True, group="Statistics"),
        BoolParameter(name="measure_std", label="Std Dev", default=True, group="Statistics"),
        BoolParameter(name="measure_median", label="Median", default=True, group="Statistics"),
        BoolParameter(name="measure_sum", label="Sum", default=False, group="Statistics"),
        StringParameter(
            name="column_prefix",
            label="Column Prefix",
            default="",
            placeholder="e.g. 'raw_'",
        ),
        BoolParameter(
            name="report_noise_floor",
            label="Report Noise Floor",
            default=False,
        ),
        BoolParameter(
            name="noise_adjusted",
            label="Noise-Adjusted Values",
            default=False,
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

    # Channel names and corresponding include_* parameter names
    _CHANNEL_NAMES = ["R", "G", "B", "A"]
    _CHANNEL_PARAMS = ["include_r", "include_g", "include_b", "include_a"]

    # -- Processing ----------------------------------------------------

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        """Legacy single-output interface — returns the image unchanged."""
        return image

    def process_ports(
        self,
        inputs: dict[str, PipelineData],
        progress_callback: Callable[[float], None],
    ) -> dict[str, PipelineData]:
        """Multi-port processing: pass image through and emit measurements."""
        image: ImageContainer = inputs.get("image_in")
        if image is None:
            raise RuntimeError("No image provided to Intensity Measurement")

        progress_callback(0.1)

        table = self._measure(image)

        progress_callback(0.9)

        return {
            "image_out": image,
            "measurements": table,
        }

    # -- Measurement logic ---------------------------------------------

    def _measure(self, image: ImageContainer) -> TableData:
        data = image.data.astype(np.float64)
        prefix = self.get_parameter("column_prefix") or ""
        mode = self.get_parameter("channel_mode")
        report_noise = self.get_parameter("report_noise_floor")
        do_adjusted = self.get_parameter("noise_adjusted")
        noise_pct = float(self.get_parameter("noise_estimate_percentile"))
        meas_bg_pct = float(self.get_parameter("measurement_bg_percentile"))

        row: dict = {}

        # Always include filename
        source = image.metadata.source_path
        row["filename"] = str(source.name) if source else ""

        is_multi = data.ndim == 3 and data.shape[2] > 1
        channels = data.shape[2] if is_multi else 1

        if not is_multi:
            gray = data if data.ndim == 2 else data[:, :, 0]
            self._add_stats(row, gray, prefix)
            if report_noise or do_adjusted:
                floor = self._compute_noise_floor(gray, noise_pct, meas_bg_pct, row, prefix)
                if do_adjusted:
                    self._add_adjusted_stats(row, gray, floor, prefix)
        elif mode == "Per-Channel":
            for ch_idx in range(channels):
                if ch_idx < len(self._CHANNEL_PARAMS):
                    if not self.get_parameter(self._CHANNEL_PARAMS[ch_idx]):
                        continue
                ch_name = self._CHANNEL_NAMES[ch_idx] if ch_idx < len(self._CHANNEL_NAMES) else f"Ch{ch_idx}"
                ch_prefix = f"{prefix}{ch_name}_"
                ch_data = data[:, :, ch_idx]
                self._add_stats(row, ch_data, ch_prefix)
                if report_noise or do_adjusted:
                    floor = self._compute_noise_floor(ch_data, noise_pct, meas_bg_pct, row, ch_prefix)
                    if do_adjusted:
                        self._add_adjusted_stats(row, ch_data, floor, ch_prefix)
        else:
            # Luminance
            if channels >= 3:
                lum = 0.2126 * data[:, :, 0] + 0.7152 * data[:, :, 1] + 0.0722 * data[:, :, 2]
            else:
                lum = data[:, :, 0]
            self._add_stats(row, lum, prefix)
            if report_noise or do_adjusted:
                floor = self._compute_noise_floor(lum, noise_pct, meas_bg_pct, row, prefix)
                if do_adjusted:
                    self._add_adjusted_stats(row, lum, floor, prefix)

        table = TableData()
        table.add_row(row)
        return table

    def _add_stats(self, row: dict, channel_data: np.ndarray, prefix: str) -> None:
        """Append selected statistics to *row* with the given column prefix."""
        if self.get_parameter("measure_mean"):
            row[f"{prefix}mean"] = float(np.mean(channel_data))
        if self.get_parameter("measure_min"):
            row[f"{prefix}min"] = float(np.min(channel_data))
        if self.get_parameter("measure_max"):
            row[f"{prefix}max"] = float(np.max(channel_data))
        if self.get_parameter("measure_std"):
            row[f"{prefix}std"] = float(np.std(channel_data))
        if self.get_parameter("measure_median"):
            row[f"{prefix}median"] = float(np.median(channel_data))
        if self.get_parameter("measure_sum"):
            row[f"{prefix}sum"] = float(np.sum(channel_data))

    @staticmethod
    def _compute_noise_floor(
        channel_data: np.ndarray,
        noise_pct: float,
        meas_bg_pct: float,
        row: dict,
        prefix: str,
    ) -> float:
        """Compute noise floor stats and add them to *row*.

        Returns the measurement_floor value for use by adjusted stats.
        """
        x = channel_data.ravel().astype(np.float64)

        # Background sample from lower-intensity region
        cutoff = float(np.percentile(x, noise_pct))
        bg = x[x <= cutoff]
        if bg.size < 100:
            bg = x

        bg_median = float(np.median(bg))
        noise_sigma = _robust_sigma(bg.astype(np.float32))
        measurement_floor = float(np.percentile(x, meas_bg_pct))

        row[f"{prefix}noise_floor_sigma"] = noise_sigma
        row[f"{prefix}bg_median"] = bg_median
        row[f"{prefix}measurement_floor"] = measurement_floor

        return measurement_floor

    def _add_adjusted_stats(
        self, row: dict, channel_data: np.ndarray, floor: float, prefix: str
    ) -> None:
        """Append noise-floor-adjusted statistics to *row*.

        Each enabled stat gets a ``net_`` counterpart with the measurement
        floor subtracted (clipped to 0 for min/mean/median, left as-is
        for std which is unaffected by a constant shift).
        """
        adjusted = np.clip(channel_data - floor, 0, None)

        if self.get_parameter("measure_mean"):
            row[f"{prefix}net_mean"] = float(np.mean(adjusted))
        if self.get_parameter("measure_min"):
            row[f"{prefix}net_min"] = float(np.min(adjusted))
        if self.get_parameter("measure_max"):
            row[f"{prefix}net_max"] = float(np.max(adjusted))
        if self.get_parameter("measure_median"):
            row[f"{prefix}net_median"] = float(np.median(adjusted))
        if self.get_parameter("measure_sum"):
            row[f"{prefix}net_sum"] = float(np.sum(adjusted))

    # -- Validation ----------------------------------------------------

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
