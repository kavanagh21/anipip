"""Intensity measurement plugin — computes per-image statistics."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer
from core.table_data import TableData
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import BoolParameter, ChoiceParameter, StringParameter


class IntensityMeasurement(BasePlugin):
    """Compute intensity statistics for each image.

    Produces two outputs:
    * **image_out** — the input image passed through unchanged
    * **measurements** — a :class:`TableData` containing one row of statistics

    For multi-channel images the plugin can measure individual channels
    or compute a single luminance (perceptual brightness) value.
    Grayscale images are measured directly regardless of mode.
    """

    name = "Intensity Measurement"
    category = "Measurements"
    description = "Compute mean, min, max, std, median, and sum intensity statistics"
    help_text = (
        "Computes per-image intensity statistics (mean, min, max, std, median, "
        "sum) and outputs them as a table row. For colour images, choose "
        "Per-Channel to measure each channel separately, or Luminance for a "
        "single perceptual brightness value. The image passes through unchanged."
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

        row: dict = {}

        # Always include filename
        source = image.metadata.source_path
        row["filename"] = str(source.name) if source else ""

        is_multi = data.ndim == 3 and data.shape[2] > 1
        channels = data.shape[2] if is_multi else 1

        if not is_multi:
            # Grayscale — measure the single channel directly
            gray = data if data.ndim == 2 else data[:, :, 0]
            self._add_stats(row, gray, prefix)
        elif mode == "Per-Channel":
            for ch_idx in range(channels):
                if ch_idx < len(self._CHANNEL_PARAMS):
                    if not self.get_parameter(self._CHANNEL_PARAMS[ch_idx]):
                        continue
                ch_name = self._CHANNEL_NAMES[ch_idx] if ch_idx < len(self._CHANNEL_NAMES) else f"Ch{ch_idx}"
                ch_data = data[:, :, ch_idx]
                self._add_stats(row, ch_data, f"{prefix}{ch_name}_")
        else:
            # Luminance — perceptual brightness from RGB
            if channels >= 3:
                lum = 0.2126 * data[:, :, 0] + 0.7152 * data[:, :, 1] + 0.0722 * data[:, :, 2]
            else:
                lum = data[:, :, 0]
            self._add_stats(row, lum, prefix)

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

    # -- Validation ----------------------------------------------------

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
