"""Mask intensity measurement plugin — intensity stats inside a mask."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.table_data import TableData
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import StringParameter


class MaskIntensityMeasurement(BasePlugin):
    """Measure intensity statistics inside a binary mask.

    Produces the input image unchanged plus a TableData row containing
    total/mean intensity, in-mask intensity, and coverage percentage.
    """

    name = "Mask Intensity Measurement"
    category = "Measurements"
    description = "Measure total/mean intensity inside mask and coverage %"
    help_text = (
        "Measures intensity statistics inside a binary mask. Reports total and "
        "mean intensity for the whole image and within the mask, plus coverage "
        "percentage and pixel counts. Requires both an image and a mask input."
    )
    icon = None

    ports: list[Port] = [
        InputPort("image_in", ImageContainer, label="Image"),
        InputPort("mask_in", ImageContainer, label="Mask"),
        OutputPort("image_out", ImageContainer, label="Image Out"),
        OutputPort("measurements", TableData, label="Measurements"),
    ]

    parameters = [
        StringParameter(
            name="column_prefix",
            label="Column Prefix",
            default="",
            placeholder="e.g. 'ch1_'",
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
        mask_cont: ImageContainer = inputs.get("mask_in")
        if image is None or mask_cont is None:
            raise RuntimeError("Mask Intensity Measurement requires both image and mask inputs")

        progress_callback(0.1)

        prefix = self.get_parameter("column_prefix") or ""
        img_2d = self._to_2d(image.data, image.image_type)
        mask_2d = self._to_mask_2d(mask_cont.data)

        # Clean NaN values
        img_clean = np.nan_to_num(img_2d, nan=0.0, posinf=0.0, neginf=0.0)
        m = mask_2d > 0
        pos = int(np.sum(m))
        tot = int(mask_2d.size)

        row: dict = {}
        source = image.metadata.source_path
        row["filename"] = str(source.name) if source else ""

        row[f"{prefix}mean_intensity"] = float(np.nanmean(img_clean)) if img_clean.size > 0 else 0.0
        row[f"{prefix}mean_intensity_in_mask"] = float(np.nanmean(img_clean[m])) if pos > 0 else 0.0
        row[f"{prefix}total_intensity"] = float(np.nansum(img_clean))
        row[f"{prefix}total_intensity_in_mask"] = float(np.nansum(img_clean[m])) if pos > 0 else 0.0
        row[f"{prefix}coverage_percent"] = (float(pos) / float(tot) * 100.0) if tot > 0 else 0.0
        row[f"{prefix}positive_pixels"] = pos
        row[f"{prefix}total_pixels"] = tot

        progress_callback(0.8)

        table = TableData()
        table.add_row(row)

        progress_callback(1.0)
        return {"image_out": image, "measurements": table}

    @staticmethod
    def _to_2d(data, image_type):
        if data.ndim == 2:
            return data.astype(np.float32)
        if image_type != ImageType.SINGLE:
            proj = np.max(data, axis=0)
            if proj.ndim == 3:
                return np.mean(proj[..., :3].astype(np.float32), axis=-1)
            return proj.astype(np.float32)
        if data.ndim == 3 and data.shape[-1] in (3, 4):
            return np.mean(data[..., :3].astype(np.float32), axis=-1)
        return data.astype(np.float32)

    @staticmethod
    def _to_mask_2d(data):
        if data.ndim == 2:
            return data
        if data.ndim == 3 and data.shape[-1] in (3, 4):
            return data[..., 0]
        if data.ndim == 3:
            return np.max(data, axis=0)
        return data

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
