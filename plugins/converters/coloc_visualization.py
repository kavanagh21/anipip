"""Colocalization visualization plugin — additive colour blending of masks."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import BoolParameter, ChoiceParameter, FloatParameter


_COLOR_MAP = {
    "Red": (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
    "Cyan": (0.0, 1.0, 1.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Yellow": (1.0, 1.0, 0.0),
    "White": (1.0, 1.0, 1.0),
}

_COLOR_NAMES = list(_COLOR_MAP.keys())


class ColocVisualization(BasePlugin):
    """Visualise mask overlap as an additive-colour RGB image.

    Each connected mask is tinted with a pseudocolour and blended
    additively.  Where masks overlap, colours mix (e.g. red + green = yellow).
    An optional brightness boost highlights overlap regions.
    """

    name = "Coloc Visualization"
    category = "Converters"
    description = "Visualise mask overlap with additive colour blending"
    help_text = (
        "Creates an RGB image showing where binary masks overlap. Each mask "
        "is tinted with a configurable colour and blended additively "
        "(red + green = yellow where both are positive). Overlap regions "
        "can be brightened for emphasis. Accepts 2 or 3 mask inputs."
    )
    icon = None

    ports: list[Port] = [
        InputPort("mask1_in", ImageContainer, label="Mask 1"),
        InputPort("mask2_in", ImageContainer, label="Mask 2"),
        InputPort("mask3_in", ImageContainer, label="Mask 3", optional=True),
        OutputPort("image_out", ImageContainer, label="Coloc Image"),
    ]

    parameters = [
        ChoiceParameter(
            name="ch1_color",
            label="Mask 1 Colour",
            choices=_COLOR_NAMES,
            default="Red",
        ),
        ChoiceParameter(
            name="ch2_color",
            label="Mask 2 Colour",
            choices=_COLOR_NAMES,
            default="Green",
        ),
        ChoiceParameter(
            name="ch3_color",
            label="Mask 3 Colour",
            choices=_COLOR_NAMES,
            default="Blue",
        ),
        BoolParameter(
            name="highlight_overlap",
            label="Highlight Overlap",
            default=True,
        ),
        FloatParameter(
            name="overlap_boost",
            label="Overlap Boost",
            default=1.3,
            min_value=1.0,
            max_value=2.0,
            step=0.05,
            decimals=2,
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
        mask1_cont: ImageContainer = inputs.get("mask1_in")
        mask2_cont: ImageContainer = inputs.get("mask2_in")
        mask3_cont: ImageContainer = inputs.get("mask3_in")

        if mask1_cont is None or mask2_cont is None:
            raise RuntimeError("Coloc Visualization requires at least two mask inputs")

        progress_callback(0.1)

        masks = [
            (self._to_mask_2d(mask1_cont.data), self.get_parameter("ch1_color")),
            (self._to_mask_2d(mask2_cont.data), self.get_parameter("ch2_color")),
        ]
        if mask3_cont is not None:
            masks.append(
                (self._to_mask_2d(mask3_cont.data), self.get_parameter("ch3_color"))
            )

        highlight = self.get_parameter("highlight_overlap")
        boost = float(self.get_parameter("overlap_boost"))

        h, w = masks[0][0].shape
        img = np.zeros((h, w, 3), dtype=np.float32)
        overlap_count = np.zeros((h, w), dtype=np.int32)

        for idx, (mask, color_name) in enumerate(masks):
            if mask.shape != (h, w):
                raise RuntimeError(
                    f"Mask {idx + 1} shape {mask.shape} does not match first mask ({h}, {w})"
                )
            rgb = _COLOR_MAP.get(color_name, (1.0, 0.0, 0.0))
            m = (mask > 0).astype(np.float32)
            for c in range(3):
                img[:, :, c] += m * float(rgb[c])
            overlap_count += (mask > 0).astype(np.int32)
            progress_callback(0.1 + 0.6 * (idx + 1) / len(masks))

        if highlight and len(masks) >= 2:
            overlap = overlap_count >= 2
            img[overlap] *= boost

        result = np.clip(img, 0.0, 1.0)

        progress_callback(0.9)

        metadata = mask1_cont.metadata.copy()
        metadata.image_type = ImageType.SINGLE
        metadata.add_history("Coloc Visualization")

        progress_callback(1.0)
        return {"image_out": ImageContainer(data=result, metadata=metadata)}

    @staticmethod
    def _to_mask_2d(data: np.ndarray) -> np.ndarray:
        """Extract a 2D mask from arbitrary-shape data."""
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
