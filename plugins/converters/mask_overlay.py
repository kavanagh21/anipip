"""Mask overlay plugin — overlay coloured mask edges onto an image."""

from typing import Callable

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import ChoiceParameter, FloatParameter, IntParameter


class MaskOverlay(BasePlugin):
    """Overlay coloured mask edges onto a grayscale image.

    Accepts an image and a binary mask and produces an RGB image with
    the mask outline (and optional fill) drawn on top.
    """

    name = "Mask Overlay"
    category = "Converters"
    description = "Overlay coloured mask edges onto an image"
    help_text = (
        "Draws coloured mask edges onto a grayscale image, producing an RGB "
        "result. Accepts separate image and mask inputs via ports. Configure "
        "edge colour, width, and optional semi-transparent fill. Useful for "
        "visualising segmentation results overlaid on the original image."
    )
    icon = None

    _COLOR_MAP = {
        "Red": (1.0, 0.0, 0.0),
        "Green": (0.0, 1.0, 0.0),
        "Blue": (0.0, 0.0, 1.0),
        "Yellow": (1.0, 1.0, 0.0),
        "Cyan": (0.0, 1.0, 1.0),
        "Magenta": (1.0, 0.0, 1.0),
        "White": (1.0, 1.0, 1.0),
    }

    ports: list[Port] = [
        InputPort("image_in", ImageContainer, label="Image"),
        InputPort("mask_in", ImageContainer, label="Mask"),
        OutputPort("image_out", ImageContainer, label="Overlay"),
    ]

    parameters = [
        ChoiceParameter(
            name="edge_color",
            label="Edge Color",
            choices=["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"],
            default="Red",
        ),
        IntParameter(
            name="edge_width",
            label="Edge Width",
            default=1,
            min_value=1,
            max_value=10,
        ),
        FloatParameter(
            name="fill_alpha",
            label="Fill Alpha",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
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
        image: ImageContainer = inputs.get("image_in")
        mask_cont: ImageContainer = inputs.get("mask_in")
        if image is None or mask_cont is None:
            raise RuntimeError("Mask Overlay requires both image and mask inputs")

        progress_callback(0.1)

        color_name = self.get_parameter("edge_color")
        color = self._COLOR_MAP.get(color_name, (1.0, 0.0, 0.0))
        width = int(self.get_parameter("edge_width"))
        alpha = float(self.get_parameter("fill_alpha"))

        gray = self._to_norm_gray(image.data)
        mask_2d = self._to_mask_2d(mask_cont.data)

        progress_callback(0.3)

        # Build RGB base from grayscale
        overlay = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

        # Compute mask edges via morphological gradient
        m = mask_2d > 0
        if width <= 1:
            edges = binary_dilation(m) & ~m
        else:
            edges = binary_dilation(m, iterations=width) & ~binary_erosion(m, iterations=max(1, width - 1))

        # Optional fill
        if alpha > 0:
            fill = m & ~edges
            for c in range(3):
                overlay[fill, c] = overlay[fill, c] * (1 - alpha) + color[c] * alpha * 0.5

        # Draw edges
        for c in range(3):
            overlay[edges, c] = color[c]

        overlay = np.clip(overlay, 0, 1)

        progress_callback(0.9)

        metadata = image.metadata.copy()
        metadata.image_type = ImageType.SINGLE
        metadata.add_history(f"Mask Overlay ({color_name})")

        progress_callback(1.0)
        return {"image_out": ImageContainer(data=overlay, metadata=metadata)}

    @staticmethod
    def _to_norm_gray(data):
        if data.ndim == 2:
            arr = data.astype(np.float32)
        elif data.ndim == 3 and data.shape[-1] in (3, 4):
            arr = np.mean(data[..., :3].astype(np.float32), axis=-1)
        elif data.ndim == 3:
            arr = np.max(data.astype(np.float32), axis=0)
        elif data.ndim == 4:
            arr = np.max(np.mean(data[..., :3].astype(np.float32), axis=-1), axis=0)
        else:
            arr = data.astype(np.float32)
        pmax = float(np.max(arr))
        return arr / pmax if pmax > 0 else arr

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
