"""Channel overlay plugin — merge up to 5 grayscale channels into an RGB composite."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import ChoiceParameter, FloatParameter


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

_DEFAULTS = ["Red", "Green", "Blue", "Cyan", "Magenta"]


class ChannelOverlay(BasePlugin):
    """Merge up to 5 single-channel images into a false-colour RGB composite.

    Each connected input is tinted with the chosen colour and weighted by its
    slider value.  Unconnected ports are silently skipped, so the node works
    with anywhere from 1 to 5 inputs.
    """

    name = "Channel Overlay"
    category = "Converters"
    description = "Merge channels into a false-colour RGB composite"
    help_text = (
        "Combines up to 5 grayscale images into a single RGB composite. "
        "Each channel is tinted with a configurable pseudocolour (Red, Green, "
        "Blue, Cyan, Magenta, Yellow, White) and its contribution is scaled by "
        "a weight slider. Unconnected inputs are ignored. Useful for creating "
        "multichannel fluorescence overlays."
    )
    icon = None

    ports: list[Port] = [
        InputPort("ch1_in", ImageContainer, label="Channel 1", optional=True),
        InputPort("ch2_in", ImageContainer, label="Channel 2", optional=True),
        InputPort("ch3_in", ImageContainer, label="Channel 3", optional=True),
        InputPort("ch4_in", ImageContainer, label="Channel 4", optional=True),
        InputPort("ch5_in", ImageContainer, label="Channel 5", optional=True),
        OutputPort("image_out", ImageContainer, label="Composite"),
    ]

    parameters = []
    for _i in range(1, 6):
        parameters.append(
            ChoiceParameter(
                name=f"ch{_i}_color",
                label=f"Ch {_i} Colour",
                choices=_COLOR_NAMES,
                default=_DEFAULTS[_i - 1],
            )
        )
        parameters.append(
            FloatParameter(
                name=f"ch{_i}_weight",
                label=f"Ch {_i} Weight",
                default=1.0,
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                decimals=2,
            )
        )

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
        # Collect connected channels
        channels: list[tuple[int, ImageContainer]] = []
        for i in range(1, 6):
            container = inputs.get(f"ch{i}_in")
            if container is not None:
                channels.append((i, container))

        if not channels:
            raise RuntimeError("Channel Overlay requires at least one connected input")

        progress_callback(0.1)

        first_data = channels[0][1].data
        is_stack = first_data.ndim == 3 and first_data.shape[-1] not in (3, 4)

        if is_stack:
            result = self._process_stack(channels, progress_callback)
        else:
            result = self._process_single(channels, progress_callback)

        progress_callback(0.9)

        metadata = channels[0][1].metadata.copy()
        metadata.image_type = ImageType.SINGLE if not is_stack else ImageType.Z_STACK
        ch_names = ", ".join(
            f"Ch{i}={self.get_parameter(f'ch{i}_color')}" for i, _ in channels
        )
        metadata.add_history(f"Channel Overlay ({ch_names})")

        progress_callback(1.0)
        return {"image_out": ImageContainer(data=result, metadata=metadata)}

    def _process_single(self, channels, progress_callback):
        """Composite a single 2D frame."""
        first = self._to_gray(channels[0][1].data)
        h, w = first.shape
        comp = np.zeros((h, w, 3), dtype=np.float32)

        for idx, (i, container) in enumerate(channels):
            gray = self._to_gray(container.data)
            if gray.shape != (h, w):
                raise RuntimeError(
                    f"Channel {i} shape {gray.shape} does not match first channel ({h}, {w})"
                )
            color = _COLOR_MAP[self.get_parameter(f"ch{i}_color")]
            weight = float(self.get_parameter(f"ch{i}_weight"))
            for c in range(3):
                comp[:, :, c] += gray * float(color[c]) * weight
            progress_callback(0.1 + 0.7 * (idx + 1) / len(channels))

        return np.clip(comp, 0.0, 1.0)

    def _process_stack(self, channels, progress_callback):
        """Composite a Z-stack slice by slice."""
        first_data = channels[0][1].data
        n_slices = first_data.shape[0]
        h, w = first_data.shape[1], first_data.shape[2]
        result_slices = []

        for z in range(n_slices):
            comp = np.zeros((h, w, 3), dtype=np.float32)
            for i, container in channels:
                stack = container.data
                if stack.ndim == 2:
                    gray = self._normalize(stack.astype(np.float32))
                elif stack.ndim == 3:
                    gray = self._normalize(stack[z].astype(np.float32))
                else:
                    gray = self._normalize(
                        np.mean(stack[z, ..., :3].astype(np.float32), axis=-1)
                    )
                color = _COLOR_MAP[self.get_parameter(f"ch{i}_color")]
                weight = float(self.get_parameter(f"ch{i}_weight"))
                for c in range(3):
                    comp[:, :, c] += gray * float(color[c]) * weight
            result_slices.append(np.clip(comp, 0.0, 1.0))
            progress_callback(0.1 + 0.7 * (z + 1) / n_slices)

        return np.stack(result_slices, axis=0)

    @staticmethod
    def _to_gray(data: np.ndarray) -> np.ndarray:
        """Convert image data to normalised 2D float32 grayscale."""
        if data.ndim == 2:
            arr = data.astype(np.float32)
        elif data.ndim == 3 and data.shape[-1] in (3, 4):
            arr = np.mean(data[..., :3].astype(np.float32), axis=-1)
        elif data.ndim == 3:
            # Z-stack — take max projection for single-frame context
            arr = np.max(data.astype(np.float32), axis=0)
        elif data.ndim == 4:
            arr = np.max(np.mean(data[..., :3].astype(np.float32), axis=-1), axis=0)
        else:
            arr = data.astype(np.float32)
        return ChannelOverlay._normalize(arr)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        pmax = float(np.max(arr))
        if pmax > 1.0:
            return arr / pmax
        if pmax > 0:
            return arr / pmax
        return arr

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
