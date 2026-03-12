"""Channel splitter plugin — split multichannel images into separate outputs."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import ChoiceParameter

_CHANNEL_CHOICES = [
    "Red / Ch1",
    "Green / Ch2",
    "Blue / Ch3",
    "Alpha / Ch4",
    "None",
]

_CHOICE_TO_INDEX = {
    "Red / Ch1": 0,
    "Green / Ch2": 1,
    "Blue / Ch3": 2,
    "Alpha / Ch4": 3,
}


class ChannelSplitter(BasePlugin):
    """Split a multichannel image into separate single-channel outputs.

    Each output port can be configured to extract a specific channel from
    the input image.  Supports SINGLE (H, W, C), Z_STACK (Z, H, W, C),
    and TIMELAPSE (T, H, W, C) layouts.  Grayscale inputs are passed
    through to all non-"None" outputs unchanged.
    """

    name = "Channel Splitter"
    category = "Converters"
    description = "Split a multichannel image into separate channel outputs"
    help_text = (
        "Splits a multichannel image (e.g. RGB) into up to three separate "
        "single-channel outputs. Each output port can be assigned to a "
        "specific channel. Set an output to \"None\" to disable it. Grayscale "
        "inputs are passed through to all active outputs unchanged."
    )
    icon = None

    accepted_image_types = {ImageType.SINGLE, ImageType.Z_STACK, ImageType.TIMELAPSE}

    ports: list[Port] = [
        InputPort("image_in", ImageContainer, label="Image"),
        OutputPort("output_1", ImageContainer, label="Output 1"),
        OutputPort("output_2", ImageContainer, label="Output 2"),
        OutputPort("output_3", ImageContainer, label="Output 3"),
    ]

    parameters = [
        ChoiceParameter(
            name="output_1_channel",
            label="Output 1 Channel",
            choices=_CHANNEL_CHOICES,
            default="Red / Ch1",
        ),
        ChoiceParameter(
            name="output_2_channel",
            label="Output 2 Channel",
            choices=_CHANNEL_CHOICES,
            default="Green / Ch2",
        ),
        ChoiceParameter(
            name="output_3_channel",
            label="Output 3 Channel",
            choices=_CHANNEL_CHOICES,
            default="Blue / Ch3",
        ),
    ]

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        # Legacy single-port fallback — not used for this multi-output plugin
        return image

    def process_ports(
        self,
        inputs: dict[str, PipelineData],
        progress_callback: Callable[[float], None],
    ) -> dict[str, PipelineData]:
        image: ImageContainer = inputs.get("image_in")
        if image is None:
            raise RuntimeError("Channel Splitter requires an input image")

        progress_callback(0.1)

        data = image.data
        is_grayscale = data.ndim == 2 or (
            data.ndim == 3 and image.metadata.image_type != ImageType.SINGLE
        )
        num_channels = 1 if is_grayscale else data.shape[-1]

        output_params = [
            ("output_1", self.get_parameter("output_1_channel")),
            ("output_2", self.get_parameter("output_2_channel")),
            ("output_3", self.get_parameter("output_3_channel")),
        ]

        results: dict[str, PipelineData] = {}
        for i, (port_name, choice) in enumerate(output_params):
            progress_callback(0.2 + 0.25 * i)

            if choice == "None":
                continue

            ch_idx = _CHOICE_TO_INDEX[choice]

            if is_grayscale:
                # All non-None outputs get the full 2D data
                channel_data = data.copy()
            elif ch_idx < num_channels:
                channel_data = data[..., ch_idx].copy()
            else:
                # Requested channel exceeds available channels — skip
                continue

            metadata = image.metadata.copy()
            metadata.color_space = "grayscale"
            metadata.add_history(f"Channel Splitter: extracted {choice}")

            results[port_name] = ImageContainer(data=channel_data, metadata=metadata)

        progress_callback(1.0)
        return results

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
