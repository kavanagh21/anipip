"""Duplicate plugin — copy one input to multiple outputs."""

from typing import Callable

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port


class Duplicate(BasePlugin):
    """Copy a single input image to multiple output ports.

    Useful for branching the pipeline — e.g., send an image to an
    exporter and continue processing it downstream via a separate
    output.  All outputs receive the same image data (shallow copy
    of the container with copied metadata).

    Note: the pipeline DAG also supports fan-out natively (one output
    port can connect to multiple input ports), but this node makes
    branch points explicit and easier to read.
    """

    name = "Duplicate"
    category = "Converters"
    description = "Copy one image input to multiple outputs"
    help_text = (
        "Copies a single input image to up to 5 output ports. Use this to "
        "branch the pipeline \u2014 e.g., send an image to an exporter on one "
        "output and continue processing on another. All outputs receive "
        "the same image data."
    )
    icon = None

    accepted_image_types = {ImageType.SINGLE, ImageType.Z_STACK, ImageType.TIMELAPSE}

    ports: list[Port] = [
        InputPort("image_in", ImageContainer, label="Image In"),
        OutputPort("out_1", ImageContainer, label="Output 1"),
        OutputPort("out_2", ImageContainer, label="Output 2"),
        OutputPort("out_3", ImageContainer, label="Output 3"),
        OutputPort("out_4", ImageContainer, label="Output 4"),
        OutputPort("out_5", ImageContainer, label="Output 5"),
    ]

    parameters = []

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
            raise RuntimeError("Duplicate requires an input image")

        progress_callback(0.5)

        return {
            "out_1": image,
            "out_2": image,
            "out_3": image,
            "out_4": image,
            "out_5": image,
        }

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
