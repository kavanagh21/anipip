"""Multi-channel folder loader — scan a folder and route files to channel outputs by pattern."""

from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tifffile

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageMetadata, ImageType, normalize_tiff_axes
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import ChoiceParameter, FileParameter, StringParameter


class MultichannelFolderLoader(BasePlugin):
    """Scan a folder and load images into separate channel outputs by filename pattern.

    Each of the five output ports has a corresponding filename pattern
    (standard wildcards: ``*``, ``?``).  The loader globs the folder for
    each non-empty pattern and loads the first matching file as a
    Z-stack (or timelapse / single image depending on the setting).

    For example, with a folder containing::

        liver_XY17_C0.tif
        liver_XY17_C1.tif
        liver_XY17_C2.tif

    Set the patterns to ``*_C0.tif``, ``*_C1.tif``, ``*_C2.tif`` on
    channels 1–3 and leave channels 4–5 empty.
    """

    name = "Multichannel Folder Loader"
    category = "Loaders"
    description = "Load multiple channel files from a folder by filename pattern"
    help_text = (
        "Scans a folder and loads one file per channel output based on a "
        "wildcard pattern (e.g. \"*_C0*.tif\"). Up to 5 channel outputs are "
        "available — leave a pattern empty to disable that output. Each "
        "matched file is loaded as a Z-stack, timelapse, or single image "
        "depending on the \"Interpret As\" setting. Case-insensitive matching."
    )
    icon = None

    ports: list[Port] = [
        OutputPort("ch1_out", ImageContainer, label="Channel 1"),
        OutputPort("ch2_out", ImageContainer, label="Channel 2"),
        OutputPort("ch3_out", ImageContainer, label="Channel 3"),
        OutputPort("ch4_out", ImageContainer, label="Channel 4"),
        OutputPort("ch5_out", ImageContainer, label="Channel 5"),
    ]

    parameters = [
        FileParameter(
            name="folder_path",
            label="Image Folder",
            default="",
            folder_mode=True,
        ),
        ChoiceParameter(
            name="image_type",
            label="Interpret As",
            choices=["Z-Stack", "Timelapse", "Auto"],
            default="Z-Stack",
        ),
        StringParameter(
            name="ch1_pattern",
            label="Channel 1 Pattern",
            default="*_C0*.tif",
            placeholder="e.g. *_C0*.tif",
        ),
        StringParameter(
            name="ch2_pattern",
            label="Channel 2 Pattern",
            default="*_C1*.tif",
            placeholder="e.g. *_C1*.tif",
        ),
        StringParameter(
            name="ch3_pattern",
            label="Channel 3 Pattern",
            default="*_C2*.tif",
            placeholder="e.g. *_C2*.tif",
        ),
        StringParameter(
            name="ch4_pattern",
            label="Channel 4 Pattern",
            default="",
            placeholder="e.g. *_C3*.tif",
        ),
        StringParameter(
            name="ch5_pattern",
            label="Channel 5 Pattern",
            default="",
            placeholder="e.g. *_C4*.tif",
        ),
    ]

    def process(
        self,
        image: Optional[ImageContainer],
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        """Legacy single-output fallback — loads channel 1 only."""
        results = self.process_ports({}, progress_callback)
        for val in results.values():
            if isinstance(val, ImageContainer):
                return val
        raise RuntimeError("No channel patterns matched any files")

    def process_ports(
        self,
        inputs: dict[str, PipelineData],
        progress_callback: Callable[[float], None],
    ) -> dict[str, PipelineData]:
        folder = Path(self.get_parameter("folder_path"))
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        progress_callback(0.05)

        # Collect active channels
        channel_specs: list[tuple[str, str]] = []  # (port_name, pattern)
        for i in range(1, 6):
            pattern = (self.get_parameter(f"ch{i}_pattern") or "").strip()
            if pattern:
                channel_specs.append((f"ch{i}_out", pattern))

        if not channel_specs:
            raise RuntimeError("No channel patterns specified")

        # List folder contents once
        all_files = sorted(f for f in folder.iterdir() if f.is_file())

        results: dict[str, PipelineData] = {}
        total = len(channel_specs)

        for idx, (port_name, pattern) in enumerate(channel_specs):
            base_progress = 0.05 + 0.9 * idx / total

            # Match files against pattern (case-insensitive)
            matched = [
                f for f in all_files
                if fnmatch(f.name.lower(), pattern.lower())
            ]

            if not matched:
                raise FileNotFoundError(
                    f"No file matching '{pattern}' found in {folder}"
                )

            # Use the first match (sorted, so deterministic)
            file_path = matched[0]

            def ch_progress(p: float, bp=base_progress, span=0.9 / total):
                progress_callback(bp + p * span)

            container = self._load_file(file_path, ch_progress)
            results[port_name] = container

        progress_callback(1.0)
        return results

    def _load_file(
        self,
        file_path: Path,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        """Load a single image file as an ImageContainer."""
        progress_callback(0.1)

        suffix = file_path.suffix.lower()

        if suffix in (".tif", ".tiff"):
            with tifffile.TiffFile(str(file_path)) as tif:
                num_pages = len(tif.pages)
                series = tif.series[0] if tif.series else None
                if series is not None:
                    data = series.asarray()
                    data = normalize_tiff_axes(data, series.axes)
                else:
                    data = tifffile.imread(str(file_path))
        else:
            from PIL import Image
            with Image.open(file_path) as img:
                if img.mode == "L":
                    data = np.array(img)
                elif img.mode in ("LA", "Pa"):
                    data = np.array(img.convert("L"))
                elif img.mode == "RGBA":
                    data = np.array(img)
                elif img.mode == "P":
                    data = np.array(img)
                else:
                    data = np.array(img.convert("RGB"))
            num_pages = 1

        progress_callback(0.7)

        # Determine image type
        type_choice = self.get_parameter("image_type")
        if type_choice == "Auto":
            if num_pages > 1 and data.ndim >= 3:
                image_type = ImageType.Z_STACK
            else:
                image_type = ImageType.SINGLE
        elif num_pages <= 1 or data.ndim < 3:
            image_type = ImageType.SINGLE
        elif type_choice == "Timelapse":
            image_type = ImageType.TIMELAPSE
        else:
            image_type = ImageType.Z_STACK

        metadata = self._create_metadata(file_path, data, image_type)

        progress_callback(1.0)
        return ImageContainer(data=data, metadata=metadata)

    @staticmethod
    def _create_metadata(
        path: Path, data: np.ndarray, image_type: ImageType
    ) -> ImageMetadata:
        """Create metadata from loaded image data."""
        suffix = path.suffix.lower().lstrip(".")

        if image_type == ImageType.SINGLE:
            num_slices = 1
            if data.ndim == 2:
                h, w = data.shape
                color_space = "grayscale"
            elif data.ndim == 3 and data.shape[-1] in (1, 2, 3, 4):
                h, w, c = data.shape
                color_space = {1: "grayscale", 2: "multichannel", 3: "rgb", 4: "rgba"}[c]
            else:
                h, w = data.shape[-2], data.shape[-1]
                color_space = "grayscale"
        else:
            if data.ndim == 3:
                num_slices, h, w = data.shape
                color_space = "grayscale"
            elif data.ndim == 4 and data.shape[-1] in (1, 2, 3, 4):
                num_slices, h, w, c = data.shape
                color_space = {1: "grayscale", 2: "multichannel", 3: "rgb", 4: "rgba"}[c]
            else:
                num_slices = data.shape[0]
                h, w = data.shape[-2], data.shape[-1]
                color_space = "grayscale"

        if data.dtype == np.uint8:
            bit_depth = 8
        elif data.dtype == np.uint16:
            bit_depth = 16
        elif data.dtype in (np.float32, np.float64):
            bit_depth = 32
        else:
            bit_depth = 8

        return ImageMetadata(
            source_path=path,
            original_format=suffix,
            bit_depth=bit_depth,
            color_space=color_space,
            dimensions=(w, h),
            image_type=image_type,
            num_slices=num_slices,
            processing_history=[f"Loaded from {path.name}"],
        )

    def validate_parameters(self) -> tuple[bool, list[str]]:
        errors = []
        folder_path = self.get_parameter("folder_path")

        if not folder_path:
            errors.append("No folder path specified")
        elif not Path(folder_path).exists():
            errors.append(f"Folder not found: {folder_path}")
        elif not Path(folder_path).is_dir():
            errors.append(f"Path is not a folder: {folder_path}")

        # Check at least one pattern is specified
        has_pattern = False
        for i in range(1, 6):
            pattern = (self.get_parameter(f"ch{i}_pattern") or "").strip()
            if pattern:
                has_pattern = True
                break

        if not has_pattern:
            errors.append("At least one channel pattern must be specified")

        return len(errors) == 0, errors
