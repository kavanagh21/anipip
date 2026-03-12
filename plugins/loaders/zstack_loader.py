"""Z-Stack / Timelapse loader plugin for multi-page TIFF files."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tifffile

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageMetadata, ImageType, normalize_tiff_axes
from core.parameters import FileParameter, ChoiceParameter


class ZStackLoader(BasePlugin):
    """Load a multi-page TIFF as a z-stack or timelapse volume."""

    name = "Z-Stack Loader"
    category = "Loaders"
    description = "Load a multi-page TIFF file as a z-stack or timelapse"
    help_text = (
        "Loads a multi-page TIFF as a Z-stack or timelapse volume. Use "
        "\"Interpret As\" to control whether pages are treated as Z-slices "
        "or time frames. Single-page TIFFs are loaded as regular 2D images."
    )
    icon = None

    parameters = [
        FileParameter(
            name="file_path",
            label="TIFF File",
            default="",
            filter="TIFF Files (*.tif *.tiff);;All Files (*.*)",
            save_mode=False,
        ),
        ChoiceParameter(
            name="image_type",
            label="Interpret As",
            choices=["Z-Stack", "Timelapse"],
            default="Z-Stack",
        ),
    ]

    def process(
        self,
        image: Optional[ImageContainer],
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        progress_callback(0.1)

        file_path = Path(self.get_parameter("file_path"))

        if not file_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in ('.tif', '.tiff'):
            raise ValueError(f"Unsupported format: {suffix}. Only TIFF files are supported.")

        progress_callback(0.2)

        # Load and normalise axes in a single open
        with tifffile.TiffFile(str(file_path)) as tif:
            num_pages = len(tif.pages)
            series = tif.series[0] if tif.series else None
            if series is not None:
                data = series.asarray()
                data = normalize_tiff_axes(data, series.axes)
            else:
                data = tifffile.imread(str(file_path))

        progress_callback(0.6)

        # Determine image type
        type_choice = self.get_parameter("image_type")
        if num_pages <= 1 or data.ndim < 3:
            image_type = ImageType.SINGLE
        elif type_choice == "Timelapse":
            image_type = ImageType.TIMELAPSE
        else:
            image_type = ImageType.Z_STACK

        progress_callback(0.8)

        metadata = self._create_metadata(file_path, data, image_type)

        progress_callback(1.0)

        return ImageContainer(data=data, metadata=metadata)

    def _create_metadata(
        self, path: Path, data: np.ndarray, image_type: ImageType
    ) -> ImageMetadata:
        """Create metadata from loaded stack data.

        Data is expected to already be axis-normalised.
        """
        suffix = path.suffix.lower().lstrip('.')

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
            # Stack: (Z, H, W) or (Z, H, W, C)
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

        # Bit depth
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
        file_path = self.get_parameter("file_path")

        if not file_path:
            errors.append("No file path specified")
        elif not Path(file_path).exists():
            errors.append(f"File not found: {file_path}")
        elif Path(file_path).suffix.lower() not in ('.tif', '.tiff'):
            errors.append(f"Only TIFF files are supported: {Path(file_path).suffix}")

        return len(errors) == 0, errors
