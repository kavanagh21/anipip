"""Batch image loader plugin for loading multiple images from a folder."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import tifffile

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageMetadata, ImageType, normalize_tiff_axes
from core.parameters import FileParameter, StringParameter, BoolParameter


class BatchImageLoader(BasePlugin):
    """Load multiple images from a folder for batch processing."""

    name = "Batch Image Loader"
    category = "Loaders"
    description = "Load all images from a folder for batch processing"
    help_text = (
        "Loads all matching images from a folder for batch processing. The file "
        "filter supports semicolon-separated glob patterns (e.g. \"GFP_*.tif;*.png\"). "
        "Enable Include Subfolders to search recursively."
    )
    icon = None

    parameters = [
        FileParameter(
            name="folder_path",
            label="Image Folder",
            default="",
            folder_mode=True,
        ),
        StringParameter(
            name="file_filter",
            label="File Filter",
            default="*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp",
            placeholder="e.g. GFP_*.tif;*.png",
        ),
        BoolParameter(
            name="recursive",
            label="Include Subfolders",
            default=False,
        ),
    ]

    # Flag to indicate this is a batch source
    is_batch_source = True

    def _parse_filter(self) -> list[str]:
        """Parse the file filter string into a list of glob patterns."""
        raw = self.get_parameter("file_filter").strip()
        if not raw:
            return ["*"]
        return [p.strip() for p in raw.split(";") if p.strip()]

    def get_image_files(self) -> list[Path]:
        """Get list of image files to process.

        Patterns are semicolon-separated globs (e.g. ``GFP_*.tif;*.png``).
        For extension-only patterns like ``*.tif`` both lower and upper
        case variants are searched automatically.
        """
        folder = Path(self.get_parameter("folder_path"))
        recursive = self.get_parameter("recursive")
        patterns = self._parse_filter()

        glob_fn = folder.rglob if recursive else folder.glob

        files = []
        for pattern in patterns:
            files.extend(glob_fn(pattern))
            # For extension-only patterns (e.g. *.tif) also try upper-case
            # extension so we catch .TIF on case-sensitive filesystems.
            # Skip this for filename wildcards (e.g. GFP_*.tif) to avoid
            # mangling the name portion.
            if pattern.startswith("*."):
                files.extend(glob_fn(pattern.upper()))

        # Sort for consistent ordering, deduplicate
        return sorted(set(files))

    def process(
        self,
        image: Optional[ImageContainer],
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        """Load the first image (for single-image mode compatibility).

        In batch mode, the pipeline will call load_image() for each file.
        """
        files = self.get_image_files()
        if not files:
            raise FileNotFoundError("No images found in the specified folder")

        return self.load_image(files[0], progress_callback)

    def load_image(
        self,
        file_path: Path,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        """Load a single image from file.

        Args:
            file_path: Path to the image file
            progress_callback: Progress callback function

        Returns:
            ImageContainer with loaded image data
        """
        progress_callback(0.1)

        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        suffix = file_path.suffix.lower()

        progress_callback(0.2)

        # Load image based on format
        num_pages = 1
        if suffix in ('.tif', '.tiff'):
            with tifffile.TiffFile(str(file_path)) as tif:
                num_pages = len(tif.pages)
                series = tif.series[0] if tif.series else None
                if series is not None:
                    data = series.asarray()
                    data = normalize_tiff_axes(data, series.axes)
                else:
                    data = tifffile.imread(str(file_path))
        else:
            data = self._load_pil(file_path)

        progress_callback(0.8)

        # Create metadata
        metadata = self._create_metadata(file_path, data, num_pages)

        progress_callback(1.0)

        return ImageContainer(data=data, metadata=metadata)

    def _load_pil(self, path: Path) -> np.ndarray:
        """Load image using PIL."""
        with Image.open(path) as img:
            if img.mode == 'L':
                data = np.array(img)
            elif img.mode == 'LA':
                data = np.array(img.convert('L'))
            elif img.mode == 'P':
                # Palette/indexed - raw index values (ignore LUT)
                data = np.array(img)
            elif img.mode == 'RGBA':
                data = np.array(img)
            else:
                data = np.array(img.convert('RGB'))
        return data

    def _create_metadata(
        self, path: Path, data: np.ndarray, num_pages: int = 1
    ) -> ImageMetadata:
        """Create metadata from loaded image.

        Data is expected to already be axis-normalised.
        """
        suffix = path.suffix.lower().lstrip('.')

        # Detect multi-page TIFFs as z-stacks
        image_type = ImageType.SINGLE
        num_slices = 1

        if num_pages > 1 and data.ndim >= 3:
            image_type = ImageType.Z_STACK
            num_slices = data.shape[0]

        # Determine color space and spatial dimensions
        if image_type == ImageType.SINGLE:
            if data.ndim == 2:
                color_space = "grayscale"
                height, width = data.shape
            elif data.ndim == 3 and data.shape[-1] in (1, 2, 3, 4):
                height, width, c = data.shape
                color_space = {1: "grayscale", 2: "multichannel", 3: "rgb", 4: "rgba"}[c]
            else:
                height, width = data.shape[-2], data.shape[-1]
                color_space = "grayscale"
        else:
            # Stack: (Z, H, W) or (Z, H, W, C)
            if data.ndim == 3:
                _, height, width = data.shape
                color_space = "grayscale"
            elif data.ndim == 4 and data.shape[-1] in (1, 2, 3, 4):
                _, height, width, c = data.shape
                color_space = {1: "grayscale", 2: "multichannel", 3: "rgb", 4: "rgba"}[c]
            else:
                height, width = data.shape[-2], data.shape[-1]
                color_space = "grayscale"

        if data.dtype == np.uint8:
            bit_depth = 8
        elif data.dtype == np.uint16:
            bit_depth = 16
        elif data.dtype in (np.float32, np.float64):
            bit_depth = 32
        else:
            bit_depth = 8

        dpi = None
        try:
            with Image.open(path) as img:
                if 'dpi' in img.info:
                    dpi = tuple(int(d) for d in img.info['dpi'])
        except Exception:
            pass

        return ImageMetadata(
            source_path=path,
            original_format=suffix,
            bit_depth=bit_depth,
            color_space=color_space,
            dimensions=(width, height),
            image_type=image_type,
            num_slices=num_slices,
            dpi=dpi,
            processing_history=[f"Loaded from {path.name}"],
        )

    def validate_parameters(self) -> tuple[bool, list[str]]:
        """Validate loader parameters."""
        errors = []
        folder_path = self.get_parameter("folder_path")

        if not folder_path:
            errors.append("No folder path specified")
        elif not Path(folder_path).exists():
            errors.append(f"Folder not found: {folder_path}")
        elif not Path(folder_path).is_dir():
            errors.append(f"Path is not a folder: {folder_path}")

        return len(errors) == 0, errors
