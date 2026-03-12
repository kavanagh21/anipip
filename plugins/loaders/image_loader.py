"""Image loader plugin for loading various image formats."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import tifffile

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageMetadata, ImageType, normalize_tiff_axes
from core.parameters import FileParameter


class ImageLoader(BasePlugin):
    """Load images from PNG, JPG, JPEG, TIF, and TIFF files."""

    name = "Image Loader"
    category = "Loaders"
    description = "Load images from common formats (PNG, JPG, TIFF)"
    help_text = (
        "Loads a single image file. Supports PNG, JPG, BMP, and TIFF formats. "
        "Multi-page TIFFs are automatically detected as Z-stacks and axis "
        "metadata is used to reorder dimensions correctly."
    )
    icon = None

    parameters = [
        FileParameter(
            name="file_path",
            label="Image File",
            default="",
            filter="Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*.*)",
            save_mode=False,
        ),
    ]

    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    def process(
        self,
        image: Optional[ImageContainer],
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        """Load an image from file.

        Args:
            image: Ignored for loader (can be None)
            progress_callback: Progress callback function

        Returns:
            ImageContainer with loaded image data
        """
        progress_callback(0.1)

        file_path = Path(self.get_parameter("file_path"))

        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}")

        progress_callback(0.2)

        # Load image based on format
        num_pages = 1
        if suffix in ('.tif', '.tiff'):
            data, num_pages = self._load_tiff(file_path)
        else:
            data = self._load_pil(file_path)

        progress_callback(0.8)

        # Create metadata
        metadata = self._create_metadata(file_path, data, num_pages)

        progress_callback(1.0)

        return ImageContainer(data=data, metadata=metadata)

    def _load_tiff(self, path: Path) -> tuple[np.ndarray, int]:
        """Load TIFF file using tifffile for scientific format support.

        Uses axis metadata from tifffile to reorder the array to our
        convention (stack, Y, X, channels-last).

        Returns:
            Tuple of (data array, number of pages).
        """
        with tifffile.TiffFile(str(path)) as tif:
            num_pages = len(tif.pages)
            series = tif.series[0] if tif.series else None
            if series is not None:
                data = series.asarray()
                data = normalize_tiff_axes(data, series.axes)
            else:
                data = tifffile.imread(str(path))
        return data, num_pages

    def _load_pil(self, path: Path) -> np.ndarray:
        """Load image using PIL."""
        with Image.open(path) as img:
            # Convert to RGB/RGBA if needed
            if img.mode == 'L':
                # Grayscale
                data = np.array(img)
            elif img.mode == 'LA':
                # Grayscale with alpha - convert to grayscale
                data = np.array(img.convert('L'))
            elif img.mode == 'P':
                # Palette/indexed - raw index values (ignore LUT)
                data = np.array(img)
            elif img.mode == 'RGBA':
                data = np.array(img)
            else:
                # Convert to RGB
                data = np.array(img.convert('RGB'))

        return data

    def _create_metadata(
        self, path: Path, data: np.ndarray, num_pages: int = 1
    ) -> ImageMetadata:
        """Create metadata from loaded image.

        Data is expected to already be axis-normalised (stack, Y, X,
        channels-last) so shape interpretation is straightforward.

        Args:
            path: Source file path
            data: Normalised image array
            num_pages: Number of pages in the TIFF (1 for non-TIFF)
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

        # Determine bit depth
        if data.dtype == np.uint8:
            bit_depth = 8
        elif data.dtype == np.uint16:
            bit_depth = 16
        elif data.dtype in (np.float32, np.float64):
            bit_depth = 32
        else:
            bit_depth = 8

        # Try to get DPI from image
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
        file_path = self.get_parameter("file_path")

        if not file_path:
            errors.append("No file path specified")
        elif not Path(file_path).exists():
            errors.append(f"File not found: {file_path}")
        elif Path(file_path).suffix.lower() not in self.SUPPORTED_FORMATS:
            errors.append(f"Unsupported format: {Path(file_path).suffix}")

        return len(errors) == 0, errors
