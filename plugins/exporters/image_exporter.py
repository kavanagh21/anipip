"""Image exporter plugin for saving output images."""

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import tifffile

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer
from core.parameters import FileParameter, ChoiceParameter, IntParameter, BoolParameter


class ImageExporter(BasePlugin):
    """Save images to various output formats.

    In batch mode, set output_path to a folder and enable batch_mode.
    Files will be saved with their original names (with new extension).
    """

    name = "Image Exporter"
    category = "Exporters"
    description = "Save images as TIFF or PNG files (supports batch output to folder)"
    help_text = (
        "Saves images to TIFF or PNG files. In batch mode, set the output "
        "path to a folder and files are saved with their original names. TIFF "
        "supports zlib compression. Float images are automatically converted "
        "to 16-bit for PNG output."
    )
    icon = None

    parameters = [
        FileParameter(
            name="output_path",
            label="Output Path",
            default="",
            filter="TIFF Files (*.tif *.tiff);;PNG Files (*.png);;All Files (*.*)",
            save_mode=True,
        ),
        ChoiceParameter(
            name="format",
            label="Output Format",
            choices=["TIFF", "PNG"],
            default="TIFF",
        ),
        IntParameter(
            name="compression",
            label="Compression Level",
            default=5,
            min_value=0,
            max_value=9,
            step=1,
        ),
        BoolParameter(
            name="batch_mode",
            label="Batch Mode (output to folder)",
            default=False,
        ),
    ]

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        """Export image to file.

        Args:
            image: Input image container
            progress_callback: Progress callback function

        Returns:
            The same image container (passthrough)
        """
        progress_callback(0.1)

        output_path = Path(self.get_parameter("output_path"))
        output_format = self.get_parameter("format")
        compression = self.get_parameter("compression")
        batch_mode = self.get_parameter("batch_mode")

        # Determine final output path
        if batch_mode:
            # Output to folder with original filename
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get source filename from metadata
            source_path = image.metadata.source_path
            if source_path:
                base_name = source_path.stem
            else:
                # Generate a name if no source
                import time
                base_name = f"output_{int(time.time() * 1000)}"

            # Add correct extension
            ext = ".tif" if output_format == "TIFF" else ".png"
            final_path = output_dir / f"{base_name}{ext}"
        else:
            # Single file output
            final_path = output_path
            final_path.parent.mkdir(parents=True, exist_ok=True)

        progress_callback(0.3)

        # Export based on format
        if output_format == "TIFF":
            self._save_tiff(image, final_path, compression)
        else:  # PNG
            self._save_png(image, final_path, compression)

        progress_callback(0.9)

        # Update metadata history
        result = image.copy()
        result.metadata.add_history(f"Exported to {final_path.name}")

        progress_callback(1.0)

        return result

    def _save_tiff(
        self, image: ImageContainer, path: Path, compression: int
    ) -> None:
        """Save image as TIFF."""
        # Ensure correct extension
        if path.suffix.lower() not in ('.tif', '.tiff'):
            path = path.with_suffix('.tif')

        data = image.data

        # Configure compression
        compress = compression if compression > 0 else None

        tifffile.imwrite(
            str(path),
            data,
            compression='zlib' if compress else None,
            compressionargs={'level': compress} if compress else None,
        )

    def _save_png(
        self, image: ImageContainer, path: Path, compression: int
    ) -> None:
        """Save image as PNG."""
        # Ensure correct extension
        if path.suffix.lower() != '.png':
            path = path.with_suffix('.png')

        data = image.data

        # PNG requires 8-bit or 16-bit
        if data.dtype == np.float32 or data.dtype == np.float64:
            # Convert float to 16-bit
            data = (np.clip(data, 0, 1) * 65535).astype(np.uint16)
        elif data.dtype == np.uint16:
            pass  # Keep as-is
        elif data.dtype != np.uint8:
            # Convert to 8-bit
            data = data.astype(np.uint8)

        # Handle different channel configurations
        if data.ndim == 2:
            mode = 'L' if data.dtype == np.uint8 else 'I;16'
        elif data.shape[2] == 1:
            data = data[:, :, 0]
            mode = 'L' if data.dtype == np.uint8 else 'I;16'
        elif data.shape[2] == 3:
            mode = 'RGB'
        elif data.shape[2] == 4:
            mode = 'RGBA'
        else:
            raise ValueError(f"Unsupported number of channels: {data.shape[2]}")

        # Create PIL image
        if mode == 'I;16':
            # PIL's 16-bit grayscale handling
            img = Image.fromarray(data.astype(np.uint16), mode='I;16')
        else:
            img = Image.fromarray(data, mode=mode)

        # Save with compression
        img.save(str(path), compress_level=compression)

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        """Validate input image."""
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""

    def validate_parameters(self) -> tuple[bool, list[str]]:
        """Validate exporter parameters."""
        errors = []
        output_path = self.get_parameter("output_path")
        batch_mode = self.get_parameter("batch_mode")

        if not output_path:
            errors.append("No output path specified")
        else:
            path = Path(output_path)
            if batch_mode:
                # For batch mode, path should be a directory or creatable
                if path.exists() and not path.is_dir():
                    errors.append(f"Batch mode requires a folder path, not a file: {path}")
            else:
                # For single file mode, check parent directory
                try:
                    parent = path.parent
                    if parent.exists() and not parent.is_dir():
                        errors.append(f"Parent path is not a directory: {parent}")
                except Exception as e:
                    errors.append(f"Invalid path: {e}")

        return len(errors) == 0, errors
