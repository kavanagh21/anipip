"""Image exporter plugin for saving output images."""

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import tifffile

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer
from core.parameters import (
    BoolParameter,
    FileParameter,
    ChoiceParameter,
    IntParameter,
    StringParameter,
)


class ImageExporter(BasePlugin):
    """Save images to various output formats.

    Point the output folder at a directory and images are saved using
    the original source filename (from metadata) with an optional
    prefix and/or suffix.  Alternatively, set a custom filename to
    replace the source name entirely.

    When *Save to Source Image Folder* is ticked the output is written
    alongside the original input files.
    """

    name = "Image Exporter"
    category = "Exporters"
    description = "Save images as TIFF or PNG files to a folder"
    help_text = (
        "Saves images to a chosen folder as TIFF or PNG. Enable \"Save to "
        "Source Image Folder\" to write output alongside the original files. "
        "Set a Custom Filename to replace the source name entirely, or use "
        "Prefix/Suffix to modify it (e.g. prefix=\"proj_\", suffix=\"_bg\" "
        "\u2192 \"proj_liver_C0_bg.tif\"). Downstream nodes will see the "
        "new filename in their metadata."
    )
    icon = None

    parameters = [
        BoolParameter(
            name="save_to_source_folder",
            label="Save to Source Image Folder",
            default=False,
        ),
        BoolParameter(
            name="use_output_subfolder",
            label="Save into 'output' subfolder",
            default=False,
        ),
        FileParameter(
            name="output_folder",
            label="Output Folder",
            default="",
            folder_mode=True,
        ),
        ChoiceParameter(
            name="format",
            label="Output Format",
            choices=["TIFF", "PNG"],
            default="TIFF",
        ),
        StringParameter(
            name="custom_filename",
            label="Custom Filename",
            default="",
            placeholder="e.g. composite (without extension)",
        ),
        StringParameter(
            name="prefix",
            label="Filename Prefix",
            default="",
            placeholder="e.g. processed_",
        ),
        StringParameter(
            name="suffix",
            label="Filename Suffix",
            default="",
            placeholder="e.g. _max",
        ),
        IntParameter(
            name="compression",
            label="Compression Level",
            default=5,
            min_value=0,
            max_value=9,
            step=1,
        ),
    ]

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        progress_callback(0.1)

        output_format = self.get_parameter("format")
        compression = self.get_parameter("compression")
        prefix = self.get_parameter("prefix") or ""
        suffix = self.get_parameter("suffix") or ""
        custom_name = (self.get_parameter("custom_filename") or "").strip()
        save_to_source = self.get_parameter("save_to_source_folder")

        # Determine output folder
        source_path = image.metadata.source_path
        output_folder = None
        if save_to_source:
            # Try image source path first, then pipeline-level fallback
            if source_path and source_path.parent.exists():
                output_folder = source_path.parent
            elif self._pipeline_source_folder and self._pipeline_source_folder.exists():
                output_folder = self._pipeline_source_folder
        if output_folder is None:
            output_folder = Path(self.get_parameter("output_folder"))

        if self.get_parameter("use_output_subfolder"):
            output_folder = output_folder / "output"

        output_folder.mkdir(parents=True, exist_ok=True)

        # Build filename
        if custom_name:
            # Custom filename replaces the source name entirely;
            # prefix/suffix are still applied around it.
            base_name = custom_name
        elif source_path:
            base_name = source_path.stem
        else:
            import time
            base_name = f"output_{int(time.time() * 1000)}"

        ext = ".tif" if output_format == "TIFF" else ".png"
        final_path = output_folder / f"{prefix}{base_name}{suffix}{ext}"

        progress_callback(0.3)

        # Normalise data to a saveable dtype
        data = self._normalise_data(image.data)

        # Export based on format
        if output_format == "TIFF":
            self._save_tiff(data, final_path, compression)
        else:
            self._save_png(data, final_path, compression)

        progress_callback(0.9)

        # Build result with updated metadata
        result = image.copy()
        result.metadata.add_history(f"Exported to {final_path.name}")
        # Update source_path so downstream nodes see the new filename
        result.metadata.source_path = final_path

        progress_callback(1.0)

        return result

    # ------------------------------------------------------------------
    # Data normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_data(data: np.ndarray) -> np.ndarray:
        """Ensure data has a standard dtype that file writers can handle."""
        if data.dtype in (np.uint8, np.uint16, np.float32, np.float64):
            return data

        # Convert other integer types to the nearest standard dtype
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            if info.max <= 255:
                return data.astype(np.uint8)
            return data.astype(np.uint16)

        if np.issubdtype(data.dtype, np.floating):
            return data.astype(np.float32)

        # Boolean or other exotic types
        return data.astype(np.uint8)

    # ------------------------------------------------------------------
    # TIFF
    # ------------------------------------------------------------------

    @staticmethod
    def _save_tiff(data: np.ndarray, path: Path, compression: int) -> None:
        if path.suffix.lower() not in ('.tif', '.tiff'):
            path = path.with_suffix('.tif')

        compress = compression if compression > 0 else None

        tifffile.imwrite(
            str(path),
            data,
            compression='zlib' if compress else None,
            compressionargs={'level': compress} if compress else None,
        )

    # ------------------------------------------------------------------
    # PNG
    # ------------------------------------------------------------------

    @staticmethod
    def _save_png(data: np.ndarray, path: Path, compression: int) -> None:
        if path.suffix.lower() != '.png':
            path = path.with_suffix('.png')

        # PNG supports: uint8 (all modes) and uint16 (grayscale only).
        # Multi-channel uint16 or float must be converted.
        is_multichannel = data.ndim == 3 and data.shape[2] > 1

        if data.dtype in (np.float32, np.float64):
            if is_multichannel:
                # Float RGB/RGBA → uint8
                data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
            else:
                # Float grayscale → uint16 for precision
                data = (np.clip(data, 0, 1) * 65535).astype(np.uint16)
        elif data.dtype == np.uint16 and is_multichannel:
            # PIL cannot save uint16 RGB — convert to uint8
            data = (data / 256).astype(np.uint8)
        elif data.dtype != np.uint8 and data.dtype != np.uint16:
            data = data.astype(np.uint8)

        # Squeeze single-channel trailing dim
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]

        # Determine PIL mode
        if data.ndim == 2:
            mode = 'I;16' if data.dtype == np.uint16 else 'L'
        elif data.shape[2] == 3:
            mode = 'RGB'
        elif data.shape[2] == 4:
            mode = 'RGBA'
        else:
            raise ValueError(f"Unsupported channel count for PNG: {data.shape[2]}")

        data = np.ascontiguousarray(data)

        if mode == 'I;16':
            img = Image.fromarray(data.astype(np.uint16), mode='I;16')
        else:
            img = Image.fromarray(data, mode=mode)

        img.save(str(path), compress_level=compression)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""

    def validate_parameters(self) -> tuple[bool, list[str]]:
        errors = []
        save_to_source = self.get_parameter("save_to_source_folder")
        output_folder = self.get_parameter("output_folder")

        if not save_to_source:
            if not output_folder:
                errors.append(
                    "No output folder specified "
                    "(or enable 'Save to Source Image Folder')"
                )
            else:
                path = Path(output_folder)
                if path.exists() and not path.is_dir():
                    errors.append(f"Output path is not a folder: {path}")

        return len(errors) == 0, errors
