"""ImageContainer data class for internal image representation."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .pipeline_data import PipelineData


class ImageType(Enum):
    """Type of image data stored in an ImageContainer."""

    SINGLE = "single"
    Z_STACK = "z_stack"
    TIMELAPSE = "timelapse"


def normalize_tiff_axes(data: np.ndarray, axes: str) -> np.ndarray:
    """Reorder a tifffile array to our axis convention.

    Target layout:
    - SINGLE:    ``(H, W)`` or ``(H, W, C)``
    - Z_STACK:   ``(Z, H, W)`` or ``(Z, H, W, C)``
    - TIMELAPSE: ``(T, H, W)`` or ``(T, H, W, C)``

    The function reads the *axes* string produced by
    ``tifffile.TiffFile.series[0].axes`` (characters such as ``Z``, ``T``,
    ``C``, ``S``, ``Y``, ``X``, ``I``, ``Q``) and transposes the array so
    that stack axes come first, then Y, X, then channels last.

    Multiple stack axes (e.g. ``TZCYX``) are flattened into a single
    leading dimension.  A length-1 channel axis is squeezed away.
    """
    axes = axes.upper()

    stack_idx: list[int] = []
    spatial_idx: list[int] = []
    channel_idx: list[int] = []

    for i, ax in enumerate(axes):
        if ax in ("Z", "T", "I", "Q"):
            stack_idx.append(i)
        elif ax in ("Y", "X"):
            spatial_idx.append(i)
        elif ax in ("C", "S"):
            channel_idx.append(i)
        # Unrecognised axes (rare) — treat as extra stack axes
        else:
            stack_idx.append(i)

    if not spatial_idx:
        # Cannot determine spatial layout — return unchanged
        return data

    # Target order: stack..., Y, X, channels...
    order = stack_idx + spatial_idx + channel_idx
    if order != list(range(len(axes))):
        data = np.transpose(data, order)

    # Flatten multiple leading stack axes into one
    n_stack = len(stack_idx)
    if n_stack > 1:
        combined = 1
        for i in range(n_stack):
            combined *= data.shape[i]
        data = data.reshape((combined,) + data.shape[n_stack:])

    # Squeeze length-1 trailing channel dim
    if channel_idx and data.shape[-1] == 1:
        data = data.squeeze(axis=-1)

    return data


@dataclass
class ImageMetadata:
    """Metadata associated with an image."""

    source_path: Optional[Path] = None
    original_format: str = ""
    bit_depth: int = 8
    color_space: str = "rgb"  # 'grayscale', 'rgb', 'rgba'
    dimensions: tuple[int, int] = (0, 0)  # (width, height)
    image_type: ImageType = ImageType.SINGLE
    num_slices: int = 1
    dpi: Optional[tuple[int, int]] = None
    custom: dict[str, Any] = field(default_factory=dict)
    processing_history: list[str] = field(default_factory=list)

    def add_history(self, operation: str) -> None:
        """Add an operation to the processing history."""
        self.processing_history.append(operation)

    def copy(self) -> "ImageMetadata":
        """Create a copy of the metadata."""
        return ImageMetadata(
            source_path=self.source_path,
            original_format=self.original_format,
            bit_depth=self.bit_depth,
            color_space=self.color_space,
            dimensions=self.dimensions,
            image_type=self.image_type,
            num_slices=self.num_slices,
            dpi=self.dpi,
            custom=self.custom.copy(),
            processing_history=self.processing_history.copy(),
        )


@dataclass
class ImageContainer(PipelineData):
    """Container for image data flowing through the pipeline.

    Attributes:
        data: Image data as NumPy array with shape (H, W, C) or (H, W) for grayscale
        metadata: Associated metadata about the image
    """

    data: np.ndarray
    metadata: ImageMetadata = field(default_factory=ImageMetadata)

    def __post_init__(self):
        """Validate and update metadata based on actual data."""
        if self.data is not None:
            self._update_metadata_from_data()

    def _update_metadata_from_data(self) -> None:
        """Update metadata dimensions and color space from actual data."""
        if self.metadata.image_type == ImageType.SINGLE:
            # SINGLE: (H, W) grayscale or (H, W, C) color
            if self.data.ndim == 2:
                h, w = self.data.shape
                self.metadata.color_space = "grayscale"
            elif self.data.ndim == 3:
                h, w, c = self.data.shape
                if c == 1:
                    self.metadata.color_space = "grayscale"
                elif c == 2:
                    self.metadata.color_space = "multichannel"
                elif c == 3:
                    self.metadata.color_space = "rgb"
                elif c == 4:
                    self.metadata.color_space = "rgba"
            else:
                h, w = 0, 0
            self.metadata.dimensions = (w, h)
        else:
            # Z_STACK / TIMELAPSE: (Z, H, W) grayscale or (Z, H, W, C) color
            if self.data.ndim == 3:
                z, h, w = self.data.shape
                self.metadata.color_space = "grayscale"
            elif self.data.ndim == 4 and self.data.shape[-1] in (1, 2, 3, 4):
                z, h, w, c = self.data.shape
                if c == 1:
                    self.metadata.color_space = "grayscale"
                elif c == 2:
                    self.metadata.color_space = "multichannel"
                elif c == 3:
                    self.metadata.color_space = "rgb"
                elif c == 4:
                    self.metadata.color_space = "rgba"
            elif self.data.ndim == 4:
                # Last dim too large to be channels — treat as extra
                # stack dim: use last two dims as spatial.
                h, w = self.data.shape[-2], self.data.shape[-1]
                self.metadata.color_space = "grayscale"
            else:
                z, h, w = 0, 0, 0
            self.metadata.num_slices = self.data.shape[0] if self.data.ndim >= 3 else 1
            self.metadata.dimensions = (w, h)

        # Determine bit depth from dtype
        if self.data.dtype == np.uint8:
            self.metadata.bit_depth = 8
        elif self.data.dtype == np.uint16:
            self.metadata.bit_depth = 16
        elif self.data.dtype in (np.float32, np.float64):
            self.metadata.bit_depth = 32

    @property
    def width(self) -> int:
        """Get image width."""
        return self.metadata.dimensions[0]

    @property
    def height(self) -> int:
        """Get image height."""
        return self.metadata.dimensions[1]

    @property
    def is_grayscale(self) -> bool:
        """Check if image is grayscale."""
        return self.metadata.color_space == "grayscale"

    @property
    def image_type(self) -> ImageType:
        """Get the image type (SINGLE, Z_STACK, TIMELAPSE)."""
        return self.metadata.image_type

    @property
    def num_slices(self) -> int:
        """Get the number of slices (1 for SINGLE images)."""
        return self.metadata.num_slices

    def copy(self) -> "ImageContainer":
        """Create a deep copy of the container."""
        return ImageContainer(
            data=self.data.copy(),
            metadata=self.metadata.copy(),
        )
