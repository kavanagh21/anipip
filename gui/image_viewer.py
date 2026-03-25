"""Image viewer with zoom, pan, and pixel inspection."""

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QPushButton,
    QComboBox,
    QFrame,
    QSlider,
    QSpinBox,
    QSplitter,
)

from core.image_container import ImageContainer, ImageType


class ImageDisplay(QLabel):
    """Image display widget with zoom and pan support."""

    mouse_moved = pyqtSignal(int, int)  # Emitted with (x, y) image coordinates
    zoom_changed = pyqtSignal(float)  # Emitted when zoom level changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        self._pan_start: Optional[QPointF] = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setMinimumSize(100, 100)

    def set_image(self, pixmap: QPixmap) -> None:
        """Set the image to display."""
        self._pixmap = pixmap
        self._update_display()

    def clear_image(self) -> None:
        """Clear the displayed image."""
        self._pixmap = None
        self.clear()

    @property
    def zoom(self) -> float:
        """Get current zoom level."""
        return self._zoom

    @zoom.setter
    def zoom(self, value: float) -> None:
        """Set zoom level."""
        self._zoom = max(self._min_zoom, min(self._max_zoom, value))
        self._update_display()
        self.zoom_changed.emit(self._zoom)

    def fit_to_window(self) -> None:
        """Fit the image to the window size."""
        if self._pixmap is None:
            return

        parent = self.parent()
        if parent:
            parent_size = parent.size()
            scale_x = parent_size.width() / self._pixmap.width()
            scale_y = parent_size.height() / self._pixmap.height()
            self.zoom = min(scale_x, scale_y) * 0.95

    def reset_zoom(self) -> None:
        """Reset zoom to 100%."""
        self.zoom = 1.0

    def _update_display(self) -> None:
        """Update the displayed image with current zoom."""
        if self._pixmap is None:
            return

        scaled = self._pixmap.scaled(
            int(self._pixmap.width() * self._zoom),
            int(self._pixmap.height() * self._zoom),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse movement for coordinate display."""
        if self._pixmap is None:
            return

        # Convert widget coordinates to image coordinates
        pos = event.position()
        pixmap = self.pixmap()

        if pixmap:
            # Calculate image position within label
            label_rect = self.rect()
            pixmap_rect = pixmap.rect()

            offset_x = (label_rect.width() - pixmap_rect.width()) / 2
            offset_y = (label_rect.height() - pixmap_rect.height()) / 2

            img_x = int((pos.x() - offset_x) / self._zoom)
            img_y = int((pos.y() - offset_y) / self._zoom)

            if 0 <= img_x < self._pixmap.width() and 0 <= img_y < self._pixmap.height():
                self.mouse_moved.emit(img_x, img_y)


class ImageViewer(QWidget):
    """Complete image viewer with controls and pixel inspection."""

    preview_file_changed = pyqtSignal(object)  # Emitted with Path when batch file selected

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_data: Optional[np.ndarray] = None
        self._stack_container: Optional[ImageContainer] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        toolbar.addWidget(self.zoom_label)

        fit_btn = QPushButton("Fit")
        fit_btn.setMaximumWidth(40)
        fit_btn.clicked.connect(self._fit_to_window)
        toolbar.addWidget(fit_btn)

        reset_btn = QPushButton("1:1")
        reset_btn.setMaximumWidth(40)
        reset_btn.clicked.connect(self._reset_zoom)
        toolbar.addWidget(reset_btn)

        # Batch file selector (hidden by default)
        self._batch_combo = QComboBox()
        self._batch_combo.setMinimumWidth(150)
        self._batch_combo.setMaximumWidth(300)
        self._batch_combo.hide()
        self._batch_combo.currentIndexChanged.connect(self._on_batch_file_selected)
        toolbar.addWidget(self._batch_combo)

        # Slice navigator (hidden by default, shown for stacks)
        self._slice_label = QLabel("Slice:")
        self._slice_label.hide()
        toolbar.addWidget(self._slice_label)

        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.setMinimumWidth(100)
        self._slice_slider.setMaximumWidth(250)
        self._slice_slider.setMinimum(0)
        self._slice_slider.hide()
        self._slice_slider.valueChanged.connect(self._on_slice_changed)
        toolbar.addWidget(self._slice_slider)

        self._slice_spinbox = QSpinBox()
        self._slice_spinbox.setMinimumWidth(55)
        self._slice_spinbox.setMaximumWidth(70)
        self._slice_spinbox.hide()
        self._slice_spinbox.valueChanged.connect(self._on_slice_spinbox_changed)
        toolbar.addWidget(self._slice_spinbox)

        self._slice_total_label = QLabel("")
        self._slice_total_label.hide()
        toolbar.addWidget(self._slice_total_label)

        toolbar.addStretch()

        # Coordinate display
        self.coord_label = QLabel("X: - Y: -")
        toolbar.addWidget(self.coord_label)

        # Pixel value display
        self.pixel_label = QLabel("Value: -")
        self.pixel_label.setMinimumWidth(100)
        toolbar.addWidget(self.pixel_label)

        layout.addLayout(toolbar)

        # Image display in scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setFrameShape(QFrame.Shape.StyledPanel)

        self.image_display = ImageDisplay()
        self.image_display.mouse_moved.connect(self._on_mouse_moved)
        self.image_display.zoom_changed.connect(self._on_zoom_changed)
        self.scroll_area.setWidget(self.image_display)

        layout.addWidget(self.scroll_area)

        # Info bar
        self.info_label = QLabel("No image loaded")
        self.info_label.setStyleSheet("color: #666;")
        layout.addWidget(self.info_label)

    def set_image(self, container: ImageContainer) -> None:
        """Set the image to display from an ImageContainer."""
        if container is None or container.data is None:
            self.clear()
            return

        is_stack = (
            container.image_type != ImageType.SINGLE
            and container.data.ndim >= 3
        )

        if is_stack:
            self._stack_container = container
            num_slices = container.data.shape[0]

            # Configure the slice slider/spinbox
            self._slice_slider.blockSignals(True)
            self._slice_spinbox.blockSignals(True)

            self._slice_slider.setMaximum(num_slices - 1)
            self._slice_spinbox.setMinimum(1)
            self._slice_spinbox.setMaximum(num_slices)
            self._slice_total_label.setText(f"/ {num_slices}")

            mid = num_slices // 2
            self._slice_slider.setValue(mid)
            self._slice_spinbox.setValue(mid + 1)

            self._slice_slider.blockSignals(False)
            self._slice_spinbox.blockSignals(False)

            self._slice_label.show()
            self._slice_slider.show()
            self._slice_spinbox.show()
            self._slice_total_label.show()

            # Display the middle slice
            self._display_slice(mid)
        else:
            self._stack_container = None
            self._hide_slice_controls()

            self._image_data = container.data
            pixmap = self._array_to_pixmap(container.data)
            self.image_display.set_image(pixmap)

            meta = container.metadata
            info = (
                f"{meta.dimensions[0]}x{meta.dimensions[1]} | "
                f"{meta.bit_depth}-bit | {meta.color_space}"
            )
            self.info_label.setText(info)

    def set_array(self, array: np.ndarray) -> None:
        """Set the image directly from a NumPy array."""
        if array is None:
            self.clear()
            return

        self._image_data = array
        pixmap = self._array_to_pixmap(array)
        self.image_display.set_image(pixmap)

        # Basic info
        if array.ndim == 2:
            h, w = array.shape
            info = f"{w}x{h} | grayscale"
        else:
            h, w, c = array.shape
            info = f"{w}x{h} | {c} channels"
        self.info_label.setText(info)

    def _array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """Convert a NumPy array to QPixmap."""
        # Collapse extra leading dimensions (e.g. a stray batch dim)
        while array.ndim > 3:
            array = array[0]

        # Normalize to 8-bit for display
        if array.dtype == np.uint16:
            array = (array / 256).astype(np.uint8)
        elif array.dtype in (np.float32, np.float64):
            array = (array * 255).clip(0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = array.astype(np.uint8)

        # Ensure contiguous array
        array = np.ascontiguousarray(array)

        if array.ndim == 2:
            # Grayscale
            h, w = array.shape
            qimage = QImage(array.data, w, h, w, QImage.Format.Format_Grayscale8)
        elif array.ndim == 3 and array.shape[2] == 3:
            # RGB
            h, w, _ = array.shape
            qimage = QImage(array.data, w, h, w * 3, QImage.Format.Format_RGB888)
        elif array.ndim == 3 and array.shape[2] == 4:
            # RGBA
            h, w, _ = array.shape
            qimage = QImage(array.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        elif array.ndim == 3 and array.shape[2] in (1, 2):
            # Single/dual channel — take first channel as grayscale
            array = np.ascontiguousarray(array[:, :, 0])
            h, w = array.shape
            qimage = QImage(array.data, w, h, w, QImage.Format.Format_Grayscale8)
        elif array.ndim == 3:
            # 3D but last dim is too large to be channels (e.g. spatial) —
            # treat as grayscale by taking max projection or first slice
            array = np.ascontiguousarray(array[:, :, 0])
            h, w = array.shape
            qimage = QImage(array.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # 1D or 0D — shouldn't happen, but guard against it
            raise ValueError(f"Cannot display array with shape {array.shape}")

        return QPixmap.fromImage(qimage)

    def clear(self) -> None:
        """Clear the displayed image."""
        self._image_data = None
        self._stack_container = None
        self._hide_slice_controls()
        self.image_display.clear_image()
        self.info_label.setText("No image loaded")
        self.coord_label.setText("X: - Y: -")
        self.pixel_label.setText("Value: -")

    def set_batch_files(self, files: list[Path]) -> None:
        """Populate the batch file selector combo box.

        Args:
            files: List of file paths. If empty, hides the combo.
        """
        self._batch_combo.blockSignals(True)
        self._batch_combo.clear()
        if files:
            for f in files:
                self._batch_combo.addItem(f.name, f)
            self._batch_combo.show()
        else:
            self._batch_combo.hide()
        self._batch_combo.blockSignals(False)

    def _on_batch_file_selected(self, index: int) -> None:
        """Handle batch file combo selection change."""
        if index < 0:
            return
        path = self._batch_combo.itemData(index)
        if path is not None:
            self.preview_file_changed.emit(path)

    def _fit_to_window(self) -> None:
        """Fit image to window."""
        self.image_display.fit_to_window()

    def _reset_zoom(self) -> None:
        """Reset zoom to 100%."""
        self.image_display.reset_zoom()

    def _on_zoom_changed(self, zoom: float) -> None:
        """Handle zoom changes."""
        self.zoom_label.setText(f"{int(zoom * 100)}%")

    def _hide_slice_controls(self) -> None:
        """Hide all slice navigation widgets."""
        self._slice_label.hide()
        self._slice_slider.hide()
        self._slice_spinbox.hide()
        self._slice_total_label.hide()

    def _display_slice(self, index: int) -> None:
        """Extract and display a single slice from the current stack."""
        if self._stack_container is None:
            return

        display_data = self._stack_container.data[index]
        self._image_data = display_data
        pixmap = self._array_to_pixmap(display_data)
        self.image_display.set_image(pixmap)

        # Update info bar
        meta = self._stack_container.metadata
        num_slices = self._stack_container.data.shape[0]
        info = (
            f"{meta.dimensions[0]}x{meta.dimensions[1]} | "
            f"{meta.bit_depth}-bit | {meta.color_space}"
            f" | Slice {index + 1}/{num_slices} | {meta.image_type.value}"
        )
        self.info_label.setText(info)

    def _on_slice_changed(self, value: int) -> None:
        """Handle slice slider movement."""
        self._slice_spinbox.blockSignals(True)
        self._slice_spinbox.setValue(value + 1)
        self._slice_spinbox.blockSignals(False)
        self._display_slice(value)

    def _on_slice_spinbox_changed(self, value: int) -> None:
        """Handle slice spinbox value change."""
        index = value - 1
        self._slice_slider.blockSignals(True)
        self._slice_slider.setValue(index)
        self._slice_slider.blockSignals(False)
        self._display_slice(index)

    def _on_mouse_moved(self, x: int, y: int) -> None:
        """Handle mouse movement over image."""
        self.coord_label.setText(f"X: {x} Y: {y}")

        if self._image_data is not None:
            try:
                if self._image_data.ndim == 2:
                    value = self._image_data[y, x]
                    self.pixel_label.setText(f"Value: {value}")
                else:
                    values = self._image_data[y, x]
                    if len(values) == 3:
                        self.pixel_label.setText(f"RGB: {tuple(values)}")
                    elif len(values) == 4:
                        self.pixel_label.setText(f"RGBA: {tuple(values)}")
                    else:
                        self.pixel_label.setText(f"Value: {values[0]}")
            except IndexError:
                self.pixel_label.setText("Value: -")
