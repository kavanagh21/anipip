"""Pipeline execution progress view."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
)


class ProgressView(QWidget):
    """Widget displaying pipeline execution progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._batch_mode = False
        self._total_files = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        # Main progress row
        main_row = QHBoxLayout()

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(140)
        main_row.addWidget(self.status_label)

        # Current node/file label
        self.node_label = QLabel("")
        self.node_label.setStyleSheet("color: #666;")
        main_row.addWidget(self.node_label)

        # Progress bar (for nodes or overall batch)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_row.addWidget(self.progress_bar, stretch=1)

        # Node progress (within current node)
        self.node_progress_label = QLabel("")
        self.node_progress_label.setMinimumWidth(80)
        self.node_progress_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_row.addWidget(self.node_progress_label)

        layout.addLayout(main_row)

        # Batch progress row (hidden by default)
        self.batch_row = QWidget()
        batch_layout = QHBoxLayout(self.batch_row)
        batch_layout.setContentsMargins(0, 0, 0, 0)

        self.file_label = QLabel("")
        self.file_label.setStyleSheet("color: #666; font-size: 11px;")
        batch_layout.addWidget(self.file_label)

        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setMinimum(0)
        self.batch_progress_bar.setMaximum(100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setTextVisible(True)
        self.batch_progress_bar.setMaximumHeight(16)
        batch_layout.addWidget(self.batch_progress_bar, stretch=1)

        self.batch_row.setVisible(False)
        layout.addWidget(self.batch_row)

    def set_batch_mode(self, enabled: bool, total_files: int = 0) -> None:
        """Enable or disable batch mode display."""
        self._batch_mode = enabled
        self._total_files = total_files
        self.batch_row.setVisible(enabled)

        if enabled:
            self.batch_progress_bar.setFormat(f"0/{total_files} files")
            self.batch_progress_bar.setValue(0)

    def set_status(self, status: str) -> None:
        """Set the status text."""
        self.status_label.setText(status)

    def set_current_node(self, node_name: str, node_index: int, total_nodes: int) -> None:
        """Set the current processing node information."""
        self.node_label.setText(f"Processing: {node_name}")
        overall_progress = int((node_index / total_nodes) * 100)
        self.progress_bar.setValue(overall_progress)

    def set_node_progress(self, progress: float) -> None:
        """Set the progress within the current node (0.0-1.0)."""
        self.node_progress_label.setText(f"{int(progress * 100)}%")

    def set_overall_progress(self, current: int, total: int) -> None:
        """Set overall progress as node count."""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{current}/{total} nodes")

    def set_batch_progress(
        self,
        file_idx: int, total_files: int,
        node_idx: int, total_nodes: int,
        progress: float, filename: str
    ) -> None:
        """Set batch processing progress."""
        # Update file progress
        if total_files > 0:
            file_progress = int(((file_idx + (node_idx + progress) / total_nodes) / total_files) * 100)
            self.batch_progress_bar.setValue(file_progress)
            self.batch_progress_bar.setFormat(f"{file_idx + 1}/{total_files} files")

        # Update current file label
        self.file_label.setText(f"File: {filename}")

        # Update node progress
        if total_nodes > 0:
            node_progress = int(((node_idx + progress) / total_nodes) * 100)
            self.progress_bar.setValue(node_progress)
            self.progress_bar.setFormat(f"{node_idx + 1}/{total_nodes} nodes")

        self.node_progress_label.setText(f"{int(progress * 100)}%")

    def reset(self) -> None:
        """Reset the progress view to initial state."""
        self.status_label.setText("Ready")
        self.node_label.setText("")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.node_progress_label.setText("")
        self.file_label.setText("")
        self.batch_progress_bar.setValue(0)
        self._batch_mode = False
        self.batch_row.setVisible(False)

    def set_complete(self, success: bool = True) -> None:
        """Mark execution as complete."""
        if success:
            self.status_label.setText("Complete")
            self.progress_bar.setValue(100)
            if self._batch_mode:
                self.batch_progress_bar.setValue(100)
        else:
            self.status_label.setText("Failed")
        self.node_label.setText("")
        self.node_progress_label.setText("")

    def set_error(self, message: str) -> None:
        """Display an error message."""
        self.status_label.setText("Error")
        self.node_label.setText(message)
        self.node_progress_label.setText("")
