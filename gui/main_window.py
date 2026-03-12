"""Main application window."""

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QMenuBar,
    QMenu,
    QToolBar,
    QFileDialog,
    QMessageBox,
    QStatusBar,
)
from PyQt6.QtGui import QAction, QKeySequence

from core.pipeline import Pipeline
from core.plugin_registry import PluginRegistry
from core.image_container import ImageContainer
from core.pipeline_data import PipelineData
from core.table_data import TableData
from core.settings import PluginSettings
from .node_canvas import NodeCanvas
from .node_widget import NodeStatus
from .plugin_browser import PluginBrowser
from .plugin_defaults_dialog import PluginDefaultsDialog
from .properties_panel import PropertiesPanel
from .progress_view import ProgressView
from .image_viewer import ImageViewer
from .spreadsheet_viewer import SpreadsheetViewer
from .viewer_window import ViewerWindow


class PipelineWorker(QThread):
    """Worker thread for pipeline execution."""

    progress = pyqtSignal(int, int, float)  # node_index, total, node_progress
    node_started = pyqtSignal(str, int, int)  # node_id, index, total
    finished = pyqtSignal(object)  # ImageContainer or None
    error = pyqtSignal(str)

    def __init__(self, pipeline: Pipeline, input_image: Optional[ImageContainer] = None):
        super().__init__()
        self.pipeline = pipeline
        self.input_image = input_image
        self._stop_requested = False

    def run(self):
        """Execute the pipeline."""
        try:
            def progress_callback(node_index: int, total: int, node_progress: float):
                self.progress.emit(node_index, total, node_progress)

                # Emit node started when progress is 0
                if node_progress == 0 and node_index < len(self.pipeline.nodes):
                    node = self.pipeline.nodes[node_index]
                    self.node_started.emit(node.node_id, node_index, total)

            result = self.pipeline.execute(
                self.input_image,
                progress_callback,
                lambda: self._stop_requested,
            )
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """Request the pipeline to stop."""
        self._stop_requested = True


class BatchPipelineWorker(QThread):
    """Worker thread for batch pipeline execution."""

    # file_index, total_files, node_index, total_nodes, progress, filename
    progress = pyqtSignal(int, int, int, int, float, str)
    file_started = pyqtSignal(int, int, str)  # file_index, total_files, filename
    node_started = pyqtSignal(str, int, int)  # node_id, node_index, total_nodes
    file_finished = pyqtSignal(int, int, str)  # file_index, total_files, filename
    finished = pyqtSignal(list)  # List of ImageContainers
    error = pyqtSignal(str)

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._stop_requested = False
        self._current_file_index = 0
        self._total_files = 0

    def run(self):
        """Execute the batch pipeline."""
        try:
            files = self.pipeline.get_batch_files()
            self._total_files = len(files)

            def progress_callback(
                file_idx: int, total_files: int,
                node_idx: int, total_nodes: int,
                node_progress: float, filename: str
            ):
                self.progress.emit(file_idx, total_files, node_idx, total_nodes, node_progress, filename)

                # Emit file started when first node starts
                if node_idx == 0 and node_progress == 0:
                    self.file_started.emit(file_idx, total_files, filename)

                # Emit node started
                if node_progress == 0 and node_idx < len(self.pipeline.nodes):
                    node = self.pipeline.nodes[node_idx]
                    self.node_started.emit(node.node_id, node_idx, total_nodes)

            results = self.pipeline.execute_batch(
                progress_callback,
                lambda: self._stop_requested,
            )

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """Request the pipeline to stop."""
        self._stop_requested = True


class PreviewWorker(QThread):
    """Worker thread for partial pipeline execution (preview)."""

    finished = pyqtSignal(str, dict)  # node_id, port_results
    error = pyqtSignal(str)

    def __init__(self, pipeline: Pipeline, node_id: str, file_path=None,
                 changed_node_id: str = None):
        super().__init__()
        self.pipeline = pipeline
        self.node_id = node_id
        self.file_path = file_path
        self.changed_node_id = changed_node_id
        self._stop_requested = False

    def run(self):
        try:
            results = self.pipeline.preview_execute(
                self.node_id,
                self.file_path,
                lambda: self._stop_requested,
                changed_node_id=self.changed_node_id,
            )
            if not self._stop_requested:
                self.finished.emit(self.node_id, results)
        except Exception as e:
            if not self._stop_requested:
                self.error.emit(str(e))

    def stop(self):
        self._stop_requested = True


class MainWindow(QMainWindow):
    """Main application window with splitter layout."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analysis Pipeline")
        self.setMinimumSize(1200, 800)

        # Initialize core components
        self.pipeline = Pipeline()
        self.registry = PluginRegistry()
        self.settings = PluginSettings()
        self.registry.set_settings(self.settings)
        self._worker: Optional[PipelineWorker] = None
        self._batch_worker: Optional[BatchPipelineWorker] = None
        self._preview_node_id: Optional[str] = None
        self._preview_file: Optional[Path] = None
        self._preview_changed_node_id: Optional[str] = None
        self._preview_worker: Optional[PreviewWorker] = None
        self._preview_debounce_timer = QTimer()
        self._preview_debounce_timer.setSingleShot(True)
        self._preview_debounce_timer.setInterval(300)
        self._preview_debounce_timer.timeout.connect(self._run_preview)

        # Discover plugins
        plugins_dir = Path(__file__).parent.parent / "plugins"
        self.registry.discover_plugins(plugins_dir)

        # Setup UI
        self._setup_menus()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_status_bar()

        # Connect signals
        self._connect_signals()

    def _setup_menus(self) -> None:
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Pipeline", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_pipeline)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Pipeline...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_pipeline)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Pipeline", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_pipeline)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save Pipeline &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._save_pipeline_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        clear_action = QAction("&Clear Pipeline", self)
        clear_action.triggered.connect(self._clear_pipeline)
        edit_menu.addAction(clear_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        self.show_browser_action = QAction("Plugin &Browser", self)
        self.show_browser_action.setCheckable(True)
        self.show_browser_action.setChecked(True)
        self.show_browser_action.triggered.connect(self._toggle_browser)
        view_menu.addAction(self.show_browser_action)

        self.show_properties_action = QAction("&Properties Panel", self)
        self.show_properties_action.setCheckable(True)
        self.show_properties_action.setChecked(True)
        self.show_properties_action.triggered.connect(self._toggle_properties)
        view_menu.addAction(self.show_properties_action)

        self.show_viewer_action = QAction("Image &Viewer", self)
        self.show_viewer_action.setCheckable(True)
        self.show_viewer_action.setChecked(True)
        self.show_viewer_action.triggered.connect(self._toggle_viewer)
        view_menu.addAction(self.show_viewer_action)

        self.show_spreadsheet_action = QAction("&Spreadsheet Viewer", self)
        self.show_spreadsheet_action.setCheckable(True)
        self.show_spreadsheet_action.setChecked(True)
        self.show_spreadsheet_action.triggered.connect(self._toggle_spreadsheet)
        view_menu.addAction(self.show_spreadsheet_action)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")

        plugin_defaults_action = QAction("Plugin &Defaults...", self)
        plugin_defaults_action.triggered.connect(self._show_plugin_defaults)
        settings_menu.addAction(plugin_defaults_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self) -> None:
        """Setup the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.run_action = QAction("Run", self)
        self.run_action.setShortcut("F5")
        self.run_action.triggered.connect(self._run_pipeline)
        toolbar.addAction(self.run_action)

        self.stop_action = QAction("Stop", self)
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self._stop_pipeline)
        toolbar.addAction(self.stop_action)

        toolbar.addSeparator()

        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self._clear_pipeline)
        toolbar.addAction(clear_action)

    def _setup_central_widget(self) -> None:
        """Setup the central widget with splitter layout."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main splitter (horizontal)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)

        # Left: Plugin browser
        self.plugin_browser = PluginBrowser(self.registry)
        self.plugin_browser.setMinimumWidth(180)
        self.plugin_browser.setMaximumWidth(300)
        main_splitter.addWidget(self.plugin_browser)

        # Center: Node canvas (full height — viewers are separate windows)
        self.canvas = NodeCanvas(self.pipeline, self.registry)
        main_splitter.addWidget(self.canvas)

        # Right: Properties panel
        self.properties_panel = PropertiesPanel(self.pipeline)
        self.properties_panel.setMinimumWidth(200)
        self.properties_panel.setMaximumWidth(350)
        main_splitter.addWidget(self.properties_panel)

        main_splitter.setSizes([200, 700, 250])

        # Bottom: Progress view
        self.progress_view = ProgressView()
        layout.addWidget(self.progress_view)

        # Separate viewer windows
        self.image_viewer = ImageViewer()
        self.image_window = ViewerWindow(self.image_viewer, "Image Viewer", self)

        self.spreadsheet_viewer = SpreadsheetViewer()
        self.spreadsheet_window = ViewerWindow(
            self.spreadsheet_viewer, "Spreadsheet Viewer", self
        )

    def _setup_status_bar(self) -> None:
        """Setup the status bar."""
        self.statusBar().showMessage("Ready")

    def _connect_signals(self) -> None:
        """Connect component signals."""
        self.canvas.node_selected.connect(self._on_node_selected)
        self.canvas.node_preview_requested.connect(self._preview_node)
        self.canvas.pipeline_changed.connect(self._on_pipeline_changed)
        self.properties_panel.parameters_changed.connect(self._on_preview_param_changed)
        self.image_viewer.preview_file_changed.connect(self._on_preview_file_changed)

    def _on_node_selected(self, node_id: str) -> None:
        """Handle node selection."""
        self.properties_panel.set_node(node_id)

    def _on_pipeline_changed(self) -> None:
        """Handle pipeline structure changes."""
        self.statusBar().showMessage(f"Pipeline: {len(self.pipeline.nodes)} nodes")

    def _preview_node(self, node_id: str) -> None:
        """Preview the output of a specific node by executing up to it."""
        self._preview_node_id = node_id
        self._preview_file = None
        self._preview_changed_node_id = None  # Full run on explicit preview

        # Populate batch combo if this is a batch pipeline
        if self.pipeline.is_batch_pipeline():
            files = self.pipeline.get_batch_files()
            self.image_viewer.set_batch_files(files)
            if files:
                self._preview_file = files[0]
        else:
            self.image_viewer.set_batch_files([])

        self._run_preview()

    def _run_preview(self) -> None:
        """Start (or restart) the preview worker for the current preview node."""
        if self._preview_node_id is None:
            return

        # Cancel any in-flight preview
        if self._preview_worker is not None:
            self._preview_worker.stop()
            self._preview_worker.wait()
            self._preview_worker = None

        self.statusBar().showMessage("Preview: running...")

        self._preview_worker = PreviewWorker(
            self.pipeline, self._preview_node_id, self._preview_file,
            changed_node_id=self._preview_changed_node_id,
        )
        self._preview_changed_node_id = None  # Consumed
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_finished(self, node_id: str, port_results: dict) -> None:
        """Handle preview execution completion — display results."""
        self._preview_worker = None
        self.statusBar().showMessage("Preview: done")

        table_shown = False
        image_shown = False

        for val in port_results.values():
            if isinstance(val, TableData) and not table_shown:
                self.spreadsheet_viewer.set_table(val)
                table_shown = True
            elif isinstance(val, ImageContainer) and not image_shown:
                self.image_viewer.set_image(val)
                image_shown = True

        # Fall back to the single-result preview
        if not table_shown and not image_shown:
            result = self.pipeline.get_node_result(node_id)
            if result and isinstance(result, TableData):
                self.spreadsheet_viewer.set_table(result)
                table_shown = True
            elif result and isinstance(result, ImageContainer):
                self.image_viewer.set_image(result)
                image_shown = True

        if not image_shown:
            self.image_viewer.clear()
        if not table_shown:
            self.spreadsheet_viewer.clear()

        # Auto-show the relevant viewer windows
        if image_shown:
            self._show_viewer_window(self.image_window)
        if table_shown:
            self._show_viewer_window(self.spreadsheet_window)

    def _on_preview_error(self, message: str) -> None:
        """Handle preview execution error."""
        self._preview_worker = None
        self.statusBar().showMessage(f"Preview error: {message}")

    def _on_preview_param_changed(self, node_id: str) -> None:
        """Re-run preview if the changed node is in the preview path."""
        if self._preview_node_id is None:
            return

        # Check if the changed node affects the preview target
        if self.pipeline.connections:
            ancestors = self.pipeline._get_ancestor_node_ids(
                self._preview_node_id
            )
            if node_id not in ancestors:
                return
        else:
            # Linear mode: check index
            target_idx = None
            changed_idx = None
            for i, node in enumerate(self.pipeline.nodes):
                if node.node_id == self._preview_node_id:
                    target_idx = i
                if node.node_id == node_id:
                    changed_idx = i
            if target_idx is None or changed_idx is None:
                return
            if changed_idx > target_idx:
                return

        # Track which node changed so the preview can skip upstream cache
        self._preview_changed_node_id = node_id
        self._preview_debounce_timer.start()

    def _on_preview_file_changed(self, file_path) -> None:
        """Handle batch file combo selection change."""
        self._preview_file = file_path
        self._preview_changed_node_id = None  # Full run on file change
        self._run_preview()

    def _new_pipeline(self) -> None:
        """Create a new empty pipeline."""
        if self.pipeline.nodes:
            reply = QMessageBox.question(
                self,
                "New Pipeline",
                "Clear current pipeline and create new?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._clear_pipeline()
        self._current_file = None
        self.setWindowTitle("Analysis Pipeline")

    def _open_pipeline(self) -> None:
        """Open a pipeline from file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Pipeline",
            "",
            "Pipeline Files (*.json);;All Files (*)",
        )
        if path:
            try:
                self.pipeline.load(Path(path), self.registry)
                self.canvas.sync_from_pipeline()
                self._current_file = path
                self.setWindowTitle(f"Analysis Pipeline - {Path(path).name}")
                self.statusBar().showMessage(f"Opened: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open pipeline: {e}")

    def _save_pipeline(self) -> None:
        """Save the current pipeline."""
        if hasattr(self, '_current_file') and self._current_file:
            try:
                self.pipeline.save(Path(self._current_file))
                self.statusBar().showMessage(f"Saved: {self._current_file}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save pipeline: {e}")
        else:
            self._save_pipeline_as()

    def _save_pipeline_as(self) -> None:
        """Save the pipeline to a new file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Pipeline",
            "",
            "Pipeline Files (*.json);;All Files (*)",
        )
        if path:
            if not path.endswith('.json'):
                path += '.json'
            try:
                self.pipeline.save(Path(path))
                self._current_file = path
                self.setWindowTitle(f"Analysis Pipeline - {Path(path).name}")
                self.statusBar().showMessage(f"Saved: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save pipeline: {e}")

    def _cancel_preview(self) -> None:
        """Cancel any active preview."""
        self._preview_node_id = None
        self._preview_file = None
        self._preview_debounce_timer.stop()
        if self._preview_worker is not None:
            self._preview_worker.stop()
            self._preview_worker.wait()
            self._preview_worker = None
        self.image_viewer.set_batch_files([])

    def _clear_pipeline(self) -> None:
        """Clear the current pipeline."""
        self._cancel_preview()
        self.canvas.clear()
        self.properties_panel.set_node(None)
        self.image_viewer.clear()
        self.spreadsheet_viewer.clear()
        self.progress_view.reset()
        self.statusBar().showMessage("Pipeline cleared")

    def _run_pipeline(self) -> None:
        """Run the pipeline."""
        self._cancel_preview()

        # Validate pipeline
        errors = self.pipeline.validate()
        if errors:
            error_msg = "\n".join(f"Node {e.node_index}: {e.message}" for e in errors)
            QMessageBox.warning(self, "Validation Errors", error_msg)
            return

        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline", "Add nodes to the pipeline first.")
            return

        # Reset UI state
        self.canvas.reset_node_statuses()
        self.progress_view.reset()

        # Disable run, enable stop
        self.run_action.setEnabled(False)
        self.stop_action.setEnabled(True)

        # Check if this is a batch pipeline
        if self.pipeline.is_batch_pipeline():
            self._run_batch_pipeline()
        else:
            self._run_single_pipeline()

    def _run_single_pipeline(self) -> None:
        """Run pipeline for a single image."""
        self.progress_view.set_status("Running...")

        self._worker = PipelineWorker(self.pipeline)
        self._worker.progress.connect(self._on_progress)
        self._worker.node_started.connect(self._on_node_started)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _run_batch_pipeline(self) -> None:
        """Run pipeline for batch processing."""
        files = self.pipeline.get_batch_files()
        self.progress_view.set_status(f"Batch: 0/{len(files)} files")
        self.progress_view.set_batch_mode(True, len(files))

        self._batch_worker = BatchPipelineWorker(self.pipeline)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.file_started.connect(self._on_file_started)
        self._batch_worker.node_started.connect(self._on_node_started)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.error.connect(self._on_error)
        self._batch_worker.start()

    def _stop_pipeline(self) -> None:
        """Stop the running pipeline."""
        if hasattr(self, '_batch_worker') and self._batch_worker:
            self._batch_worker.stop()
            self.progress_view.set_status("Stopping...")
        elif self._worker:
            self._worker.stop()
            self.progress_view.set_status("Stopping...")

    def _on_progress(self, node_index: int, total: int, node_progress: float) -> None:
        """Handle progress updates."""
        if node_index < len(self.pipeline.nodes):
            node = self.pipeline.nodes[node_index]
            self.canvas.set_node_progress(node.node_id, node_progress)
            self.progress_view.set_node_progress(node_progress)
            self.progress_view.set_overall_progress(node_index, total)

    def _on_node_started(self, node_id: str, index: int, total: int) -> None:
        """Handle node execution start."""
        node = self.pipeline.get_node(node_id)
        if node:
            self.canvas.set_node_status(node_id, NodeStatus.PROCESSING)
            self.progress_view.set_current_node(node.plugin.name, index, total)

            # Mark previous nodes as complete
            for i in range(index):
                prev_node = self.pipeline.nodes[i]
                self.canvas.set_node_status(prev_node.node_id, NodeStatus.COMPLETE)

    def _on_finished(self, result) -> None:
        """Handle pipeline completion."""
        self.run_action.setEnabled(True)
        self.stop_action.setEnabled(False)

        if result:
            # Mark all nodes complete
            for node in self.pipeline.nodes:
                self.canvas.set_node_status(node.node_id, NodeStatus.COMPLETE)

            self.progress_view.set_complete(True)

            if isinstance(result, ImageContainer):
                self.image_viewer.set_image(result)
                self._show_viewer_window(self.image_window)
            else:
                self.image_viewer.clear()

            self.statusBar().showMessage("Pipeline completed successfully")
        else:
            self.progress_view.set_complete(False)
            self.statusBar().showMessage("Pipeline stopped or failed")

        # Always scan for table data — sink nodes may not return a result
        # but upstream nodes (e.g. Intensity Measurement) still have TableData
        self._show_last_table_result()

    def _on_batch_progress(
        self, file_idx: int, total_files: int,
        node_idx: int, total_nodes: int,
        progress: float, filename: str
    ) -> None:
        """Handle batch progress updates."""
        if node_idx < len(self.pipeline.nodes):
            node = self.pipeline.nodes[node_idx]
            self.canvas.set_node_progress(node.node_id, progress)

        self.progress_view.set_batch_progress(file_idx, total_files, node_idx, total_nodes, progress, filename)

    def _on_file_started(self, file_idx: int, total_files: int, filename: str) -> None:
        """Handle start of processing a new file in batch."""
        self.canvas.reset_node_statuses()
        self.progress_view.set_status(f"Batch: {file_idx + 1}/{total_files} files")
        self.statusBar().showMessage(f"Processing: {filename}")

    def _on_batch_finished(self, results: list) -> None:
        """Handle batch pipeline completion."""
        self.run_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self._batch_worker = None

        # Mark all nodes complete
        for node in self.pipeline.nodes:
            self.canvas.set_node_status(node.node_id, NodeStatus.COMPLETE)

        self.progress_view.set_batch_mode(False, 0)
        self.progress_view.set_complete(True)

        if results:
            # Show the last processed image (if it's an image)
            last = results[-1]
            if isinstance(last, ImageContainer):
                self.image_viewer.set_image(last)
                self._show_viewer_window(self.image_window)
            else:
                self.image_viewer.clear()

            self.statusBar().showMessage(f"Batch complete: {len(results)} images processed")
        else:
            self.statusBar().showMessage("Batch processing stopped or failed")

        # Always scan for table data — works even when last node is a sink
        self._show_last_table_result()

    def _on_error(self, message: str) -> None:
        """Handle pipeline errors."""
        self.run_action.setEnabled(True)
        self.stop_action.setEnabled(False)

        self.progress_view.set_error(message)
        QMessageBox.critical(self, "Pipeline Error", message)

    def _show_viewer_window(self, window) -> None:
        """Show and raise a viewer window."""
        window.show()
        window.raise_()
        window.activateWindow()

    def _toggle_browser(self) -> None:
        """Toggle plugin browser visibility."""
        self.plugin_browser.setVisible(self.show_browser_action.isChecked())

    def _toggle_properties(self) -> None:
        """Toggle properties panel visibility."""
        self.properties_panel.setVisible(self.show_properties_action.isChecked())

    def _toggle_viewer(self) -> None:
        """Toggle image viewer window visibility."""
        if self.show_viewer_action.isChecked():
            self._show_viewer_window(self.image_window)
        else:
            self.image_window.hide()

    def _toggle_spreadsheet(self) -> None:
        """Toggle spreadsheet viewer window visibility."""
        if self.show_spreadsheet_action.isChecked():
            self._show_viewer_window(self.spreadsheet_window)
        else:
            self.spreadsheet_window.hide()

    def _show_last_table_result(self) -> None:
        """Scan all nodes for TableData results and display the last one found."""
        last_table = None
        for node in self.pipeline.nodes:
            port_results = self.pipeline.get_node_port_results(node.node_id)
            for val in port_results.values():
                if isinstance(val, TableData):
                    last_table = val

            # Also check the single-result store
            result = self.pipeline.get_node_result(node.node_id)
            if isinstance(result, TableData):
                last_table = result

        if last_table is not None:
            self.spreadsheet_viewer.set_table(last_table)
            self._show_viewer_window(self.spreadsheet_window)
        else:
            self.spreadsheet_viewer.clear()

    def _show_plugin_defaults(self) -> None:
        """Show the plugin defaults settings dialog."""
        dialog = PluginDefaultsDialog(self.registry, self.settings, self)
        dialog.exec()

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Analysis Pipeline",
            "Scientific Analysis Pipeline Tool\n\n"
            "A visual pipeline editor for image analysis.\n\n"
            "Drag plugins from the browser onto the canvas,\n"
            "connect them to build your pipeline, and run.",
        )
