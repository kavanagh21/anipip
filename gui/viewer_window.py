"""Standalone viewer window for hosting widgets outside the main window."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class ViewerWindow(QWidget):
    """Floating window that hosts a viewer widget (image or spreadsheet).

    Closing the window only hides it; the widget is preserved.
    """

    def __init__(self, widget: QWidget, title: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setWindowTitle(title)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)

        self.resize(800, 600)

    def closeEvent(self, event) -> None:
        # Hide instead of destroying so the widget stays alive
        self.hide()
        event.ignore()
