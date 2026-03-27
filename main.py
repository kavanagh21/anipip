#!/usr/bin/env python3
"""Application entry point for the Scientific Analysis Pipeline Tool."""

__version__ = "0.1.1"

import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """Launch the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Analysis Pipeline")
    app.setOrganizationName("AnalysisPipeline")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
