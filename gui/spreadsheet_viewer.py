"""Spreadsheet viewer for displaying TableData inline."""

import csv
import io
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QApplication,
    QAbstractItemView,
)

from core.table_data import TableData


class NumericTableItem(QTableWidgetItem):
    """Table item that sorts numerically when the value is a number."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        try:
            return float(self.text()) < float(other.text())
        except (ValueError, TypeError):
            return super().__lt__(other)


class SpreadsheetViewer(QWidget):
    """Displays TableData in a sortable table with copy/export controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._table_data: Optional[TableData] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()

        self.info_label = QLabel("No data")
        toolbar.addWidget(self.info_label)

        toolbar.addStretch()

        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        toolbar.addWidget(copy_btn)

        export_btn = QPushButton("Export CSV...")
        export_btn.clicked.connect(self._export_csv)
        toolbar.addWidget(export_btn)

        layout.addLayout(toolbar)

        # Table widget
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(24)
        layout.addWidget(self.table)

        # Ctrl+C shortcut for copying selection
        copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self.table)
        copy_shortcut.activated.connect(self._copy_selection)

    def set_table(self, table_data: TableData) -> None:
        """Populate the table widget from a TableData object."""
        self._table_data = table_data

        # Disable sorting while populating to avoid re-sorts on each insert
        self.table.setSortingEnabled(False)
        self.table.clear()

        if not table_data or not table_data.columns:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.info_label.setText("No data")
            self.table.setSortingEnabled(True)
            return

        columns = table_data.columns
        rows = table_data.rows

        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(rows))

        for r, row in enumerate(rows):
            for c, col in enumerate(columns):
                value = row.get(col)
                if value is None:
                    text = ""
                elif isinstance(value, float):
                    text = f"{value:.6f}"
                else:
                    text = str(value)

                item = NumericTableItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, item)

        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.info_label.setText(f"{len(rows)} rows, {len(columns)} columns")

    def clear(self) -> None:
        """Clear the table."""
        self._table_data = None
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.info_label.setText("No data")

    def _copy_to_clipboard(self) -> None:
        """Copy entire table as tab-separated values to clipboard."""
        if self._table_data is None:
            return

        lines = ["\t".join(self._table_data.columns)]
        for row in self._table_data.rows:
            values = []
            for col in self._table_data.columns:
                v = row.get(col)
                if v is None:
                    values.append("")
                elif isinstance(v, float):
                    values.append(f"{v:.6f}")
                else:
                    values.append(str(v))
            lines.append("\t".join(values))

        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(lines))

    def _copy_selection(self) -> None:
        """Copy selected cells as tab-separated values."""
        selection = self.table.selectedRanges()
        if not selection:
            return

        lines = []
        for sel_range in selection:
            for r in range(sel_range.topRow(), sel_range.bottomRow() + 1):
                row_values = []
                for c in range(sel_range.leftColumn(), sel_range.rightColumn() + 1):
                    item = self.table.item(r, c)
                    row_values.append(item.text() if item else "")
                lines.append("\t".join(row_values))

        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(lines))

    def _export_csv(self) -> None:
        """Export current table data to a CSV file."""
        if self._table_data is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        if not path.endswith(".csv"):
            path += ".csv"

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._table_data.columns, extrasaction="ignore"
            )
            writer.writeheader()
            for row in self._table_data.rows:
                writer.writerow(row)
