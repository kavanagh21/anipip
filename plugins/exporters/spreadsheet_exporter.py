"""Spreadsheet exporter plugin — writes TableData to CSV or Excel."""

import csv
from pathlib import Path
from typing import Callable

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer
from core.table_data import TableData
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import ChoiceParameter, FileParameter, BoolParameter


class SpreadsheetExporter(BasePlugin):
    """Write measurement / tabular data to CSV or Excel.

    This is a *sink* node — it has an input port but no output ports.

    In **batch mode** (the default when running a batch pipeline), rows are
    accumulated via :meth:`batch_initialize` / :meth:`batch_finalize` and
    written once at the end.  In **single mode**, the file is written
    immediately on each execution.
    """

    name = "Spreadsheet Exporter"
    category = "Exporters"
    description = "Export tabular measurement data to CSV or Excel"
    help_text = (
        "Writes measurement table data to a CSV or Excel (.xlsx) file. In "
        "batch mode, rows are accumulated across all images and written once "
        "at the end. Enable \"Show in internal viewer\" to display results in "
        "the app. Supports append mode for adding to existing files. Excel "
        "export requires the openpyxl package."
    )
    icon = None

    # -- Ports ---------------------------------------------------------

    ports: list[Port] = [
        InputPort("measurements", TableData, label="Measurements"),
        OutputPort("table_out", TableData, label="Table Out"),
    ]

    # -- Parameters ----------------------------------------------------

    parameters = [
        ChoiceParameter(
            name="format",
            label="Format",
            choices=["CSV", "Excel (.xlsx)"],
            default="CSV",
        ),
        FileParameter(
            name="output_path",
            label="Output Path",
            default="",
            filter="CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*.*)",
            save_mode=True,
        ),
        BoolParameter(
            name="append_mode",
            label="Append to existing file",
            default=False,
        ),
        BoolParameter(
            name="show_in_viewer",
            label="Show in internal viewer",
            default=True,
        ),
    ]

    def __init__(self):
        super().__init__()
        self._accumulated: TableData | None = None
        self._last_table: TableData | None = None

    # -- Batch lifecycle -----------------------------------------------

    def batch_initialize(self) -> None:
        """Start a fresh accumulator for the batch run."""
        self._accumulated = TableData()

    def batch_finalize(self) -> None:
        """Write the accumulated table after the batch completes."""
        if self._accumulated and self._accumulated.rows:
            if self.get_parameter("output_path"):
                self._write_table(self._accumulated)
            # Keep the final accumulated table available for the viewer
            self._last_table = self._accumulated
        self._accumulated = None

    # -- Processing ----------------------------------------------------

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        """Legacy interface — no-op pass-through."""
        return image

    def process_ports(
        self,
        inputs: dict[str, PipelineData],
        progress_callback: Callable[[float], None],
    ) -> dict[str, PipelineData]:
        """Receive table data and either accumulate or write immediately."""
        progress_callback(0.1)

        table: TableData | None = inputs.get("measurements")
        if table is None:
            progress_callback(1.0)
            return {}

        if self._accumulated is not None:
            # Batch mode — accumulate rows
            self._accumulated = self._accumulated.merge(table)
        else:
            # Single mode — write to file if a path is configured
            if self.get_parameter("output_path"):
                self._write_table(table)

        # Store the current table for viewer access
        self._last_table = table

        progress_callback(1.0)

        # Expose table on output port when show_in_viewer is enabled
        if self.get_parameter("show_in_viewer"):
            out = self._accumulated if self._accumulated is not None else table
            return {"table_out": out}

        return {}

    # -- Writing ---------------------------------------------------

    def _write_table(self, table: TableData) -> None:
        """Write *table* using the configured format."""
        fmt = self.get_parameter("format")
        if fmt == "Excel (.xlsx)":
            self._write_xlsx(table)
        else:
            self._write_csv(table)

    def _write_csv(self, table: TableData) -> None:
        """Write *table* to the configured CSV path."""
        output_path = Path(self.get_parameter("output_path"))
        append = self.get_parameter("append_mode")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if append and output_path.exists():
            # Append rows (skip header if file already has content)
            with open(output_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=table.columns, extrasaction="ignore")
                # Write header only if the file is empty
                if output_path.stat().st_size == 0:
                    writer.writeheader()
                for row in table.rows:
                    writer.writerow(row)
        else:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=table.columns, extrasaction="ignore")
                writer.writeheader()
                for row in table.rows:
                    writer.writerow(row)

    def _write_xlsx(self, table: TableData) -> None:
        """Write *table* to an Excel .xlsx file via openpyxl."""
        try:
            import openpyxl
        except ImportError:
            raise RuntimeError(
                "Excel export requires the openpyxl package. "
                "Install it with: pip install openpyxl"
            )

        output_path = Path(self.get_parameter("output_path"))
        append = self.get_parameter("append_mode")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if append and output_path.exists():
            wb = openpyxl.load_workbook(output_path)
            ws = wb.active
        else:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Measurements"
            # Write header row
            for col_idx, col_name in enumerate(table.columns, start=1):
                ws.cell(row=1, column=col_idx, value=col_name)

        start_row = ws.max_row + 1
        for row_idx, row_data in enumerate(table.rows, start=start_row):
            for col_idx, col_name in enumerate(table.columns, start=1):
                ws.cell(row=row_idx, column=col_idx, value=row_data.get(col_name, ""))

        wb.save(output_path)

    # -- Validation ----------------------------------------------------

    def validate_parameters(self) -> tuple[bool, list[str]]:
        errors = []
        output_path = self.get_parameter("output_path")
        show_in_viewer = self.get_parameter("show_in_viewer")
        if not output_path and not show_in_viewer:
            errors.append("Either specify an output file path or enable 'Show in internal viewer'")
        return len(errors) == 0, errors
