"""Tabular data container for measurement results."""

import csv
import io
from dataclasses import dataclass, field

from .pipeline_data import PipelineData


@dataclass
class TableData(PipelineData):
    """Tabular data with named columns and row dictionaries.

    Used for measurement results, statistics, and other structured data
    that can be exported to CSV or displayed in a table view.

    Attributes:
        columns: Ordered list of column names
        rows: List of row dictionaries mapping column name to value
    """

    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)

    def add_row(self, row: dict) -> None:
        """Add a row of data.

        Any keys not in columns are added as new columns automatically.

        Args:
            row: Dictionary mapping column names to values
        """
        for key in row:
            if key not in self.columns:
                self.columns.append(key)
        self.rows.append(row)

    def merge(self, other: "TableData") -> "TableData":
        """Merge another TableData into a new combined TableData.

        Columns from both tables are unioned; missing values become None.

        Args:
            other: Another TableData to merge with

        Returns:
            New TableData with all rows from both tables
        """
        merged_columns = list(self.columns)
        for col in other.columns:
            if col not in merged_columns:
                merged_columns.append(col)

        merged_rows = []
        for row in self.rows:
            merged_rows.append(row.copy())
        for row in other.rows:
            merged_rows.append(row.copy())

        return TableData(columns=merged_columns, rows=merged_rows)

    def to_csv_string(self) -> str:
        """Serialize to a CSV-formatted string.

        Returns:
            CSV string with header row and data rows
        """
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.columns, extrasaction="ignore")
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row)
        return output.getvalue()

    def copy(self) -> "TableData":
        """Create a deep copy.

        Returns:
            New TableData with copied columns and rows
        """
        return TableData(
            columns=list(self.columns),
            rows=[row.copy() for row in self.rows],
        )
