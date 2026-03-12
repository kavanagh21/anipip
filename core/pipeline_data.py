"""Base class for all data types flowing through the pipeline."""

from dataclasses import dataclass


@dataclass
class PipelineData:
    """Root base class for all pipeline data types.

    All data flowing through pipeline ports must inherit from this class.
    This enables type checking on port connections.
    """

    pass
