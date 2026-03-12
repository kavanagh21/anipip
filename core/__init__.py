"""Core components for the analysis pipeline."""

from .pipeline_data import PipelineData
from .image_container import ImageContainer, ImageMetadata, ImageType, normalize_tiff_axes
from .table_data import TableData
from .parameters import (
    Parameter,
    IntParameter,
    FloatParameter,
    ChoiceParameter,
    BoolParameter,
    FileParameter,
    ActionParameter,
)
from .ports import Port, InputPort, OutputPort, PortDirection, Connection
from .plugin_base import BasePlugin
from .plugin_registry import PluginRegistry
from .pipeline import Pipeline, PipelineNode

__all__ = [
    "PipelineData",
    "ImageContainer",
    "ImageMetadata",
    "ImageType",
    "normalize_tiff_axes",
    "TableData",
    "Parameter",
    "IntParameter",
    "FloatParameter",
    "ChoiceParameter",
    "BoolParameter",
    "FileParameter",
    "ActionParameter",
    "Port",
    "InputPort",
    "OutputPort",
    "PortDirection",
    "Connection",
    "BasePlugin",
    "PluginRegistry",
    "Pipeline",
    "PipelineNode",
]
