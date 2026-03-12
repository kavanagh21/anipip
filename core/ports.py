"""Port and connection types for the DAG pipeline."""

from dataclasses import dataclass
from enum import Enum
from typing import Type

from .pipeline_data import PipelineData


class PortDirection(Enum):
    """Direction of a port on a plugin node."""

    INPUT = "input"
    OUTPUT = "output"


class PortSide(Enum):
    """Which side of the node a port handle is rendered on."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class Port:
    """A typed input or output slot on a plugin.

    Attributes:
        name: Unique identifier within the plugin (e.g. "image_in")
        direction: INPUT or OUTPUT
        data_type: The PipelineData subclass this port accepts/produces
        label: Human-readable display label
        optional: If True, the port does not need to be connected
    """

    name: str
    direction: PortDirection
    data_type: Type[PipelineData]
    label: str = ""
    optional: bool = False

    def __post_init__(self):
        if not self.label:
            object.__setattr__(self, "label", self.name.replace("_", " ").title())

    def is_compatible(self, other: "Port") -> bool:
        """Check if this output port can connect to another input port.

        An output can connect to an input if the output's data_type is the same
        as or a subclass of the input's data_type.

        Args:
            other: The target input port

        Returns:
            True if compatible
        """
        if self.direction != PortDirection.OUTPUT or other.direction != PortDirection.INPUT:
            return False
        return issubclass(self.data_type, other.data_type)


def InputPort(
    name: str,
    data_type: Type[PipelineData],
    label: str = "",
    optional: bool = False,
) -> Port:
    """Convenience factory for creating an input port."""
    return Port(
        name=name,
        direction=PortDirection.INPUT,
        data_type=data_type,
        label=label,
        optional=optional,
    )


def OutputPort(
    name: str,
    data_type: Type[PipelineData],
    label: str = "",
) -> Port:
    """Convenience factory for creating an output port."""
    return Port(
        name=name,
        direction=PortDirection.OUTPUT,
        data_type=data_type,
        label=label,
    )


@dataclass(frozen=True)
class Connection:
    """A directed edge in the pipeline DAG.

    Attributes:
        source_node_id: ID of the node producing data
        source_port: Name of the output port on the source node
        target_node_id: ID of the node consuming data
        target_port: Name of the input port on the target node
    """

    source_node_id: str
    source_port: str
    target_node_id: str
    target_port: str

    def to_dict(self) -> dict:
        """Serialize to a dictionary for JSON."""
        return {
            "source_node_id": self.source_node_id,
            "source_port": self.source_port,
            "target_node_id": self.target_node_id,
            "target_port": self.target_port,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Connection":
        """Deserialize from a dictionary."""
        return cls(
            source_node_id=data["source_node_id"],
            source_port=data["source_port"],
            target_node_id=data["target_node_id"],
            target_port=data["target_port"],
        )
