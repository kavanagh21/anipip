"""Pipeline execution engine with DAG support."""

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .image_container import ImageContainer, ImageType
from .pipeline_data import PipelineData
from .plugin_base import BasePlugin
from .plugin_registry import PluginRegistry
from .ports import Connection, PortDirection
from .table_data import TableData


@dataclass
class ValidationError:
    """Represents a validation error in the pipeline."""

    node_index: int
    message: str


@dataclass
class PipelineNode:
    """A node in the pipeline representing a processing step.

    Attributes:
        plugin: The plugin instance for this node
        parameters: Current parameter values
        position: Canvas position (x, y) for GUI
        node_id: Unique identifier for the node
    """

    plugin: BasePlugin
    parameters: dict[str, any] = field(default_factory=dict)
    position: tuple[int, int] = (0, 0)
    node_id: str = ""
    port_sides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.node_id:
            import uuid
            self.node_id = str(uuid.uuid4())[:8]

        # Initialize parameters from plugin defaults
        if not self.parameters:
            self.parameters = self.plugin.get_all_parameters()
        else:
            self.plugin.set_all_parameters(self.parameters)


class Pipeline:
    """Manages execution of connected pipeline nodes.

    Supports both legacy linear execution (v1.0 — nodes run in list order)
    and DAG execution (v2.0 — explicit connections, topological ordering).
    """

    def __init__(self):
        self._nodes: list[PipelineNode] = []
        self._connections: list[Connection] = []
        self._last_results: dict[str, PipelineData] = {}
        # Per-node, per-port results from last execution
        self._port_results: dict[str, dict[str, PipelineData]] = {}

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> list[PipelineNode]:
        """Get the list of nodes."""
        return self._nodes

    @property
    def connections(self) -> list[Connection]:
        """Get the list of connections."""
        return list(self._connections)

    def add_node(self, node: PipelineNode, index: Optional[int] = None) -> None:
        """Add a node to the pipeline.

        Args:
            node: The node to add
            index: Optional position to insert at. If None, appends to end.
        """
        if index is None:
            self._nodes.append(node)
        else:
            self._nodes.insert(index, node)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connections from the pipeline.

        Args:
            node_id: The ID of the node to remove

        Returns:
            True if node was removed, False if not found
        """
        for i, node in enumerate(self._nodes):
            if node.node_id == node_id:
                self._nodes.pop(i)
                # Clean up associated connections
                self._connections = [
                    c for c in self._connections
                    if c.source_node_id != node_id and c.target_node_id != node_id
                ]
                self._last_results.pop(node_id, None)
                self._port_results.pop(node_id, None)
                return True
        return False

    def get_node(self, node_id: str) -> Optional[PipelineNode]:
        """Get a node by its ID."""
        for node in self._nodes:
            if node.node_id == node_id:
                return node
        return None

    def move_node(self, node_id: str, new_index: int) -> bool:
        """Move a node to a new position in the pipeline."""
        old_index = None
        for i, node in enumerate(self._nodes):
            if node.node_id == node_id:
                old_index = i
                break

        if old_index is None:
            return False

        node = self._nodes.pop(old_index)
        self._nodes.insert(new_index, node)
        return True

    def clear(self) -> None:
        """Remove all nodes and connections from the pipeline."""
        self._nodes.clear()
        self._connections.clear()
        self._last_results.clear()
        self._port_results.clear()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def add_connection(self, connection: Connection) -> None:
        """Add a connection between two ports with validation.

        Args:
            connection: The connection to add

        Raises:
            ValueError: If the connection is invalid (missing nodes/ports,
                type mismatch, duplicate, or would create a cycle)
        """
        # Validate source node exists
        source_node = self.get_node(connection.source_node_id)
        if source_node is None:
            raise ValueError(f"Source node not found: {connection.source_node_id}")

        # Validate target node exists
        target_node = self.get_node(connection.target_node_id)
        if target_node is None:
            raise ValueError(f"Target node not found: {connection.target_node_id}")

        # Cannot connect a node to itself
        if connection.source_node_id == connection.target_node_id:
            raise ValueError("Cannot connect a node to itself")

        # Validate source port exists and is an output
        source_ports = {p.name: p for p in source_node.plugin.get_output_ports()}
        if connection.source_port not in source_ports:
            raise ValueError(
                f"Output port '{connection.source_port}' not found on "
                f"'{source_node.plugin.name}'"
            )

        # Validate target port exists and is an input
        target_ports = {p.name: p for p in target_node.plugin.get_input_ports()}
        if connection.target_port not in target_ports:
            raise ValueError(
                f"Input port '{connection.target_port}' not found on "
                f"'{target_node.plugin.name}'"
            )

        # Type compatibility check
        src_port = source_ports[connection.source_port]
        tgt_port = target_ports[connection.target_port]
        if not src_port.is_compatible(tgt_port):
            raise ValueError(
                f"Type mismatch: {src_port.data_type.__name__} output cannot "
                f"connect to {tgt_port.data_type.__name__} input"
            )

        # Check for duplicate connection to same input port
        for existing in self._connections:
            if (existing.target_node_id == connection.target_node_id
                    and existing.target_port == connection.target_port):
                raise ValueError(
                    f"Input port '{connection.target_port}' on "
                    f"'{target_node.plugin.name}' is already connected"
                )

        # Cycle detection
        if self._would_create_cycle(connection):
            raise ValueError("Connection would create a cycle in the pipeline")

        self._connections.append(connection)

    def remove_connection(self, connection: Connection) -> bool:
        """Remove a connection.

        Args:
            connection: The connection to remove

        Returns:
            True if removed, False if not found
        """
        try:
            self._connections.remove(connection)
            return True
        except ValueError:
            return False

    def get_connections_from(self, node_id: str) -> list[Connection]:
        """Get all connections originating from a node."""
        return [c for c in self._connections if c.source_node_id == node_id]

    def get_connections_to(self, node_id: str) -> list[Connection]:
        """Get all connections targeting a node."""
        return [c for c in self._connections if c.target_node_id == node_id]

    def get_connection_for_input(
        self, node_id: str, port_name: str
    ) -> Optional[Connection]:
        """Get the connection feeding a specific input port."""
        for c in self._connections:
            if c.target_node_id == node_id and c.target_port == port_name:
                return c
        return None

    def _would_create_cycle(self, new_connection: Connection) -> bool:
        """Check if adding a connection would create a cycle (DFS)."""
        # Build adjacency list including the proposed connection
        adj: dict[str, set[str]] = {}
        for node in self._nodes:
            adj[node.node_id] = set()

        for c in self._connections:
            adj.setdefault(c.source_node_id, set()).add(c.target_node_id)

        # Add the proposed edge
        adj.setdefault(new_connection.source_node_id, set()).add(
            new_connection.target_node_id
        )

        # DFS from target to see if we can reach source (cycle)
        visited = set()
        stack = [new_connection.target_node_id]
        while stack:
            current = stack.pop()
            if current == new_connection.source_node_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adj.get(current, set()))

        return False

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def _topological_sort(self) -> list[PipelineNode]:
        """Sort nodes in topological order using Kahn's algorithm.

        Returns:
            Nodes in execution order

        Raises:
            RuntimeError: If a cycle is detected (should not happen if
                add_connection validates properly)
        """
        node_map = {n.node_id: n for n in self._nodes}

        # Build in-degree map
        in_degree: dict[str, int] = {n.node_id: 0 for n in self._nodes}
        adj: dict[str, list[str]] = {n.node_id: [] for n in self._nodes}

        for c in self._connections:
            adj[c.source_node_id].append(c.target_node_id)
            in_degree[c.target_node_id] += 1

        # Start with zero in-degree nodes
        queue = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        ordered: list[PipelineNode] = []

        while queue:
            nid = queue.popleft()
            ordered.append(node_map[nid])
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(self._nodes):
            raise RuntimeError("Cycle detected in pipeline graph")

        return ordered

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[ValidationError]:
        """Check if the pipeline is valid before execution.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self._nodes:
            errors.append(ValidationError(-1, "Pipeline has no nodes"))
            return errors

        for i, node in enumerate(self._nodes):
            # Validate plugin parameters
            is_valid, param_errors = node.plugin.validate_parameters()
            if not is_valid:
                for error in param_errors:
                    errors.append(ValidationError(i, error))

        return errors

    # ------------------------------------------------------------------
    # Auto-iteration for stack/timelapse data
    # ------------------------------------------------------------------

    def _execute_node_with_auto_iteration(
        self,
        node: PipelineNode,
        inputs: dict[str, PipelineData],
        progress_callback: Callable[[float], None],
    ) -> dict[str, PipelineData]:
        """Execute a node, auto-iterating over slices when needed.

        If any ImageContainer input has an image_type not in the plugin's
        ``accepted_image_types``, the stack is split along axis 0, the
        plugin is called once per slice, and outputs are reassembled.

        Args:
            node: The pipeline node to execute
            inputs: Mapping of input port name to data object
            progress_callback: Progress callback (0.0-1.0)

        Returns:
            Mapping of output port name to produced data object
        """
        plugin = node.plugin

        # Check if any ImageContainer input needs iteration
        needs_iteration = False
        for data in inputs.values():
            if isinstance(data, ImageContainer):
                if data.image_type not in plugin.accepted_image_types:
                    needs_iteration = True
                    break

        if not needs_iteration:
            return plugin.process_ports(inputs, progress_callback)

        # Find the stack to iterate over and determine slice count
        stack_type = ImageType.SINGLE
        num_slices = 0
        for data in inputs.values():
            if isinstance(data, ImageContainer) and data.image_type not in plugin.accepted_image_types:
                num_slices = data.num_slices
                stack_type = data.image_type
                break

        # Collect per-slice results
        all_slice_outputs: list[dict[str, PipelineData]] = []

        for s in range(num_slices):
            # Build single-slice inputs
            slice_inputs: dict[str, PipelineData] = {}
            for port_name, data in inputs.items():
                if isinstance(data, ImageContainer) and data.image_type not in plugin.accepted_image_types:
                    slice_data = data.data[s]
                    slice_meta = data.metadata.copy()
                    slice_meta.image_type = ImageType.SINGLE
                    slice_meta.num_slices = 1
                    slice_inputs[port_name] = ImageContainer(data=slice_data, metadata=slice_meta)
                else:
                    slice_inputs[port_name] = data

            def slice_progress(p: float, s_idx=s):
                # Map slice progress into overall progress
                progress_callback((s_idx + p) / num_slices)

            slice_outputs = plugin.process_ports(slice_inputs, slice_progress)
            all_slice_outputs.append(slice_outputs or {})

        # Reassemble outputs
        if not all_slice_outputs:
            return {}

        result: dict[str, PipelineData] = {}
        all_keys = all_slice_outputs[0].keys()

        for key in all_keys:
            first = all_slice_outputs[0].get(key)

            if isinstance(first, ImageContainer):
                # Stack image arrays back together
                stacked_data = np.stack(
                    [out[key].data for out in all_slice_outputs], axis=0
                )
                stacked_meta = first.metadata.copy()
                stacked_meta.image_type = stack_type
                stacked_meta.num_slices = num_slices
                result[key] = ImageContainer(data=stacked_data, metadata=stacked_meta)

            elif isinstance(first, TableData):
                # Merge all slice tables
                merged = all_slice_outputs[0][key]
                for out in all_slice_outputs[1:]:
                    merged = merged.merge(out[key])
                result[key] = merged

            else:
                # Pass through last slice result
                result[key] = all_slice_outputs[-1].get(key)

        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        input_image: Optional[ImageContainer],
        progress_callback: Callable[[int, int, float], None],
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[PipelineData]:
        """Execute the pipeline.

        If connections exist, runs DAG execution. Otherwise falls back to
        legacy linear execution for v1.0 compatibility.

        Args:
            input_image: Initial input image (can be None if first node is a loader)
            progress_callback: Callback receiving (current_node_index, total_nodes, node_progress)
            stop_check: Optional callback that returns True if execution should stop

        Returns:
            Final output data or None if execution failed
        """
        if self._connections:
            return self._execute_dag(progress_callback, stop_check)
        return self._execute_linear(input_image, progress_callback, stop_check)

    def _execute_linear(
        self,
        input_image: Optional[ImageContainer],
        progress_callback: Callable[[int, int, float], None],
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[ImageContainer]:
        """Execute the pipeline sequentially (legacy v1.0 mode).

        Args:
            input_image: Initial input image (can be None if first node is a loader)
            progress_callback: Callback receiving (current_node_index, total_nodes, node_progress)
            stop_check: Optional callback that returns True if execution should stop

        Returns:
            Final output image or None if execution failed
        """
        if not self._nodes:
            return input_image

        current_image = input_image
        total_nodes = len(self._nodes)
        # Port outputs from the previous v2.0 node, used to route
        # compatible data (e.g. TableData) between nodes in linear mode.
        previous_outputs: dict[str, PipelineData] = {}

        for i, node in enumerate(self._nodes):
            if stop_check and stop_check():
                return None

            # Sync parameters to plugin
            node.plugin.set_all_parameters(node.parameters)

            # Create progress callback for this node
            def node_progress(progress: float, node_index=i):
                progress_callback(node_index, total_nodes, progress)

            # Execute the node
            try:
                node_progress(0.0)

                if node.plugin.ports:
                    # v2.0 node: use process_ports for full multi-port output
                    inputs: dict[str, PipelineData] = {}
                    if current_image is not None:
                        for port in node.plugin.get_input_ports():
                            if issubclass(ImageContainer, port.data_type):
                                inputs[port.name] = current_image
                                break

                    # Route compatible data from previous node's outputs
                    for port in node.plugin.get_input_ports():
                        if port.name not in inputs:
                            for out_data in previous_outputs.values():
                                if out_data is not None and isinstance(out_data, port.data_type):
                                    inputs[port.name] = out_data
                                    break

                    outputs = self._execute_node_with_auto_iteration(node, inputs, node_progress)
                    self._port_results[node.node_id] = outputs or {}
                    previous_outputs = outputs or {}

                    # Continue linear chain with image output
                    new_image = (outputs or {}).get("image_out")
                    if isinstance(new_image, ImageContainer):
                        current_image = new_image

                    # Store preview (prefer ImageContainer)
                    preview = None
                    for val in (outputs or {}).values():
                        if isinstance(val, ImageContainer):
                            preview = val
                            break
                    self._last_results[node.node_id] = preview if preview is not None else current_image
                else:
                    # Legacy node: route through auto-iteration wrapper
                    if current_image is not None:
                        is_valid, error = node.plugin.validate_input(current_image)
                        if not is_valid:
                            raise RuntimeError(f"Node {i} ({node.plugin.name}): {error}")

                    inputs = {}
                    if current_image is not None:
                        inputs["image_in"] = current_image
                    outputs = self._execute_node_with_auto_iteration(node, inputs, node_progress)
                    new_image = (outputs or {}).get("image_out")
                    if isinstance(new_image, ImageContainer):
                        current_image = new_image
                    self._last_results[node.node_id] = current_image

                node_progress(1.0)

            except Exception as e:
                raise RuntimeError(f"Node {i} ({node.plugin.name}) failed: {e}")

        return current_image

    def _execute_dag(
        self,
        progress_callback: Callable[[int, int, float], None],
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[PipelineData]:
        """Execute the pipeline in DAG (topological) order.

        Each node receives its inputs from connected upstream ports and
        produces outputs that are routed to downstream nodes.

        Returns:
            The output of the last executed node (first ImageContainer found
            among its outputs, or first output of any type), or None.
        """
        ordered = self._topological_sort()
        total_nodes = len(ordered)
        self._port_results.clear()

        last_result: Optional[PipelineData] = None
        # Track source folder from any ImageContainer seen during execution
        source_folder = None

        for i, node in enumerate(ordered):
            if stop_check and stop_check():
                return None

            node.plugin.set_all_parameters(node.parameters)
            # Propagate source folder to all plugins
            node.plugin._pipeline_source_folder = source_folder

            # Gather inputs from upstream connections
            inputs: dict[str, PipelineData] = {}
            for conn in self.get_connections_to(node.node_id):
                upstream_outputs = self._port_results.get(conn.source_node_id, {})
                data = upstream_outputs.get(conn.source_port)
                if data is not None:
                    inputs[conn.target_port] = data

            def node_progress(progress: float, node_index=i):
                progress_callback(node_index, total_nodes, progress)

            try:
                node_progress(0.0)
                outputs = self._execute_node_with_auto_iteration(node, inputs, node_progress)
                node_progress(1.0)

                self._port_results[node.node_id] = outputs or {}

                # Store first ImageContainer (or any output) for preview
                # and track source folder
                preview = None
                for val in (outputs or {}).values():
                    if isinstance(val, ImageContainer):
                        if preview is None:
                            preview = val
                        if val.metadata.source_path and source_folder is None:
                            source_folder = val.metadata.source_path.parent
                if preview is None and outputs:
                    preview = next(iter(outputs.values()), None)

                self._last_results[node.node_id] = preview
                if preview is not None:
                    last_result = preview

            except Exception as e:
                raise RuntimeError(f"Node {i} ({node.plugin.name}) failed: {e}")

        return last_result

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def get_node_result(self, node_id: str) -> Optional[PipelineData]:
        """Get the last execution result for a node.

        Returns the first ImageContainer output if available, otherwise the
        first output of any type.
        """
        return self._last_results.get(node_id)

    def get_node_port_results(self, node_id: str) -> dict[str, PipelineData]:
        """Get the per-port results from the last DAG execution for a node.

        Returns:
            Dictionary mapping port name to PipelineData, or empty dict.
        """
        return self._port_results.get(node_id, {})

    def get_node_inputs(self, node_id: str) -> dict[str, PipelineData]:
        """Gather cached input data for a node from the last execution."""
        inputs: dict[str, PipelineData] = {}

        # DAG mode: follow connections
        conns = self.get_connections_to(node_id)
        if conns:
            for conn in conns:
                upstream = self._port_results.get(conn.source_node_id, {})
                data = upstream.get(conn.source_port)
                if data is not None:
                    inputs[conn.target_port] = data
            return inputs

        # Linear mode: previous node's output → image_in
        for i, node in enumerate(self._nodes):
            if node.node_id == node_id and i > 0:
                prev = self._port_results.get(self._nodes[i - 1].node_id, {})
                prev_image = prev.get("image_out")
                if prev_image is not None:
                    inputs["image_in"] = prev_image
                break

        return inputs

    # ------------------------------------------------------------------
    # Preview execution (partial pipeline for live node preview)
    # ------------------------------------------------------------------

    def _get_ancestor_node_ids(self, node_id: str) -> set[str]:
        """Get all upstream node IDs that the target depends on (BFS backwards).

        Returns:
            Set of node IDs including the target itself.
        """
        ancestors = {node_id}
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for conn in self.get_connections_to(current):
                if conn.source_node_id not in ancestors:
                    ancestors.add(conn.source_node_id)
                    queue.append(conn.source_node_id)
        return ancestors

    def _get_descendant_node_ids(self, node_id: str) -> set[str]:
        """Get all downstream node IDs (BFS forwards).

        Returns:
            Set of node IDs including the node itself.
        """
        descendants = {node_id}
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for conn in self.get_connections_from(current):
                if conn.target_node_id not in descendants:
                    descendants.add(conn.target_node_id)
                    queue.append(conn.target_node_id)
        return descendants

    def preview_execute(
        self,
        node_id: str,
        file_path: Optional[Path] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        changed_node_id: Optional[str] = None,
    ) -> dict[str, PipelineData]:
        """Execute the pipeline up to the target node for preview.

        When *changed_node_id* is provided and cached results exist for
        upstream nodes, only the changed node and its downstream
        dependents are re-executed.  Upstream nodes reuse their cached
        ``_port_results`` from the previous preview run.

        Args:
            node_id: The node to preview
            file_path: Optional file path to override batch loader (use specific image)
            stop_check: Optional callback that returns True if execution should stop
            changed_node_id: If set, only re-run this node and its dependents

        Returns:
            Port results dict for the target node
        """
        if self._connections:
            return self._preview_execute_dag(node_id, file_path, stop_check, changed_node_id)
        return self._preview_execute_linear(node_id, file_path, stop_check, changed_node_id)

    def _preview_execute_dag(
        self,
        target_node_id: str,
        file_path: Optional[Path],
        stop_check: Optional[Callable[[], bool]],
        changed_node_id: Optional[str] = None,
    ) -> dict[str, PipelineData]:
        """Preview execution in DAG mode — runs only ancestor nodes.

        When *changed_node_id* is set, nodes upstream of the change that
        already have cached ``_port_results`` are skipped.
        """
        ancestor_ids = self._get_ancestor_node_ids(target_node_id)
        ordered = self._topological_sort()
        # Filter to only ancestors of the target
        ordered = [n for n in ordered if n.node_id in ancestor_ids]

        # Determine which nodes must be re-executed
        if changed_node_id is not None:
            needs_rerun = self._get_descendant_node_ids(changed_node_id) & ancestor_ids
        else:
            needs_rerun = ancestor_ids

        # Identify batch loader nodes among ancestors (must have outgoing
        # connections — disconnected loaders are ignored)
        nodes_with_incoming = {c.target_node_id for c in self._connections}
        nodes_with_outgoing = {c.source_node_id for c in self._connections}
        batch_loader_ids = set()
        for node in ordered:
            if (node.node_id not in nodes_with_incoming
                    and node.node_id in nodes_with_outgoing):
                if getattr(node.plugin, "is_batch_source", False):
                    batch_loader_ids.add(node.node_id)

        no_op = lambda *_args: None
        source_folder = None

        # Scan cached results for source folder (from previous runs)
        for nid, results in self._port_results.items():
            for val in results.values():
                if isinstance(val, ImageContainer) and val.metadata.source_path:
                    source_folder = val.metadata.source_path.parent
                    break
            if source_folder:
                break

        for node in ordered:
            if stop_check and stop_check():
                return {}

            # Skip nodes with valid cache that don't need re-execution
            if (node.node_id not in needs_rerun
                    and node.node_id in self._port_results):
                continue

            node.plugin.set_all_parameters(node.parameters)
            node.plugin._pipeline_source_folder = source_folder

            try:
                if node.node_id in batch_loader_ids:
                    # Batch loader: load specific file or first file
                    if file_path is not None:
                        loaded = node.plugin.load_image(file_path, no_op)
                    else:
                        files = node.plugin.get_image_files()
                        if not files:
                            raise RuntimeError(
                                f"{node.plugin.name}: No files found"
                            )
                        loaded = node.plugin.load_image(files[0], no_op)
                    outputs = {"image_out": loaded}
                    for port in node.plugin.get_output_ports():
                        if port.name not in outputs:
                            outputs[port.name] = loaded
                else:
                    # Normal node: gather inputs from upstream
                    inputs: dict[str, PipelineData] = {}
                    for conn in self.get_connections_to(node.node_id):
                        upstream = self._port_results.get(
                            conn.source_node_id, {}
                        )
                        data = upstream.get(conn.source_port)
                        if data is not None:
                            inputs[conn.target_port] = data

                    outputs = self._execute_node_with_auto_iteration(node, inputs, no_op)

                self._port_results[node.node_id] = outputs or {}

                # Store preview result and track source folder
                preview = None
                for val in (outputs or {}).values():
                    if isinstance(val, ImageContainer):
                        if preview is None:
                            preview = val
                        if val.metadata.source_path and source_folder is None:
                            source_folder = val.metadata.source_path.parent
                if preview is None and outputs:
                    preview = next(iter(outputs.values()), None)
                self._last_results[node.node_id] = preview

            except Exception as e:
                raise RuntimeError(
                    f"Node ({node.plugin.name}) failed: {e}"
                )

        return self._port_results.get(target_node_id, {})

    def _preview_execute_linear(
        self,
        target_node_id: str,
        file_path: Optional[Path],
        stop_check: Optional[Callable[[], bool]],
        changed_node_id: Optional[str] = None,
    ) -> dict[str, PipelineData]:
        """Preview execution in linear mode — runs nodes[0..target].

        When *changed_node_id* is set, nodes before the changed node that
        already have cached ``_port_results`` are skipped and their
        outputs are replayed to reconstruct the pipeline state.
        """
        # Find target index
        target_index = None
        for i, node in enumerate(self._nodes):
            if node.node_id == target_node_id:
                target_index = i
                break
        if target_index is None:
            return {}

        # Determine where to start executing based on cache
        start_index = 0
        if changed_node_id is not None:
            for i, node in enumerate(self._nodes[: target_index + 1]):
                if node.node_id == changed_node_id:
                    # Check all nodes before this one have cached results
                    all_cached = all(
                        self._nodes[j].node_id in self._port_results
                        for j in range(i)
                    )
                    if all_cached:
                        start_index = i
                    break

        no_op = lambda *_args: None
        current_image: Optional[ImageContainer] = None
        previous_outputs: dict[str, PipelineData] = {}

        # Replay cached state for skipped nodes
        if start_index > 0:
            for j in range(start_index):
                cached = self._port_results.get(self._nodes[j].node_id, {})
                img = cached.get("image_out")
                if isinstance(img, ImageContainer):
                    current_image = img
                previous_outputs = cached

        for i, node in enumerate(self._nodes[: target_index + 1]):
            if i < start_index:
                continue

            if stop_check and stop_check():
                return {}

            node.plugin.set_all_parameters(node.parameters)

            try:
                # Check if this is a batch loader
                if getattr(node.plugin, "is_batch_source", False):
                    if file_path is not None:
                        current_image = node.plugin.load_image(
                            file_path, no_op
                        )
                    else:
                        files = node.plugin.get_image_files()
                        if not files:
                            raise RuntimeError(
                                f"{node.plugin.name}: No files found"
                            )
                        current_image = node.plugin.load_image(
                            files[0], no_op
                        )
                    self._last_results[node.node_id] = current_image
                    self._port_results[node.node_id] = {
                        "image_out": current_image
                    }
                    continue

                if node.plugin.ports:
                    inputs: dict[str, PipelineData] = {}
                    if current_image is not None:
                        for port in node.plugin.get_input_ports():
                            if issubclass(ImageContainer, port.data_type):
                                inputs[port.name] = current_image
                                break

                    for port in node.plugin.get_input_ports():
                        if port.name not in inputs:
                            for out_data in previous_outputs.values():
                                if out_data is not None and isinstance(
                                    out_data, port.data_type
                                ):
                                    inputs[port.name] = out_data
                                    break

                    outputs = self._execute_node_with_auto_iteration(node, inputs, no_op)
                    self._port_results[node.node_id] = outputs or {}
                    previous_outputs = outputs or {}

                    new_image = (outputs or {}).get("image_out")
                    if isinstance(new_image, ImageContainer):
                        current_image = new_image

                    preview = None
                    for val in (outputs or {}).values():
                        if isinstance(val, ImageContainer):
                            preview = val
                            break
                    self._last_results[node.node_id] = (
                        preview if preview is not None else current_image
                    )
                else:
                    if current_image is not None:
                        is_valid, error = node.plugin.validate_input(
                            current_image
                        )
                        if not is_valid:
                            raise RuntimeError(
                                f"Node {i} ({node.plugin.name}): {error}"
                            )

                    inputs = {}
                    if current_image is not None:
                        inputs["image_in"] = current_image
                    outputs = self._execute_node_with_auto_iteration(node, inputs, no_op)
                    new_image = (outputs or {}).get("image_out")
                    if isinstance(new_image, ImageContainer):
                        current_image = new_image
                    self._last_results[node.node_id] = current_image

            except Exception as e:
                raise RuntimeError(
                    f"Node {i} ({node.plugin.name}) failed: {e}"
                )

        return self._port_results.get(target_node_id, {})

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def is_batch_pipeline(self) -> bool:
        """Check if the pipeline has a batch source.

        In DAG mode, checks connected root nodes (no incoming connections
        but at least one outgoing connection — disconnected nodes are
        ignored).  In linear mode, checks the first node.
        """
        if self._connections:
            # DAG mode: check connected root nodes
            nodes_with_incoming = {c.target_node_id for c in self._connections}
            nodes_with_outgoing = {c.source_node_id for c in self._connections}
            for node in self._nodes:
                if (node.node_id not in nodes_with_incoming
                        and node.node_id in nodes_with_outgoing):
                    if getattr(node.plugin, "is_batch_source", False):
                        return True
            return False

        # Linear mode
        if not self._nodes:
            return False
        first_plugin = self._nodes[0].plugin
        return getattr(first_plugin, "is_batch_source", False)

    def get_batch_files(self) -> list[Path]:
        """Get the list of files for batch processing."""
        if not self.is_batch_pipeline():
            return []

        if self._connections:
            nodes_with_incoming = {c.target_node_id for c in self._connections}
            nodes_with_outgoing = {c.source_node_id for c in self._connections}
            for node in self._nodes:
                if (node.node_id not in nodes_with_incoming
                        and node.node_id in nodes_with_outgoing):
                    if getattr(node.plugin, "is_batch_source", False):
                        node.plugin.set_all_parameters(node.parameters)
                        return node.plugin.get_image_files()
            return []

        first_plugin = self._nodes[0].plugin
        first_plugin.set_all_parameters(self._nodes[0].parameters)
        return first_plugin.get_image_files()

    def execute_batch(
        self,
        progress_callback: Callable[[int, int, int, int, float, str], None],
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> list[PipelineData]:
        """Execute the pipeline for all files in batch mode.

        Calls ``batch_initialize()`` on all nodes before the loop and
        ``batch_finalize()`` on all nodes after completion.

        Args:
            progress_callback: Callback receiving (file_index, total_files,
                             node_index, total_nodes, node_progress, filename)
            stop_check: Optional callback that returns True if execution should stop

        Returns:
            List of processed results
        """
        if not self._nodes:
            return []

        files = self.get_batch_files()
        if not files:
            raise RuntimeError("No files found for batch processing")

        # Batch lifecycle: initialize all nodes
        for node in self._nodes:
            node.plugin.set_all_parameters(node.parameters)
            node.plugin.batch_initialize()

        try:
            if self._connections:
                return self._execute_batch_dag(
                    files, progress_callback, stop_check
                )
            else:
                return self._execute_batch_linear(
                    files, progress_callback, stop_check
                )
        finally:
            # Batch lifecycle: finalize all nodes
            for node in self._nodes:
                node.plugin.batch_finalize()

    def _execute_batch_linear(
        self,
        files: list[Path],
        progress_callback: Callable[[int, int, int, int, float, str], None],
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> list[PipelineData]:
        """Batch execution in linear mode."""
        results = []
        total_files = len(files)
        first_node = self._nodes[0]
        batch_loader = first_node.plugin

        for file_idx, file_path in enumerate(files):
            if stop_check and stop_check():
                break

            filename = file_path.name

            # Load the current file
            def file_progress(progress: float, fi=file_idx, fn=filename):
                progress_callback(fi, total_files, 0, len(self._nodes), progress, fn)

            batch_loader.set_all_parameters(first_node.parameters)
            current_image = batch_loader.load_image(file_path, file_progress)
            self._last_results[first_node.node_id] = current_image

            # Process through remaining nodes
            previous_outputs: dict[str, PipelineData] = {}
            for node_idx, node in enumerate(self._nodes[1:], start=1):
                if stop_check and stop_check():
                    return results

                node.plugin.set_all_parameters(node.parameters)

                def node_progress(progress: float, fi=file_idx, ni=node_idx, fn=filename):
                    progress_callback(fi, total_files, ni, len(self._nodes), progress, fn)

                try:
                    node_progress(0.0)

                    if node.plugin.ports:
                        # v2.0 node: use process_ports
                        inputs: dict[str, PipelineData] = {}
                        if current_image is not None:
                            for port in node.plugin.get_input_ports():
                                if issubclass(ImageContainer, port.data_type):
                                    inputs[port.name] = current_image
                                    break

                        # Route compatible data from previous node
                        for port in node.plugin.get_input_ports():
                            if port.name not in inputs:
                                for out_data in previous_outputs.values():
                                    if out_data is not None and isinstance(out_data, port.data_type):
                                        inputs[port.name] = out_data
                                        break

                        outputs = self._execute_node_with_auto_iteration(node, inputs, node_progress)
                        self._port_results[node.node_id] = outputs or {}
                        previous_outputs = outputs or {}

                        new_image = (outputs or {}).get("image_out")
                        if isinstance(new_image, ImageContainer):
                            current_image = new_image

                        preview = None
                        for val in (outputs or {}).values():
                            if isinstance(val, ImageContainer):
                                preview = val
                                break
                        self._last_results[node.node_id] = preview if preview is not None else current_image
                    else:
                        # Legacy node: route through auto-iteration wrapper
                        if current_image is not None:
                            is_valid, error = node.plugin.validate_input(current_image)
                            if not is_valid:
                                raise RuntimeError(f"Node {node_idx} ({node.plugin.name}): {error}")

                        inputs = {}
                        if current_image is not None:
                            inputs["image_in"] = current_image
                        outputs = self._execute_node_with_auto_iteration(node, inputs, node_progress)
                        new_image = (outputs or {}).get("image_out")
                        if isinstance(new_image, ImageContainer):
                            current_image = new_image
                        self._last_results[node.node_id] = current_image

                    node_progress(1.0)
                except Exception as e:
                    raise RuntimeError(f"Node {node_idx} ({node.plugin.name}) failed on {filename}: {e}")

            results.append(current_image)

        return results

    def _execute_batch_dag(
        self,
        files: list[Path],
        progress_callback: Callable[[int, int, int, int, float, str], None],
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> list[PipelineData]:
        """Batch execution in DAG mode."""
        results = []
        total_files = len(files)

        ordered = self._topological_sort()
        total_nodes = len(ordered)

        # Identify batch loader nodes (connected roots with is_batch_source)
        nodes_with_incoming = {c.target_node_id for c in self._connections}
        nodes_with_outgoing = {c.source_node_id for c in self._connections}
        batch_loaders = []
        for node in ordered:
            if (node.node_id not in nodes_with_incoming
                    and node.node_id in nodes_with_outgoing):
                if getattr(node.plugin, "is_batch_source", False):
                    batch_loaders.append(node)

        for file_idx, file_path in enumerate(files):
            if stop_check and stop_check():
                break

            filename = file_path.name
            self._port_results.clear()
            last_result = None

            for node_idx, node in enumerate(ordered):
                if stop_check and stop_check():
                    return results

                node.plugin.set_all_parameters(node.parameters)

                def node_progress(progress: float, fi=file_idx, ni=node_idx, fn=filename):
                    progress_callback(fi, total_files, ni, total_nodes, progress, fn)

                try:
                    node_progress(0.0)

                    # Batch loaders load the current file
                    if node in batch_loaders:
                        loaded_image = node.plugin.load_image(file_path, node_progress)
                        outputs = {"image_out": loaded_image}
                        # Also populate default port results
                        for port in node.plugin.get_output_ports():
                            if port.name not in outputs:
                                outputs[port.name] = loaded_image
                    else:
                        # Gather inputs from upstream
                        inputs: dict[str, PipelineData] = {}
                        for conn in self.get_connections_to(node.node_id):
                            upstream = self._port_results.get(conn.source_node_id, {})
                            data = upstream.get(conn.source_port)
                            if data is not None:
                                inputs[conn.target_port] = data

                        outputs = self._execute_node_with_auto_iteration(node, inputs, node_progress)

                    node_progress(1.0)
                    self._port_results[node.node_id] = outputs or {}

                    # Store preview
                    preview = None
                    for val in (outputs or {}).values():
                        if isinstance(val, ImageContainer):
                            preview = val
                            break
                    if preview is None and outputs:
                        preview = next(iter(outputs.values()), None)

                    self._last_results[node.node_id] = preview
                    if preview is not None:
                        last_result = preview

                except Exception as e:
                    raise RuntimeError(
                        f"Node {node_idx} ({node.plugin.name}) failed on {filename}: {e}"
                    )

            results.append(last_result)

        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize the pipeline to JSON.

        Uses v2.0 format when connections exist, v1.0 otherwise.
        """
        version = "2.0" if self._connections else "1.0"

        data = {
            "version": version,
            "nodes": [],
        }

        for node in self._nodes:
            node_data = {
                "node_id": node.node_id,
                "plugin_name": node.plugin.name,
                "parameters": node.parameters,
                "position": list(node.position),
            }
            if node.port_sides:
                node_data["port_sides"] = node.port_sides
            data["nodes"].append(node_data)

        if self._connections:
            data["connections"] = [c.to_dict() for c in self._connections]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path, registry: PluginRegistry) -> None:
        """Load a pipeline from JSON. Handles both v1.0 and v2.0 formats."""
        with open(path, "r") as f:
            data = json.load(f)

        self.clear()

        for node_data in data.get("nodes", []):
            plugin_name = node_data["plugin_name"]
            plugin = registry.create_instance(plugin_name)

            if plugin is None:
                raise ValueError(f"Unknown plugin: {plugin_name}")

            node = PipelineNode(
                plugin=plugin,
                parameters=node_data.get("parameters", {}),
                position=tuple(node_data.get("position", [0, 0])),
                node_id=node_data.get("node_id", ""),
                port_sides=node_data.get("port_sides", {}),
            )
            self.add_node(node)

        # Load connections (v2.0)
        for conn_data in data.get("connections", []):
            try:
                conn = Connection.from_dict(conn_data)
                self.add_connection(conn)
            except ValueError:
                # Skip invalid connections (e.g. missing plugins)
                pass
