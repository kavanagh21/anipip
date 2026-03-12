"""Visual pipeline editor canvas with interactive connection support."""

from typing import Optional

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
    QDragEnterEvent,
    QDropEvent,
    QPainterPath,
)
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPathItem,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QWidget,
)

from core.pipeline import Pipeline, PipelineNode
from core.pipeline_data import PipelineData
from core.ports import Connection, PortDirection, PortSide
from core.plugin_registry import PluginRegistry
from core.image_container import ImageContainer
from core.table_data import TableData
from .node_widget import NodeWidget, NodeStatus, PortItem, PORT_TYPE_COLORS, DEFAULT_PORT_COLOR


# Connection line colours keyed by data type
CONNECTION_TYPE_COLORS = {
    ImageContainer: QColor(80, 180, 80),
    TableData: QColor(80, 130, 220),
}
DEFAULT_CONNECTION_COLOR = QColor(120, 120, 120)


class NodeCanvas(QGraphicsView):
    """Canvas for visual pipeline editing.

    Supports drag-and-drop from plugin browser, node selection,
    interactive port-to-port connection creation, and explicit
    connection rendering from the pipeline's connection list.
    """

    node_selected = pyqtSignal(str)
    node_delete_requested = pyqtSignal(str)
    node_preview_requested = pyqtSignal(str)
    pipeline_changed = pyqtSignal()

    GRID_SIZE = 20
    GRID_COLOR = QColor(230, 230, 230)

    def __init__(self, pipeline: Pipeline, registry: PluginRegistry, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.registry = registry
        self._node_widgets: dict[str, NodeWidget] = {}
        self._connection_items: list[QGraphicsPathItem] = []

        # Interactive connection dragging state
        self._dragging_connection = False
        self._drag_source_widget: Optional[NodeWidget] = None
        self._drag_source_port: Optional[PortItem] = None
        self._drag_temp_item: Optional[QGraphicsPathItem] = None
        self._drag_end_pos = QPointF()

        # Setup scene
        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(-2000, -2000, 4000, 4000)
        self.setScene(self._scene)

        # Configure view
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Connect signals
        self._scene.selectionChanged.connect(self._on_selection_changed)
        self.node_delete_requested.connect(self._delete_node)

        # Background color
        self.setBackgroundBrush(QBrush(QColor(250, 250, 250)))

        # Zoom state
        self._zoom_level = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 3.0
        self._zoom_step = 1.25

        self._setup_zoom_controls()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _setup_zoom_controls(self) -> None:
        """Create floating zoom buttons in the bottom-right corner."""
        self._zoom_bar = QWidget(self)
        layout = QHBoxLayout(self._zoom_bar)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        btn_style = (
            "QPushButton { background: #fff; border: 1px solid #bbb;"
            " border-radius: 3px; font-weight: bold; min-width: 24px;"
            " min-height: 24px; }"
            "QPushButton:hover { background: #e8e8e8; }"
            "QPushButton:pressed { background: #d0d0d0; }"
        )

        btn_minus = QPushButton("\u2212")  # minus sign
        btn_minus.setToolTip("Zoom Out")
        btn_minus.setStyleSheet(btn_style)
        btn_minus.clicked.connect(self.zoom_out)
        layout.addWidget(btn_minus)

        self._zoom_label = QPushButton("100%")
        self._zoom_label.setToolTip("Reset Zoom")
        self._zoom_label.setStyleSheet(btn_style)
        self._zoom_label.clicked.connect(self.zoom_reset)
        layout.addWidget(self._zoom_label)

        btn_plus = QPushButton("+")
        btn_plus.setToolTip("Zoom In")
        btn_plus.setStyleSheet(btn_style)
        btn_plus.clicked.connect(self.zoom_in)
        layout.addWidget(btn_plus)

        self._zoom_bar.adjustSize()

    def resizeEvent(self, event) -> None:
        """Reposition zoom bar on resize."""
        super().resizeEvent(event)
        bar = self._zoom_bar
        bar.move(self.width() - bar.width() - 8, self.height() - bar.height() - 8)

    def _apply_zoom(self, new_level: float) -> None:
        """Apply a new zoom level, clamped to min/max."""
        new_level = max(self._zoom_min, min(self._zoom_max, new_level))
        factor = new_level / self._zoom_level
        self._zoom_level = new_level
        self.scale(factor, factor)
        self._zoom_label.setText(f"{round(self._zoom_level * 100)}%")

    def zoom_in(self) -> None:
        self._apply_zoom(self._zoom_level * self._zoom_step)

    def zoom_out(self) -> None:
        self._apply_zoom(self._zoom_level / self._zoom_step)

    def zoom_reset(self) -> None:
        self._apply_zoom(1.0)

    def wheelEvent(self, event) -> None:
        """Zoom with Ctrl+scroll wheel."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect) -> None:
        """Draw grid background."""
        super().drawBackground(painter, rect)

        pen = QPen(self.GRID_COLOR, 1)
        painter.setPen(pen)

        left = int(rect.left()) - (int(rect.left()) % self.GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % self.GRID_SIZE)

        lines = []
        x = left
        while x < rect.right():
            lines.append((QPointF(x, rect.top()), QPointF(x, rect.bottom())))
            x += self.GRID_SIZE

        y = top
        while y < rect.bottom():
            lines.append((QPointF(rect.left(), y), QPointF(rect.right(), y)))
            y += self.GRID_SIZE

        for start, end in lines:
            painter.drawLine(start, end)

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, plugin_name: str, position: tuple[int, int] = None) -> Optional[str]:
        """Add a new node to the canvas."""
        plugin = self.registry.create_instance(plugin_name)
        if plugin is None:
            return None

        if position is None:
            position = self._calculate_next_position()

        pipeline_node = PipelineNode(plugin=plugin, position=position)
        self.pipeline.add_node(pipeline_node)

        node_widget = NodeWidget(pipeline_node)
        self._scene.addItem(node_widget)
        self._node_widgets[pipeline_node.node_id] = node_widget

        self._update_connections()
        self.pipeline_changed.emit()
        return pipeline_node.node_id

    def _calculate_next_position(self) -> tuple[int, int]:
        """Calculate position for a new node."""
        if not self._node_widgets:
            return (100, 100)

        rightmost_x = 0
        y = 100
        for widget in self._node_widgets.values():
            pos = widget.scenePos()
            if pos.x() > rightmost_x:
                rightmost_x = pos.x()
                y = pos.y()

        return (int(rightmost_x) + NodeWidget.NODE_WIDTH + 60, int(y))

    def _delete_node(self, node_id: str) -> None:
        """Delete a node from the canvas and pipeline."""
        if node_id in self._node_widgets:
            widget = self._node_widgets.pop(node_id)
            self._scene.removeItem(widget)
            self.pipeline.remove_node(node_id)
            self._update_connections()
            self.pipeline_changed.emit()

    # ------------------------------------------------------------------
    # Connection drawing
    # ------------------------------------------------------------------

    def _update_connections(self) -> None:
        """Redraw all connections."""
        # Remove old connection graphics
        for item in self._connection_items:
            self._scene.removeItem(item)
        self._connection_items.clear()

        if self.pipeline.connections:
            self._draw_explicit_connections()

    def _get_port_side(self, node_id: str, port_name: str, direction: PortDirection) -> PortSide:
        """Return the PortSide for a port, consulting node overrides."""
        widget = self._node_widgets.get(node_id)
        if widget:
            for pi in widget.port_items:
                if pi.port.name == port_name and pi.port.direction == direction:
                    return pi.side
        # Default
        return PortSide.RIGHT if direction == PortDirection.OUTPUT else PortSide.LEFT

    def _draw_explicit_connections(self) -> None:
        """Draw connections from the pipeline's connection list."""
        for conn in self.pipeline.connections:
            src_widget = self._node_widgets.get(conn.source_node_id)
            tgt_widget = self._node_widgets.get(conn.target_node_id)
            if not src_widget or not tgt_widget:
                continue

            start = src_widget.get_port_scene_pos(conn.source_port, PortDirection.OUTPUT)
            end = tgt_widget.get_port_scene_pos(conn.target_port, PortDirection.INPUT)
            if start is None or end is None:
                continue

            # Determine colour from source port data type
            color = DEFAULT_CONNECTION_COLOR
            for pi in src_widget.port_items:
                if pi.port.name == conn.source_port:
                    color = CONNECTION_TYPE_COLORS.get(pi.port.data_type, DEFAULT_CONNECTION_COLOR)
                    break

            start_side = self._get_port_side(conn.source_node_id, conn.source_port, PortDirection.OUTPUT)
            end_side = self._get_port_side(conn.target_node_id, conn.target_port, PortDirection.INPUT)

            path_item = self._make_bezier(start, end, color, start_side=start_side, end_side=end_side)
            # Store connection data on the item for right-click deletion
            path_item.setData(0, conn)
            self._scene.addItem(path_item)
            self._connection_items.append(path_item)

    def _draw_linear_connections(self) -> None:
        """Draw connections between consecutive nodes in list order.

        For v2.0 nodes with explicit ports, draws coloured lines between
        compatible port pairs.  Falls back to a single grey line for
        legacy nodes.
        """
        ordered = []
        for node in self.pipeline.nodes:
            if node.node_id in self._node_widgets:
                ordered.append((node, self._node_widgets[node.node_id]))

        for i in range(len(ordered) - 1):
            src_node, src_widget = ordered[i]
            tgt_node, tgt_widget = ordered[i + 1]

            # Try port-to-port matching when both nodes have explicit ports
            if src_node.plugin.ports and tgt_node.plugin.ports:
                matched = False
                for out_port in src_node.plugin.get_output_ports():
                    for in_port in tgt_node.plugin.get_input_ports():
                        if out_port.is_compatible(in_port):
                            start = src_widget.get_port_scene_pos(
                                out_port.name, PortDirection.OUTPUT)
                            end = tgt_widget.get_port_scene_pos(
                                in_port.name, PortDirection.INPUT)
                            if start and end:
                                color = CONNECTION_TYPE_COLORS.get(
                                    out_port.data_type, DEFAULT_CONNECTION_COLOR)
                                item = self._make_bezier(start, end, color)
                                self._scene.addItem(item)
                                self._connection_items.append(item)
                                matched = True
                if matched:
                    continue

            # Fallback: generic grey line
            sx, sy = src_widget.get_output_point()
            ex, ey = tgt_widget.get_input_point()
            path_item = self._make_bezier(
                QPointF(sx, sy), QPointF(ex, ey),
                QColor(100, 100, 100),
            )
            self._scene.addItem(path_item)
            self._connection_items.append(path_item)

    @staticmethod
    def _make_bezier(
        start: QPointF,
        end: QPointF,
        color: QColor,
        dashed: bool = False,
        start_side: PortSide = PortSide.RIGHT,
        end_side: PortSide = PortSide.LEFT,
    ) -> QGraphicsPathItem:
        """Create a bezier curve path item between two points.

        Control points face outward from the port side for natural routing.
        """
        path = QPainterPath()
        path.moveTo(start)

        ctrl_offset = min(max(abs(end.x() - start.x()), abs(end.y() - start.y())) / 2, 100)

        # Start control point faces outward from start_side
        if start_side == PortSide.RIGHT:
            sc = QPointF(start.x() + ctrl_offset, start.y())
        elif start_side == PortSide.LEFT:
            sc = QPointF(start.x() - ctrl_offset, start.y())
        elif start_side == PortSide.TOP:
            sc = QPointF(start.x(), start.y() - ctrl_offset)
        else:  # BOTTOM
            sc = QPointF(start.x(), start.y() + ctrl_offset)

        # End control point faces outward from end_side
        if end_side == PortSide.LEFT:
            ec = QPointF(end.x() - ctrl_offset, end.y())
        elif end_side == PortSide.RIGHT:
            ec = QPointF(end.x() + ctrl_offset, end.y())
        elif end_side == PortSide.TOP:
            ec = QPointF(end.x(), end.y() - ctrl_offset)
        else:  # BOTTOM
            ec = QPointF(end.x(), end.y() + ctrl_offset)

        path.cubicTo(sc.x(), sc.y(), ec.x(), ec.y(), end.x(), end.y())

        item = QGraphicsPathItem(path)
        pen = QPen(color, 2)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)
        item.setZValue(-1)
        return item

    # ------------------------------------------------------------------
    # Interactive connection creation (drag from output port to input port)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        """Start a connection drag if the user clicks on an output port."""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            for widget in self._node_widgets.values():
                local = widget.mapFromScene(scene_pos)
                pi = widget.port_at(local)
                if pi and pi.port.direction == PortDirection.OUTPUT:
                    self._dragging_connection = True
                    self._drag_source_widget = widget
                    self._drag_source_port = pi
                    self._drag_end_pos = scene_pos

                    # Create temporary dashed line
                    start = widget.mapToScene(pi.local_center)
                    color = PORT_TYPE_COLORS.get(pi.port.data_type, DEFAULT_PORT_COLOR)
                    self._drag_temp_item = self._make_bezier(
                        start, scene_pos, color, dashed=True, start_side=pi.side,
                    )
                    self._scene.addItem(self._drag_temp_item)

                    # Highlight compatible input ports
                    self._highlight_compatible_ports(pi)
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Update the temporary connection line while dragging."""
        if self._dragging_connection and self._drag_source_port:
            scene_pos = self.mapToScene(event.pos())
            self._drag_end_pos = scene_pos

            # Redraw temp line
            if self._drag_temp_item:
                self._scene.removeItem(self._drag_temp_item)

            start = self._drag_source_widget.mapToScene(self._drag_source_port.local_center)
            color = PORT_TYPE_COLORS.get(self._drag_source_port.port.data_type, DEFAULT_PORT_COLOR)
            self._drag_temp_item = self._make_bezier(
                start, scene_pos, color, dashed=True, start_side=self._drag_source_port.side,
            )
            self._scene.addItem(self._drag_temp_item)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Complete or cancel the connection drag."""
        if self._dragging_connection:
            # Clean up temp line
            if self._drag_temp_item:
                self._scene.removeItem(self._drag_temp_item)
                self._drag_temp_item = None

            self._clear_port_highlights()

            # Check if we released over a compatible input port
            scene_pos = self.mapToScene(event.pos())
            for widget in self._node_widgets.values():
                local = widget.mapFromScene(scene_pos)
                pi = widget.port_at(local)
                if pi and pi.port.direction == PortDirection.INPUT:
                    # Attempt to create connection
                    conn = Connection(
                        source_node_id=self._drag_source_widget.node_id,
                        source_port=self._drag_source_port.port.name,
                        target_node_id=widget.node_id,
                        target_port=pi.port.name,
                    )
                    try:
                        self.pipeline.add_connection(conn)
                        self._update_connections()
                        self.pipeline_changed.emit()
                    except ValueError:
                        pass  # Invalid connection — silently ignore
                    break

            self._dragging_connection = False
            self._drag_source_widget = None
            self._drag_source_port = None
            return

        super().mouseReleaseEvent(event)

    def _highlight_compatible_ports(self, source_port_item: PortItem) -> None:
        """Highlight input ports compatible with the source output port."""
        # For now this is a no-op visually; in a future iteration we could
        # change the port brush of compatible inputs.  We store compatible
        # info for potential use.
        pass

    def _clear_port_highlights(self) -> None:
        """Remove any port highlights."""
        pass

    # ------------------------------------------------------------------
    # Connection deletion via right-click
    # ------------------------------------------------------------------

    def contextMenuEvent(self, event):
        """Show context menu on right-click — handle connection deletion."""
        scene_pos = self.mapToScene(event.pos())

        # Check if click is near a connection
        for item in self._connection_items:
            if item.shape().contains(scene_pos):
                conn = item.data(0)
                if isinstance(conn, Connection):
                    menu = QMenu(self)
                    delete_action = menu.addAction("Delete Connection")
                    action = menu.exec(event.globalPos())
                    if action == delete_action:
                        self.pipeline.remove_connection(conn)
                        self._update_connections()
                        self.pipeline_changed.emit()
                    return

        super().contextMenuEvent(event)

    # ------------------------------------------------------------------
    # Selection / status helpers
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        """Handle selection changes."""
        selected = self._scene.selectedItems()
        if selected and isinstance(selected[0], NodeWidget):
            self.node_selected.emit(selected[0].node_id)

    def get_selected_node(self) -> Optional[str]:
        selected = self._scene.selectedItems()
        if selected and isinstance(selected[0], NodeWidget):
            return selected[0].node_id
        return None

    def select_node(self, node_id: str) -> None:
        self._scene.clearSelection()
        if node_id in self._node_widgets:
            self._node_widgets[node_id].setSelected(True)

    def set_node_status(self, node_id: str, status: NodeStatus) -> None:
        if node_id in self._node_widgets:
            self._node_widgets[node_id].status = status

    def set_node_progress(self, node_id: str, progress: float) -> None:
        if node_id in self._node_widgets:
            self._node_widgets[node_id].progress = progress

    def reset_node_statuses(self) -> None:
        for widget in self._node_widgets.values():
            widget.status = NodeStatus.PENDING
            widget.progress = 0.0

    def clear(self) -> None:
        for node_id in list(self._node_widgets.keys()):
            self._delete_node(node_id)
        self.pipeline.clear()

    # ------------------------------------------------------------------
    # Drag & drop from plugin browser
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasText():
            plugin_name = event.mimeData().text()
            pos = self.mapToScene(event.position().toPoint())
            self.add_node(plugin_name, (int(pos.x()), int(pos.y())))
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync_from_pipeline(self) -> None:
        """Synchronize visual widgets from pipeline state."""
        node_ids = {node.node_id for node in self.pipeline.nodes}
        for widget_id in list(self._node_widgets.keys()):
            if widget_id not in node_ids:
                widget = self._node_widgets.pop(widget_id)
                self._scene.removeItem(widget)

        for node in self.pipeline.nodes:
            if node.node_id not in self._node_widgets:
                widget = NodeWidget(node)
                self._scene.addItem(widget)
                self._node_widgets[node.node_id] = widget

        self._update_connections()
