"""Visual representation of a pipeline node with multi-port support."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QLinearGradient,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QStyleOptionGraphicsItem,
    QWidget,
    QMenu,
)

from core.pipeline import PipelineNode
from core.ports import Port, PortDirection, PortSide
from core.image_container import ImageContainer
from core.table_data import TableData


class NodeStatus(Enum):
    """Status of a pipeline node."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


# Map data types to port colours
PORT_TYPE_COLORS = {
    ImageContainer: QColor(80, 180, 80),   # green
    TableData: QColor(80, 130, 220),       # blue
}
DEFAULT_PORT_COLOR = QColor(180, 180, 180)  # grey fallback


@dataclass
class PortItem:
    """Metadata for a rendered port on the node widget."""

    port: Port
    local_center: QPointF  # Position relative to the node's top-left
    side: PortSide = PortSide.LEFT


class NodeWidget(QGraphicsRectItem):
    """Visual representation of a pipeline step.

    Displays the node with name, category, preview thumbnail, status
    indicator, and typed connection ports.
    """

    NODE_WIDTH = 160
    MIN_NODE_HEIGHT = 80
    PORT_SPACING = 20
    PORT_HORIZ_SPACING = 24
    PORT_RADIUS = 6
    PORT_Y_START = 50  # vertical offset for first port
    CORNER_RADIUS = 8

    # Colors for different statuses
    STATUS_COLORS = {
        NodeStatus.PENDING: QColor(200, 200, 200),
        NodeStatus.PROCESSING: QColor(100, 180, 255),
        NodeStatus.COMPLETE: QColor(100, 200, 100),
        NodeStatus.ERROR: QColor(255, 100, 100),
    }

    def __init__(self, pipeline_node: PipelineNode, parent=None):
        super().__init__(parent)
        self.pipeline_node = pipeline_node
        self._status = NodeStatus.PENDING
        self._selected = False
        self._preview_pixmap: Optional[QPixmap] = None
        self._progress = 0.0
        self._top_port_count = 0
        self._bottom_port_count = 0

        # Build port items from plugin's port list
        self._port_items: list[PortItem] = []
        self._calculate_dimensions()

        # Setup the item
        self.setRect(0, 0, self.NODE_WIDTH, self.NODE_HEIGHT)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Set initial position from pipeline node
        self.setPos(*pipeline_node.position)

    # ------------------------------------------------------------------
    # Port helpers
    # ------------------------------------------------------------------

    def _get_port_side(self, port: Port) -> PortSide:
        """Determine which side a port should be on."""
        side_str = self.pipeline_node.port_sides.get(port.name)
        if side_str:
            try:
                return PortSide(side_str)
            except ValueError:
                pass
        # Default: inputs left, outputs right
        if port.direction == PortDirection.INPUT:
            return PortSide.LEFT
        return PortSide.RIGHT

    def _calculate_dimensions(self) -> None:
        """Calculate node dimensions and build port items."""
        all_ports = self.pipeline_node.plugin.get_ports()

        # Group ports by side
        sides: dict[PortSide, list[Port]] = {s: [] for s in PortSide}
        for port in all_ports:
            sides[self._get_port_side(port)].append(port)

        self._top_port_count = len(sides[PortSide.TOP])
        self._bottom_port_count = len(sides[PortSide.BOTTOM])

        # Vertical offset when top ports exist
        top_offset = 20 if self._top_port_count > 0 else 0
        y_start = self.PORT_Y_START + top_offset

        # Height from left/right port counts
        max_side_ports = max(len(sides[PortSide.LEFT]), len(sides[PortSide.RIGHT]), 1)
        self.NODE_HEIGHT = max(
            self.MIN_NODE_HEIGHT + top_offset,
            y_start + max_side_ports * self.PORT_SPACING + 10,
        )
        if self._bottom_port_count > 0:
            self.NODE_HEIGHT += 20

        # Width: expand if top/bottom ports need more space
        top_bottom_max = max(self._top_port_count, self._bottom_port_count)
        min_width_for_horiz = top_bottom_max * self.PORT_HORIZ_SPACING + 20
        self.NODE_WIDTH = max(160, min_width_for_horiz)

        # Build port items
        self._port_items.clear()

        # Left ports
        for idx, port in enumerate(sides[PortSide.LEFT]):
            y = y_start + idx * self.PORT_SPACING
            self._port_items.append(
                PortItem(port=port, local_center=QPointF(0, y), side=PortSide.LEFT)
            )

        # Right ports
        for idx, port in enumerate(sides[PortSide.RIGHT]):
            y = y_start + idx * self.PORT_SPACING
            self._port_items.append(
                PortItem(port=port, local_center=QPointF(self.NODE_WIDTH, y), side=PortSide.RIGHT)
            )

        # Top ports — distributed horizontally
        if self._top_port_count > 0:
            spacing = self.NODE_WIDTH / (self._top_port_count + 1)
            for idx, port in enumerate(sides[PortSide.TOP]):
                x = spacing * (idx + 1)
                self._port_items.append(
                    PortItem(port=port, local_center=QPointF(x, 0), side=PortSide.TOP)
                )

        # Bottom ports — distributed horizontally
        if self._bottom_port_count > 0:
            spacing = self.NODE_WIDTH / (self._bottom_port_count + 1)
            for idx, port in enumerate(sides[PortSide.BOTTOM]):
                x = spacing * (idx + 1)
                self._port_items.append(
                    PortItem(port=port, local_center=QPointF(x, self.NODE_HEIGHT), side=PortSide.BOTTOM)
                )

    def port_at(self, local_pos: QPointF) -> Optional[PortItem]:
        """Hit-test: return the PortItem at *local_pos*, or None."""
        hit_radius = self.PORT_RADIUS + 4  # generous hit area
        for pi in self._port_items:
            dx = local_pos.x() - pi.local_center.x()
            dy = local_pos.y() - pi.local_center.y()
            if dx * dx + dy * dy <= hit_radius * hit_radius:
                return pi
        return None

    def get_port_scene_pos(self, port_name: str, direction: PortDirection) -> Optional[QPointF]:
        """Get the scene-space position of a named port."""
        for pi in self._port_items:
            if pi.port.name == port_name and pi.port.direction == direction:
                return self.mapToScene(pi.local_center)
        return None

    @property
    def port_items(self) -> list[PortItem]:
        return list(self._port_items)

    # ------------------------------------------------------------------
    # Legacy single-port access (backwards compat for auto-connections)
    # ------------------------------------------------------------------

    def get_input_point(self) -> tuple[float, float]:
        """Get the position of the first input connection point."""
        for pi in self._port_items:
            if pi.port.direction == PortDirection.INPUT:
                sp = self.mapToScene(pi.local_center)
                return sp.x(), sp.y()
        # Fallback
        pos = self.scenePos()
        return pos.x(), pos.y() + self.NODE_HEIGHT / 2

    def get_output_point(self) -> tuple[float, float]:
        """Get the position of the first output connection point."""
        for pi in self._port_items:
            if pi.port.direction == PortDirection.OUTPUT:
                sp = self.mapToScene(pi.local_center)
                return sp.x(), sp.y()
        # Fallback
        pos = self.scenePos()
        return pos.x() + self.NODE_WIDTH, pos.y() + self.NODE_HEIGHT / 2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_id(self) -> str:
        """Get the pipeline node ID."""
        return self.pipeline_node.node_id

    @property
    def status(self) -> NodeStatus:
        return self._status

    @status.setter
    def status(self, value: NodeStatus) -> None:
        self._status = value
        self.update()

    @property
    def progress(self) -> float:
        return self._progress

    @progress.setter
    def progress(self, value: float) -> None:
        self._progress = max(0.0, min(1.0, value))
        self.update()

    def set_preview(self, pixmap: QPixmap) -> None:
        """Set a preview thumbnail."""
        self._preview_pixmap = pixmap.scaled(
            40, 40, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Paint the node widget."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background gradient
        gradient = QLinearGradient(0, 0, 0, self.NODE_HEIGHT)
        base_color = self.STATUS_COLORS[self._status]

        if self.isSelected():
            gradient.setColorAt(0, base_color.lighter(120))
            gradient.setColorAt(1, base_color)
            pen_color = QColor(50, 150, 255)
            pen_width = 3
        else:
            gradient.setColorAt(0, base_color.lighter(110))
            gradient.setColorAt(1, base_color.darker(110))
            pen_color = QColor(100, 100, 100)
            pen_width = 1

        # Draw rounded rectangle background
        painter.setPen(QPen(pen_color, pen_width))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(
            self.rect(), self.CORNER_RADIUS, self.CORNER_RADIUS
        )

        # Vertical offset when top ports exist
        top_offset = 20 if self._top_port_count > 0 else 0

        # Draw plugin name
        painter.setPen(QPen(QColor(40, 40, 40)))
        font = QFont("Sans Serif", 10, QFont.Weight.Bold)
        painter.setFont(font)

        name_rect = QRectF(10, 8 + top_offset, self.NODE_WIDTH - 20, 20)
        painter.drawText(
            name_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.pipeline_node.plugin.name,
        )

        # Draw category
        painter.setPen(QPen(QColor(80, 80, 80)))
        font = QFont("Sans Serif", 8)
        painter.setFont(font)

        category_rect = QRectF(10, 28 + top_offset, self.NODE_WIDTH - 20, 16)
        painter.drawText(
            category_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.pipeline_node.plugin.category,
        )

        # Draw preview thumbnail if available
        if self._preview_pixmap:
            preview_x = self.NODE_WIDTH - 50
            preview_y = 10 + top_offset
            painter.drawPixmap(int(preview_x), int(preview_y), self._preview_pixmap)

        # Draw progress bar if processing
        if self._status == NodeStatus.PROCESSING and self._progress > 0:
            progress_rect = QRectF(10, self.NODE_HEIGHT - 18, self.NODE_WIDTH - 20, 8)
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.setBrush(QBrush(QColor(220, 220, 220)))
            painter.drawRoundedRect(progress_rect, 3, 3)

            fill_width = (self.NODE_WIDTH - 20) * self._progress
            fill_rect = QRectF(10, self.NODE_HEIGHT - 18, fill_width, 8)
            painter.setBrush(QBrush(QColor(50, 150, 255)))
            painter.drawRoundedRect(fill_rect, 3, 3)

        # Draw ports
        self._draw_ports(painter)

    def _draw_ports(self, painter: QPainter) -> None:
        """Draw typed port circles with labels."""
        label_font = QFont("Sans Serif", 7)
        painter.setFont(label_font)

        for pi in self._port_items:
            color = PORT_TYPE_COLORS.get(pi.port.data_type, DEFAULT_PORT_COLOR)

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(140), 1))

            cx = pi.local_center.x()
            cy = pi.local_center.y()
            r = self.PORT_RADIUS

            painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

            # Port label — only for left/right sides (top/bottom use tooltips)
            if pi.side in (PortSide.LEFT, PortSide.RIGHT):
                painter.setPen(QPen(QColor(60, 60, 60)))
                if pi.side == PortSide.LEFT:
                    label_rect = QRectF(r + 4, cy - 8, self.NODE_WIDTH / 2 - r - 8, 16)
                    painter.drawText(
                        label_rect,
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                        pi.port.label,
                    )
                else:
                    label_rect = QRectF(
                        self.NODE_WIDTH / 2, cy - 8,
                        self.NODE_WIDTH / 2 - r - 4, 16,
                    )
                    painter.drawText(
                        label_rect,
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                        pi.port.label,
                    )

    # ------------------------------------------------------------------
    # Layout rebuild
    # ------------------------------------------------------------------

    def _rebuild_layout(self) -> None:
        """Recalculate dimensions and port positions after a port side change."""
        self._calculate_dimensions()
        self.setRect(0, 0, self.NODE_WIDTH, self.NODE_HEIGHT)
        self.update()
        # Trigger connection redraw
        if self.scene() and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, '_update_connections'):
                view._update_connections()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def itemChange(self, change, value):
        """Handle item changes like position updates."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            pos = self.scenePos()
            self.pipeline_node.position = (int(pos.x()), int(pos.y()))
            if self.scene() and self.scene().views():
                view = self.scene().views()[0]
                if hasattr(view, '_update_connections'):
                    view._update_connections()

        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        """Show context menu on right-click — port side menu or node menu."""
        local = event.pos()
        pi = self.port_at(local)
        if pi:
            self._show_port_side_menu(pi, event.screenPos())
            return

        menu = QMenu()
        delete_action = menu.addAction("Delete")
        preview_action = menu.addAction("Preview Output")

        action = menu.exec(event.screenPos())

        if action == delete_action:
            if self.scene():
                self.scene().views()[0].node_delete_requested.emit(self.node_id)
        elif action == preview_action:
            if self.scene():
                self.scene().views()[0].node_preview_requested.emit(self.node_id)

    def _show_port_side_menu(self, pi: PortItem, screen_pos) -> None:
        """Show a context menu to change which side a port is on."""
        menu = QMenu()
        menu.setTitle(f"Move '{pi.port.label}'")

        current_side = pi.side
        for side in PortSide:
            action = menu.addAction(side.value.capitalize())
            action.setCheckable(True)
            action.setChecked(side == current_side)
            action.setData(side)

        chosen = menu.exec(screen_pos)
        if chosen and chosen.data() != current_side:
            new_side: PortSide = chosen.data()
            # Determine if this is the default side for this port
            default_side = PortSide.LEFT if pi.port.direction == PortDirection.INPUT else PortSide.RIGHT
            if new_side == default_side:
                # Remove override — use default
                self.pipeline_node.port_sides.pop(pi.port.name, None)
            else:
                self.pipeline_node.port_sides[pi.port.name] = new_side.value
            self._rebuild_layout()
            # Signal pipeline changed
            if self.scene() and self.scene().views():
                view = self.scene().views()[0]
                if hasattr(view, 'pipeline_changed'):
                    view.pipeline_changed.emit()
