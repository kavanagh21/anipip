"""Plugin library sidebar for browsing available plugins."""

from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QLineEdit,
    QAbstractItemView,
)

from core.plugin_registry import PluginRegistry


class PluginBrowser(QWidget):
    """Sidebar widget for browsing and dragging plugins onto the canvas."""

    def __init__(self, registry: PluginRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry

        self._setup_ui()
        self._populate_plugins()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QLabel("Plugins")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search plugins...")
        self.search_box.textChanged.connect(self._filter_plugins)
        layout.addWidget(self.search_box)

        # Plugin tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree)

        # Override drag behavior
        self.tree.startDrag = self._start_drag

    def _populate_plugins(self) -> None:
        """Populate the tree with available plugins."""
        self.tree.clear()

        # Group plugins by category
        categories = self.registry.get_categories()

        for category in sorted(categories):
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(
                category_item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled
            )
            self.tree.addTopLevelItem(category_item)

            plugins = self.registry.get_plugins_by_category(category)
            for plugin_class in plugins:
                plugin_item = QTreeWidgetItem([plugin_class.name])
                plugin_item.setToolTip(0, plugin_class.description)
                plugin_item.setData(0, Qt.ItemDataRole.UserRole, plugin_class.name)
                category_item.addChild(plugin_item)

            category_item.setExpanded(True)

    def _filter_plugins(self, text: str) -> None:
        """Filter visible plugins based on search text."""
        text = text.lower()

        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            category_visible = False

            for j in range(category_item.childCount()):
                plugin_item = category_item.child(j)
                plugin_name = plugin_item.text(0).lower()
                visible = text in plugin_name or not text
                plugin_item.setHidden(not visible)

                if visible:
                    category_visible = True

            category_item.setHidden(not category_visible)

    def _start_drag(self, supported_actions) -> None:
        """Start a drag operation with the selected plugin."""
        item = self.tree.currentItem()
        if item is None or item.parent() is None:
            return  # Don't drag category items

        plugin_name = item.data(0, Qt.ItemDataRole.UserRole)
        if not plugin_name:
            return

        drag = QDrag(self.tree)
        mime_data = QMimeData()
        mime_data.setText(plugin_name)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.CopyAction)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle double-click on a plugin item."""
        if item.parent() is None:
            return  # Ignore category items

        # Signal that a plugin should be added (handled by main window)
        plugin_name = item.data(0, Qt.ItemDataRole.UserRole)
        if plugin_name:
            # Find parent main window and add the node
            parent = self.parent()
            while parent:
                if hasattr(parent, 'canvas'):
                    parent.canvas.add_node(plugin_name)
                    break
                parent = parent.parent()

    def refresh(self) -> None:
        """Refresh the plugin list."""
        self._populate_plugins()
