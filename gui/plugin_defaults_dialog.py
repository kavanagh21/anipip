"""Dialog for configuring plugin default parameter values."""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QFrame,
    QWidget,
    QDialogButtonBox,
)

from core.plugin_registry import PluginRegistry
from core.settings import PluginSettings
from core.parameters import (
    Parameter,
    IntParameter,
    FloatParameter,
    ChoiceParameter,
    BoolParameter,
    StringParameter,
    FileParameter,
    ActionParameter,
)


class PluginDefaultsDialog(QDialog):
    """Dialog for editing default parameter values for all plugins."""

    def __init__(self, registry: PluginRegistry, settings: PluginSettings, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._settings = settings
        # Working copy of defaults being edited: {plugin_name: {param_name: value}}
        self._edited_defaults: dict[str, dict[str, Any]] = settings.get_all_defaults()
        self._controls: dict[str, QWidget] = {}
        self._current_plugin: str | None = None

        self.setWindowTitle("Plugin Defaults")
        self.resize(650, 450)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Main splitter area
        content = QHBoxLayout()
        layout.addLayout(content, 1)

        # Left: plugin tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMaximumWidth(220)
        self._tree.currentItemChanged.connect(self._on_plugin_selected)
        content.addWidget(self._tree)

        # Right: parameter editor
        right = QVBoxLayout()
        content.addLayout(right, 1)

        self._plugin_label = QLabel("Select a plugin")
        self._plugin_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        right.addWidget(self._plugin_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        right.addWidget(scroll)

        self._params_container = QWidget()
        self._params_layout = QVBoxLayout(self._params_container)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_layout.addStretch()
        scroll.setWidget(self._params_container)

        # Reset button for current plugin
        self._reset_btn = QPushButton("Reset to Built-in Defaults")
        self._reset_btn.clicked.connect(self._reset_current)
        self._reset_btn.setEnabled(False)
        right.addWidget(self._reset_btn)

        # Bottom: OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate_tree()

    def _populate_tree(self) -> None:
        for category in sorted(self._registry.get_categories()):
            cat_item = QTreeWidgetItem([category])
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._tree.addTopLevelItem(cat_item)
            for plugin_class in sorted(
                self._registry.get_plugins_by_category(category),
                key=lambda p: p.name,
            ):
                child = QTreeWidgetItem([plugin_class.name])
                child.setData(0, Qt.ItemDataRole.UserRole, plugin_class.name)
                cat_item.addChild(child)
            cat_item.setExpanded(True)

    def _on_plugin_selected(self, current: QTreeWidgetItem, _previous) -> None:
        if current is None:
            return
        plugin_name = current.data(0, Qt.ItemDataRole.UserRole)
        if plugin_name is None:
            return
        self._show_plugin(plugin_name)

    def _show_plugin(self, plugin_name: str) -> None:
        self._current_plugin = plugin_name
        self._plugin_label.setText(plugin_name)
        self._reset_btn.setEnabled(True)
        self._clear_controls()

        plugin_class = self._registry.get_plugin(plugin_name)
        if plugin_class is None:
            return

        overrides = self._edited_defaults.get(plugin_name, {})

        for param in plugin_class.parameters:
            value = overrides.get(param.name, param.default)
            control = self._create_control(param, value, plugin_name)
            if control is None:
                continue

            group = QWidget()
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(0, 4, 0, 4)

            label = QLabel(param.label)
            label.setStyleSheet("font-weight: 500;")
            group_layout.addWidget(label)
            group_layout.addWidget(control)

            self._params_layout.insertWidget(self._params_layout.count() - 1, group)

    def _clear_controls(self) -> None:
        self._controls.clear()
        while self._params_layout.count() > 1:
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _create_control(self, param: Parameter, value: Any, plugin_name: str) -> QWidget | None:
        if isinstance(param, ActionParameter):
            return None

        if isinstance(param, IntParameter):
            control = QSpinBox()
            control.setMinimum(param.min_value)
            control.setMaximum(param.max_value)
            control.setSingleStep(param.step)
            control.setValue(value if isinstance(value, int) else param.default)
            control.valueChanged.connect(
                lambda v, n=param.name, pn=plugin_name: self._on_value_changed(pn, n, v)
            )
            return control

        elif isinstance(param, FloatParameter):
            control = QDoubleSpinBox()
            control.setMinimum(param.min_value)
            control.setMaximum(param.max_value)
            control.setSingleStep(param.step)
            control.setDecimals(param.decimals)
            control.setValue(value if isinstance(value, (int, float)) else param.default)
            control.valueChanged.connect(
                lambda v, n=param.name, pn=plugin_name: self._on_value_changed(pn, n, v)
            )
            return control

        elif isinstance(param, ChoiceParameter):
            control = QComboBox()
            control.addItems(param.choices)
            if value in param.choices:
                control.setCurrentText(value)
            control.currentTextChanged.connect(
                lambda v, n=param.name, pn=plugin_name: self._on_value_changed(pn, n, v)
            )
            return control

        elif isinstance(param, BoolParameter):
            control = QCheckBox()
            control.setChecked(bool(value))
            control.stateChanged.connect(
                lambda state, n=param.name, pn=plugin_name: self._on_value_changed(
                    pn, n, state == Qt.CheckState.Checked.value
                )
            )
            return control

        elif isinstance(param, StringParameter):
            control = QLineEdit()
            control.setText(str(value))
            if param.placeholder:
                control.setPlaceholderText(param.placeholder)
            control.textChanged.connect(
                lambda v, n=param.name, pn=plugin_name: self._on_value_changed(pn, n, v)
            )
            return control

        elif isinstance(param, FileParameter):
            container = QWidget()
            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)

            line_edit = QLineEdit()
            line_edit.setText(str(value))
            line_edit.textChanged.connect(
                lambda v, n=param.name, pn=plugin_name: self._on_value_changed(pn, n, v)
            )
            h_layout.addWidget(line_edit)

            browse_btn = QPushButton("Browse")
            browse_btn.setMaximumWidth(60)
            browse_btn.clicked.connect(
                lambda checked, le=line_edit, p=param: self._browse_file(le, p)
            )
            h_layout.addWidget(browse_btn)
            return container

        return None

    def _browse_file(self, line_edit: QLineEdit, param: FileParameter) -> None:
        if param.folder_mode:
            path = QFileDialog.getExistingDirectory(self, f"Select {param.label}")
        elif param.save_mode:
            path, _ = QFileDialog.getSaveFileName(self, f"Select {param.label}", "", param.filter)
        else:
            path, _ = QFileDialog.getOpenFileName(self, f"Select {param.label}", "", param.filter)
        if path:
            line_edit.setText(path)

    def _on_value_changed(self, plugin_name: str, param_name: str, value: Any) -> None:
        if plugin_name not in self._edited_defaults:
            self._edited_defaults[plugin_name] = {}
        self._edited_defaults[plugin_name][param_name] = value

    def _reset_current(self) -> None:
        if self._current_plugin is None:
            return
        self._edited_defaults.pop(self._current_plugin, None)
        self._show_plugin(self._current_plugin)

    def _accept(self) -> None:
        # Save all edited defaults
        for plugin_name, params in self._edited_defaults.items():
            self._settings.set_plugin_defaults(plugin_name, params)
        # Remove any plugins that were reset (no longer in edited defaults)
        for plugin_name in list(self._settings.get_all_defaults().keys()):
            if plugin_name not in self._edited_defaults:
                self._settings.clear_plugin_defaults(plugin_name)
        self._settings.save()
        self.accept()
