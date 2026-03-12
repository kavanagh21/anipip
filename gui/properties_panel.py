"""Node parameter editor panel with auto-generated controls."""

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QFrame,
    QGroupBox,
)

from core.pipeline import Pipeline, PipelineNode
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


class PropertiesPanel(QWidget):
    """Panel for editing the parameters of the selected node.

    Auto-generates form controls based on the plugin's parameter definitions.
    """

    parameters_changed = pyqtSignal(str)  # Emitted with node_id when params change

    def __init__(self, pipeline: Pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self._current_node_id: Optional[str] = None
        self._controls: dict[str, QWidget] = {}
        self._updating = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        self.header = QLabel("Properties")
        self.header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.header)

        # Node name label
        self.node_name_label = QLabel("No node selected")
        self.node_name_label.setStyleSheet("color: #666;")
        layout.addWidget(self.node_name_label)

        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(scroll)

        # Container for parameter controls
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.addStretch()
        scroll.setWidget(self.params_container)

    def set_node(self, node_id: Optional[str]) -> None:
        """Set the node to edit.

        Args:
            node_id: The ID of the node to edit, or None to clear
        """
        self._current_node_id = node_id
        self._clear_controls()

        if node_id is None:
            self.node_name_label.setText("No node selected")
            return

        node = self.pipeline.get_node(node_id)
        if node is None:
            self.node_name_label.setText("Node not found")
            return

        self.node_name_label.setText(f"{node.plugin.name}")
        self._create_controls(node)

    def _clear_controls(self) -> None:
        """Clear all parameter controls."""
        self._controls.clear()

        # Remove all widgets except the stretch
        while self.params_layout.count() > 1:
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _create_controls(self, node: PipelineNode) -> None:
        """Create controls for the node's parameters.

        Consecutive ``BoolParameter``s that share the same non-empty
        ``group`` value are rendered together inside a compact
        ``QGroupBox`` with a 2-column grid of inline-label checkboxes.
        All other parameters render with the standard label-above-control
        layout.
        """
        self._updating = True

        # Show help text if the plugin provides it
        help_text = getattr(node.plugin, "help_text", "")
        if help_text:
            help_label = QLabel(help_text)
            help_label.setWordWrap(True)
            help_label.setStyleSheet(
                "color: #999; font-size: 11px; padding: 2px 0 6px 0;"
            )
            self.params_layout.insertWidget(
                self.params_layout.count() - 1, help_label
            )

        params = list(node.plugin.parameters)
        i = 0
        while i < len(params):
            param = params[i]

            # Detect a run of grouped BoolParameters
            if isinstance(param, BoolParameter) and param.group:
                group_name = param.group
                group_params: list[BoolParameter] = []
                while (
                    i < len(params)
                    and isinstance(params[i], BoolParameter)
                    and params[i].group == group_name
                ):
                    group_params.append(params[i])
                    i += 1

                self._create_grouped_bools(group_name, group_params, node)
                continue

            # ActionParameter: render as button directly (no label wrapper)
            if isinstance(param, ActionParameter):
                btn = QPushButton(param.button_label)
                btn.clicked.connect(
                    lambda checked, name=param.name: self._on_action_clicked(name)
                )
                self.params_layout.insertWidget(
                    self.params_layout.count() - 1, btn
                )
                self._controls[param.name] = btn
                i += 1
                continue

            # Standard single-parameter row
            control = self._create_control_for_param(param, node)
            if control:
                wrapper = QWidget()
                wrapper_layout = QVBoxLayout(wrapper)
                wrapper_layout.setContentsMargins(0, 4, 0, 4)

                label = QLabel(param.label)
                label.setStyleSheet("font-weight: 500;")
                wrapper_layout.addWidget(label)
                wrapper_layout.addWidget(control)

                self.params_layout.insertWidget(
                    self.params_layout.count() - 1, wrapper
                )
                self._controls[param.name] = control

            i += 1

        self._updating = False

    def _create_grouped_bools(
        self,
        group_name: str,
        params: list[BoolParameter],
        node: PipelineNode,
    ) -> None:
        """Render a list of BoolParameters as a compact 2-column grid."""
        box = QGroupBox(group_name)
        grid = QGridLayout(box)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setSpacing(4)

        for idx, param in enumerate(params):
            value = node.parameters.get(param.name, param.default)
            cb = QCheckBox(param.label)
            cb.setChecked(value)
            cb.stateChanged.connect(
                lambda state, name=param.name: self._on_value_changed(
                    name, state == Qt.CheckState.Checked.value
                )
            )
            row = idx // 2
            col = idx % 2
            grid.addWidget(cb, row, col)
            self._controls[param.name] = cb

        self.params_layout.insertWidget(
            self.params_layout.count() - 1, box
        )

    def _create_control_for_param(
        self, param: Parameter, node: PipelineNode
    ) -> Optional[QWidget]:
        """Create the appropriate control widget for a parameter type."""
        value = node.parameters.get(param.name, param.default)

        if isinstance(param, IntParameter):
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(param.min_value)
            slider.setMaximum(param.max_value)
            slider.setSingleStep(param.step)
            slider.setValue(value)
            layout.addWidget(slider, stretch=1)

            spin = QSpinBox()
            spin.setMinimum(param.min_value)
            spin.setMaximum(param.max_value)
            spin.setSingleStep(param.step)
            spin.setValue(value)
            spin.setMinimumWidth(60)
            layout.addWidget(spin)

            # Bidirectional sync
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            spin.valueChanged.connect(
                lambda v, name=param.name: self._on_value_changed(name, v)
            )

            container.spin = spin
            return container

        elif isinstance(param, FloatParameter):
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            # Slider operates on integer ticks; map float range to int
            slider_resolution = max(
                int((param.max_value - param.min_value) / param.step), 1
            )
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(slider_resolution)
            slider.setValue(
                round((value - param.min_value) / param.step)
            )
            layout.addWidget(slider, stretch=1)

            spin = QDoubleSpinBox()
            spin.setMinimum(param.min_value)
            spin.setMaximum(param.max_value)
            spin.setSingleStep(param.step)
            spin.setDecimals(param.decimals)
            spin.setValue(value)
            spin.setMinimumWidth(70)
            layout.addWidget(spin)

            # Slider → spin: convert tick to float
            def _slider_to_spin(tick, s=spin, p=param):
                s.setValue(p.min_value + tick * p.step)

            # Spin → slider: convert float to tick
            def _spin_to_slider(val, sl=slider, p=param):
                sl.blockSignals(True)
                sl.setValue(round((val - p.min_value) / p.step))
                sl.blockSignals(False)

            slider.valueChanged.connect(_slider_to_spin)
            spin.valueChanged.connect(_spin_to_slider)
            spin.valueChanged.connect(
                lambda v, name=param.name: self._on_value_changed(name, v)
            )

            container.spin = spin
            return container

        elif isinstance(param, ChoiceParameter):
            control = QComboBox()
            control.addItems(param.choices)
            if value in param.choices:
                control.setCurrentText(value)
            control.currentTextChanged.connect(
                lambda v, name=param.name: self._on_value_changed(name, v)
            )
            return control

        elif isinstance(param, BoolParameter):
            control = QCheckBox()
            control.setChecked(value)
            control.stateChanged.connect(
                lambda state, name=param.name: self._on_value_changed(
                    name, state == Qt.CheckState.Checked.value
                )
            )
            return control

        elif isinstance(param, StringParameter):
            control = QLineEdit()
            control.setText(str(value))
            if param.placeholder:
                control.setPlaceholderText(param.placeholder)
            control.textChanged.connect(
                lambda v, name=param.name: self._on_value_changed(name, v)
            )
            return control

        elif isinstance(param, FileParameter):
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)

            line_edit = QLineEdit()
            line_edit.setText(value)
            line_edit.textChanged.connect(
                lambda v, name=param.name: self._on_value_changed(name, v)
            )
            layout.addWidget(line_edit)

            browse_btn = QPushButton("Browse")
            browse_btn.setMaximumWidth(60)
            browse_btn.clicked.connect(
                lambda checked, le=line_edit, p=param: self._browse_file(le, p)
            )
            layout.addWidget(browse_btn)

            # Store the line edit as the control for value access
            container.line_edit = line_edit
            return container

        return None

    def _browse_file(self, line_edit: QLineEdit, param: FileParameter) -> None:
        """Open file dialog for file parameter."""
        if param.folder_mode:
            path = QFileDialog.getExistingDirectory(
                self, f"Select {param.label}"
            )
        elif param.save_mode:
            path, _ = QFileDialog.getSaveFileName(
                self, f"Select {param.label}", "", param.filter
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Select {param.label}", "", param.filter
            )

        if path:
            line_edit.setText(path)

    def _on_value_changed(self, param_name: str, value) -> None:
        """Handle parameter value changes."""
        if self._updating or self._current_node_id is None:
            return

        node = self.pipeline.get_node(self._current_node_id)
        if node:
            node.parameters[param_name] = value
            node.plugin.set_parameter(param_name, value)
            self.parameters_changed.emit(self._current_node_id)

    def _on_action_clicked(self, action_name: str) -> None:
        """Handle an ActionParameter button click."""
        if self._current_node_id is None:
            return
        node = self.pipeline.get_node(self._current_node_id)
        if node is None:
            return

        inputs = self.pipeline.get_node_inputs(self._current_node_id)
        updates = node.plugin.execute_action(action_name, inputs)
        if not updates:
            return

        for param_name, value in updates.items():
            node.parameters[param_name] = value
            node.plugin.set_parameter(param_name, value)

        # Refresh the panel to show new values and trigger preview
        self.set_node(self._current_node_id)
        self.parameters_changed.emit(self._current_node_id)

    def refresh(self) -> None:
        """Refresh the display for the current node."""
        if self._current_node_id:
            self.set_node(self._current_node_id)
