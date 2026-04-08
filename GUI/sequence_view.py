"""Sequence analysis sub-window view.

Provides the UI for selecting multiple scans from an HDF5 file, choosing
x/y variables, running batch Gaussian fits, and visualizing the results.
Supports post-processing curve fits (linear, exponential, Gaussian, temperature)
on the resulting scatter plot.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui


class SequenceWindow(QtWidgets.QMainWindow):
    """Window for batch sequence analysis across multiple scan groups.

    Displays a scatter plot of extracted measurements and a control panel
    for selecting scans, variables, and triggering the analysis.

    Signals:
        processSequenceRequested: Emitted with (indices, x_var, y_var,
            fit_type) when the user clicks "Process Sequence".
        showMetadataRequested: Emitted when the user wants to see file
            metadata.
        xVariableChanged: Emitted with the new x-variable name when
            the combo box selection changes.
        saveCsvRequested: Emitted with the chosen filepath when the user
            wants to export results.
    """

    processSequenceRequested = QtCore.pyqtSignal(list, str, str, str)
    showMetadataRequested = QtCore.pyqtSignal()
    xVariableChanged = QtCore.pyqtSignal(str)
    saveCsvRequested = QtCore.pyqtSignal(str)
    fitScatterRequested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        """Initializes the sequence window.

        Args:
            parent: Optional parent widget (typically the main FitView).
        """
        super().__init__(parent)
        self.setWindowTitle("Sequence Analysis")
        self.resize(800, 600)
        self._setup_ui()

    def _setup_ui(self):
        """Builds the UI: scatter plot on the left, control panel on the right."""
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QHBoxLayout(main_widget)

        # --- Left side: plot + results ---
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # --- Plot Area ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', "X Variable")
        self.plot_widget.setLabel('left', "Y Variable")
        # Actual points (faint/transparent)
        self.raw_scatter_item = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 114, 178, 50)
        )
        self.plot_widget.addItem(self.raw_scatter_item)

        # Error bars for the mean points
        self.error_bars_item = pg.ErrorBarItem(
            pen=pg.mkPen(color=(0, 114, 178), width=1.5)
        )
        self.plot_widget.addItem(self.error_bars_item)

        # Mean points
        self.mean_scatter_item = pg.ScatterPlotItem(
            size=15, pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 114, 178)
        )
        self.plot_widget.addItem(self.mean_scatter_item)

        # Fit overlay curve on the scatter plot
        self.fit_curve_item = pg.PlotDataItem(
            pen=pg.mkPen(color=(220, 50, 50), width=2.5)
        )
        self.plot_widget.addItem(self.fit_curve_item)

        left_layout.addWidget(self.plot_widget, stretch=3)

        # --- Fit Results Panel ---
        self.results_group = QtWidgets.QGroupBox("Fit Results")
        self.results_group.setVisible(False)
        results_layout = QtWidgets.QVBoxLayout(self.results_group)

        self.lbl_fit_model_used = QtWidgets.QLabel("")
        _bold = QtGui.QFont()
        _bold.setBold(True)
        self.lbl_fit_model_used.setFont(_bold)
        results_layout.addWidget(self.lbl_fit_model_used)

        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value", "± Uncertainty"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.results_table.setMaximumHeight(160)
        results_layout.addWidget(self.results_table)

        self.lbl_fit_quality = QtWidgets.QLabel("")
        self.lbl_fit_quality.setWordWrap(True)
        results_layout.addWidget(self.lbl_fit_quality)

        left_layout.addWidget(self.results_group, stretch=1)

        layout.addWidget(left_widget, stretch=2)

        # --- Control Panel ---
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setAlignment(QtCore.Qt.AlignTop)

        # Scan group selection list
        control_layout.addWidget(QtWidgets.QLabel("Select Scans to Analyze:"))
        self.list_scans = QtWidgets.QListWidget()
        self.list_scans.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection
        )
        control_layout.addWidget(self.list_scans)

        btn_select_all = QtWidgets.QPushButton("Select All")
        btn_select_all.clicked.connect(self._select_all_scans)
        control_layout.addWidget(btn_select_all)

        # X variable combo
        control_layout.addWidget(
            QtWidgets.QLabel("X Variable (from Parameters):")
        )
        self.combo_x = QtWidgets.QComboBox()
        self.combo_x.currentTextChanged.connect(self.xVariableChanged.emit)
        control_layout.addWidget(self.combo_x)

        # Fit model combo
        control_layout.addWidget(QtWidgets.QLabel("Fit Model:"))
        self.combo_fit = QtWidgets.QComboBox()
        self.combo_fit.addItems(["2D Gaussian"])
        control_layout.addWidget(self.combo_fit)

        # Y variable combo
        control_layout.addWidget(QtWidgets.QLabel("Y Variable (from Fit):"))
        self.combo_y = QtWidgets.QComboBox()
        self.combo_y.addItems([
            "Light Absorption (Total Area)",
            "Cloud Width X (pixels)",
            "Cloud Width Y (pixels)",
            "Cloud Width X (mm)",
            "Cloud Width Y (mm)",
            "Cloud Width X (m)",
            "Cloud Width Y (m)",
        ])
        control_layout.addWidget(self.combo_y)

        # --- Scatter Plot Fit Section ---
        control_layout.addWidget(QtWidgets.QLabel("─" * 30))
        control_layout.addWidget(QtWidgets.QLabel("Fit on Scatter Plot:"))
        self.combo_scatter_fit = QtWidgets.QComboBox()
        self.combo_scatter_fit.addItems([
            "None",
            "Linear  (y = a·x + b)",
            "Exponential  (y = A·exp(-x/τ) + C)",
            "Gaussian 1D  (y = A·exp(-(x-x0)²/2σ²) + B)",
            "Temperature TOF  (σ²(t) = σ₀² + kT/m·t²)",
        ])
        control_layout.addWidget(self.combo_scatter_fit)

        self.btn_fit_scatter = QtWidgets.QPushButton("Fit Scatter Plot")
        self.btn_fit_scatter.setStyleSheet(
            "background-color: #e67e22; color: white; "
            "font-weight: bold; padding: 6px;"
        )
        self.btn_fit_scatter.clicked.connect(self._on_fit_scatter_clicked)
        control_layout.addWidget(self.btn_fit_scatter)

        # --- Temperature constants (for TOF fit → T directly) ---
        tof_group = QtWidgets.QGroupBox("TOF Temperature Constants")
        tof_layout = QtWidgets.QFormLayout(tof_group)

        self.spin_kb = QtWidgets.QDoubleSpinBox()
        self.spin_kb.setDecimals(6)
        self.spin_kb.setRange(1e-30, 1e-10)
        self.spin_kb.setValue(1.380649e-23)   # Boltzmann constant [J/K]
        self.spin_kb.setSingleStep(1e-25)
        self.spin_kb.setToolTip("Boltzmann constant k_B [J/K]")
        self.spin_kb.setDecimals(4)  # show in scientific notation via large range
        # Use a line-edit so the user can type directly:
        self.edit_kb = QtWidgets.QLineEdit("1.380649e-23")
        self.edit_kb.setToolTip("Boltzmann constant k_B [J/K]")
        tof_layout.addRow("k_B  [J/K]:", self.edit_kb)

        self.edit_mass = QtWidgets.QLineEdit("2.207e-25")
        self.edit_mass.setToolTip(
            "Atom mass [kg]"
        )
        tof_layout.addRow("Mass  [kg]:", self.edit_mass)

        control_layout.addWidget(tof_group)

        self.btn_clear_fit = QtWidgets.QPushButton("Clear Fit")
        self.btn_clear_fit.setStyleSheet("padding: 5px;")
        self.btn_clear_fit.clicked.connect(self._on_clear_fit_clicked)
        control_layout.addWidget(self.btn_clear_fit)

        control_layout.addWidget(QtWidgets.QLabel("─" * 30))

        # Action buttons
        self.btn_metadata = QtWidgets.QPushButton("Show Metadata")
        self.btn_metadata.clicked.connect(self.showMetadataRequested.emit)
        control_layout.addWidget(self.btn_metadata)

        self.btn_process = QtWidgets.QPushButton("Process Sequence")
        self.btn_process.setStyleSheet(
            "background-color: #28a745; color: white; "
            "font-weight: bold; padding: 8px;"
        )
        self.btn_process.clicked.connect(self._on_process_clicked)
        control_layout.addWidget(self.btn_process)

        self.btn_save_csv = QtWidgets.QPushButton("Save CSV")
        self.btn_save_csv.setStyleSheet("padding: 5px;")
        self.btn_save_csv.clicked.connect(self._on_save_csv_clicked)
        control_layout.addWidget(self.btn_save_csv)

        layout.addWidget(control_panel, stretch=1)

    def _select_all_scans(self):
        """Selects all items in the scan list."""
        for i in range(self.list_scans.count()):
            self.list_scans.item(i).setSelected(True)

    def set_scan_groups(self, groups):
        """Replaces the scan list with new group paths.

        Args:
            groups: List of HDF5 group path strings.
        """
        self.list_scans.clear()
        self.list_scans.addItems(groups)

    def update_scan_group_labels(self, labels):
        """Updates the display text of existing list items.

        Preserves selection state by modifying text in-place rather than
        clearing and re-adding items.

        Args:
            labels: List of display label strings, one per scan.
        """
        for i in range(self.list_scans.count()):
            if i < len(labels):
                self.list_scans.item(i).setText(labels[i])

    def set_x_variables(self, variables):
        """Populates the x-variable combo box.

        Args:
            variables: List of parameter name strings from the HDF5 file.
        """
        self.combo_x.clear()
        self.combo_x.addItems(variables)

    def _on_process_clicked(self):
        """Validates selection and emits processSequenceRequested."""
        selected_indices = [
            item.row() for item in self.list_scans.selectedIndexes()
        ]
        if not selected_indices:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please select at least one scan group."
            )
            return

        x_var = self.combo_x.currentText()
        y_var = self.combo_y.currentText()
        fit_type = self.combo_fit.currentText()

        self.plot_widget.setLabel('bottom', x_var)
        self.plot_widget.setLabel('left', y_var)

        # Reset progressive plot data
        self._seq_x = []
        self._seq_y = []
        self.raw_scatter_item.setData([], [])
        self.mean_scatter_item.setData([], [])
        self.error_bars_item.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]))
        # Clear any previous fit overlay when starting a new sequence
        self._on_clear_fit_clicked()

        self.btn_process.setText("Processing...")
        self.btn_process.setEnabled(False)

        self.processSequenceRequested.emit(
            selected_indices, x_var, y_var, fit_type
        )

    def add_result_point(self, x, y, name):
        """Adds a single data point to the scatter plot progressively.

        Called once per successfully processed scan during sequence analysis.

        Args:
            x: X-axis value for the point.
            y: Y-axis value for the point.
            name: Scan name (currently unused for display, but available
                for future tooltip support).
        """
        self._seq_x.append(x)
        self._seq_y.append(y)
        self._update_plot()

    def _update_plot(self):
        """Recalculates means and error bars, and updates the view."""
        if not self._seq_x:
            return

        x_raw = np.array(self._seq_x)
        y_raw = np.array(self._seq_y)
        
        self.raw_scatter_item.setData(x_raw, y_raw)
        
        unique_x = np.unique(x_raw)
        mean_x = []
        mean_y = []
        std_y = []

        for ux in unique_x:
            ys = y_raw[x_raw == ux]
            mean_x.append(ux)
            mean_y.append(np.mean(ys))
            if len(ys) > 1:
                std_y.append(np.std(ys, ddof=1)) # sample std
            else:
                std_y.append(0.0)

        mean_x = np.array(mean_x)
        mean_y = np.array(mean_y)
        std_y = np.array(std_y)

        self.mean_scatter_item.setData(mean_x, mean_y)
        
        # Adjust the cap widths (beam)
        beam_w = 0.05
        if len(mean_x) > 1:
            span = np.max(mean_x) - np.min(mean_x)
            if span > 0:
                beam_w = span * 0.02
        
        self.error_bars_item.setData(x=mean_x, y=mean_y, top=std_y, bottom=std_y, beam=beam_w)
        self.plot_widget.autoRange()

    def on_sequence_finished(self):
        """Re-enables the process button when all scans are done."""
        self.btn_process.setText("Process Sequence")
        self.btn_process.setEnabled(True)

    # --- Scatter-plot fitting ---

    def _on_fit_scatter_clicked(self):
        """Emits fitScatterRequested with the selected scatter fit model."""
        model_text = self.combo_scatter_fit.currentText()
        if model_text == "None":
            self._on_clear_fit_clicked()
            return
        # Extract model key (first word)
        model_key = model_text.split()[0]
        self.fitScatterRequested.emit(model_key)

    def _on_clear_fit_clicked(self):
        """Removes the fit overlay and hides the results panel."""
        self.fit_curve_item.setData([], [])
        self.results_group.setVisible(False)

    def show_scatter_fit(self, model_name, x_fit, y_fit, param_labels,
                         param_values, param_errors, quality_text):
        """Draws the fit curve and populates the results panel.

        Args:
            model_name: Human-readable name of the fit model.
            x_fit: 1D array of x values for the smooth fit curve.
            y_fit: 1D array of y values for the smooth fit curve.
            param_labels: List of parameter name strings.
            param_values: List of float parameter values.
            param_errors: List of float parameter standard deviations.
            quality_text: String summarising fit quality (R², χ², …).
        """
        self.fit_curve_item.setData(x_fit, y_fit)

        self.lbl_fit_model_used.setText(f"Model: {model_name}")

        self.results_table.setRowCount(len(param_labels))
        for row, (label, val, err) in enumerate(
            zip(param_labels, param_values, param_errors)
        ):
            self.results_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(label)
            )
            self.results_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(f"{val:.6g}")
            )
            err_text = f"± {err:.6g}" if err is not None else "n/a"
            self.results_table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(err_text)
            )
        self.results_table.resizeColumnsToContents()

        self.lbl_fit_quality.setText(quality_text)
        self.results_group.setVisible(True)

    def show_scatter_fit_error(self, message):
        """Shows a warning dialog when the scatter fit fails.

        Args:
            message: Human-readable description of the failure.
        """
        QtWidgets.QMessageBox.warning(self, "Fit Failed", message)

    def _on_save_csv_clicked(self):
        """Opens a file dialog and emits saveCsvRequested with the path."""
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", "", "CSV Files (*.csv)"
        )
        if filepath:
            self.saveCsvRequested.emit(filepath)

    def show_metadata_dialog(self, metadata_dict):
        """Displays HDF5 file metadata in a modal table dialog.

        Args:
            metadata_dict: Dictionary of metadata key-value pairs.
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Metadata")
        dialog.resize(400, 500)
        layout = QtWidgets.QVBoxLayout(dialog)

        table = QtWidgets.QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Parameter", "Value"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setRowCount(len(metadata_dict))

        for row, (key, value) in enumerate(metadata_dict.items()):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(key)))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))

        layout.addWidget(table)
        dialog.exec_()

    def get_temperature_constants(self):
        """Returns the user-provided Boltzmann constant and atom mass.

        Returns:
            Tuple of (k_B, mass) as floats.
        """
        try:
            k_b = float(self.edit_kb.text())
        except ValueError:
            k_b = 1.380649e-23
        try:
            mass = float(self.edit_mass.text())
        except ValueError:
            mass = 2.207e-25
        return k_b, mass
