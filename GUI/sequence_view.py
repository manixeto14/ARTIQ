"""Sequence analysis sub-window view.

Provides the UI for selecting multiple scans from an HDF5 file, choosing
x/y variables, running batch Gaussian fits, and visualizing the results.
"""

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets


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

        # --- Plot Area ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', "X Variable")
        self.plot_widget.setLabel('left', "Y Variable")
        self.scatter_item = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 114, 178)
        )
        self.plot_widget.addItem(self.scatter_item)
        layout.addWidget(self.plot_widget, stretch=2)

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
        ])
        control_layout.addWidget(self.combo_y)

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
        self.scatter_item.setData([], [])

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
        self.scatter_item.setData(self._seq_x, self._seq_y)
        self.plot_widget.autoRange()

    def on_sequence_finished(self):
        """Re-enables the process button when all scans are done."""
        self.btn_process.setText("Process Sequence")
        self.btn_process.setEnabled(True)

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
