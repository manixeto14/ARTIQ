"""Controller for the sequence analysis sub-window.

Connects the SequenceWindow view to the FitModel for batch processing
of multiple scan groups.
"""

from PyQt5 import QtCore, QtWidgets


class SequenceController(QtCore.QObject):
    """Bridges user actions in SequenceWindow to the FitModel.

    Handles populating the UI from model state, dispatching sequence
    analysis requests, and relaying results/errors back to the view.

    Args:
        model: The FitModel instance holding application state.
        view: The SequenceWindow instance.
        main_view: Optional reference to the main FitView, used to read
            the pixel calibration value.
    """

    def __init__(self, model, view, main_view=None):
        """Initializes the controller and connects all signals.

        Args:
            model: FitModel instance.
            view: SequenceWindow instance.
            main_view: Optional FitView instance for pixel calibration.
        """
        super().__init__()
        self.model = model
        self.view = view
        self.main_view = main_view

        # View → Controller
        self.view.processSequenceRequested.connect(self.on_process_sequence)
        self.view.showMetadataRequested.connect(self.on_show_metadata)
        self.view.xVariableChanged.connect(self.on_x_variable_changed)
        self.view.saveCsvRequested.connect(self.on_save_csv)

        # Model → View (progressive results)
        self.model.sequenceScanReady.connect(self.view.add_result_point)
        self.model.sequenceFinished.connect(self.view.on_sequence_finished)
        self.model.sequenceScanError.connect(self._on_scan_error)

        self.populate_ui()

    def populate_ui(self):
        """Populates the sequence window with current model state.

        Loads scan group list and x-variable options from the model.
        Should be called whenever the sequence window is shown.
        """
        if self.model.scan_groups:
            self.view.set_scan_groups(self.model.scan_groups)

        if self.model.sequence_x_variables:
            self.view.set_x_variables(self.model.sequence_x_variables)
            self.on_x_variable_changed(self.model.sequence_x_variables[0])

    def on_x_variable_changed(self, x_var_name):
        """Updates scan labels when the selected x-variable changes.

        Args:
            x_var_name: Name of the newly selected x-variable parameter.
        """
        labels = self.model.get_scan_labels_for_variable(x_var_name)
        self.view.update_scan_group_labels(labels)

    def on_show_metadata(self):
        """Displays the metadata dialog, or a message if none is available."""
        if self.model.sequence_metadata:
            self.view.show_metadata_dialog(self.model.sequence_metadata)
        else:
            QtWidgets.QMessageBox.information(
                self.view, "No Metadata",
                "No metadata is available in the currently loaded HDF5 file."
            )

    def on_process_sequence(self, selected_indices, x_var_name,
                            y_var_name, fit_type):
        """Dispatches a sequence analysis request to the model.

        Reads the pixel calibration from the main view if available.

        Args:
            selected_indices: List of scan group indices to process.
            x_var_name: HDF5 parameter name for the x-axis.
            y_var_name: Display name of the y-axis measurement.
            fit_type: Display name of the fit model.
        """
        pixel_size_mm = 1.0
        if (self.main_view is not None
                and hasattr(self.main_view, 'get_pixel_size_mm')):
            pixel_size_mm = self.main_view.get_pixel_size_mm()
        self.model.run_sequence_analysis(
            selected_indices, x_var_name, y_var_name, fit_type, pixel_size_mm
        )

    def on_save_csv(self, filepath):
        """Forwards a CSV save request to the model.

        Args:
            filepath: Absolute path for the output CSV file.
        """
        self.model.save_sequence_to_csv(filepath)

    def _on_scan_error(self, scan_name, error_msg):
        """Logs per-scan errors without interrupting the analysis.

        Args:
            scan_name: Display name of the failed scan.
            error_msg: Description of what went wrong.
        """
        print(f"[Sequence] Scan '{scan_name}' error: {error_msg}")
