"""Controller connecting the FitModel and FitView in the MVC architecture.

Handles user actions from the view, dispatches them to the model, and
updates the view when the model state changes.
"""

from PyQt5 import QtCore, QtWidgets
from model import FitModel
from view import FitView


class FitController(QtCore.QObject):
    """Main controller for the single-image analysis window.

    Wires signals between FitModel and FitView, manages the ROI workflow,
    file loading dialogs, and opening the sequence analysis sub-window.

    Args:
        model: The FitModel instance holding application state.
        view: The FitView widget displaying images and controls.
    """

    def __init__(self, model, view):
        """Initializes the controller and connects all signals.

        Args:
            model: FitModel instance.
            view: FitView instance.
        """
        super().__init__()
        self.model = model
        self.view = view
        self.view.controller = self

        # View → Controller / Model
        self.view.loadScanGroupClicked.connect(self.on_load_scan_group)
        self.view.scanGroupSelected.connect(self.model.load_scan_group_by_index)
        self.view.displayTypeChanged.connect(self.model.set_current_image_by_type)
        self.view.fitRequested.connect(self.model.start_fit)
        self.view.setRoiRequested.connect(self.on_set_roi)
        self.view.resetRoiRequested.connect(self.on_reset_roi)
        self.view.openSequenceRequested.connect(self.on_open_sequence)
        
        # Grid layout toggles and secondary viewer requests
        self.view.toggleGridRequested.connect(self.view.set_grid_visible)
        self.view.requestExtraViewerImage.connect(self.on_request_extra_viewer_image)

        self.sequence_window = None
        self.sequence_controller = None

        # Model → View
        self.model.scanGroupsUpdated.connect(self.view.set_scan_groups)
        self.model.dataUpdated.connect(self._on_model_data_updated)
        self.model.fitCompleted.connect(
            lambda img, roi, popt: self.view.set_fitted_image(img, roi, popt)
        )
        self.model.fitError.connect(self._show_error)
        self.model.errorOccurred.connect(self._show_error)

        self.view.enable_display_combo(False)

    def on_set_roi(self):
        """Reads the ROI from the view and applies it to the model."""
        slice_tuple = self.view.get_roi_slice()
        if slice_tuple:
            self.model.apply_roi(slice_tuple)
            self.view.set_roi_locked_visuals(True)

    def on_reset_roi(self):
        """Clears the ROI in the model and resets the view visuals."""
        self.model.reset_roi()
        self.view.set_roi_locked_visuals(False)

    def on_load_scan_group(self):
        """Opens a file dialog and loads scan groups from the selected HDF5 file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.view, "Select HDF5 file with scan data", "",
            "HDF5 files (*.h5 *.hdf5)"
        )
        if file_path:
            self.model.load_scan_groups(file_path)

    def _on_model_data_updated(self):
        """Updates the view when the model's current image changes."""
        self.view.set_original_image(self.model.current_img)
        self.view.clear_fitted_image()
        if self.model.raw_background is not None:
            self.view.enable_display_combo(True)

    def _show_error(self, message):
        """Displays an error message dialog to the user.

        Args:
            message: Error description string to show.
        """
        QtWidgets.QMessageBox.critical(self.view, "Error", message)

    def on_request_extra_viewer_image(self, viewer_idx, scan_index, disp_type):
        """Fetches directly from the model without changing the global state and updates the extra viewer."""
        img = self.model.get_image_data(scan_index, disp_type)
        if img is not None:
            self.view.viewers[viewer_idx].set_image(img, disp_type, preserve_levels=False)

    def data_changed(self, value, metadata, persist, mods):
        """Handles ARTIQ dataset change events for live camera updates.

        Called by the applet framework when the subscribed dataset changes.
        Forwards the new image to the model if the view is not locked.

        Args:
            value: Dictionary of dataset values.
            metadata: Dataset metadata (unused).
            persist: Persistence flag (unused).
            mods: Modification descriptors (unused).
        """
        if self.view.args is None or not hasattr(self.view.args, 'img'):
            return
        if self.view.args.img in value:
            img_data = value[self.view.args.img]
            self.model.update_from_data_changed(img_data, self.view.is_locked())

    def on_open_sequence(self):
        """Opens the sequence analysis sub-window, creating it if needed."""
        from sequence_view import SequenceWindow
        from sequence_controller import SequenceController
        if self.sequence_window is None:
            self.sequence_window = SequenceWindow(self.view)
            self.sequence_controller = SequenceController(
                self.model, self.sequence_window, self.view
            )

        self.sequence_controller.populate_ui()
        self.sequence_window.show()
        self.sequence_window.raise_()
        self.sequence_window.activateWindow()