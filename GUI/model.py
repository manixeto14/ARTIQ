"""Data model for the MOT image analysis application.

Contains the FitModel (main application state), FitWorker (threaded single-
image fitting), and SequenceWorker (threaded batch sequence analysis).
"""

import numpy as np
from PyQt5 import QtCore
from Analysis import fit_function, gaussian_2d, optical_density
import h5py

# Width definition factor: 2*sigma by default.
# Change to 2.355 for FWHM, or 4.0 for 1/e^2 radius.
WIDTH_FACTOR = 2.0


class FitWorker(QtCore.QObject):
    """Worker that performs a 2D fit on a single image in a background thread.

    Signals:
        finished: Emitted when the worker completes (success or failure).
        update_image: Emitted on success with (fitted_image, roi_slice, popt).
        error: Emitted on failure with an error message string.
    """

    finished = QtCore.pyqtSignal()
    update_image = QtCore.pyqtSignal(np.ndarray, object, object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, img_data, fit_model, roi_slice=None, fit_params=None):
        """Initializes the FitWorker.

        Args:
            img_data: 2D numpy array of the image region to fit.
            fit_model: Callable model function (e.g., gaussian_2d).
            roi_slice: Optional tuple of slices indicating the ROI origin,
                used to map fit coordinates back to full-image coordinates.
            fit_params: Optional list of initial parameter guesses. If None,
                parameters are estimated automatically.
        """
        super().__init__()
        self.img_data = img_data
        self.roi_slice = roi_slice
        self.fit_params = fit_params
        self.fit_model = fit_model

    def run(self):
        """Executes the fit and emits results or error signals."""
        try:
            fitted_image, popt = fit_function(
                self.img_data, self.fit_model, self.fit_params
            )
            fitted_image = np.nan_to_num(fitted_image, nan=0.0)
            self.update_image.emit(fitted_image, self.roi_slice, popt)
        except (ValueError, RuntimeError) as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Unexpected fit error: {e}")
        finally:
            self.finished.emit()


class SequenceWorker(QtCore.QObject):
    """Worker that processes a batch of scans from an HDF5 file.

    For each selected scan group, computes the OD image, fits a 2D Gaussian,
    extracts the requested Y variable, and emits the (x, y) data point.
    Invalid fits are reported as errors and skipped rather than crashing
    the analysis.

    Signals:
        scanProcessed: Emitted per scan with (x_val, y_val, scan_name).
        scanError: Emitted per failed scan with (scan_name, error_msg).
        allFinished: Emitted when all scans have been processed.
    """

    scanProcessed = QtCore.pyqtSignal(float, float, str)
    scanError = QtCore.pyqtSignal(str, str)
    allFinished = QtCore.pyqtSignal()

    def __init__(self, h5_path, scan_groups, indices, roi_slice,
                 x_var_name, y_var_name, fit_model, pixel_size_mm=1.0):
        """Initializes the SequenceWorker.

        Args:
            h5_path: Path to the HDF5 file.
            scan_groups: List of all HDF5 group paths containing scan data.
            indices: List of integer indices into scan_groups to process.
            roi_slice: Optional tuple of slices for cropping images.
            x_var_name: Name of the HDF5 parameter to use as x-axis variable.
            y_var_name: Display name of the y-axis variable to extract
                from the fit (e.g., "Cloud Width X (pixels)").
            fit_model: Callable model function for 2D fitting.
            pixel_size_mm: Physical size of one pixel in millimeters,
                used for unit conversion of cloud widths.
        """
        super().__init__()
        self.h5_path = h5_path
        self.scan_groups = scan_groups
        self.indices = indices
        self.roi_slice = roi_slice
        self.x_var_name = x_var_name
        self.y_var_name = y_var_name
        self.fit_model = fit_model
        self.pixel_size_mm = pixel_size_mm

    def run(self):
        """Processes each selected scan, fitting and extracting data."""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                for idx in self.indices:
                    group_path = self.scan_groups[idx]
                    group_name = (group_path.split('/')[-1]
                                  if group_path != '/' else '/')
                    try:
                        self._process_single_scan(f, group_path, group_name)
                    except Exception as e:
                        self.scanError.emit(group_name, str(e))
        except Exception as e:
            self.scanError.emit("File", str(e))
        finally:
            self.allFinished.emit()

    def _process_single_scan(self, f, group_path, group_name):
        """Processes a single scan group: reads data, computes OD, fits, extracts y.

        Args:
            f: Open h5py.File handle.
            group_path: HDF5 group path string for this scan.
            group_name: Short display name for the scan.

        Raises:
            ValueError: If the x-parameter is missing or the fit is rejected.
            KeyError: If required datasets are missing from the HDF5 group.
        """
        group = f[group_path] if group_path != '/' else f

        # Read X variable
        param_path = (f"{group_path}/Parameters/{self.x_var_name}"
                      if group_path != '/'
                      else f"Parameters/{self.x_var_name}")
        if param_path not in f:
            raise ValueError(f"Missing parameter '{self.x_var_name}'")
        x_val = float(f[param_path][()])

        # Read raw images
        bg = group['_background'][()]
        wa = group['_withatoms'][()]
        woa = group['_withoutatoms'][()]

        # Apply ROI if set
        if self.roi_slice is not None:
            bg = bg[self.roi_slice]
            wa = wa[self.roi_slice]
            woa = woa[self.roi_slice]

        # Compute OD and fit (ValueError propagates if fit is rejected)
        od = np.nan_to_num(
            optical_density(I_at=wa, I_0=woa, I_bg=bg), nan=0.0
        )
        fitted_img, popt = fit_function(od, self.fit_model, None)

        # Extract requested Y variable from fit results
        # popt = [x0, y0, sigma_x, sigma_y, A, B]
        y_val = self._extract_y_value(fitted_img, popt)

        self.scanProcessed.emit(x_val, y_val, group_name)

    def _extract_y_value(self, fitted_img, popt):
        """Extracts the requested Y measurement from fit parameters.

        Args:
            fitted_img: 2D numpy array of the fitted Gaussian model.
            popt: Fitted parameters [x0, y0, sigma_x, sigma_y, A, B].

        Returns:
            Float value of the requested Y variable.
        """
        if self.y_var_name == "Cloud Width X (pixels)":
            return float(WIDTH_FACTOR * abs(popt[2]))
        elif self.y_var_name == "Cloud Width Y (pixels)":
            return float(WIDTH_FACTOR * abs(popt[3]))
        elif self.y_var_name == "Cloud Width X (mm)":
            return float(WIDTH_FACTOR * abs(popt[2]) * self.pixel_size_mm)
        elif self.y_var_name == "Cloud Width Y (mm)":
            return float(WIDTH_FACTOR * abs(popt[3]) * self.pixel_size_mm)
        elif self.y_var_name == "Cloud Width X (m)":
            return float(WIDTH_FACTOR * abs(popt[2]) * self.pixel_size_mm * 1e-3)
        elif self.y_var_name == "Cloud Width Y (m)":
            return float(WIDTH_FACTOR * abs(popt[3]) * self.pixel_size_mm * 1e-3)
        else:
            # Default: Light Absorption (Total Area)
            return float(np.sum(fitted_img))


class FitModel(QtCore.QObject):
    """Central data model for the MOT image analysis application.

    Manages HDF5 file loading, scan group navigation, image state (raw,
    OD), ROI management, single-image fitting, and batch sequence analysis.

    Signals:
        scanGroupsUpdated: Emitted with list of group paths after loading.
        dataUpdated: Emitted when the current display image changes.
        odUpdated: Emitted when the OD image is recomputed.
        fitCompleted: Emitted with (fitted_image, roi_slice, popt) on
            successful fit.
        fitError: Emitted with error message string on fit failure.
        errorOccurred: Emitted with error message for general errors.
        sequenceScanReady: Emitted per scan with (x, y, name) during
            sequence analysis.
        sequenceFinished: Emitted when sequence analysis completes.
        sequenceScanError: Emitted per failed scan with (name, error).
    """

    scanGroupsUpdated = QtCore.pyqtSignal(list)
    dataUpdated = QtCore.pyqtSignal()
    odUpdated = QtCore.pyqtSignal()
    fitCompleted = QtCore.pyqtSignal(np.ndarray, object, object)
    fitError = QtCore.pyqtSignal(str)
    errorOccurred = QtCore.pyqtSignal(str)

    sequenceScanReady = QtCore.pyqtSignal(float, float, str)
    sequenceFinished = QtCore.pyqtSignal()
    sequenceScanError = QtCore.pyqtSignal(str, str)

    def __init__(self):
        """Initializes all model state to empty/None defaults."""
        super().__init__()
        self.current_img = None
        self.raw_background = None
        self.raw_withatoms = None
        self.raw_withoutatoms = None
        self.od_image = None
        self.full_raw_background = None
        self.full_raw_withatoms = None
        self.full_raw_withoutatoms = None
        self.current_h5_path = None
        self.scan_groups = []
        self.roi_slice = None
        self.last_seq_job = None
        self.sequence_metadata = {}
        self.sequence_x_variables = []

    def load_scan_groups(self, file_path):
        """Loads scan groups from an HDF5 file.

        Recursively searches the file for groups containing the required
        datasets ('_background', '_withatoms', '_withoutatoms'). Also
        extracts any metadata and available x-axis parameter names.

        Args:
            file_path: Absolute path to the HDF5 file.
        """
        try:
            with h5py.File(file_path, 'r') as f:
                groups = []
                self._find_scan_groups(f, groups)
                if not groups:
                    self.errorOccurred.emit(
                        "File does not contain '_background', "
                        "'_withatoms' and '_withoutatoms' datasets."
                    )
                    return
                self.current_h5_path = file_path
                self.scan_groups = groups

                # Extract metadata
                self.sequence_metadata = {}
                if 'datasets/Metadata' in f:
                    meta_group = f['datasets/Metadata']
                    for key in meta_group.keys():
                        val = meta_group[key][()]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8')
                        self.sequence_metadata[key] = val

                # Extract available X variables from the first scan
                self.sequence_x_variables = []
                if groups:
                    first_scan = groups[0]
                    param_path = (f"{first_scan}/Parameters"
                                  if first_scan != '/'
                                  else "Parameters")
                    if param_path in f:
                        self.sequence_x_variables = list(
                            f[param_path].keys()
                        )

                self.scanGroupsUpdated.emit(groups)
        except Exception as e:
            self.errorOccurred.emit(f"Can't load file: {e}")

    def get_scan_labels_for_variable(self, var_name):
        """Creates display labels for each scan showing the variable value.

        Args:
            var_name: Name of the HDF5 parameter to read from each scan.

        Returns:
            List of strings like "scan_name (value)" for each scan group.
            Returns "(–)" if the parameter is missing in a particular scan.
        """
        if not self.current_h5_path or not self.scan_groups:
            return []

        labels = []
        try:
            with h5py.File(self.current_h5_path, 'r') as f:
                for group_path in self.scan_groups:
                    group_name = (group_path.split('/')[-1]
                                  if group_path != '/' else '/')
                    param_path = (f"{group_path}/Parameters/{var_name}"
                                  if group_path != '/'
                                  else f"Parameters/{var_name}")
                    if param_path in f:
                        val = f[param_path][()]
                        labels.append(f"{group_name} ({val})")
                    else:
                        labels.append(f"{group_name} (-)")
        except Exception:
            labels = [g.split('/')[-1] for g in self.scan_groups]
        return labels

    def _find_scan_groups(self, group, result, current_path=""):
        """Recursively finds HDF5 groups that contain all required datasets.

        Args:
            group: h5py.Group to search within.
            result: List to append found group paths to (modified in place).
            current_path: Accumulated HDF5 path string for recursion.
        """
        required = {'_background', '_withatoms', '_withoutatoms'}
        if all(name in group for name in required):
            result.append(current_path if current_path else '/')
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                new_path = (f"{current_path}/{key}"
                            if current_path else key)
                self._find_scan_groups(item, result, new_path)

    def load_scan_group_by_index(self, index):
        """Loads the raw image datasets for a specific scan group.

        Reads '_background', '_withatoms', and '_withoutatoms' from the
        HDF5 file, validates them as 2D arrays, computes the OD image,
        and emits dataUpdated.

        Args:
            index: Integer index into self.scan_groups.
        """
        if index < 0 or not self.current_h5_path:
            return
        group_path = self.scan_groups[index]
        try:
            with h5py.File(self.current_h5_path, 'r') as f:
                group = f[group_path] if group_path != '/' else f
                bg = group['_background'][()]
                wa = group['_withatoms'][()]
                woa = group['_withoutatoms'][()]

                for name, arr in [('background', bg),
                                  ('withatoms', wa),
                                  ('withoutatoms', woa)]:
                    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                        self.errorOccurred.emit(
                            f"Invalid dataset '{name}': expected 2D array."
                        )
                        return

                self.full_raw_background = bg
                self.full_raw_withatoms = wa
                self.full_raw_withoutatoms = woa

                self.raw_background = bg
                self.raw_withatoms = wa
                self.raw_withoutatoms = woa

                self._compute_od()
                self.current_img = self.od_image
                self.dataUpdated.emit()
        except Exception as e:
            self.errorOccurred.emit(f"Couldn't load datasets: {e}")

    def _compute_od(self):
        """Recomputes the OD image from the current raw images.

        Requires raw_withatoms and raw_withoutatoms to be set. Emits
        odUpdated after computation.
        """
        if (self.raw_withatoms is not None
                and self.raw_withoutatoms is not None):
            self.od_image = np.nan_to_num(
                optical_density(
                    I_at=self.raw_withatoms,
                    I_0=self.raw_withoutatoms,
                    I_bg=self.raw_background
                ),
                nan=0.0
            )
            self.odUpdated.emit()

    def set_current_image_by_type(self, img_type):
        """Switches the displayed image between OD and raw channels.

        Args:
            img_type: One of "OD", "Background", "With atoms",
                "Without atoms".
        """
        type_map = {
            "OD": self.od_image,
            "Background": self.raw_background,
            "With atoms": self.raw_withatoms,
            "Without atoms": self.raw_withoutatoms,
        }
        img = type_map.get(img_type)
        if img is not None:
            self.current_img = img
            self.dataUpdated.emit()

    def start_fit(self, model_name):
        """Starts a background-thread 2D fit on the current image.

        If an ROI is set, only the ROI region is fitted. The fit type
        is always 2D Gaussian.

        Args:
            model_name: Display name of the fit model (currently only
                "2D Gaussian" is supported).
        """
        if self.current_img is None:
            self.fitError.emit("No image to fit.")
            return

        fit_model = gaussian_2d

        img_to_fit = self.current_img
        if self.roi_slice is not None:
            img_to_fit = self.current_img[self.roi_slice]

        self.thread = QtCore.QThread()
        self.worker = FitWorker(img_to_fit, fit_model, self.roi_slice)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.update_image.connect(self.fitCompleted)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self.fitError)

        self.thread.start()

    def run_sequence_analysis(self, selected_indices, x_var_name,
                              y_var_name, fit_type, pixel_size_mm=1.0):
        """Starts a background-thread batch analysis over selected scans.

        For each scan, computes OD and fits a 2D Gaussian, extracting
        the requested Y measurement. Results are emitted progressively
        via sequenceScanReady.

        Args:
            selected_indices: List of integer indices into scan_groups.
            x_var_name: HDF5 parameter name for the x-axis variable.
            y_var_name: Display name of the y-axis measurement.
            fit_type: Display name of the fit model (currently only
                "2D Gaussian" is supported).
            pixel_size_mm: Physical size of one pixel in mm for unit
                conversion.
        """
        if not self.current_h5_path:
            self.errorOccurred.emit("No HDF5 file loaded.")
            return

        fit_model = gaussian_2d

        self.last_seq_job = {
            'x_var': x_var_name,
            'y_var': y_var_name,
            'x_data': [],
            'y_data': [],
            'scan_names': []
        }

        self.seq_thread = QtCore.QThread()
        self.seq_worker = SequenceWorker(
            self.current_h5_path, self.scan_groups, selected_indices,
            self.roi_slice, x_var_name, y_var_name, fit_model, pixel_size_mm
        )
        self.seq_worker.moveToThread(self.seq_thread)

        self.seq_thread.started.connect(self.seq_worker.run)
        self.seq_worker.scanProcessed.connect(self._on_scan_processed)
        self.seq_worker.scanError.connect(self._on_scan_error)
        self.seq_worker.allFinished.connect(self._on_sequence_all_finished)
        self.seq_worker.allFinished.connect(self.seq_thread.quit)
        self.seq_worker.allFinished.connect(self.seq_worker.deleteLater)
        self.seq_thread.finished.connect(self.seq_thread.deleteLater)

        self.seq_thread.start()

    def _on_scan_processed(self, x_val, y_val, scan_name):
        """Accumulates a successful scan result into the current job.

        Args:
            x_val: X-axis value for this scan point.
            y_val: Y-axis value extracted from the fit.
            scan_name: Display name of the scan group.
        """
        if self.last_seq_job is not None:
            self.last_seq_job['x_data'].append(x_val)
            self.last_seq_job['y_data'].append(y_val)
            self.last_seq_job['scan_names'].append(scan_name)
        self.sequenceScanReady.emit(x_val, y_val, scan_name)

    def _on_scan_error(self, scan_name, error_msg):
        """Logs and re-emits a per-scan error.

        Args:
            scan_name: Display name of the failed scan.
            error_msg: Description of what went wrong.
        """
        print(f"[Sequence] Scan '{scan_name}' failed: {error_msg}")
        self.sequenceScanError.emit(scan_name, error_msg)

    def _on_sequence_all_finished(self):
        """Emits sequenceFinished when batch analysis is complete."""
        self.sequenceFinished.emit()

    def save_sequence_to_csv(self, filepath):
        """Saves the last sequence analysis results to a CSV file.

        Writes metadata (if available) followed by a table of
        (scan_name, x_value, y_value) rows.

        Args:
            filepath: Absolute path for the output CSV file.
        """
        if self.last_seq_job is None or not self.last_seq_job['x_data']:
            self.errorOccurred.emit("No sequence data to save.")
            return

        import csv
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write metadata header
                writer.writerow(["Metadata"])
                if self.sequence_metadata:
                    for k, v in self.sequence_metadata.items():
                        writer.writerow([k, v])

                # Write data
                writer.writerow([])
                writer.writerow(["Sequence Data"])
                writer.writerow([
                    "Scan Name",
                    self.last_seq_job['x_var'],
                    self.last_seq_job['y_var']
                ])
                for x, y, name in zip(
                    self.last_seq_job['x_data'],
                    self.last_seq_job['y_data'],
                    self.last_seq_job['scan_names']
                ):
                    writer.writerow([name, x, y])
        except Exception as e:
            self.errorOccurred.emit(f"Failed to save CSV: {e}")

    def apply_roi(self, slice_tuple):
        """Stores the ROI slice for use in fitting and sequence analysis.

        The ROI does not crop the visual display; it only affects which
        region is sent to curve_fit.

        Args:
            slice_tuple: Tuple of slice objects (row_slice, col_slice).
        """
        self.roi_slice = slice_tuple

    def reset_roi(self):
        """Clears the stored ROI so the full image is used for fitting."""
        self.roi_slice = None

    def update_from_data_changed(self, img_data, lock_checked):
        """Updates the current image from an external data change event.

        Used by the ARTIQ applet integration to receive live camera frames.

        Args:
            img_data: 2D numpy array of the new image data.
            lock_checked: If True, the display is locked and the update
                is ignored.
        """
        if not lock_checked and isinstance(img_data, np.ndarray):
            if self.roi_slice is not None:
                self.current_img = img_data[self.roi_slice]
            else:
                self.current_img = img_data
            self.dataUpdated.emit()