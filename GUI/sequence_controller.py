"""Controller for the sequence analysis sub-window.

Connects the SequenceWindow view to the FitModel for batch processing
of multiple scan groups.  Scatter-plot fitting uses the model functions
from Analysis.py; all control logic (registry, p0, dispatch) lives here.
"""

import numpy as np
from scipy.optimize import curve_fit
from PyQt5 import QtCore, QtWidgets

from Analysis import (
    fit_linear,
    fit_exponential,
    fit_gaussian_1d,
    fit_temperature_tof,
)


# ---------------------------------------------------------------------------
# Scatter-plot fit registry
# Each entry: function, display label, parameter names shown in the results.
# ---------------------------------------------------------------------------
SCATTER_FIT_MODELS = {
    "Linear": {
        "func":         fit_linear,
        "label":        "Linear  (y = a·x + b)",
        "param_labels": ["a  (slope)", "b  (intercept)"],
    },
    "Exponential": {
        "func":         fit_exponential,
        "label":        "Exponential  (y = A·exp(−x/τ) + C)",
        "param_labels": ["A  (amplitude)", "τ  (time constant)", "C  (offset)"],
    },
    "Gaussian": {
        "func":         fit_gaussian_1d,
        "label":        "Gaussian 1D  (y = A·exp(−(x−x₀)²/2σ²) + B)",
        "param_labels": ["A  (amplitude)", "x₀  (center)", "σ  (width)",
                         "B  (offset)"],
    },
    "Temperature": {
        "func":         fit_temperature_tof,
        "label":        "Temperature TOF  (σ²(t) = σ₀² + kT/m·t²)",
        "param_labels": ["σ₀²  (initial variance)", "kT/m  [units²/time²]"],
    },
}


def _initial_guess(model_key, x, y):
    """Returns sensible starting parameters for curve_fit."""
    if model_key == "Linear":
        dx = x[-1] - x[0]
        a0 = (y[-1] - y[0]) / dx if dx != 0 else 1.0
        return [a0, float(y.mean() - a0 * x.mean())]

    if model_key == "Exponential":
        return [float(y.max() - y.min()),
                float((x.max() - x.min()) / 3.0) or 1.0,
                float(y.min())]

    if model_key == "Gaussian":
        return [float(y.max() - y.min()),
                float(x[np.argmax(y)]),
                float((x.max() - x.min()) / 4.0) or 1.0,
                float(y.min())]

    if model_key == "Temperature":
        denom = float(x.max() ** 2 - x.min() ** 2) or 1e-30
        return [max(float(y.min()), 1e-6),
                max(float(y.max() - y.min()) / denom, 1e-6)]

    return None


class SequenceController(QtCore.QObject):
    """Bridges the SequenceWindow view to the FitModel.

    Args:
        model: FitModel instance holding application state.
        view: SequenceWindow instance.
        main_view: Optional FitView, used to read pixel calibration.
    """

    def __init__(self, model, view, main_view=None):
        super().__init__()
        self.model = model
        self.view = view
        self.main_view = main_view

        # View → Controller
        self.view.processSequenceRequested.connect(self.on_process_sequence)
        self.view.showMetadataRequested.connect(self.on_show_metadata)
        self.view.xVariableChanged.connect(self.on_x_variable_changed)
        self.view.saveCsvRequested.connect(self.on_save_csv)
        self.view.fitScatterRequested.connect(self.on_fit_scatter)

        # Model → View
        self.model.sequenceScanReady.connect(self.view.add_result_point)
        self.model.sequenceFinished.connect(self.view.on_sequence_finished)
        self.model.sequenceScanError.connect(self._on_scan_error)

        self.populate_ui()

    def populate_ui(self):
        """Populates the window with current model state."""
        if self.model.scan_groups:
            self.view.set_scan_groups(self.model.scan_groups)
        if self.model.sequence_x_variables:
            self.view.set_x_variables(self.model.sequence_x_variables)
            self.on_x_variable_changed(self.model.sequence_x_variables[0])

    def on_x_variable_changed(self, x_var_name):
        """Updates scan labels when the x-variable selection changes."""
        labels = self.model.get_scan_labels_for_variable(x_var_name)
        self.view.update_scan_group_labels(labels)

    def on_show_metadata(self):
        """Shows the metadata dialog, or a message if none is available."""
        if self.model.sequence_metadata:
            self.view.show_metadata_dialog(self.model.sequence_metadata)
        else:
            QtWidgets.QMessageBox.information(
                self.view, "No Metadata",
                "No metadata found in the currently loaded HDF5 file."
            )

    def on_process_sequence(self, selected_indices, x_var_name,
                            y_var_name, fit_type):
        """Dispatches a sequence analysis request to the model."""
        pixel_size_mm = 1.0
        if self.main_view is not None:
            pixel_size_mm = self.main_view.get_pixel_size_mm()
        self.model.run_sequence_analysis(
            selected_indices, x_var_name, y_var_name, fit_type, pixel_size_mm
        )

    def on_save_csv(self, filepath):
        """Forwards a CSV save request to the model."""
        self.model.save_sequence_to_csv(filepath)

    def _on_scan_error(self, scan_name, error_msg):
        """Logs per-scan errors without interrupting the analysis."""
        print(f"[Sequence] Scan '{scan_name}' error: {error_msg}")

    # ------------------------------------------------------------------
    # Scatter-plot fitting
    # ------------------------------------------------------------------

    def on_fit_scatter(self, model_key):
        """Fits the scatter data with the selected model and shows results.

        Args:
            model_key: One of the keys in SCATTER_FIT_MODELS.
        """
        if self.model.last_seq_job is None:
            self.view.show_scatter_fit_error(
                "No sequence data yet. Run a sequence analysis first."
            )
            return

        x = np.array(self.model.last_seq_job.get('x_data', []), dtype=float)
        y = np.array(self.model.last_seq_job.get('y_data', []), dtype=float)

        if len(x) < 2:
            self.view.show_scatter_fit_error(
                "Need at least 2 data points to fit."
            )
            return

        meta = SCATTER_FIT_MODELS[model_key]
        func = meta["func"]
        p0   = _initial_guess(model_key, x, y)

        try:
            popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=20000)
        except Exception as exc:
            self.view.show_scatter_fit_error(f"Fit did not converge: {exc}")
            return

        perr  = np.sqrt(np.diag(pcov))
        x_fit = np.linspace(x.min(), x.max(), 400)
        y_fit = func(x_fit, *popt)

        # Coefficient of determination R²
        ss_res = np.sum((y - func(x, *popt)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        
        param_labels = meta["param_labels"][:]
        param_values = popt.tolist()
        param_errors = perr.tolist()

        if model_key == "Temperature":
            kT_over_m = popt[1]
            kT_over_m_err = perr[1]

            k_b, mass = self.view.get_temperature_constants()
            
            temp = kT_over_m * mass / k_b
            temp_err = (kT_over_m_err * mass / k_b) if kT_over_m_err is not None else 0.0

            param_labels.append("T  [K]")
            param_values.append(temp)
            param_errors.append(temp_err)

        self.view.show_scatter_fit(
            model_name=meta["label"],
            x_fit=x_fit,
            y_fit=y_fit,
            param_labels=param_labels,
            param_values=param_values,
            param_errors=param_errors,
            quality_text=f"R² = {r2:.5f}   |   N = {len(x)} points",
        )
