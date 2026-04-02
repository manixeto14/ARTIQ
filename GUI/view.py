"""Main view widget for the MOT image analysis application.

Displays the OD/raw image with linked marginal distribution plots,
an interactive ROI, cloud width overlays, and a scale bar.
"""

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from model import WIDTH_FACTOR


class FitView(QtWidgets.QWidget):
    """Main image viewer with ROI, fit overlay, and analysis controls.

    Displays a 2D image with linked X/Y marginal distribution plots,
    an interactive rectangular ROI, fitted image panel, cloud width
    measurement overlay, and a physical scale bar.

    Signals:
        loadScanGroupClicked: User wants to load an HDF5 file.
        scanGroupSelected: User selected a scan group by index.
        displayTypeChanged: User changed the display type combo.
        fitRequested: User requested a fit with the given model name.
        lockToggled: User toggled the manual lock checkbox.
        setRoiRequested: User clicked 'Set ROI'.
        resetRoiRequested: User clicked 'Reset ROI'.
        openSequenceRequested: User clicked 'Sequence Window'.
        calibrationChanged: User changed the pixel size calibration.
    """
    loadScanGroupClicked = QtCore.pyqtSignal()
    scanGroupSelected = QtCore.pyqtSignal(int)
    displayTypeChanged = QtCore.pyqtSignal(str)
    fitRequested = QtCore.pyqtSignal(str)
    lockToggled = QtCore.pyqtSignal(bool)
    setRoiRequested = QtCore.pyqtSignal()
    resetRoiRequested = QtCore.pyqtSignal()
    openSequenceRequested = QtCore.pyqtSignal()
    calibrationChanged = QtCore.pyqtSignal(float)  # pixel_size_mm

    def __init__(self, args, req):
        """Initializes the view and builds the UI.

        Args:
            args: ARTIQ applet arguments (has 'img' attribute for
                dataset subscription). None in standalone mode.
            req: ARTIQ applet request object. None in standalone mode.
        """
        super().__init__()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.args = args
        self.req = req
        self.controller = None
        self._setup_ui()

    def _setup_ui(self):
        """Builds the complete UI layout with plots, controls, and ROI."""
        main_layout = QtWidgets.QVBoxLayout(self)

        self.images_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Unified GraphicsLayoutWidget for exact pixel alignment
        self.glw = pg.GraphicsLayoutWidget()
        
        self.p_y = self.glw.addPlot()
        self.p_y.setMaximumWidth(150)
        self.p_y.hideAxis('bottom')
        
        self.p_img = self.glw.addPlot()
        self.p_img.hideAxis('bottom')
        self.p_img.hideAxis('left')
        self.p_img.setAspectLocked(True)
        
        self.glw.nextRow()
        self.glw.nextCol()
        
        self.p_x = self.glw.addPlot()
        self.p_x.setMaximumHeight(150)
        self.p_x.hideAxis('left')

        # Link axes
        self.p_x.setXLink(self.p_img)
        self.p_y.setYLink(self.p_img)

        # Main Image Item
        self.img_item = pg.ImageItem()
        self.img_item.setColorMap(pg.colormap.get('jet', source='matplotlib'))
        self.p_img.addItem(self.img_item)

        # --- Cloud width overlay items ---
        self.width_line_x = pg.PlotDataItem(pen=pg.mkPen('black', width=2, style=QtCore.Qt.DashLine))
        self.p_img.addItem(self.width_line_x)
        self.width_text = pg.TextItem('', color='black', anchor=(0.5, 1.0))
        self.width_text.setFont(QtWidgets.QApplication.font())
        self.p_img.addItem(self.width_text)
        self.width_text.hide()
        self.width_line_x.hide()

        # --- Scale bar (map style) ---
        self.scale_bar_line = pg.PlotDataItem(pen=pg.mkPen('w', width=4))
        self.p_img.addItem(self.scale_bar_line)
        self.scale_bar_text = pg.TextItem('', color='b', anchor=(0.5, 1.0))
        self.p_img.addItem(self.scale_bar_text)
        self.scale_bar_line.hide()
        self.scale_bar_text.hide()

        # Add ROI to image
        self.roi = pg.RectROI([0, 0], [50, 50], pen=(0, 9))
        self.p_img.addItem(self.roi)
        self.first_image_loaded = False

        self.fitted_view = pg.ImageView()
        # Keep fitted_view stylized identically
        self.fitted_view.setColorMap(pg.colormap.get('jet', source='matplotlib'))
        self.fitted_view.ui.roiBtn.hide()
        self.fitted_view.ui.menuBtn.hide()
        # Match the original image plot's coordinate convention
        self.fitted_view.getView().invertY(False)
        self.fitted_view.getView().setAspectLocked(True)
        self.fitted_view.hide()

        self.images_splitter.addWidget(self.glw)
        self.images_splitter.addWidget(self.fitted_view)
        
        # Set equal stretch so both viewers occupy 50/50 of horizontal space if open
        self.images_splitter.setSizes([500, 500])

        # Pre-create curve items for marginals
        self.data_curve_y = self.p_y.plot(pen='k')
        self.fit_curve_y = self.p_y.plot(pen='r')

        self.data_curve_x = self.p_x.plot(pen='k')
        self.fit_curve_x = self.p_x.plot(pen='r')

        #Buttons

        controls_layout = QtWidgets.QHBoxLayout()

        self.btn_load_scan = QtWidgets.QPushButton("Load Scan Group")
        self.btn_load_scan.clicked.connect(self.loadScanGroupClicked.emit)
        controls_layout.addWidget(self.btn_load_scan)

        self.combo_scan_groups = QtWidgets.QComboBox()
        self.combo_scan_groups.setEnabled(False)
        self.combo_scan_groups.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.combo_scan_groups.setMinimumWidth(200)
        self.combo_scan_groups.currentIndexChanged.connect(self._on_scan_group_index_changed)
        controls_layout.addWidget(self.combo_scan_groups)

        self.combo_display = QtWidgets.QComboBox()
        self.combo_display.addItems(["OD", "Background", "With atoms", "Without atoms"])
        self.combo_display.setEnabled(False)
        self.combo_display.currentTextChanged.connect(self.displayTypeChanged.emit)
        controls_layout.addWidget(self.combo_display)

        self.check_lock = QtWidgets.QCheckBox("Lock to manual")
        self.check_lock.toggled.connect(self.lockToggled.emit)
        controls_layout.addWidget(self.check_lock)

        # Pixel calibration
        controls_layout.addWidget(QtWidgets.QLabel("px size (mm):"))
        self.spin_pixel_size = QtWidgets.QDoubleSpinBox()
        self.spin_pixel_size.setDecimals(6)
        self.spin_pixel_size.setRange(0.000001, 100.0)
        self.spin_pixel_size.setValue(0.007)  # default 5 µm/pixel
        self.spin_pixel_size.setSingleStep(0.001)
        self.spin_pixel_size.setToolTip("Physical size of one pixel in mm")
        self.spin_pixel_size.valueChanged.connect(self.calibrationChanged.emit)
        controls_layout.addWidget(self.spin_pixel_size)

        controls_layout.addStretch()

        self.combo_fit_type = QtWidgets.QComboBox()
        self.fit_models = ["2D Gaussian"]
        self.combo_fit_type.addItems(self.fit_models)
        controls_layout.addWidget(self.combo_fit_type)

        self.btn_set_roi = QtWidgets.QPushButton("Set ROI")
        self.btn_set_roi.clicked.connect(self.setRoiRequested.emit)
        controls_layout.addWidget(self.btn_set_roi)
        
        self.btn_reset_roi = QtWidgets.QPushButton("Reset ROI")
        self.btn_reset_roi.clicked.connect(self.resetRoiRequested.emit)
        controls_layout.addWidget(self.btn_reset_roi)

        self.btn_fit = QtWidgets.QPushButton("Start Fit")
        self.btn_fit.clicked.connect(lambda: self.fitRequested.emit(self.combo_fit_type.currentText()))
        controls_layout.addWidget(self.btn_fit)
        
        # Add Sequence Window Button
        self.btn_seq = QtWidgets.QPushButton("Sequence Window")
        self.btn_seq.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.btn_seq.clicked.connect(self.openSequenceRequested.emit)
        controls_layout.addWidget(self.btn_seq)

        # Manual ROI layout
        roi_controls_layout = QtWidgets.QHBoxLayout()
        roi_controls_layout.addWidget(QtWidgets.QLabel("Manual ROI:  X:"))
        self.spin_roi_x = QtWidgets.QSpinBox()
        self.spin_roi_x.setRange(0, 10000)
        roi_controls_layout.addWidget(self.spin_roi_x)
        
        roi_controls_layout.addWidget(QtWidgets.QLabel("Y:"))
        self.spin_roi_y = QtWidgets.QSpinBox()
        self.spin_roi_y.setRange(0, 10000)
        roi_controls_layout.addWidget(self.spin_roi_y)

        roi_controls_layout.addWidget(QtWidgets.QLabel("W:"))
        self.spin_roi_w = QtWidgets.QSpinBox()
        self.spin_roi_w.setRange(1, 10000)
        self.spin_roi_w.setValue(50)
        roi_controls_layout.addWidget(self.spin_roi_w)

        roi_controls_layout.addWidget(QtWidgets.QLabel("H:"))
        self.spin_roi_h = QtWidgets.QSpinBox()
        self.spin_roi_h.setRange(1, 10000)
        self.spin_roi_h.setValue(50)
        roi_controls_layout.addWidget(self.spin_roi_h)
        
        roi_controls_layout.addStretch()

        # Marginal ROI Regions
        self.roi_x_region = pg.LinearRegionItem([0, 50], orientation=pg.LinearRegionItem.Vertical)
        self.roi_x_region.setZValue(10)
        self.p_x.addItem(self.roi_x_region)
        self.roi_x_region.setBrush(pg.mkBrush((0, 255, 0, 50)))

        self.roi_y_region = pg.LinearRegionItem([0, 50], orientation=pg.LinearRegionItem.Horizontal)
        self.roi_y_region.setZValue(10)
        self.p_y.addItem(self.roi_y_region)
        self.roi_y_region.setBrush(pg.mkBrush((0, 255, 0, 50)))

        # Add colorbar
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        self.glw.addItem(self.hist, row=0, col=2, rowspan=2)

        def on_roi_spin_changed():
            # Only update ROI if user explicitly changed the spinbox
            self.roi.sigRegionChanged.disconnect(on_roi_dragged)
            self.roi.setPos([self.spin_roi_x.value(), self.spin_roi_y.value()])
            self.roi.setSize([self.spin_roi_w.value(), self.spin_roi_h.value()])
            update_marginal_regions()
            self.roi.sigRegionChanged.connect(on_roi_dragged)

        def update_marginal_regions():
            pos = self.roi.pos()
            size = self.roi.size()
            
            self.roi_x_region.blockSignals(True)
            self.roi_x_region.setRegion((pos[0], pos[0] + size[0]))
            self.roi_x_region.blockSignals(False)

            self.roi_y_region.blockSignals(True)
            self.roi_y_region.setRegion((pos[1], pos[1] + size[1]))
            self.roi_y_region.blockSignals(False)

        def update_spinboxes():
            pos = self.roi.pos()
            size = self.roi.size()
            self.spin_roi_x.blockSignals(True)
            self.spin_roi_y.blockSignals(True)
            self.spin_roi_w.blockSignals(True)
            self.spin_roi_h.blockSignals(True)
            
            self.spin_roi_x.setValue(int(pos[0]))
            self.spin_roi_y.setValue(int(pos[1]))
            self.spin_roi_w.setValue(int(size[0]))
            self.spin_roi_h.setValue(int(size[1]))
            
            self.spin_roi_x.blockSignals(False)
            self.spin_roi_y.blockSignals(False)
            self.spin_roi_w.blockSignals(False)
            self.spin_roi_h.blockSignals(False)

        def on_roi_dragged():
            update_spinboxes()
            update_marginal_regions()

        def on_x_region_changed():
            rx = self.roi_x_region.getRegion()
            self.roi.sigRegionChanged.disconnect(on_roi_dragged)
            self.roi.setPos((rx[0], self.roi.pos()[1]))
            self.roi.setSize((rx[1] - rx[0], self.roi.size()[1]))
            update_spinboxes()
            self.roi.sigRegionChanged.connect(on_roi_dragged)

        def on_y_region_changed():
            ry = self.roi_y_region.getRegion()
            self.roi.sigRegionChanged.disconnect(on_roi_dragged)
            self.roi.setPos((self.roi.pos()[0], ry[0]))
            self.roi.setSize((self.roi.size()[0], ry[1] - ry[0]))
            update_spinboxes()
            self.roi.sigRegionChanged.connect(on_roi_dragged)

        self.spin_roi_x.valueChanged.connect(on_roi_spin_changed)
        self.spin_roi_y.valueChanged.connect(on_roi_spin_changed)
        self.spin_roi_w.valueChanged.connect(on_roi_spin_changed)
        self.spin_roi_h.valueChanged.connect(on_roi_spin_changed)
        self.roi.sigRegionChanged.connect(on_roi_dragged)
        self.roi_x_region.sigRegionChanged.connect(on_x_region_changed)
        self.roi_y_region.sigRegionChanged.connect(on_y_region_changed)
        
        # Initialize spinbox values based on initial ROI
        on_roi_dragged()

        #Main
        main_layout.addWidget(self.images_splitter)
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(roi_controls_layout)

    def _on_scan_group_index_changed(self, index):
        """Emits scanGroupSelected when a valid scan group is selected.

        Args:
            index: Current combo box index (-1 if no selection).
        """
        if index >= 0:
            self.scanGroupSelected.emit(index)

    def set_scan_groups(self, groups):
        """Populates the scan group combo box.

        Args:
            groups: List of HDF5 group path strings.
        """
        self.combo_scan_groups.clear()
        self.combo_scan_groups.addItems(groups)
        self.combo_scan_groups.setEnabled(True)
        # Clear color levels lock when loading new file
        self._last_disp_type = None

    def set_original_image(self, image):
        """Displays a new image and updates marginal distribution plots.

        Preserves color levels between scans of the same display type
        so the user can compare them with a consistent color scale.
        Resets color levels when switching display types.

        Args:
            image: 2D numpy array to display.
        """
        # We only autolevel if the display mode just changed, so the user can keep 
        # a locked color scale between different scans to compare them perfectly.
        current_disp = self.combo_display.currentText()
        if not hasattr(self, '_last_disp_type') or self._last_disp_type != current_disp:
            self._last_disp_type = current_disp
            levels = None
        else:
            levels = self.hist.getLevels()
            
        if levels is None:
            self.img_item.setImage(image, autoLevels=True)
            # Reapply jet colormap if autolevels overwrote it internally
            self.hist.gradient.setColorMap(pg.colormap.get('jet', source='matplotlib'))
        else:
            self.img_item.setImage(image, autoLevels=False, levels=levels)

        # Adjust ROI if it's outside the bounds initially
        max_x, max_y = image.shape
        roi_pos = self.roi.pos()
        if roi_pos.x() == 0 and roi_pos.y() == 0:
            self.roi.setPos([max_x // 4, max_y // 4])
            self.roi.setSize([max_x // 2, max_y // 2])

        # Update marginals (image in pyqtgraph is typically (x, y) coordinates)
        x_margin = np.sum(image, axis=1) # Length max_x
        y_margin = np.sum(image, axis=0) # Length max_y
        
        self.data_curve_x.setData(np.arange(max_x), x_margin)
        self.data_curve_y.setData(y_margin, np.arange(max_y))

        if not self.first_image_loaded:
            self.first_image_loaded = True
            
        # Force the view to zoom into the new image dimensions (crucial for ROI cropping)
        # Calling this AFTER `setData` ensures the marginal axes scale properly.
        self.p_img.autoRange()
        self.p_x.autoRange()
        self.p_y.autoRange()
        
        # Clear fit curves and overlays when new image loaded
        self.fit_curve_x.setData([], [])
        self.fit_curve_y.setData([], [])
        self._clear_width_overlay()

    def set_fitted_image(self, image, roi_slice=None, popt=None):
        """Displays the fitted image and overlays fit marginals.

        Maps the fitted image coordinates back to full-image coordinates
        using the ROI offset, overlays fit marginals on the X/Y plots,
        and draws the cloud width indicator.

        Args:
            image: 2D numpy array of the fitted model.
            roi_slice: Optional tuple of slices for coordinate offset.
            popt: Optional fitted parameters [x0, y0, sigma_x, sigma_y,
                A, B] for drawing the cloud width overlay.
        """
        self.fitted_view.setImage(image, autoLevels=False)
        self.fitted_view.show()
        
        offset_x, offset_y = 0, 0
        if roi_slice is not None:
            offset_x = roi_slice[0].start if roi_slice[0].start is not None else 0
            offset_y = roi_slice[1].start if roi_slice[1].start is not None else 0
            
        # Overlay fit marginals
        x_margin_fit = np.sum(image, axis=1)
        y_margin_fit = np.sum(image, axis=0)
        self.fit_curve_x.setData(np.arange(len(x_margin_fit)) + offset_x, x_margin_fit)
        self.fit_curve_y.setData(y_margin_fit, np.arange(len(y_margin_fit)) + offset_y)

        # Draw cloud width overlay if we have fit params
        if popt is not None:
            self._draw_width_overlay(popt, offset_x, offset_y)

    def clear_fitted_image(self):
        """Hides the fitted image panel and clears all fit overlays."""
        self.fitted_view.clear()
        self.fitted_view.hide()
        self._clear_width_overlay()

    # --- Cloud Width Overlay Drawing ---

    def _draw_width_overlay(self, popt, offset_x=0, offset_y=0):
        """Draw a horizontal line indicating the cloud width centred on the fit.
        popt = [x0, y0, sigma_x, sigma_y, A, B] from the fit, where:
            x0 = column center (array axis 1)
            y0 = row center    (array axis 0)
            sigma_x = width along columns (array axis 1)
            sigma_y = width along rows    (array axis 0)
        pyqtgraph ImageItem maps: axis 0 → display X, axis 1 → display Y.
        offset_x = roi_slice[0].start (axis 0 offset → display X offset)
        offset_y = roi_slice[1].start (axis 1 offset → display Y offset)
        """
        fit_x0, fit_y0 = popt[0], popt[1]         # column, row in fit coords
        fit_sigma_x, fit_sigma_y = abs(popt[2]), abs(popt[3])  # col-width, row-width

        # Map to display coordinates (swap axes)
        display_cx = fit_y0 + offset_x    # row index  → display X
        display_cy = fit_x0 + offset_y    # col index  → display Y

        # Horizontal line uses the row-direction width (sigma_y) since
        # rows map to display-X
        diameter_px = WIDTH_FACTOR * fit_sigma_y
        pixel_mm = self.spin_pixel_size.value()
        diameter_mm = diameter_px * pixel_mm

        # Horizontal line from (cx - r, cy) to (cx + r, cy)
        half = diameter_px / 2.0
        self.width_line_x.setData([display_cx - half, display_cx + half],
                                  [display_cy, display_cy])
        self.width_line_x.show()

        # Text label above the line
        self.width_text.setText(f"{diameter_px:.1f} px | {diameter_mm:.4f} mm")
        self.width_text.setPos(display_cx, display_cy - 3)
        self.width_text.show()

        # Update the scale bar whenever a fit is drawn
        self._draw_scale_bar()

    def _clear_width_overlay(self):
        """Hides the cloud width line, label, and scale bar."""
        self.width_line_x.setData([], [])
        self.width_line_x.hide()
        self.width_text.hide()
        self.scale_bar_line.hide()
        self.scale_bar_text.hide()

    def _draw_scale_bar(self):
        """Draw a map-style scale bar in the bottom-left corner of the image."""
        img_data = self.img_item.image
        if img_data is None:
            return
        pixel_mm = self.spin_pixel_size.value()
        img_w, img_h = img_data.shape  # pyqtgraph images are (rows, cols)

        # Choose a nice round scale bar length in mm
        # Target ~15% of image width in pixels
        target_px = img_w * 0.15
        target_mm = target_px * pixel_mm
        # Round to a nice number
        nice_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        bar_mm = nice_values[0]
        for nv in nice_values:
            if nv <= target_mm * 1.5:
                bar_mm = nv
        bar_px = bar_mm / pixel_mm

        # Position: bottom-left corner with some padding
        margin_x = img_w * 0.05
        margin_y = img_h * 0.05
        x_start = margin_x
        y_pos = margin_y

        self.scale_bar_line.setData([x_start, x_start + bar_px], [y_pos, y_pos])
        self.scale_bar_line.show()

        self.scale_bar_text.setText(f"{bar_mm} mm")
        self.scale_bar_text.setPos(x_start + bar_px / 2, y_pos + img_h * 0.02)
        self.scale_bar_text.show()

    def get_roi_slice(self):
        """Returns the current ROI as a tuple of array slices.

        Returns:
            Tuple of (row_slice, col_slice) mapping the ROI rectangle
            to array indices, or None if no image is loaded.
        """
        img_data = self.img_item.image
        if img_data is None:
            return None
        slice_tuple, _ = self.roi.getArraySlice(img_data, self.img_item)
        return slice_tuple

    def enable_display_combo(self, enabled):
        """Enables or disables the image display type combo box.

        Args:
            enabled: True to enable, False to disable.
        """
        self.combo_display.setEnabled(enabled)

    def is_locked(self):
        """Returns True if the manual lock checkbox is checked."""
        return self.check_lock.isChecked()

    def get_pixel_size_mm(self):
        """Returns the current pixel size calibration in millimeters."""
        return self.spin_pixel_size.value()

    def data_changed(self, value, metadata, persist, mods):
        """ARTIQ applet callback — forwards to the controller.

        Args:
            value: Dictionary of dataset values.
            metadata: Dataset metadata.
            persist: Persistence flag.
            mods: Modification descriptors.
        """
        if self.controller:
            self.controller.data_changed(value, metadata, persist, mods)

    def set_roi_locked_visuals(self, locked):
        """Updates the ROI pen and button style to reflect lock state.

        Args:
            locked: True to show green 'locked' styling, False to reset
                to the default blue ROI pen.
        """
        if locked:
            self.roi.setPen(pg.mkPen('g', width=3))
            self.btn_set_roi.setText("ROI: SET (Locked)")
            self.btn_set_roi.setStyleSheet(
                "background-color: #28a745; color: white; font-weight: bold;"
            )
        else:
            # pyqtgraph pen color (0, 9) = palette color index 9 (blue)
            self.roi.setPen(pg.mkPen((0, 9), width=2))
            self.btn_set_roi.setText("Set ROI")
            self.btn_set_roi.setStyleSheet("")