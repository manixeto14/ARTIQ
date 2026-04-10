"""Main view widget for the MOT image analysis application.

Displays the OD/raw image with linked marginal distribution plots,
an interactive ROI, cloud width overlays, and a scale bar.
"""

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from model import WIDTH_FACTOR

class SingleImageView(QtWidgets.QWidget):
    """Encapsulates a single image viewer with marginals and optional local controls."""
    requestData = QtCore.pyqtSignal(int, int, str) # viewer_idx, scan_index, disp_type

    def __init__(self, viewer_idx, is_main=False):
        super().__init__()
        self.viewer_idx = viewer_idx
        self.is_main = is_main
        self._last_disp_type = None
        self.first_image_loaded = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not self.is_main:
            self.toolbar = QtWidgets.QHBoxLayout()
            
            self.lbl_fit = QtWidgets.QLabel("Fit Result")
            self.lbl_fit.setStyleSheet("font-weight: bold; color: #d35400; padding: 2px;")
            self.lbl_fit.hide()
            self.toolbar.addWidget(self.lbl_fit)
            
            self.combo_scan = QtWidgets.QComboBox()
            self.combo_scan.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            self.combo_scan.currentIndexChanged.connect(self._on_local_selection)
            
            self.combo_display = QtWidgets.QComboBox()
            self.combo_display.addItems(["OD", "Background", "With atoms", "Without atoms"])
            self.combo_display.currentTextChanged.connect(self._on_local_selection)
            
            self.toolbar.addWidget(self.combo_scan)
            self.toolbar.addWidget(self.combo_display)
            layout.addLayout(self.toolbar)

        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        self.p_y = self.glw.addPlot()
        self.p_y.setMaximumWidth(100 if not self.is_main else 150)
        self.p_y.hideAxis('bottom')
        
        self.p_img = self.glw.addPlot()
        self.p_img.hideAxis('bottom')
        self.p_img.hideAxis('left')
        self.p_img.setAspectLocked(True)
        
        self.glw.nextRow()
        self.glw.nextCol()
        
        self.p_x = self.glw.addPlot()
        self.p_x.setMaximumHeight(100 if not self.is_main else 150)
        self.p_x.hideAxis('left')

        self.p_x.setXLink(self.p_img)
        self.p_y.setYLink(self.p_img)

        self.img_item = pg.ImageItem()
        self.img_item.setColorMap(pg.colormap.get('jet', source='matplotlib'))
        self.p_img.addItem(self.img_item)

        self.width_line_x = pg.PlotDataItem(pen=pg.mkPen('black', width=2, style=QtCore.Qt.DashLine))
        self.p_img.addItem(self.width_line_x)
        self.width_text = pg.TextItem('', color='black', anchor=(0.5, 1.0))
        self.width_text.setFont(QtWidgets.QApplication.font())
        self.p_img.addItem(self.width_text)
        self.width_text.hide()
        self.width_line_x.hide()

        self.scale_bar_line = pg.PlotDataItem(pen=pg.mkPen('w', width=4))
        self.p_img.addItem(self.scale_bar_line)
        self.scale_bar_text = pg.TextItem('', color='b', anchor=(0.5, 1.0))
        self.p_img.addItem(self.scale_bar_text)
        self.scale_bar_line.hide()
        self.scale_bar_text.hide()

        self.data_curve_y = self.p_y.plot(pen='k')
        self.fit_curve_y = self.p_y.plot(pen='r')
        self.data_curve_x = self.p_x.plot(pen='k')
        self.fit_curve_x = self.p_x.plot(pen='r')

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        if self.is_main:
            self.glw.addItem(self.hist, row=0, col=2, rowspan=2)

    def _on_local_selection(self, _=None):
        if self.is_main: return
        idx = self.combo_scan.currentIndex()
        disp = self.combo_display.currentText()
        if idx >= 0:
            self.requestData.emit(self.viewer_idx, idx, disp)

    def set_scan_groups(self, groups):
        if not self.is_main:
            self.combo_scan.blockSignals(True)
            self.combo_scan.clear()
            self.combo_scan.addItems(groups)
            self.combo_scan.blockSignals(False)

    def set_image(self, image, disp_type=None, preserve_levels=True):
        if image is None:
            self.img_item.clear()
            return
            
        if not hasattr(self, '_last_disp_type') or self._last_disp_type != disp_type:
            self._last_disp_type = disp_type
            levels = None
        else:
            levels = self.hist.getLevels() if hasattr(self, 'hist') else None
            
        if not preserve_levels or levels is None:
            self.img_item.setImage(image, autoLevels=True)
            if hasattr(self, 'hist'):
                self.hist.gradient.setColorMap(pg.colormap.get('jet', source='matplotlib'))
        else:
            self.img_item.setImage(image, autoLevels=False, levels=levels)
            
        # Strictly enforce color map native to ImageItem
        self.img_item.setColorMap(pg.colormap.get('jet', source='matplotlib'))

        max_x, max_y = image.shape
        x_margin = np.sum(image, axis=1)
        y_margin = np.sum(image, axis=0)
        self.data_curve_x.setData(np.arange(max_x), x_margin)
        self.data_curve_y.setData(y_margin, np.arange(max_y))
        
        if not self.first_image_loaded:
            self.first_image_loaded = True
            
        self.p_img.autoRange()
        self.p_x.autoRange()
        self.p_y.autoRange()
        
        self.fit_curve_x.setData([], [])
        self.fit_curve_y.setData([], [])
        self.clear_overlays()

    def set_fit_overlays(self, image, offset_x=0, offset_y=0, popt=None, pixel_size_mm=1.0):
        # Overlays marginal curves and cloud width on top of the existing image (Viewer 1)
        x_margin_fit = np.sum(image, axis=1)
        y_margin_fit = np.sum(image, axis=0)
        self.fit_curve_x.setData(np.arange(len(x_margin_fit)) + offset_x, x_margin_fit)
        self.fit_curve_y.setData(y_margin_fit, np.arange(len(y_margin_fit)) + offset_y)
        
        if popt is not None:
            self.draw_width_overlay(popt, offset_x, offset_y, pixel_size_mm)

    def set_fitted_surface(self, image):
        # Displays a standalone fitted 2D Gaussian image (Viewer 2)
        self.img_item.setImage(image, autoLevels=True)
        self.img_item.setColorMap(pg.colormap.get('jet', source='matplotlib'))
        
        max_x, max_y = image.shape
        x_margin = np.sum(image, axis=1)
        y_margin = np.sum(image, axis=0)
        self.data_curve_x.setData(np.arange(max_x), x_margin)
        self.data_curve_y.setData(y_margin, np.arange(max_y))
        
        self.p_img.autoRange()
        self.p_x.autoRange()
        self.p_y.autoRange()
        
        self.fit_curve_x.setData([], [])
        self.fit_curve_y.setData([], [])
        self.clear_overlays()

    def clear_overlays(self):
        self.width_line_x.setData([], [])
        self.width_line_x.hide()
        self.width_text.hide()
        self.scale_bar_line.hide()
        self.scale_bar_text.hide()

    def clear(self):
        self.img_item.clear()
        self.data_curve_x.setData([], [])
        self.data_curve_y.setData([], [])
        self.fit_curve_x.setData([], [])
        self.fit_curve_y.setData([], [])
        self.clear_overlays()

    def draw_width_overlay(self, popt, offset_x, offset_y, pixel_size_mm):
        fit_x0, fit_y0 = popt[0], popt[1]         
        fit_sigma_x, fit_sigma_y = abs(popt[2]), abs(popt[3])
        display_cx = fit_y0 + offset_x    
        display_cy = fit_x0 + offset_y    
        diameter_px = WIDTH_FACTOR * fit_sigma_y
        diameter_mm = diameter_px * pixel_size_mm

        half = diameter_px / 2.0
        self.width_line_x.setData([display_cx - half, display_cx + half], [display_cy, display_cy])
        self.width_line_x.show()

        self.width_text.setText(f"{diameter_px:.1f} px | {diameter_mm:.4f} mm")
        self.width_text.setPos(display_cx, display_cy - 3)
        self.width_text.show()

        self._draw_scale_bar(pixel_size_mm)

    def _draw_scale_bar(self, pixel_size_mm):
        img_data = self.img_item.image
        if img_data is None: return
        img_w, img_h = img_data.shape  
        target_px = img_w * 0.15
        target_mm = target_px * pixel_size_mm
        nice_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        bar_mm = nice_values[0]
        for nv in nice_values:
            if nv <= target_mm * 1.5:
                bar_mm = nv
        bar_px = bar_mm / pixel_size_mm

        margin_x = img_w * 0.05
        margin_y = img_h * 0.05
        x_start = margin_x
        y_pos = margin_y

        self.scale_bar_line.setData([x_start, x_start + bar_px], [y_pos, y_pos])
        self.scale_bar_line.show()

        self.scale_bar_text.setText(f"{bar_mm} mm")
        self.scale_bar_text.setPos(x_start + bar_px / 2, y_pos + img_h * 0.02)
        self.scale_bar_text.show()

    def show_as_fit_result(self, is_fit):
        if self.is_main: return
        if is_fit:
            self.combo_scan.hide()
            self.combo_display.hide()
            self.lbl_fit.show()
        else:
            self.lbl_fit.hide()
            self.combo_scan.show()
            self.combo_display.show()

class FitView(QtWidgets.QWidget):
    loadScanGroupClicked = QtCore.pyqtSignal()
    scanGroupSelected = QtCore.pyqtSignal(int)
    displayTypeChanged = QtCore.pyqtSignal(str)
    fitRequested = QtCore.pyqtSignal(str)
    lockToggled = QtCore.pyqtSignal(bool)
    setRoiRequested = QtCore.pyqtSignal()
    resetRoiRequested = QtCore.pyqtSignal()
    openSequenceRequested = QtCore.pyqtSignal()
    calibrationChanged = QtCore.pyqtSignal(float)
    
    toggleGridRequested = QtCore.pyqtSignal(bool)
    requestExtraViewerImage = QtCore.pyqtSignal(int, int, str)

    def __init__(self, args, req):
        super().__init__()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.args = args
        self.req = req
        self.controller = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0,0,0,0)

        self.viewers = []
        for i in range(4):
            viewer = SingleImageView(viewer_idx=i, is_main=(i==0))
            self.viewers.append(viewer)
            self.grid_layout.addWidget(viewer, i//2, i%2)
            if i > 0:
                viewer.hide()
                viewer.requestData.connect(self.requestExtraViewerImage.emit)

        self.roi = pg.RectROI([0, 0], [50, 50], pen=(0, 9))
        self.viewers[0].p_img.addItem(self.roi)

        main_layout.addWidget(self.grid_widget)

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

        controls_layout.addWidget(QtWidgets.QLabel("px size (mm):"))
        self.spin_pixel_size = QtWidgets.QDoubleSpinBox()
        self.spin_pixel_size.setDecimals(6)
        self.spin_pixel_size.setRange(0.000001, 100.0)
        self.spin_pixel_size.setValue(0.007)
        self.spin_pixel_size.setSingleStep(0.001)
        self.spin_pixel_size.valueChanged.connect(self.calibrationChanged.emit)
        controls_layout.addWidget(self.spin_pixel_size)

        self.btn_toggle_grid = QtWidgets.QPushButton("Toggle 2x2 Grid")
        self.btn_toggle_grid.setCheckable(True)
        self.btn_toggle_grid.toggled.connect(self.toggleGridRequested.emit)
        controls_layout.addWidget(self.btn_toggle_grid)

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
        
        self.btn_seq = QtWidgets.QPushButton("Sequence Window")
        self.btn_seq.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.btn_seq.clicked.connect(self.openSequenceRequested.emit)
        controls_layout.addWidget(self.btn_seq)

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

        self.roi_x_region = pg.LinearRegionItem([0, 50], orientation=pg.LinearRegionItem.Vertical)
        self.roi_x_region.setZValue(10)
        self.viewers[0].p_x.addItem(self.roi_x_region)
        self.roi_x_region.setBrush(pg.mkBrush((0, 255, 0, 50)))

        self.roi_y_region = pg.LinearRegionItem([0, 50], orientation=pg.LinearRegionItem.Horizontal)
        self.roi_y_region.setZValue(10)
        self.viewers[0].p_y.addItem(self.roi_y_region)
        self.roi_y_region.setBrush(pg.mkBrush((0, 255, 0, 50)))

        def on_roi_spin_changed():
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
        
        on_roi_dragged()

        main_layout.addLayout(controls_layout)
        main_layout.addLayout(roi_controls_layout)

    def _on_scan_group_index_changed(self, index):
        if index >= 0:
            self.scanGroupSelected.emit(index)

    def set_scan_groups(self, groups):
        self.combo_scan_groups.clear()
        self.combo_scan_groups.addItems(groups)
        self.combo_scan_groups.setEnabled(True)
        for i in range(1, 4):
            self.viewers[i].set_scan_groups(groups)

    def set_original_image(self, image):
        disp = self.combo_display.currentText()
        self.viewers[0].set_image(image, disp)
        
        max_x, max_y = image.shape
        roi_pos = self.roi.pos()
        if roi_pos.x() == 0 and roi_pos.y() == 0:
            self.roi.setPos([max_x // 4, max_y // 4])
            self.roi.setSize([max_x // 2, max_y // 2])

    def set_fitted_image(self, image, roi_slice=None, popt=None):
        self.viewers[1].show()
        self.viewers[1].show_as_fit_result(True)
        offset_x = roi_slice[0].start if roi_slice and roi_slice[0].start is not None else 0
        offset_y = roi_slice[1].start if roi_slice and roi_slice[1].start is not None else 0
        px_mm = self.get_pixel_size_mm()
        
        # Put the 2D Gaussian surface in viewer 2
        self.viewers[1].set_fitted_surface(image)
        # Overlay the red marginal curves and cloud width marker onto viewer 1
        self.viewers[0].set_fit_overlays(image, offset_x, offset_y, popt, px_mm)

    def clear_fitted_image(self):
        self.viewers[1].show_as_fit_result(False)
        self.viewers[1].clear()
        if not self.btn_toggle_grid.isChecked():
            self.viewers[1].hide()

    def set_grid_visible(self, visible):
        for i in range(1, 4):
            if visible:
                self.viewers[i].show()
                # Trigger an initial image load if they didn't have one
                if self.viewers[i].combo_scan.count() > 0 and self.viewers[i].img_item.image is None:
                    self.viewers[i]._on_local_selection()
            else:
                self.viewers[i].hide()

    def get_roi_slice(self):
        img_data = self.viewers[0].img_item.image
        if img_data is None:
            return None
        slice_tuple, _ = self.roi.getArraySlice(img_data, self.viewers[0].img_item)
        return slice_tuple

    def enable_display_combo(self, enabled):
        self.combo_display.setEnabled(enabled)

    def is_locked(self):
        return self.check_lock.isChecked()

    def get_pixel_size_mm(self):
        return self.spin_pixel_size.value()

    def set_roi_locked_visuals(self, locked):
        if locked:
            self.roi.setPen(pg.mkPen('g', width=3))
            self.btn_set_roi.setText("ROI: SET (Locked)")
            self.btn_set_roi.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        else:
            self.roi.setPen(pg.mkPen((0, 9), width=2))
            self.btn_set_roi.setText("Set ROI")
            self.btn_set_roi.setStyleSheet("")