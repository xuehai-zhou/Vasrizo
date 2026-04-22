"""Right-side panel: primary action buttons + small tunables."""
from __future__ import annotations
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QCheckBox, QGroupBox,
    QFormLayout, QDoubleSpinBox, QLabel, QHBoxLayout,
)


class ControlsPanel(QWidget):
    # Visibility toggles
    toggleLabel = Signal(bool)
    toggleCT = Signal(bool)
    # Actions
    trace = Signal()
    undoPath = Signal()
    clearPaths = Signal()
    save = Signal()
    resetView = Signal()
    autoLockRootPlane = Signal()
    cameraViewRequested = Signal(str)
    turntableChanged = Signal(bool)
    # Noise deletion
    startDeletion = Signal()
    undoDeletion = Signal()
    # Pot-wall peel (mm_xy, mm_base)
    applyPotWall = Signal(float, float)
    # Tuning knobs
    fillRadiusChanged = Signal(float)
    betaChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # --- Visibility ---
        vis_box = QGroupBox("Visibility")
        vis_lay = QVBoxLayout(vis_box)
        self.cb_label = QCheckBox("Show label (orange)")
        self.cb_label.setChecked(True)
        self.cb_ct = QCheckBox("Show thresholded CT (brown)")
        self.cb_ct.setChecked(True)
        vis_lay.addWidget(self.cb_label)
        vis_lay.addWidget(self.cb_ct)
        self.cb_label.toggled.connect(self.toggleLabel.emit)
        self.cb_ct.toggled.connect(self.toggleCT.emit)
        root.addWidget(vis_box)

        # --- Pot-wall peel ---
        # Replaces the slow batch preprocess_imagesTr.py step: we compute
        # the interior mask live via a 2D per-slice EDT. Apply-on-click
        # because the EDT takes a few seconds on full volumes — too slow
        # for live slider drag.
        pw_box = QGroupBox("Pot wall peel")
        pw_form = QFormLayout(pw_box)
        self.sp_peel_xy = QDoubleSpinBox()
        self.sp_peel_xy.setRange(0.0, 50.0)
        self.sp_peel_xy.setSingleStep(0.5)
        self.sp_peel_xy.setValue(15.0)
        self.sp_peel_xy.setSuffix(" mm")
        self.sp_peel_xy.setToolTip(
            "Radial peel from the outer pot wall (2D per-slice EDT). "
            "15 mm is equivalent to the current batch preprocessing.")
        pw_form.addRow("Outer wall:", self.sp_peel_xy)

        self.sp_peel_base = QDoubleSpinBox()
        self.sp_peel_base.setRange(0.0, 50.0)
        self.sp_peel_base.setSingleStep(0.5)
        self.sp_peel_base.setValue(0.0)
        self.sp_peel_base.setSuffix(" mm")
        self.sp_peel_base.setToolTip(
            "Extra trim from the pot base (frustum bottom), applied on "
            "top of the outer-wall peel.")
        pw_form.addRow("Base:", self.sp_peel_base)

        self.btn_apply_pot_wall = QPushButton("Apply pot-wall peel")
        self.btn_apply_pot_wall.setToolTip(
            "Re-compute the interior mask using the values above. Takes "
            "a few seconds per volume.")
        pw_form.addRow(self.btn_apply_pot_wall)
        self.btn_apply_pot_wall.clicked.connect(
            lambda: self.applyPotWall.emit(
                self.sp_peel_xy.value(), self.sp_peel_base.value()))
        root.addWidget(pw_box)

        # --- Actions ---
        act_box = QGroupBox("Annotation")
        act_lay = QVBoxLayout(act_box)
        self.btn_trace = QPushButton("Trace between waypoints")
        self.btn_trace.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; "
            "background: #4a90e2; color: white; border-radius: 4px; }"
            "QPushButton:hover { background: #5aa0f2; }"
            "QPushButton:disabled { background: #aaa; }")
        self.btn_undo_path = QPushButton("Undo last path")
        self.btn_clear = QPushButton("Clear all paths")
        self.btn_save = QPushButton("Save refined label")
        self.btn_save.setStyleSheet(
            "QPushButton { padding: 6px; font-weight: bold; }")
        self.btn_reset_view = QPushButton("Reset view")
        self.btn_auto_lock_plane = QPushButton("Auto fit root slab")
        self.btn_auto_lock_plane.setToolTip(
            "Use the current waypoints as hints and fit the current "
            "screen-parallel front/back slab to the target root segment.")
        act_lay.addWidget(self.btn_trace)
        act_lay.addWidget(self.btn_undo_path)
        act_lay.addWidget(self.btn_clear)
        act_lay.addSpacing(4)
        act_lay.addWidget(self.btn_save)
        act_lay.addWidget(self.btn_auto_lock_plane)
        act_lay.addWidget(self.btn_reset_view)
        self.btn_trace.clicked.connect(self.trace.emit)
        self.btn_undo_path.clicked.connect(self.undoPath.emit)
        self.btn_clear.clicked.connect(self.clearPaths.emit)
        self.btn_save.clicked.connect(self.save.emit)
        self.btn_auto_lock_plane.clicked.connect(self.autoLockRootPlane.emit)
        self.btn_reset_view.clicked.connect(self.resetView.emit)
        root.addWidget(act_box)

        cam_box = QGroupBox("Camera")
        cam_lay = QVBoxLayout(cam_box)
        cam_lay.addWidget(QLabel("Axis colors: X red, Y green, Z blue"))

        row1 = QHBoxLayout()
        self.btn_view_px = QPushButton("+X")
        self.btn_view_nx = QPushButton("-X")
        self.btn_view_py = QPushButton("+Y")
        row1.addWidget(self.btn_view_px)
        row1.addWidget(self.btn_view_nx)
        row1.addWidget(self.btn_view_py)
        cam_lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_view_ny = QPushButton("-Y")
        self.btn_view_pz = QPushButton("+Z")
        self.btn_view_nz = QPushButton("-Z")
        row2.addWidget(self.btn_view_ny)
        row2.addWidget(self.btn_view_pz)
        row2.addWidget(self.btn_view_nz)
        cam_lay.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_view_iso = QPushButton("ISO")
        row3.addWidget(self.btn_view_iso)
        cam_lay.addLayout(row3)

        self.cb_turntable = QCheckBox("Turntable orbit (left-drag)")
        self.cb_turntable.setChecked(True)
        self.cb_turntable.setToolTip(
            "More stable orbiting than free trackball rotation. Keeps the "
            "camera easier to understand relative to the axes.")
        cam_lay.addWidget(self.cb_turntable)

        self.btn_view_px.clicked.connect(lambda: self.cameraViewRequested.emit("+X"))
        self.btn_view_nx.clicked.connect(lambda: self.cameraViewRequested.emit("-X"))
        self.btn_view_py.clicked.connect(lambda: self.cameraViewRequested.emit("+Y"))
        self.btn_view_ny.clicked.connect(lambda: self.cameraViewRequested.emit("-Y"))
        self.btn_view_pz.clicked.connect(lambda: self.cameraViewRequested.emit("+Z"))
        self.btn_view_nz.clicked.connect(lambda: self.cameraViewRequested.emit("-Z"))
        self.btn_view_iso.clicked.connect(lambda: self.cameraViewRequested.emit("ISO"))
        self.cb_turntable.toggled.connect(self.turntableChanged.emit)
        root.addWidget(cam_box)

        # --- Noise deletion (CloudCompare-style segment tool) ---
        del_box = QGroupBox("Noise deletion")
        del_lay = QVBoxLayout(del_box)
        self.btn_start_deletion = QPushButton("Delete noise points…")
        self.btn_start_deletion.setToolTip(
            "Orient the view first, then start. Left-click to add polyline "
            "vertices; right-click to chain a new segment; Enter to close "
            "the loop and preview; Delete to confirm; Esc to cancel.")
        self.btn_start_deletion.setStyleSheet(
            "QPushButton { padding: 6px; background: #c0392b; color: white; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #d04433; }"
            "QPushButton:disabled { background: #aaa; }")
        self.btn_undo_deletion = QPushButton("Undo last deletion")
        del_lay.addWidget(self.btn_start_deletion)
        del_lay.addWidget(self.btn_undo_deletion)
        self.btn_start_deletion.clicked.connect(self.startDeletion.emit)
        self.btn_undo_deletion.clicked.connect(self.undoDeletion.emit)
        root.addWidget(del_box)

        # --- Tuning knobs ---
        tun_box = QGroupBox("Tuning")
        tun_form = QFormLayout(tun_box)
        self.sp_radius = QDoubleSpinBox()
        self.sp_radius.setRange(0.05, 5.0)
        self.sp_radius.setSingleStep(0.05)
        self.sp_radius.setValue(0.5)
        self.sp_radius.setSuffix(" mm")
        self.sp_radius.valueChanged.connect(self.fillRadiusChanged.emit)
        tun_form.addRow("Fill radius:", self.sp_radius)

        self.sp_beta = QDoubleSpinBox()
        self.sp_beta.setRange(0.1, 1.0)
        self.sp_beta.setSingleStep(0.05)
        self.sp_beta.setValue(0.40)
        self.sp_beta.setDecimals(2)
        self.sp_beta.valueChanged.connect(self.betaChanged.emit)
        tun_form.addRow("JS β:", self.sp_beta)
        root.addWidget(tun_box)

        root.addStretch(1)

    # ------------------------------------------------------------------
    # External setters (used when switching tabs so controls match state)
    # ------------------------------------------------------------------

    def set_state(self, show_label: bool, show_ct: bool,
                  fill_radius_mm: float, beta: float):
        self.cb_label.blockSignals(True)
        self.cb_ct.blockSignals(True)
        self.sp_radius.blockSignals(True)
        self.sp_beta.blockSignals(True)
        self.cb_label.setChecked(show_label)
        self.cb_ct.setChecked(show_ct)
        self.sp_radius.setValue(fill_radius_mm)
        self.sp_beta.setValue(beta)
        self.cb_label.blockSignals(False)
        self.cb_ct.blockSignals(False)
        self.sp_radius.blockSignals(False)
        self.sp_beta.blockSignals(False)

    def set_trace_enabled(self, enabled: bool):
        self.btn_trace.setEnabled(enabled)
