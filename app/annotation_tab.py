"""One tab in the main window = one sample being annotated.

Owns its DocumentState, 3D viewer, threshold slider, controls panel, and
waypoint panel. Nothing leaks between tabs.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QFrame,
    QMessageBox, QFileDialog,
)

from .models.app_settings import AppSettings
from .models.document_state import DocumentState, Waypoint
from .io.data_loader import SampleData
# save_label is invoked indirectly via SaveWorker
from .services.root_model import RootModel
from .services.tracing_service import paint_tube, paint_tube_local
from .services.threshold_service import threshold_ct_to_coords
from .utils.config import (
    SPACING, HU_LOWER_DEFAULT, HU_UPPER_DEFAULT, HU_STEP,
    HU_MIN_LIMIT, HU_MAX_LIMIT, POINT_SIZE_DEFAULT,
)
from .utils.geometry_utils import label_to_coords, voxel_to_physical

from .ui.viewer_3d import Viewer3D
from .ui.threshold_range_slider import ThresholdRangeSlider
from .ui.point_size_slider import PointSizeSlider
from .ui.screen_slice_slider import ScreenSliceSlider
from .ui.controls_panel import ControlsPanel
from .ui.waypoint_panel import WaypointPanel
from .services.deletion_service import build_voxel_deletion_mask

from .workers.load_worker import LoadWorker
from .workers.trace_worker import TraceWorker
from .workers.threshold_worker import ThresholdWorker
from .workers.save_worker import SaveWorker
from .workers.pot_wall_worker import PotWallWorker


class AnnotationTab(QWidget):
    """Self-contained editor widget for one sample."""

    # Emitted so the MainWindow can update the tab title, mark dirty, etc.
    statusMessage = Signal(str)
    dirtyChanged = Signal(bool)
    sampleSaved = Signal(str)   # emits sample name

    def __init__(self, sample_name: str, settings: AppSettings,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.sample_name = sample_name
        self.settings = settings

        # Per-tab state (populated after LoadWorker finishes)
        self.doc: Optional[DocumentState] = None

        # Threads we own (must keep references so they're not GC'd)
        self._load_thread: Optional[QThread] = None
        self._load_worker: Optional[LoadWorker] = None
        self._trace_thread: Optional[QThread] = None
        self._trace_worker: Optional[TraceWorker] = None
        self._thr_thread: Optional[QThread] = None
        self._thr_worker: Optional[ThresholdWorker] = None
        self._save_thread: Optional[QThread] = None
        self._save_worker: Optional[SaveWorker] = None
        self._potwall_thread: Optional[QThread] = None
        self._potwall_worker = None

        self._build_ui()
        self._start_loading()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Center column: viewer + threshold slider + a pick-button row
        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(4, 4, 4, 4)
        center_lay.setSpacing(6)

        self.viewer = Viewer3D(self, point_size=POINT_SIZE_DEFAULT)
        center_lay.addWidget(self.viewer, 1)
        self.viewer.set_camera_changed_callback(self._on_camera_changed)

        # Noise-deletion polyline is drawn inside VTK (see DeletionController);
        # the Viewer3D exposes signals we react to here.
        self.viewer.polylineFinished.connect(self._on_polyline_finished)
        self.viewer.deletionCancelled.connect(self._on_deletion_cancel)
        self.viewer.statusMessage.connect(self.statusMessage.emit)

        # Deletion mode state
        self._deletion_mode: bool = False
        self._deletion_preview_mask: Optional[np.ndarray] = None

        # Point-size slider row (sits directly below the 3D viewer)
        ps_box = QFrame()
        ps_box.setFrameShape(QFrame.StyledPanel)
        ps_lay = QVBoxLayout(ps_box)
        ps_lay.setContentsMargins(0, 0, 0, 0)
        self.point_size_slider = PointSizeSlider(
            minimum=1, maximum=20, initial=POINT_SIZE_DEFAULT)
        ps_lay.addWidget(self.point_size_slider)
        center_lay.addWidget(ps_box)

        slice_box = QFrame()
        slice_box.setFrameShape(QFrame.StyledPanel)
        slice_lay = QVBoxLayout(slice_box)
        slice_lay.setContentsMargins(0, 0, 0, 0)
        self.screen_slice_slider = ScreenSliceSlider(
            minimum=-100, maximum=100, initial=0.0)
        slice_lay.addWidget(self.screen_slice_slider)
        center_lay.addWidget(slice_box)

        # Threshold control row
        thr_box = QFrame()
        thr_box.setFrameShape(QFrame.StyledPanel)
        thr_lay = QVBoxLayout(thr_box)
        thr_lay.setContentsMargins(8, 6, 8, 4)
        head = QHBoxLayout()
        head.addWidget(QLabel("<b>HU threshold</b>"))
        head.addStretch(1)
        self.lbl_thr_values = QLabel("")
        self.lbl_thr_values.setStyleSheet("color: #444; font-family: monospace;")
        head.addWidget(self.lbl_thr_values)
        thr_lay.addLayout(head)
        self.slider = ThresholdRangeSlider(
            minimum=HU_MIN_LIMIT, maximum=HU_MAX_LIMIT, step=HU_STEP,
            lower=HU_LOWER_DEFAULT, upper=HU_UPPER_DEFAULT)
        thr_lay.addWidget(self.slider)
        center_lay.addWidget(thr_box)

        # Loading placeholder
        self.lbl_loading = QLabel(f"Loading {self.sample_name}…")
        self.lbl_loading.setAlignment(Qt.AlignCenter)
        self.lbl_loading.setStyleSheet(
            "color: #888; font-size: 14pt; padding: 40px;")
        center_lay.addWidget(self.lbl_loading)

        # Right column: controls + waypoint list
        right = QWidget()
        right.setMaximumWidth(340)
        right.setMinimumWidth(280)
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(0)
        self.controls = ControlsPanel()
        self.waypoints = WaypointPanel()
        r_lay.addWidget(self.controls)
        r_lay.addWidget(self.waypoints, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 320])
        root.addWidget(splitter)

        # Wire signals
        self.slider.valuesChanged.connect(self._on_slider_values_changed)
        self.slider.valuesCommitted.connect(self._on_slider_commit)
        self.controls.toggleLabel.connect(self._on_toggle_label)
        self.controls.toggleCT.connect(self._on_toggle_ct)
        self.controls.trace.connect(self._on_trace)
        self.controls.undoPath.connect(self._on_undo_path)
        self.controls.clearPaths.connect(self._on_clear_paths)
        self.controls.save.connect(self._on_save)
        self.controls.resetView.connect(lambda: self.viewer.reset_view())
        self.controls.fillRadiusChanged.connect(self._on_radius_changed)
        self.controls.betaChanged.connect(self._on_beta_changed)
        self.controls.startDeletion.connect(self._on_start_deletion)
        self.controls.undoDeletion.connect(self._on_undo_deletion)
        self.controls.applyPotWall.connect(self._on_apply_pot_wall)

        self.waypoints.undoLast.connect(self._on_undo_last_waypoint)
        self.waypoints.deleteSelected.connect(self._on_delete_waypoint)
        self.waypoints.clearAll.connect(self._on_clear_waypoints)

        self.point_size_slider.valueChanged.connect(self._on_point_size_changed)
        self.screen_slice_slider.enabledChanged.connect(
            self._on_screen_slice_enabled_changed)
        self.screen_slice_slider.offsetChanged.connect(
            self._on_screen_slice_offset_changed)
        self.screen_slice_slider.reverseChanged.connect(
            self._on_screen_slice_reverse_changed)
        self.screen_slice_slider.lockedChanged.connect(
            self._on_screen_slice_locked_changed)
        self.screen_slice_slider.thicknessChanged.connect(
            self._on_screen_slice_thickness_changed)
        self.screen_slice_slider.guidesChanged.connect(
            self._on_screen_slice_guides_changed)
        self.screen_slice_slider.freezeDirectionChanged.connect(
            self._on_screen_slice_freeze_direction_changed)

        # Live values label
        self._refresh_thr_label(*self.slider.values())

        # Disable trace until doc is loaded
        self.controls.set_trace_enabled(False)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _start_loading(self):
        self._load_thread = QThread(self)
        self._load_worker = LoadWorker(
            self.sample_name,
            labels_dir=self.settings.labels_dir,
            images_dir=self.settings.images_dir,
            beta=0.40, col_diff=80.0,
        )
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self.statusMessage.emit)
        self._load_worker.finished.connect(self._on_loaded)
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_thread.start()

    def _on_load_failed(self, msg: str):
        self.lbl_loading.setText(f"Failed to load {self.sample_name}\n{msg}")
        self.lbl_loading.setStyleSheet("color: #c0392b; padding: 40px;")
        self.statusMessage.emit(f"Load failed: {msg}")

    def _on_loaded(self, sample: SampleData, model: RootModel):
        self.doc = DocumentState(sample)
        self.doc.root_model = model
        self.lbl_loading.setVisible(False)

        # Push current state into the UI
        self.controls.set_state(
            show_label=self.doc.show_label, show_ct=self.doc.show_ct,
            fill_radius_mm=self.doc.fill_radius_mm, beta=self.doc.beta)
        self.point_size_slider.set_value(self.doc.point_size)
        self.screen_slice_slider.set_enabled(self.doc.screen_slice_enabled)
        self.screen_slice_slider.set_offset(
            self.doc.screen_slice_offset_mm, emit=False)
        self.screen_slice_slider.set_reverse(self.doc.screen_slice_reverse)
        self.screen_slice_slider.set_locked(self.doc.screen_slice_locked)
        self.screen_slice_slider.set_thickness(
            self.doc.screen_slice_thickness_mm)
        self.screen_slice_slider.set_guides(
            self.doc.screen_slice_show_guides)
        self.screen_slice_slider.set_freeze_direction(
            self.doc.screen_slice_freeze_direction)
        self.viewer.set_point_size(self.doc.point_size)

        # Initial cloud rendering
        self._refresh_label_cloud()
        self._refresh_ct_cloud()
        self._refresh_waypoints()
        self._refresh_paths()
        self.viewer.reset_view()
        self._sync_screen_slice_to_camera(force_range=True)

        # Enable picking
        self.viewer.pointPicked.connect(self._on_point_picked)
        self.viewer.enable_picking(lambda p, s: None)

        self.controls.set_trace_enabled(True)
        self.statusMessage.emit(
            f"Loaded {sample.name}: shape={sample.image.shape}, "
            f"voxels={int(sample.label.sum()):,}")

    # ------------------------------------------------------------------
    # Cloud refresh helpers
    # ------------------------------------------------------------------

    def _refresh_label_cloud(self):
        if self.doc is None:
            return
        coords = label_to_coords(self.doc.label, downsample=1)
        self.viewer.set_label_points(coords, visible=self.doc.show_label)

    def _refresh_ct_cloud(self):
        """Threshold overlay is computed directly from the image volume.

        If the pot-wall peel has populated `sample.interior_mask`, it is
        AND-ed into the threshold result so pot walls are suppressed.
        """
        if self.doc is None:
            return
        if self.doc.sample.image is None:
            self.viewer.set_ct_points(np.empty((0, 3)), visible=False)
            return
        self._kick_threshold_worker()

    def _kick_threshold_worker(self):
        if self.doc is None or self.doc.sample.image is None:
            return
        # Detach any in-flight worker (coarse — old result just gets discarded)
        if self._thr_thread is not None and self._thr_thread.isRunning():
            self._thr_worker = None
        self._thr_thread = QThread(self)
        self._thr_worker = ThresholdWorker(
            ct_vol=self.doc.sample.image,
            interior=self.doc.sample.interior_mask,  # None is fine
            hu_lower=self.doc.hu_lower,
            hu_upper=self.doc.hu_upper,
            downsample=1,
        )
        self._thr_worker.moveToThread(self._thr_thread)
        self._thr_thread.started.connect(self._thr_worker.run)
        self._thr_worker.finished.connect(self._on_threshold_done)
        self._thr_worker.finished.connect(self._thr_thread.quit)
        self._thr_worker.failed.connect(self._thr_thread.quit)
        self._thr_thread.start()

    def _on_threshold_done(self, coords):
        if self.doc is None:
            return
        self.viewer.set_ct_points(coords, visible=self.doc.show_ct)

    def _refresh_waypoints(self):
        if self.doc is None:
            return
        self.waypoints.update_from_waypoints(self.doc.waypoints)
        phys = [wp.phys for wp in self.doc.waypoints]
        self.viewer.update_waypoints(phys)

    def _refresh_paths(self):
        if self.doc is None:
            return
        phys_paths = []
        for tp in self.doc.traced_paths:
            phys_paths.append(voxel_to_physical(tp.path_voxels))
        self.viewer.update_traced_paths(phys_paths)

    # ------------------------------------------------------------------
    # Threshold slider
    # ------------------------------------------------------------------

    def _refresh_thr_label(self, lo: float, up: float):
        self.lbl_thr_values.setText(f"[{lo:.0f}, {up:.0f}]  (step {int(HU_STEP)})")

    def _on_slider_values_changed(self, lo: float, up: float):
        self._refresh_thr_label(lo, up)

    def _on_slider_commit(self, lo: float, up: float):
        if self.doc is None:
            return
        if (self.doc.hu_lower, self.doc.hu_upper) == (lo, up):
            return
        self.doc.hu_lower = lo
        self.doc.hu_upper = up
        self._kick_threshold_worker()

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def _on_toggle_label(self, on: bool):
        if self.doc is None:
            return
        self.doc.show_label = on
        self.viewer.set_label_visible(on)

    def _on_point_size_changed(self, size: float):
        if self.doc is not None:
            self.doc.point_size = float(size)
        self.viewer.set_point_size(size)

    def _on_screen_slice_enabled_changed(self, enabled: bool):
        self.screen_slice_slider.set_enabled(enabled)
        if self.doc is None:
            return
        self.doc.screen_slice_enabled = bool(enabled)
        if enabled and self.doc.screen_slice_freeze_direction:
            self._capture_frozen_screen_slice_frame()
        self._sync_screen_slice_to_camera(force_range=True)

    def _on_screen_slice_offset_changed(self, offset_mm: float):
        if self.doc is None:
            return
        self.doc.screen_slice_offset_mm = float(offset_mm)
        self._sync_screen_slice_to_camera(force_range=False)

    def _on_screen_slice_reverse_changed(self, enabled: bool):
        if self.doc is None:
            return
        self.doc.screen_slice_reverse = bool(enabled)
        self._sync_screen_slice_to_camera(force_range=False)

    def _on_screen_slice_locked_changed(self, enabled: bool):
        if self.doc is None:
            return
        self.doc.screen_slice_locked = bool(enabled)
        self.screen_slice_slider.set_locked(enabled)
        self._sync_screen_slice_to_camera(force_range=True)

    def _on_screen_slice_thickness_changed(self, thickness_mm: float):
        if self.doc is None:
            return
        self.doc.screen_slice_thickness_mm = max(0.1, float(thickness_mm))
        self._sync_screen_slice_to_camera(force_range=False)

    def _on_screen_slice_guides_changed(self, enabled: bool):
        if self.doc is None:
            return
        self.doc.screen_slice_show_guides = bool(enabled)
        self._sync_screen_slice_to_camera(force_range=False)

    def _on_screen_slice_freeze_direction_changed(self, enabled: bool):
        if self.doc is None:
            return
        self.doc.screen_slice_freeze_direction = bool(enabled)
        self.screen_slice_slider.set_freeze_direction(enabled)
        if enabled:
            self._capture_frozen_screen_slice_frame()
        else:
            self.doc.screen_slice_frozen_origin = None
            self.doc.screen_slice_frozen_normal = None
        self._sync_screen_slice_to_camera(force_range=True)

    def _on_toggle_ct(self, on: bool):
        if self.doc is None:
            return
        self.doc.show_ct = on
        self.viewer.set_ct_visible(on)

    def _on_camera_changed(self):
        if self.doc is None or not self.doc.screen_slice_enabled:
            return
        if self.doc.screen_slice_freeze_direction:
            return
        self._sync_screen_slice_to_camera(force_range=True)

    def _volume_corners_phys(self) -> np.ndarray:
        if self.doc is None or self.doc.sample.image is None:
            return np.empty((0, 3), dtype=np.float64)
        shape = np.asarray(self.doc.sample.image.shape, dtype=np.int64)
        max_idx = np.maximum(shape - 1, 0)
        corners = []
        for x in (0, int(max_idx[0])):
            for y in (0, int(max_idx[1])):
                for z in (0, int(max_idx[2])):
                    corners.append((x, y, z))
        return voxel_to_physical(np.asarray(corners, dtype=np.float64))

    def _current_view_frame(self):
        pos, focal = self.viewer.current_camera_pose()
        if pos is None or focal is None:
            return None, None
        normal = np.asarray(focal - pos, dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            return None, None
        return np.asarray(focal, dtype=np.float64), normal / norm

    def _capture_frozen_screen_slice_frame(self):
        if self.doc is None:
            return
        origin, normal = self._current_view_frame()
        if origin is None or normal is None:
            return
        self.doc.screen_slice_frozen_origin = origin.copy()
        self.doc.screen_slice_frozen_normal = normal.copy()

    def _sync_screen_slice_to_camera(self, force_range: bool):
        if self.doc is None:
            return
        if self.doc.screen_slice_freeze_direction:
            if (self.doc.screen_slice_frozen_origin is None or
                    self.doc.screen_slice_frozen_normal is None):
                self._capture_frozen_screen_slice_frame()
            focal = self.doc.screen_slice_frozen_origin
            normal = self.doc.screen_slice_frozen_normal
            if focal is None or normal is None:
                return
            focal = np.asarray(focal, dtype=np.float64)
            normal = np.asarray(normal, dtype=np.float64)
        else:
            focal, normal = self._current_view_frame()
            if focal is None or normal is None:
                return

        corners = self._volume_corners_phys()
        if force_range:
            if len(corners) > 0:
                rel = (corners - focal[None, :]) @ normal
                depth = float(rel.max() - rel.min())
                pad = max(5.0, 0.05 * depth)
                lo = float(np.floor(rel.min() - pad))
                hi = float(np.ceil(rel.max() + pad))
                self.screen_slice_slider.set_range(lo, hi)
                clamped = self.screen_slice_slider.offset()
                self.doc.screen_slice_offset_mm = clamped
                thickness_max = max(5.0, depth + 2.0 * pad)
                self.screen_slice_slider.set_thickness_range(0.1, thickness_max)
                self.doc.screen_slice_thickness_mm = max(
                    self.screen_slice_slider.thickness_minimum(),
                    min(thickness_max, self.doc.screen_slice_thickness_mm),
                )
                self.screen_slice_slider.set_thickness(
                    self.doc.screen_slice_thickness_mm)

        origin = focal + normal * float(self.doc.screen_slice_offset_mm)
        plane_size = 100.0
        if len(corners) > 0:
            mins = corners.min(axis=0)
            maxs = corners.max(axis=0)
            plane_size = 1.25 * float(np.linalg.norm(maxs - mins))
        self.viewer.set_screen_slice(
            enabled=self.doc.screen_slice_enabled,
            plane_origin=origin,
            plane_normal=normal,
            reverse=self.doc.screen_slice_reverse,
            locked=self.doc.screen_slice_locked,
            thickness_mm=self.doc.screen_slice_thickness_mm,
            show_guides=self.doc.screen_slice_show_guides,
            plane_size=plane_size,
        )

    # ------------------------------------------------------------------
    # Picking -> waypoints
    # ------------------------------------------------------------------

    def _on_point_picked(self, phys, source: str):
        if self.doc is None:
            return
        self.doc.add_waypoint(phys, source)
        self._refresh_waypoints()
        self.statusMessage.emit(
            f"Picked waypoint {len(self.doc.waypoints)} ({source}) "
            f"at ({phys[0]:.1f}, {phys[1]:.1f}, {phys[2]:.1f})")

    def _on_undo_last_waypoint(self):
        if self.doc and self.doc.undo_last_waypoint():
            self._refresh_waypoints()

    def _on_delete_waypoint(self, index: int):
        if self.doc and self.doc.delete_waypoint(index):
            self._refresh_waypoints()

    def _on_clear_waypoints(self):
        if self.doc:
            self.doc.clear_waypoints()
            self._refresh_waypoints()

    # ------------------------------------------------------------------
    # Tracing
    # ------------------------------------------------------------------

    def _on_radius_changed(self, mm: float):
        if self.doc is not None:
            self.doc.fill_radius_mm = float(mm)

    def _on_beta_changed(self, beta: float):
        if self.doc is None:
            return
        self.doc.beta = float(beta)
        if self.doc.root_model is not None:
            self.doc.root_model.beta = float(beta)

    def _on_trace(self):
        if self.doc is None or len(self.doc.waypoints) < 2:
            QMessageBox.information(
                self, "Need waypoints",
                "Pick at least 2 waypoints (Shift+Click in the 3D view) "
                "before tracing.")
            return
        if self._trace_thread is not None and self._trace_thread.isRunning():
            QMessageBox.information(self, "Trace running",
                                    "A trace is already in progress.")
            return

        waypoints_phys = [wp.phys for wp in self.doc.waypoints]
        self._trace_thread = QThread(self)
        self._trace_worker = TraceWorker(
            image=self.doc.sample.image,
            model=self.doc.root_model,
            waypoints_phys=waypoints_phys,
            fill_radius_mm=self.doc.fill_radius_mm,
            mean_spacing=float(np.mean(SPACING)),
            # Enables the anchor early-termination trick: Dijkstra stops
            # as soon as it reaches a labeled voxel near the end
            # waypoint (see services/tracing_service.py for details).
            label=self.doc.label,
        )
        self._trace_worker.moveToThread(self._trace_thread)
        self._trace_thread.started.connect(self._trace_worker.run)
        self._trace_worker.progress.connect(self.statusMessage.emit)
        self._trace_worker.finished.connect(self._on_trace_finished)
        self._trace_worker.failed.connect(self._on_trace_failed)
        self._trace_worker.finished.connect(self._trace_thread.quit)
        self._trace_worker.failed.connect(self._trace_thread.quit)
        self.controls.set_trace_enabled(False)
        self.statusMessage.emit("Tracing…")
        self._trace_thread.start()

    def _on_trace_finished(self, path_voxels, r_vox: float):
        self.controls.set_trace_enabled(True)
        if path_voxels is None or len(path_voxels) == 0:
            self.statusMessage.emit("Trace: no path found.")
            return
        # Compute the dilated footprint once (bbox-local) and cache it on
        # the TracedPath so a future undo can rebuild the label by OR'ing
        # remaining footprints, without re-running binary_dilation over
        # the whole volume per path.
        bbox, footprint = paint_tube_local(
            self.doc.label.shape, path_voxels, r_vox)
        self.doc.add_traced_path(path_voxels, r_vox,
                                 bbox=bbox, footprint=footprint)
        if bbox is not None:
            self.doc.label[bbox] |= footprint
        self._refresh_label_cloud()
        self._refresh_paths()
        # After a successful trace, clear the waypoints so the user can
        # move on without accidentally reusing them.
        self.doc.clear_waypoints()
        self._refresh_waypoints()
        self.dirtyChanged.emit(True)
        self.statusMessage.emit(
            f"Trace added: {len(path_voxels)} voxels (radius "
            f"{self.doc.fill_radius_mm:.2f} mm).")

    def _on_trace_failed(self, msg: str):
        self.controls.set_trace_enabled(True)
        QMessageBox.warning(self, "Trace failed", msg)
        self.statusMessage.emit(f"Trace failed: {msg}")

    def _rebuild_label_from_state(self) -> np.ndarray:
        """Return a fresh label = label_orig + all traced paths − all deletions.

        Shared by undo-path and clear-paths. Crucially, the re-application
        of `doc.deleted_regions` at the end keeps prior noise-deletion
        work intact when a traced path is undone or cleared — without
        this step, we'd be rebuilding from `label_orig` and silently
        resurrecting every deleted voxel.
        """
        new_label = self.doc.label_orig.copy()
        for tp in self.doc.traced_paths:
            if tp.bbox is not None and tp.footprint is not None:
                new_label[tp.bbox] |= tp.footprint
            else:
                # Path added before the bbox cache existed — fall back
                # to the full-volume painter just for it.
                new_label = paint_tube(
                    new_label, tp.path_voxels, tp.radius_voxels)
        for del_mask in self.doc.deleted_regions:
            new_label &= ~del_mask
        return new_label

    def _on_undo_path(self):
        if self.doc and self.doc.undo_last_path():
            # Rebuild label from original + remaining paths − deletions.
            # Using each path's cached bbox-local footprint makes this
            # O(N · bbox_volume) instead of the old O(N · full_volume_dilation).
            self.doc.label = self._rebuild_label_from_state()
            self._refresh_label_cloud()
            self._refresh_paths()
            self.dirtyChanged.emit(self.doc.unsaved)
            self.statusMessage.emit(
                f"Undone. {len(self.doc.traced_paths)} paths remain.")

    def _on_clear_paths(self):
        if self.doc is None:
            return
        if not self.doc.traced_paths:
            return
        self.doc.clear_paths()
        self.doc.label = self._rebuild_label_from_state()
        self._refresh_label_cloud()
        self._refresh_paths()
        self.dirtyChanged.emit(self.doc.unsaved)
        self.statusMessage.emit("Cleared all traced paths.")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _on_save(self):
        """Save the refined label in a background thread so the UI stays
        responsive even on big volumes (gzip of a ~256 MB uint8 mask is
        still ~1-2 seconds of CPU)."""
        if self.doc is None:
            return
        if self._save_thread is not None and self._save_thread.isRunning():
            self.statusMessage.emit("Save already in progress…")
            return

        # Disable the save button while we work (visual feedback)
        try:
            self.controls.btn_save.setEnabled(False)
        except AttributeError:
            pass
        self.statusMessage.emit("Saving…")

        self._save_thread = QThread(self)
        self._save_worker = SaveWorker(
            out_dir=self.settings.output_dir,
            name=self.doc.sample.name,
            label=self.doc.label,
            reference_nii=self.doc.sample.lbl_nii,
        )
        self._save_worker.moveToThread(self._save_thread)
        self._save_thread.started.connect(self._save_worker.run)
        self._save_worker.finished.connect(self._on_save_finished)
        self._save_worker.failed.connect(self._on_save_failed)
        self._save_worker.finished.connect(self._save_thread.quit)
        self._save_worker.failed.connect(self._save_thread.quit)
        self._save_thread.start()

    def _on_save_finished(self, out_path: str):
        if self.doc is not None:
            self.doc.mark_saved()
            self.dirtyChanged.emit(False)
            self.sampleSaved.emit(self.doc.sample.name)
        try:
            self.controls.btn_save.setEnabled(True)
        except AttributeError:
            pass
        self.statusMessage.emit(f"Saved {out_path}")

    def _on_save_failed(self, msg: str):
        try:
            self.controls.btn_save.setEnabled(True)
        except AttributeError:
            pass
        QMessageBox.warning(self, "Save failed", msg)
        self.statusMessage.emit(f"Save failed: {msg}")

    # ------------------------------------------------------------------
    # Noise-point deletion mode (CloudCompare-style segment tool)
    #
    # Workflow:
    #   1. User orients the view however they want.
    #   2. Clicks "Delete noise points…" → enter _deletion_mode.
    #      - Camera becomes non-interactive.
    #      - Polyline overlay takes over mouse events on top of the viewer.
    #      - Left-click adds vertices; right-click starts a new chained
    #        polyline continuing from the last vertex.
    #      - Enter closes the loop (last→first) → preview mask.
    #      - Delete confirms the deletion; Esc cancels.
    #   3. Preview shows the to-be-deleted label voxels in yellow; the
    #      overlay is dismissed so the user can judge whether it looks right.
    #   4. Confirm → label is modified, mask pushed onto undo stack.
    # ------------------------------------------------------------------

    def _on_start_deletion(self):
        if self.doc is None:
            return
        if self._deletion_mode:
            return
        if self._label_points() is None or len(self._label_points()) == 0:
            QMessageBox.information(
                self, "Nothing to delete",
                "The label is empty — there are no points to delete.")
            return
        self._enter_deletion_mode()

    def _label_points(self) -> Optional[np.ndarray]:
        """Current label-cloud coordinates as seen in the viewer."""
        return self.viewer._label_points  # set by set_label_points()

    def _enter_deletion_mode(self):
        self._deletion_mode = True
        self._deletion_preview_mask = None
        # Disable other actions while the user is drawing
        self.controls.btn_start_deletion.setEnabled(False)
        self.controls.btn_trace.setEnabled(False)
        # Viewer3D handles camera lock + VTK 2D overlay + event capture
        self.viewer.enter_deletion_mode()
        self.statusMessage.emit(
            "Deletion mode: left-click to add vertices, right-click to start "
            "a new segment, Enter to close the loop, Esc to cancel.")

    def _exit_deletion_mode(self):
        self._deletion_mode = False
        self.viewer.exit_deletion_mode()
        self.viewer.clear_deletion_preview()
        self._deletion_preview_mask = None
        self.controls.btn_start_deletion.setEnabled(True)
        self.controls.btn_trace.setEnabled(True)

    def _on_polyline_finished(self, polygon_px: np.ndarray):
        """User pressed Enter — build the preview mask in 3D."""
        if self.doc is None:
            return
        coords = self._label_points()
        if coords is None or len(coords) == 0:
            self.statusMessage.emit("No label to delete.")
            return

        sx, sy, valid_idx = self.viewer.project_world_points_to_screen(coords)
        if len(sx) == 0:
            self.statusMessage.emit(
                "No label points are in view — rotate to include them, "
                "then restart deletion.")
            return

        # Polygon vertices come from the VTK controller already in VTK
        # display coordinates (bottom-left origin), matching `sx`/`sy`.
        # No flip required.
        from .utils.config import SPACING
        mask = build_voxel_deletion_mask(
            label=self.doc.label,
            label_coords_world=coords,
            valid_indices_of_world_coords=valid_idx,
            screen_x=sx, screen_y=sy,
            polygon_px=polygon_px,
            spacing=SPACING,
        )
        n = int(mask.sum())
        if n == 0:
            self.statusMessage.emit(
                "The drawn region contains no label voxels. Keep drawing, "
                "press Z to undo the last segment, or Esc to cancel.")
            return

        # Temporarily exit drawing mode so the preview is uncluttered; we
        # re-enter if the user picks Discard in the dialog below.
        self.viewer.exit_deletion_mode()
        self._deletion_preview_mask = mask
        self._show_preview_from_mask(mask)

        keep = QMessageBox.question(
            self,
            "Confirm deletion",
            f"{n:,} label voxels will be removed.\n\n"
            "Apply this deletion?",
            QMessageBox.Apply | QMessageBox.Discard,
            QMessageBox.Apply,
        )
        if keep == QMessageBox.Apply:
            removed = self.doc.apply_deletion(mask)
            self._refresh_label_cloud()
            self.dirtyChanged.emit(self.doc.unsaved)
            self.statusMessage.emit(
                f"Deleted {removed:,} voxels. Use 'Undo last deletion' "
                "to restore them.")
            self._exit_deletion_mode()
        else:
            # User wants to redraw — clear preview, re-enter drawing state
            self.viewer.clear_deletion_preview()
            self._deletion_preview_mask = None
            self.viewer.enter_deletion_mode()
            self.statusMessage.emit(
                "Redraw the deletion region, or press Esc to cancel.")

    def _show_preview_from_mask(self, mask: np.ndarray):
        """Highlight voxels in `mask` in the viewer as a deletion preview."""
        from .utils.config import SPACING
        coords = np.argwhere(mask).astype(np.float64) * SPACING
        self.viewer.show_deletion_preview(coords)

    def _on_deletion_cancel(self):
        self._exit_deletion_mode()
        self.statusMessage.emit("Deletion cancelled.")

    def _on_undo_deletion(self):
        if self.doc is None:
            return
        n = self.doc.undo_last_deletion()
        if n == 0:
            self.statusMessage.emit("Nothing to undo.")
            return
        self._refresh_label_cloud()
        self.dirtyChanged.emit(self.doc.unsaved)
        self.statusMessage.emit(f"Restored {n:,} voxels from last deletion.")

    # ------------------------------------------------------------------
    # Pot-wall peel
    #
    # Runs per-slice 2D anisotropic EDT with rim-aware top preservation
    # (see services/pot_wall_service.py). We run it in a worker since it
    # still takes a few seconds on a full volume; on completion we swap
    # the resulting interior mask into DocumentState and re-fire the
    # thresholded-CT overlay worker so the 3D view reflects the new peel.
    # ------------------------------------------------------------------

    def _on_apply_pot_wall(self, peel_xy_mm: float, peel_base_mm: float):
        if self.doc is None:
            self.statusMessage.emit("No sample loaded.")
            return
        if (self._potwall_thread is not None
                and self._potwall_thread.isRunning()):
            self.statusMessage.emit("Pot-wall peel already running…")
            return

        try:
            self.controls.btn_apply_pot_wall.setEnabled(False)
        except AttributeError:
            pass

        self._potwall_thread = QThread(self)
        self._potwall_worker = PotWallWorker(
            image=self.doc.sample.image,
            peel_xy_mm=peel_xy_mm,
            peel_base_mm=peel_base_mm,
        )
        self._potwall_worker.moveToThread(self._potwall_thread)
        self._potwall_thread.started.connect(self._potwall_worker.run)
        self._potwall_worker.progress.connect(self.statusMessage.emit)
        self._potwall_worker.finished.connect(self._on_pot_wall_done)
        self._potwall_worker.failed.connect(self._on_pot_wall_failed)
        self._potwall_worker.finished.connect(self._potwall_thread.quit)
        self._potwall_worker.failed.connect(self._potwall_thread.quit)
        self._potwall_thread.start()

    def _on_pot_wall_done(self, interior_mask):
        try:
            self.controls.btn_apply_pot_wall.setEnabled(True)
        except AttributeError:
            pass
        if self.doc is None:
            return
        # Store the fresh mask on the sample so the threshold overlay
        # worker uses it. Fire a cloud refresh.
        self.doc.sample.interior_mask = np.ascontiguousarray(
            interior_mask.astype(np.uint8))
        n_in = int(interior_mask.sum())
        self.statusMessage.emit(
            f"Pot-wall peel done — interior = {n_in:,} voxels. "
            "Updating threshold overlay…")
        self._refresh_ct_cloud()

    def _on_pot_wall_failed(self, msg: str):
        try:
            self.controls.btn_apply_pot_wall.setEnabled(True)
        except AttributeError:
            pass
        self.statusMessage.emit(f"Pot-wall peel failed: {msg}")

    # ------------------------------------------------------------------
    # Public helpers used by MainWindow
    # ------------------------------------------------------------------

    def is_dirty(self) -> bool:
        return self.doc is not None and self.doc.unsaved

    def toggle_label_visibility(self):
        """Invert the 'Show label' state (invoked from a main-window shortcut)."""
        self.controls.cb_label.toggle()  # fires toggled signal → _on_toggle_label

    def toggle_ct_visibility(self):
        """Invert the 'Show thresholded CT' state."""
        self.controls.cb_ct.toggle()

    def shutdown(self):
        """Stop every background worker before the tab is destroyed.

        Workers here use the ``moveToThread + started.connect(run)`` pattern,
        and their ``run()`` bodies are synchronous (numpy/nibabel/etc.).
        ``QThread.quit()`` only exits the thread's event loop — it does NOT
        interrupt a running ``run()``. So we ALSO need to force-terminate
        any thread that doesn't return in the graceful window; otherwise
        Qt's destructor finds the thread still alive and aborts with
        "QThread: Destroyed while thread '' is still running".

        Save is handled with a longer graceful window — we'd rather wait
        for a clean write than leave a dangling `.tmp` file on disk.
        """
        # Save: generous graceful timeout; safer to finish the atomic rename
        self._stop_thread(self._save_thread,
                          graceful_ms=5000, force_ms=2000)
        # Load / trace / threshold / pot-wall: short graceful window then force
        for th in (self._load_thread, self._trace_thread,
                   self._thr_thread, self._potwall_thread):
            self._stop_thread(th, graceful_ms=300, force_ms=2000)

    @staticmethod
    def _stop_thread(thread, graceful_ms: int = 300, force_ms: int = 2000):
        """Stop a QThread robustly: quit() + wait, then terminate() if needed."""
        if thread is None:
            return
        try:
            if not thread.isRunning():
                return
            thread.quit()
            if thread.wait(graceful_ms):
                return
            # Worker is still inside a blocking run(); force it down so the
            # upcoming QObject delete doesn't hit a live QThread.
            thread.terminate()
            thread.wait(force_ms)
        except Exception:
            # Qt can raise if the thread is already being torn down; we're
            # doing best-effort cleanup so swallow and move on.
            pass
