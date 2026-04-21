"""Background tracing: speed field + Dijkstra between a list of waypoints."""
from __future__ import annotations
import traceback
import numpy as np
from PySide6.QtCore import QObject, Signal

from ..services.root_model import RootModel
from ..services.tracing_service import find_path_between
from ..utils.geometry_utils import physical_to_voxel


class TraceWorker(QObject):
    progress = Signal(str)
    # Emits (concatenated_path_voxels, radius_voxels). Path may be empty.
    finished = Signal(object, float)
    failed = Signal(str)

    def __init__(self, image: np.ndarray, model: RootModel,
                 waypoints_phys: list, fill_radius_mm: float,
                 mean_spacing: float,
                 label: np.ndarray | None = None):
        super().__init__()
        self.image = image
        self.model = model
        self.waypoints = waypoints_phys
        self.fill_radius_mm = fill_radius_mm
        self.mean_spacing = mean_spacing
        # Label volume is used by find_path_between as the "anchor" for
        # early Dijkstra termination — we stop as soon as the search
        # reaches a labeled voxel near the end waypoint.
        self.label = label

    def run(self):
        try:
            paths = []
            n = len(self.waypoints)
            for i in range(n - 1):
                start_vox = physical_to_voxel(self.waypoints[i])
                end_vox = physical_to_voxel(self.waypoints[i + 1])
                dist = float(np.linalg.norm(
                    np.asarray(self.waypoints[i + 1], dtype=np.float64)
                    - np.asarray(self.waypoints[i], dtype=np.float64)))
                self.progress.emit(
                    f"Segment {i + 1}/{n - 1}: distance={dist:.1f} mm")
                seg = find_path_between(
                    self.image, self.model, start_vox, end_vox,
                    progress=lambda m: self.progress.emit(m),
                    label=self.label,
                )
                if seg is not None:
                    paths.append(seg)
                    self.progress.emit(f"  -> {len(seg)} voxels")
                else:
                    self.progress.emit("  -> no path found")

            if paths:
                full = np.concatenate(paths)
                r_vox = self.fill_radius_mm / self.mean_spacing
                self.finished.emit(full, float(r_vox))
            else:
                self.finished.emit(None, 0.0)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
