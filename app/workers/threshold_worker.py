"""Background HU-threshold recompute (keeps slider-drag UI responsive)."""
from __future__ import annotations
import numpy as np
from PySide6.QtCore import QObject, Signal

from ..services.threshold_service import threshold_ct_to_coords


class ThresholdWorker(QObject):
    """Recomputes the brown CT point cloud from cached CT vol + interior mask."""

    finished = Signal(object)   # (N, 3) float64 coord array
    failed = Signal(str)

    def __init__(self, ct_vol: np.ndarray, interior: np.ndarray,
                 hu_lower: float, hu_upper: float, downsample: int = 1):
        super().__init__()
        self.ct_vol = ct_vol
        self.interior = interior
        self.hu_lower = hu_lower
        self.hu_upper = hu_upper
        self.downsample = downsample

    def run(self):
        try:
            coords = threshold_ct_to_coords(
                self.ct_vol, self.interior,
                self.hu_lower, self.hu_upper, self.downsample)
            self.finished.emit(coords)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
