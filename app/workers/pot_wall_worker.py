"""Background pot-wall computation — keeps the UI responsive during EDT."""
from __future__ import annotations
import traceback
import numpy as np

from PySide6.QtCore import QObject, Signal

from ..services.pot_wall_service import (
    estimate_pot_cylinder_geometry,
    remove_pot_walls,
)


class PotWallWorker(QObject):
    """Run `remove_pot_walls` in a QThread. Emits the boolean mask."""

    progress = Signal(str)
    finished = Signal(object)   # (np.ndarray mask, PotCylinderGeometry|None)
    failed = Signal(str)

    def __init__(self, image: np.ndarray,
                 peel_xy_mm: float, peel_base_mm: float):
        super().__init__()
        self.image = image
        self.peel_xy_mm = float(peel_xy_mm)
        self.peel_base_mm = float(peel_base_mm)

    def run(self):
        try:
            self.progress.emit(
                f"Pot-wall peel running  (xy={self.peel_xy_mm:.1f} mm, "
                f"base={self.peel_base_mm:.1f} mm)…")
            geom = estimate_pot_cylinder_geometry(self.image)
            interior = remove_pot_walls(
                self.image,
                peel_xy_mm=self.peel_xy_mm,
                peel_base_mm=self.peel_base_mm,
            )
            self.finished.emit((interior, geom))
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
