"""Background label-save worker — keeps the UI responsive during gzip."""
from __future__ import annotations
import traceback
import numpy as np

from PySide6.QtCore import QObject, Signal

from ..io.data_saver import save_label


class SaveWorker(QObject):
    """Runs in a QThread. Emits `finished(path)` or `failed(msg)`."""

    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, out_dir: str, name: str, label: np.ndarray,
                 reference_nii):
        super().__init__()
        self.out_dir = out_dir
        self.name = name
        # Cast to uint8 *and* take a contiguous snapshot in one shot so
        # (a) the main thread can keep mutating `doc.label` while we
        # compress, and (b) save_label doesn't have to allocate a second
        # copy. `astype(np.uint8)` already returns a fresh contiguous
        # array, so no separate `ascontiguousarray` is needed.
        self.label = label.astype(np.uint8, copy=True)
        self.reference_nii = reference_nii

    def run(self):
        try:
            path = save_label(self.out_dir, self.name,
                              self.label, self.reference_nii)
            self.finished.emit(path)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
