"""Background sample loading (NIfTI read + RootModel learning)."""
from __future__ import annotations
import traceback

from PySide6.QtCore import QObject, Signal

from ..io.data_loader import load_sample
from ..services.root_model import RootModel


class LoadWorker(QObject):
    """Runs in a QThread. Emits `finished(SampleData, RootModel)` or `failed(str)`."""

    progress = Signal(str)
    finished = Signal(object, object)   # SampleData, RootModel
    failed = Signal(str)

    def __init__(self, name: str, labels_dir: str, images_dir: str,
                 beta: float, col_diff: float):
        super().__init__()
        self.name = name
        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.beta = beta
        self.col_diff = col_diff

    def run(self):
        try:
            self.progress.emit(f"Loading {self.name} …")
            sample = load_sample(self.name, self.labels_dir, self.images_dir)
            self.progress.emit("Learning root appearance model …")
            # TODO(pybind11): RootModel training could be accelerated if it
            # became a hotspot; currently it's a small histogram + KDE.
            model = RootModel(sample.image, sample.label,
                              beta=self.beta, col_diff=self.col_diff)
            self.finished.emit(sample, model)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
