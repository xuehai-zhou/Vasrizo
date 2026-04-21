"""Small horizontal point-size slider + live value label.

Emits `valueChanged(size)` continuously while dragging. Intended to live
directly below the 3D viewer, next to the threshold slider.
"""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider


class PointSizeSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, minimum: int = 1, maximum: int = 20,
                 initial: float = 3.0, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 2, 8, 4)
        root.setSpacing(8)

        root.addWidget(QLabel("<b>Point size</b>"))

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(minimum, maximum)
        self._slider.setValue(int(initial))
        self._slider.setMinimumWidth(160)
        self._slider.valueChanged.connect(self._on_changed)
        root.addWidget(self._slider, 1)

        self._lbl = QLabel(f"{int(initial)} px")
        self._lbl.setStyleSheet("color: #444; font-family: monospace;")
        self._lbl.setMinimumWidth(48)
        root.addWidget(self._lbl)

    def _on_changed(self, v: int):
        self._lbl.setText(f"{v} px")
        self.valueChanged.emit(float(v))

    def value(self) -> float:
        return float(self._slider.value())

    def set_value(self, v: float):
        self._slider.blockSignals(True)
        self._slider.setValue(int(round(v)))
        self._lbl.setText(f"{int(round(v))} px")
        self._slider.blockSignals(False)
