"""Axis-aligned single-plane slice control."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QCheckBox, QPushButton,
)


class AxisSliceSlider(QWidget):
    enabledChanged = Signal(bool)
    positionChanged = Signal(float)
    reverseChanged = Signal(bool)

    def __init__(self, title: str, minimum: int = -100, maximum: int = 100,
                 initial: float = 0.0, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(6)

        top = QHBoxLayout()
        self._enable = QCheckBox(title)
        self._enable.toggled.connect(self._on_enable_toggled)
        top.addWidget(self._enable)
        self._reverse = QCheckBox("Reverse")
        self._reverse.toggled.connect(self.reverseChanged.emit)
        top.addWidget(self._reverse)
        top.addStretch(1)
        root.addLayout(top)

        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Position</b>"))
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(int(minimum), int(maximum))
        self._slider.setValue(int(round(initial)))
        self._slider.setMinimumWidth(180)
        self._slider.valueChanged.connect(self._on_changed)
        row.addWidget(self._slider, 1)
        self._lbl = QLabel()
        self._lbl.setStyleSheet("color: #444; font-family: monospace;")
        self._lbl.setMinimumWidth(76)
        row.addWidget(self._lbl)
        self._reset = QPushButton("Zero")
        self._reset.clicked.connect(lambda: self.set_position(0.0, emit=True))
        row.addWidget(self._reset)
        root.addLayout(row)

        self._refresh_label()
        self._set_interactive(False)

    def _set_interactive(self, enabled: bool):
        self._slider.setEnabled(enabled)
        self._reverse.setEnabled(enabled)
        self._reset.setEnabled(enabled)

    def _refresh_label(self):
        self._lbl.setText(f"{int(self._slider.value()):+d} mm")

    def _on_enable_toggled(self, enabled: bool):
        self._set_interactive(enabled)
        self.enabledChanged.emit(enabled)

    def _on_changed(self, value: int):
        self._refresh_label()
        self.positionChanged.emit(float(value))

    def set_enabled(self, enabled: bool):
        self._enable.blockSignals(True)
        self._enable.setChecked(enabled)
        self._enable.blockSignals(False)
        self._set_interactive(enabled)

    def set_reverse(self, enabled: bool):
        self._reverse.blockSignals(True)
        self._reverse.setChecked(enabled)
        self._reverse.blockSignals(False)

    def set_position(self, position_mm: float, emit: bool = False):
        value = int(round(position_mm))
        self._slider.blockSignals(not emit)
        self._slider.setValue(value)
        self._refresh_label()
        self._slider.blockSignals(False)
        if emit:
            self.positionChanged.emit(float(value))

    def set_range(self, minimum: float, maximum: float):
        lo = int(min(minimum, maximum))
        hi = int(max(minimum, maximum))
        if lo == hi:
            hi = lo + 1
        self._slider.blockSignals(True)
        self._slider.setRange(lo, hi)
        cur = max(lo, min(hi, self._slider.value()))
        self._slider.setValue(cur)
        self._refresh_label()
        self._slider.blockSignals(False)
