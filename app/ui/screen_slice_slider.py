"""Screen-aligned slicing control.

The slicing plane is always parallel to the current screen plane; the
slider moves that plane along the current view normal (into/out of the
screen). The consumer owns the actual camera math and rendering update.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QCheckBox,
    QPushButton, QDoubleSpinBox,
)


class ScreenSliceSlider(QWidget):
    enabledChanged = Signal(bool)
    offsetChanged = Signal(float)
    reverseChanged = Signal(bool)
    lockedChanged = Signal(bool)
    thicknessChanged = Signal(float)
    guidesChanged = Signal(bool)
    freezeDirectionChanged = Signal(bool)

    def __init__(self, minimum: int = -100, maximum: int = 100,
                 initial: float = 0.0, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)

        self._enable = QCheckBox("Screen slice")
        self._enable.toggled.connect(self._on_enable_toggled)
        top.addWidget(self._enable)

        self._reverse = QCheckBox("Reverse")
        self._reverse.toggled.connect(self._on_reverse_toggled)
        top.addWidget(self._reverse)

        self._lock = QCheckBox("Lock thickness")
        self._lock.toggled.connect(self._on_lock_toggled)
        top.addWidget(self._lock)

        self._guides = QCheckBox("Guide planes")
        self._guides.toggled.connect(self.guidesChanged.emit)
        top.addWidget(self._guides)

        self._freeze = QCheckBox("Freeze direction")
        self._freeze.toggled.connect(self.freezeDirectionChanged.emit)
        top.addWidget(self._freeze)
        top.addStretch(1)
        root.addLayout(top)

        mid = QHBoxLayout()
        mid.setSpacing(8)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(int(minimum), int(maximum))
        self._slider.setValue(int(round(initial)))
        self._slider.setMinimumWidth(180)
        self._slider.valueChanged.connect(self._on_changed)
        mid.addWidget(QLabel("<b>Offset</b>"))
        mid.addWidget(self._slider, 1)

        self._lbl = QLabel()
        self._lbl.setStyleSheet("color: #444; font-family: monospace;")
        self._lbl.setMinimumWidth(76)
        mid.addWidget(self._lbl)

        self._reset = QPushButton("Zero")
        self._reset.clicked.connect(lambda: self.set_offset(0.0, emit=True))
        mid.addWidget(self._reset)
        root.addLayout(mid)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(QLabel("Thickness"))
        self._thickness = QDoubleSpinBox()
        self._thickness.setRange(0.1, 1000.0)
        self._thickness.setDecimals(1)
        self._thickness.setSingleStep(0.5)
        self._thickness.setValue(5.0)
        self._thickness.setSuffix(" mm")
        self._thickness.valueChanged.connect(self.thicknessChanged.emit)
        bottom.addWidget(self._thickness)
        bottom.addStretch(1)
        root.addLayout(bottom)

        self._refresh_label()
        self._set_interactive(False)

    def _refresh_label(self):
        v = int(self._slider.value())
        self._lbl.setText(f"{v:+d} mm")

    def _set_interactive(self, enabled: bool):
        self._slider.setEnabled(enabled)
        self._reset.setEnabled(enabled)
        self._reverse.setEnabled(enabled)
        self._lock.setEnabled(enabled)
        self._guides.setEnabled(enabled)
        self._freeze.setEnabled(enabled)
        self._thickness.setEnabled(enabled and self._lock.isChecked())

    def _on_changed(self, v: int):
        self._refresh_label()
        self.offsetChanged.emit(float(v))

    def set_enabled(self, enabled: bool):
        self._enable.blockSignals(True)
        self._enable.setChecked(enabled)
        self._enable.blockSignals(False)
        self._set_interactive(enabled)

    def set_reverse(self, enabled: bool):
        self._reverse.blockSignals(True)
        self._reverse.setChecked(enabled)
        self._reverse.blockSignals(False)

    def set_locked(self, enabled: bool):
        self._lock.blockSignals(True)
        self._lock.setChecked(enabled)
        self._lock.blockSignals(False)
        self._thickness.setEnabled(self._enable.isChecked() and enabled)

    def set_guides(self, enabled: bool):
        self._guides.blockSignals(True)
        self._guides.setChecked(enabled)
        self._guides.blockSignals(False)

    def set_freeze_direction(self, enabled: bool):
        self._freeze.blockSignals(True)
        self._freeze.setChecked(enabled)
        self._freeze.blockSignals(False)

    def set_thickness(self, thickness_mm: float):
        self._thickness.blockSignals(True)
        self._thickness.setValue(float(thickness_mm))
        self._thickness.blockSignals(False)

    def set_thickness_range(self, minimum: float, maximum: float):
        lo = min(float(minimum), float(maximum))
        hi = max(float(minimum), float(maximum))
        if hi <= lo:
            hi = lo + 0.1
        self._thickness.blockSignals(True)
        self._thickness.setRange(max(0.1, lo), hi)
        self._thickness.setValue(max(self._thickness.minimum(),
                                     min(self._thickness.maximum(),
                                         self._thickness.value())))
        self._thickness.blockSignals(False)

    def set_offset(self, offset_mm: float, emit: bool = False):
        val = int(round(offset_mm))
        self._slider.blockSignals(not emit)
        self._slider.setValue(val)
        self._refresh_label()
        self._slider.blockSignals(False)
        if emit:
            self.offsetChanged.emit(float(val))

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

    def offset(self) -> float:
        return float(self._slider.value())

    def thickness(self) -> float:
        return float(self._thickness.value())

    def thickness_minimum(self) -> float:
        return float(self._thickness.minimum())

    def _on_enable_toggled(self, enabled: bool):
        self._set_interactive(enabled)
        self.enabledChanged.emit(enabled)

    def _on_reverse_toggled(self, enabled: bool):
        self.reverseChanged.emit(enabled)

    def _on_lock_toggled(self, enabled: bool):
        self._thickness.setEnabled(self._enable.isChecked() and enabled)
        self.lockedChanged.emit(enabled)
