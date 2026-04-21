"""Dual-handle horizontal HU threshold slider.

One track, two round handles. Drag either handle; step snaps to `step`
(default 5). Emits `valuesChanged(lower, upper)` while dragging.
"""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal, QPointF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget, QSizePolicy


class ThresholdRangeSlider(QWidget):
    valuesChanged = Signal(float, float)
    # Throttled companion: fires after the user stops moving for ~120ms
    valuesCommitted = Signal(float, float)

    def __init__(self, minimum: float = -1000.0, maximum: float = 1500.0,
                 step: float = 5.0, lower: float = -500.0, upper: float = 700.0,
                 parent=None):
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._lower = float(lower)
        self._upper = float(upper)
        self._dragging = None       # 'lower' | 'upper' | None
        self._handle_radius = 9
        self._track_height = 5
        self.setMinimumHeight(62)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.setInterval(120)
        self._commit_timer.timeout.connect(
            lambda: self.valuesCommitted.emit(self._lower, self._upper))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def values(self) -> tuple[float, float]:
        return self._lower, self._upper

    def set_values(self, lower: float, upper: float):
        lower = self._snap(lower)
        upper = self._snap(upper)
        if upper - lower < self._step:
            upper = lower + self._step
        if (lower, upper) != (self._lower, self._upper):
            self._lower, self._upper = lower, upper
            self.update()
            self.valuesChanged.emit(self._lower, self._upper)
            self._commit_timer.start()

    def set_range(self, minimum: float, maximum: float):
        self._min, self._max = float(minimum), float(maximum)
        self._lower = max(self._lower, self._min)
        self._upper = min(self._upper, self._max)
        self.update()

    # ------------------------------------------------------------------
    # Internal geometry
    # ------------------------------------------------------------------

    def _track_rect(self) -> tuple[float, float, float]:
        pad = self._handle_radius + 6
        x1 = pad
        x2 = self.width() - pad
        y = self.height() * 0.45
        return x1, x2, y

    def _snap(self, v: float) -> float:
        v = round(v / self._step) * self._step
        return max(self._min, min(self._max, v))

    def _value_to_x(self, v: float) -> float:
        x1, x2, _ = self._track_rect()
        frac = (v - self._min) / (self._max - self._min)
        return x1 + frac * (x2 - x1)

    def _x_to_value(self, x: float) -> float:
        x1, x2, _ = self._track_rect()
        if x2 <= x1:
            return self._min
        frac = (x - x1) / (x2 - x1)
        frac = max(0.0, min(1.0, frac))
        return self._snap(self._min + frac * (self._max - self._min))

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        x1, x2, y = self._track_rect()

        # Track background
        p.setPen(QPen(QColor(200, 200, 205), self._track_height, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(x1, y), QPointF(x2, y))

        # Active range
        lx = self._value_to_x(self._lower)
        ux = self._value_to_x(self._upper)
        p.setPen(QPen(QColor(70, 130, 200), self._track_height, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(lx, y), QPointF(ux, y))

        # Handles
        p.setBrush(QBrush(QColor(235, 245, 255)))
        p.setPen(QPen(QColor(50, 90, 160), 1.4))
        p.drawEllipse(QPointF(lx, y), self._handle_radius, self._handle_radius)
        p.drawEllipse(QPointF(ux, y), self._handle_radius, self._handle_radius)

        # Labels
        font = QFont(self.font())
        font.setPointSize(max(8, font.pointSize() - 1))
        p.setFont(font)
        p.setPen(QColor(60, 60, 60))
        metrics = p.fontMetrics()
        lower_text = f"{self._lower:.0f}"
        upper_text = f"{self._upper:.0f}"
        lw = metrics.horizontalAdvance(lower_text)
        uw = metrics.horizontalAdvance(upper_text)
        text_y = y + self._handle_radius + metrics.ascent() + 2
        p.drawText(QPointF(lx - lw / 2, text_y), lower_text)
        p.drawText(QPointF(ux - uw / 2, text_y), upper_text)

        # Min / max end labels
        p.setPen(QColor(140, 140, 140))
        p.drawText(QPointF(x1 - metrics.horizontalAdvance(f"{int(self._min)}") / 2,
                           y - self._handle_radius - 4),
                   f"{int(self._min)}")
        p.drawText(QPointF(x2 - metrics.horizontalAdvance(f"{int(self._max)}") / 2,
                           y - self._handle_radius - 4),
                   f"{int(self._max)}")

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _handle_under(self, pos) -> str | None:
        lx = self._value_to_x(self._lower)
        ux = self._value_to_x(self._upper)
        dl = abs(pos.x() - lx)
        du = abs(pos.x() - ux)
        hot = self._handle_radius + 4
        if dl < du and dl <= hot:
            return 'lower'
        if du <= hot:
            return 'upper'
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        h = self._handle_under(event.position())
        if h is None:
            # Clicking on the track — move the nearer handle to that x
            v = self._x_to_value(event.position().x())
            if abs(v - self._lower) < abs(v - self._upper):
                h = 'lower'
            else:
                h = 'upper'
            self._move_to(h, v)
        self._dragging = h
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            # Hover feedback
            self.setCursor(Qt.PointingHandCursor if self._handle_under(event.position())
                           else Qt.ArrowCursor)
            return
        v = self._x_to_value(event.position().x())
        self._move_to(self._dragging, v)

    def _move_to(self, which: str, v: float):
        changed = False
        if which == 'lower':
            v = min(v, self._upper - self._step)
            if v != self._lower:
                self._lower = v
                changed = True
        else:
            v = max(v, self._lower + self._step)
            if v != self._upper:
                self._upper = v
                changed = True
        if changed:
            self.update()
            self.valuesChanged.emit(self._lower, self._upper)
            self._commit_timer.start()

    def mouseReleaseEvent(self, event):
        self._dragging = None
        self.setCursor(Qt.ArrowCursor)
        # Ensure a final commit fires
        self._commit_timer.start()
