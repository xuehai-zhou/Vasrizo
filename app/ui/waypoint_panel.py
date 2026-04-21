"""List of current waypoints with undo / delete / clear buttons."""
from __future__ import annotations
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QHBoxLayout, QPushButton, QLabel,
)


class WaypointPanel(QWidget):
    undoLast = Signal()
    deleteSelected = Signal(int)   # index in the list
    clearAll = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)

        root.addWidget(QLabel("<b>Waypoints</b>  (Shift+Click in 3D view to add)"))

        self.list_w = QListWidget()
        self.list_w.setAlternatingRowColors(True)
        root.addWidget(self.list_w)

        btns = QHBoxLayout()
        self.btn_undo = QPushButton("Undo last")
        self.btn_del = QPushButton("Delete selected")
        self.btn_clear = QPushButton("Clear all")
        btns.addWidget(self.btn_undo)
        btns.addWidget(self.btn_del)
        btns.addWidget(self.btn_clear)
        root.addLayout(btns)

        self.btn_undo.clicked.connect(self.undoLast.emit)
        self.btn_clear.clicked.connect(self.clearAll.emit)
        self.btn_del.clicked.connect(self._on_delete)

    def _on_delete(self):
        row = self.list_w.currentRow()
        if row >= 0:
            self.deleteSelected.emit(row)

    def update_from_waypoints(self, waypoints):
        """waypoints: iterable of DocumentState.Waypoint"""
        self.list_w.clear()
        for i, wp in enumerate(waypoints):
            x, y, z = wp.phys
            self.list_w.addItem(
                f"[{i+1}] ({wp.source:5s})  "
                f"({x:7.1f}, {y:7.1f}, {z:7.1f}) mm")
