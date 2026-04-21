"""Bottom status strip — text + progress indicator + dirty indicator."""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStatusBar, QLabel, QProgressBar


class AppStatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._msg = QLabel("")
        self._dirty = QLabel("")
        self._dirty.setStyleSheet("color: #c0392b; font-weight: bold;")
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(140)
        self._progress.setVisible(False)
        self.addWidget(self._msg, 1)
        self.addPermanentWidget(self._dirty)
        self.addPermanentWidget(self._progress)

    def say(self, text: str):
        self._msg.setText(text)

    def set_dirty(self, dirty: bool, label_name: str = ""):
        self._dirty.setText(f"● unsaved: {label_name}" if dirty else "")

    def start_busy(self):
        self._progress.setRange(0, 0)
        self._progress.setVisible(True)

    def stop_busy(self):
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setVisible(False)
