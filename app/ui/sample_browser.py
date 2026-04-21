"""Left panel: list of samples on disk, with an 'open in new tab' action."""
from __future__ import annotations
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QLineEdit,
    QHBoxLayout, QPushButton,
)


class SampleBrowser(QWidget):
    """Emits `openSample(name)` on double-click or when Open is pressed."""
    openSample = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(QLabel("<b>Samples</b>"))

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter…")
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        self.list_w = QListWidget()
        self.list_w.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.list_w)

        btns = QHBoxLayout()
        self.btn_open = QPushButton("Open in new tab")
        self.btn_open.clicked.connect(self._on_open)
        btns.addWidget(self.btn_open)
        layout.addLayout(btns)

        self._all_names: list[str] = []
        self._completed: set[str] = set()

    # ------------------------------------------------------------------

    def set_samples(self, names: list[str], completed: set[str] | None = None):
        self._all_names = list(names)
        self._completed = set(completed or ())
        self._apply_filter(self._filter.text())

    def mark_completed(self, name: str):
        self._completed.add(name)
        self._apply_filter(self._filter.text())

    def _apply_filter(self, text: str):
        text = text.strip().lower()
        self.list_w.clear()
        for n in self._all_names:
            if text and text not in n.lower():
                continue
            tag = "  ✓" if n in self._completed else ""
            item = QListWidgetItem(f"{n}{tag}")
            item.setData(Qt.UserRole, n)
            if n in self._completed:
                item.setForeground(Qt.darkGreen)
            self.list_w.addItem(item)

    # ------------------------------------------------------------------

    def _on_double_click(self, item: QListWidgetItem):
        name = item.data(Qt.UserRole)
        if name:
            self.openSample.emit(name)

    def _on_open(self):
        item = self.list_w.currentItem()
        if item is not None:
            name = item.data(Qt.UserRole)
            if name:
                self.openSample.emit(name)
