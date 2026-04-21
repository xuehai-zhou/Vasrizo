"""Dialog for manually configuring a dataset (images / labels / output dirs).

Shown when:
  - the user picks File > Configure Dataset…
  - auto-detection of a dropped/opened folder didn't find both images and
    labels subdirs (fields are prefilled with whatever was detected).
"""
from __future__ import annotations
import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel,
    QFileDialog, QDialogButtonBox, QFrame,
)

from ..utils.layout_detect import default_output_dir


class _DirPickerRow(QHBoxLayout):
    """Label + line edit + browse button."""

    def __init__(self, caption: str, initial: str = ""):
        super().__init__()
        self.caption_label = QLabel(caption)
        self.caption_label.setMinimumWidth(90)
        self.edit = QLineEdit(initial)
        self.edit.setMinimumWidth(420)
        self.btn = QPushButton("Browse…")
        self.btn.clicked.connect(self._browse)
        self.addWidget(self.caption_label)
        self.addWidget(self.edit, 1)
        self.addWidget(self.btn)

    def _browse(self):
        current = self.edit.text().strip() or os.path.expanduser("~")
        # Start from parent dir if current path exists, for convenience
        start = current if os.path.isdir(current) else os.path.dirname(current)
        d = QFileDialog.getExistingDirectory(
            self.btn.parentWidget(), "Select directory", start)
        if d:
            self.edit.setText(d)

    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, v: str):
        self.edit.setText(v or "")


class DatasetDialog(QDialog):
    """Let the user specify images / labels / output directories."""

    def __init__(self, parent=None, images: str = "", labels: str = "",
                 output: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Configure Dataset")
        self.setMinimumWidth(640)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "<b>Dataset configuration</b><br>"
            "Choose the directories containing your <i>image volumes</i>, "
            "<i>partial labels</i>, and the <i>output</i> folder for refined labels.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Form rows
        self.row_images = _DirPickerRow("Images:", images)
        self.row_labels = _DirPickerRow("Labels:", labels)
        self.row_output = _DirPickerRow(
            "Output:", output or default_output_dir(labels))
        layout.addLayout(self.row_images)
        layout.addLayout(self.row_labels)
        layout.addLayout(self.row_output)

        # Validation hint label
        self.hint = QLabel("")
        self.hint.setStyleSheet("color: #a33;")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        # Separator + buttons
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Live re-evaluate the output default when labels change
        self.row_labels.edit.textChanged.connect(self._on_labels_changed)

    # ------------------------------------------------------------------

    def _on_labels_changed(self, text: str):
        # Only auto-fill output when output is empty or a previous auto default
        cur = self.row_output.value()
        if cur == "" or cur == default_output_dir(""):
            self.row_output.set_value(default_output_dir(text))

    def _on_ok(self):
        imgs, lbls, out = self.row_images.value(), self.row_labels.value(), self.row_output.value()
        if not imgs or not os.path.isdir(imgs):
            self.hint.setText(f"Images directory does not exist: {imgs!r}")
            return
        if not lbls or not os.path.isdir(lbls):
            self.hint.setText(f"Labels directory does not exist: {lbls!r}")
            return
        if not out:
            self.hint.setText("Please choose an output directory.")
            return
        self.accept()

    # ------------------------------------------------------------------

    def values(self) -> tuple[str, str, str]:
        return (self.row_images.value(), self.row_labels.value(),
                self.row_output.value())

    @staticmethod
    def run(parent=None, images: str = "", labels: str = "",
            output: str = "") -> Optional[tuple[str, str, str]]:
        """Convenience: returns (images, labels, output) on OK, None on Cancel."""
        dlg = DatasetDialog(parent, images=images, labels=labels, output=output)
        if dlg.exec() == QDialog.Accepted:
            return dlg.values()
        return None
