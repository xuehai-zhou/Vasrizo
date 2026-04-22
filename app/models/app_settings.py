"""App-wide settings carried from CLI args into all tabs.

Two required directories per run:
    labels_dir  — partial/broken labels per sample
    images_dir  — preprocessed image volumes per sample
                  (non-anatomy voxels filled with an out-of-range intensity)
    output_dir  — refined labels are saved here

Directories can be:
  - supplied via CLI defaults in DataPaths (optional), or
  - set at runtime via set_dirs() after the user opens/drops a folder.

Runtime overrides take precedence over CLI defaults.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
from typing import Optional

from ..utils.config import DataPaths


@dataclass
class AppSettings:
    paths: DataPaths = field(default_factory=DataPaths)

    # Runtime overrides (absolute paths) set after File > Open Dataset
    _images_override: Optional[str] = None
    _labels_override: Optional[str] = None
    _output_override: Optional[str] = None

    # ------------------------------------------------------------------
    # Public: runtime dataset configuration
    # ------------------------------------------------------------------

    def set_dirs(self, images_dir: str, labels_dir: str,
                 output_dir: Optional[str] = None) -> None:
        """Replace the current dataset with absolute paths."""
        self._images_override = os.path.abspath(images_dir) if images_dir else None
        self._labels_override = os.path.abspath(labels_dir) if labels_dir else None
        self._output_override = os.path.abspath(output_dir) if output_dir else None

    def is_configured(self) -> bool:
        """True iff both images and labels dirs resolve to existing directories."""
        return os.path.isdir(self.images_dir) and os.path.isdir(self.labels_dir)

    # ------------------------------------------------------------------
    # Resolved paths
    # ------------------------------------------------------------------

    def _resolve(self, subdir: str) -> str:
        """Resolve a subdir against data_dir, with sibling/cwd fallbacks."""
        if not subdir:
            return ""
        primary = os.path.join(self.paths.data_dir, subdir)
        if os.path.isdir(primary):
            return primary
        parent = os.path.dirname(self.paths.data_dir.rstrip("/")) or "."
        sibling = os.path.join(parent, subdir)
        if os.path.isdir(sibling):
            return sibling
        cwd_side = os.path.join(".", subdir)
        if os.path.isdir(cwd_side):
            return cwd_side
        return primary

    @property
    def labels_dir(self) -> str:
        if self._labels_override:
            return self._labels_override
        return self._resolve(self.paths.labels_subdir)

    @property
    def images_dir(self) -> str:
        if self._images_override:
            return self._images_override
        return self._resolve(self.paths.images_subdir)

    @property
    def output_dir(self) -> str:
        if self._output_override:
            return self._output_override
        return os.path.join(self.paths.data_dir, self.paths.output_subdir)
