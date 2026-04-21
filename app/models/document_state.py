"""Per-tab annotation state: everything about one sample being edited."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..io.data_loader import SampleData
from ..services.root_model import RootModel
from ..utils.config import (
    HU_LOWER_DEFAULT, HU_UPPER_DEFAULT,
    BETA_DEFAULT, COL_DIFF_DEFAULT, FILL_RADIUS_MM_DEFAULT,
    POINT_SIZE_DEFAULT,
)


@dataclass
class Waypoint:
    """One picked waypoint (source="label"|"ct")."""
    phys: np.ndarray          # (3,) physical mm
    source: str               # "label" or "ct"


@dataclass
class TracedPath:
    """A single accepted traced path + the radius used for painting it.

    `bbox` and `footprint` cache the path's dilated local mask so undo
    can rebuild the label by OR-ing bbox-sized chunks instead of
    re-dilating across the whole volume per path. See
    `paint_tube_local` for the packing convention.
    """
    path_voxels: np.ndarray   # (N, 3) int64 voxel coords
    radius_voxels: float
    bbox: Optional[Tuple[slice, slice, slice]] = None
    footprint: Optional[np.ndarray] = None   # bool, shape matches bbox


class DocumentState:
    """All state for one open sample. Owned by one AnnotationTab.

    Mutations are pure Python — the tab connects Qt signals to trigger UI
    refreshes after mutating this object.
    """

    def __init__(self, sample: SampleData):
        self.sample = sample

        # Mutable label (original kept separately so Undo/Clear work)
        self.label: np.ndarray = sample.label.copy()
        self.label_orig: np.ndarray = sample.label.copy()

        # Learned root appearance model (built lazily after load)
        self.root_model: Optional[RootModel] = None

        # Thresholding state
        self.hu_lower: float = HU_LOWER_DEFAULT
        self.hu_upper: float = HU_UPPER_DEFAULT

        # Visibility + rendering
        self.show_label: bool = True
        self.show_ct: bool = True
        self.point_size: float = POINT_SIZE_DEFAULT
        self.screen_slice_enabled: bool = False
        self.screen_slice_offset_mm: float = 0.0
        self.screen_slice_reverse: bool = False
        self.screen_slice_locked: bool = False
        self.screen_slice_thickness_mm: float = 5.0
        self.screen_slice_show_guides: bool = False
        self.screen_slice_freeze_direction: bool = False
        self.screen_slice_frozen_origin: Optional[np.ndarray] = None
        self.screen_slice_frozen_normal: Optional[np.ndarray] = None

        # Annotation state
        self.waypoints: list[Waypoint] = []
        self.traced_paths: list[TracedPath] = []
        # Stack of boolean masks representing voxels removed by each
        # noise-deletion operation. Popping the top mask re-adds those
        # voxels to `self.label` (see `undo_last_deletion`).
        self.deleted_regions: list[np.ndarray] = []

        # Tuning knobs (per-tab overrides of defaults)
        self.beta: float = BETA_DEFAULT
        self.col_diff: float = COL_DIFF_DEFAULT
        self.fill_radius_mm: float = FILL_RADIUS_MM_DEFAULT

        # Dirty flag
        self._unsaved: bool = False

    # ------------------------------------------------------------------
    # Dirty-flag management
    # ------------------------------------------------------------------

    @property
    def unsaved(self) -> bool:
        return self._unsaved

    def mark_unsaved(self):
        self._unsaved = True

    def mark_saved(self):
        self._unsaved = False

    # ------------------------------------------------------------------
    # Waypoint ops
    # ------------------------------------------------------------------

    def add_waypoint(self, phys: np.ndarray, source: str):
        self.waypoints.append(Waypoint(phys=np.asarray(phys, dtype=np.float64),
                                       source=source))

    def undo_last_waypoint(self) -> bool:
        if not self.waypoints:
            return False
        self.waypoints.pop()
        return True

    def delete_waypoint(self, index: int) -> bool:
        if 0 <= index < len(self.waypoints):
            self.waypoints.pop(index)
            return True
        return False

    def clear_waypoints(self):
        self.waypoints = []

    # ------------------------------------------------------------------
    # Traced-path ops
    # ------------------------------------------------------------------

    def add_traced_path(self, path_voxels: np.ndarray, radius_voxels: float,
                        bbox=None, footprint=None):
        self.traced_paths.append(
            TracedPath(path_voxels=path_voxels, radius_voxels=radius_voxels,
                       bbox=bbox, footprint=footprint))
        self.mark_unsaved()

    def undo_last_path(self) -> bool:
        if not self.traced_paths:
            return False
        self.traced_paths.pop()
        self.mark_unsaved()
        return True

    def clear_paths(self):
        if self.traced_paths:
            self.mark_unsaved()
        self.traced_paths = []

    # ------------------------------------------------------------------
    # Noise deletion
    # ------------------------------------------------------------------

    def apply_deletion(self, mask: np.ndarray) -> int:
        """Remove `mask` voxels from `self.label`, remembering them for undo.

        Returns the number of voxels actually removed.
        """
        if mask is None or not mask.any():
            return 0
        # Only remember voxels that were actually present
        really = mask & self.label
        n = int(really.sum())
        if n == 0:
            return 0
        self.label = self.label & (~really)
        self.deleted_regions.append(really)
        self.mark_unsaved()
        return n

    def undo_last_deletion(self) -> int:
        """Restore voxels removed by the most recent deletion. Returns count."""
        if not self.deleted_regions:
            return 0
        mask = self.deleted_regions.pop()
        n = int(mask.sum())
        self.label = self.label | mask
        self.mark_unsaved()
        return n
