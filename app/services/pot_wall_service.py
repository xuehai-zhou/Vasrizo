"""Fast pot-wall removal — interactive replacement for offline pot cropping.

Rationale
---------
A common preprocessing step for pot-grown root CT is, per axial slice,
threshold → fill → ``binary_erosion(disk(R))``. With R ≈ 38 px, that
erosion is O(N · R²) per slice and takes several minutes per volume.

We replace the erosion with a **per-slice 2D Euclidean distance transform**.
For each axial slice independently:

  1. non_air_slice = image_slice > air_threshold  (outer wall + contents)
  2. edt_mm = distance_transform_edt(non_air_slice, sampling=[dy, dx])
  3. interior_slice = edt_mm ≥ peel_xy_mm

EDT is O(N) per slice regardless of radius, and runs orders of magnitude
faster than the equivalent binary erosion (tens of ms per 512² slice vs.
seconds), giving ~2–5 s per full volume — fast enough to drive from a
button rather than an offline batch.

Rim-aware top preservation
--------------------------
Per-slice 2D EDT on its own over-peels the plant shoot / root crown
region above the pot rim: in those slices the only non-air material is
a narrow shoot, every voxel of which is within peel_xy_mm of air, so
the entire shoot would be thresholded away. We only want to peel the
pot — the sides and the base — not the shoot.

To avoid this, we identify the topmost slice whose 2D-EDT maximum
exceeds peel_xy_mm (i.e. the last slice where the pot wall still
bounds an interior wider than the peel). Every slice above that rim
keeps its non-air voxels verbatim; no peel is applied there.

The pot base (the frustum's flat bottom) is handled separately as an
explicit axial crop.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.ndimage import distance_transform_edt

from ..utils.config import SPACING


# Default air/outside threshold (HU). Anything below this is treated as
# "outside the pot". −500 HU sits well below soil/water/root intensities
# but comfortably above the −1000 HU of air, so the interface is stable
# even in the presence of partial-volume haze around the pot wall.
AIR_THRESHOLD_HU = -500.0


def remove_pot_walls(image: np.ndarray,
                     peel_xy_mm: float = 10.0,
                     peel_base_mm: float = 0.0,
                     base_axis: int = 2,
                     base_is_low: bool = True,
                     air_threshold: float = AIR_THRESHOLD_HU,
                     spacing: Optional[np.ndarray] = None,
                     ) -> np.ndarray:
    """Return a boolean interior mask of the pot.

    Parameters
    ----------
    image : (D, H, W) CT/MRI volume.
    peel_xy_mm : how far (mm) to peel inward from the outer pot wall.
        Value is a 3D Euclidean distance, so it also trims top/bottom
        slices that are thinner than `peel_xy_mm` — but only to the same
        depth they'd be trimmed by proximity to air, not to zero.
    peel_base_mm : if > 0, additionally remove this many mm from the
        pot base (the frustum's flat bottom). Applied AFTER the EDT peel,
        so it's an explicit hard crop of the base region.
    base_axis : which numpy axis runs vertically through the pot.
        Default 2 matches the convention in this repo.
    base_is_low : if True, the base of the pot is at the LOW index end of
        `base_axis` (e.g. z=0 is the bottom). Flip to False if your data
        is stored upside-down.
    air_threshold : HU cutoff for "outside the pot".
    spacing : (3,) voxel size in mm along each axis. Defaults to the
        repo's SPACING constant.

    Returns
    -------
    interior : (D, H, W) bool array — True inside the pot.
    """
    if spacing is None:
        spacing = SPACING
    spacing = np.asarray(spacing, dtype=np.float64)

    # Step 1: non-air mask — everything that belongs to the pot or its
    # contents. On raw CT this ALSO includes the plastic pot wall, which
    # is exactly what we're about to peel off. On already-preprocessed
    # volumes where the wall has been replaced with an out-of-window
    # fill, this is identical to the interior and the peel is a no-op.
    non_air = image > air_threshold
    if not non_air.any():
        return non_air.copy()

    # Step 2: per-slice 2D EDT along the two in-plane axes (the axes that
    # aren't `base_axis`). Cheap and gives the correct behavior for the
    # pot body — but naively applied it would *also* erase the shoot/
    # crown region above the pot rim, because in those slices the only
    # non-air material is a thin shoot whose every voxel sits within
    # peel_xy_mm of air. We guard against that below with a rim-aware
    # second pass.
    in_plane_axes = tuple(i for i in range(3) if i != base_axis)
    dy = spacing[in_plane_axes[0]]
    dx = spacing[in_plane_axes[1]]

    interior = np.zeros_like(non_air)
    n_slices = non_air.shape[base_axis]
    # Max 2D-EDT per slice. For pot-body slices this is roughly the pot's
    # interior radius in mm; for shoot/crown slices above the rim it
    # collapses to whatever half-thickness the shoot has — much smaller
    # than peel_xy_mm, which is how we detect "above the rim".
    slice_max_edt = np.zeros(n_slices, dtype=np.float64)
    for z in range(n_slices):
        # Index the slice along whichever axis is vertical
        sl = [slice(None), slice(None), slice(None)]
        sl[base_axis] = z
        sl_t = tuple(sl)
        slice_na = non_air[sl_t]
        if not slice_na.any():
            continue
        # 2D anisotropic EDT — mm-accurate, independent of peel radius.
        edt2d = distance_transform_edt(slice_na, sampling=(dy, dx))
        slice_max_edt[z] = edt2d.max()
        interior[sl_t] = edt2d >= peel_xy_mm

    # Step 3: rim-aware top preservation. The top of the pot is open
    # (plant shoot / root crown sticks out), and we do NOT want to peel
    # the shoot the way we peel the pot wall. The highest slice whose
    # 2D-EDT still exceeds peel_xy_mm is the pot rim: above it we only
    # see the shoot, so we keep those slices' non-air content verbatim.
    pot_body = slice_max_edt > peel_xy_mm
    if pot_body.any():
        body_idx = np.where(pot_body)[0]
        if base_is_low:
            # Base at low z → top is at high z. Keep everything above rim.
            z_rim = int(body_idx.max())
            if z_rim + 1 < n_slices:
                sl = [slice(None), slice(None), slice(None)]
                sl[base_axis] = slice(z_rim + 1, n_slices)
                sl_t = tuple(sl)
                interior[sl_t] = non_air[sl_t]
        else:
            # Base at high z → top is at low z. Keep everything below rim.
            z_rim = int(body_idx.min())
            if z_rim > 0:
                sl = [slice(None), slice(None), slice(None)]
                sl[base_axis] = slice(0, z_rim)
                sl_t = tuple(sl)
                interior[sl_t] = non_air[sl_t]

    # Step 4: optional base crop. The pot's flat bottom is a different,
    # asymmetric geometric feature from the side walls and isn't fully
    # handled by radial peeling alone. Top is intentionally preserved
    # above, so the base is the only direction still needing an axial
    # trim. We measure relative to the actual interior (after the radial
    # peel), not to the raw non-air — that way "peel 5 mm from the base"
    # removes the bottom 5 mm of the pot floor even if the raw image
    # contains extra air voxels below it.
    if peel_base_mm > 0:
        n_trim = int(np.ceil(peel_base_mm / spacing[base_axis]))
        if n_trim > 0:
            axes_others = tuple(i for i in range(interior.ndim)
                                if i != base_axis)
            live_after_peel = interior.any(axis=axes_others)
            live_idx = np.where(live_after_peel)[0]
            if len(live_idx) > 0:
                sl = [slice(None)] * interior.ndim
                if base_is_low:
                    z0 = int(live_idx[0])
                    sl[base_axis] = slice(z0, z0 + n_trim)
                else:
                    z1 = int(live_idx[-1])
                    sl[base_axis] = slice(max(0, z1 - n_trim + 1), z1 + 1)
                interior[tuple(sl)] = False

    return interior


# TODO(pybind11): `distance_transform_edt` is already C-optimized, but the
# non-air threshold + mask allocation could be fused in a single pass if
# this ever becomes the long pole.
