"""Pot-wall removal based on a fitted cylindrical pot model.

The user's requirement is intentionally *not* a generic 3D erosion:

1. Peel only the outer radial shell of the pot wall, i.e. the region
   ``R in [R_max - outer_wall, R_max]`` around the fitted pot axis.
2. Peel only the bottom slab of thickness ``outer_wall`` / ``base``
   measured from the pot bottom toward the top.
3. Do not peel the top horizontal slab near ``Y=0`` at all.

To do that we first estimate a cylindrical pot axis from the occupied
volume itself, then build the mask analytically in cylindrical
coordinates around that axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils.config import SPACING


AIR_THRESHOLD_HU = -500.0


@dataclass
class PotCylinderGeometry:
    """Estimated cylindrical pot geometry in physical mm coordinates."""

    base_axis: int
    radial_axes: tuple[int, int]
    center_mm: np.ndarray
    radius_mm: float
    y_min_index: int
    y_max_index: int
    axis_start_mm: np.ndarray
    axis_end_mm: np.ndarray


def _slice_tuple(axis: int, idx: int, ndim: int = 3) -> tuple:
    sl = [slice(None)] * ndim
    sl[axis] = idx
    return tuple(sl)


def estimate_pot_cylinder_geometry(
    image: np.ndarray,
    base_axis: int = 1,
    air_threshold: float = AIR_THRESHOLD_HU,
    spacing: Optional[np.ndarray] = None,
) -> Optional[PotCylinderGeometry]:
    """Fit a pot axis and radius from the occupied part of the volume."""
    if spacing is None:
        spacing = SPACING
    spacing = np.asarray(spacing, dtype=np.float64)

    non_air = image > air_threshold
    if not non_air.any():
        return None

    radial_axes = tuple(i for i in range(3) if i != base_axis)
    live_y = non_air.any(axis=radial_axes)
    live_idx = np.where(live_y)[0]
    if len(live_idx) == 0:
        return None

    # Estimate the axis from the stable middle body of the pot rather than
    # from the top rim or bottom cap, which are more likely to be irregular.
    n_live = len(live_idx)
    trim = int(max(1, round(n_live * 0.18)))
    body_idx = live_idx[trim:-trim] if n_live > 2 * trim + 4 else live_idx

    center_a = []
    center_b = []
    radii = []
    for y_idx in body_idx:
        sl = non_air[_slice_tuple(base_axis, int(y_idx), non_air.ndim)]
        pts = np.argwhere(sl)
        if len(pts) < 64:
            continue
        a_mm = pts[:, 0].astype(np.float64) * spacing[radial_axes[0]]
        b_mm = pts[:, 1].astype(np.float64) * spacing[radial_axes[1]]
        ca = 0.5 * (float(a_mm.min()) + float(a_mm.max()))
        cb = 0.5 * (float(b_mm.min()) + float(b_mm.max()))
        r = np.sqrt((a_mm - ca) ** 2 + (b_mm - cb) ** 2)
        center_a.append(ca)
        center_b.append(cb)
        # Use a high quantile rather than the absolute max to avoid
        # isolated outliers from shoots / roots skewing the fitted wall.
        radii.append(float(np.quantile(r, 0.985)))

    if not radii:
        return None

    center = np.zeros(3, dtype=np.float64)
    center[radial_axes[0]] = float(np.median(center_a))
    center[radial_axes[1]] = float(np.median(center_b))
    center[base_axis] = 0.5 * float(
        (live_idx[0] + live_idx[-1]) * spacing[base_axis]
    )

    radius_mm = float(np.quantile(np.asarray(radii, dtype=np.float64), 0.85))

    axis_start = center.copy()
    axis_end = center.copy()
    axis_start[base_axis] = float(live_idx[0] * spacing[base_axis])
    axis_end[base_axis] = float(live_idx[-1] * spacing[base_axis])

    return PotCylinderGeometry(
        base_axis=base_axis,
        radial_axes=radial_axes,
        center_mm=center,
        radius_mm=radius_mm,
        y_min_index=int(live_idx[0]),
        y_max_index=int(live_idx[-1]),
        axis_start_mm=axis_start,
        axis_end_mm=axis_end,
    )


def remove_pot_walls(
    image: np.ndarray,
    peel_xy_mm: float = 15.0,
    peel_base_mm: float = 0.0,
    base_axis: int = 1,
    base_is_low: bool = False,
    air_threshold: float = AIR_THRESHOLD_HU,
    spacing: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a boolean interior mask after cylindrical wall / base peel.

    This removes only:
    - the radial shell near the fitted pot wall, and
    - the bottom slab near the fitted pot base.

    It does *not* remove a top slab near Y=0.
    """
    if spacing is None:
        spacing = SPACING
    spacing = np.asarray(spacing, dtype=np.float64)

    non_air = image > air_threshold
    if not non_air.any():
        return non_air.copy()

    geom = estimate_pot_cylinder_geometry(
        image=image,
        base_axis=base_axis,
        air_threshold=air_threshold,
        spacing=spacing,
    )
    if geom is None:
        return non_air.copy()

    radial_axes = geom.radial_axes
    radial_shape = tuple(image.shape[ax] for ax in radial_axes)
    coords_a = (
        np.arange(radial_shape[0], dtype=np.float64) * spacing[radial_axes[0]]
        - geom.center_mm[radial_axes[0]]
    )
    coords_b = (
        np.arange(radial_shape[1], dtype=np.float64) * spacing[radial_axes[1]]
        - geom.center_mm[radial_axes[1]]
    )
    grid_a, grid_b = np.meshgrid(coords_a, coords_b, indexing="ij")
    radial_dist_mm = np.sqrt(grid_a ** 2 + grid_b ** 2)

    inner_radius_mm = max(0.0, float(geom.radius_mm) - float(peel_xy_mm))
    keep_radially = radial_dist_mm <= (inner_radius_mm + 1e-6)

    interior = np.zeros_like(non_air, dtype=bool)
    n_slices = image.shape[base_axis]
    for y_idx in range(n_slices):
        sl_t = _slice_tuple(base_axis, y_idx, non_air.ndim)
        slice_na = non_air[sl_t]
        if not slice_na.any():
            continue
        interior[sl_t] = slice_na & keep_radially

    if peel_base_mm > 0:
        n_trim = int(np.ceil(float(peel_base_mm) / spacing[base_axis]))
        if n_trim > 0:
            sl = [slice(None)] * interior.ndim
            if base_is_low:
                y0 = geom.y_min_index
                sl[base_axis] = slice(y0, min(n_slices, y0 + n_trim))
            else:
                y1 = geom.y_max_index
                sl[base_axis] = slice(max(0, y1 - n_trim + 1), y1 + 1)
            interior[tuple(sl)] = False

    return interior
