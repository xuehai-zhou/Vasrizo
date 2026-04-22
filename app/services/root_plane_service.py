"""Estimate a root-following slab whose planes are parallel to the pot axis.

The user wants the fitted slicing planes to be vertical with respect to the
pot, i.e. both planes are parallel to the pot center axis and the slab just
captures the selected root between them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pot_wall_service import estimate_pot_cylinder_geometry


@dataclass
class RootScreenSlabEstimate:
    slab_origin: np.ndarray
    slab_normal: np.ndarray
    offset_mm: float
    thickness_mm: float
    support_points: int
    support_source: str


def _closest_distance_to_polyline(points: np.ndarray,
                                  polyline: np.ndarray) -> np.ndarray:
    if len(polyline) < 2 or len(points) == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    best = np.full(len(points), np.inf, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    line = np.asarray(polyline, dtype=np.float64)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq <= 1e-10:
            d = np.linalg.norm(pts - a[None, :], axis=1)
        else:
            t = ((pts - a[None, :]) @ ab) / ab_len_sq
            t = np.clip(t, 0.0, 1.0)
            closest = a[None, :] + t[:, None] * ab[None, :]
            d = np.linalg.norm(pts - closest, axis=1)
        best = np.minimum(best, d)
    return best


def _select_support_points(points: Optional[np.ndarray],
                           polyline: np.ndarray,
                           corridor_radius_mm: float) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    dist = _closest_distance_to_polyline(pts, polyline)
    return pts[dist <= corridor_radius_mm]


def _robust_span(values: np.ndarray,
                 lo_pct: float = 2.0,
                 hi_pct: float = 98.0) -> tuple[float, float]:
    lo = float(np.percentile(values, lo_pct))
    hi = float(np.percentile(values, hi_pct))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def estimate_root_screen_slab(waypoints_phys: list[np.ndarray],
                              label_points: Optional[np.ndarray],
                              ct_points: Optional[np.ndarray],
                              image_volume: Optional[np.ndarray],
                              view_origin: np.ndarray,
                              view_normal: np.ndarray,
                              corridor_radius_mm: float = 8.0,
                              min_support_points: int = 12) -> RootScreenSlabEstimate:
    """Estimate a vertical slab aligned with the root's gross growth trend."""
    polyline = np.asarray(waypoints_phys, dtype=np.float64)
    if len(polyline) < 2:
        raise ValueError("Need at least 2 waypoints to estimate a slab.")

    pot_geom = None
    if image_volume is not None:
        pot_geom = estimate_pot_cylinder_geometry(image_volume)
    if pot_geom is None:
        raise ValueError("Could not estimate the pot center axis.")

    pot_axis = np.zeros(3, dtype=np.float64)
    pot_axis[pot_geom.base_axis] = 1.0

    support_label = _select_support_points(label_points, polyline, corridor_radius_mm)
    support_ct = _select_support_points(
        ct_points, polyline, max(corridor_radius_mm, 12.0))

    if len(support_label) >= min_support_points:
        pts = support_label
        source = "label"
    elif len(support_ct) >= min_support_points:
        pts = support_ct
        source = "ct"
    else:
        pts = np.concatenate(
            [arr for arr in (support_label, support_ct) if len(arr) > 0],
            axis=0,
        ) if (len(support_label) > 0 or len(support_ct) > 0) else polyline.copy()
        source = "mixed" if len(pts) > len(polyline) else "waypoints"

    if len(pts) < 2:
        raise ValueError("Not enough nearby root points were found.")

    growth_dir = polyline[-1] - polyline[0]
    lateral_dir = growth_dir - pot_axis * float(np.dot(growth_dir, pot_axis))

    lat_norm = float(np.linalg.norm(lateral_dir))
    if lat_norm <= 1e-8:
        radial_pts = pts.copy()
        radial_pts[:, pot_geom.base_axis] = 0.0
        centered = radial_pts - radial_pts.mean(axis=0, keepdims=True)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        lateral_dir = eigvecs[:, int(np.argmax(eigvals))]
        lateral_dir[pot_geom.base_axis] = 0.0
        lat_norm = float(np.linalg.norm(lateral_dir))

    if lat_norm <= 1e-8:
        fallback = np.asarray(view_normal, dtype=np.float64)
        fallback = fallback - pot_axis * float(np.dot(fallback, pot_axis))
        lat_norm = float(np.linalg.norm(fallback))
        if lat_norm <= 1e-8:
            raise ValueError("Could not determine the root-aligned slab direction.")
        lateral_dir = fallback / lat_norm
    else:
        lateral_dir = lateral_dir / lat_norm

    # The slab planes must be parallel to both the pot axis and the root's
    # overall growth direction. Their normal is therefore orthogonal to both.
    normal = np.cross(pot_axis, lateral_dir)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        raise ValueError("Could not build a vertical slab from the selected root direction.")
    normal = normal / norm

    origin = pts.mean(axis=0)

    signed = (pts - origin[None, :]) @ normal
    lo, hi = _robust_span(signed)
    pad = 1.0
    lo -= pad
    hi += pad
    thickness = max(2.0, hi - lo)

    return RootScreenSlabEstimate(
        slab_origin=origin.astype(np.float64),
        slab_normal=normal.astype(np.float64),
        offset_mm=float(lo),
        thickness_mm=float(thickness),
        support_points=int(len(pts)),
        support_source=source,
    )
