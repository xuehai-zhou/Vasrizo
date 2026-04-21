"""Noise-point deletion: determine which label voxels fall inside a
closed polygon drawn in screen space, and apply the deletion.

Inspired by CloudCompare's ccGraphicalSegmentationTool (qCC/...). The
core routine mirrors its `CCCoreLib::ManualSegmentationTools::isPointInsidePoly`
2D ray-cast test, but operates on a numpy batch of projected points.
"""
from __future__ import annotations
import numpy as np


def points_in_polygon(sx: np.ndarray, sy: np.ndarray,
                      polygon: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test.

    Parameters
    ----------
    sx, sy   : (N,) arrays of screen-space x and y
    polygon  : (M, 2) closed polygon vertices (last vertex != first is OK,
               the algorithm closes implicitly)

    Returns
    -------
    inside : (N,) boolean array, True if the point is inside the polygon.
    """
    n = len(sx)
    if n == 0 or len(polygon) < 3:
        return np.zeros(n, dtype=bool)

    inside = np.zeros(n, dtype=bool)
    m = len(polygon)
    j = m - 1
    for i in range(m):
        x_i, y_i = polygon[i]
        x_j, y_j = polygon[j]
        # Edge (j -> i): does a horizontal ray from (sx, sy) to +∞ cross it?
        cond = ((y_i > sy) != (y_j > sy))
        # Avoid divide-by-zero when the edge is horizontal
        denom = (y_j - y_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = x_i + (sy - y_i) * (x_j - x_i) / denom
        crosses = cond & (sx < x_intersect)
        inside ^= crosses
        j = i
    return inside


def build_voxel_deletion_mask(label: np.ndarray,
                              label_coords_world: np.ndarray,
                              valid_indices_of_world_coords: np.ndarray,
                              screen_x: np.ndarray, screen_y: np.ndarray,
                              polygon_px: np.ndarray,
                              spacing: np.ndarray) -> np.ndarray:
    """Compute a boolean mask (same shape as `label`) of voxels to delete.

    The caller has already projected label points to screen space and kept
    only those inside the view frustum — we operate on those.

    Parameters
    ----------
    label                           : (D, H, W) bool array
    label_coords_world              : (N_all, 3) world coords of all label
                                      voxels, in same order as label_to_coords
                                      returned them
    valid_indices_of_world_coords   : (K,) indices into label_coords_world for
                                      the K points that fell inside the view
                                      frustum (and whose screen coords are in
                                      screen_x / screen_y)
    screen_x, screen_y              : (K,) screen-space projected coords
    polygon_px                      : (M, 2) polygon vertices in screen pixels
    spacing                         : (3,) voxel size in mm

    Returns
    -------
    to_delete : bool array with the shape of `label`
    """
    inside = points_in_polygon(screen_x, screen_y, polygon_px)
    if not inside.any():
        return np.zeros_like(label, dtype=bool)

    # Map visible-and-inside projected points back to voxel indices.
    selected_world = label_coords_world[valid_indices_of_world_coords[inside]]
    voxel_indices = np.round(selected_world / spacing).astype(np.int64)

    to_delete = np.zeros_like(label, dtype=bool)
    # Clip to volume bounds and flag
    shape = np.array(label.shape, dtype=np.int64)
    valid = ((voxel_indices >= 0).all(axis=1) &
             (voxel_indices < shape).all(axis=1))
    vi = voxel_indices[valid]
    to_delete[vi[:, 0], vi[:, 1], vi[:, 2]] = True
    # Only keep voxels that are actually currently in the label
    to_delete &= label
    return to_delete
