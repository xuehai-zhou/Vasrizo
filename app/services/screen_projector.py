"""Project world-space coordinates to screen pixels.

Shared by the screen-space picker and the noise-deletion polygon test.
Pure numpy; no VTK types leak out (only a vtkCamera is needed at input).
"""
from __future__ import annotations
from typing import Optional
import numpy as np


def composite_world_to_ndc_matrix(camera, aspect: float) -> np.ndarray:
    """Return the 4x4 world-to-NDC matrix as a numpy array.

    Parameters
    ----------
    camera : vtkCamera
    aspect : float
        Renderer's tiled aspect ratio.
    """
    vtk_mat = camera.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
    return np.array([[vtk_mat.GetElement(i, j) for j in range(4)]
                     for i in range(4)], dtype=np.float64)


def project_world_to_screen(coords: np.ndarray, M: np.ndarray,
                            width: float, height: float
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project Nx3 world coords to screen pixels (VTK bottom-left origin).

    Returns
    -------
    sx, sy : (K,) arrays of screen x and y for points that fell inside the
             view frustum
    orig_indices : (K,) int array mapping each projected point back to the
             row of `coords` it came from
    """
    n = coords.shape[0]
    hc = np.empty((n, 4), dtype=np.float64)
    hc[:, :3] = coords
    hc[:, 3] = 1.0
    proj = hc @ M.T
    w = proj[:, 3]
    valid = w > 1e-8
    if not valid.any():
        empty = np.empty(0)
        return empty, empty, np.empty(0, dtype=np.int64)
    vp = proj[valid]
    vw = w[valid]
    ndc = vp[:, :3] / vw[:, None]
    in_view = (ndc[:, 0] >= -1.05) & (ndc[:, 0] <= 1.05) & \
              (ndc[:, 1] >= -1.05) & (ndc[:, 1] <= 1.05) & \
              (ndc[:, 2] >= -1.05) & (ndc[:, 2] <= 1.05)
    if not in_view.any():
        empty = np.empty(0)
        return empty, empty, np.empty(0, dtype=np.int64)
    v_ndc = ndc[in_view]
    valid_idx = np.where(valid)[0][in_view]
    sx = (v_ndc[:, 0] + 1.0) * 0.5 * width
    sy = (v_ndc[:, 1] + 1.0) * 0.5 * height
    return sx, sy, valid_idx
