"""Build intensity-thresholded point clouds from an image volume.

If an interior mask is supplied it is applied as an AND; if not, the
threshold is applied to the image alone (the preprocessed image is
expected to carry out-of-window fills for non-anatomy regions).
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from ..utils.config import SPACING


def threshold_ct_to_coords(ct_vol: np.ndarray,
                           interior: Optional[np.ndarray],
                           hu_lower: float, hu_upper: float,
                           downsample: int = 1) -> np.ndarray:
    """Apply intensity thresholding and return (N, 3) physical coords.

    `interior` is optional. If supplied, voxels outside the mask are
    rejected even if they fall inside the intensity window.
    """
    if ct_vol is None:
        return np.empty((0, 3), dtype=np.float64)
    if downsample > 1:
        ct_vol = ct_vol[::downsample, ::downsample, ::downsample]
        if interior is not None:
            interior = interior[::downsample, ::downsample, ::downsample]
        sp = SPACING * downsample
    else:
        sp = SPACING.copy()
    keep = (ct_vol > hu_lower) & (ct_vol < hu_upper)
    if interior is not None:
        keep = keep & (interior > 0)
    return np.argwhere(keep).astype(np.float64) * sp


# TODO(pybind11): Called on every threshold-slider commit. Numpy vectorization
# is already adequate; a C++ version could skip the argwhere memory churn
# for very large volumes.
