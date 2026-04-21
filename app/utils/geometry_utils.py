"""Pure-function geometry helpers — no Qt, no Open3D, no state."""
import numpy as np
from .config import SPACING


def physical_to_voxel(phys_pos: np.ndarray) -> np.ndarray:
    """(mm) -> voxel indices, rounded."""
    return np.round(np.asarray(phys_pos, dtype=np.float64) / SPACING).astype(np.int64)


def voxel_to_physical(vox_pos) -> np.ndarray:
    """voxel indices -> (mm)."""
    return np.asarray(vox_pos, dtype=np.float64) * SPACING


def label_to_coords(label: np.ndarray, downsample: int = 1) -> np.ndarray:
    """Convert a binary label volume to an (N, 3) array of physical coords."""
    mask = label
    sp = SPACING.copy()
    if downsample > 1:
        mask = mask[::downsample, ::downsample, ::downsample]
        sp = sp * downsample
    return np.argwhere(mask).astype(np.float64) * sp
