"""Application-wide constants and default paths."""
from dataclasses import dataclass
import numpy as np


# Voxel spacing in mm — must match how labels were authored
SPACING: np.ndarray = np.array([0.39, 0.39, 0.2], dtype=np.float64)

# HU thresholding (Rootrak-style CT preprocessing)
HU_LOWER_DEFAULT: float = -500.0
HU_UPPER_DEFAULT: float = 700.0
HU_STEP: float = 5.0           # slider increment
HU_MIN_LIMIT: float = -1000.0
HU_MAX_LIMIT: float = 1500.0

# Default tracing / speed function hyperparameters
BETA_DEFAULT: float = 0.40
COL_DIFF_DEFAULT: float = 80.0
FILL_RADIUS_MM_DEFAULT: float = 0.5

# Visualization defaults
POINT_SIZE_DEFAULT: float = 3.0
DOWNSAMPLE_DEFAULT: int = 1
CT_DOWNSAMPLE_DEFAULT: int = 1

# Colors (RGB 0..1)
COLOR_LABEL = (1.0, 0.5, 0.0)       # orange — original/refined label
COLOR_CT = (0.4, 0.3, 0.2)          # brown — thresholded CT
COLOR_TRACED = (0.0, 0.8, 0.2)      # green — newly traced path
COLOR_WAYPOINT = (1.0, 0.0, 0.0)    # red — waypoint marker
COLOR_BG = (1.0, 1.0, 1.0)          # white background


@dataclass
class DataPaths:
    """Directory layout — override via CLI args or the in-app dataset dialog.

    Vasrizo is domain-agnostic: the image volume can be either raw or
    already preprocessed. If raw CT contains a pot/container wall, the
    interactive "Apply pot-wall peel" control removes it live. Works
    equally well for vessel tracing, neuron reconstruction, or any
    partial-label repair task on a 3D image volume.

    These defaults are placeholders only; normal use is to pick a
    dataset folder from File → Open Dataset Folder… (Ctrl+O) or by
    dropping a folder onto the main window.
    """
    data_dir: str = "."
    labels_subdir: str = "labels"
    images_subdir: str = "images"
    output_subdir: str = "refined_labels"
