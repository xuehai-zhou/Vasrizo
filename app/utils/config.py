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

# Default tracing / speed function hyperparameters (match interactive_gap_tool_B3T3)
BETA_DEFAULT: float = 0.40
COL_DIFF_DEFAULT: float = 80.0
FILL_RADIUS_MM_DEFAULT: float = 0.5

# Visualization defaults
POINT_SIZE_DEFAULT: float = 3.0
DOWNSAMPLE_DEFAULT: int = 1
CT_DOWNSAMPLE_DEFAULT: int = 2

# Colors (RGB 0..1)
COLOR_LABEL = (1.0, 0.5, 0.0)       # orange — original/refined label
COLOR_CT = (0.4, 0.3, 0.2)          # brown — thresholded CT
COLOR_TRACED = (0.0, 0.8, 0.2)      # green — newly traced path
COLOR_WAYPOINT = (1.0, 0.0, 0.0)    # red — waypoint marker
COLOR_BG = (1.0, 1.0, 1.0)          # white background


@dataclass
class DataPaths:
    """Directory layout — override via CLI args.

    Vasrizo is domain-agnostic: the image volume supplied here is assumed
    to already be preprocessed (non-anatomy regions filled with an
    out-of-range value so HU thresholding alone excludes them). Works for
    pot-wall-removed root CT, skull-stripped MRI, background-subtracted
    microscopy, etc. An optional `{name}_interior.nii.gz` sitting next to
    the image will be picked up if present.
    """
    data_dir: str = ".."
    labels_subdir: str = "Training_data_v3/B3T3_nifti_picked"
    # Raw (pot-wall-intact) volumes. Pot-wall removal now happens live
    # inside Vasrizo via the "Apply pot-wall peel" control, so we no
    # longer need the heavy offline preprocessing stage.
    images_subdir: str = "Training_data_v3/B3T3_nifti_all"
    output_subdir: str = "Training_data_v3/B3T3_complete_picked"
