"""Sample discovery and loading.

Contract per sample:
    {images_dir}/{name}.nii.gz         OR  {images_dir}/{name}_0000.nii.gz
    {labels_dir}/{name}.nii.gz

The image volume can be raw or preprocessed. If raw CT contains a pot
or container wall, `SampleData.interior_mask` is populated live by the
in-app pot-wall peel (see services/pot_wall_service.py).
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Optional

import nibabel as nib
import numpy as np


# Accepted image filename suffixes (tried in order).
IMAGE_SUFFIXES = ("_0000.nii.gz", ".nii.gz")


@dataclass
class SampleData:
    """In-memory payload for one sample.

    `interior_mask` starts as None and gets populated at runtime by the
    "Apply pot-wall peel" action when the user needs it.
    """
    name: str
    image: np.ndarray              # float32 image volume
    label: np.ndarray              # bool label volume
    lbl_nii: nib.Nifti1Image       # for saving with matching affine/header
    interior_mask: Optional[np.ndarray] = None  # uint8 0/1, populated live


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _image_path(images_dir: str, name: str) -> Optional[str]:
    """Return the first existing image file for `name`, or None."""
    for suffix in IMAGE_SUFFIXES:
        p = os.path.join(images_dir, f"{name}{suffix}")
        if os.path.exists(p):
            return p
    return None


def _image_names_in_dir(images_dir: str) -> set[str]:
    """Set of sample names discovered in `images_dir` (either suffix)."""
    if not os.path.isdir(images_dir):
        return set()
    names: set[str] = set()
    for f in os.listdir(images_dir):
        for suffix in IMAGE_SUFFIXES:
            if f.endswith(suffix):
                names.add(f[: -len(suffix)])
                break
    return names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_samples(labels_dir: str, images_dir: str) -> list[str]:
    """Intersection of label names and image names, sorted."""
    if not os.path.isdir(labels_dir):
        return []
    labels = {f[: -len(".nii.gz")]
              for f in os.listdir(labels_dir) if f.endswith(".nii.gz")}
    images = _image_names_in_dir(images_dir)
    if images:
        return sorted(labels & images)
    return sorted(labels)


def _read_image_array(path: str) -> np.ndarray:
    """Decode a CT volume to float32. Isolated so it can run in a thread."""
    return nib.load(path).get_fdata(dtype=np.float32)


def _read_label_array(lbl_nii: nib.Nifti1Image) -> np.ndarray:
    """Decode a binary label without the float32 round-trip.

    `get_fdata(dtype=float32) > 0.5` allocates a whole float32 volume just
    to threshold it back to bool. `dataobj` hands us the stored dtype
    (usually uint8/int16) directly, which is both faster to decompress
    and avoids the 4× memory blow-up.
    """
    raw = np.asarray(lbl_nii.dataobj)
    return raw > 0


def load_sample(name: str, labels_dir: str, images_dir: str) -> SampleData:
    """Load one sample. Raises FileNotFoundError if required files missing.

    Image and label are decoded concurrently because zlib releases the GIL
    during inflate — a 10-20% wall-clock win over the sequential path on
    the typical CT+mask pair, and more when the label is sizeable.
    """
    img_path = _image_path(images_dir, name)
    if img_path is None:
        raise FileNotFoundError(
            f"No image for '{name}' in {images_dir} "
            f"(tried suffixes {IMAGE_SUFFIXES})")

    lbl_path = os.path.join(labels_dir, f"{name}.nii.gz")
    if not os.path.exists(lbl_path):
        raise FileNotFoundError(lbl_path)

    # Open the label header on the main thread (cheap) so the worker
    # only pays for the data fetch. Parallelize the two big decodes.
    lbl_nii = nib.load(lbl_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_image = pool.submit(_read_image_array, img_path)
        f_label = pool.submit(_read_label_array, lbl_nii)
        image = f_image.result()
        label = f_label.result()

    return SampleData(
        name=name, image=image, label=label, lbl_nii=lbl_nii,
        interior_mask=None,
    )
