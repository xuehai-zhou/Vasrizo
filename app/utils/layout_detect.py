"""Auto-detect images/labels subdirectories inside a dropped/opened folder.

Heuristic: look for common conventions. Returns (images_dir, labels_dir)
with absolute paths, or None for anything we couldn't identify.
"""
from __future__ import annotations
import os
from typing import Optional


# Common subdir names, tried in order
IMAGE_CANDIDATES = (
    "images", "imagesTr", "imagesTs", "image", "img", "imgs",
    "volumes", "raw", "ct", "mri",
    # Root-CT specific defaults used by this repo
    "B3T3_nifti_all_preprocessed", "B3T3_nifti_all",
)

LABEL_CANDIDATES = (
    "labels", "labelsTr", "labelsTs", "label", "masks", "seg",
    "segmentation", "segmentations", "annot", "annotations",
    "B3T3_nifti_picked", "labelsCompleteNi_v1",
)


def _first_existing(root: str, names: tuple) -> Optional[str]:
    for n in names:
        p = os.path.join(root, n)
        if os.path.isdir(p):
            return os.path.abspath(p)
    # Also check nested: some datasets are root/Training_data/images/
    # (one level deep)
    try:
        entries = os.listdir(root)
    except OSError:
        return None
    for entry in entries:
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        for n in names:
            p = os.path.join(sub, n)
            if os.path.isdir(p):
                return os.path.abspath(p)
    return None


def detect_layout(folder: str) -> tuple[Optional[str], Optional[str]]:
    """Try to find (images_dir, labels_dir) under `folder`.

    Rules, in order:
      1. Look for well-known subdir names (IMAGE_CANDIDATES / LABEL_CANDIDATES)
         at the top level of `folder`.
      2. If not found, descend one level and look again — catches layouts
         like `folder/dataset_name/images/`.
      3. If only .nii.gz files are at the top level (no subdirs that match),
         treat `folder` itself as images and hope labels is somewhere else
         (handled later by the config dialog).
    """
    if not os.path.isdir(folder):
        return None, None

    images = _first_existing(folder, IMAGE_CANDIDATES)
    labels = _first_existing(folder, LABEL_CANDIDATES)
    return images, labels


def default_output_dir(labels_dir: Optional[str]) -> str:
    """A sensible output dir default: sibling of labels_dir named 'refined_labels'."""
    if not labels_dir:
        return ""
    return os.path.join(os.path.dirname(os.path.abspath(labels_dir)),
                        "refined_labels")
