"""Save refined labels back to NIfTI, preserving affine/header.

Optimization notes
------------------
* Labels are binary, so we save as uint8 rather than float32 — 4× less
  data to feed into the gzip compressor.
* We pin nibabel's gzip compresslevel to 1 (fast + still shrinks a binary
  mask 10-30×). Older nibabel defaulted to 9, which is ~5× slower.
* If the caller already handed us a contiguous uint8 array, we skip the
  re-cast entirely (SaveWorker does this in the background thread).
* Atomic rename via tempfile prevents a crash mid-save from leaving a
  half-written file in the output directory.
"""
import os
import tempfile
import nibabel as nib
from nibabel.openers import Opener
import numpy as np


# Pin low compresslevel. nibabel 5.x defaults to 1, but being explicit
# protects us from a future version regression.
Opener.default_compresslevel = 1


def save_label(out_dir: str, name: str, label: np.ndarray,
               reference_nii: nib.Nifti1Image) -> str:
    """Write `label` (bool/uint8 accepted) to `{out_dir}/{name}.nii.gz`.

    Saved dtype is uint8 (binary mask); header dtype is updated to match.
    Returns the final output path.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.nii.gz")

    if label.dtype == np.uint8 and label.flags.c_contiguous:
        data = label
    else:
        data = np.ascontiguousarray(label.astype(np.uint8))
    # Don't mutate the caller's header; copy it and re-stamp the dtype
    hdr = reference_nii.header.copy()
    try:
        hdr.set_data_dtype(np.uint8)
    except Exception:
        pass
    out_nii = nib.Nifti1Image(data, reference_nii.affine, hdr)

    # nibabel identifies format by filename suffix, so the temp path must
    # end in `.nii.gz`. We prefix with a dot so the in-progress file is
    # hidden from most file managers / scripts.
    fd, tmp_path = tempfile.mkstemp(
        suffix=".nii.gz", prefix=f".{name}.", dir=out_dir)
    os.close(fd)
    try:
        nib.save(out_nii, tmp_path)
        os.replace(tmp_path, out_path)
    except Exception:
        # Don't leave a stale .hidden.tmp on failure
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    return out_path
