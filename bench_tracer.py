"""Micro-benchmark the fast tracer vs. the baseline, on real volume shapes.

Run from repo root as:
    python Vasrizo/bench_tracer.py
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.root_model import RootModel
from app.services.tracing_service import (
    find_path_between, compute_speed_field, _dijkstra_python,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="../B3T3_nifti_all_preprocessed/B3T3G10S1_0000.nii.gz")
    ap.add_argument("--label", default="../Training_data_v3/B3T3_nifti_picked/B3T3G10S1.nii.gz")
    args = ap.parse_args()

    print("Loading…")
    t0 = time.time()
    image = nib.load(args.image).get_fdata(dtype=np.float32)
    label = nib.load(args.label).get_fdata(dtype=np.float32) > 0.5
    print(f"  image={image.shape} label.sum={int(label.sum()):,}   "
          f"({time.time()-t0:.1f}s)")

    print("Training RootModel…")
    t0 = time.time()
    model = RootModel(image, label, beta=0.4, col_diff=80.0)
    print(f"  {time.time()-t0:.2f}s")

    # Pick two label voxels separated by ~40 vx as a realistic "gap"
    coords = np.argwhere(label)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(coords), 1)[0]
    start = coords[idx]
    # Find another labeled voxel roughly 40-60 voxels away
    d = np.linalg.norm(coords - start, axis=1)
    cand = np.where((d > 30) & (d < 70))[0]
    if len(cand) == 0:
        print("No suitable endpoints — relaxing distance constraint.")
        cand = np.where(d > 10)[0]
    end = coords[rng.choice(cand)]
    print(f"Start={tuple(start.tolist())}  End={tuple(end.tolist())}  "
          f"distance={np.linalg.norm(end - start):.1f} vx")

    # ---- Trace with new fast tracer ----
    print("\nFast tracer (vectorized speed field + anchor termination):")
    t0 = time.time()
    path1 = find_path_between(image, model, start, end,
                              label=label, goal_radius=4,
                              progress=lambda m: print(f"  {m}"))
    t_fast = time.time() - t0
    print(f"  total: {t_fast:.2f}s   path={None if path1 is None else len(path1)} voxels")

    # ---- Same but without anchor (to gauge early-term savings) ----
    print("\nFast tracer (no anchor termination, for comparison):")
    t0 = time.time()
    path2 = find_path_between(image, model, start, end,
                              label=None,
                              progress=lambda m: print(f"  {m}"))
    t_no_anchor = time.time() - t0
    print(f"  total: {t_no_anchor:.2f}s   path={None if path2 is None else len(path2)} voxels")


if __name__ == "__main__":
    main()
