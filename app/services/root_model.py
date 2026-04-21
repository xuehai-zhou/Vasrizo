"""Rootrak-style learned root appearance model.

Pure-Python, no Qt. Safe to call from background workers.

Two evaluation paths:
  * `speed_at(...)`        — original per-voxel local-histogram + JS divergence
                             matcher (slow; kept for API compatibility).
  * `speed_field_block(...)` — vectorized numpy pass that scores an entire
                             subvolume at once via a precomputed intensity
                             LUT. Orders of magnitude faster; used by the
                             tracer.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_dilation
from scipy.stats import entropy
from skimage.morphology import ball


class RootModel:
    """Learns p_root from labelled voxels; evaluates speed(voxel) via JS divergence."""

    def __init__(self, image: np.ndarray, label: np.ndarray,
                 n_bins: int = 256, kde_bw: float = 2.0,
                 col_diff: float = 80.0, beta: float = 0.40):
        root_vals = image[label]
        se = ball(3)
        shell = binary_dilation(label, se) & ~label
        bg_vals = image[shell]

        # Intensity bounds from labels
        self.b_lower = float(np.percentile(root_vals, 2))
        self.b_upper = float(np.percentile(root_vals, 98))
        margin = 0.15 * (self.b_upper - self.b_lower)
        self.b_lower -= margin
        self.b_upper += margin

        self.hist_min = float(min(root_vals.min(), bg_vals.min()))
        self.hist_max = float(max(root_vals.max(), bg_vals.max()))
        self.n_bins = n_bins
        self.col_diff = col_diff
        self.beta = beta

        bins = np.linspace(self.hist_min, self.hist_max, n_bins + 1)
        p_raw, _ = np.histogram(root_vals, bins=bins, density=False)
        sigma = kde_bw * n_bins / (self.hist_max - self.hist_min)
        self.p_root = self._kde_smooth(p_raw.astype(np.float64), sigma)

        # Background (shell) distribution — used to build the LUT below.
        p_bg_raw, _ = np.histogram(bg_vals, bins=bins, density=False)
        self.p_bg = self._kde_smooth(p_bg_raw.astype(np.float64), sigma)

        # Precomputed per-bin "rootness" score ∈ [0, 1]:
        #     score[b] = p_root[b] / (p_root[b] + p_bg[b])
        # This is the Bayesian posterior P(root | intensity) under an
        # uninformative prior. High = bin looks root-like; 0.5 = ambiguous.
        # We bake it into a LUT so `speed_field_block()` can score a
        # whole subvolume with one numpy gather.
        self._score_lut = (self.p_root /
                           (self.p_root + self.p_bg + 1e-12)).astype(np.float32)
        self._hist_min = float(self.hist_min)
        self._hist_range = float(self.hist_max - self.hist_min)

    @staticmethod
    def _kde_smooth(hist: np.ndarray, sigma: float) -> np.ndarray:
        s = gaussian_filter1d(hist, sigma=sigma)
        s = s / (s.sum() + 1e-12) + 1e-10
        return s / s.sum()

    def _bin(self, val: float) -> int:
        frac = (val - self.hist_min) / (self.hist_max - self.hist_min)
        return max(0, min(self.n_bins - 1, int(frac * self.n_bins)))

    def speed_at(self, image: np.ndarray, z: int, y: int, x: int,
                 kernel_d: int = 5) -> tuple[float, float]:
        """Evaluate speed function at (z,y,x). Returns (speed, js_divergence)."""
        shape = image.shape
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            return -1.0, 1.0

        center_val = image[z, y, x]
        k = kernel_d
        block = image[max(0, z - k):min(shape[0], z + k + 1),
                      max(0, y - k):min(shape[1], y + k + 1),
                      max(0, x - k):min(shape[2], x + k + 1)]
        selected = block[np.abs(block - center_val) <= self.col_diff]
        if len(selected) < 3:
            return -1.0, 1.0

        q = np.zeros(self.n_bins, dtype=np.float64)
        for v in selected:
            q[self._bin(v)] += 1
        d_mean = selected.mean()

        sigma = 2.0 * self.n_bins / (self.hist_max - self.hist_min)
        q = self._kde_smooth(q, sigma)

        m = 0.5 * (self.p_root + q)
        js = float(0.5 * entropy(self.p_root, m, base=2)
                   + 0.5 * entropy(q, m, base=2))

        if js < self.beta and self.b_lower <= d_mean <= self.b_upper:
            return +1.0, js
        return -1.0, js

    # ------------------------------------------------------------------
    # Fast bulk evaluation (used by the tracer)
    # ------------------------------------------------------------------

    def speed_field_block(self, image_block: np.ndarray,
                          corridor_mask: Optional[np.ndarray] = None
                          ) -> np.ndarray:
        """Score an entire image subvolume in one numpy pass.

        Parameters
        ----------
        image_block : (D, H, W) float array — CT/MRI subvolume inside the
            corridor bounding box.
        corridor_mask : optional bool array of the same shape. Voxels
            where this is False are returned as −1 (non-root) without
            evaluation, matching the behavior of the old corridor gate.

        Returns
        -------
        speed : (D, H, W) float32 array
            +1  → voxel's intensity looks root-like AND falls inside the
                  learned [b_lower, b_upper] bounds
            −1  → voxel rejected
        """
        # 1. Map every voxel's intensity to an integer bin (vectorized).
        hr = self._hist_range if self._hist_range > 0 else 1.0
        bins = ((image_block - self._hist_min) / hr * self.n_bins)
        bins = np.clip(bins.astype(np.int32), 0, self.n_bins - 1)

        # 2. Gather the precomputed score per voxel.
        score = self._score_lut[bins]

        # 3. Apply the same two gating conditions that speed_at() uses:
        #      score > 0.5  (i.e. JS < beta proxy — more root than bg)
        #      b_lower ≤ intensity ≤ b_upper
        speed = np.where(
            (score > 0.5)
            & (image_block >= self.b_lower)
            & (image_block <= self.b_upper),
            np.float32(1.0), np.float32(-1.0),
        )

        # 4. Mask out voxels outside the corridor if requested.
        if corridor_mask is not None:
            speed = np.where(corridor_mask, speed, np.float32(-1.0))
        return speed
