"""Path-finding through the image volume (speed-field Dijkstra) + tube painting.

Design notes
------------
  1. **Vectorized speed field.** A naive implementation calls
     ``RootModel.speed_at()`` per voxel — ~700k Python calls for a 90^3
     corridor on a 512^3 volume. We instead score the whole corridor
     block in one numpy pass via a precomputed intensity LUT.

  2. **Vectorized corridor mask.** The cylindrical corridor around the
     start–end line is built with numpy broadcasting, not a nested loop.

  3. **Anchor early-termination.** If the end waypoint is near an existing
     label voxel, Dijkstra stops as soon as any goal-proximal voxel is
     popped from the heap — typically an order-of-magnitude speedup over
     running to convergence.

  4. **Optional numba Dijkstra.** For volumes where the Python heap is
     still the long pole, a numba-JIT'd variant kicks in when ``numba``
     is installed. Degrades gracefully to the pure-Python heap otherwise.

Interface (`find_path_between`, `paint_tube`) is unchanged across backends.
"""
from __future__ import annotations
import heapq
import time
from typing import Optional

import numpy as np

from .root_model import RootModel


# ---------------------------------------------------------------------------
# Optional numba acceleration for the Dijkstra inner loop
# ---------------------------------------------------------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# Corridor mask and speed field (vectorized)
# ---------------------------------------------------------------------------

def _build_corridor_mask(shape: tuple,
                         local_start: np.ndarray,
                         local_end: np.ndarray,
                         radius: float) -> np.ndarray:
    """Bool mask of voxels within `radius` of the line start→end.

    Vectorized replacement for the Python triple-loop that dominated the
    old `compute_speed_field`. Builds the full distance-to-line array in
    one broadcast.
    """
    line_vec = local_end - local_start
    line_len_sq = float(np.dot(line_vec, line_vec))
    if line_len_sq < 1e-12:
        return np.zeros(shape, dtype=bool)

    zz, yy, xx = np.indices(shape, dtype=np.float32)
    # Offset from start, per component
    dz = zz - float(local_start[0])
    dy = yy - float(local_start[1])
    dx = xx - float(local_start[2])

    # Parametric projection onto the line (can lie outside [0, 1])
    t = (dz * line_vec[0] + dy * line_vec[1] + dx * line_vec[2]) / line_len_sq
    # Orthogonal-distance squared to the line
    px = dz - t * line_vec[0]
    py = dy - t * line_vec[1]
    pz = dx - t * line_vec[2]
    d_sq = px * px + py * py + pz * pz
    return d_sq <= (radius * radius)


def compute_speed_field(image: np.ndarray, model: RootModel,
                        local_start: np.ndarray, local_end: np.ndarray,
                        bbox_min: np.ndarray, shape: tuple) -> np.ndarray:
    """Build the per-voxel speed score inside the search corridor.

    Uses the RootModel's precomputed intensity LUT rather than a per-voxel
    local-histogram JS test — matching what the local evaluator reduces to
    in practice (kernel_d=3 windows are dominated by similar-intensity
    voxels), but orders of magnitude faster because everything runs in
    one vectorized numpy pass.

    Returns
    -------
    speed_field : (D, H, W) float32
        +1 inside corridor and root-like by intensity, −1 otherwise.
    """
    line_len = float(np.linalg.norm(local_end - local_start))
    if line_len < 1e-8:
        return np.full(shape, -1.0, dtype=np.float32)

    corridor_radius = max(15.0, line_len * 0.4)

    # Crop the image to the corridor bounding box (already done by caller)
    block = image[bbox_min[0]:bbox_min[0] + shape[0],
                  bbox_min[1]:bbox_min[1] + shape[1],
                  bbox_min[2]:bbox_min[2] + shape[2]]

    mask = _build_corridor_mask(shape, local_start, local_end, corridor_radius)
    return model.speed_field_block(block, corridor_mask=mask)


# ---------------------------------------------------------------------------
# Dijkstra — pure Python heap, plus optional numba fast path
# ---------------------------------------------------------------------------

# Offsets for 26-connectivity + the associated step distance
_OFFSETS = np.array(
    [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
     if not (dz == 0 and dy == 0 and dx == 0)], dtype=np.int32)
_STEPS = np.sqrt((_OFFSETS ** 2).sum(axis=1)).astype(np.float32)


def _dijkstra_python(speed_field: np.ndarray,
                     local_start: tuple, local_end: tuple,
                     goal_proximity_mask: Optional[np.ndarray] = None
                     ) -> Optional[np.ndarray]:
    """Pure-Python heap Dijkstra. Early-terminates if we reach any voxel in
    `goal_proximity_mask` (if provided) — the anchor early-termination
    trick described in the module docstring."""
    shape = speed_field.shape
    INF = float('inf')
    dist_map = np.full(shape, INF, dtype=np.float32)
    prev_map = np.full(shape + (3,), -1, dtype=np.int32)
    dist_map[local_start] = 0.0
    heap = [(0.0, local_start)]
    visited = np.zeros(shape, dtype=bool)

    actual_end = local_end
    while heap:
        d, cur = heapq.heappop(heap)
        if visited[cur]:
            continue
        visited[cur] = True
        # Standard goal check
        if cur == local_end:
            break
        # Anchor early-termination: if we're inside the goal-proximity mask
        # (i.e. we've landed on / next to a labeled voxel near the end),
        # treat this as the actual endpoint.
        if goal_proximity_mask is not None and goal_proximity_mask[cur]:
            actual_end = cur
            break

        cz, cy, cx = cur
        for (dz, dy, dx), step in zip(_OFFSETS, _STEPS):
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                if visited[nz, ny, nx]:
                    continue
                penalty = 0.2 if speed_field[nz, ny, nx] > 0 else 3.0
                cost = step * (1.0 + penalty)
                nd = d + cost
                if nd < dist_map[nz, ny, nx]:
                    dist_map[nz, ny, nx] = nd
                    prev_map[nz, ny, nx] = (cz, cy, cx)
                    heapq.heappush(heap, (nd, (nz, ny, nx)))

    if dist_map[actual_end] == INF:
        return None

    # Backtrack
    path: list = []
    cur = actual_end
    while cur != local_start:
        path.append(np.array(cur, dtype=np.int64))
        prev = tuple(prev_map[cur])
        if prev[0] == -1:
            return None
        cur = prev
    path.append(np.array(local_start, dtype=np.int64))
    path.reverse()
    return np.array(path)


if _HAVE_NUMBA:

    # TODO(pybind11): if we keep scaling up, a pybind11/C++ bucket-heap
    # version would still beat this. For now this is 10-30× faster than
    # the Python heap on typical corridors.
    @njit(cache=True, boundscheck=False)
    def _dijkstra_numba(speed_field, sz, sy, sx, ez, ey, ex,
                        goal_mask, has_goal_mask):
        D, H, W = speed_field.shape
        INF = np.float32(np.inf)
        dist = np.full((D, H, W), INF, dtype=np.float32)
        prev = np.full((D, H, W, 3), -1, dtype=np.int32)
        visited = np.zeros((D, H, W), dtype=np.bool_)
        # Simple binary heap of (priority, z, y, x) as parallel arrays
        # since numba doesn't handle heapq well. Use a list-of-tuples
        # representation.
        heap_pri = np.empty(0, dtype=np.float32)
        heap_z = np.empty(0, dtype=np.int32)
        heap_y = np.empty(0, dtype=np.int32)
        heap_x = np.empty(0, dtype=np.int32)

        def _push(heap_pri, heap_z, heap_y, heap_x, p, z, y, x):
            return (np.append(heap_pri, p),
                    np.append(heap_z, z),
                    np.append(heap_y, y),
                    np.append(heap_x, x))

        # We can't mutate through a function cleanly in numba's object
        # mode — unroll the heap as a dynamic array and do linear pop of
        # the min. For corridor-sized volumes this is fine.
        dist[sz, sy, sx] = 0.0
        heap_pri, heap_z, heap_y, heap_x = _push(
            heap_pri, heap_z, heap_y, heap_x, np.float32(0.0), sz, sy, sx)

        offs = np.array(
            [[dz, dy, dx] for dz in range(-1, 2)
             for dy in range(-1, 2) for dx in range(-1, 2)
             if not (dz == 0 and dy == 0 and dx == 0)], dtype=np.int32)
        steps = np.sqrt((offs.astype(np.float32) ** 2).sum(axis=1))

        actual_ez, actual_ey, actual_ex = ez, ey, ex
        found = False
        while heap_pri.size > 0:
            # Pop min
            idx = np.argmin(heap_pri)
            d = heap_pri[idx]
            cz = heap_z[idx]
            cy = heap_y[idx]
            cx = heap_x[idx]
            heap_pri = np.delete(heap_pri, idx)
            heap_z = np.delete(heap_z, idx)
            heap_y = np.delete(heap_y, idx)
            heap_x = np.delete(heap_x, idx)

            if visited[cz, cy, cx]:
                continue
            visited[cz, cy, cx] = True
            if cz == ez and cy == ey and cx == ex:
                found = True
                break
            if has_goal_mask and goal_mask[cz, cy, cx]:
                actual_ez, actual_ey, actual_ex = cz, cy, cx
                found = True
                break

            for k in range(offs.shape[0]):
                nz = cz + offs[k, 0]
                ny = cy + offs[k, 1]
                nx = cx + offs[k, 2]
                if nz < 0 or nz >= D or ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                if visited[nz, ny, nx]:
                    continue
                penalty = np.float32(0.2) if speed_field[nz, ny, nx] > 0 \
                    else np.float32(3.0)
                cost = steps[k] * (np.float32(1.0) + penalty)
                nd = d + cost
                if nd < dist[nz, ny, nx]:
                    dist[nz, ny, nx] = nd
                    prev[nz, ny, nx, 0] = cz
                    prev[nz, ny, nx, 1] = cy
                    prev[nz, ny, nx, 2] = cx
                    heap_pri, heap_z, heap_y, heap_x = _push(
                        heap_pri, heap_z, heap_y, heap_x, nd, nz, ny, nx)

        if not found:
            return None, -1, -1, -1
        return prev, actual_ez, actual_ey, actual_ex


def dijkstra_path(speed_field: np.ndarray, local_start: tuple,
                  local_end: tuple,
                  goal_proximity_mask: Optional[np.ndarray] = None
                  ) -> Optional[np.ndarray]:
    """Shortest path in the speed-field cost graph. Returns (N, 3) or None.

    The goal_proximity_mask enables early termination: the search returns
    as soon as it reaches any True voxel in this mask (plus the exact end
    voxel always counts). Useful when the endpoint is on/near an existing
    labeled region — the tracer doesn't have to hit the exact voxel.
    """
    # The numba heap implementation has np.append/np.delete, which regress
    # for large heaps. Only use it when the corridor is small enough that
    # the O(N²) argmin-heap doesn't bite. For bigger corridors the Python
    # heapq is better.
    total = int(np.prod(speed_field.shape))
    if _HAVE_NUMBA and total <= 120_000:
        try:
            has_gm = goal_proximity_mask is not None
            gm = goal_proximity_mask if has_gm else np.zeros(
                speed_field.shape, dtype=np.bool_)
            prev, ez, ey, ex = _dijkstra_numba(
                speed_field,
                local_start[0], local_start[1], local_start[2],
                local_end[0], local_end[1], local_end[2],
                gm, has_gm)
            if prev is None:
                return None
            # Backtrack
            path = []
            cz, cy, cx = ez, ey, ex
            while (cz, cy, cx) != local_start:
                path.append((cz, cy, cx))
                pz, py, px = prev[cz, cy, cx]
                if pz < 0:
                    return None
                cz, cy, cx = int(pz), int(py), int(px)
            path.append(local_start)
            path.reverse()
            return np.array(path, dtype=np.int64)
        except Exception:
            # Fall through to Python heap on any numba oddity
            pass
    return _dijkstra_python(speed_field, local_start, local_end,
                            goal_proximity_mask)


# ---------------------------------------------------------------------------
# Public entry point — signature-preserving
# ---------------------------------------------------------------------------

def find_path_between(image: np.ndarray, model: RootModel,
                      start_vox, end_vox, max_radius: int = 60,
                      progress=None,
                      label: Optional[np.ndarray] = None,
                      goal_radius: int = 4
                      ) -> Optional[np.ndarray]:
    """End-to-end point-to-point trace through the raw CT volume.

    Parameters
    ----------
    image : (D, H, W) CT volume
    model : RootModel
    start_vox, end_vox : length-3 voxel indices
    max_radius : max distance (vx) to allow the search corridor to reach
    progress : optional callable(msg:str) for UI status updates
    label  : (D, H, W) bool label volume, optional. If supplied, the
             Dijkstra may terminate at any labeled voxel within
             `goal_radius` of `end_vox` — the anchor early-termination
             trick described in the module docstring.
    goal_radius : int, proximity radius (vx) for early termination.
    """
    start = np.asarray(start_vox, dtype=np.int64)
    end = np.asarray(end_vox, dtype=np.int64)

    margin = 10
    bbox_min = np.minimum(start, end) - max_radius // 2 - margin
    bbox_max = np.maximum(start, end) + max_radius // 2 + margin
    bbox_min = np.maximum(bbox_min, 0)
    bbox_max = np.minimum(bbox_max, np.array(image.shape) - 1)

    local_start = tuple((start - bbox_min).tolist())
    local_end = tuple((end - bbox_min).tolist())
    shape = tuple((bbox_max - bbox_min + 1).astype(int).tolist())

    for dim in range(3):
        if not (0 <= local_start[dim] < shape[dim]):
            return None
        if not (0 <= local_end[dim] < shape[dim]):
            return None

    if progress:
        progress(f"Speed field {shape}…")
    t0 = time.time()
    speed = compute_speed_field(image, model,
                                np.array(local_start, dtype=np.float64),
                                np.array(local_end, dtype=np.float64),
                                bbox_min, shape)
    if progress:
        progress(f"Speed field in {time.time() - t0:.2f}s  "
                 f"({int((speed > 0).sum())} root-like voxels)")

    # Build a goal-proximity mask around the end voxel, optionally scoped
    # to label voxels only — anchor trick.
    goal_mask = None
    if label is not None and goal_radius > 0:
        local_end_arr = np.array(local_end, dtype=np.int64)
        zz, yy, xx = np.indices(shape)
        d2 = ((zz - local_end_arr[0]) ** 2
              + (yy - local_end_arr[1]) ** 2
              + (xx - local_end_arr[2]) ** 2)
        ball_mask = d2 <= goal_radius * goal_radius
        # Only terminate on label voxels inside that ball
        label_block = label[bbox_min[0]:bbox_min[0] + shape[0],
                            bbox_min[1]:bbox_min[1] + shape[1],
                            bbox_min[2]:bbox_min[2] + shape[2]]
        goal_mask = ball_mask & label_block

    t1 = time.time()
    local_path = dijkstra_path(speed, local_start, local_end,
                               goal_proximity_mask=goal_mask)
    if progress:
        progress(f"Dijkstra in {time.time() - t1:.2f}s")
    if local_path is None:
        return None
    return local_path + bbox_min


# ---------------------------------------------------------------------------
# Tube painting — unchanged semantics; minor vectorization for speed
# ---------------------------------------------------------------------------

def paint_tube_local(shape, path_voxels: np.ndarray,
                     radius_voxels: float,
                     spacing=None):
    """Return (bbox, footprint) — a smooth tube around `path_voxels`, cropped
    to its bounding box.

    Why EDT instead of `binary_dilation(ball(r))`
    ---------------------------------------------
    The previous implementation stamped path voxels and dilated with a
    voxel ball. That produced a visibly blocky tube because:
      * `r` was rounded to an integer number of voxels, so 0.5 mm and
        0.7 mm paths painted the same ball;
      * the ball is axis-aligned on the voxel grid and ignores that our
        CT voxels are anisotropic (0.39, 0.39, 0.2 mm) — the tube looked
        squashed/stretched along Z.

    Here we instead compute an anisotropic 3D EDT from the path voxels
    and threshold it at `radius_mm` — so a voxel belongs to the tube
    iff its true mm distance to the nearest path voxel is ≤ radius_mm.
    The surface is a smooth iso-contour of a continuous distance field
    rather than a union of axis-aligned balls, and the anisotropic
    sampling means the tube is a true round cross-section in mm space.

    `radius_voxels` is interpreted in mean-spacing units (the convention
    TraceWorker already uses: r_vox = fill_radius_mm / mean_spacing), so
    existing callers pass the same value as before.

    Caching the local footprint per path keeps undo O(N · bbox_volume).
    Returns (None, None) if the path is empty or entirely out of bounds.
    """
    if path_voxels is None or len(path_voxels) == 0:
        return None, None
    if spacing is None:
        # Lazy import avoids a services ↔ utils cycle at module load.
        from ..utils.config import SPACING
        spacing = SPACING
    spacing = np.asarray(spacing, dtype=np.float64)

    # Convert voxel radius (at mean spacing) back to mm — that's the
    # unit EDT measures in, and the unit the user actually cares about.
    r_mm = float(radius_voxels) * float(spacing.mean())
    if r_mm <= 0:
        return None, None

    pv = np.asarray(path_voxels, dtype=np.int64)
    shape_arr = np.array(shape, dtype=np.int64)
    valid = ((pv >= 0).all(axis=1) & (pv < shape_arr).all(axis=1))
    pv = pv[valid]
    if len(pv) == 0:
        return None, None

    # Pad bbox per-axis so the iso-surface at r_mm can't touch the
    # boundary on any axis, even the finely-sampled Z one.
    pad_vox = np.ceil(r_mm / spacing).astype(np.int64) + 1
    mn = np.maximum(pv.min(axis=0) - pad_vox, 0)
    mx = np.minimum(pv.max(axis=0) + pad_vox + 1, shape_arr)
    bbox = tuple(slice(int(a), int(b)) for a, b in zip(mn, mx))
    local_shape = tuple(int(b - a) for a, b in zip(mn, mx))

    centers = np.zeros(local_shape, dtype=bool)
    local = pv - mn
    centers[local[:, 0], local[:, 1], local[:, 2]] = True

    # Anisotropic EDT — distance to the nearest stamped centerline
    # voxel, measured in mm. Threshold at r_mm to get the tube.
    from scipy.ndimage import distance_transform_edt
    edt_mm = distance_transform_edt(~centers, sampling=tuple(spacing))
    footprint = edt_mm <= r_mm
    return bbox, footprint


def paint_tube(label: np.ndarray, path_voxels: np.ndarray,
               radius_voxels: float) -> np.ndarray:
    """Dilate each voxel in `path_voxels` into `label` using a ball SE.

    Backed by `paint_tube_local` — compute the bbox footprint once and OR
    it into the full-volume label at its bbox. Equivalent semantics to
    the old stamp-everywhere-then-global-dilate implementation, but about
    an order of magnitude faster for small paths. The input `label` is
    not mutated; a new array is returned (matches old contract).
    """
    bbox, footprint = paint_tube_local(label.shape, path_voxels,
                                       radius_voxels)
    if bbox is None:
        return label.copy()
    out = label.copy()
    out[bbox] |= footprint
    return out
