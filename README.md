# Vasrizo

Desktop annotation tool for refining broken segmentation labels on
volumetric image data. Originated from a root-CT gap-completion task,
but the input contract is deliberately generic — equally applicable to
vessel tracing, neuron reconstruction, or any partial-label repair on a
3D image volume.

Browser-style tabs let you work on several samples at once. The core
tracer is a learned-intensity speed function driving Dijkstra path
finding through the image, accelerated with a vectorized speed-field
block evaluator and optional numba JIT.

## Features

- **Gap tracing** between user-picked waypoints (Shift+Click) with a
  Rootrak-style speed function learned from the partial label.
- **Noise-point deletion** — CloudCompare-style screen-space polyline
  lasso with Undo.
- **Live pot / container wall peel** — anisotropic per-slice EDT with
  rim-aware top preservation, replaces a slow offline preprocessing
  step (5+ min → ~3 s).
- **HU threshold overlay** with dual-handle slider, live update.
- **Smooth tubular painting** of traced paths via anisotropic EDT
  (true mm radius, no axis-aligned voxel blockiness).
- **Atomic NIfTI save** (uint8, compresslevel 1) and parallel
  decompression on load.

## Input contract

Per sample:

```
{images_dir}/{name}.nii.gz             OR  {images_dir}/{name}_0000.nii.gz
{labels_dir}/{name}.nii.gz
```

Refined labels are written to `{output_dir}/{name}.nii.gz` as uint8
binary masks preserving the input affine and header.

The image volume can be raw or preprocessed. If raw CT contains a pot
or container wall, use the interactive **Apply pot-wall peel** control
to remove it in ~3 s.

## Install

Python 3.10+ is recommended. We suggest a dedicated virtualenv or conda
environment:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`numba` is optional; installing it will JIT-compile the Dijkstra inner
loop for an additional tracer speedup.

## Run

```bash
python main.py
```

Then use **File → Open Dataset Folder…** (Ctrl+O), or drag a folder
onto the window. The app auto-detects common `images/` and `labels/`
subdirectory conventions.

For scripted launches:

```bash
python main.py --folder path/to/dataset
python main.py --images_dir path/to/images --labels_dir path/to/labels \
               [--output_dir path/to/output] [--sample NAME]
```

## Key controls

| Action | Shortcut |
|---|---|
| Open Dataset Folder | Ctrl+O |
| Configure Dataset (manual) | Ctrl+Shift+O |
| Open Next Pending Sample | Ctrl+N |
| Save Current Tab | Ctrl+S |
| Close Tab | Ctrl+W |
| Reset 3D View | R |
| Toggle Label Visibility | L |
| Toggle CT Overlay Visibility | T |
| Add Waypoint | Shift + Left-Click |
| Trace between waypoints | **Trace** button |
| Delete noise points | **Delete noise points…** button → draw polyline |
| Apply pot-wall peel | **Apply pot-wall peel** button |

## Layout

```
Vasrizo/
  main.py                       Entry point
  bench_tracer.py               Tracer benchmark (requires --image --label)
  app/
    main_window.py              Top-level QMainWindow with tab bar
    annotation_tab.py           One tab = one sample, owns its DocumentState
    models/
      document_state.py         Per-tab state (label, waypoints, paths, …)
      app_settings.py           Dataset directories
    io/
      data_loader.py            Parallel NIfTI load (image + label)
      data_saver.py             Atomic uint8 NIfTI save
    services/                   Pure algorithms (no Qt)
      root_model.py             Rootrak-style intensity-histogram model
      tracing_service.py        Speed field + Dijkstra + tube painting
      threshold_service.py      HU threshold → point cloud
      pot_wall_service.py       Per-slice 2D EDT pot-wall peel
      deletion_service.py       Screen-space polyline → voxel mask
      screen_projector.py       World ↔ screen projection
    ui/
      viewer_3d.py              pyvistaqt-backed 3D view
      controls_panel.py         Action buttons & tuning knobs
      threshold_range_slider.py Dual-handle HU slider
      waypoint_panel.py         Waypoint list
      sample_browser.py         Left dock: sample list
      dataset_dialog.py         Manual dataset configuration
      deletion_controller.py    VTK 2D overlay for deletion lasso
      status_bar.py
    workers/                    QThread background workers
      load_worker.py            NIfTI load + RootModel fit
      trace_worker.py           Waypoint → path
      threshold_worker.py       Threshold volume → coords
      save_worker.py            NIfTI save
      pot_wall_worker.py        Pot-wall peel
    utils/
      config.py                 Defaults and color palette
      geometry_utils.py         Voxel ↔ physical conversion
      layout_detect.py          Auto-detect images/ and labels/ subdirs
```

## Algorithmic notes

* **Speed field.** For each voxel, speed = `exp(-β · JS(p_local, p_model))`,
  where `p_model` is a learned intensity histogram of label voxels and
  `p_local` is a histogram of a small window around the voxel. The
  tracer replaces the per-voxel Python loop with a vectorized numpy
  pass that scores an entire corridor via a precomputed intensity LUT.
* **Dijkstra with anchor termination.** If an end waypoint is near an
  existing label voxel, the search stops as soon as any goal-proximal
  voxel is popped — typically 10–50× faster than running to convergence.
* **Tube painting.** Traced paths are painted via anisotropic EDT
  thresholded at `radius_mm`, not a voxel-ball dilation — surfaces are
  smooth iso-contours rather than unions of axis-aligned balls.
* **Pot-wall peel.** Per-slice 2D EDT gives the mm distance from each
  voxel to its same-slice air boundary; threshold at `peel_xy_mm` to
  peel the wall. A rim-aware second pass preserves the plant shoot /
  root crown above the pot rim by detecting the topmost slice whose
  max EDT still exceeds the peel radius, and leaving everything above
  it untouched.

## License

MIT — see [LICENSE](LICENSE).
