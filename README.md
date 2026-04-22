# Vasrizo

Desktop annotation tool for refining broken segmentation labels on
volumetric image data. Vasrizo started from a root-CT gap-completion
workflow and has since evolved into a broader 3D label-repair tool with
interactive slicing, pot-wall removal, root-oriented slab fitting, and
camera aids for difficult root systems.

This repository state corresponds to the `v2.0.0` update based on the
local `V15` workflow.

## Highlights in v2.0.0

- Screen-aligned slicing with front/back filtering
- Reverse slicing, locked slab thickness, and optional guide planes
- Y-axis slicing with an independent slider
- Pot wall peel based on a fitted cylindrical pot model
- Automatic pot center-axis estimation and on-screen axis display
- Root-oriented auto-fit slab aligned to the pot axis and selected root trend
- Camera axis buttons and orientation aids for understanding XYZ directions
- Improved interactive workflow for noisy root / pot volumes

## Input contract

Per sample:

```text
{images_dir}/{name}.nii.gz             OR  {images_dir}/{name}_0000.nii.gz
{labels_dir}/{name}.nii.gz
{images_dir}/{name}_interior.nii.gz    OPTIONAL, auto-detected if present
```

The image volume can be raw or preprocessed. If raw CT contains a pot or
container wall, use the interactive `Apply pot-wall peel` control after load.

## Install

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

On macOS we recommend a dedicated conda or virtualenv environment. The app
depends on:

- `PySide6`
- `numpy`
- `scipy`
- `nibabel`
- `scikit-image`
- `pyvista`
- `pyvistaqt`

## Run

```bash
python main.py
```

Then use `File -> Open Dataset Folder...` or drag a dataset folder onto the
window.

## Main features

- Gap tracing between user-picked waypoints
- Brown CT overlay from HU thresholding
- Pot wall peel for cylindrical pot datasets
- Screen-normal slicing with thickness lock
- Y-slice filtering parallel to the XZ plane
- Auto-fit root slab for selected roots
- Camera snap buttons and orientation axes
- Save refined labels back to NIfTI

## Layout

```text
Vasrizo/
  main.py
  bench_tracer.py
  app/
    main_window.py
    annotation_tab.py
    models/
      document_state.py
      app_settings.py
    io/
      data_loader.py
      data_saver.py
    services/
      root_model.py
      tracing_service.py
      threshold_service.py
      pot_wall_service.py
      root_plane_service.py
      deletion_service.py
      screen_projector.py
    ui/
      viewer_3d.py
      controls_panel.py
      threshold_range_slider.py
      screen_slice_slider.py
      axis_slice_slider.py
      waypoint_panel.py
      sample_browser.py
      dataset_dialog.py
      deletion_controller.py
      status_bar.py
    workers/
      load_worker.py
      trace_worker.py
      threshold_worker.py
      save_worker.py
      pot_wall_worker.py
```

## Notes

- `v2.0.0` is focused on interactive root CT cleanup and inspection.
- The repository should contain only the current app code, not nested
  historical `V12/V13/...` folders.
- For very large CT volumes, increasing display downsampling can make the
  viewer much smoother.

## License

MIT - see [LICENSE](LICENSE).
