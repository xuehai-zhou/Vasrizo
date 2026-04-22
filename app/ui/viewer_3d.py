"""Embedded 3D viewer (pyvistaqt + PyVista + VTK).

Exposes a small API so the rest of the app doesn't need to know about VTK.

Picking is implemented via a raw VTK observer on LeftButtonPressEvent that
checks the interactor's Shift key state. PyVista's `enable_point_picking`
does not natively support Shift+Click — it uses the "P" key or plain
left-click — so we register our own observer.
"""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    import vtk
    HAVE_PYVISTA = True
except ImportError:  # pragma: no cover
    HAVE_PYVISTA = False

from ..utils.config import (
    COLOR_LABEL, COLOR_CT, COLOR_TRACED, COLOR_WAYPOINT, COLOR_BG,
    POINT_SIZE_DEFAULT,
)
from ..services.screen_projector import (
    composite_world_to_ndc_matrix, project_world_to_screen,
)


# Highlight color for to-be-deleted preview points (yellow)
COLOR_DELETION_PREVIEW = (1.0, 0.95, 0.2)
COLOR_SLICE_GUIDE_PRIMARY = (0.15, 0.55, 0.95)
COLOR_SLICE_GUIDE_SECONDARY = (0.95, 0.25, 0.25)
COLOR_POT_AXIS = (0.0, 0.7, 0.95)


class Viewer3D(QWidget):
    """Thin wrapper around pyvistaqt.QtInteractor.

    Signals:
        pointPicked(numpy.ndarray, str) — fires on Shift+Left-click on a visible
            point cloud. The str is 'label' or 'ct' depending on nearest cloud.
        polylineFinished(numpy.ndarray) — fires when the user closes the
            noise-deletion polyline (Enter); the array is (M, 2) of VTK
            display pixel coords (bottom-left origin).
        deletionCancelled() — fires when the user aborts deletion (Esc).
        statusMessage(str) — status updates while drawing.
    """

    pointPicked = Signal(object, str)
    polylineFinished = Signal(object)
    deletionCancelled = Signal()
    statusMessage = Signal(str)

    def __init__(self, parent=None, point_size: float = POINT_SIZE_DEFAULT):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plotter: Optional[QtInteractor] = None
        self._actors: dict[str, object] = {}  # name -> actor
        self._cloud_raw: dict[str, Optional[np.ndarray]] = {
            "label": None,
            "ct": None,
            "paths": None,
            "waypoints": None,
        }
        self._cloud_visible: dict[str, bool] = {
            "label": True,
            "ct": True,
            "paths": True,
            "waypoints": True,
        }
        self._label_points: Optional[np.ndarray] = None
        self._ct_points: Optional[np.ndarray] = None
        self._path_points: Optional[np.ndarray] = None
        self._waypoint_points: Optional[np.ndarray] = None
        self._point_size = float(point_size)
        self._pick_enabled = False
        self._on_pick_cb: Optional[Callable] = None
        self._screen_slice_enabled = False
        self._screen_slice_origin: Optional[np.ndarray] = None
        self._screen_slice_normal: Optional[np.ndarray] = None
        self._screen_slice_reverse = False
        self._screen_slice_locked = False
        self._screen_slice_thickness_mm = 5.0
        self._screen_slice_show_guides = False
        self._screen_slice_plane_size = 100.0
        self._y_slice_enabled = False
        self._y_slice_position_mm = 0.0
        self._y_slice_reverse = False
        self._camera_changed_cb: Optional[Callable[[], None]] = None
        self._turntable_enabled = True
        self._pot_axis_start: Optional[np.ndarray] = None
        self._pot_axis_end: Optional[np.ndarray] = None

        if HAVE_PYVISTA:
            self._plotter = QtInteractor(self)
            self._plotter.set_background(COLOR_BG)
            layout.addWidget(self._plotter)
            self._setup_orientation_axes()
            self.set_turntable_interaction(True)
        else:
            from PySide6.QtWidgets import QLabel
            warn = QLabel(
                "pyvista / pyvistaqt not installed — 3D view disabled.\n"
                "pip install pyvista pyvistaqt")
            warn.setStyleSheet("color: #900; padding: 20px;")
            layout.addWidget(warn)

    # ------------------------------------------------------------------
    # Point cloud updates
    # ------------------------------------------------------------------

    def set_label_points(self, coords: np.ndarray, visible: bool = True):
        """Replace the label cloud's data. Does NOT reset the camera."""
        self._cloud_raw["label"] = coords
        self._cloud_visible["label"] = bool(visible)
        self._rebuild_label_cloud()

    def set_ct_points(self, coords: np.ndarray, visible: bool = True):
        """Replace the CT cloud's data. Does NOT reset the camera."""
        self._cloud_raw["ct"] = coords
        self._cloud_visible["ct"] = bool(visible)
        self._rebuild_ct_cloud()

    def set_label_visible(self, visible: bool):
        """Toggle label actor visibility without disturbing the camera."""
        self._cloud_visible["label"] = bool(visible)
        self._rebuild_label_cloud()

    def set_ct_visible(self, visible: bool):
        """Toggle CT actor visibility without disturbing the camera."""
        self._cloud_visible["ct"] = bool(visible)
        self._rebuild_ct_cloud()

    # ---- internal ----

    def _set_cloud_visibility(self, name: str, visible: bool,
                               coords_fallback: Optional[np.ndarray],
                               color) -> None:
        """Flip VTK's visibility flag when possible; no camera reset.

        If the actor doesn't exist yet (e.g. was never built), and we're
        being asked to show it, rebuild — but still suppress camera reset.
        """
        if self._plotter is None:
            return
        actor = self._actors.get(name)
        if actor is not None:
            try:
                actor.SetVisibility(1 if visible else 0)
            except Exception:
                pass
            self._plotter.render()
            return
        # Actor isn't built yet — only build if we need to show it
        if visible and coords_fallback is not None and len(coords_fallback) > 0:
            self._replace_cloud(name, coords_fallback, color, True)

    def _replace_cloud(self, name: str, coords: Optional[np.ndarray],
                       color, visible: bool):
        """Remove any existing actor with this name and (optionally) add a fresh
        one. Camera is NEVER reset here — the initial camera framing is the
        caller's responsibility (they should call `reset_view()` once after
        the first sample loads)."""
        if self._plotter is None:
            return
        if name in self._actors:
            try:
                # render=False keeps the intermediate state from flashing
                self._plotter.remove_actor(self._actors[name],
                                           reset_camera=False, render=False)
            except TypeError:
                # Older pyvista signatures may not accept those kwargs
                try:
                    self._plotter.remove_actor(self._actors[name])
                except Exception:
                    pass
            except Exception:
                pass
            del self._actors[name]
        if coords is None or len(coords) == 0:
            self._plotter.render()
            return
        poly = pv.PolyData(np.asarray(coords, dtype=np.float32))
        actor = self._plotter.add_mesh(
            poly, color=color, point_size=self._point_size,
            render_points_as_spheres=False,
            pickable=(name in ("label", "ct")),
            reset_camera=False,   # ← the key fix: don't re-frame on every update
        )
        if not visible:
            try:
                actor.SetVisibility(0)
            except Exception:
                pass
        self._actors[name] = actor
        self._plotter.render()

    def _filter_screen_slice(self, coords: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if coords is None:
            return None
        if len(coords) == 0:
            return coords
        if not self._screen_slice_enabled:
            return self._filter_y_slice(coords)
        if self._screen_slice_origin is None or self._screen_slice_normal is None:
            return self._filter_y_slice(coords)
        delta = np.asarray(coords, dtype=np.float64) - self._screen_slice_origin[None, :]
        signed = delta @ self._screen_slice_normal
        eps = 1e-6
        if self._screen_slice_reverse:
            keep = signed <= eps
            if self._screen_slice_locked:
                keep &= signed >= -(float(self._screen_slice_thickness_mm) + eps)
        else:
            keep = signed >= -eps
            if self._screen_slice_locked:
                keep &= signed <= float(self._screen_slice_thickness_mm) + eps
        filtered = coords[keep]
        return self._filter_y_slice(filtered)

    def _filter_y_slice(self, coords: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if coords is None:
            return None
        if len(coords) == 0:
            return coords
        if not self._y_slice_enabled:
            return coords
        y = np.asarray(coords, dtype=np.float64)[:, 1]
        eps = 1e-6
        if self._y_slice_reverse:
            keep = y <= (self._y_slice_position_mm + eps)
        else:
            keep = y >= (self._y_slice_position_mm - eps)
        return coords[keep]

    def _rebuild_label_cloud(self):
        raw = self._cloud_raw["label"]
        filtered = self._filter_screen_slice(raw)
        self._label_points = filtered
        self._replace_cloud("label", filtered, COLOR_LABEL, self._cloud_visible["label"])

    def _rebuild_ct_cloud(self):
        raw = self._cloud_raw["ct"]
        filtered = self._filter_screen_slice(raw)
        self._ct_points = filtered
        self._replace_cloud("ct", filtered, COLOR_CT, self._cloud_visible["ct"])

    def _rebuild_path_cloud(self):
        raw = self._cloud_raw["paths"]
        filtered = self._filter_screen_slice(raw)
        self._path_points = filtered
        self._clear_actor("paths")
        if filtered is None or len(filtered) == 0 or self._plotter is None:
            if self._plotter is not None:
                self._plotter.render()
            return
        poly = pv.PolyData(filtered.astype(np.float32))
        actor = self._plotter.add_mesh(poly, color=COLOR_TRACED,
                                       point_size=self._point_size + 1,
                                       render_points_as_spheres=True,
                                       reset_camera=False)
        try:
            actor.SetPickable(False)
        except Exception:
            pass
        self._actors["paths"] = actor
        self._plotter.render()

    def _rebuild_waypoints(self):
        raw = self._cloud_raw["waypoints"]
        filtered = self._filter_screen_slice(raw)
        self._waypoint_points = filtered
        self._clear_actor("waypoints")
        if filtered is None or len(filtered) == 0 or self._plotter is None:
            return
        coords = np.asarray(filtered, dtype=np.float32)
        merged = pv.MultiBlock()
        for p in coords:
            merged.append(pv.Sphere(radius=1.0, center=p))
        combined = merged.combine()
        actor = self._plotter.add_mesh(combined, color=COLOR_WAYPOINT,
                                       specular=0.3, reset_camera=False)
        try:
            actor.SetPickable(False)
        except Exception:
            pass
        self._actors["waypoints"] = actor
        self._plotter.render()

    def _rebuild_all_slice_filtered_clouds(self):
        self._rebuild_label_cloud()
        self._rebuild_ct_cloud()
        self._rebuild_path_cloud()
        self._rebuild_waypoints()
        self._update_screen_slice_guides()

    def set_pot_axis(self,
                     start_mm: Optional[np.ndarray],
                     end_mm: Optional[np.ndarray]):
        self._pot_axis_start = (
            np.asarray(start_mm, dtype=np.float64).copy()
            if start_mm is not None else None
        )
        self._pot_axis_end = (
            np.asarray(end_mm, dtype=np.float64).copy()
            if end_mm is not None else None
        )
        self._rebuild_pot_axis()

    def _rebuild_pot_axis(self):
        self._clear_actor("pot_axis")
        if self._plotter is None:
            return
        if self._pot_axis_start is None or self._pot_axis_end is None:
            self._plotter.render()
            return
        if float(np.linalg.norm(self._pot_axis_end - self._pot_axis_start)) <= 1e-6:
            self._plotter.render()
            return
        line = pv.Line(
            self._pot_axis_start.astype(np.float32),
            self._pot_axis_end.astype(np.float32),
            resolution=1,
        )
        actor = self._plotter.add_mesh(
            line,
            color=COLOR_POT_AXIS,
            line_width=5.0,
            render_lines_as_tubes=True,
            reset_camera=False,
        )
        try:
            actor.SetPickable(False)
        except Exception:
            pass
        self._actors["pot_axis"] = actor
        self._plotter.render()

    # ------------------------------------------------------------------
    # Waypoints + traced paths
    # ------------------------------------------------------------------

    def update_waypoints(self, phys_positions: list):
        self._cloud_raw["waypoints"] = (
            np.asarray(phys_positions, dtype=np.float32)
            if phys_positions else np.empty((0, 3), dtype=np.float32)
        )
        self._rebuild_waypoints()

    def update_traced_paths(self, all_path_voxels_phys: list):
        """all_path_voxels_phys: list of (N, 3) float physical coords."""
        self._cloud_raw["paths"] = (
            np.concatenate(all_path_voxels_phys, axis=0).astype(np.float32)
            if all_path_voxels_phys else np.empty((0, 3), dtype=np.float32)
        )
        self._rebuild_path_cloud()

    def _clear_actor(self, name: str):
        if self._plotter is None or name not in self._actors:
            return
        try:
            self._plotter.remove_actor(self._actors[name],
                                       reset_camera=False, render=False)
        except TypeError:
            try:
                self._plotter.remove_actor(self._actors[name])
            except Exception:
                pass
        except Exception:
            pass
        del self._actors[name]

    # ------------------------------------------------------------------
    # Appearance
    # ------------------------------------------------------------------

    def set_point_size(self, size: float):
        """Apply a new point size to all existing point-cloud actors."""
        self._point_size = float(size)
        if self._plotter is None:
            return
        for name, actor in self._actors.items():
            try:
                prop = actor.GetProperty()
                # Traced paths were drawn a bit thicker for visibility
                ps = self._point_size + 1 if name == "paths" else self._point_size
                prop.SetPointSize(ps)
            except Exception:
                pass
        self._plotter.render()

    def reset_view(self):
        if self._plotter is not None:
            self._plotter.reset_camera()
            self._plotter.render()
        self._notify_camera_changed()

    def set_screen_slice(self, enabled: bool,
                         plane_origin: Optional[np.ndarray],
                         plane_normal: Optional[np.ndarray],
                         reverse: bool = False,
                         locked: bool = False,
                         thickness_mm: float = 5.0,
                         show_guides: bool = False,
                         plane_size: float = 100.0):
        self._screen_slice_enabled = bool(enabled)
        self._screen_slice_origin = (
            np.asarray(plane_origin, dtype=np.float64).copy()
            if plane_origin is not None else None
        )
        if plane_normal is None:
            self._screen_slice_normal = None
        else:
            n = np.asarray(plane_normal, dtype=np.float64)
            norm = float(np.linalg.norm(n))
            self._screen_slice_normal = n / norm if norm > 1e-8 else None
        self._screen_slice_reverse = bool(reverse)
        self._screen_slice_locked = bool(locked)
        self._screen_slice_thickness_mm = max(0.1, float(thickness_mm))
        self._screen_slice_show_guides = bool(show_guides)
        self._screen_slice_plane_size = max(1.0, float(plane_size))
        self._rebuild_all_slice_filtered_clouds()

    def set_y_slice(self, enabled: bool, position_mm: float, reverse: bool = False):
        self._y_slice_enabled = bool(enabled)
        self._y_slice_position_mm = float(position_mm)
        self._y_slice_reverse = bool(reverse)
        self._rebuild_all_slice_filtered_clouds()

    def current_camera_pose(self):
        if self._plotter is None:
            return None, None
        renderer = self._plotter.renderer
        if renderer is None:
            return None, None
        camera = renderer.GetActiveCamera()
        pos = np.asarray(camera.GetPosition(), dtype=np.float64)
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        return pos, focal

    def _setup_orientation_axes(self):
        if self._plotter is None:
            return
        try:
            axes = vtk.vtkAxesActor()
            axes.SetXAxisLabelText("X")
            axes.SetYAxisLabelText("Y")
            axes.SetZAxisLabelText("Z")
            axes.SetTotalLength(1.0, 1.0, 1.0)
            widget = vtk.vtkOrientationMarkerWidget()
            widget.SetOrientationMarker(axes)
            iren_wrapper = getattr(self._plotter, "iren", None)
            if iren_wrapper is not None and hasattr(iren_wrapper, "interactor"):
                widget.SetInteractor(iren_wrapper.interactor)
            else:
                widget.SetInteractor(self._plotter.interactor)
            widget.SetViewport(0.0, 0.0, 0.18, 0.18)
            widget.SetEnabled(1)
            widget.InteractiveOff()
            self._axes_widget = widget
            self._axes_actor = axes
        except Exception:
            self._axes_widget = None
            self._axes_actor = None

    def set_turntable_interaction(self, enabled: bool):
        self._turntable_enabled = bool(enabled)
        if self._plotter is None:
            return
        iren_wrapper = getattr(self._plotter, "iren", None)
        if iren_wrapper is None:
            return
        try:
            iren = iren_wrapper.interactor
        except AttributeError:
            iren = self._plotter.interactor
        # Terrain style is what caused the repeated VTK "Resetting view-up"
        # warnings near top/bottom views. We keep the interaction intuitive
        # by using trackball in both modes and separately sanitizing view-up.
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)

    def set_camera_to_axis(self, axis_name: str):
        if self._plotter is None:
            return
        renderer = self._plotter.renderer
        if renderer is None:
            return
        camera = renderer.GetActiveCamera()
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        pos = np.asarray(camera.GetPosition(), dtype=np.float64)
        distance = float(np.linalg.norm(pos - focal))
        if distance <= 1e-6:
            distance = 200.0

        axis_map = {
            "+X": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "-X": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "+Y": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "-Y": (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "+Z": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
            "-Z": (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
        }
        if axis_name == "ISO":
            direction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            direction /= np.linalg.norm(direction)
            view_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            direction, view_up = axis_map.get(
                axis_name, (np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0])))
        new_pos = focal + distance * direction
        camera.SetPosition(*new_pos.tolist())
        camera.SetFocalPoint(*focal.tolist())
        camera.SetViewUp(*view_up.tolist())
        self._sanitize_camera_view_up(camera)
        renderer.ResetCameraClippingRange()
        self._plotter.render()
        self._notify_camera_changed()

    def align_camera_to_plane(self, center: np.ndarray, normal: np.ndarray,
                              up_hint: Optional[np.ndarray] = None):
        if self._plotter is None:
            return
        renderer = self._plotter.renderer
        if renderer is None:
            return
        camera = renderer.GetActiveCamera()
        center = np.asarray(center, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        n_norm = float(np.linalg.norm(normal))
        if n_norm <= 1e-8:
            return
        normal = normal / n_norm

        cur_pos = np.asarray(camera.GetPosition(), dtype=np.float64)
        cur_focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        distance = float(np.linalg.norm(cur_pos - cur_focal))
        if distance <= 1e-6:
            distance = 200.0

        if up_hint is None:
            up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            up_hint = np.asarray(up_hint, dtype=np.float64)
        up_hint = up_hint - normal * float(np.dot(up_hint, normal))
        if float(np.linalg.norm(up_hint)) <= 1e-8:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            up_hint = up_hint - normal * float(np.dot(up_hint, normal))
        up_hint = up_hint / max(float(np.linalg.norm(up_hint)), 1e-8)

        new_pos = center - distance * normal
        camera.SetFocalPoint(*center.tolist())
        camera.SetPosition(*new_pos.tolist())
        camera.SetViewUp(*up_hint.tolist())
        self._sanitize_camera_view_up(camera)
        renderer.ResetCameraClippingRange()
        self._plotter.render()
        self._notify_camera_changed()

    def set_camera_changed_callback(self, callback: Optional[Callable[[], None]]):
        self._camera_changed_cb = callback
        if self._plotter is None:
            return
        if getattr(self, "_camera_observer_installed", False):
            return

        def _on_camera_event(*_args):
            self._notify_camera_changed()

        iren_wrapper = getattr(self._plotter, "iren", None)
        if iren_wrapper is not None and hasattr(iren_wrapper, "add_observer"):
            iren_wrapper.add_observer("EndInteractionEvent", _on_camera_event)
        else:
            vtk_iren = self._plotter.interactor
            vtk_iren.AddObserver("EndInteractionEvent", _on_camera_event)
        self._camera_observer_installed = True

    def _notify_camera_changed(self):
        if self._plotter is not None:
            renderer = self._plotter.renderer
            if renderer is not None:
                self._sanitize_camera_view_up(renderer.GetActiveCamera())
        if self._camera_changed_cb is None:
            return
        try:
            self._camera_changed_cb()
        except Exception:
            pass

    def _sanitize_camera_view_up(self, camera):
        if camera is None:
            return
        pos = np.asarray(camera.GetPosition(), dtype=np.float64)
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        up = np.asarray(camera.GetViewUp(), dtype=np.float64)
        view = focal - pos
        view_norm = float(np.linalg.norm(view))
        up_norm = float(np.linalg.norm(up))
        if view_norm <= 1e-8 or up_norm <= 1e-8:
            return
        view /= view_norm
        up /= up_norm
        parallel = abs(float(np.dot(view, up)))
        if parallel < 0.98:
            return
        for cand in (
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        ):
            cand = cand - view * float(np.dot(cand, view))
            n = float(np.linalg.norm(cand))
            if n > 1e-6:
                camera.SetViewUp(*(cand / n).tolist())
                return

    def _update_screen_slice_guides(self):
        self._clear_actor("slice_plane_primary")
        self._clear_actor("slice_plane_secondary")
        if self._plotter is None:
            return
        if not self._screen_slice_enabled or not self._screen_slice_show_guides:
            self._plotter.render()
            return
        if self._screen_slice_origin is None or self._screen_slice_normal is None:
            self._plotter.render()
            return

        primary = self._make_plane_actor(
            "slice_plane_primary",
            center=self._screen_slice_origin,
            normal=self._screen_slice_normal,
            color=COLOR_SLICE_GUIDE_PRIMARY,
        )
        if primary is not None:
            self._actors["slice_plane_primary"] = primary

        if self._screen_slice_locked:
            direction = -1.0 if self._screen_slice_reverse else 1.0
            other_center = (
                self._screen_slice_origin +
                direction * self._screen_slice_thickness_mm * self._screen_slice_normal
            )
            secondary = self._make_plane_actor(
                "slice_plane_secondary",
                center=other_center,
                normal=self._screen_slice_normal,
                color=COLOR_SLICE_GUIDE_SECONDARY,
            )
            if secondary is not None:
                self._actors["slice_plane_secondary"] = secondary
        self._plotter.render()

    def _make_plane_actor(self, name: str, center: np.ndarray,
                          normal: np.ndarray, color):
        if self._plotter is None:
            return None
        plane = pv.Plane(
            center=np.asarray(center, dtype=np.float32),
            direction=np.asarray(normal, dtype=np.float32),
            i_size=float(self._screen_slice_plane_size),
            j_size=float(self._screen_slice_plane_size),
            i_resolution=1,
            j_resolution=1,
        )
        actor = self._plotter.add_mesh(
            plane,
            color=color,
            style="wireframe",
            line_width=2.0,
            opacity=0.9,
            reset_camera=False,
            pickable=False,
        )
        try:
            actor.SetPickable(False)
        except Exception:
            pass
        return actor

    # ------------------------------------------------------------------
    # Picking — Shift + Left-click
    # ------------------------------------------------------------------

    def enable_picking(self, on_pick: Callable[[np.ndarray, str], None]):
        """Register a Shift+Left-click pick handler.

        Uses the same algorithm as Open3D's VisualizerWithEditing:
        **screen-space nearest-neighbor picking**. We project every visible
        point to screen space via the VTK camera matrix, then pick the one
        closest to the click in pixels. This matches what the user sees —
        no ray tolerance, no front-point tie-breaking.

        Label points get a small priority zone: if the nearest label is
        within LABEL_PRIORITY_RADIUS_PX of the click, we pick it even if
        a CT noise point is slightly closer. Outside that zone, whichever
        cloud is nearest wins.
        """
        if self._plotter is None or self._pick_enabled:
            return
        self._pick_enabled = True
        self._on_pick_cb = on_pick

        def _handler(caller, _event):
            if not caller.GetShiftKey():
                return  # plain click → let rotation proceed
            x, y = caller.GetEventPosition()
            renderer = self._plotter.renderer
            if renderer is None:
                return
            pt, source = self._screen_nearest_pick(x, y)
            if pt is None:
                return
            self.pointPicked.emit(pt, source)
            if self._on_pick_cb is not None:
                try:
                    self._on_pick_cb(pt, source)
                except Exception:
                    pass

        # pyvistaqt's iren wrapper exposes add_observer; fall back to the
        # raw vtk interactor if the wrapper isn't there in some version.
        iren_wrapper = getattr(self._plotter, "iren", None)
        if iren_wrapper is not None and hasattr(iren_wrapper, "add_observer"):
            iren_wrapper.add_observer("LeftButtonPressEvent", _handler)
        else:
            # Direct VTK fallback
            vtk_iren = self._plotter.interactor
            vtk_iren.AddObserver("LeftButtonPressEvent", _handler)

    # ---- screen-space picking constants ----
    # Same rationale as Open3D's VisualizerWithEditing: prefer the nearest
    # on-screen point. Two knobs tune the UX:
    #   LABEL_PRIORITY_RADIUS_PX — if the nearest label point is within this
    #       many pixels of the click, snap to label regardless of whether
    #       CT noise sits closer. Keeps "click on a visible labeled voxel"
    #       from getting hijacked by CT points stacked on top of it.
    #   MAX_PICK_RADIUS_PX       — outside this radius the click is ignored
    #       (treated as "clicked on empty space").
    LABEL_PRIORITY_RADIUS_PX = 12
    MAX_PICK_RADIUS_PX = 40

    def _screen_nearest_pick(self, click_x: int, click_y: int):
        """Return (physical_pos, source) of the visible point whose on-screen
        projection is closest to the click. Mirrors Open3D picker semantics.
        """
        renderer = self._plotter.renderer if self._plotter else None
        if renderer is None:
            return None, None

        candidates = self._visible_cloud_bundles()
        if not candidates:
            return None, None

        render_window = renderer.GetRenderWindow()
        if render_window is None:
            return None, None
        size = render_window.GetSize()
        width, height = float(size[0]), float(size[1])
        if width <= 0 or height <= 0:
            return None, None

        # World → NDC transform as a numpy 4×4
        camera = renderer.GetActiveCamera()
        aspect = renderer.GetTiledAspectRatio()
        vtk_mat = camera.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
        M = np.array([[vtk_mat.GetElement(i, j) for j in range(4)]
                      for i in range(4)], dtype=np.float64)

        best_label = (float('inf'), None)   # (dist², phys_pos)
        best_ct = (float('inf'), None)

        for coords, name in candidates:
            pts, dist_sq = self._project_and_nearest(
                coords, M, width, height, click_x, click_y)
            if pts is None:
                continue
            if name == "label" and dist_sq < best_label[0]:
                best_label = (dist_sq, pts)
            elif name == "ct" and dist_sq < best_ct[0]:
                best_ct = (dist_sq, pts)

        prio_sq = self.LABEL_PRIORITY_RADIUS_PX ** 2
        max_sq = self.MAX_PICK_RADIUS_PX ** 2

        # Rule 1: snap to label if it's inside the priority ring
        if best_label[1] is not None and best_label[0] <= prio_sq:
            return best_label[1], "label"

        # Rule 2: otherwise, nearest overall wins (within MAX_PICK_RADIUS_PX)
        ld, lp = best_label
        cd, cp = best_ct
        if lp is None and cp is None:
            return None, None
        if lp is None:
            return (cp, "ct") if cd <= max_sq else (None, None)
        if cp is None:
            return (lp, "label") if ld <= max_sq else (None, None)
        if ld <= cd:
            return (lp, "label") if ld <= max_sq else (None, None)
        return (cp, "ct") if cd <= max_sq else (None, None)

    def _visible_cloud_bundles(self) -> list:
        """List of (coords_ndarray, name) for clouds whose actor is visible."""
        out = []
        for name, coords in (("label", self._label_points),
                             ("ct", self._ct_points)):
            if coords is None or len(coords) == 0:
                continue
            actor = self._actors.get(name)
            if actor is None:
                continue
            try:
                if not actor.GetVisibility():
                    continue
            except Exception:
                pass
            out.append((coords, name))
        return out

    @staticmethod
    def _project_and_nearest(coords: np.ndarray, M: np.ndarray,
                             width: float, height: float,
                             click_x: float, click_y: float):
        """Project `coords` (N,3) to screen space using world-to-NDC matrix M,
        then return (nearest_phys_pos, dist_sq_pixels) for the closest
        on-screen point. Returns (None, inf) if no point is in view.
        """
        n = coords.shape[0]
        hc = np.empty((n, 4), dtype=np.float64)
        hc[:, :3] = coords
        hc[:, 3] = 1.0
        proj = hc @ M.T
        w = proj[:, 3]
        valid = w > 1e-8
        if not valid.any():
            return None, float('inf')
        vp = proj[valid]
        vw = w[valid]
        ndc = vp[:, :3] / vw[:, None]
        # Slightly generous frustum test so edge points still participate
        in_view = (ndc[:, 0] >= -1.05) & (ndc[:, 0] <= 1.05) & \
                  (ndc[:, 1] >= -1.05) & (ndc[:, 1] <= 1.05) & \
                  (ndc[:, 2] >= -1.05) & (ndc[:, 2] <= 1.05)
        if not in_view.any():
            return None, float('inf')
        v_ndc = ndc[in_view]
        valid_idx = np.where(valid)[0][in_view]
        # VTK event coords are bottom-left origin; same as our NDC→screen.
        sx = (v_ndc[:, 0] + 1.0) * 0.5 * width
        sy = (v_ndc[:, 1] + 1.0) * 0.5 * height
        dx = sx - click_x
        dy = sy - click_y
        d_sq = dx * dx + dy * dy
        k = int(np.argmin(d_sq))
        return coords[valid_idx[k]].astype(np.float64).copy(), float(d_sq[k])

    # ------------------------------------------------------------------
    # Deletion mode — camera lock + preview highlight
    # ------------------------------------------------------------------

    def viewport_size(self) -> tuple[int, int]:
        """Return (width, height) in pixels of the render window, or (0,0)."""
        if self._plotter is None:
            return 0, 0
        rw = self._plotter.renderer.GetRenderWindow()
        if rw is None:
            return 0, 0
        w, h = rw.GetSize()
        return int(w), int(h)

    def current_projection_matrix(self) -> Optional[np.ndarray]:
        """World→NDC 4x4 for the current camera; None if not ready."""
        if self._plotter is None:
            return None
        renderer = self._plotter.renderer
        if renderer is None:
            return None
        camera = renderer.GetActiveCamera()
        return composite_world_to_ndc_matrix(
            camera, renderer.GetTiledAspectRatio())

    def set_camera_interactive(self, interactive: bool):
        """Lock (False) or unlock (True) camera manipulation via mouse.

        Implementation: swap the VTK interactor style. When locked, we
        install a no-op style that ignores all interaction events so the
        user can't rotate/pan/zoom. When unlocked, restore the trackball
        camera style that pyvista uses by default.
        """
        if self._plotter is None:
            return
        iren_wrap = getattr(self._plotter, "iren", None)
        if iren_wrap is None:
            return
        try:
            iren = iren_wrap.interactor
        except AttributeError:
            iren = self._plotter.interactor
        if interactive:
            if getattr(self, "_saved_style", None) is not None:
                iren.SetInteractorStyle(self._saved_style)
                self._saved_style = None
        else:
            # Save current style, replace with a locked one
            self._saved_style = iren.GetInteractorStyle()
            try:
                from vtkmodules.vtkInteractionStyle import (
                    vtkInteractorStyleUser,
                )
                locked = vtkInteractorStyleUser()
            except ImportError:
                locked = vtk.vtkInteractorStyleUser()
            iren.SetInteractorStyle(locked)

    def project_world_points_to_screen(self, coords: np.ndarray):
        """Project coords to (sx, sy, original_indices) using the *current* camera.

        Wrapper over screen_projector.project_world_to_screen that supplies
        the camera matrix and viewport size. Returns empty arrays if the
        viewer isn't ready.
        """
        if self._plotter is None or coords is None or len(coords) == 0:
            empty = np.empty(0)
            return empty, empty, np.empty(0, dtype=np.int64)
        M = self.current_projection_matrix()
        if M is None:
            empty = np.empty(0)
            return empty, empty, np.empty(0, dtype=np.int64)
        w, h = self.viewport_size()
        if w <= 0 or h <= 0:
            empty = np.empty(0)
            return empty, empty, np.empty(0, dtype=np.int64)
        return project_world_to_screen(coords, M, float(w), float(h))

    def show_deletion_preview(self, world_coords: np.ndarray):
        """Highlight these world-space points as 'about to be deleted' (yellow)."""
        self._clear_actor("del_preview")
        if world_coords is None or len(world_coords) == 0 or self._plotter is None:
            self._plotter.render() if self._plotter else None
            return
        poly = pv.PolyData(world_coords.astype(np.float32))
        actor = self._plotter.add_mesh(
            poly,
            color=COLOR_DELETION_PREVIEW,
            point_size=max(self._point_size + 2, 5),
            render_points_as_spheres=True,
            pickable=False,
            reset_camera=False,
        )
        try:
            actor.SetPickable(False)
        except Exception:
            pass
        self._actors["del_preview"] = actor
        self._plotter.render()

    def clear_deletion_preview(self):
        self._clear_actor("del_preview")
        if self._plotter is not None:
            self._plotter.render()

    # ---- deletion-mode entry/exit ------------------------------------

    def enter_deletion_mode(self):
        """Activate polyline-drawing mode: lock camera, install VTK overlay."""
        if self._plotter is None:
            return
        if getattr(self, "_deletion_ctrl", None) is not None:
            return
        self.set_camera_interactive(False)
        # Lazily import so this file still imports when VTK isn't available
        from .deletion_controller import DeletionController
        self._deletion_ctrl = DeletionController(self)
        self._deletion_ctrl.enter()
        # VTK needs the interactor to have focus to receive key events
        try:
            self._plotter.setFocus()
        except Exception:
            pass

    def exit_deletion_mode(self):
        """Tear down polyline-drawing mode and re-enable camera."""
        ctrl = getattr(self, "_deletion_ctrl", None)
        if ctrl is not None:
            ctrl.exit()
            self._deletion_ctrl = None
        self.set_camera_interactive(True)
