"""Noise-deletion polyline controller that draws directly in VTK.

Replacement for the previous Qt-overlay approach: a transparent QWidget
on top of a VTK OpenGL viewport gets composited as solid black on macOS
(and unpredictably on other platforms) because the GL surface is a native
layer Qt cannot blend over. So instead we draw the polyline inside VTK
with `vtkActor2D` / `vtkPolyDataMapper2D` in display (pixel) coordinates,
which is how CloudCompare's segmentation tool renders.

Input events come from VTK interactor observers:
  LeftButtonPress       → add vertex to current polyline
  RightButtonPress      → start a new polyline chained from the last vertex
  MouseMove             → update the live cursor tail
  KeyPress(Return)      → close loop → emit polylineFinished
  KeyPress(Escape)      → emit deletionCancelled
  KeyPress(BackSpace)   → undo last vertex
  KeyPress(z)           → undo last polyline
"""
from __future__ import annotations
from typing import Optional

import numpy as np

try:
    import vtk
    HAVE_VTK = True
except ImportError:   # pragma: no cover
    HAVE_VTK = False


def _make_display_mapper() -> "vtk.vtkPolyDataMapper2D":
    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToDisplay()
    m = vtk.vtkPolyDataMapper2D()
    m.SetTransformCoordinate(coord)
    return m


class DeletionController:
    """Owned transiently by a `Viewer3D` while deletion mode is active.

    Uses VTK 2D overlay actors so nothing on the Qt side needs to sit on
    top of the GL widget. The viewer wires enter()/exit() and keeps a
    reference; we dispose cleanly on exit().
    """

    # Stipple patterns (16-bit)
    DASHED = 0x00FF
    DOTTED = 0x0F0F

    def __init__(self, viewer):
        self.viewer = viewer
        # list[list[(x, y)]] — each sub-list is one polyline "stroke".
        # Consecutive strokes share their join vertex (i.e. strokes[k+1][0]
        # equals strokes[k][-1]), matching CloudCompare's segment chaining.
        self.strokes: list[list[tuple[float, float]]] = [[]]
        self.cursor: Optional[tuple[float, float]] = None
        self._observer_ids: list[int] = []

        # VTK objects created on enter()
        self._poly_data = None
        self._vert_data = None
        self._live_data = None
        self._close_data = None
        self._poly_actor = None
        self._vert_actor = None
        self._live_actor = None
        self._close_actor = None

        # Flat list of vertices after duplicate-join removal — cached here
        # after each rebuild so _finish() can emit the polygon directly.
        self._flat_verts: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def enter(self):
        plotter = self.viewer._plotter
        if plotter is None or not HAVE_VTK:
            return
        self._build_actors()
        iren = self._interactor()
        if iren is None:
            return
        # High priority so we run before any other style (even though the
        # viewer has already swapped in vtkInteractorStyleUser, belt & suspenders)
        p = 20.0
        self._observer_ids = [
            iren.AddObserver("LeftButtonPressEvent",  self._on_left,  p),
            iren.AddObserver("RightButtonPressEvent", self._on_right, p),
            iren.AddObserver("MouseMoveEvent",        self._on_move,  p),
            iren.AddObserver("KeyPressEvent",         self._on_key,   p),
        ]
        self._rebuild()
        self._emit_status()

    def exit(self):
        iren = self._interactor()
        if iren is not None:
            for oid in self._observer_ids:
                try:
                    iren.RemoveObserver(oid)
                except Exception:
                    pass
        self._observer_ids.clear()
        self._remove_actors()
        plotter = self.viewer._plotter
        if plotter is not None:
            plotter.render()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _interactor(self):
        plotter = self.viewer._plotter
        iren_wrap = getattr(plotter, "iren", None)
        if iren_wrap is not None and hasattr(iren_wrap, "interactor"):
            return iren_wrap.interactor
        return getattr(plotter, "interactor", None)

    def _build_actors(self):
        renderer = self.viewer._plotter.renderer

        # Committed polyline (solid green line through all vertices)
        self._poly_data = vtk.vtkPolyData()
        self._poly_data.SetPoints(vtk.vtkPoints())
        self._poly_data.SetLines(vtk.vtkCellArray())
        m = _make_display_mapper()
        m.SetInputData(self._poly_data)
        a = vtk.vtkActor2D()
        a.SetMapper(m)
        prop = a.GetProperty()
        prop.SetColor(0.2, 0.8, 0.3)
        prop.SetLineWidth(2.5)
        renderer.AddActor2D(a)
        self._poly_actor = a

        # Vertex dots
        self._vert_data = vtk.vtkPolyData()
        self._vert_data.SetPoints(vtk.vtkPoints())
        self._vert_data.SetVerts(vtk.vtkCellArray())
        m = _make_display_mapper()
        m.SetInputData(self._vert_data)
        a = vtk.vtkActor2D()
        a.SetMapper(m)
        prop = a.GetProperty()
        prop.SetColor(0.1, 0.5, 0.2)
        prop.SetPointSize(8.0)
        renderer.AddActor2D(a)
        self._vert_actor = a

        # Live cursor tail (dashed, light green)
        self._live_data = vtk.vtkPolyData()
        self._live_data.SetPoints(vtk.vtkPoints())
        self._live_data.SetLines(vtk.vtkCellArray())
        m = _make_display_mapper()
        m.SetInputData(self._live_data)
        a = vtk.vtkActor2D()
        a.SetMapper(m)
        prop = a.GetProperty()
        prop.SetColor(0.4, 0.85, 0.35)
        prop.SetLineWidth(1.5)
        prop.SetLineStipplePattern(self.DASHED)
        renderer.AddActor2D(a)
        self._live_actor = a

        # Closure hint (dotted yellow, last → first)
        self._close_data = vtk.vtkPolyData()
        self._close_data.SetPoints(vtk.vtkPoints())
        self._close_data.SetLines(vtk.vtkCellArray())
        m = _make_display_mapper()
        m.SetInputData(self._close_data)
        a = vtk.vtkActor2D()
        a.SetMapper(m)
        prop = a.GetProperty()
        prop.SetColor(0.95, 0.7, 0.2)
        prop.SetLineWidth(1.5)
        prop.SetLineStipplePattern(self.DOTTED)
        renderer.AddActor2D(a)
        self._close_actor = a

    def _remove_actors(self):
        renderer = self.viewer._plotter.renderer
        for a in (self._poly_actor, self._vert_actor,
                  self._live_actor, self._close_actor):
            if a is None:
                continue
            try:
                renderer.RemoveActor2D(a)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Geometry rebuild
    # ------------------------------------------------------------------

    def _flatten_strokes(self) -> list[tuple[float, float]]:
        """Join all strokes into one vertex list, deduping shared join vertices."""
        out: list[tuple[float, float]] = []
        for i, stroke in enumerate(self.strokes):
            for j, v in enumerate(stroke):
                if i > 0 and j == 0 and out:
                    px, py = out[-1]
                    if abs(v[0] - px) < 0.5 and abs(v[1] - py) < 0.5:
                        continue
                out.append(v)
        return out

    def _rebuild(self):
        """Refill all VTK polydata from self.strokes + self.cursor."""
        verts = self._flatten_strokes()
        self._flat_verts = verts

        # Committed polyline
        pts = self._poly_data.GetPoints()
        cells = self._poly_data.GetLines()
        pts.Reset()
        cells.Reset()
        for x, y in verts:
            pts.InsertNextPoint(x, y, 0.0)
        if len(verts) >= 2:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(verts))
            for i in range(len(verts)):
                line.GetPointIds().SetId(i, i)
            cells.InsertNextCell(line)
        self._poly_data.Modified()

        # Vertex dots
        vpts = self._vert_data.GetPoints()
        vcells = self._vert_data.GetVerts()
        vpts.Reset()
        vcells.Reset()
        for i, (x, y) in enumerate(verts):
            vpts.InsertNextPoint(x, y, 0.0)
            vx = vtk.vtkVertex()
            vx.GetPointIds().SetId(0, i)
            vcells.InsertNextCell(vx)
        self._vert_data.Modified()

        self._rebuild_live_and_closure()
        self.viewer._plotter.render()

    def _rebuild_live_and_closure(self):
        verts = self._flat_verts
        # Live tail (last vertex → cursor)
        lpts = self._live_data.GetPoints()
        lcells = self._live_data.GetLines()
        lpts.Reset()
        lcells.Reset()
        if verts and self.cursor is not None:
            lx, ly = verts[-1]
            cx, cy = self.cursor
            lpts.InsertNextPoint(lx, ly, 0.0)
            lpts.InsertNextPoint(cx, cy, 0.0)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, 1)
            lcells.InsertNextCell(line)
        self._live_data.Modified()

        # Closure hint (last → first) — only once there are ≥3 vertices
        cpts = self._close_data.GetPoints()
        ccells = self._close_data.GetLines()
        cpts.Reset()
        ccells.Reset()
        if len(verts) >= 3:
            fx, fy = verts[0]
            lx, ly = verts[-1]
            cpts.InsertNextPoint(lx, ly, 0.0)
            cpts.InsertNextPoint(fx, fy, 0.0)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, 1)
            ccells.InsertNextCell(line)
        self._close_data.Modified()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_left(self, iren, _evt):
        x, y = iren.GetEventPosition()
        self.strokes[-1].append((float(x), float(y)))
        self._rebuild()
        self._emit_status()

    def _on_right(self, iren, _evt):
        # Chain a new polyline whose first vertex is the current last vertex
        if not self.strokes[-1]:
            return
        last = self.strokes[-1][-1]
        self.strokes.append([last])
        self._rebuild()
        self._emit_status()

    def _on_move(self, iren, _evt):
        x, y = iren.GetEventPosition()
        self.cursor = (float(x), float(y))
        # Only the cursor/closure overlays change on mouse move — cheap path.
        self._rebuild_live_and_closure()
        self.viewer._plotter.render()

    def _on_key(self, iren, _evt):
        key = iren.GetKeySym() or ""
        if key in ("Return", "KP_Enter"):
            self._finish()
        elif key == "Escape":
            self.viewer.deletionCancelled.emit()
        elif key == "BackSpace":
            self._undo_vertex()
        elif key in ("z", "Z"):
            self._undo_stroke()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _undo_vertex(self):
        # Collapse empty strokes first
        while self.strokes and not self.strokes[-1] and len(self.strokes) > 1:
            self.strokes.pop()
        if not self.strokes:
            self.strokes = [[]]
        elif self.strokes[-1]:
            self.strokes[-1].pop()
            if not self.strokes[-1] and len(self.strokes) > 1:
                self.strokes.pop()
        self._rebuild()
        self._emit_status()

    def _undo_stroke(self):
        if len(self.strokes) > 1:
            self.strokes.pop()
        elif self.strokes and self.strokes[0]:
            self.strokes = [[]]
        self._rebuild()
        self._emit_status()

    def _finish(self):
        verts = self._flat_verts
        if len(verts) < 3:
            self.viewer.statusMessage.emit(
                "Need at least 3 vertices to close the region.")
            return
        poly = np.array(verts, dtype=np.float64)  # VTK display coords (y from bottom)
        self.viewer.polylineFinished.emit(poly)

    def _emit_status(self):
        n_v = sum(len(s) for s in self.strokes)
        n_s = len([s for s in self.strokes if s])
        self.viewer.statusMessage.emit(
            f"Deletion mode — {n_v} vertex(es), {n_s} segment(s). "
            "Left-click: add vertex  •  Right-click: new segment  •  "
            "Enter: close & preview  •  Backspace: undo vertex  •  "
            "Z: undo segment  •  Esc: cancel")
