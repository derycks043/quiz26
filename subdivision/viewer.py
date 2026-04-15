from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .loop import subdivide
from .mesh import Mesh


class MeshViewer:
    """Interactive matplotlib 3D viewer for control + Loop-subdivided mesh."""

    def __init__(self, control_mesh: Mesh):
        self.control_mesh = control_mesh.copy()
        self.subdivision_level = 0
        self.smooth_mesh = self.control_mesh.copy()

        self.fig = plt.figure("Subdivision")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("black")

        self._drag_vertex = None
        self._press_xy = None
        self._press_vertex_pos = None

        self._connect_events()
        self._redraw()

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _project_points(self, points: np.ndarray) -> np.ndarray:
        """Project 3D points to display coordinates."""
        x2, y2, _ = proj3d.proj_transform(points[:, 0], points[:, 1], points[:, 2], self.ax.get_proj())
        disp = self.ax.transData.transform(np.column_stack([x2, y2]))
        return disp

    def _pick_vertex(self, event, threshold_px: float = 15.0):
        if event.x is None or event.y is None:
            return None
        p2 = self._project_points(self.control_mesh.vertices)
        d2 = np.sum((p2 - np.array([event.x, event.y])) ** 2, axis=1)
        i = int(np.argmin(d2))
        if float(np.sqrt(d2[i])) <= threshold_px:
            return i
        return None

    def _on_key(self, event) -> None:
        if event.key is None:
            return
        if event.key.lower() == "s":
            self.subdivision_level += 1
            self.smooth_mesh = subdivide(self.control_mesh, self.subdivision_level)
            self._redraw()
        elif event.key.lower() == "r":
            self.subdivision_level = 0
            self.smooth_mesh = self.control_mesh.copy()
            self._redraw()

    def _on_press(self, event) -> None:
        if event.button != 1:
            return
        idx = self._pick_vertex(event)
        if idx is None:
            return
        self._drag_vertex = idx
        self._press_xy = np.array([event.x, event.y], dtype=float)
        self._press_vertex_pos = self.control_mesh.vertices[idx].copy()

    def _on_motion(self, event) -> None:
        if self._drag_vertex is None or event.x is None or event.y is None:
            return

        # Screen-space drag mapped to current view basis (simple but effective).
        dx, dy = np.array([event.x, event.y], dtype=float) - self._press_xy
        elev = np.deg2rad(self.ax.elev)
        azim = np.deg2rad(self.ax.azim)

        right = np.array([-np.sin(azim), np.cos(azim), 0.0])
        up = np.array(
            [
                -np.cos(azim) * np.sin(elev),
                -np.sin(azim) * np.sin(elev),
                np.cos(elev),
            ]
        )
        scale = 0.005 * max(np.linalg.norm(np.ptp(self.control_mesh.vertices, axis=0)), 1.0)
        delta = scale * (dx * right + -dy * up)

        self.control_mesh.vertices[self._drag_vertex] = self._press_vertex_pos + delta
        self.smooth_mesh = subdivide(self.control_mesh, self.subdivision_level)
        self._redraw()

    def _on_release(self, _event) -> None:
        self._drag_vertex = None
        self._press_xy = None
        self._press_vertex_pos = None

    def _mesh_edges(self, mesh: Mesh):
        edges = set()
        for a, b, c in mesh.faces:
            a, b, c = int(a), int(b), int(c)
            edges.add(tuple(sorted((a, b))))
            edges.add(tuple(sorted((b, c))))
            edges.add(tuple(sorted((c, a))))
        segments = [(mesh.vertices[u], mesh.vertices[v]) for u, v in edges]
        return segments

    def _redraw(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("black")
        self.ax.set_title(
            f"Loop subdivision (level={self.subdivision_level}) | S=subdivide, R=reset", color="white"
        )

        smooth_faces = [self.smooth_mesh.vertices[f] for f in self.smooth_mesh.faces]
        poly = Poly3DCollection(smooth_faces, facecolor=(0.7, 0.7, 0.7, 0.65), edgecolor=(0.2, 0.2, 0.2, 0.2))
        self.ax.add_collection3d(poly)

        control_edges = self._mesh_edges(self.control_mesh)
        line = Line3DCollection(control_edges, colors=(0.0, 0.9, 0.2, 1.0), linewidths=1.2)
        self.ax.add_collection3d(line)
        self.ax.scatter(
            self.control_mesh.vertices[:, 0],
            self.control_mesh.vertices[:, 1],
            self.control_mesh.vertices[:, 2],
            color="#00ff33",
            s=25,
            depthshade=False,
        )

        pts = np.vstack([self.control_mesh.vertices, self.smooth_mesh.vertices])
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = max(np.max(maxs - mins) * 0.55, 1e-3)

        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)
        self.ax.set_box_aspect((1, 1, 1))

        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0, 0, 0, 1))
            axis.set_tick_params(colors="white")

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()
