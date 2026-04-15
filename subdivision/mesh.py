from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Edge = Tuple[int, int]


def canon_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


@dataclass
class Mesh:
    """Triangle mesh representation using indexed vertices/faces."""

    vertices: np.ndarray  # shape (N, 3)
    faces: np.ndarray  # shape (M, 3), dtype=int

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.faces = np.asarray(self.faces, dtype=int)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be shaped (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be shaped (M, 3)")

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())

    def edge_faces(self) -> Dict[Edge, List[int]]:
        edge_map: Dict[Edge, List[int]] = {}
        for fi, (a, b, c) in enumerate(self.faces):
            for u, v in ((a, b), (b, c), (c, a)):
                edge_map.setdefault(canon_edge(int(u), int(v)), []).append(fi)
        return edge_map

    def vertex_neighbors(self) -> List[List[int]]:
        neighbors = [set() for _ in range(len(self.vertices))]
        for a, b, c in self.faces:
            a, b, c = int(a), int(b), int(c)
            neighbors[a].update((b, c))
            neighbors[b].update((a, c))
            neighbors[c].update((a, b))
        return [sorted(ns) for ns in neighbors]

    def boundary_edges(self) -> List[Edge]:
        return [e for e, fs in self.edge_faces().items() if len(fs) == 1]

    def boundary_vertices(self) -> List[int]:
        b = set()
        for u, v in self.boundary_edges():
            b.add(u)
            b.add(v)
        return sorted(b)

    def boundary_neighbors(self) -> Dict[int, List[int]]:
        m: Dict[int, List[int]] = {}
        for u, v in self.boundary_edges():
            m.setdefault(u, []).append(v)
            m.setdefault(v, []).append(u)
        for k in m:
            m[k] = sorted(m[k])
        return m
