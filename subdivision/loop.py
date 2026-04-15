from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .mesh import Mesh, canon_edge


def _beta(valence: int) -> float:
    if valence <= 2:
        return 3.0 / 16.0
    return 3.0 / (8.0 * valence)


def _boundary_even_vertex(old: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    return 0.75 * old + 0.125 * (n0 + n1)


def subdivide_once(mesh: Mesh) -> Mesh:
    vertices = mesh.vertices
    faces = mesh.faces
    edge_faces = mesh.edge_faces()
    v_neighbors = mesh.vertex_neighbors()
    b_neighbors = mesh.boundary_neighbors()

    # Compute updated old vertices ("even" vertices)
    new_even = np.zeros_like(vertices)
    for i, p in enumerate(vertices):
        if i in b_neighbors:
            bns = b_neighbors[i]
            if len(bns) >= 2:
                new_even[i] = _boundary_even_vertex(p, vertices[bns[0]], vertices[bns[-1]])
            elif len(bns) == 1:
                new_even[i] = 0.875 * p + 0.125 * vertices[bns[0]]
            else:
                new_even[i] = p
        else:
            nbs = v_neighbors[i]
            n = len(nbs)
            beta = _beta(n)
            new_even[i] = (1.0 - n * beta) * p + beta * np.sum(vertices[nbs], axis=0)

    # Compute edge points ("odd" vertices)
    edge_to_new_index: Dict[Tuple[int, int], int] = {}
    odd_vertices = []

    # Build opposite vertex lookup for each edge from each incident face
    opp_lookup: Dict[Tuple[int, int], list[int]] = {}
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        opp_lookup.setdefault(canon_edge(a, b), []).append(c)
        opp_lookup.setdefault(canon_edge(b, c), []).append(a)
        opp_lookup.setdefault(canon_edge(c, a), []).append(b)

    for edge, incident_faces in edge_faces.items():
        u, v = edge
        p_u, p_v = vertices[u], vertices[v]
        opposites = opp_lookup[edge]

        if len(incident_faces) == 1 or len(opposites) == 1:
            p_new = 0.5 * (p_u + p_v)
        else:
            p0, p1 = vertices[opposites[0]], vertices[opposites[1]]
            p_new = 0.375 * (p_u + p_v) + 0.125 * (p0 + p1)

        edge_to_new_index[edge] = len(vertices) + len(odd_vertices)
        odd_vertices.append(p_new)

    all_vertices = np.vstack([new_even, np.asarray(odd_vertices)])

    # Split each triangle into four
    new_faces = []
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        ab = edge_to_new_index[canon_edge(a, b)]
        bc = edge_to_new_index[canon_edge(b, c)]
        ca = edge_to_new_index[canon_edge(c, a)]

        new_faces.extend(
            [
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            ]
        )

    return Mesh(all_vertices, np.asarray(new_faces, dtype=int))


def subdivide(mesh: Mesh, levels: int) -> Mesh:
    out = mesh.copy()
    for _ in range(levels):
        out = subdivide_once(out)
    return out
