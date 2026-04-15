from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .mesh import Mesh


def load_ray(path: str | Path) -> Mesh:
    """Load a triangle mesh from a .ray-style file.

    Supported formats:
    - OBJ-like subset: lines beginning with `v x y z` and `f i j k` (1-based indices).
    - Flat triangles: each non-comment line has 9 floats: x1 y1 z1 x2 y2 z2 x3 y3 z3.
      Vertices are deduplicated with exact matching.
    """
    path = Path(path)
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    has_obj_style = any(ln.startswith("v ") or ln.startswith("f ") for ln in lines)

    if has_obj_style:
        for ln in lines:
            if ln.startswith("v "):
                _, x, y, z, *_ = ln.split()
                vertices.append([float(x), float(y), float(z)])
            elif ln.startswith("f "):
                parts = ln.split()[1:4]
                idxs = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(idxs)
    else:
        v_map = {}

        def get_idx(v):
            t = tuple(v)
            if t not in v_map:
                v_map[t] = len(vertices)
                vertices.append(list(t))
            return v_map[t]

        for ln in lines:
            vals = [float(x) for x in ln.split()]
            if len(vals) != 9:
                raise ValueError(
                    f"Unsupported line format in {path}: expected 9 floats, got {len(vals)} ({ln})"
                )
            tri = [vals[0:3], vals[3:6], vals[6:9]]
            faces.append([get_idx(v) for v in tri])

    if not vertices or not faces:
        raise ValueError(f"No mesh data parsed from {path}")

    return Mesh(np.asarray(vertices), np.asarray(faces, dtype=int))
