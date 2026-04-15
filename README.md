# Quiz 26 — Interactive Loop Subdivision (Python)

This project implements a **basic interactive modeling tool** in Python that mirrors the assignment goal shown in your screenshot:

- Load a triangular control mesh from a `.ray` file.
- Render the control cage (green wireframe + points).
- Render a smooth subdivision surface (gray shaded mesh).
- Press `S` to apply one more level of **Loop subdivision**.
- Drag control vertices with the mouse and see the smooth surface update continuously.

## Features implemented

1. **Mesh representation**
   - Indexed triangle mesh (`vertices`, `faces`).
   - Edge/neighbor/boundary queries used by subdivision.

2. **Loop subdivision algorithm**
   - Even vertex update for interior and boundary vertices.
   - Odd edge vertex creation for interior and boundary edges.
   - 1-to-4 triangle split per face.

3. **Interactive viewer (matplotlib 3D)**
   - Left click near a control point to select it.
   - Drag to move selected point in view plane.
   - `S` key increases subdivision level.
   - `R` key resets to level 0.

4. **`.ray` loader**
   - Supports `v/f` style lines:
     - `v x y z`
     - `f i j k` (1-based)
   - Also supports flat triangle lines with 9 floats.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py assets/octahedron.ray
```

Optional initial level:

```bash
python main.py assets/octahedron.ray --level 2
```

## Controls

- **Rotate camera**: matplotlib default (left-drag not on a point).
- **Zoom**: scroll wheel.
- **Select + drag vertex**: left-click near green point, then drag.
- **Subdivide**: `S`
- **Reset level**: `R`

## Project structure

- `main.py` — CLI entry point.
- `subdivision/mesh.py` — mesh data model + topology helpers.
- `subdivision/loop.py` — Loop subdivision core.
- `subdivision/ray_loader.py` — input parsing.
- `subdivision/viewer.py` — interactive display + editing.
- `tests/test_loop.py` — unit tests for subdivision behavior.

## Notes

- This is designed to be clear and assignment-friendly, with explicit topology computations.
- For large meshes/high levels, repeated full recomputation can be slow; this can be optimized later with caching or a half-edge structure.
