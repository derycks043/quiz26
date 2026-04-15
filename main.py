from __future__ import annotations

import argparse

from subdivision.loop import subdivide
from subdivision.ray_loader import load_ray
from subdivision.viewer import MeshViewer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Loop subdivision viewer")
    p.add_argument("mesh", help="Path to .ray mesh file")
    p.add_argument("-l", "--level", type=int, default=0, help="Initial subdivision level")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    control = load_ray(args.mesh)
    viewer = MeshViewer(control)
    if args.level > 0:
        viewer.subdivision_level = args.level
        viewer.smooth_mesh = subdivide(viewer.control_mesh, args.level)
        viewer._redraw()
    viewer.show()


if __name__ == "__main__":
    main()
