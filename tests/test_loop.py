import numpy as np

from subdivision.loop import subdivide_once
from subdivision.mesh import Mesh


def test_single_triangle_counts():
    mesh = Mesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        np.array([[0, 1, 2]]),
    )
    out = subdivide_once(mesh)
    assert out.vertices.shape[0] == 6  # 3 old + 3 edge points
    assert out.faces.shape[0] == 4


def test_tetrahedron_face_growth():
    mesh = Mesh(
        np.array(
            [
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
            ]
        ),
        np.array(
            [
                [0, 1, 2],
                [0, 3, 1],
                [0, 2, 3],
                [1, 3, 2],
            ]
        ),
    )
    out = subdivide_once(mesh)
    assert out.faces.shape[0] == 16
    assert np.all(np.isfinite(out.vertices))
