"""Struts and boxes: the pieces that towers, gates and frames are put together from.

Every piece is a closed shape wound to face outward, which is what the renderer needs: it
draws one side of a triangle only.
"""
import numpy as np


def strut(a, b, radius, sides=4):
    """A straight bar from a to b: (vertices, faces) of a prism with that many sides, ends capped."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    along = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    side = np.cross(along, [0.0, 0.0, 1.0] if abs(along[2]) < 0.9 else [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    other = np.cross(along, side)
    turn = (np.arange(sides) + 0.5) * 2 * np.pi / sides
    ring = radius * (np.cos(turn)[:, None] * side + np.sin(turn)[:, None] * other)
    vertices = np.concatenate([a + ring, b + ring])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [(i, j, sides + j), (i, sides + j, sides + i)]
    for i in range(1, sides - 1):
        faces += [(0, i + 1, i), (sides, sides + i, sides + i + 1)]
    return vertices, np.array(faces)


def box(centre, half_axes):
    """A box about centre with the three given half edges, as vectors: (vertices, faces)."""
    centre, (u, v, w) = np.asarray(centre, float), (np.asarray(h, float) for h in half_axes)
    if np.dot(np.cross(u, v), w) < 0:
        w = -w                                  # keep the three right-handed, so the winding below faces out
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    vertices = centre + signs[:, :1] * u + signs[:, 1:2] * v + signs[:, 2:] * w
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return vertices, np.array(faces)


def merge(pieces):
    """Several (vertices, faces) as one."""
    vertices, faces, base = [], [], 0
    for v, f in pieces:
        vertices.append(v)
        faces.append(f + base)
        base += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def volume(vertices, faces):
    """The volume a mesh encloses: positive when every piece is closed and faces outward."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
