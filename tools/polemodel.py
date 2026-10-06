"""A street light: concrete footing, tapered steel pole, one arm, one lamp head.

Generated, not scanned. The drone shows how tall each pole is and where it stands, and that a
single arm carries a lamp out over the pavement. The rest is the ordinary shape of such a pole.
Metres, Z up, the pole's axis on the origin, the ground at z = 0, the arm along +x.

Three parts, so they can be coloured apart: "footing", "pole" (shaft and arm) and "lamp".
"""
import os

import numpy as np

SIDES = 8
FOOTING = (0.28, 0.45)     # radius and height above ground of the concrete footing
BURIED = 0.4               # how far the footing carries on below z = 0, for sloping ground
COLOURS = {"footing": (0.74, 0.73, 0.70), "pole": (0.66, 0.68, 0.70), "lamp": (0.80, 0.81, 0.83)}


def _tube(path, radii, sides=SIDES, cap=True):
    """A tube along a 3D path with a radius at each point. Returns (vertices, faces).

    Faces are wound to face outward: the renderer draws one side of a triangle only.
    """
    path = np.asarray(path, float)
    tangent = np.gradient(path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    # A frame carried along the path: pick a side vector once and keep it from twisting.
    side = np.cross(tangent[0], [0.0, 1.0, 0.0])
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    rings = []
    for p, t, r in zip(path, tangent, radii):
        side = side - (side @ t) * t
        side /= np.linalg.norm(side)
        up = np.cross(t, side)
        angle = np.arange(sides) * 2 * np.pi / sides
        rings.append(p + r * (np.cos(angle)[:, None] * side + np.sin(angle)[:, None] * up))
    v = np.concatenate(rings)
    f = []
    for k in range(len(path) - 1):
        for s in range(sides):
            a, b = k * sides + s, k * sides + (s + 1) % sides
            f += [(a, b, b + sides), (a, b + sides, a + sides)]
    if cap:
        for k, flip in ((0, True), (len(path) - 1, False)):
            centre = len(v)
            v = np.concatenate([v, path[k:k + 1]])
            for s in range(sides):
                a, b = k * sides + s, k * sides + (s + 1) % sides
                f.append((centre, b, a) if flip else (centre, a, b))
    return v, np.array(f)


def _box(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


def make(height):
    """{"footing": (vertices, faces), "pole": ..., "lamp": ...} for a pole this tall, in metres."""
    # Proportions grow with the pole: a 15 m mast is a heavier thing than a 7 m street light.
    thick = 0.085 + 0.006 * height                 # shaft radius at the footing
    reach = float(np.clip(0.2 * height, 1.5, 2.6))  # how far the arm carries the lamp out
    rise = 0.5
    top = height - rise

    footing = _tube([[0, 0, -BURIED], [0, 0, FOOTING[1]]], [FOOTING[0], FOOTING[0]])
    shaft = _tube([[0, 0, FOOTING[1]], [0, 0, top]], [thick, 0.55 * thick])
    # The arm leaves the top of the shaft and sweeps up and out in a quarter ellipse.
    t = np.linspace(0.0, np.pi / 2, 6)
    arm_path = np.stack([reach * np.sin(t), np.zeros_like(t), top - 0.1 + (rise + 0.1) * (1 - np.cos(t))], axis=1)
    arm = _tube(arm_path, np.full(len(t), 0.4 * thick), sides=6)
    pole_v = np.concatenate([shaft[0], arm[0]])
    pole_f = np.concatenate([shaft[1], arm[1] + len(shaft[0])])

    size = 0.045 * height + 0.3                    # the lamp head's length
    lamp = _box([reach - 0.1, -0.28 * size, height - 0.02 - 0.2 * size], [reach - 0.1 + size, 0.28 * size, height + 0.03])
    return {"footing": footing, "pole": (pole_v, pole_f), "lamp": lamp}


def write_obj(path, vertices, faces):
    """Flat-shaded OBJ: every triangle gets its own vertices and normal."""
    lines, count = [], 0
    for a, b, c in faces:
        n = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        length = np.linalg.norm(n)
        if length < 1e-12:
            continue
        n /= length
        for p in (a, b, c):
            lines.append(f"v {vertices[p][0]:.4f} {vertices[p][1]:.4f} {vertices[p][2]:.4f}\nvn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        count += 1
    with open(path, "w") as out:
        out.write("".join(lines))
        out.write("".join(f"f {3 * t + 1}//{3 * t + 1} {3 * t + 2}//{3 * t + 2} {3 * t + 3}//{3 * t + 3}\n" for t in range(count)))
    return count


def write(height, directory):
    """Write one pole's three meshes. Returns (asset name, triangle count)."""
    os.makedirs(directory, exist_ok=True)
    name = f"pole_{int(round(height * 10)):03d}"
    total = 0
    for part, (v, f) in make(height).items():
        total += write_obj(os.path.join(directory, f"{name}_{part}.obj"), v, f)
    return name, total
