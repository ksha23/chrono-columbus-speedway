"""Rebuild the scanned buildings as clean models: straight walls and a gable roof.

Photogrammetry from overhead gives a building a good roof and melted walls. Each building here
is refitted as a rectangle in plan with a ridge along its long side, which is what all four on
this site are. Eave and ridge heights come from the scan's height above the lidar ground. The
roof keeps its own photograph as a texture. The walls get one flat colour, taken from what
little of them the drone saw.
"""
import os

import numpy as np
from PIL import Image
from scipy import ndimage

import renderer

OVERHANG = 0.3      # metres the roof extends past the walls
SUNK = 0.4          # metres the walls continue below the lowest ground at the footprint


def fit_rectangle(xs, ys):
    """Smallest rectangle around plan points: (centre, unit long axis, half length, half width)."""
    pts = np.stack([xs, ys], 1)
    centre = pts.mean(0)
    best = None
    for angle in np.radians(np.arange(0, 180, 0.5)):
        u = np.array([np.cos(angle), np.sin(angle)])
        v = np.array([-u[1], u[0]])
        a, b = (pts - centre) @ u, (pts - centre) @ v
        # Percentiles, not extremes: the mask's edge is ragged by a cell or two.
        a0, a1 = np.percentile(a, [1, 99])
        b0, b1 = np.percentile(b, [1, 99])
        area = (a1 - a0) * (b1 - b0)
        if best is None or area < best[0]:
            best = (area, u, v, a0, a1, b0, b1)
    _, u, v, a0, a1, b0, b1 = best
    centre = centre + u * (a0 + a1) / 2 + v * (b0 + b1) / 2
    half_u, half_v = (a1 - a0) / 2, (b1 - b0) / 2
    if half_v > half_u:
        u, v, half_u, half_v = v, -u, half_v, half_u
    return centre, u, half_u, half_v


def measure(mask, height, gx, gy, ref):
    """One dict per building in the mask: plan rectangle, eave and ridge height, ground level."""
    labels, count = ndimage.label(mask)
    found = []
    for n in range(1, count + 1):
        sel = labels == n
        xs, ys, hs = gx[sel], gy[sel], height[sel]
        centre, u, half_u, half_v = fit_rectangle(xs, ys)
        v = np.array([-u[1], u[0]])
        across = (np.stack([xs, ys], 1) - centre) @ v
        # A gable roof is highest along the middle and lowest at the two long edges.
        ridge = float(np.percentile(hs[np.abs(across) < 0.15 * half_v], 60))
        eave = float(np.percentile(hs[np.abs(across) > 0.8 * half_v], 40))
        corners = [centre + su * half_u * u + sv * half_v * v for su in (-1, 1) for sv in (-1, 1)]
        ground = [float(ref.elevation(c[0], c[1])) for c in corners]
        found.append({"centre": [float(centre[0]), float(centre[1])], "axis": [float(u[0]), float(u[1])],
                      "half_length": float(half_u), "half_width": float(half_v),
                      "eave": eave, "ridge": max(ridge, eave + 0.05), "ground": float(np.mean(ground)), "ground_low": float(min(ground))})
    return found


def _quad(vertices, normals, uvs, faces, corners, normal, uv=None):
    base = len(vertices)
    vertices += [list(map(float, c)) for c in corners]
    normals += [list(map(float, normal))] * len(corners)
    uvs += uv if uv is not None else [[0.0, 0.0]] * len(corners)
    faces += [[base, base + 1, base + 2]] + ([[base, base + 2, base + 3]] if len(corners) == 4 else [])


def model(b):
    """Geometry of one building: {"roof": (v, vn, vt, f), "walls": (v, vn, vt, f)}, scene coordinates."""
    c = np.array(b["centre"])
    u = np.array(b["axis"])
    v = np.array([-u[1], u[0]])
    L, Wd = b["half_length"], b["half_width"]
    z0, ze, zr = b["ground_low"] - SUNK, b["ground"] + b["eave"], b["ground"] + b["ridge"]

    def p(a, bb, z):
        return [c[0] + a * u[0] + bb * v[0], c[1] + a * u[1] + bb * v[1], z]

    walls = ([], [], [], [])
    for (a0, b0), (a1, b1), n in [((-L, -Wd), (L, -Wd), -v), ((L, -Wd), (L, Wd), u), ((L, Wd), (-L, Wd), v), ((-L, Wd), (-L, -Wd), -u)]:
        _quad(*walls, [p(a0, b0, z0), p(a1, b1, z0), p(a1, b1, ze), p(a0, b0, ze)], [n[0], n[1], 0.0])
    for s in (-1, 1):   # the two gable ends, triangles above the eave line
        tri = [p(s * L, -Wd, ze), p(s * L, Wd, ze), p(s * L, 0, zr)]
        _quad(*walls, tri if s > 0 else tri[::-1], [s * u[0], s * u[1], 0.0])

    roof = ([], [], [], [])
    Lo, Wo = L + OVERHANG, Wd + OVERHANG
    drop = (zr - ze) * OVERHANG / Wd     # the roof plane carried on down past the wall
    rise = np.arctan2(zr - ze, Wd)

    def uv(a, bb):
        """The roof's photo is its plan view: u along the ridge, v across."""
        return [(a + Lo) / (2 * Lo), (bb + Wo) / (2 * Wo)]

    for s in (-1, 1):
        n = [s * v[0] * np.sin(rise), s * v[1] * np.sin(rise), np.cos(rise)]
        quad = [p(-Lo, s * Wo, ze - drop), p(Lo, s * Wo, ze - drop), p(Lo, 0, zr), p(-Lo, 0, zr)]
        coords = [uv(-Lo, s * Wo), uv(Lo, s * Wo), uv(Lo, 0), uv(-Lo, 0)]
        if s > 0:
            quad, coords = quad[::-1], coords[::-1]
        _quad(*roof, quad, n, coords)
    return {"roof": tuple(np.array(x) for x in roof), "walls": tuple(np.array(x) for x in walls)}


def roof_photo(b, photo, raster, pixel=0.05):
    """The roof's plan view cut from the ground photo and turned square to the building."""
    c, u = np.array(b["centre"]), np.array(b["axis"])
    v = np.array([-u[1], u[0]])
    Lo, Wo = b["half_length"] + OVERHANG, b["half_width"] + OVERHANG
    w, h = int(round(2 * Lo / pixel)), int(round(2 * Wo / pixel))
    a = (np.arange(w) + 0.5) / w * 2 * Lo - Lo
    bb = Wo - (np.arange(h) + 0.5) / h * 2 * Wo      # image rows run from v = 1 down to v = 0
    A, B = np.meshgrid(a, bb)
    x, y = c[0] + A * u[0] + B * v[0], c[1] + A * u[1] + B * v[1]
    cols = (x - raster["x0"]) / raster["res"] - 0.5
    rows = (raster["y1"] - y) / raster["res"] - 0.5
    r0, c0 = int(rows.min()) - 2, int(cols.min()) - 2
    window = np.asarray(photo[r0:int(rows.max()) + 3, c0:int(cols.max()) + 3], dtype=np.float32)
    return np.stack([ndimage.map_coordinates(window[..., ch], [rows - r0, cols - c0], order=1, mode="nearest") for ch in range(3)], -1).astype(np.uint8)


def write_obj(path, part, with_uv):
    v, vn, vt, f = part
    with open(path, "w") as out:
        out.write("".join(f"v {x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in v))
        if with_uv:
            out.write("".join(f"vt {a:.5f} {b:.5f}\n" for a, b in vt))
        out.write("".join(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in vn))
        for a, b, c in f + 1:
            out.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" if with_uv else f"f {a}//{a} {b}//{b} {c}//{c}\n")


def write(found, photo, raster, scene_dir, wall_colours):
    """Write every building's meshes and roof photo. Returns manifest assets."""
    os.makedirs(os.path.join(scene_dir, "buildings"), exist_ok=True)
    assets = []
    for n, b in enumerate(found, start=1):
        parts = model(b)
        name = f"building_{n}"
        write_obj(os.path.join(scene_dir, "buildings", name + "_roof.obj"), parts["roof"], True)
        write_obj(os.path.join(scene_dir, "buildings", name + "_walls.obj"), parts["walls"], False)
        roof = roof_photo(b, photo, raster)
        Image.fromarray(renderer.for_renderer(roof)).save(os.path.join(scene_dir, "buildings", name + "_roof.jpg"), quality=90, optimize=True)
        assets.append({"name": name, "parts": [
            {"name": "roof", "mesh": f"buildings/{name}_roof.obj", "texture": f"buildings/{name}_roof.jpg", "colour": [1.0, 1.0, 1.0], "roughness_value": 0.6},
            {"name": "walls", "mesh": f"buildings/{name}_walls.obj", "colour": renderer.colour_for_renderer(wall_colours[n - 1]), "roughness_value": 0.8},
        ], "footprint": b})
    return assets
