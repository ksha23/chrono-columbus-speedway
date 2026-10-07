"""Rebuild the scanned buildings as clean models: straight walls under a gable roof.

Photogrammetry from overhead gives a building a good roof and melted walls. Each building here
is refitted from its roof: the roof's outline is its plan (footprint.py), the roof's heights
are its two slopes (roofmodel.py), and the walls stand under its edge (wallmodel.py). The roof
keeps its own photograph as a texture.

The walls are painted (facade.py). What each is clad in and where its doors and windows are
was read by hand from what little of the walls the drone saw, and is kept in facades.json. A
building that file does not name gets plain walls in the colour the scan shows.
"""
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

import facade
import footprint
import railmodel
import renderer
import roofmodel
import tubes
import wallmodel

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "facades.json")
OVERHANG = 0.3      # metres the roof extends past the walls, unless facades.json says otherwise
NEAR = 6.0          # metres between a building's middle and the point facades.json names it by
EXTRAS = {"canopy": (62, 70, 82), "posts": (150, 152, 155), "bollards": (228, 190, 40)}


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
    """One dict per building in the mask: its frame, its plan, its roof and its ground level.

    The frame is the rectangle fitted round the roof: centre, axis (the unit vector along the
    ridge), half_length and half_width. plan is the roof's outline in that frame and roof its
    two slopes. eave and ridge are the roof's lowest and highest edge above the ground.
    """
    labels, count = ndimage.label(mask)
    cell = float(gx[0, 1] - gx[0, 0])
    x0, y1 = float(gx[0, 0]) - cell / 2, float(gy[0, 0]) + cell / 2
    found = []
    for n in range(1, count + 1):
        sel = labels == n
        xs, ys, hs = gx[sel], gy[sel], height[sel]
        centre, u, half_u, half_v = fit_rectangle(xs, ys)
        v = np.array([-u[1], u[0]])

        def covered(a, q):
            x, y = centre[0] + a * u[0] + q * v[0], centre[1] + a * u[1] + q * v[1]
            rows, cols = ((y1 - y) / cell).astype(int), ((x - x0) / cell).astype(int)
            inside = (rows >= 0) & (rows < sel.shape[0]) & (cols >= 0) & (cols < sel.shape[1])
            return inside & sel[np.clip(rows, 0, sel.shape[0] - 1), np.clip(cols, 0, sel.shape[1] - 1)]

        plan = footprint.strips(covered, half_u, half_v)
        roof = roofmodel.fit((np.stack([xs, ys], 1) - centre) @ v, hs, plan)
        low, high = min(strip[2] for strip in plan), max(strip[3] for strip in plan)
        eave = float(min(roofmodel.height(roof, low), roofmodel.height(roof, high)))
        corners = [centre + su * half_u * u + sv * half_v * v for su in (-1, 1) for sv in (-1, 1)]
        ground = [float(ref.elevation(c[0], c[1])) for c in corners]
        found.append({"centre": [float(centre[0]), float(centre[1])], "axis": [float(u[0]), float(u[1])],
                      "half_length": float(half_u), "half_width": float(half_v),
                      "plan": [[round(float(x), 3) for x in strip] for strip in plan], "roof": roof,
                      "eave": eave, "ridge": max(roof["ridge"], eave + 0.05), "ground": float(np.mean(ground)), "ground_low": float(min(ground))})
    return found


def frame(b):
    """The two ways between a building's frame and the scene: (place, to_frame).

    place(u, w, z) is the scene point at frame coordinates u and w and z metres above the
    lowest ground at the building. to_frame(x, y) is the frame coordinates of a scene point.
    """
    c, u = np.array(b["centre"]), np.array(b["axis"])
    v = np.array([-u[1], u[0]])

    def place(a, q, z):
        return np.stack([c[0] + a * u[0] + q * v[0], c[1] + a * u[1] + q * v[1], b["ground_low"] + z + 0.0 * (a + q)])

    def to_frame(x, y):
        return (x - c[0]) * u[0] + (y - c[1]) * u[1], (x - c[0]) * v[0] + (y - c[1]) * v[1]

    return place, to_frame


def facades(path=FILE):
    with open(path) as f:
        return json.load(f)["buildings"]


def style_of(b, read, wall_colour):
    """What facades.json says of a building, or plain walls in the scan's colour if it says nothing."""
    for entry in read:
        if np.hypot(entry["near"][0] - b["centre"][0], entry["near"][1] - b["centre"][1]) < NEAR:
            return entry
    return {"name": None, "walls": {"cladding": "plain", "colour": [round(255 * float(c)) for c in wall_colour]}, "features": []}


def outline(b, style):
    """A building's walls, as wallmodel.walls gives them: under the roof's edge, set in by its overhang."""
    ring = footprint.inset(footprint.polygon(b["plan"]), style.get("overhang", OVERHANG))
    return wallmodel.walls(ring, b["roof"], b["ground"] - b["ground_low"])


def model(b, style, photo, raster, seed, log=print):
    """One building: {"meshes": {part: mesh}, "roof_photo": picture, "wall_photo": picture}."""
    place, to_frame = frame(b)
    lift = b["ground"] - b["ground_low"]
    ring = footprint.polygon(b["plan"])
    meshes = roofmodel.model(b["plan"], ring, b["roof"], lambda u, w, h: place(u, w, h + lift), style.get("fascia", roofmodel.FASCIA))
    found = outline(b, style)
    features = wallmodel.assign(found, style.get("features", []), to_frame, log)
    walls = dict(style["walls"], trim=style.get("trim", facade.WHITE))
    pictures = [facade.paint(wall, walls, here, wallmodel.SUNK, seed * 100 + n) for n, (wall, here) in enumerate(zip(found, features))]
    sheet, places = facade.atlas(pictures)
    meshes["walls"] = wallmodel.mesh(found, pictures, places, sheet.shape[:2], wallmodel.SUNK, place)
    extras = {}
    for wall, here in zip(found, features):
        for f in here:
            if f["kind"] == "canopy":
                for part, piece in wallmodel.canopy(wall, f, wallmodel.SUNK, place).items():
                    extras.setdefault(part, []).append(piece)
        posts = [f for f in here if f["kind"] == "bollard"]
        if posts:
            extras.setdefault("bollards", []).append(wallmodel.bollards(wall, posts, wallmodel.SUNK, place))
    return {"meshes": meshes, "extras": {part: tubes.merge(pieces) for part, pieces in extras.items()},
            "roof_photo": roofmodel.photo(b["plan"], place, photo, raster), "wall_photo": sheet, "walls": len(found),
            "features": sum(len(here) for here in features)}


def write_obj(path, part):
    """A mesh with a normal for every vertex: (v, vn, vt, f) with a picture, (v, vn, f) without."""
    v, vn, f = part[0], part[1], part[-1]
    vt = part[2] if len(part) == 4 else None
    with open(path, "w") as out:
        out.write("".join(f"v {x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in v))
        if vt is not None:
            out.write("".join(f"vt {a:.5f} {b:.5f}\n" for a, b in vt))
        out.write("".join(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in vn))
        for a, b, c in f + 1:
            out.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" if vt is not None else f"f {a}//{a} {b}//{b} {c}//{c}\n")


def write(found, photo, raster, scene_dir, wall_colours, log=print):
    """Write every building's meshes and pictures. Returns manifest assets, each with its building as "footprint"."""
    folder = os.path.join(scene_dir, "buildings")
    os.makedirs(folder, exist_ok=True)
    read = facades()
    assets = []
    for n, b in enumerate(found, start=1):
        name = f"building_{n}"
        style = style_of(b, read, wall_colours[n - 1])
        made = model(b, style, photo, raster, n, log)
        for part in ("roof", "walls", "trim"):
            write_obj(os.path.join(folder, f"{name}_{part}.obj"), made["meshes"][part])
        for part, picture in (("roof", made["roof_photo"]), ("walls", made["wall_photo"])):
            Image.fromarray(renderer.for_renderer(picture)).save(os.path.join(folder, f"{name}_{part}.jpg"), quality=90, optimize=True)
        trim = renderer.colour_for_renderer(np.asarray(style.get("trim", facade.WHITE), float) / 255)
        parts = [{"name": "roof", "mesh": f"buildings/{name}_roof.obj", "texture": f"buildings/{name}_roof.jpg", "colour": [1.0, 1.0, 1.0], "roughness_value": 0.6},
                 {"name": "walls", "mesh": f"buildings/{name}_walls.obj", "texture": f"buildings/{name}_walls.jpg", "colour": [1.0, 1.0, 1.0], "roughness_value": 0.8},
                 {"name": "trim", "mesh": f"buildings/{name}_trim.obj", "colour": [round(c, 3) for c in trim], "roughness_value": 0.7}]
        for part, (v, f) in made["extras"].items():
            railmodel.write_obj(os.path.join(folder, f"{name}_{part}.obj"), v, f)
            parts.append({"name": part, "mesh": f"buildings/{name}_{part}.obj", "roughness_value": 0.7,
                          "colour": [round(c, 3) for c in renderer.colour_for_renderer(np.asarray(EXTRAS[part], float) / 255)]})
        log(f"  {name}" + (f" ({style['name']})" if style.get("name") else "") + f": {2 * b['half_length']:.1f} x {2 * b['half_width']:.1f} m, eave {b['eave']:.1f} m,"
            f" ridge {b['ridge']:.1f} m, {made['walls']} walls, {made['features']} doors, windows and the like")
        assets.append({"name": name, "parts": parts, "footprint": b})
    return assets
