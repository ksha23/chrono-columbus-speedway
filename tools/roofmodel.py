"""A gable roof over a building's plan: its two slopes fitted to the scan, with an edge to it.

All four roofs on this site are gables with the ridge along the building. The ridge is not
always in the middle: the main building's roof carries on down over a wing on one side, and
its ridge sits over the middle of the main block. So the two slopes are fitted as two lines to
the scan's heights across the building, and the ridge is where they meet.

A roof drawn as a sheet has no edge, and from the ground the edge is most of what shows: so
the roof is a slab. Its top keeps the roof's own photograph. Its edge and its underside are
the trim.
"""
import numpy as np
from scipy import ndimage

BIN = 0.25          # metres across the building that one reading of the height covers
EDGE = 0.75         # metres from the roof's edge and from its ridge that readings are left out
FASCIA = 0.22       # metres from the roof's top to its underside
PHOTO = 0.05        # metres a pixel of the roof's photograph covers


def fit(across, heights, plan):
    """The roof's shape across the building: {"ridge_at", "ridge", "fall": (low side, high side)}.

    across and heights are the frame coordinate and the height above the ground of each cell
    the roof covers. The falls are metres down per metre away from the ridge.
    """
    low, high = min(strip[2] for strip in plan), max(strip[3] for strip in plan)
    edges = np.arange(low + EDGE, high - EDGE + BIN, BIN)
    which = np.digitize(across, edges) - 1
    middle, level = [], []
    for n in range(len(edges) - 1):
        here = heights[which == n]
        if len(here) >= 6:
            middle.append((edges[n] + edges[n + 1]) / 2)
            level.append(np.median(here))
    middle, level = np.array(middle), np.array(level)
    top = float(middle[np.argmax(ndimage.uniform_filter1d(level, 5, mode="nearest"))])
    sides = []
    for side in (middle < top - EDGE, middle > top + EDGE):
        if side.sum() >= 3:
            sides.append(np.polyfit(middle[side], level[side], 1))
        else:
            sides.append(np.array([0.0, float(level.max())]))
    (rise, start), (drop, end) = sides
    if rise - drop > 0.02:           # two slopes that meet: the ridge is where
        ridge_at = float(np.clip((end - start) / (rise - drop), low + 2 * EDGE, high - 2 * EDGE))
    else:
        ridge_at = top
    ridge = float(max(start + rise * ridge_at, end + drop * ridge_at))
    return {"ridge_at": ridge_at, "ridge": ridge, "fall": (float(max(rise, 0.0)), float(max(-drop, 0.0)))}


def height(roof, across):
    """The roof's height above the ground at a distance across the building."""
    across = np.asarray(across, float)
    away = across - roof["ridge_at"]
    return roof["ridge"] - np.where(away < 0, -away * roof["fall"][0], away * roof["fall"][1])


def _pieces(plan, roof):
    """The plan cut into rectangles that each lie on one slope: (u0, u1, w0, w1)."""
    out = []
    for u0, u1, low, high in plan:
        if low < roof["ridge_at"] < high:
            out += [(u0, u1, low, roof["ridge_at"]), (u0, u1, roof["ridge_at"], high)]
        else:
            out.append((u0, u1, low, high))
    return out


def _edges(ring, roof):
    """The outline's edges, an edge that crosses the ridge cut in two there: pairs of (u, w) points."""
    out = []
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        if min(a[1], b[1]) < roof["ridge_at"] - 1e-6 and max(a[1], b[1]) > roof["ridge_at"] + 1e-6:
            mid = np.array([a[0], roof["ridge_at"]])
            out += [(a, mid), (mid, b)]
        else:
            out.append((a, b))
    return out


def extent(plan):
    """The rectangle the plan lies in: (u0, u1, w0, w1)."""
    return plan[0][0], plan[-1][1], min(strip[2] for strip in plan), max(strip[3] for strip in plan)


def model(plan, ring, roof, place, fascia=FASCIA):
    """The roof as meshes: {"roof": (v, vn, vt, f), "trim": (v, vn, f)}.

    ring is the plan's outline, anticlockwise. place(u, w, h) gives the scene point at frame
    coordinates u and w and height h above the building's ground. The roof's top is mapped to
    its photograph (see photo()): the trim is its edge all round and its underside.
    """
    u0, u1, w0, w1 = extent(plan)

    def uv(u, w):
        return [(u - u0) / (u1 - u0), (w - w0) / (w1 - w0)]

    top = ([], [], [], [])
    trim = ([], [], [])
    for a0, a1, b0, b1 in _pieces(plan, roof):
        corners = [(a0, b0), (a1, b0), (a1, b1), (a0, b1)]
        heights = [float(height(roof, w)) for _, w in corners]
        points = [place(u, w, h) for (u, w), h in zip(corners, heights)]
        normal = np.cross(np.subtract(points[1], points[0]), np.subtract(points[3], points[0]))
        normal /= np.linalg.norm(normal)
        base = len(top[0])
        top[0].extend(points)
        top[1].extend([normal.tolist()] * 4)
        top[2].extend(uv(u, w) for u, w in corners)
        top[3].extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
        # The underside: the same piece lower down, facing the ground.
        base = len(trim[0])
        trim[0].extend(place(u, w, h - fascia) for (u, w), h in zip(corners, heights))
        trim[1].extend([(-normal).tolist()] * 4)
        trim[2].extend([[base, base + 2, base + 1], [base, base + 3, base + 2]])
    for a, b in _edges(ring, roof):
        ha, hb = float(height(roof, a[1])), float(height(roof, b[1]))
        points = [place(a[0], a[1], ha - fascia), place(b[0], b[1], hb - fascia), place(b[0], b[1], hb), place(a[0], a[1], ha)]
        out = np.cross(np.subtract(points[1], points[0]), [0.0, 0.0, 1.0])
        out /= np.linalg.norm(out)
        base = len(trim[0])
        trim[0].extend(points)
        trim[1].extend([out.tolist()] * 4)
        trim[2].extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    return {"roof": tuple(np.array(part) for part in top), "trim": tuple(np.array(part) for part in trim)}


def photo(plan, place, picture, raster, pixel=PHOTO):
    """The roof's plan view cut from the site's photo and turned square to the building.

    Covers the rectangle the plan lies in, u from left to right and w from the bottom row up:
    what model() maps the roof's top to.
    """
    u0, u1, w0, w1 = extent(plan)
    wide, tall = int(round((u1 - u0) / pixel)), int(round((w1 - w0) / pixel))
    u = u0 + (np.arange(wide) + 0.5) / wide * (u1 - u0)
    w = w1 - (np.arange(tall) + 0.5) / tall * (w1 - w0)
    U, W = np.meshgrid(u, w)
    x, y, _ = place(U, W, 0.0)
    cols = (x - raster["x0"]) / raster["res"] - 0.5
    rows = (raster["y1"] - y) / raster["res"] - 0.5
    r0, c0 = int(rows.min()) - 2, int(cols.min()) - 2
    window = np.asarray(picture[r0:int(rows.max()) + 3, c0:int(cols.max()) + 3, :3], dtype=np.float32)
    return np.stack([ndimage.map_coordinates(window[..., ch], [rows - r0, cols - c0], order=1, mode="nearest") for ch in range(3)], -1).astype(np.uint8)
