"""The walls under a roof: where they stand, how high each reaches, and the mesh of them.

A wall stands under the roof's edge, set in by the roof's overhang, and reaches up to the
roof wherever the roof is above it: level along the eaves, and up to the ridge and down again
at a gable end. Each wall is one flat piece mapped to its own picture (facade.py).
"""
import numpy as np

import facade
import roofmodel
import tubes

SUNK = 0.4          # metres the walls continue below the lowest ground at the building
TUCK = 0.03         # metres a wall's top is kept under the roof's top, so the two never share a surface
REACH = 1.5         # metres from a wall that a door or window read by hand may be and still be put in it


def walls(ring, roof, lift):
    """One dict per wall of an anticlockwise outline, in the building's frame.

    start is the wall's left end as seen from outside, along the unit vector toward its right
    end, out the unit vector it faces. tops are (s, z) points along its top edge, z above the
    lowest ground at the building: lift is how far the building's own ground level is above that.
    """
    out = []
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        length = float(np.hypot(*(b - a)))
        along = (b - a) / length
        gable = abs(along[1]) > 0.5
        at = [0.0, length]
        if gable and min(a[1], b[1]) < roof["ridge_at"] < max(a[1], b[1]):
            at.insert(1, abs(roof["ridge_at"] - a[1]))
        tops = [(s, float(roofmodel.height(roof, a[1] + s * along[1])) + lift - TUCK) for s in at]
        out.append({"start": a, "along": along, "out": np.array([along[1], -along[0]]), "length": length, "gable": gable, "tops": tops})
    return out


def assign(found, features, to_frame, log=print):
    """Give each hand-read feature to the wall it is in: a list of feature lists, one per wall.

    A feature's "at" is a scene point on or near its wall. to_frame turns scene x and y into
    frame coordinates. Each feature gets s, the metres along its wall of its middle, and off,
    how far out from the wall the point was.
    """
    per_wall = [[] for _ in found]
    for f in features:
        if "at" not in f:
            continue
        point = np.array(to_frame(*f["at"]))
        best = None
        for n, wall in enumerate(found):
            rel = point - wall["start"]
            s, off = float(rel @ wall["along"]), float(rel @ wall["out"])
            if -0.2 <= s <= wall["length"] + 0.2 and abs(off) <= REACH and (best is None or abs(off) < abs(best[0])):
                best = (off, n, s)
        if best is None:
            log(f"    a {f['kind']} at {f['at']} is not within {REACH} m of any wall: left out")
            continue
        half = f.get("wide", 0.0) / 2
        s = float(np.clip(best[2], half + 0.05, max(found[best[1]]["length"] - half - 0.05, half + 0.05)))
        per_wall[best[1]].append(dict(f, s=s, off=best[0]))
    return per_wall


def mesh(found, pictures, places, size, sunk, place):
    """The walls as one mesh mapped to their shared picture: (v, vn, vt, f).

    pictures and places are the walls' pictures and where each sits in the shared one, of the
    given size in (rows, columns). place(u, w, z) gives the scene point for frame coordinates
    and a height above the lowest ground.
    """
    v, vn, vt, f = [], [], [], []
    for wall, picture, (row, col) in zip(found, pictures, places):
        top = picture.shape[0] * facade.PIX - sunk             # the height the picture's first row is at
        corners = [(0.0, -sunk), (wall["length"], -sunk)] + [(s, z) for s, z in reversed(wall["tops"])]
        base = len(v)
        for s, z in corners:
            point = wall["start"] + s * wall["along"]
            v.append(place(point[0], point[1], z))
            vt.append([(col + s / facade.PIX) / size[1], 1.0 - (row + (top - z) / facade.PIX) / size[0]])
        east, north = place(wall["out"][0], wall["out"][1], 0.0)[:2] - place(0.0, 0.0, 0.0)[:2]
        vn += [[float(east), float(north), 0.0]] * len(corners)
        f += [[base, base + n, base + n + 1] for n in range(1, len(corners) - 1)]
    return np.array(v), np.array(vn), np.array(vt), np.array(f)


def canopy(wall, f, sunk, place):
    """A roof over a door, standing out from the wall on two posts: {"canopy": (v, f), "posts": (v, f)}."""
    def at(s, off, z):
        point = wall["start"] + s * wall["along"] + off * wall["out"]
        return place(point[0], point[1], z)

    half, deep = f["wide"] / 2, f["deep"]
    slab = tubes.box((at(f["s"], deep / 2, f["high"]) + at(f["s"], deep / 2, f["low"])) / 2 + [0, 0, 0.0],
                     [at(f["s"] + half, 0, 0) - at(f["s"], 0, 0), (at(f["s"], deep, f["low"]) - at(f["s"], 0, f["high"])) / 2, [0, 0, f.get("thick", 0.25) / 2]])
    posts = [tubes.strut(at(f["s"] + side * (half - 0.15), deep - 0.15, -sunk), at(f["s"] + side * (half - 0.15), deep - 0.15, f["low"]), 0.08) for side in (-1, 1)]
    return {"canopy": slab, "posts": tubes.merge(posts)}


def bollards(wall, posts, sunk, place):
    """Guard posts standing in front of a wall, as at either side of a wide door: (v, f)."""
    out = []
    for f in posts:
        point = wall["start"] + f["s"] * wall["along"] + max(f["off"], 0.3) * wall["out"]
        out.append(tubes.strut(place(point[0], point[1], -sunk), place(point[0], point[1], f.get("tall", 1.1)), f.get("radius", 0.08), sides=10))
    return tubes.merge(out)
