"""Turn the crowns measured in the scan into a planting plan for generated tree models.

The scan says where each tree stands, how tall it is, how wide its crown is and what colour its
leaves were that day. This picks a model for each one from a small library of generated trees
and says how to stretch, turn and colour it. A crown too wide to be one tree is a clump whose
tops ran together in the scan, and is replanted as several.
"""
import numpy as np
from scipy import ndimage

import renderer

# The library of generated models: kind, the height and crown radius it is generated at, its
# triangle budget, and the tallest tree it is used for. A placed tree is one of these stretched
# to the measured height and crown width. More than one proportion per kind keeps the stretch
# small, and young trees get lighter models because stock Chrono::VSG draws every triangle of
# every tree every frame.
LIBRARY = [
    ("broadleaf", 13.0, 5.4, 7000, 99.0), ("broadleaf", 13.0, 3.9, 7000, 99.0),
    ("broadleaf", 5.5, 2.4, 3200, 7.5), ("broadleaf", 5.5, 1.6, 3200, 7.5),
    ("upright", 14.0, 3.0, 7000, 99.0), ("upright", 6.0, 1.3, 3200, 7.5),
    ("willow", 10.0, 5.0, 7000, 99.0),
    ("shrub", 2.6, 1.5, 1800, 99.0), ("shrub", 2.6, 0.9, 1800, 99.0),
]
SEEDS = 3
# Stock Chrono::VSG draws every triangle of every tree once for the picture and again for each
# shadow map, at about 5 ms per million triangles on an M4 Pro. So trees a driver passes close
# to get the full model, and trees further than NEAR from any pavement get one with LIGHT of
# the triangles. From there the difference does not show.
NEAR = 12.0
LIGHT = 0.4
CLUMP_SPACING = 1.8   # trees planted in a merged canopy stand this many crown radii apart
PALETTE = 10          # leaf colours kept, so the scene needs only this many leaf materials
WOOD = (0.33, 0.29, 0.25)


def widest_single(height):
    """Largest crown radius one tree of this height is allowed before it counts as a clump."""
    return float(np.clip(0.42 * height + 0.6, 1.0, 8.5))


def split_clumps(trees, crowns, height, x0, y1, cell, rng):
    """Replace over-wide crowns by several trees spread over the same ground."""
    out = []
    for t in trees:
        limit = widest_single(t["height"])
        if t["radius"] <= limit:
            out.append(dict(t))
            continue
        rows, cols = np.nonzero(crowns == t["id"])
        # Dart throwing: accept points no closer together than a typical crown width.
        spacing = CLUMP_SPACING * limit
        order = rng.permutation(len(rows))
        taken = []
        for k in order:
            px, py = x0 + (cols[k] + 0.5) * cell, y1 - (rows[k] + 0.5) * cell
            if all((px - qx) ** 2 + (py - qy) ** 2 >= spacing**2 for qx, qy, _ in taken):
                taken.append((px, py, k))
        for px, py, k in taken:
            r0, r1 = max(rows[k] - 6, 0), rows[k] + 7
            c0, c1 = max(cols[k] - 6, 0), cols[k] + 7
            local = np.nanmax(np.where(crowns[r0:r1, c0:c1] == t["id"], height[r0:r1, c0:c1], np.nan))
            h = float(max(local, 0.6 * t["height"]))
            out.append({"id": t["id"], "x": float(px), "y": float(py), "height": h, "radius": float(min(limit, widest_single(h)) * rng.uniform(0.8, 1.0))})
    return out


def leaf_colours(trees, crowns, small_photo):
    """Typical sunlit leaf colour of each crown, sRGB 0..1, from the photo on the crown grid.

    Crowns whose colour is not a leaf's are left out, and the caller plants nothing for them.
    """
    ids = np.array(sorted({t["id"] for t in trees}))
    rgb = small_photo.astype(np.float32) / 255
    bright = rgb.max(-1)
    colours = {}
    slices = ndimage.find_objects(crowns, max_label=int(ids.max()))
    for i in ids:
        sl = slices[i - 1]
        if sl is None:
            continue
        sel = crowns[sl] == i
        px = rgb[sl][sel]
        b = bright[sl][sel]
        # The brighter half: the half in its own shade says more about the sun than the leaf.
        colour = np.median(px[b >= np.median(b)], axis=0)
        if is_foliage(colour):
            colours[int(i)] = colour
    return colours


def is_foliage(rgb):
    """Whether a colour could be leaves: green through yellow to brown, never grey or blue.

    Tall things that are not trees get through the height test: a tank, a light mast, a
    parked truck. Their colour gives them away.
    """
    r, g, b = (float(v) for v in rgb)
    top = max(r, g, b)
    return top > 0.08 and (top - min(r, g, b)) / top > 0.15 and b < 0.9 * max(r, g)


def palette(colours, count, rng):
    """Reduce a dict of colours to count representatives. Returns (id -> index, list of colours)."""
    ids = list(colours)
    data = np.array([colours[i] for i in ids])
    centres = data[rng.choice(len(data), size=min(count, len(data)), replace=False)]
    for _ in range(25):
        nearest = np.argmin(((data[:, None] - centres[None]) ** 2).sum(-1), axis=1)
        for k in range(len(centres)):
            if (nearest == k).any():
                centres[k] = data[nearest == k].mean(0)
    return dict(zip(ids, nearest.tolist())), centres


def kind_of(t, near_water):
    if t["height"] < 3.5:
        return "shrub"
    slender = t["height"] / (2 * t["radius"])
    if near_water and t["height"] > 5 and slender < 1.3:
        return "willow"
    return "upright" if slender > 2.0 else "broadleaf"


CLEAR = 2.0     # metres a trunk is kept from pavement and from walls


def off_the_road(trees, keep_out, x0, y1, cell):
    """Move trunks that landed on pavement or in a building to the nearest ground that is clear.

    A tree's position is the top of its crown, and a crown that leans out over a road has its
    top over the road. The trunk cannot be there, so it is stepped back to the verge. In place.
    """
    blocked = ndimage.distance_transform_edt(~keep_out) * cell < CLEAR
    _, (near_r, near_c) = ndimage.distance_transform_edt(blocked, return_indices=True)
    moved = 0
    for t in trees:
        row = int(np.clip((y1 - t["y"]) / cell, 0, blocked.shape[0] - 1))
        col = int(np.clip((t["x"] - x0) / cell, 0, blocked.shape[1] - 1))
        if blocked[row, col]:
            t["x"], t["y"] = float(x0 + (near_c[row, col] + 0.5) * cell), float(y1 - (near_r[row, col] + 0.5) * cell)
            moved += 1
    return moved


OVERHANG = -0.3     # metres a crown may reach over pavement. Negative: it stops short of it
BACK = 4.0          # metres a trunk may be stepped back to make room for its crown


def clear_of_road(trees, paved, keep_out, x0, y1, cell):
    """Keep every crown off the pavement. In place.

    From above, a crown beside a road looks as if it reaches over it, and often it does. Drawn
    that way, branches and leaves hang into the lane at windscreen height, because a generated
    tree does not know to grow up and over a road the way a real one is pruned to. So a tree
    whose crown would cross the pavement's edge is first stepped back from the road, up to BACK,
    and whatever still crosses is taken off its width.
    """
    away = ndimage.distance_transform_edt(~paved) * cell
    gy, gx = np.gradient(away, cell)        # rows run south, so +row is -y
    blocked = ndimage.distance_transform_edt(~keep_out) * cell < CLEAR
    h, w = paved.shape
    moved = narrowed = 0
    for t in trees:
        width = min(t["radius"], widest_single(t["height"]))
        start = (t["x"], t["y"])
        for _ in range(int(BACK / 0.25)):
            row = int(np.clip((y1 - t["y"]) / cell, 0, h - 1))
            col = int(np.clip((t["x"] - x0) / cell, 0, w - 1))
            if away[row, col] + OVERHANG >= width:
                break
            step = np.array([gx[row, col], -gy[row, col]])
            size = np.hypot(*step)
            if size < 0.3:      # on the ridge between two roads: nowhere better to go
                break
            nx, ny = float(t["x"] + 0.25 * step[0] / size), float(t["y"] + 0.25 * step[1] / size)
            if blocked[int(np.clip((y1 - ny) / cell, 0, h - 1)), int(np.clip((nx - x0) / cell, 0, w - 1))]:
                break           # a building is in the way
            t["x"], t["y"] = nx, ny
        moved += (t["x"], t["y"]) != start
        row = int(np.clip((y1 - t["y"]) / cell, 0, h - 1))
        col = int(np.clip((t["x"] - x0) / cell, 0, w - 1))
        room = away[row, col] + OVERHANG
        if room < width:
            t["radius"] = float(max(room, 0.6))
            narrowed += 1
    return moved, narrowed


def plan(trees, crowns, height, small_photo, water, paved, buildings, grid, ref, seed=0, not_trees=None, log=print):
    """Return (library entries to generate, placements).

    grid is (x0, y1, cell) of the crown grid. water, paved and buildings are boolean masks on
    it. A library entry is (name, kind, height, radius, seed, triangles). A placement is a dict
    ready for the manifest, with "model" naming its library entry.

    not_trees is a mask of ground where a tree the survey found is not one: it is the thing
    that stands there, a tank or a yard of plant. Such a tree is left out. Every other tree
    comes out exactly as it would without the mask, name and all.
    """
    rng = np.random.default_rng(seed)
    x0, y1, cell = grid
    colours = leaf_colours(trees, crowns, small_photo)
    trees = [t for t in split_clumps(trees, crowns, height, x0, y1, cell, rng) if t["id"] in colours]
    if not_trees is not None:
        for t in trees:      # judged where the survey found it, before anything moves it
            t["gone"] = bool(not_trees[int(np.clip((y1 - t["y"]) / cell, 0, crowns.shape[0] - 1)), int(np.clip((t["x"] - x0) / cell, 0, crowns.shape[1] - 1))])
        gone = [t for t in trees if t["gone"]]
        log(f"  {len(gone)} of the survey's trees stand in a small structure or under a roof's edge, and are left out: "
            + ", ".join(f"({t['x']:.0f}, {t['y']:.0f})" for t in gone))
    index, centres = palette(colours, PALETTE, rng)
    off_the_road(trees, paved | buildings, x0, y1, cell)
    clear_of_road(trees, paved, paved | buildings, x0, y1, cell)
    by_water = ndimage.binary_dilation(water, iterations=int(round(12 / cell)))
    from_pavement = ndimage.distance_transform_edt(~paved) * cell

    library = []
    for n, (kind, h, r, triangles, _) in enumerate(LIBRARY):
        for s in range(SEEDS):
            library.append((f"{kind}_{n}_{s}", kind, h, r, 1000 * n + s, triangles))
            library.append((f"{kind}_{n}_{s}_light", kind, h, r, 1000 * n + s, int(triangles * LIGHT)))
    wood = renderer.colour_for_renderer(WOOD)
    placements = []
    for n, t in enumerate(trees):
        row = int(np.clip((y1 - t["y"]) / cell, 0, crowns.shape[0] - 1))
        col = int(np.clip((t["x"] - x0) / cell, 0, crowns.shape[1] - 1))
        kind = kind_of(t, bool(by_water[row, col]))
        radius = float(np.clip(t["radius"], 0.5, widest_single(t["height"])))
        # Among this kind's models for a tree this tall, the proportion nearest its own.
        fits = [m for m, (k, _, _, _, tallest) in enumerate(LIBRARY) if k == kind and t["height"] <= tallest]
        light = min(LIBRARY[m][3] for m in fits)
        fits = [m for m in fits if LIBRARY[m][3] == light]
        which = min(fits, key=lambda m: abs(np.log((t["height"] / radius) / (LIBRARY[m][1] / LIBRARY[m][2]))))
        h, r = LIBRARY[which][1], LIBRARY[which][2]
        yaw = rng.uniform(0, 2 * np.pi)
        variant = int(rng.integers(SEEDS))
        if t.get("gone"):
            continue
        leaf = renderer.colour_for_renderer(centres[index[t["id"]]])
        # A crown reaches toward the road, so distance is measured from its edge, not its trunk.
        near = from_pavement[row, col] - radius < NEAR
        placements.append({
            "model": f"{kind}_{which}_{variant}" + ("" if near else "_light"), "group": "Trees", "name": f"tree_{n:04d}", "kind": kind,
            "pos": [round(t["x"], 2), round(t["y"], 2), round(float(ref.elevation(t["x"], t["y"])), 2)],
            "rot": [round(float(np.cos(yaw / 2)), 5), 0.0, 0.0, round(float(np.sin(yaw / 2)), 5)],
            "scale": [round(radius / r, 3), round(radius / r, 3), round(t["height"] / h, 3)],
            "height": round(t["height"], 1), "crown_radius": round(radius, 1),
            "colours": {"leaves": [round(c, 3) for c in leaf], "wood": [round(c, 3) for c in wood]},
        })
    return library, placements
