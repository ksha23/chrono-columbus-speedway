#!/usr/bin/env python3
"""Stand grass up along the pavement's edges. For the sensor copy only.

    python -I verge.py SENSOR_SCENE      (a copy made by sensor_scene.py: it is changed)

The lawn is a flat picture and it meets the pavement in a line as sharp as a ruler's. From
above nobody can tell. From a car's camera 1.3 m up it is the plainest sign that the world is
drawn: a real verge has height, and its edge against the road is ragged.

So tufts of blades are stood on the land beside the pavement (vergemodel.py): a close row
right along the edge, whose blades lean a few centimetres out over it, and behind that a
scatter that is thick in the first metre or two and gone by REACH.

THE TUFTS ARE SYNTHESIZED. Where they stand, how tall they are and which way they lean come
from hashed noise (worldnoise.py). The site's grass was not measured, only photographed from
above. What is taken from the scene: a tuft stands only where the land's own triangles are
under it, at their height, only where the ground's picture has the colour of grass, not under
a building or a tank, and it is given the colour of the picture where it stands.

The lawns here are mown, so most tufts are 5 to 10 cm, with patches left rougher. Bare soil
that is yellowish cannot be told from dry lawn by colour (groundgrain.grass), and gets tufts
in its own colour. Nothing is known here of trees, rocks or parked cars: a tuft may stand
against a trunk or under a bumper.

There are fewer tufts than there could be, for two reasons. Each one shades the ground, which
the photo's lawn already is as dark as: thicker than this and the verge shows from above as a
darker band. And Chrono::Sensor's Metal renderer, as it is, opens a triangle's texture file
once for every triangle, so these 350,000 add about 5 s to its start.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, spatial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groundgrain  # noqa: E402
import groundmesh  # noqa: E402
import vergemodel  # noqa: E402
import worldnoise  # noqa: E402

REACH = 8.0           # metres from the pavement where the last tufts stand
FADE = 1.7            # metres over which the scatter thins to about a third
THIN = 2.5            # metres before REACH over which what is left goes to nothing
CELL = 0.32           # metres: the scatter has at most one tuft in each square of this size
ROW = (0.10, 0.22)    # metres between neighbours in the row along the edge
GAPS = (1.2, 0.5)     # the row has thinner stretches: metres along, and the share of its tufts that stay in the thinnest
BACK = (0.015, 0.07)  # metres from the edge to a tuft of the row
CLEAR = 0.01          # metres a card's foot keeps from the pavement
OVERHANG = 0.04       # metres a blade's tip may lean out over the pavement. The edge line's paint starts at 0.25
MOWN = 0.068          # metres: the common height of a tuft. Heights spread about it by HEIGHTS
HEIGHTS = 0.30
ROUGH = (6.0, 1.0, 1.0)     # rough patches: metres across, how rare (in spreads of the noise), how much taller at most
TALLEST = 0.20
SUNK = 0.015          # metres a card's foot is below the ground, so none shows a gap on a slope
TILT = 0.06           # how far a tuft's normal leans from the ground's, as a spread: some catch more sun
COLOURS = 24          # colours the tufts come in: rows of the sheet
AROUND = 0.3          # metres of ground picture a tuft's colour is the average of
# The picture pales toward the pavement over its last 30 cm, where the photo is a blend of the two:
# over the whole site it has a quarter more light at 3 cm from the edge than at 1 m, and two thirds
# more blue. The grass at the edge is the lawn's grass, so a tuft nearer than this takes its colour
# from this far in.
SETTLED = 0.4
COARSEST = 0.15       # metres a pixel: a picture coarser than this (the surround's) cannot say what is grass
SEED = 1511


def candidates(edges):
    """Where tufts could stand: (x, y, a number of their own, whether in the row along the edge)."""
    along = edges[:, 1] - edges[:, 0]
    length = np.linalg.norm(along, axis=1)
    along = along / np.maximum(length, 1e-9)[:, None]
    middle = edges.mean(1)
    own = worldnoise.bits(np.round(middle[:, 0] * 1000), np.round(middle[:, 1] * 1000), SEED) >> np.uint64(34)
    most = int(np.ceil(length.max() / ROW[0])) + 1
    h = worldnoise.bits(np.arange(most)[None, :], own[:, None], SEED + 1)
    at = np.cumsum(ROW[0] + (ROW[1] - ROW[0]) * worldnoise.uniform(h, 0), axis=1) - ROW[1] * worldnoise.uniform(h[:, :1], 1)
    back = BACK[0] + (BACK[1] - BACK[0]) * worldnoise.uniform(h, 2) ** 2
    use = (at > 0) & (at < length[:, None])
    e = np.nonzero(use)[0]
    # The pavement lies to the left of its edge, so the land is to the right.
    out = np.stack([along[e, 1], -along[e, 0]], 1)
    row = edges[e, 0] + along[e] * at[use][:, None] + out * back[use][:, None]

    low, high = edges.reshape(-1, 2).min(0) - REACH, edges.reshape(-1, 2).max(0) + REACH
    ix = np.arange(int(np.floor(low[0] / CELL)), int(np.ceil(high[0] / CELL)))
    iy = np.arange(int(np.floor(low[1] / CELL)), int(np.ceil(high[1] / CELL)))
    g = worldnoise.bits(ix[None, :], iy[:, None], SEED + 2)
    fx = (ix[None, :] + worldnoise.uniform(g, 0)) * CELL
    fy = (iy[:, None] + worldnoise.uniform(g, 1)) * CELL
    x = np.concatenate([row[:, 0], fx.reshape(-1)])
    y = np.concatenate([row[:, 1], fy.reshape(-1)])
    number = np.concatenate([h[use], g.reshape(-1)]) >> np.uint64(34)
    return x, y, number.astype(np.int64), np.arange(len(x)) < len(row)


def covered(doc, x, y, margin=0.3):
    """Whether points are under a building or a small structure, by what the manifest says each covers."""
    inside = np.zeros(len(x), bool)
    for inst in doc["instances"]:
        if inst.get("group") == "Buildings" and "plan" in inst:
            (cx, cy), (ux, uy) = inst["centre"], inst["axis"]
            along, across = (x - cx) * ux + (y - cy) * uy, -(x - cx) * uy + (y - cy) * ux
            for u0, u1, low, high in inst["plan"]:
                inside |= (along > u0 - margin) & (along < u1 + margin) & (across > low - margin) & (across < high + margin)
        elif inst.get("group") == "Props" and "outline" in inst:
            ring = np.array(inst["outline"])
            ahead = np.roll(ring, -1, axis=0) - ring
            ahead /= np.maximum(np.linalg.norm(ahead, axis=1, keepdims=True), 1e-9)
            # An outline is a convex ring: a point inside is on the same side of every edge.
            left = ahead[None, :, 0] * (y[:, None] - ring[None, :, 1]) - ahead[None, :, 1] * (x[:, None] - ring[None, :, 0])
            inside |= (left > -margin).all(1) | (left < margin).all(1)
    return inside


def ground_colour(doc, out, x, y):
    """The ground picture's colour at points, averaged over AROUND, as light: (n, 3). NaN where no photo tile is."""
    colour = np.full((len(x), 3), np.nan, np.float32)
    group = {inst["asset"]: inst.get("group") for inst in doc["instances"]}
    for n, asset in enumerate(doc["assets"]):
        part = asset["parts"][0]
        if group.get(n) != "Terrain" or len(asset["parts"]) != 1 or not part.get("texture"):
            continue
        lies = groundmesh.frame(*groundmesh.read(os.path.join(out, part["mesh"])))
        if lies is None:
            continue
        x0, y0, wide, tall = lies
        here = np.nonzero(np.isnan(colour[:, 0]) & (x >= x0) & (x < x0 + wide) & (y >= y0) & (y < y0 + tall))[0]
        if not len(here):
            continue
        picture = Image.open(os.path.join(out, part.get("photo", part["texture"]).replace("<level>", groundgrain.LEVEL))).convert("RGB")
        if wide / picture.width > COARSEST:
            continue
        k = max(1, int(round(AROUND / 3 / (wide / picture.width))))
        light = (np.asarray(picture, np.float32) / 255.0) ** vergemodel.GAMMA
        rows, cols = light.shape[0] // k, light.shape[1] // k
        light = light[:rows * k, :cols * k].reshape(rows, k, cols, k, 3).mean((1, 3))
        light = ndimage.uniform_filter(light, size=(3, 3, 1), mode="nearest")
        col = np.clip(((x[here] - x0) / wide * picture.width / k).astype(int), 0, cols - 1)
        row = np.clip(((y0 + tall - y[here]) / tall * picture.height / k).astype(int), 0, rows - 1)
        colour[here] = light[row, col]
    return colour


def classes(colour, count):
    """Sort colours (light) into count kinds: (the kind of each, each kind's colour). The same every run."""
    seen = colour ** (1 / vergemodel.GAMMA)                   # as the eye spaces them
    order = np.argsort(seen @ np.array([0.3, 0.6, 0.1], np.float32), kind="stable")
    centre = np.stack([seen[part].mean(0) for part in np.array_split(order, count)])
    for _ in range(12):
        kind = ((seen[:, None, :] - centre[None]) ** 2).sum(-1).argmin(1)
        centre = np.stack([seen[kind == k].mean(0) if (kind == k).any() else centre[k] for k in range(count)])
    kind = ((seen[:, None, :] - centre[None]) ** 2).sum(-1).argmin(1)
    return kind, np.stack([colour[kind == k].mean(0) if (kind == k).any() else centre[k] ** vergemodel.GAMMA for k in range(count)])


def apply(doc, out, log=print):
    """Add tufts of grass beside the pavement to the copy at out, and to its manifest doc as one asset in group Terrain."""
    if any(asset["name"] == "verge" for asset in doc["assets"]):
        log("this copy has a verge already")
        return
    road_v, _, road_f, _ = groundmesh.read(os.path.join(out, doc["collision"]["road"]))
    land_v, _, land_f, _ = groundmesh.read(os.path.join(out, doc["collision"]["terrain"]))
    edges = groundmesh.outline(road_v, road_f)
    land = groundmesh.Surface(land_v, land_f)
    # The edge as points 1 cm apart: the nearest of them says how far the pavement is, and which way.
    pieces = np.maximum(np.ceil(np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1) / 0.01).astype(int), 1)
    t = (np.arange(pieces.sum()) - np.repeat(np.cumsum(pieces) - pieces, pieces) + 0.5) / np.repeat(pieces, pieces)
    marks = np.repeat(edges[:, 0], pieces, axis=0) + (np.repeat(edges[:, 1] - edges[:, 0], pieces, axis=0)) * t[:, None]
    tree = spatial.cKDTree(marks)

    x, y, number, row = candidates(edges)
    away, nearest = tree.query(np.stack([x, y], 1), distance_upper_bound=REACH)
    chance = np.exp(-away / FADE) * np.clip((REACH - away) / THIN, 0, 1) * (away > BACK[1])
    chance[row] = np.clip(0.85 + 0.35 * worldnoise.smooth_at(x[row], y[row], GAPS[0], SEED + 6), GAPS[1], 1.0)
    keep = np.isfinite(away) & (worldnoise.uniform(worldnoise.bits(number, 0, SEED + 3)) < chance)
    x, y, number, row, away, nearest = (a[keep] for a in (x, y, number, row, away, nearest))
    # Only on land: where one of the land's own triangles is under the tuft. And not under a roof.
    _, under = land.under(x, y)
    keep = (under >= 0) & ~covered(doc, x, y)
    x, y, number, row, away, nearest, under = (a[keep] for a in (x, y, number, row, away, nearest, under))
    # Only on grass: the greener or more like straw the picture is at its foot, the likelier a tuft stays.
    inland = (np.stack([x, y], 1) - marks[nearest]) / np.maximum(away, 1e-6)[:, None] * np.maximum(0.0, SETTLED - away)[:, None]
    colour = ground_colour(doc, out, x + inland[:, 0], y + inland[:, 1])
    seen = np.nan_to_num(colour) ** (1 / vergemodel.GAMMA)
    keep = ~np.isnan(colour[:, 0]) & (worldnoise.uniform(worldnoise.bits(number, 1, SEED + 3)) < groundgrain.grass(seen[:, 0], seen[:, 1], seen[:, 2]))
    x, y, number, row, away, nearest, under, colour = (a[keep] for a in (x, y, number, row, away, nearest, under, colour))
    toward = (marks[nearest] - np.stack([x, y], 1)) / np.maximum(away, 1e-6)[:, None]     # unit, from the tuft to the pavement

    def draw(what, k=0):
        return worldnoise.uniform(worldnoise.bits(number, what, SEED + 4), k)

    rough = np.clip(worldnoise.smooth_at(x, y, ROUGH[0], SEED + 5) - ROUGH[1], 0, 1) * ROUGH[2]
    tall = MOWN * np.exp(HEIGHTS * worldnoise.normal(worldnoise.bits(number, 2, SEED + 4))) * (1 + rough)
    tall = np.clip(tall, 0.04, TALLEST)
    kind, shades = classes(colour, COLOURS)
    normal = land.normals(under)
    normal[:, :2] += TILT * np.stack([worldnoise.normal(worldnoise.bits(number, 3, SEED + 4), k) for k in (0, 1)], 1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)

    cards = []
    for c in range(2):
        # The first card of a row tuft runs along the edge and the second across it. In the scatter they turn any way.
        turn = np.where(row, np.arctan2(toward[:, 1], toward[:, 0]) + np.pi / 2 + (draw(4 + c) - 0.5) * 0.7, 2 * np.pi * draw(4)) + c * (np.pi / 2 + (draw(6) - 0.5) * 0.9)
        way = np.stack([np.cos(turn), np.sin(turn)], 1)
        half = np.where(row, (0.10 + 0.07 * draw(7 + c)) * (1.0 - 0.45 * c), 0.5 * (0.22 + 0.16 * draw(7 + c))) * np.clip(np.sqrt(tall / MOWN), 0.8, 1.5)
        # A card that would reach the pavement is moved back from it.
        back = np.maximum(0.0, half * np.abs((way * toward).sum(1)) + CLEAR + 0.005 - away)
        middle = np.stack([x, y], 1) - toward * back[:, None]
        lean = np.where(row, 0.15 + 0.45 * draw(9 + c), 0.45 * draw(9 + c) * np.cos(2 * np.pi * draw(11 + c))) * tall
        lean = np.minimum(lean, away + back - half * np.abs((way * toward).sum(1)) + OVERHANG)[:, None] * toward
        lean += (np.stack([-toward[:, 1], toward[:, 0]], 1) * ((draw(13 + c) - 0.5) * 0.5 * tall)[:, None])
        ends = [middle - way * half[:, None], middle + way * half[:, None]]
        heights = [land.under(p[:, 0], p[:, 1])[0] for p in ends]
        # A card stands only if both ends of its foot are on land and clear of the pavement, whichever edge is nearest.
        good = ~np.isnan(heights[0]) & ~np.isnan(heights[1]) & (tree.query(ends[0])[0] > CLEAR + 0.002) & (tree.query(ends[1])[0] > CLEAR + 0.002)
        # And it leans only if neither top corner then hangs further over the pavement than it may.
        for p in ends:
            lean[np.isnan(land.under(p[:, 0] + lean[:, 0], p[:, 1] + lean[:, 1])[0]) & (tree.query(p + lean)[0] > OVERHANG)] = 0.0
        foot = [np.column_stack([p, z - SUNK])[good] for p, z in zip(ends, heights)]
        top = [f + np.column_stack([lean[good], tall[good] + SUNK]) for f in foot]
        picture = kind * vergemodel.SHAPES + np.minimum((draw(15 + c) * vergemodel.SHAPES).astype(int), vergemodel.SHAPES - 1)
        cards.append((foot[0], foot[1], top[0], top[1], picture[good], draw(16 + c)[good] < 0.5, np.nonzero(good)[0]))
    foot_a, foot_b, top_a, top_b, picture, flipped, tuft = (np.concatenate([card[k] for card in cards]) for k in range(7))

    os.makedirs(os.path.join(out, "verge"), exist_ok=True)
    Image.fromarray(vergemodel.sheet(shades)).save(os.path.join(out, "verge", "blades.png.new"), format="PNG", optimize=True)
    os.replace(os.path.join(out, "verge", "blades.png.new"), os.path.join(out, "verge", "blades.png"))
    triangles = vergemodel.write(os.path.join(out, "verge", "verge.obj.new"), foot_a, foot_b, top_a, top_b, picture, flipped, normal, tuft)
    os.replace(os.path.join(out, "verge", "verge.obj.new"), os.path.join(out, "verge", "verge.obj"))
    doc["assets"].append({"name": "verge", "parts": [{"name": "blades", "mesh": "verge/verge.obj", "texture": "verge/blades.png", "colour": [1.0, 1.0, 1.0],
                                                      "ks": [0.0, 0.0, 0.0], "ns": 1.0, "roughness_value": 1.0, "double_sided": True}]})
    stand = np.unique(tuft)
    doc["instances"].append({"asset": len(doc["assets"]) - 1, "group": "Terrain", "name": "verge", "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0],
                             "scale": [1.0, 1.0, 1.0], "tufts": int(len(stand)), "triangles": int(triangles)})
    doc.setdefault("synthesized", {})["verge"] = (f"{len(stand)} tufts of grass within {REACH:g} m of the pavement, made by tools/verge.py: "
                                                  "placed and shaped by noise, coloured from the ground picture, not measured on the site")
    log(f"{len(stand)} tufts ({int(row[stand].sum())} along {np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1).sum():.0f} m of pavement edge, {int((~row[stand]).sum())} scattered behind),"
        f" {triangles} triangles, {tall[stand].mean() * 100:.1f} cm tall on average and {tall[stand].max() * 100:.0f} cm at most, in {COLOURS} colours on one sheet")
    groundmesh.count_pictures(doc, log, "verge")           # one picture is added: the sheet of blades


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(os.path.join(sys.argv[1], "speedway_scene.json")) as f:
        manifest = json.load(f)
    apply(manifest, sys.argv[1])
    with open(os.path.join(sys.argv[1], "speedway_scene.json.new"), "w") as f:
        json.dump(manifest, f)
    os.replace(os.path.join(sys.argv[1], "speedway_scene.json.new"), os.path.join(sys.argv[1], "speedway_scene.json"))
