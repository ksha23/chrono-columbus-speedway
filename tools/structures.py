"""The small structures as what they are: tanks, a shed, a deck with its ramps, a yard of plant.

The block finder (blocks.py) says where each small man-made thing stands, which way it lies,
how big it is and what colour. It cannot say what the thing is: a white lump four metres long
is a block to it whether it is a propane tank or a freezer. There are a handful of them on
this site and no two alike, so that one fact is read by hand, once, and kept in
structures.json, together with whatever else of the thing the finder could not measure: the
diameter of a tank whose block includes its frame, which end of a shed is the high one,
where the bollards stand.

Some things the finder does not find at all, because they are not one colourless lump: a deck
of ramps and rails, plant behind screen walls, pipes in the grass. Their entries carry their
own place and are built whether or not a block was found there.

A found block that no entry names stays a plain block of its size and colour, as before.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

import deckmodel
import propmodel
import railmodel
import tankmodel
import tubes
import yardmodel
from blocks import write_unit
from propmodel import Site

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "structures.json")
NEAR = 3.0             # metres within which a found block is the thing an entry names
METAL = {"steel"}      # parts that are bare galvanised tube


def load(path=FILE):
    with open(path) as f:
        return json.load(f)


def _site(entry, block, ground):
    """Where a thing stands and which way it lies: its entry's word if it has one, or else its block's."""
    if "at" in entry:
        along = np.asarray(entry.get("along", (1.0, 0.0)), float)
        return Site(entry["at"], along, (-along[1], along[0]), ground)
    return Site.heading(block["x"], block["y"], block["yaw"], ground)


def _size(entry, block, key):
    """A measurement of the thing: its entry's if it gives one, or else its block's."""
    if key not in entry and block is None:
        raise ValueError(f"{entry['name']}: no block was found for it, and its entry gives no {key}")
    return entry[key] if key in entry else block[key]


def _pad(entry, site, ground):
    """The concrete an entry says its thing stands on: (slab, the level of its top)."""
    spec = entry["pad"]
    return propmodel.pad(Site(spec.get("at", site.origin), site.along, site.out, ground), spec["length"], spec["width"], spec.get("kerb", 0.0))


def _tank(entry, block, ground):
    site = _site(entry, block, ground)
    parts = tankmodel.tank(site, entry["length"], entry["diameter"], _size(entry, block, "height"))
    if entry.get("guards"):
        parts["steel"] = tubes.merge([tankmodel.guard(a, b, entry.get("guard_height", 0.8), ground) for a, b in entry["guards"]])
    return parts, {"shell": block["colour"]} if block else {}, site.origin


def _fuel_tank(entry, block, ground):
    """A skid tank in a kerbed concrete basin, with bollards on the sides a vehicle can come from."""
    site = _site(entry, block, ground)
    slab, level = _pad(entry, site, ground)
    top = max(site.ground(0.0, 0.0) + _size(entry, block, "height") - level, entry["diameter"] + 0.15)
    parts = tankmodel.tank(site, entry["length"], entry["diameter"], top, head=0.05, stand=level, skid=True)
    parts["concrete"] = slab
    parts["cap"] = tankmodel.fuel_fittings(site, entry["length"], entry["diameter"], top, level)
    if entry.get("bollards"):
        parts["bollard"] = tankmodel.bollards(entry["bollards"], entry.get("bollard_height", 1.1), ground)
    return parts, {"shell": block["colour"]} if block else {}, site.origin


def _shed(entry, block, ground):
    site = _site(entry, block, ground)
    length, width, high = (_size(entry, block, key) for key in ("length", "width", "height"))
    # Of the four ways the shed's sides face, the one nearest the azimuth read for it is the high end.
    toward = np.radians(entry.get("high_end", 0.0))
    want = np.array([np.sin(toward), np.cos(toward)])
    faces = [site.along, site.out, -site.along, -site.out]
    turn = int(np.argmax([want @ f for f in faces]))
    if turn % 2:
        length, width = width, length
    site = Site(site.origin, faces[turn], faces[(turn + 1) % 4], ground)
    parts = propmodel.shed(site, length, width, high, high - entry.get("roof_drop", 0.0))
    return parts, {"roof": block["colour"]} if block else {}, site.origin


def _cabinet(entry, block, ground):
    site = _site(entry, block, ground)
    length, width, height = (_size(entry, block, key) for key in ("length", "width", "height"))
    parts = {}
    base = site.span((-length / 2, length / 2), (-width / 2, width / 2))[1]
    if "pad" in entry:
        parts["concrete"], base = _pad(entry, site, ground)
    parts["body"] = propmodel.cabinet(site, length, width, height, base)
    if entry.get("bollards"):
        parts["bollard"] = tankmodel.bollards(entry["bollards"], entry.get("bollard_height", 1.1), ground)
    seen = entry.get("colour") or (block or {}).get("colour")
    return parts, {"body": seen} if seen else {}, site.origin


def _against_wall(model):
    def make(entry, block, ground):
        site = Site(entry["origin"], entry["along"], entry["out"], ground)
        return model.make(site, entry), {}, site.origin
    return make


def _pipes(entry, block, ground):
    parts = propmodel.pipes(entry["pipes"], entry["diameter"], entry["rest"], ground)
    return parts, {}, np.mean([p for run in entry["pipes"] for p in run], axis=0)


# kind -> (how to build it, its parts' colours as seen, what an entry must have to stand without a block)
KINDS = {
    "tank": (_tank, tankmodel.COLOURS, "at"),
    "fuel_tank": (_fuel_tank, tankmodel.COLOURS, "at"),
    "shed": (_shed, propmodel.COLOURS, "at"),
    "cabinet": (_cabinet, {**tankmodel.COLOURS, **propmodel.COLOURS}, "at"),
    "deck": (_against_wall(deckmodel), deckmodel.COLOURS, "origin"),
    "hvac_yard": (_against_wall(yardmodel), yardmodel.COLOURS, "origin"),
    "pipes": (_pipes, propmodel.COLOURS, "pipes"),
}


def _named(entries, blocks):
    """Which found block each entry names: {index of a block: index of its entry}."""
    named = {}
    for k, entry in enumerate(entries):
        if "near" in entry:
            reach = [(np.hypot(b["x"] - entry["near"][0], b["y"] - entry["near"][1]), n) for n, b in enumerate(blocks) if n not in named]
            if reach and min(reach)[0] <= NEAR:
                named[min(reach)[1]] = k
    return named


def _made(blocks, ground, log=print):
    """Every structure there is to build, in the order of the entries: (entry, its block or None, parts, colours as seen, centre)."""
    entries = load()["structures"]
    block_of = {k: blocks[n] for n, k in _named(entries, blocks).items()}
    for k, entry in enumerate(entries):
        make, seen, own = KINDS[entry["kind"]]
        block = block_of.get(k)
        if block is None and own not in entry:
            log(f"  {entry['name']}: no block found within {NEAR:g} m of ({entry['near'][0]:.1f}, {entry['near'][1]:.1f}), left out")
            continue
        parts, found, centre = make(entry, block, ground)
        yield entry, block, parts, {**seen, **found}, centre


def cover(blocks, x0, y1, res, shape, margin=0.0, whole=False):
    """The ground the structures stand on, as a boolean map of cells res metres across with its corner at (x0, y1).

    Every structure is counted, found by the block finder or not, as its own shape seen from
    above and margin metres round it: what has to be painted out of the photo for the models
    to stand in for. With whole, a structure is all the ground inside its outline, the open
    floor of a yard between its walls too: what a tree cannot be growing in. Nothing is written.
    """
    out = np.zeros(shape, bool)
    reach = int(np.ceil(margin / res))
    yy, xx = np.ogrid[-reach:reach + 1, -reach:reach + 1]
    round_it = (xx * xx + yy * yy) * res * res <= margin * margin
    for _, _, parts, _, _ in _made(blocks, lambda xs, ys: np.zeros(np.shape(xs)), log=lambda *_: None):
        triangles = np.concatenate([v[f][:, :, :2] for v, f in parts.values()])
        c0, c1 = int((triangles[..., 0].min() - x0) / res) - reach - 1, int((triangles[..., 0].max() - x0) / res) + reach + 2
        r0, r1 = int((y1 - triangles[..., 1].max()) / res) - reach - 1, int((y1 - triangles[..., 1].min()) / res) + reach + 2
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, shape[1]), min(r1, shape[0])
        if c1 <= c0 or r1 <= r0:
            continue
        picture = Image.new("L", (c1 - c0, r1 - r0), 0)
        draw = ImageDraw.Draw(picture)
        if whole:
            flat = triangles.reshape(-1, 2)
            triangles = [flat[ConvexHull(flat).vertices]]
        for corners in triangles:
            draw.polygon([((x - x0) / res - c0, (y1 - y) / res - r0) for x, y in corners], fill=1, outline=1)
        out[r0:r1, c0:c1] |= ndimage.binary_dilation(np.asarray(picture, dtype=bool), structure=round_it)
    return out


def build(blocks, ground, scene_dir, colour, log=print):
    """Write the small structures' meshes under scene_dir. Returns (assets, instances) for the manifest.

    blocks is what blocks.find found. ground(xs, ys) gives the ground's height. colour turns a
    colour as seen into the one the renderer needs. An instance's "asset" is its index in the
    returned assets: the caller adds its own offset. Each structure's instance carries its
    "outline", the ground it covers as a ring of points.
    """
    named = _named(load()["structures"], blocks)
    os.makedirs(os.path.join(scene_dir, "props"), exist_ok=True)
    assets, instances, count = [], [], {}
    for entry, block, parts, seen, centre in _made(blocks, ground, log):
        triangles = 0
        for part, (v, f) in parts.items():
            if tubes.volume(v, f) <= 0:
                raise ValueError(f"{entry['name']}: part {part} is not closed and facing outward")
            triangles += railmodel.write_obj(os.path.join(scene_dir, "props", f"{entry['name']}_{part}.obj"), v, f)
        # The ground it covers, as the hull of all of it seen from above.
        plan = np.concatenate([v[:, :2] for v, _ in parts.values()])
        outline = plan[ConvexHull(plan).vertices]
        assets.append({"name": entry["name"], "parts": [
            {"name": part, "mesh": f"props/{entry['name']}_{part}.obj", "colour": [round(float(c), 3) for c in colour(seen[part])],
             "roughness_value": 0.5 if part in METAL else 0.8, "metallic_value": 0.4 if part in METAL else 0.0} for part in parts]})
        n = count.get(entry["kind"], 0)
        count[entry["kind"]] = n + 1
        instances.append({"asset": len(assets) - 1, "group": "Props", "name": f"{entry['kind']}_{n:02d}", "kind": entry["kind"],
                          "centre": [round(float(centre[0]), 2), round(float(centre[1]), 2)],
                          "outline": [[round(float(x), 2), round(float(y), 2)] for x, y in outline],
                          "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
        log(f"  {entry['name']}: {entry['kind']} at ({centre[0]:.1f}, {centre[1]:.1f}), {triangles} triangles" + ("" if block else ", no block"))

    # What nobody named is a block of the size and colour the scan gives it.
    plain = [n for n in range(len(blocks)) if n not in named]
    if plain:
        write_unit(os.path.join(scene_dir, "props", "block.obj"))
        assets.append({"name": "block", "parts": [{"name": "block", "mesh": "props/block.obj", "colour": [0.8, 0.8, 0.8], "roughness_value": 0.8}]})
        for n in plain:
            b = blocks[n]
            z = float(ground(np.array([b["x"]]), np.array([b["y"]]))[0])
            instances.append({"asset": len(assets) - 1, "group": "Props", "name": f"block_{n:02d}",
                              "pos": [round(b["x"], 2), round(b["y"], 2), round(z, 3)],
                              "rot": [round(float(np.cos(b["yaw"] / 2)), 5), 0.0, 0.0, round(float(np.sin(b["yaw"] / 2)), 5)],
                              "scale": [b["length"], b["width"], b["height"]],
                              "colours": {"block": [round(float(v), 3) for v in colour(b["colour"])]}})
        log(f"  {len(plain)} more left as plain blocks")
    return assets, instances
