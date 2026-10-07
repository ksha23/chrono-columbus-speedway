"""Give the trees of a sensor copy fine detail: sprays of small leaves, and bark.

    apply(doc, out, log=print, every_colour=True)

doc is the manifest of a copy made by sensor_scene.py, as a dict, and out is the copy's
directory. The copy's tree files are changed and doc is edited in place: the caller writes it.

NOTHING HERE IS MEASURED. The scan gives a tree its place, height, width and one colour. The
leaves, their shapes, the gaps between them, the bark and its furrows are synthesized
(leafatlas.py, barkmodel.py), so that a camera at car height sees foliage and wood and not
flat triangles. Whoever tests an algorithm on these trees is testing it on drawn ones.

What changes:

  Leaves. A leaf was one flat triangle. It becomes a spray of small leaves with gaps, by a
  picture with holes laid on the triangle. The triangle count stays. Each triangle is made
  larger about its own middle, since a spray fills only about half of it: FILL sets how much
  leaf shows afterwards against before. Which spray a triangle shows, and which way round,
  is drawn at random, so that neighbours differ. The normals stay the rounded ones the copy
  has: a leaf's corners keep theirs.

  Colour. The renderer ignores a part's colour once it has a picture, so a tree's leaf colour
  has to be in the picture. Every colour has its own sprays in the one picture, and a crown
  is written once per colour it comes in, each pointing at its colour's sprays. An asset is
  split the same way, and placements are moved to the asset of their colour. An asset keeps
  its name and file for its commonest colour. The others are named NAME_tN.

  Wood. Trunks and limbs get a bark picture and the coordinates that wrap it round them.

Two pictures are added, and no more, on purpose: Chrono::Sensor's Metal ray tracer takes 64
pictures in all and silently draws what is beyond them in plain colour. This scene's ground
and buildings use 61. apply says how many the copy ends up with.

The cost is in loading, not in drawing. A crown in five colours is five meshes to load, and
the renderer is slow to take in a triangle that has a picture. every_colour=False writes
each crown once, in its commonest colour, and costs a fifth of that.

Files of a copy may be hard links to the scene's own. None is ever opened for writing: each
is written beside itself and moved into place, which leaves the scene's file alone.

The same copy always gives the same files: every choice comes from fixed numbers.
"""
import copy
import os
import zlib
from collections import Counter

import numpy as np
from PIL import Image

import barkmodel
import leafatlas

LEAVES = "foliage.png"       # the two pictures, put beside the tree meshes
BARK = "bark.png"
FILL = 1.0                   # leaf showing after, over leaf showing before. At 1.0 as much sky shows through a crown as did
LIMIT = 64                   # pictures the ray tracer takes
SEED = 20261007


def _read(path):
    """An OBJ file's vertices, normals, and per face the indices of both."""
    v, n, fv, fn = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("vn "):
                n.append(line.split()[1:4])
            elif line.startswith("f "):
                corners = [corner.split("/") for corner in line.split()[1:4]]
                fv.append([c[0] for c in corners])
                fn.append([c[-1] for c in corners])
    return np.array(v, dtype=float), np.array(n, dtype=float), np.array(fv, dtype=int) - 1, np.array(fn, dtype=int) - 1


def _write(path, text):
    """Never in place: the file may be a hard link to the scene's own."""
    with open(path + ".new", "w") as f:
        f.write(text)
    os.replace(path + ".new", path)


def _save(path, picture):
    Image.fromarray(picture).save(path + ".new", format="PNG", optimize=True)
    os.replace(path + ".new", path)


def _lines(pattern, rows):
    return "".join(pattern % tuple(row) for row in rows)


def _crown(path, grow, kind):
    """A crown's leaves, each enlarged about its middle and given a spray: the OBJ file's text before and after its texture coordinates."""
    v, n, fv, fn = _read(path)
    corners, normals = v[fv], n[fn]                                   # (leaves, 3, 3): base, tip, base
    corners = corners.mean(1, keepdims=True) + grow * (corners - corners.mean(1, keepdims=True))
    rng = np.random.default_rng([SEED, zlib.crc32(os.path.basename(path).encode())])
    k = leafatlas.KINDS[leafatlas.kind_of(kind)]
    count = len(corners)
    spray, turn, flip = rng.integers(0, k["sprays"], count), rng.integers(0, k["turns"], count), rng.integers(0, 2, count)
    # Which corner of the spray each corner of the leaf takes: turned round, and mirrored or not.
    takes = (np.arange(3)[None] + turn[:, None]) % 3
    takes = np.where(flip[:, None] == 1, 2 - takes, takes)
    t = 3 * spray[:, None] + takes + 1
    i = 3 * np.arange(count)[:, None] + np.arange(3)[None] + 1
    faces = np.stack([i, t, i], 2).reshape(count, 9)
    head = _lines("v %.3f %.3f %.3f\n", corners.reshape(-1, 3))
    tail = _lines("vn %.3f %.3f %.3f\n", normals.reshape(-1, 3)) + _lines("f %d/%d/%d %d/%d/%d %d/%d/%d\n", faces)
    return head, tail, count


def _barked(path):
    """Write a tree's wood again with texture coordinates. False if it is not made of treegen's tubes."""
    v, n, fv, fn = _read(path)
    wrapped = barkmodel.wrap(v, fv)
    if wrapped is None:
        return False
    coords, ft = wrapped
    faces = np.stack([fv, ft, fn], 2).reshape(len(fv), 9) + 1
    _write(path, _lines("v %.4f %.4f %.4f\n", v) + _lines("vt %.4f %.4f\n", coords) + _lines("vn %.4f %.4f %.4f\n", n)
           + _lines("f %d/%d/%d %d/%d/%d %d/%d/%d\n", faces))
    return True


def pictures(doc):
    """How many different pictures the placed assets of a manifest use, at one level of ground detail."""
    used = {inst["asset"] for inst in doc["instances"]}
    return len({part["texture"] for a in used for part in doc["assets"][a]["parts"] if part.get("texture")})


def apply(doc, out, log=print, every_colour=True):
    """See the top of the file. With every_colour=False, trees of one shape are all the colour most of them have."""
    if "foliage" in doc:
        raise ValueError("this copy's trees have their detail already: start from a fresh copy")
    assets = doc["assets"]

    def part(asset, name):
        return next((p for p in asset["parts"] if p["name"] == name), None)

    # A tree is an asset with leaves drawn from both sides. Its colours are those its placements ask for.
    colours, kinds, bark, wants = {}, {}, Counter(), {}
    for nth, inst in enumerate(doc["instances"]):
        asset = assets[inst["asset"]]
        leaves = part(asset, "leaves")
        if leaves is None or not leaves.get("double_sided"):
            continue
        asked = inst.get("colours", {})
        wants[nth] = tuple(asked.get("leaves", leaves.get("colour", [0.14, 0.2, 0.06])))
        colours.setdefault(inst["asset"], Counter())[wants[nth]] += 1
        kinds.setdefault(inst["asset"], Counter())[inst.get("kind", asset["name"].split("_")[0])] += 1
        if part(asset, "wood"):
            bark[tuple(asked.get("wood", part(asset, "wood").get("colour", [0.07, 0.055, 0.04])))] += 1
    if not colours:
        log("no trees: nothing done")
        return
    kinds = {a: kinds[a].most_common(1)[0][0] for a in kinds}
    if not every_colour:
        colours = {a: Counter({max(c, key=lambda colour: (c[colour], colour)): sum(c.values())}) for a, c in colours.items()}
    often = Counter()
    for a in colours:
        for colour, count in colours[a].items():
            often[(kinds[a], colour)] += count
    pairs = sorted(often, key=lambda pair: (-often[pair], pair))          # the commonest first, and always in the same order
    picture, places, cover = leafatlas.build(pairs)
    folder = os.path.dirname(part(assets[min(colours)], "leaves")["mesh"])
    _save(os.path.join(out, folder, LEAVES), picture)
    tints = sorted({colour for _, colour in pairs}, key=lambda colour: (-sum(often[p] for p in pairs if p[1] == colour), colour))

    # Leaves: one file for each colour a crown comes in, alike but for where in the picture they point.
    moved, crowns, written = {}, 0, 0
    for a in sorted(colours):
        asset, kind = assets[a], kinds[a]
        leaves = part(asset, "leaves")
        source = os.path.join(out, leaves["mesh"])
        head, tail, count = _crown(source, (FILL / cover[leafatlas.kind_of(kind)]) ** 0.5, kind)
        for nth, (colour, _) in enumerate(sorted(colours[a].items(), key=lambda item: (-item[1], item[0]))):
            mesh = leaves["mesh"] if nth == 0 else "%s_t%d.obj" % (os.path.splitext(leaves["mesh"])[0], tints.index(colour))
            _write(os.path.join(out, mesh), head + _lines("vt %.5f %.5f\n", np.concatenate(places[(kind, colour)])) + tail)
            target = asset if nth == 0 else copy.deepcopy(asset)
            part(target, "leaves").update(mesh=mesh, colour=list(colour), texture=os.path.join(folder, LEAVES))
            if nth > 0:
                target["name"] = "%s_t%d" % (asset["name"], tints.index(colour))
                assets.append(target)
            moved[(a, colour)] = a if nth == 0 else len(assets) - 1
            written += count
        crowns += 1

    # Wood: one bark picture, in the colour most of the wood has.
    wood = sorted({part(assets[a], "wood")["mesh"] for a in colours if part(assets[a], "wood")})
    barked = {mesh for mesh in wood if _barked(os.path.join(out, mesh))}
    if barked:
        shade = bark.most_common(1)[0][0]
        _save(os.path.join(out, folder, BARK), barkmodel.picture(shade))
        for asset in assets:
            if part(asset, "wood") and part(asset, "wood")["mesh"] in barked and part(asset, "leaves"):
                part(asset, "wood")["texture"] = os.path.join(folder, BARK)

    for nth, colour in wants.items():
        a = doc["instances"][nth]["asset"]
        doc["instances"][nth]["asset"] = moved.get((a, colour), a)
    doc["foliage"] = {"synthesized": "leaves, gaps, bark: drawn by tools/foliage.py, not measured", "leaf_colours": [list(c) for c in tints],
                      "fill": FILL, "cover": {kind: round(share, 3) for kind, share in cover.items()}}
    total = pictures(doc)
    log(f"{crowns} crowns written {len(moved)} times for {len(tints)} leaf colours ({written} leaf triangles in files), sprays fill"
        f" {', '.join('%s %.2f' % item for item in sorted(cover.items()))} of a triangle, {len(barked)} of {len(wood)} wood meshes barked"
        + (f" ({len(bark)} wood colours, one bark: the commonest)" if len(bark) > 1 else "") + f", {total} pictures in all")
    if total > LIMIT:
        log(f"WARNING: {total} pictures, and the ray tracer takes {LIMIT}: the last ones loaded will show as plain colour")
