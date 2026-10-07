#!/usr/bin/env python3
"""Give the ground's pictures the fine grain a drone photo does not have. For the sensor copy only.

    python -I groundgrain.py SENSOR_SCENE      (a copy made by sensor_scene.py: it is changed)

The ground's picture is a drone photo at 5 cm a pixel. From a car's camera 1.3 m up, one pixel
of it is spread over 14 pixels of a 960 wide picture at 3 m, so the first ten metres of road
are a smooth blur. Real pavement there shows its aggregate and real grass its blades.

Each tile's picture is made twice the size each way, and the new pixels, 2.5 cm of ground, get
texture the photo could not hold: on pavement the speckle of aggregate with a few lighter stones
and darker pits, stronger in some patches than others, and a sparse hairline crack. On land the
light and dark of blades and thatch where the photo is green or straw, and a quieter grain
where it is bare. Anything 5 cm across or larger, stains and mottling included, is left to the
photo, which has it.

THIS TEXTURE IS SYNTHESIZED. It is noise of a plausible kind and strength. No stone, blade or
crack in it is one that is there on the site, and nothing should be measured from it.

The photo stays in charge. In the light the camera gathers (a texture value to the power
GAMMA), the four new pixels of every photo pixel average to exactly what that pixel was, colour
by colour: the grain moves light about inside a photo pixel and adds or removes none. So from
far off, or through a camera that averages, the ground is the photo, and every line and stain
in the photo is still there. A crack is held to the same rule: it darkens the pixels it crosses
and the rest of that photo pixel gets the light back, so it is a hairline with a faintly lit
lip, and from the car it can be made out 4 to 5 m ahead. No crack is drawn in a copy of a
scene built as found (build_scene.py --worn): its photo has the site's own. The pictures are
stored as JPEG, which moves a photo pixel's mean by about one level of 255 either way and by
nothing on average.

The grain is a function of the place in the world (worldnoise.py), found for each pixel
through the tile's own mesh, so two tiles agree along their shared edge and nothing repeats.

It costs memory. A tile's picture goes from 15 to 60 MB in the renderer, which holds them
uncompressed. Measured on this scene's 52 tiles: the pictures take 273 MB on disk where the
photos take 43, the renderer's memory goes from 2.8 to 5.2 GB and its start from 9 to 12 s.
A picture is drawn no slower. No picture is added to the scene's count: each tile names its
grained picture in place of its photo, and keeps the photo's name under "photo".
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groundmesh  # noqa: E402
import worldnoise  # noqa: E402

SCALE = 2           # new pixels per photo pixel, each way
GAMMA = 2.2         # the camera's light from a texture value: Chrono::Sensor raises it to this power
LEVEL = "standard"  # the texture level that is given grain
FOLDER = "textures/grain"
# "all", or "paved": the pavement's tiles and the land tiles that have pavement in them, 40 of 52 here,
# for three quarters of the memory. It leaves land without grain 0.8 m from the pavement in two places,
# where the north loop runs along the edge of its tile.
TILES = "all"

# How strong the grain is: the spread of a 2.5 cm pixel's light about its photo pixel's, as a share of
# it. Every strength here is such a share, never a number of grey levels, so the grain is the same
# on a photo stored darker or brighter, and near white it is drawn in as far as it must be (hold).
# Pavement: the photo's own finest detail (5 cm against 10 cm) is 2 to 3% there as the drone took
# it, and half that where the scene has squeezed its highlights, which is what makes it look
# moulded. The renderer smooths between texture pixels and that takes a good third off, so 6% here
# comes to 1.7% of the tone in the car's picture at 4 m and 1.6% at 9 m, where the bare photo gives
# 0.4% and 1%: grain that can be seen, and not gravel. At 4% it came to 0.8%, which a camera's own
# noise would bury.
PAVED = 0.060
# Grass: the photo's finest detail is 18 to 23% there and grows toward small sizes. A little under
# that, because without a camera that averages, every bit of it shows as far as the eye can see.
GRASS = 0.15
BARE = 0.06         # bare soil, gravel and whatever else the land's picture holds
STONES = 0.07       # the share of pavement pixels that hold one lighter stone or one darker pit
PATCH = (0.6, 2.5)  # metres: sizes of the patches where the grain is stronger or weaker
UNEVEN = 0.30       # by how much, as a spread
TINT = 0.04         # grass only: how far a pixel leans to straw or to green, as a spread
SEED = 151

CRACK_CELL = 14.0        # metres: the world is cut into squares this size
CRACK_SHARE = 0.45       # the share of them that hold one crack
CRACK_LENGTH = (1.5, 6.0)
CRACK_STEP = 0.10        # metres between the points of a crack
CRACK_WANDER = 0.22      # radians a crack turns by at each, as a spread
CRACK_DEPTH = 0.30       # the light a crack takes from a pixel it crosses, at its strongest. Less cannot be seen from the car


def enlarge(light):
    """A picture of light, SCALE times the size, smooth, whose pixels average to the original's."""
    h, w = light.shape[:2]

    def up(channel):
        return np.asarray(Image.fromarray(np.ascontiguousarray(channel)).resize((w * SCALE, h * SCALE), Image.BILINEAR))

    def down(big):
        return big.reshape(h, SCALE, w, SCALE).mean((1, 3))

    out = np.empty((h * SCALE, w * SCALE, light.shape[2]), np.float32)
    for c in range(light.shape[2]):
        big = up(light[..., c])
        for _ in range(2):          # plain interpolation blurs: hand back what it lost, twice
            big = big + up(light[..., c] - down(big))
        out[..., c] = np.maximum(big, 0.0)
    return out


def hold(fine, light):
    """Scale each photo pixel's new pixels so they average to the photo pixel, and none passes white."""
    h, w = light.shape[:2]
    blocks = fine.reshape(h, SCALE, w, SCALE, -1)
    blocks *= (light / np.maximum(blocks.mean((1, 3)), 1e-6))[:, None, :, None, :]
    blocks += (light - blocks.mean((1, 3)))[:, None, :, None, :]      # what scaling cannot mend: four new pixels that are all black
    # Where the brightest would pass white, pull that photo pixel's new pixels toward their mean.
    over = blocks.max((1, 3)) - light
    blocks -= light[:, None, :, None, :]
    blocks *= np.minimum(1.0, (1.0 - light) / np.maximum(over, 1e-6))[:, None, :, None, :]
    blocks += light[:, None, :, None, :]


def cracks(x0, y0, x1, y1):
    """The cracks that reach into a piece of the world: a list of (points (n, 2), strength (n,))."""
    reach = CRACK_LENGTH[1]
    found = []
    for cy in range(int(np.floor((y0 - reach) / CRACK_CELL)), int(np.floor((y1 + reach) / CRACK_CELL)) + 1):
        for cx in range(int(np.floor((x0 - reach) / CRACK_CELL)), int(np.floor((x1 + reach) / CRACK_CELL)) + 1):
            h = worldnoise.bits(cx, cy, SEED + 10)
            if worldnoise.uniform(h, 0) > CRACK_SHARE:
                continue
            n = int((CRACK_LENGTH[0] + (CRACK_LENGTH[1] - CRACK_LENGTH[0]) * worldnoise.uniform(h, 3) ** 2) / CRACK_STEP)
            turns = worldnoise.normal(worldnoise.bits(np.arange(n), cy * 100003 + cx, SEED + 11)) * CRACK_WANDER
            heading = 2 * np.pi * worldnoise.uniform(worldnoise.bits(cx, cy, SEED + 12)) + np.cumsum(turns)
            start = np.array([cx + worldnoise.uniform(h, 1), cy + worldnoise.uniform(h, 2)]) * CRACK_CELL
            points = start + np.cumsum(np.stack([np.cos(heading), np.sin(heading)], 1) * CRACK_STEP, axis=0)
            found.append((points, np.sin(np.linspace(0, np.pi, n)) ** 0.5))       # it fades out at both ends
    return found


def grass(r, g, b):
    """How much a colour is that of grass, 0 to 1: green through straw, and not grey. sRGB values, any scale.

    Hue runs from red at 0 through yellow at 60 to green at 120 degrees. This lawn lies between
    40 and 100 and the fields and bare earth round it mostly below 38, but dry lawn and
    yellowish soil overlap, and where they do this cannot tell them apart.
    """
    hue = np.degrees(np.arctan2(np.sqrt(3.0) * (g - b), 2 * r - g - b))
    top = np.maximum(np.maximum(r, g), b)
    strong = (top - np.minimum(np.minimum(r, g), b)) / np.maximum(top, 1e-6)
    return np.clip((hue - 36.0) / 10.0, 0, 1) * np.clip((strong - 0.10) / 0.10, 0, 1)


def grain(photo, x0, y0, width, height, paved, cracked=True):
    """One tile's picture with grain: sRGB uint8 in and out, the result SCALE times the size. cracked: draw hairline cracks on pavement."""
    rows, cols = photo.shape[0] * SCALE, photo.shape[1] * SCALE
    xs = x0 + (np.arange(cols) + 0.5) * width / cols             # the world, at the middle of each new pixel
    ys = y0 + height - (np.arange(rows) + 0.5) * height / rows   # row 0 is the north edge
    pitch = abs(width) / cols
    ix, iy = np.floor(xs / pitch).astype(np.int64), np.floor(ys / pitch).astype(np.int64)
    light = (photo.astype(np.float32) / 255.0) ** GAMMA
    fine = enlarge(light)

    uneven = 1.0 + UNEVEN * 0.7 * sum(worldnoise.smooth_grid(xs, ys, size, SEED + 1 + k) for k, size in enumerate(PATCH))
    uneven = np.clip(uneven, 0.35, 1.8).astype(np.float32)
    h = worldnoise.bits(ix[None, :], iy[:, None], SEED)
    if paved:
        speck = worldnoise.normal(h, 0) * 0.72
        pick, size = worldnoise.uniform(h, 1), 1.6 + 1.6 * worldnoise.uniform(h, 2)
        speck = np.where(pick < STONES / 2, size, np.where(pick > 1 - STONES / 2, -size, speck))    # unit spread overall
        factor = 1.0 + PAVED * uneven * speck
        if cracked:
            line = Image.new("L", (cols, rows), 0)
            draw = ImageDraw.Draw(line)
            for points, strength in cracks(xs[0], ys[-1], xs[-1], ys[0]):
                px, py = (points[:, 0] - x0) / width * cols, (y0 + height - points[:, 1]) / height * rows
                for k in range(len(points) - 1):
                    draw.line([(px[k], py[k]), (px[k + 1], py[k + 1])], fill=int(40 + 215 * strength[k]))
            factor *= 1.0 - CRACK_DEPTH * (np.asarray(line, np.float32) / 255.0)
        fine *= np.maximum(factor, 0.05)[..., None]
    else:
        # How green or straw-coloured the photo is here decides between the grain of grass and of bare ground.
        r, g, b = (np.asarray(Image.fromarray(photo[..., c]).resize((cols, rows), Image.BILINEAR), np.float32) for c in range(3))
        grassy = grass(r, g, b)
        spread = (BARE + (GRASS - BARE) * grassy) * uneven
        factor = np.exp(spread * worldnoise.normal(h, 0))
        lean = TINT * grassy * worldnoise.normal(h, 1)
        for c, gain in enumerate((1.0, -0.3, -1.6)):         # toward straw: more red, less blue. Toward green: the other way
            fine[..., c] *= factor * (1.0 + gain * lean)
    hold(fine, light)
    return (np.clip(fine, 0, 1) ** (1.0 / GAMMA) * 255.0 + 0.5).astype(np.uint8)


def apply(doc, out, log=print, tiles=TILES):
    """Replace the ground tiles' pictures in the copy at out with grained ones, and name them in its manifest doc.

    New files are written under FOLDER and each tile's part is pointed at its own. Nothing that
    was in the copy is written to: its other files may be links to the scene's.
    """
    group = {inst["asset"]: inst.get("group") for inst in doc["instances"]}
    todo = [(n, a) for n, a in enumerate(doc["assets"]) if n in group and len(a["parts"]) == 1 and "<level>" in (a["parts"][0].get("texture") or "")]
    meshes = {n: groundmesh.read(os.path.join(out, a["parts"][0]["mesh"])) for n, a in todo}
    frames = {n: groundmesh.frame(*meshes[n]) for n, _ in todo}
    if None in frames.values():
        raise SystemExit("a ground picture does not lie square to the world: " + ", ".join(a["name"] for n, a in todo if frames[n] is None))
    with_pavement = {tuple(round(v, 2) for v in frames[n]) for n, _ in todo if group[n] == "Road"}
    if not todo:
        log("no ground pictures left to give grain to: has this copy had it already?")
        return
    os.makedirs(os.path.join(out, FOLDER), exist_ok=True)
    # A scene built as found has the site's own cracks in its photo. Drawn ones among them
    # would be taken for real, so there none is drawn.
    cracked = not doc.get("worn")
    done = before = after = 0
    for n, asset in todo:
        part = asset["parts"][0]
        if tiles == "paved" and tuple(round(v, 2) for v in frames[n]) not in with_pavement:
            continue
        source = os.path.join(out, part["texture"].replace("<level>", LEVEL))
        photo = np.asarray(Image.open(source).convert("RGB"))
        picture = grain(photo, *frames[n], paved=group[n] == "Road", cracked=cracked)
        name = f"{FOLDER}/{asset['name']}.jpg"
        Image.fromarray(picture).save(os.path.join(out, name + ".new"), format="JPEG", quality=92, subsampling=0, optimize=True)
        os.replace(os.path.join(out, name + ".new"), os.path.join(out, name))
        part["photo"], part["texture"] = part["texture"], name
        done, before, after = done + 1, before + os.path.getsize(source), after + os.path.getsize(os.path.join(out, name))
    doc.setdefault("synthesized", {})["grain"] = (f"{done} ground pictures carry synthesized grain at {SCALE} pixels to the photo's one, "
                                                  "made by tools/groundgrain.py: texture of a plausible kind, not of the site")
    log(f"{done} of {len(todo)} ground pictures given grain at {SCALE}x{'' if cracked else ', no cracks drawn (the photo has its own)'}:"
        f" {before / 1e6:.0f} MB of pictures became {after / 1e6:.0f} MB")
    groundmesh.count_pictures(doc, log, "groundgrain")     # each tile names its new picture in place of its photo: none is added


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(os.path.join(sys.argv[1], "speedway_scene.json")) as f:
        manifest = json.load(f)
    apply(manifest, sys.argv[1])
    with open(os.path.join(sys.argv[1], "speedway_scene.json.new"), "w") as f:
        json.dump(manifest, f)
    os.replace(os.path.join(sys.argv[1], "speedway_scene.json.new"), os.path.join(sys.argv[1], "speedway_scene.json"))
