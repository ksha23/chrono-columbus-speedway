#!/usr/bin/env python3
"""Make a copy of the scene whose pictures and colours suit Chrono::Sensor's camera.

    python -I sensor_scene.py SCENE_DIR OUT_DIR [--detail]

The scene is stored for Chrono::VSG, which takes a texture's values as linear light and shows
flat sunlit ground at renderer.LIT of them (see renderer.py). Chrono::Sensor's camera does what
most renderers do: a texture is an ordinary sRGB picture, a plain colour is linear light, and
the lights say how bright things are. Shown the VSG scene as it is, it decodes every texture a
second time and the picture comes out far too dark.

So in the copy every texture is the picture Chrono::VSG would put on screen for it, and every
plain colour is scaled by renderer.LIT. Under lights that give flat ground about 1.0, the two
renderers then agree on how bright the ground and the paint on it are.

Leaves are the other difference. A leaf is one triangle with two sides. Chrono::VSG is told to
draw both and lights whichever it is looking at. The ray tracer lights a triangle by the
normals its corners carry, from wherever it is seen, so leaves at every angle make a crown of
bright and black specks. In the copy a leaf's corners carry the direction out of the middle of
its crown, leaning upward: the crown is then lit as the rounded thing it is, bright on the
sunny side and darker on the other, and a leaf is black only where another shades it.

A scene built with --worn carries the ground's pictures a second time, as the drone took
them. Those are used as they are, and the copy is then at the photo's own exposure: sunlit
concrete near white, as it was, with its joints and stains at their full contrast. The
scene's own pictures are darker and have their highlights squeezed, which Chrono::VSG needs
and which leaves bright concrete nearly blank.

A placement's size and colour go into files too (placements.py): this renderer takes
neither from the placement.

Other meshes are not copied: they are hard links to the scene's own, so the copy costs only
its textures and its leaves. The copy's manifest also says what colour the ground is at the
scene's rim ("beyond"), for a renderer whose sky picture has ground of its own to cover.

--detail adds what a camera at car height sees and a scan from the air cannot give: leaves
and bark on the trees (foliage.py), fine grain in the ground's pictures (groundgrain.py),
and grass that stands up along the pavement's edge (verge.py). All of it is synthesized. It
is there so that the picture has the fine texture a real one has, where the scan's five
centimetres a pixel leaves a smooth blur, and none of it is measured.
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import placements  # noqa: E402
import renderer  # noqa: E402

PICTURES = (".jpg", ".jpeg", ".png")
GROUND = ("textures", os.path.join("textures", "standard"))     # the folders the ground's pictures are in
METAL_TAKES = 64     # pictures in one scene that Chrono::Sensor's Metal ray tracer takes
MANIFEST = "speedway_scene.json"


UP = 0.5          # how far a leaf's normal leans upward from the direction out of its crown
RIM = 0.04        # the share of the surround's picture, in from each edge, that its rim colour is taken over


def rounded(source, copy):
    """Write a crown's leaves again with normals that point out of the crown's middle. Returns the number of leaves."""
    vertices, faces = [], []
    with open(source) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(corner.split("/")[0]) for corner in line.split()[1:4]])
    vertices = np.array(vertices)
    out = vertices - vertices.mean(0)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    out[:, 2] += UP
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    with open(copy, "w") as f:
        f.write("".join(f"v {x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in vertices))
        f.write("".join(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in out))
        f.write("".join(f"f {a}//{a} {b}//{b} {c}//{c}\n" for a, b, c in faces))
    return len(faces)


def convert(scene, out, log=print, detail=False):
    with open(os.path.join(scene, MANIFEST)) as f:
        doc = json.load(f)
    leaves = {os.path.normpath(part["mesh"]) for asset in doc["assets"] for part in asset["parts"] if part["name"] == "leaves" and part.get("double_sided")}
    # A worn build keeps the ground's pictures as the drone took them. They are used as they
    # are, and everything else is brought to the same exposure: the photo's, not the darker
    # one the scene is stored at for Chrono::VSG.
    photographed = os.path.normpath(doc.get("as_photographed") or "") if doc.get("as_photographed") else None
    gain = 1.0 / renderer.EXPOSURE if photographed else 1.0
    pictures = linked = turned = taken = 0
    for folder, _, files in os.walk(scene):
        where = os.path.normpath(os.path.relpath(folder, scene))
        if photographed and where == photographed:
            continue
        target = os.path.join(out, where)
        os.makedirs(target, exist_ok=True)
        for name in sorted(files):
            source, copy = os.path.join(folder, name), os.path.join(target, name)
            if name == MANIFEST:
                continue
            as_taken = os.path.join(scene, photographed, name) if photographed and where in GROUND else None
            if as_taken and os.path.isfile(as_taken):
                shutil.copyfile(as_taken, copy)
                taken += 1
            elif name.lower().endswith(PICTURES):
                picture = Image.open(source)
                shown = Image.fromarray(renderer.shown(np.asarray(picture.convert("RGB")), gain))
                if name.lower().endswith(".png"):
                    shown.save(copy, optimize=True)
                else:
                    shown.save(copy, quality=92, subsampling=0, optimize=True)
                pictures += 1
            elif os.path.normpath(os.path.relpath(source, scene)) in leaves:
                turned += rounded(source, copy)
            else:
                try:
                    os.link(source, copy)
                except OSError:               # another volume, or a file system without links
                    shutil.copyfile(source, copy)
                linked += 1

    # A part with a texture keeps its white: the texture is its colour, and that is converted above.
    plain = set()
    for asset in doc["assets"]:
        for part in asset["parts"]:
            if part.get("texture"):
                continue
            plain.add((asset["name"], part["name"]))
            if "colour" in part:
                part["colour"] = [round(min(c * renderer.LIT * gain, 1.0), 4) for c in part["colour"]]
    for inst in doc["instances"]:
        asset = doc["assets"][inst["asset"]]["name"]
        for part, colour in inst.get("colours", {}).items():
            if (asset, part) in plain:
                inst["colours"][part] = [round(min(c * renderer.LIT * gain, 1.0), 4) for c in colour]
    doc["for"] = "Chrono::Sensor: textures are sRGB pictures and colours are linear, made by tools/sensor_scene.py"
    doc["exposure"] = "the photo's own" if photographed else "the scene's, 0.8 of the photo's, with the ground's highlights squeezed"
    doc.pop("as_photographed", None)
    # The colour of the ground at the scene's rim, as seen: the surround's picture along its four edges.
    for asset in doc["assets"]:
        if asset["name"] == "surround" and asset["parts"][0].get("texture"):
            picture = np.asarray(Image.open(os.path.join(out, asset["parts"][0]["texture"])).convert("RGB"), dtype=np.float32) / 255.0
            edge = max(int(RIM * min(picture.shape[:2])), 1)
            rim = np.concatenate([picture[:edge].reshape(-1, 3), picture[-edge:].reshape(-1, 3), picture[:, :edge].reshape(-1, 3), picture[:, -edge:].reshape(-1, 3)])
            doc["beyond"] = [round(float(c), 4) for c in rim.mean(0)]
    log(f"{pictures} textures converted, {taken} of the ground's taken as photographed, {turned} leaves in {len(leaves)} crowns given rounded normals,"
        f" {linked} other files linked, {len(plain)} plain parts' colours scaled by {renderer.LIT * gain:.3f}, ground at the rim {doc.get('beyond')}")
    placements.separate(doc, out, log)
    if detail:
        # Each of these changes files of the copy and the manifest in hand. They are imported
        # here, so that the plain copy needs none of them.
        import foliage
        import groundgrain
        import verge
        for step in (foliage, groundgrain, verge):
            step.apply(doc, out, log)
        doc["detail"] = "leaves, bark, ground grain and verge grass are synthesized, not measured: tools/sensor_scene.py --detail"
    used = {inst["asset"] for inst in doc["instances"]}
    textures = {part["texture"] for n, asset in enumerate(doc["assets"]) if n in used for part in asset["parts"] if part.get("texture")}
    log(f"{len(textures)} pictures in all" + ("" if len(textures) <= METAL_TAKES else
        f": MORE THAN THE {METAL_TAKES} that Chrono::Sensor's Metal ray tracer takes. It draws the rest in plain colour and says nothing"))
    with open(os.path.join(out, MANIFEST), "w") as f:
        json.dump(doc, f)


if __name__ == "__main__":
    wanted = [a for a in sys.argv[1:] if a != "--detail"]
    if len(wanted) != 2:
        sys.exit(__doc__)
    if os.path.exists(wanted[1]) and os.listdir(wanted[1]):
        sys.exit(f"{wanted[1]} exists and is not empty: give a new directory")
    convert(wanted[0], wanted[1], detail="--detail" in sys.argv[1:])
