#!/usr/bin/env python3
"""Make a copy of the scene whose pictures and colours suit Chrono::Sensor's camera.

    python -I sensor_scene.py SCENE_DIR OUT_DIR

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

Other meshes are not copied: they are hard links to the scene's own, so the copy costs only
its textures and its leaves. The copy's manifest also says what colour the ground is at the
scene's rim ("beyond"), for a renderer whose sky picture has ground of its own to cover.
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import renderer  # noqa: E402

PICTURES = (".jpg", ".jpeg", ".png")
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


def convert(scene, out, log=print):
    with open(os.path.join(scene, MANIFEST)) as f:
        doc = json.load(f)
    leaves = {os.path.normpath(part["mesh"]) for asset in doc["assets"] for part in asset["parts"] if part["name"] == "leaves" and part.get("double_sided")}
    pictures = linked = turned = 0
    for folder, _, files in os.walk(scene):
        target = os.path.join(out, os.path.relpath(folder, scene))
        os.makedirs(target, exist_ok=True)
        for name in sorted(files):
            source, copy = os.path.join(folder, name), os.path.join(target, name)
            if name == MANIFEST:
                continue
            if name.lower().endswith(PICTURES):
                picture = Image.open(source)
                shown = Image.fromarray(renderer.shown(np.asarray(picture.convert("RGB"))))
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
                part["colour"] = [round(c * renderer.LIT, 4) for c in part["colour"]]
    for inst in doc["instances"]:
        asset = doc["assets"][inst["asset"]]["name"]
        for part, colour in inst.get("colours", {}).items():
            if (asset, part) in plain:
                inst["colours"][part] = [round(c * renderer.LIT, 4) for c in colour]
    doc["for"] = "Chrono::Sensor: textures are sRGB pictures and colours are linear, made by tools/sensor_scene.py"
    # The colour of the ground at the scene's rim, as seen: the surround's picture along its four edges.
    for asset in doc["assets"]:
        if asset["name"] == "surround" and asset["parts"][0].get("texture"):
            picture = np.asarray(Image.open(os.path.join(out, asset["parts"][0]["texture"])).convert("RGB"), dtype=np.float32) / 255.0
            edge = max(int(RIM * min(picture.shape[:2])), 1)
            rim = np.concatenate([picture[:edge].reshape(-1, 3), picture[-edge:].reshape(-1, 3), picture[:, :edge].reshape(-1, 3), picture[:, -edge:].reshape(-1, 3)])
            doc["beyond"] = [round(float(c), 4) for c in rim.mean(0)]
    with open(os.path.join(out, MANIFEST), "w") as f:
        json.dump(doc, f)
    log(f"{pictures} textures converted, {turned} leaves in {len(leaves)} crowns given rounded normals, {linked} other files linked,"
        f" {len(plain)} plain parts' colours scaled by {renderer.LIT}, ground at the rim {doc.get('beyond')}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    if os.path.exists(sys.argv[2]) and os.listdir(sys.argv[2]):
        sys.exit(f"{sys.argv[2]} exists and is not empty: give a new directory")
    convert(sys.argv[1], sys.argv[2])
