"""Give every placement its size and its colour in files of its own.

A scene places one mesh many times: one cone model at four heights and three colours, one
tree model at a hundred sizes. The size and the colour are on the placement, and Chrono::VSG
takes them from there. Chrono::Sensor's Metal ray tracer does not. It draws a triangle mesh at
the size it has in its file whatever scale the shape carries, and it shares one set of
triangles among all placements of a mesh, colour and all, so each takes the colour of the
first. (Read in ChMetalSceneBuilder.cpp: a mesh shape's scale is passed on as one, and the
key meshes are shared by leaves the plain colour out.) Every cone then stands a metre tall
and orange.

So here each size and colour a mesh is placed at becomes a mesh file and an asset of its
own, with the size worked into the vertices and the colour into the asset, and placements
are left with a scale of one and no colours. It costs disk and loading time, mostly for the
trees: nearly every tree has a size of its own.
"""
import copy
import os

import numpy as np


def _scaled(source, target, scale):
    """Write an OBJ file again with its vertices scaled about the origin, and its normals turned to match."""
    scale = np.asarray(scale, float)
    out = []
    with open(source) as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = (float(v) for v in line.split()[1:4])
                out.append("v %.4f %.4f %.4f\n" % (x * scale[0], y * scale[1], z * scale[2]))
            elif line.startswith("vn "):
                n = np.array([float(v) for v in line.split()[1:4]]) / scale        # a normal goes by the inverse
                out.append("vn %.4f %.4f %.4f\n" % tuple(n / max(np.linalg.norm(n), 1e-12)))
            else:
                out.append(line)
    with open(target, "w") as f:
        f.writelines(out)


def separate(doc, out, log=print):
    """Change the copy's files and its manifest, in place, so that no placement has a scale or colours of its own."""
    assets = doc["assets"]
    made, count, files = {}, {}, 0
    for inst in doc["instances"]:
        scale = tuple(round(float(s), 3) for s in inst.get("scale", (1.0, 1.0, 1.0)))
        colours = tuple(sorted((part, tuple(colour)) for part, colour in inst.get("colours", {}).items()))
        if scale != (1.0, 1.0, 1.0) or colours:
            key = (inst["asset"], scale, colours)
            if key not in made:
                asset = copy.deepcopy(assets[inst["asset"]])
                nth = count[inst["asset"]] = count.get(inst["asset"], 0) + 1
                asset["name"] = f"{asset['name']}_p{nth}"
                for part in asset["parts"]:
                    if part["name"] in dict(colours):
                        part["colour"] = list(dict(colours)[part["name"]])
                    stem, ext = os.path.splitext(part["mesh"])
                    mesh = f"{stem}_p{nth}{ext}"
                    if scale == (1.0, 1.0, 1.0):
                        os.link(os.path.join(out, part["mesh"]), os.path.join(out, mesh))      # the same triangles under a name of their own
                    else:
                        _scaled(os.path.join(out, part["mesh"]), os.path.join(out, mesh), scale)
                    part["mesh"] = mesh
                    files += 1
                assets.append(asset)
                made[key] = len(assets) - 1
            inst["asset"] = made[key]
        inst["scale"] = [1.0, 1.0, 1.0]
        inst.pop("colours", None)
    log(f"{len(made)} sizes and colours of {len(count)} meshes written out as {files} files of their own")
