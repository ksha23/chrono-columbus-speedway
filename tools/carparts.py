"""Sorting a chassis mesh into parts by what its surfaces are: paint, glass, dark trim, lamps.

Chrono's older vehicle meshes have a plain material for each of these. The newer ones have
one texture for the whole body, with the paint, the glass and the trim as flat areas of it.
Those are told apart by looking the texture up: see-through in the opacity map is glass, a
texel that differs between two colour variants of the same chassis is paint, and the rest
goes by its colour. Colours come out as linear light.
"""
import os

import numpy as np

import carmesh


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


_TEXTURES = {}


def _texture(path):
    if path not in _TEXTURES:
        from PIL import Image
        _TEXTURES[path] = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return _TEXTURES[path]


def _lookup(image, uv):
    h, w = image.shape[:2]
    col = np.floor(uv[..., 0] * w).astype(np.int64) % w
    row = np.floor((1.0 - uv[..., 1]) * h).astype(np.int64) % h
    return image[row, col]


PAINT, GLASS, DARK, BRIGHT, LAMP = range(5)


def texel_kinds(uv, rule, folder):
    """What the texture says is at each UV point: (kind, sRGB colour).

    See-through in the opacity map is glass. A texel that differs between two colour variants
    of the same chassis is paint. The rest goes by its colour: dark, red or amber (lamps), or
    bright. The rule may move the line between dark and bright with "dark_below".
    """
    texel = _lookup(_texture(os.path.join(folder, rule["texture"])), uv)
    kind = np.full(uv.shape[:-1], BRIGHT, dtype=np.int64)
    r, g, b = texel[..., 0], texel[..., 1], texel[..., 2]
    kind[(r > 0.3) & (r > 1.4 * g) & (r > 2.0 * b)] = LAMP
    kind[texel.max(axis=-1) < rule.get("dark_below", 0.25)] = DARK
    if "variant" in rule:
        other = _lookup(_texture(os.path.join(folder, rule["variant"])), uv)
        kind[np.abs(texel - other).max(axis=-1) > 0.15] = PAINT
    if "opacity" in rule:
        kind[_lookup(_texture(os.path.join(folder, rule["opacity"])), uv)[..., 0] < 0.75] = GLASS
    return kind, texel


def cut_along_borders(P, N, UV, rule, folder):
    """Cut the triangles that a border in the texture runs across, along that border.

    The meshes were made to be textured, so the line between paint and glass, say, crosses
    triangles where it likes. Sorting whole triangles to one side or the other would leave a
    saw-toothed edge. Instead, where the kind of texel differs between the two ends of an
    edge, the place it changes is found by halving the interval, the same from both
    triangles that share the edge, and a triangle with two such edges becomes three.
    """
    ids, _ = carmesh.weld(P)
    crossing = np.zeros((len(P), 3), dtype=bool)
    where = np.zeros((len(P), 3))                    # along the edge from corner k to corner k + 1
    for k in range(3):
        a, b = k, (k + 1) % 3
        swap = ids[:, a] > ids[:, b]                 # always walk from the lower vertex to the higher
        start = np.where(swap[:, None], UV[:, b], UV[:, a])
        step = np.where(swap[:, None], UV[:, a], UV[:, b]) - start
        first, _ = texel_kinds(start + 0.04 * step, rule, folder)
        last, _ = texel_kinds(start + 0.96 * step, rule, folder)
        chosen = np.flatnonzero((first != last) & (ids[:, a] != ids[:, b]))
        low, high = np.full(len(chosen), 0.04), np.full(len(chosen), 0.96)
        for _ in range(9):
            middle = (low + high) / 2
            same, _ = texel_kinds(start[chosen] + middle[:, None] * step[chosen], rule, folder)
            same = same == first[chosen]
            low, high = np.where(same, middle, low), np.where(same, high, middle)
        crossing[chosen, k] = True
        where[chosen, k] = np.where(swap[chosen], 1.0 - (low + high) / 2, (low + high) / 2)
    split = np.flatnonzero(crossing.sum(axis=1) == 2)
    if not len(split):
        return P, N, UV
    apex = (np.argmin(crossing[split], axis=1) + 2) % 3      # the corner between the two crossed edges

    def corner(array, k):
        return array[split, (apex + k) % 3]

    t_ab = where[split, apex][:, None]
    t_ca = where[split, (apex + 2) % 3][:, None]
    out = []
    for array in (P, N, UV):
        a, b, c = corner(array, 0), corner(array, 1), corner(array, 2)
        ab = a + t_ab * (b - a)
        ca = c + t_ca * (a - c)
        pieces = np.concatenate([np.stack([a, ab, ca], axis=1), np.stack([ab, b, c], axis=1), np.stack([ab, c, ca], axis=1)])
        whole = np.delete(array, split, axis=0)
        out.append(np.concatenate([whole, pieces]))
    return out[0], carmesh.unit(out[1]), out[2]


def classify(mesh, spec, folder):
    """The chassis mesh sorted into parts: (P, N, part name per triangle, linear colour per triangle).

    A material with plain colours maps to a part by name. A textured one is first cut along
    the borders in its texture, then looked up at seven points on each triangle, and the
    triangle takes the majority. Materials the spec does not list are left out.
    """
    all_N = mesh["N"] if mesh["N"] is not None else carmesh.smooth_normals(mesh["P"])
    mtl = carmesh.load_mtl(os.path.join(folder, mesh["mtllib"])) if mesh["mtllib"] else {}
    out_P, out_N, out_part, out_colour = [], [], [], []
    for index, name in enumerate(mesh["materials"]):
        chosen = np.flatnonzero(mesh["material"] == index)
        rule = spec["materials"].get(name)
        if rule is None or not len(chosen):
            continue
        P, N = mesh["P"][chosen], all_N[chosen]
        if isinstance(rule, str):
            part = np.full(len(P), rule, dtype=object)
            colour = np.repeat(np.array([mtl[name]["Kd"]], dtype=np.float64), len(P), axis=0)
        else:
            UV = mesh["UV"][chosen]
            if "variant" in rule or "opacity" in rule:
                for _ in range(3):         # a triangle is cut along one border at a time, and may hold two
                    before = len(P)
                    P, N, UV = cut_along_borders(P, N, UV, rule, folder)
                    if len(P) == before:
                        break
            kind, texel = texel_kinds(np.einsum("sk,tkc->tsc", carmesh.SAMPLES, UV), rule, folder)
            votes = np.stack([(kind == k).sum(axis=1) for k in range(5)], axis=1)
            votes[np.arange(len(P)), kind[:, 0]] += 1                     # the centre breaks a tie
            winner = votes.argmax(axis=1)
            names = ["body", "glass", rule.get("dark", "trim"), rule.get("bright", "bright"), rule.get("lamps", "taillights")]
            part = np.array(names, dtype=object)[winner]
            agree = kind == winner[:, None]
            colour = (srgb_to_linear(texel) * agree[..., None]).sum(axis=1) / np.maximum(agree.sum(axis=1), 1)[:, None]
        out_P.append(P)
        out_N.append(N)
        out_part.append(part)
        out_colour.append(colour)
    return np.concatenate(out_P), np.concatenate(out_N), np.concatenate(out_part), np.concatenate(out_colour)


def plain_colour(mesh, folder, fallback):
    """One linear colour for a whole mesh: its material's Kd, or the mean of its texture over
    the surface, or the fallback when the mesh names no material file."""
    if not mesh["mtllib"]:
        return [float(x) for x in fallback]
    mtl = carmesh.load_mtl(os.path.join(folder, mesh["mtllib"]))
    _, area = carmesh.face_normals(mesh["P"])
    total = np.zeros(3)
    for index, name in enumerate(mesh["materials"]):
        chosen = mesh["material"] == index
        entry = mtl.get(name, {})
        if "map_Kd" in entry and mesh["UV"] is not None:
            uv = np.einsum("sk,tkc->tsc", carmesh.SAMPLES, mesh["UV"][chosen])
            texel = srgb_to_linear(_lookup(_texture(os.path.join(folder, entry["map_Kd"])), uv)).mean(axis=1)
            total += (texel * area[chosen, None]).sum(axis=0)
        else:
            total += np.array(entry.get("Kd", fallback)) * area[chosen].sum()
    return [float(x) for x in total / max(area.sum(), 1e-12)]
