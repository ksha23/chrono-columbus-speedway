#!/usr/bin/env python3
"""Look at every painted line in the scene from above, and list the places where one wobbles.

    python -I audit_lines.py WORK_DIR SCENE_DIR OUT_DIR [--tile 80] [--scale 0.08]

Reads WORK_DIR/ground_photo.npy and pavement.npy (build_scene.py --save-photo writes both) and
SCENE_DIR/markings/strokes.json and edge_lines.json. Writes one picture per stretch of
pavement, with the paint drawn at its true width and colour on the finished ground photo, the
pavement's outline in thin red, and a circle on every kink in an edge line. Edge lines are
drawn in blue, since white on sunlit concrete does not show from this far up.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pavement  # noqa: E402

TURN = 8.0        # degrees: a turn this sharp one way...
WITHIN = 5.0      # ...followed within this many metres by one as sharp the other way, is a kink
CHORD = 1.0       # metres: headings are taken over chords this long, turns between neighbouring chords
PAINT = {"yellow": (240, 190, 20), "white": (255, 255, 255), "blue": (40, 80, 200), "edge": (0, 210, 255)}


def resample(points, spacing):
    p = np.asarray(points, float)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if d[-1] < spacing:
        return p
    s = np.linspace(0.0, d[-1], int(d[-1] / spacing) + 1)
    return np.stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], 1)


def kinks(line, spacing=0.25):
    """Places where a line turns sharply one way and then back: [(x, y, degrees)].

    A junction's corner turns one way only, and a road's S-bend is far gentler than this. What
    is left is a line following a notch or a bump in an outline that should not have one.
    """
    p = resample(line, spacing)
    n = int(round(CHORD / spacing))
    if len(p) < 4 * n:
        return []
    chord = p[n:] - p[:-n]
    heading = np.unwrap(np.arctan2(chord[:, 1], chord[:, 0]))
    turn = np.degrees(heading[n:] - heading[:-n])          # at p[n:-n]
    reach = 2 * int(round(WITHIN / spacing)) + 1
    left = np.minimum(ndimage.maximum_filter1d(turn, reach, mode="nearest"), -ndimage.minimum_filter1d(turn, reach, mode="nearest"))
    out = []
    labels, count = ndimage.label((left > TURN) & (np.abs(turn) > TURN))
    for k in range(1, count + 1):
        idx = np.nonzero(labels == k)[0]
        worst = idx[np.argmax(np.abs(turn[idx]))]
        out.append((float(p[worst + n, 0]), float(p[worst + n, 1]), float(left[worst])))
    # One kink is a turn and its answer, a few metres apart: report it once.
    out.sort(key=lambda w: -w[2])
    kept = []
    for w in out:
        if all(np.hypot(w[0] - q[0], w[1] - q[1]) > WITHIN for q in kept):
            kept.append(w)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("scene")
    parser.add_argument("out")
    parser.add_argument("--tile", type=float, default=64.0, help="metres along each side of a picture")
    parser.add_argument("--scale", type=float, default=0.08, help="metres per pixel of the pictures")
    parser.add_argument("--pavement", help="a pavement map to use in place of WORK_DIR/pavement.npy")
    parser.add_argument("--edges", help="edge lines to use in place of the scene's")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raster = json.load(open(os.path.join(args.work, "raster.json")))
    photo = np.load(os.path.join(args.work, "ground_photo.npy"), mmap_mode="r")
    paved = np.load(args.pavement or os.path.join(args.work, "pavement.npy"))
    strokes = json.load(open(os.path.join(args.scene, "markings", "strokes.json")))
    edges = json.load(open(args.edges or os.path.join(args.scene, "markings", "edge_lines.json")))
    res = raster["res"]

    found = []
    for line in edges:
        found += kinks(line["points"])
    found.sort(key=lambda w: -w[2])
    total = sum(float(np.hypot(*np.diff(np.asarray(e["points"], float), axis=0).T).sum()) for e in edges)
    print(f"edge lines: {len(edges)} pieces, {total:.0f} m. Kinks (over {TURN:g} degrees one way and back within {WITHIN:g} m): {len(found)}")
    for x, y, size in found[:40]:
        print(f"   at ({x:7.1f}, {y:7.1f})  {size:.0f} degrees")
    json.dump(found, open(os.path.join(args.out, "kinks.json"), "w"))

    step = max(int(round(args.scale / res)), 1)
    scale = step * res
    outline = paved ^ ndimage.binary_erosion(paved)
    kf = int(round(pavement.RES / res))
    names = []
    for ty in np.arange(raster["y1"], raster["y1"] - photo.shape[0] * res, -args.tile):
        for tx in np.arange(raster["x0"], raster["x0"] + photo.shape[1] * res, args.tile):
            r0, c0 = int(round((raster["y1"] - ty) / res)), int(round((tx - raster["x0"]) / res))
            r1, c1 = min(r0 + int(round(args.tile / res)), photo.shape[0]), min(c0 + int(round(args.tile / res)), photo.shape[1])
            part = paved[r0 // kf:r1 // kf, c0 // kf:c1 // kf]
            if part.mean() < 0.02:
                continue
            picture = np.asarray(photo[r0:r1:step, c0:c1:step]).copy()
            edge = outline[r0 // kf:r1 // kf, c0 // kf:c1 // kf]
            er, ec = np.nonzero(edge)
            pr, pc = (er * kf) // step, (ec * kf) // step
            ok = (pr < picture.shape[0]) & (pc < picture.shape[1])
            picture[pr[ok], pc[ok]] = (0.5 * picture[pr[ok], pc[ok]] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            image = Image.fromarray(picture)
            draw = ImageDraw.Draw(image)

            def px(points):
                q = np.asarray(points, float)
                return [((x - tx) / scale, (ty - y) / scale) for x, y in q]

            for colour, lines in ((None, strokes), ("edge", edges)):
                for s in lines:
                    q = np.asarray(s["points"], float)
                    if q[:, 0].max() < tx or q[:, 0].min() > tx + args.tile or q[:, 1].max() < ty - args.tile or q[:, 1].min() > ty:
                        continue
                    draw.line(px(q), fill=PAINT[colour or s["colour"]], width=max(int(round(s["width"] / scale)), 1))
            for x, y, size in found:
                if tx <= x < tx + args.tile and ty - args.tile < y <= ty:
                    (u, v), = px([[x, y]])
                    draw.ellipse([u - 25, v - 25, u + 25, v + 25], outline=(255, 0, 255), width=2)
            name = f"lines_{int(round(tx)):+05d}_{int(round(ty)):+05d}.jpg"
            image.save(os.path.join(args.out, name), quality=88)
            names.append((name, tx, ty, float(part.mean())))
    for name, tx, ty, share in names:
        print(f"  {name}: x {tx:.0f}..{tx + args.tile:.0f}, y {ty - args.tile:.0f}..{ty:.0f}, {share * 100:.0f}% paved")


if __name__ == "__main__":
    main()
