#!/usr/bin/env python3
"""Find the paint on the pavement in the drone photo and describe it as vector strokes.

    find(photo, raster, paved, taken, height, cell, shadow=None) -> list of strokes
    python -I markings.py WORK_DIR OUT_DIR      writes WORK_DIR/markings.json and overlay pictures

Each stroke is {"colour": "yellow" or "white", "width": metres, "points": [[x, y], ...]} with
the points along the centre line in scene metres. A dash is its own stroke.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import markings_fit as fit          # noqa: E402
import markings_join as join        # noqa: E402
import markings_ridge as ridge      # noqa: E402

WIDE = 2.85                         # pixels across at half height: under this a 10 cm line, over it a 15 cm one


def seeds(photo, paved, taken, height, scale, window=None):
    """Rough lines to measure, best first: (kind, points, straight)."""
    pts = ridge.points(photo, paved, taken, height, scale, window)
    out = []
    for kind in ("white", "yellow"):
        for xy, strength in ridge.chains(pts[kind], kind, photo.shape[1]):
            for a, b, straight in ridge.pieces(xy):
                out.append((float(strength[a:b].sum()), kind, xy[a:b], straight))
    out.sort(key=lambda s: -s[0])
    return [s[1:] for s in out]


def measure(ground, xy, straight, kind):
    """One rough line to one measured stroke, straight if the paint is straight, else curved. None if neither fits."""
    def line_of(path):
        _, mid, along, half = fit.bow(path)
        return fit.fit_line(ground, mid, along, half, kind)

    def curve_or_line(points, line=None):
        curve = fit.fit_curve(ground, points, kind)
        if curve is None or fit.judge(curve):
            return line if line is not None else curve
        if curve["ring"] or fit.bow(curve["path"])[0] > 0.25:
            return curve                                       # the paint bends: keep the curve
        if line is None:
            line = line_of(curve["path"])
        if line is not None and not fit.judge(line) and line["length"] > 0.9 * curve["length"]:
            return line
        return curve

    if straight or len(xy) < 40:
        line = line_of(xy)
        if line is not None and not fit.judge(line):
            if line["length"] <= 60:
                return line
            t = np.arange(0.0, line["length"] + 1.0)[:, None] / line["length"]
            return curve_or_line(line["a"] + t * (line["b"] - line["a"]), line)     # long: is it really straight?
        if len(xy) < 12:
            return line
    return curve_or_line(xy)


def measure_all(ground, rough, log=None):
    """Fit every rough line that is not already covered by a stroke. Returns measured strokes (pixels).

    White goes first. Where a yellow line passes under a white one the white hides it, so the
    yellow fit treats those pixels as unseen rather than as the end of the line.
    """
    covered = np.zeros(ground.photo.shape[:2], bool)
    ground.used = covered
    strokes, why = [], {}
    for kind in ("white", "yellow"):
        refused = np.zeros(ground.photo.shape[:2], bool)       # measured already and found not to be paint
        for seed_kind, xy, straight in rough:
            if seed_kind != kind:
                continue
            r, c = np.rint(xy[:, 1]).astype(int), np.rint(xy[:, 0]).astype(int)
            if covered[r, c].mean() > 0.5 or refused[r, c].mean() > 0.5:
                continue
            st = measure(ground, xy, straight, kind)
            reason = "no fit" if st is None else fit.judge(st)
            why[kind + " " + (reason or "kept")] = why.get(kind + " " + (reason or "kept"), 0) + 1
            if log is not None and "each" in log:
                log["each"].append((kind, xy, st, reason))
            if reason:
                if st is not None:
                    fit.cover(refused, centre_line(st), 1)
                continue
            fit.cover(covered, centre_line(st), 1)
            strokes.append(st)
        if kind == "white":
            ground.hide = np.zeros(ground.photo.shape[:2], bool)
            for st in strokes:
                fit.cover(ground.hide, centre_line(st), 2)
    ground.hide = ground.used = None
    if log is not None:
        log.update(why)
    return strokes


def centre_line(st):
    return np.stack([st["a"], st["b"]]) if "a" in st else st["path"]


def to_scene(raster, xy):
    xy = np.asarray(xy, float)
    return np.stack([raster["x0"] + (xy[:, 0] + 0.5) * raster["res"], raster["y1"] - (xy[:, 1] + 0.5) * raster["res"]], 1)


def export(raster, it):
    """One stroke as the caller wants it: colour, painted width in metres, centre line in scene metres."""
    pts = it["pts"]
    if it["form"] != "line" and len(pts) > 2:
        pts = pts[ridge.simplify(pts, 0.3)]                        # corners stay: they are far from any chord
        if it["closed"]:
            pts = np.concatenate([pts[:-1], pts[:1]])
    width = 0.10 if it["width"] < WIDE else 0.15
    return {"colour": it["kind"], "width": width, "points": [[round(float(x), 4), round(float(y), 4)] for x, y in to_scene(raster, pts)]}


def find(photo, raster, paved, taken, height, cell, shadow=None, window=None, log=None):
    scale = int(round(cell / raster["res"]))
    ground = fit.Ground(photo, paved, taken, height, scale, shadow)
    strokes = measure_all(ground, seeds(photo, paved, taken, height, scale, window), log)
    its = join.tidy(ground, strokes, log)
    return [export(raster, it) for it in its]


def main():
    work, out_dir = sys.argv[1], sys.argv[2]
    cache = os.path.join(work, "cache") if os.path.isdir(os.path.join(work, "cache")) else work
    with open(os.path.join(cache, "raster.json")) as f:
        raster = json.load(f)
    photo = np.load(os.path.join(cache, "photo.npy"), mmap_mode="r")
    paved = np.load(os.path.join(cache, "pavement.npy"))          # the masks are read whole: other tools rewrite them,
    taken = np.load(os.path.join(cache, "taken.npy"))             # and a mapped file that changes underneath kills the run
    objects = np.load(os.path.join(cache, "objects.npz"))
    shadow_path = os.path.join(cache, "shadows", "photo_shadow.npy")
    shadow = np.load(shadow_path) if os.path.isfile(shadow_path) else None
    import resource
    import time
    started = time.time()
    log = {}
    strokes = find(photo, raster, paved, taken, objects["height"], float(objects["cell"]), shadow, log=log)
    with open(os.path.join(work, "markings.json"), "w") as f:
        json.dump(strokes, f)
    for colour in ("yellow", "white"):
        mine = [s for s in strokes if s["colour"] == colour]
        metres = sum(float(np.hypot(*np.diff(np.array(s["points"]), axis=0).T).sum()) for s in mine)
        print(f"{colour}: {len(mine)} strokes, {metres:.0f} m")
    print({k: v for k, v in log.items() if k[:5] not in ("white", "yello")})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    print(f"{time.time() - started:.0f} s, peak {peak:.0f} MB")
    import markings_overlay
    markings_overlay.write_all(photo, raster, strokes, out_dir)


if __name__ == "__main__":
    main()
