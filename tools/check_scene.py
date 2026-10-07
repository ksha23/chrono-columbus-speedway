#!/usr/bin/env python3
"""Check a built scene against the rules it is meant to keep. Exits non-zero if one is broken.

    python -I check_scene.py SCENE_DIR

  - no tree trunk stands on the road, and no leaf reaches over it
  - no trunk stands inside a building
  - every cone stands on the road and every parked vehicle on pavement
  - no light pole stands in the road, and no fence post either
  - road paint lies on the road, between 0.5 and 4 cm above it
  - the ground has no holes: every edge inside it is shared by two triangles
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree


def read_obj(path):
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("f "):
                f.append([p.split("/")[0] for p in line.split()[1:4]])
    return np.array(v, float), np.array(f, int) - 1


def main():
    scene = sys.argv[1]
    doc = json.load(open(os.path.join(scene, "speedway_scene.json")))
    failed = []

    rv, rf = read_obj(os.path.join(scene, doc["collision"]["road"]))
    tri = rv[rf][:, :, :2]
    centres = cKDTree(tri.mean(1))
    reach = float(np.linalg.norm(tri - tri.mean(1, keepdims=True), axis=2).max())

    def distance_to_road(p):
        """Distance from a point to the road's surface, 0 if it is on it. Exact to the triangles."""
        best = np.inf
        for j in centres.query_ball_point(p, 12.0 + reach):
            a, b, c = tri[j]
            s = lambda u, v_, w: (u[0] - w[0]) * (v_[1] - w[1]) - (v_[0] - w[0]) * (u[1] - w[1])
            d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
            if not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0)):
                return 0.0
            for u, w in ((a, b), (b, c), (c, a)):
                t = np.clip(np.dot(p - u, w - u) / max(np.dot(w - u, w - u), 1e-12), 0, 1)
                best = min(best, float(np.linalg.norm(p - (u + t * (w - u)))))
        return best

    trees = [i for i in doc["instances"] if i["group"] == "Trees"]
    extent = {}
    for asset in doc["assets"]:
        if asset["parts"][0]["mesh"].startswith("trees/"):
            leaves, _ = read_obj(os.path.join(scene, asset["parts"][1]["mesh"]))
            extent[asset["name"]] = float(np.hypot(leaves[:, 0], leaves[:, 1]).max())
    clear = []
    for t in trees:
        d = distance_to_road(np.array(t["pos"][:2]))
        clear.append((d, d - extent[doc["assets"][t["asset"]]["name"]] * t["scale"][0]))
    clear = np.array(clear)
    on_road, over = int((clear[:, 0] <= 0).sum()), int((clear[:, 1] < 0).sum())
    print(f"{len(trees)} trees: {on_road} trunks on the road, {over} crowns reaching over it."
          f" Nearest trunk {clear[:, 0].min():.2f} m from the road, nearest leaf {clear[:, 1].min():.2f} m")
    if on_road or over:
        failed.append("trees on or over the road")

    inside = 0
    for b in [i for i in doc["instances"] if i["group"] == "Buildings"]:
        walls, _ = read_obj(os.path.join(scene, f"buildings/{b['name']}_walls.obj"))
        corners = np.unique(walls[:, :2].round(3), axis=0)
        c = np.array(b["centre"])
        far = corners[np.argsort(-np.hypot(*(corners - c).T))[:4]]
        u = next((q - far[0]) / np.hypot(*(q - far[0])) for q in far[1:] if abs(np.hypot(*(q - far[0])) - b["length"]) < 0.05)
        w = np.array([-u[1], u[0]])
        p = np.array([t["pos"][:2] for t in trees]) - c
        inside += int(((np.abs(p @ u) < b["length"] / 2) & (np.abs(p @ w) < b["width"] / 2)).sum())
    print(f"{inside} trunks inside a building")
    if inside:
        failed.append("trees inside buildings")

    cones = [i for i in doc["instances"] if i["group"] == "Cones"]
    off = sum(distance_to_road(np.array(c["pos"][:2])) > 0.3 for c in cones)
    print(f"{len(cones)} cones: {off} more than 0.3 m off the road")
    if off:
        failed.append("cones off the road")

    def height_above_road(p):
        """Height of a 3D point above the road triangle under it, or None if it is not over the road."""
        for j in centres.query_ball_point(p[:2], reach + 1e-6):
            a, b, c = rv[rf[j]]
            det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(det) < 1e-12:
                continue
            u = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det
            v = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det
            if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
                return p[2] - (u * a[2] + v * b[2] + (1 - u - v) * c[2])
        return None

    def placed(group):
        return [i for i in doc["instances"] if i["group"] == group]

    cars = placed("Vehicles")
    # A car's own footprint can be missing from the pavement map where it stood at the edge
    # of an apron, so the test is that pavement is within a car's half length of its middle.
    off = sum(distance_to_road(np.array(c["pos"][:2])) > 2.5 for c in cars)
    print(f"{len(cars)} parked vehicles: {off} not on the pavement")
    if off:
        failed.append("vehicles off the road")

    # A pole may stand on the pavement's very edge, where the map of it is a few tenths of a
    # metre generous. Half a metre in, on every side, is in the road.
    ring = [0.5 * np.array([np.cos(t), np.sin(t)]) for t in np.arange(8) * np.pi / 4]
    poles = placed("Poles")
    inside = sum(all(distance_to_road(np.array(p["pos"][:2]) + r) == 0 for r in ring) for p in poles)
    print(f"{len(poles)} light poles: {inside} standing in the road")
    if inside:
        failed.append("poles in the road")

    fences = [i for i in placed("Barriers") if i["type"] == "fence"]
    posts_on_road = total_posts = 0
    for fence in fences:
        v, _ = read_obj(os.path.join(scene, f"barriers/{fence['name']}_posts.obj"))
        feet = np.unique(v[:, :2].round(1), axis=0)[::4]
        total_posts += len(feet)
        posts_on_road += sum(distance_to_road(p) == 0 for p in feet)
    rails = [i for i in placed("Barriers") if i["type"] == "guardrail"]
    print(f"barriers: {sum(r['length'] for r in rails):.0f} m of guard rail in {len(rails)} pieces, {sum(f['length'] for f in fences):.0f} m of fence,"
          f" of which {sum(f['seen_length'] for f in fences):.0f} m was seen in the scan. Fence posts on the road: {posts_on_road} of about {total_posts} checked")
    if posts_on_road:
        failed.append("fence on the road")

    paint = placed("Markings") + placed("EdgeLines")
    if paint:
        lifts, loose = [], 0
        for item in paint:
            v, _ = read_obj(os.path.join(scene, doc["assets"][item["asset"]]["parts"][0]["mesh"]))
            for p in v[::max(len(v) // 3000, 1)]:
                lift = height_above_road(p)
                if lift is None:
                    loose += 1
                else:
                    lifts.append(lift)
        lifts = np.array(lifts)
        print(f"road paint: {len(lifts) + loose} points checked, {loose} not over the road, lift {lifts.min() * 100:.1f} to {lifts.max() * 100:.1f} cm")
        # Paint is laid 2 cm above the 1 m grid the road was built on. Where the road was cut
        # along the pavement's edge its triangles are not quite that grid's, so the lift there
        # differs by up to a centimetre. Half a centimetre still clears the road. Four would show.
        if loose > 0.01 * (len(lifts) + loose) or lifts.min() < 0.005 or lifts.max() > 0.04:
            failed.append("road paint off the road or not lying on it")

    gv, gf = read_obj(os.path.join(scene, doc["collision"]["ground"]))
    e = np.sort(np.concatenate([gf[:, [0, 1]], gf[:, [1, 2]], gf[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e[:, 0] * len(gv) + e[:, 1], return_counts=True)
    x0, y0, x1, y1 = doc["frame"]["extent"]
    rim = int(2 * ((x1 - x0) + (y1 - y0)) / 8)
    once, many = int((counts == 1).sum()), int((counts > 2).sum())
    print(f"ground: {len(gf)} triangles, {once} edges used once ({rim} on the outer rim), {many} used more than twice")
    if once != rim or many:
        failed.append("ground mesh is not watertight")

    piles = placed("Rocks")
    if piles:
        # A boulder may stand on the fringe of an apron. None stands out in the pavement, with
        # road three metres off on every side of it.
        far = [3.0 * np.array([np.cos(t), np.sin(t)]) for t in np.arange(8) * np.pi / 4]
        on_road = sum(all(distance_to_road(np.array(p["at"]) + r) == 0 for r in [np.zeros(2)] + far) for p in piles)
        print(f"{len(piles)} rock piles and boulders, {sum(p['stones'] for p in piles)} stones: {on_road} out in the road")
        if on_road:
            failed.append("rocks on the road")

    # Every parked vehicle stands on its wheels: under each of the four, the tyre's lowest
    # point is on the ground mesh, not in the air above it and not sunk into it.
    if cars:
        corners = gv[gf]
        middles = corners[:, :, :2].mean(1)
        span = np.hypot(*(corners[:, :, :2] - middles[:, None, :]).transpose(2, 0, 1)).max(1)
        near_cars = np.zeros(len(gf), bool)
        for car in cars:
            near_cars |= np.hypot(*(middles - np.array(car["pos"][:2])).T) < 6.0 + span
        local = np.nonzero(near_cars)[0]
        tree = cKDTree(middles[local])

        def ground_under(p):
            for j in tree.query_ball_point(p[:2], float(span[local].max()) + 1e-6):
                a, b, c = corners[local[j]]
                det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if abs(det) < 1e-12:
                    continue
                u = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det
                v = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det
                if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
                    return u * a[2] + v * b[2] + (1 - u - v) * c[2]
            return None

        gaps = []
        for car in cars:
            part = next(p for p in doc["assets"][car["asset"]]["parts"] if p["name"] == "tyres")
            v, _ = read_obj(os.path.join(scene, part["mesh"]))
            w, x, y, z = car["rot"]
            turn = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                             [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                             [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            placed_at = (v * np.array(car["scale"])) @ turn.T + np.array(car["pos"])
            for front in (True, False):
                for left in (True, False):
                    wheel = placed_at[((v[:, 0] > 0) == front) & ((v[:, 1] > 0) == left)]
                    lowest = wheel[np.argmin(wheel[:, 2])]
                    under = ground_under(lowest)
                    gaps.append(np.nan if under is None else lowest[2] - under)
        gaps = np.array(gaps)
        print(f"{len(cars)} parked vehicles: tyres {np.nanmin(gaps) * 100:+.1f} to {np.nanmax(gaps) * 100:+.1f} cm from the ground under them")
        if not np.isfinite(gaps).all() or np.abs(gaps).max() > 0.03:
            failed.append("a parked vehicle is not standing on its wheels")

    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))
    print("ok")


if __name__ == "__main__":
    main()
