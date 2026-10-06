#!/usr/bin/env python3
"""Check a built scene against the rules it is meant to keep. Exits non-zero if one is broken.

    python -I check_scene.py SCENE_DIR

  - no tree trunk stands on the road, and no leaf reaches over it
  - no trunk stands inside a building
  - every cone stands on the road
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

    gv, gf = read_obj(os.path.join(scene, doc["collision"]["ground"]))
    e = np.sort(np.concatenate([gf[:, [0, 1]], gf[:, [1, 2]], gf[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e[:, 0] * len(gv) + e[:, 1], return_counts=True)
    x0, y0, x1, y1 = doc["frame"]["extent"]
    rim = int(2 * ((x1 - x0) + (y1 - y0)) / 8)
    once, many = int((counts == 1).sum()), int((counts > 2).sum())
    print(f"ground: {len(gf)} triangles, {once} edges used once ({rim} on the outer rim), {many} used more than twice")
    if once != rim or many:
        failed.append("ground mesh is not watertight")

    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))
    print("ok")


if __name__ == "__main__":
    main()
