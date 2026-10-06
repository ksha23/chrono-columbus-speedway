#!/usr/bin/env python3
"""Find the trees, buildings and vehicles in the scan and save them for the scene build.

    python -I survey.py WORK_DIR REFERENCE_DIR [PREVIEW.png]

Writes WORK_DIR/objects.npz (the masks on 0.25 m cells, with the grid) and trees.json.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import objects  # noqa: E402
import reference  # noqa: E402


def main():
    work, ref_dir = sys.argv[1], sys.argv[2]
    raster = json.load(open(os.path.join(work, "raster.json")))
    ref = reference.Reference(ref_dir)
    surface = np.load(os.path.join(work, "surface.npy"), mmap_mode="r")
    photo = np.load(os.path.join(work, "photo.npy"), mmap_mode="r")
    factor = int(round(objects.CELL / raster["res"]))
    height, gx, gy = objects.height_above_ground(np.asarray(surface), raster, ref)
    small = np.stack([objects.block(np.asarray(photo[..., ch], dtype=np.float32), factor, "mean") for ch in range(3)], -1)
    masks = objects.classify(height, small)
    trees, crowns = objects.find_trees(height, masks["trees"], gx, gy)

    valid = np.isfinite(height)
    area = objects.CELL**2
    print(f"scan covers {valid.sum() * area / 1e4:.1f} ha")
    for name in ("buildings", "trees", "vehicles"):
        print(f"  {name:10s} {masks[name].sum() * area:8.0f} m2")
    hs = np.array([t["height"] for t in trees])
    rs = np.array([t["radius"] for t in trees])
    print(f"  {len(trees)} trees: height median {np.median(hs):.1f} m, 90th percentile {np.percentile(hs, 90):.1f}, tallest {hs.max():.1f};"
          f" crown radius median {np.median(rs):.1f} m, largest {rs.max():.1f}")
    for lo, hi in [(1.5, 3), (3, 6), (6, 10), (10, 15), (15, 40)]:
        print(f"    {lo:4.1f} to {hi:4.1f} m: {int(((hs >= lo) & (hs < hi)).sum())}")
    open_ground = valid & ~masks["trees"] & ~masks["buildings"] & ~masks["vehicles"]
    print(f"  open ground: scan top minus lidar, median {np.nanmedian(height[open_ground]):+.3f} m,"
          f" 5th/95th percentile {np.nanpercentile(height[open_ground], 5):+.2f} / {np.nanpercentile(height[open_ground], 95):+.2f} m")

    np.savez_compressed(os.path.join(work, "objects.npz"), height=height, crowns=crowns,
                        buildings=masks["buildings"], trees=masks["trees"], vehicles=masks["vehicles"],
                        x0=raster["x0"], y1=raster["y1"], cell=objects.CELL)
    json.dump(trees, open(os.path.join(work, "trees.json"), "w"))

    if len(sys.argv) > 3:
        view = (small * 0.55).astype(np.uint8)
        view[masks["trees"]] = (0.5 * view[masks["trees"]] + np.array([0, 110, 0])).astype(np.uint8)
        view[masks["buildings"]] = (0.4 * view[masks["buildings"]] + np.array([150, 0, 0])).astype(np.uint8)
        view[masks["vehicles"]] = (0.4 * view[masks["vehicles"]] + np.array([0, 0, 160])).astype(np.uint8)
        for t in trees:
            c = int((t["x"] - raster["x0"]) / objects.CELL)
            r = int((raster["y1"] - t["y"]) / objects.CELL)
            view[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = (255, 255, 0)
        Image.fromarray(view).save(sys.argv[3])


if __name__ == "__main__":
    main()
