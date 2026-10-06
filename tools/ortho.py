#!/usr/bin/env python3
"""Rasterise the scan from above: an orthophoto, a surface height map and a slope map.

    python -I ortho.py SCAN_DIR WORK_DIR [metres_per_pixel]

Writes WORK_DIR/ortho.png, dsm.npy (float32, NaN where empty) and ortho_meta.json. The top
surface wins where the mesh overlaps itself, so tree crowns hide the ground under them.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scanlib  # noqa: E402


def main():
    scan, work = sys.argv[1], sys.argv[2]
    res = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    v, vt, tri_v, tri_t, tri_mat, names = scanlib.load_scan(scan, work)
    lo, hi = v.min(0), v.max(0)
    W = int(np.ceil((hi[0] - lo[0]) / res)) + 1
    H = int(np.ceil((hi[1] - lo[1]) / res)) + 1
    rgb = np.zeros((H, W, 3), np.uint8)
    dsm = np.full((H, W), -np.inf, np.float32)
    _, area = scanlib.triangle_normals(v, tri_v)
    rng = np.random.default_rng(0)

    # Enough random samples per triangle to leave no holes: about 6 per pixel it covers.
    per_px = 6.0
    for m, name in enumerate(names):
        sel = np.nonzero(tri_mat == m)[0]
        if not len(sel):
            continue
        tex = np.asarray(Image.open(scanlib.texture_path(scan, name)).convert("RGB"))
        th, tw = tex.shape[:2]
        count = np.maximum(3, np.ceil(area[sel] / (res * res) * per_px)).astype(np.int64)
        which = np.repeat(sel, count)
        for part in np.array_split(np.arange(len(which)), max(1, len(which) // 4_000_000)):
            t = which[part]
            w3 = scanlib.barycentric_samples(len(t), rng)
            p = (v[tri_v[t]] * w3[:, :, None]).sum(1)
            uv = (vt[tri_t[t]] * w3[:, :, None]).sum(1)
            px = np.clip(((p[:, 0] - lo[0]) / res).astype(np.int64), 0, W - 1)
            py = np.clip(((hi[1] - p[:, 1]) / res).astype(np.int64), 0, H - 1)
            tx = np.clip((uv[:, 0] * tw).astype(np.int64), 0, tw - 1)
            ty = np.clip(((1.0 - uv[:, 1]) * th).astype(np.int64), 0, th - 1)
            z = p[:, 2].astype(np.float32)
            order = np.argsort(z, kind="stable")  # ascending, so the highest sample is written last
            px, py, z, col = px[order], py[order], z[order], tex[ty[order], tx[order]]
            keep = z >= dsm[py, px]
            px, py, z, col = px[keep], py[keep], z[keep], col[keep]
            dsm[py, px] = z
            rgb[py, px] = col
        del tex
        print(f"  {name}: {len(sel)} triangles, {len(which)} samples", flush=True)

    dsm[~np.isfinite(dsm)] = np.nan
    Image.fromarray(rgb).save(os.path.join(work, "ortho.png"))
    np.save(os.path.join(work, "dsm.npy"), dsm)
    meta = {"res": res, "x0": float(lo[0]), "y1": float(hi[1]), "width": W, "height": H}
    with open(os.path.join(work, "ortho_meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"ortho {W}x{H} at {res} m/px, covered {100 * np.isfinite(dsm).mean():.1f}%")


if __name__ == "__main__":
    main()
