#!/usr/bin/env python3
"""Redraw the scan from straight above in the scene frame: a photo and a top-surface height map.

    python -I rasterize.py SCAN_DIR WORK_DIR [metres_per_pixel] [--layout-only]

Every triangle of the scan is moved into the scene frame (plane map and height correction from
transform.py) and sampled densely. A pixel takes the average colour of the samples that fall in
it and lie on its top surface, which is a box filter and so does not alias when the output is
coarser than the scan's own texels (about 2 cm).

Writes WORK_DIR/photo.npy (uint8), surface.npy (float32 elevation, NaN where the scan has
nothing) and raster.json, which also lists the texture tiles the scan reaches.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout  # noqa: E402
import scanlib  # noqa: E402
import transform  # noqa: E402

SAMPLES_PER_PIXEL = 10.0
CHUNK = 6_000_000
TOP = 0.25   # samples within this height of a pixel's highest sample count as its top surface
MARGIN = 8.0


def fill_pinholes(photo, top):
    """Fill single uncovered pixels that random sampling happened to miss, from their neighbours.

    With about ten samples a pixel a few in every hundred thousand get none. Only holes with
    covered pixels on most sides are filled, so the scan's real edge is left alone. In place.
    """
    H, W = top.shape
    filled = 0
    for r0 in range(0, H, 2048):   # in bands with a one-pixel overlap, to bound memory
        a, b = max(r0 - 1, 0), min(r0 + 2049, H)
        t = top[a:b]
        have = np.isfinite(t)
        pad = np.pad(have, 1)
        count = sum(pad[1 + dy:1 + dy + have.shape[0], 1 + dx:1 + dx + have.shape[1]].astype(np.uint8)
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
        rows, cols = np.nonzero(~have & (count >= 6))
        keep = (rows + a >= r0) & (rows + a < r0 + 2048)
        rows, cols = rows[keep], cols[keep]
        if not len(rows):
            continue
        acc = np.zeros((len(rows), 3), np.float32)
        zacc = np.zeros(len(rows), np.float32)
        n = np.zeros(len(rows), np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                rr, cc = np.clip(rows + dy, 0, have.shape[0] - 1), np.clip(cols + dx, 0, W - 1)
                ok = have[rr, cc]
                acc[ok] += photo[a:b][rr[ok], cc[ok]]
                zacc[ok] += t[rr[ok], cc[ok]]
                n[ok] += 1
        photo[rows + a, cols] = (acc / n[:, None] + 0.5).astype(np.uint8)
        top[rows + a, cols] = zacc / n
        filled += len(rows)
    return filled


def main():
    scan, work = sys.argv[1], sys.argv[2]
    res = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    v, vt, tri_v, tri_t, tri_mat, names = scanlib.load_scan(scan, work)
    t = transform.Transform(work)
    sx, sy = t.to_scene(v[:, 0], v[:, 1])
    p = np.stack([sx, sy, t.to_elevation(v[:, 0], v[:, 1], v[:, 2])], axis=1)
    _, area = scanlib.triangle_normals(p, tri_v)

    # Tiles the scan reaches, from triangle centroids weighted by area.
    c = p[tri_v].mean(1)
    ti, tj = np.floor(c[:, 0] / layout.TILE).astype(int), np.floor(c[:, 1] / layout.TILE).astype(int)
    covered = {}
    for i, j, a in zip(ti, tj, area):
        covered[(i, j)] = covered.get((i, j), 0.0) + a
    tiles = sorted((int(i), int(j)) for (i, j), a in covered.items() if a > 0.03 * layout.TILE**2)
    print(f"{len(tiles)} tiles hold scan data (of {len(covered)} touched)")

    x0 = min(i for i, _ in tiles) * layout.TILE - MARGIN
    x1 = (max(i for i, _ in tiles) + 1) * layout.TILE + MARGIN
    y0 = min(j for _, j in tiles) * layout.TILE - MARGIN
    y1 = (max(j for _, j in tiles) + 1) * layout.TILE + MARGIN
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    print(f"raster {W} x {H} at {res} m, x {x0:.0f}..{x1:.0f}, y {y0:.0f}..{y1:.0f}")
    json.dump({"res": res, "x0": x0, "y1": y1, "width": W, "height": H, "tiles": [list(k) for k in tiles]},
              open(os.path.join(work, "raster.json"), "w"), indent=1)
    if "--layout-only" in sys.argv:
        return

    def samples(m, seed):
        """Yield (flat pixel index, z, texture row, texture col) for material m, in chunks."""
        sel = np.nonzero(tri_mat == m)[0]
        if not len(sel):
            return
        rng = np.random.default_rng(seed)
        count = np.maximum(3, np.ceil(area[sel] / (res * res) * SAMPLES_PER_PIXEL)).astype(np.int64)
        which = np.repeat(sel, count)
        for part in np.array_split(np.arange(len(which)), max(1, len(which) // CHUNK)):
            tr = which[part]
            w3 = scanlib.barycentric_samples(len(tr), rng)
            q = np.einsum("nk,nkd->nd", w3, p[tri_v[tr]])
            px = ((q[:, 0] - x0) / res).astype(np.int64)
            py = ((y1 - q[:, 1]) / res).astype(np.int64)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            uv = np.einsum("nk,nkd->nd", w3[ok], vt[tri_t[tr[ok]]])
            yield py[ok] * W + px[ok], q[ok, 2].astype(np.float32), uv

    top = np.full(W * H, -np.inf, np.float32)
    for m in range(len(names)):
        for idx, z, _ in samples(m, m):
            np.maximum.at(top, idx, z)
        print(f"  heights {m + 1}/{len(names)}", end="\r", flush=True)
    print()

    total = np.zeros((W * H, 3), np.float32)
    hits = np.zeros(W * H, np.float32)
    for m, name in enumerate(names):
        tex = np.asarray(Image.open(scanlib.texture_path(scan, name)).convert("RGB"))
        th, tw = tex.shape[:2]
        for idx, z, uv in samples(m, m):   # the same seed, so the same samples as the first pass
            keep = z >= top[idx] - TOP
            idx, uv = idx[keep], uv[keep]
            tx = np.clip((uv[:, 0] * tw).astype(np.int64), 0, tw - 1)
            ty = np.clip(((1.0 - uv[:, 1]) * th).astype(np.int64), 0, th - 1)
            col = tex[ty, tx].astype(np.float32)
            for ch in range(3):
                np.add.at(total[:, ch], idx, col[:, ch])
            np.add.at(hits, idx, 1.0)
        del tex
        print(f"  colour {m + 1}/{len(names)}", end="\r", flush=True)
    print()

    have = hits > 0
    photo = np.zeros((W * H, 3), np.uint8)
    photo[have] = np.clip(total[have] / hits[have, None] + 0.5, 0, 255).astype(np.uint8)
    top[~have] = np.nan
    photo, top = photo.reshape(H, W, 3), top.reshape(H, W)
    print(f"filled {fill_pinholes(photo, top)} pinholes")
    np.save(os.path.join(work, "photo.npy"), photo)
    np.save(os.path.join(work, "surface.npy"), top)
    print(f"covered {100 * have.mean():.1f}% of the raster, median {np.median(hits[have]):.0f} samples per covered pixel")


if __name__ == "__main__":
    main()
