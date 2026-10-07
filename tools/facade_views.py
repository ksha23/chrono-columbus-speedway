#!/usr/bin/env python3
"""Show what the scan has of each wall, with a metre grid: what facades.json was read from.

    python -I facade_views.py SCAN_DIR WORK_DIR REFERENCE_DIR OUT_DIR

One picture per wall of every building, as seen from outside: s along the wall from its left
end, z up from the lowest ground at the building. The title gives the scene point of the left
end and the direction of s, so a door seen at s is at (left end + s * direction): that point
is what goes into facades.json as "at".
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import buildings  # noqa: E402
import reference  # noqa: E402
import scanlib  # noqa: E402
import transform  # noqa: E402

PIX = 0.025         # metres a pixel covers
DENSITY = 5000.0    # points taken from the scan per square metre of it
BEHIND, BEFORE = 1.2, 1.5      # metres behind the wall and in front of it that are shown


def main():
    scan_dir, work, reference_dir, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    t = transform.Transform(work)
    v, vt, tri_v, tri_t, tri_mat, names = scanlib.load_scan(scan_dir, work)
    sx, sy = t.to_scene(v[:, 0], v[:, 1])
    p = np.stack([sx, sy, t.to_elevation(v[:, 0], v[:, 1], v[:, 2])], 1)
    middle = p[tri_v].mean(1)
    _, area = scanlib.triangle_normals(p, tri_v)
    obj = np.load(os.path.join(work, "objects.npz"))
    cell = float(obj["cell"])
    gy, gx = np.mgrid[0:obj["height"].shape[0], 0:obj["height"].shape[1]]
    gx, gy = float(obj["x0"]) + (gx + 0.5) * cell, float(obj["y1"]) - (gy + 0.5) * cell
    found = buildings.measure(obj["buildings"], obj["height"], gx, gy, reference.Reference(reference_dir))
    read = buildings.facades()
    textures = {}
    for n, b in enumerate(found, start=1):
        place, to_frame = buildings.frame(b)
        a, q = to_frame(middle[:, 0], middle[:, 1])
        sel = np.nonzero((np.abs(a) < b["half_length"] + 5) & (np.abs(q) < b["half_width"] + 5))[0]
        rng = np.random.default_rng(n)
        idx = np.repeat(sel, np.maximum((area[sel] * DENSITY).astype(int), 1))
        weight = scanlib.barycentric_samples(len(idx), rng)
        pos = (p[tri_v[idx]] * weight[:, :, None]).sum(1)
        uv = (vt[tri_t[idx]] * weight[:, :, None]).sum(1)
        colour = np.zeros((len(idx), 3), np.uint8)
        for m in np.unique(tri_mat[idx]):
            if m not in textures:
                textures[m] = np.asarray(Image.open(scanlib.texture_path(scan_dir, names[m])).convert("RGB"))
            k, tex = tri_mat[idx] == m, textures[m]
            colour[k] = tex[np.clip(((1 - uv[k, 1]) * tex.shape[0]).astype(int), 0, tex.shape[0] - 1), np.clip((uv[k, 0] * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)]
        fa, fq = to_frame(pos[:, 0], pos[:, 1])
        frame_pos = np.stack([fa, fq], 1)
        top = b["ground"] - b["ground_low"] + b["ridge"] + 0.5
        for k, wall in enumerate(buildings.outline(b, buildings.style_of(b, read, (0.7, 0.7, 0.7)))):
            rel = frame_pos - wall["start"]
            s, off, z = rel @ wall["along"], rel @ wall["out"], pos[:, 2] - b["ground_low"]
            keep = (off > -BEHIND) & (off < BEFORE) & (s > -1.0) & (s < wall["length"] + 1.0) & (z > -0.5) & (z < top)
            wide, tall = int((wall["length"] + 2.0) / PIX) + 1, int((top + 0.5) / PIX) + 1
            cols, rows = ((s[keep] + 1.0) / PIX).astype(int), ((top - z[keep]) / PIX).astype(int)
            order = np.argsort(off[keep])                     # nearest the eye last, so it wins
            picture, got = np.zeros((tall, wide, 3), np.uint8), np.zeros((tall, wide), bool)
            picture[rows[order], cols[order]] = colour[keep][order]
            got[rows[order], cols[order]] = True
            far, (near_r, near_c) = ndimage.distance_transform_edt(~got, return_indices=True)
            picture[far <= 2.5] = picture[near_r[far <= 2.5], near_c[far <= 2.5]]         # pin holes
            sheet = Image.fromarray(picture)
            draw = ImageDraw.Draw(sheet)
            for metre in range(0, int(wall["length"]) + 1):
                x = int((metre + 1.0) / PIX)
                draw.line([(x, tall - (14 if metre % 5 == 0 else 6)), (x, tall)], fill=(255, 255, 0))
                if metre % 5 == 0:
                    draw.text((x + 2, tall - 24), str(metre), fill=(255, 255, 0))
            for metre in range(0, int(top) + 1):
                y = int((top - metre) / PIX)
                draw.line([(0, y), (10, y)], fill=(0, 255, 255))
                draw.text((12, y - 5), str(metre), fill=(0, 255, 255))
            start = place(wall["start"][0], wall["start"][1], 0.0)
            ahead = place(*(wall["start"] + wall["along"]), 0.0) - start
            draw.text((60, 2), f"building_{n} wall {k}: left end ({start[0]:.2f}, {start[1]:.2f}), s toward ({ahead[0]:.3f}, {ahead[1]:.3f}), {wall['length']:.1f} m long", fill=(255, 255, 0))
            sheet.save(os.path.join(out_dir, f"building_{n}_wall_{k}.png"))
            print(f"building_{n} wall {k}: left end ({start[0]:.2f}, {start[1]:.2f}), s toward ({ahead[0]:.3f}, {ahead[1]:.3f}), {wall['length']:.1f} m long, {'gable' if wall['gable'] else 'eave'}")


if __name__ == "__main__":
    main()
