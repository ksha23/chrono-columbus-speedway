#!/usr/bin/env python3
"""Look for what should not be in the finished ground photo: dark marks left on the pavement.

    python -I audit_ground.py WORK_DIR OUT_DIR

Reads WORK_DIR/ground_photo.npy and pavement.npy (build_scene.py --save-photo writes both).
Pavement is one material, so anything on it much darker than the pavement around it is a
shadow that was missed, a crack, a tyre mark or a stain. This finds them, measures each one's
length, width and darkness, and writes a map and a sheet of the longest for a person to judge.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pavement  # noqa: E402

DARK = 0.80       # a pixel counts when it is below this share of the pavement tone around it
TONE = 2.0        # metres over which that tone is taken


def darkness(photo, paved_fine, res):
    """Luminance over local pavement tone, at the pavement map's resolution. NaN off the pavement."""
    k = int(round(pavement.RES / res))
    h, w = paved_fine.shape
    lum = np.zeros((h, w), np.float32)
    for r0 in range(0, h, 512):
        r1 = min(r0 + 512, h)
        block = np.asarray(photo[r0 * k:r1 * k, :w * k], dtype=np.float32).reshape(r1 - r0, k, w, k, 3).mean((1, 3))
        lum[r0:r1] = block @ np.array([0.30, 0.59, 0.11], np.float32)
    inside = ndimage.binary_erosion(paved_fine, iterations=int(round(0.4 / pavement.RES)))
    weight = inside.astype(np.float32)
    sigma = TONE / pavement.RES
    tone = ndimage.gaussian_filter(lum * weight, sigma) / np.maximum(ndimage.gaussian_filter(weight, sigma), 1e-3)
    # Again without the dark pixels themselves, so a wide shadow does not drag its own reference down.
    weight = (inside & (lum > 0.9 * tone)).astype(np.float32)
    seen = ndimage.gaussian_filter(weight, sigma)
    tone = np.where(seen > 0.05, ndimage.gaussian_filter(lum * weight, sigma) / np.maximum(seen, 1e-3), tone)
    ratio = lum / np.maximum(tone, 1.0)
    ratio[~inside] = np.nan
    return ratio


def main():
    work, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    raster = json.load(open(os.path.join(work, "raster.json")))
    photo = np.load(os.path.join(work, "ground_photo.npy"), mmap_mode="r")
    paved = np.load(os.path.join(work, "pavement.npy"))
    # Nothing on the ground is black. A black patch is a fill that drew no colour in.
    black = sum(int((np.asarray(photo[r0:r0 + 1024]).max(-1) < 40).sum()) for r0 in range(0, photo.shape[0], 1024))
    print(f"near-black ground anywhere in the photo: {black * raster['res'] ** 2:.1f} m2")
    ratio = darkness(photo, paved, raster["res"])
    dark = np.nan_to_num(ratio, nan=1.0) < DARK
    labels, count = ndimage.label(dark, structure=np.ones((3, 3)))
    area_total = np.isfinite(ratio).sum() * pavement.RES**2
    print(f"pavement checked: {area_total:.0f} m2. Darker than {DARK:.2f} of the tone around: {dark.sum() * pavement.RES**2:.1f} m2 in {count} marks")

    marks = []
    for n, sl in enumerate(ndimage.find_objects(labels), start=1):
        rows, cols = np.nonzero(labels[sl] == n)
        if len(rows) < 8:
            continue
        pts = np.stack([cols, rows], 1).astype(np.float64)
        pts -= pts.mean(0)
        evals = np.linalg.eigvalsh(np.cov(pts.T) + 1e-9 * np.eye(2))
        length = 4 * np.sqrt(evals[1]) * pavement.RES
        width = len(rows) * pavement.RES**2 / max(length, pavement.RES)
        x = raster["x0"] + (sl[1].start + cols.mean() + 0.5) * pavement.RES
        y = raster["y1"] - (sl[0].start + rows.mean() + 0.5) * pavement.RES
        marks.append({"x": float(x), "y": float(y), "length": float(length), "width": float(width), "area": float(len(rows) * pavement.RES**2),
                      "darkness": float(np.median(ratio[sl][labels[sl] == n]))})
    marks.sort(key=lambda m: -m["area"])
    for lo, hi, what in [(0.0, 0.12, "under 12 cm wide (cracks, joints)"), (0.12, 0.5, "12 to 50 cm wide (pole and rail shadows, tyre marks)"), (0.5, 99, "over 50 cm wide (shadows, stains, patches)")]:
        sel = [m for m in marks if lo <= m["width"] < hi and m["length"] > 1.0]
        print(f"  {what}: {len(sel)} marks over 1 m long, {sum(m['area'] for m in sel):.1f} m2")
    json.dump(marks[:400], open(os.path.join(out, "marks.json"), "w"))

    # A map of where they are, and a sheet of the largest.
    step = 4
    small = np.asarray(photo[::step * 2, ::step * 2]).copy()
    d = dark[::step, ::step]
    grown = ndimage.binary_dilation(dark, iterations=3)[::step, ::step]
    hh, ww = min(small.shape[0], grown.shape[0]), min(small.shape[1], grown.shape[1])
    small = small[:hh, :ww]
    small[grown[:hh, :ww]] = (255, 0, 255)
    Image.fromarray(small).save(os.path.join(out, "marks_map.png"))

    res = raster["res"]
    size = int(8.0 / res)
    cols_n = 8
    pick = [m for m in marks if m["length"] > 1.0][:40]
    sheet = np.zeros(((len(pick) + cols_n - 1) // cols_n * size, cols_n * size, 3), np.uint8)
    for q, m in enumerate(pick):
        c = int((m["x"] - raster["x0"]) / res) - size // 2
        r = int((raster["y1"] - m["y"]) / res) - size // 2
        patch = np.asarray(photo[max(r, 0):r + size, max(c, 0):c + size])
        sheet[(q // cols_n) * size:(q // cols_n) * size + patch.shape[0], (q % cols_n) * size:(q % cols_n) * size + patch.shape[1]] = patch
        print(f"   {q:2d}: at ({m['x']:7.1f}, {m['y']:7.1f})  {m['length']:5.1f} m long, {m['width']:.2f} m wide, {m['area']:5.1f} m2, {m['darkness']:.2f} of tone")
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2), Image.BOX).save(os.path.join(out, "marks_sheet.png"))


if __name__ == "__main__":
    main()
