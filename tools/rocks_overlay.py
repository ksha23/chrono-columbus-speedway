#!/usr/bin/env python3
"""Pictures for judging what rocks.py found: every pile and boulder drawn on the drone photo.

    python -I rocks_overlay.py WORK_DIR OUT_DIR

Reads WORK_DIR/photo.npy, raster.json, pavement.npy and objects.npz, runs rocks.find on the
whole site and writes

    overview.png   the whole site at 0.4 m per pixel, every find circled and numbered
    sheet.png      one 12 m crop of the photo per find, numbered the same way, with its outline
                   and what is to be painted out
    rocks.json     the finds as rocks.find returned them, in the same order

Piles are drawn in magenta and boulders in cyan. The edge of what would be painted out is
drawn in thin yellow. Each crop has a one metre bar.
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rocks  # noqa: E402

COLOURS = {"pile": (255, 0, 255), "boulder": (0, 255, 255)}
PAINT = (255, 255, 0)   # the edge of the patch to paint out
CROP = 12.0             # metres, side of a crop on the sheet
ZOOM = 2                # crop pixels per photo pixel
ACROSS = 5              # crops per row of the sheet
FACTOR = 8              # photo pixels per overview pixel


def overview(photo, raster, found, path):
    """The whole site, reduced, with a numbered ring round every find."""
    rows = photo.shape[0] // FACTOR * FACTOR
    cols = photo.shape[1] // FACTOR * FACTOR
    small = np.empty((rows // FACTOR, cols // FACTOR, 3), np.uint8)
    for r in range(0, rows, 1600):
        block = np.asarray(photo[r:min(rows, r + 1600), :cols, :3])
        small[r // FACTOR:(r + block.shape[0]) // FACTOR] = block.reshape(block.shape[0] // FACTOR, FACTOR, cols // FACTOR, FACTOR, 3).mean(axis=(1, 3))
    image = Image.fromarray(small)
    draw = ImageDraw.Draw(image)
    scale = raster["res"] * FACTOR
    for gx in range(-350, 400, 50):
        px = (gx - raster["x0"]) / scale
        draw.line([(px, 0), (px, 8)], fill=(255, 255, 255))
        draw.text((px + 2, 0), str(gx), fill=(255, 255, 255))
    for gy in range(250, -300, -50):
        py = (raster["y1"] - gy) / scale
        draw.line([(0, py), (8, py)], fill=(255, 255, 255))
        draw.text((10, py - 5), str(gy), fill=(255, 255, 255))
    for n, rock in enumerate(found):
        px, py = (rock["x"] - raster["x0"]) / scale, (raster["y1"] - rock["y"]) / scale
        radius = max(7.0, 0.5 * rock["length"] / scale + 4.0)
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], outline=COLOURS[rock["kind"]], width=2)
        draw.text((px + radius + 2, py - 6), str(n), fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    image.save(path)


def crop(photo, raster, mask, found, n):
    """One find in the middle of a CROP-sized window of the photo, outlined."""
    res = raster["res"]
    rock = found[n]
    half = int(round(CROP / res / 2))
    row, col = int((raster["y1"] - rock["y"]) / res), int((rock["x"] - raster["x0"]) / res)
    r0, c0 = row - half, col - half
    a0, a1 = max(r0, 0), min(r0 + 2 * half, photo.shape[0])
    b0, b1 = max(c0, 0), min(c0 + 2 * half, photo.shape[1])
    window = np.zeros((2 * half, 2 * half, 3), np.uint8)
    window[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = photo[a0:a1, b0:b1, :3]
    painted = np.zeros((2 * half, 2 * half), bool)
    painted[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = mask[a0:a1, b0:b1]
    big = np.repeat(np.repeat(window, ZOOM, axis=0), ZOOM, axis=1)
    edge = np.repeat(np.repeat(painted, ZOOM, axis=0), ZOOM, axis=1)
    big[edge & ~ndimage.binary_erosion(edge)] = PAINT
    image = Image.fromarray(big)
    draw = ImageDraw.Draw(image)
    left, top = raster["x0"] + c0 * res, raster["y1"] - r0 * res

    def to_pixels(points):
        return [((x - left) / res * ZOOM, (top - y) / res * ZOOM) for x, y in points]

    for m, other in enumerate(found):
        if abs(other["x"] - rock["x"]) < CROP and abs(other["y"] - rock["y"]) < CROP:
            draw.line(to_pixels(other["outline"]), fill=COLOURS[other["kind"]], width=1)
            if m != n:
                x, y = to_pixels([(other["x"], other["y"])])[0]
                draw.text((x, y), str(m), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    title = f"{n} {rock['kind']} ({rock['x']:.1f}, {rock['y']:.1f})  {rock['length']:.1f} x {rock['width']:.1f} m  {rock['area']:.1f} m2"
    draw.text((4, 2), title, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    bar = 1.0 / res * ZOOM
    draw.line([(6, image.height - 8), (6 + bar, image.height - 8)], fill=(255, 255, 255), width=2)
    draw.text((10 + bar, image.height - 14), "1 m", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return image


def sheet(photo, raster, mask, found, path):
    """Every find's crop, in order, ACROSS to a row."""
    side = int(round(CROP / raster["res"])) * ZOOM
    down = max(1, -(-len(found) // ACROSS))
    page = Image.new("RGB", (ACROSS * (side + 4), down * (side + 4)), (255, 255, 255))
    for n in range(len(found)):
        page.paste(crop(photo, raster, mask, found, n), ((n % ACROSS) * (side + 4), (n // ACROSS) * (side + 4)))
    page.save(path)


def main():
    work, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(work, "raster.json")) as f:
        raster = json.load(f)
    photo = np.load(os.path.join(work, "photo.npy"), mmap_mode="r")
    paved = np.load(os.path.join(work, "pavement.npy"))
    obj = np.load(os.path.join(work, "objects.npz"))
    start = time.time()
    found, mask = rocks.find(photo, raster, paved, obj["height"], obj["trees"], obj["buildings"], float(obj["cell"]))
    print(f"found in {time.time() - start:.1f} s")
    with open(os.path.join(out, "rocks.json"), "w") as f:
        json.dump(found, f, indent=1)
    overview(photo, raster, found, os.path.join(out, "overview.png"))
    sheet(photo, raster, mask, found, os.path.join(out, "sheet.png"))
    for n, rock in enumerate(found):
        print(f"{n:3d} {rock['kind']:8s} ({rock['x']:7.1f}, {rock['y']:7.1f})  {rock['length']:5.1f} x {rock['width']:4.1f} m  {rock['area']:5.1f} m2")


if __name__ == "__main__":
    main()
