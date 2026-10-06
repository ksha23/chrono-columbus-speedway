"""Pictures for judging what barriers.py found: each line drawn on the drone photo.

write(work_dir, out_dir, barriers, rejected) makes

    overview.png            the whole site at 0.4 m per pixel, every barrier numbered
    overview_rejected.png   the same with the chains that were turned down, for hunting misses
    NN_type.png             one barrier at 0.1 m per pixel: the bare photo beside the photo
                            with the line on it
    NN_type_pK.png          the same at the photo's own 0.05 m, in stretches of at most 40 m

Guard rails are magenta. The fence is cyan where it was seen and yellow where it was only
inferred, with its gate posts ringed in red, and is shown in 100 m stretches at 0.1 m per
pixel. Vertices are dotted so that a wandering line shows.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

COLOURS = {"guardrail": (255, 0, 255), "fence": (0, 255, 255)}
REJECTED = (255, 160, 0)
INFERRED = (255, 255, 0)  # stretches of the fence that were not seen
GATE = (255, 0, 0)        # gate posts, ringed
LOOP_STRETCH = 100.0      # metres, side of a picture along the fence
MARGIN = 6.0              # metres of photo kept around a barrier
STRETCH = 40.0            # metres, side of a close-up


def _grid(work_dir):
    with open(os.path.join(work_dir, "raster.json")) as f:
        raster = json.load(f)
    return float(raster["x0"]), float(raster["y1"]), float(raster["res"])


def _crop(photo, grid, x_lo, x_hi, y_lo, y_hi, factor):
    """The photo between the given scene bounds, averaged over factor x factor pixels.
    Returns (image array, x of its left edge, y of its top edge, metres per pixel)."""
    x0, y1, res = grid
    c0 = max(0, int(np.floor((x_lo - x0) / res)))
    c1 = min(photo.shape[1], int(np.ceil((x_hi - x0) / res)))
    r0 = max(0, int(np.floor((y1 - y_hi) / res)))
    r1 = min(photo.shape[0], int(np.ceil((y1 - y_lo) / res)))
    r1 -= (r1 - r0) % factor
    c1 -= (c1 - c0) % factor
    block = np.asarray(photo[r0:r1, c0:c1])
    if factor > 1:
        block = block.reshape((r1 - r0) // factor, factor, (c1 - c0) // factor, factor, 3).mean(axis=(1, 3))
    return block.astype(np.uint8), x0 + c0 * res, y1 - r0 * res, res * factor


def _draw(draw, points, left, top, scale, colour, width=1, dots=True, seen=None, gates=()):
    xy = [((x - left) / scale, (top - y) / scale) for x, y in points]
    if seen is None:
        draw.line(xy, fill=colour, width=width)
    else:
        for a, b, shown in zip(xy[:-1], xy[1:], seen):
            draw.line([a, b], fill=colour if shown else INFERRED, width=width)
    for gate in gates:
        for x, y in gate:
            x, y = (x - left) / scale, (top - y) / scale
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=GATE, width=2)
    if dots:
        for x, y in xy:
            draw.rectangle([x - 1, y - 1, x + 1, y + 1], outline=(255, 255, 0))


def _pair(bare, marked):
    """The bare photo beside (or above) the marked one, whichever wastes less space."""
    gap = np.full((bare.shape[0], 4, 3), 255, np.uint8)
    if bare.shape[1] <= bare.shape[0] * 1.3:
        return np.concatenate([bare, gap, marked], axis=1)
    gap = np.full((4, bare.shape[1], 3), 255, np.uint8)
    return np.concatenate([bare, gap, marked], axis=0)


def _length(points):
    p = np.asarray(points, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def _along(points, spacing):
    """Points every `spacing` metres along a polyline, starting half a spacing in."""
    p = np.asarray(points, float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    n = max(1, int(np.ceil(s[-1] / spacing)))
    t = (np.arange(n) + 0.5) * s[-1] / n
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def overview(photo, grid, barriers, rejected, path, show_rejected):
    x0, y1, res = grid
    factor = 8
    rows = photo.shape[0] // factor * factor
    cols = photo.shape[1] // factor * factor
    small = np.empty((rows // factor, cols // factor, 3), np.uint8)
    for r in range(0, rows, 1600):
        block = np.asarray(photo[r:min(rows, r + 1600), :cols])
        small[r // factor:(r + block.shape[0]) // factor] = block.reshape(
            block.shape[0] // factor, factor, cols // factor, factor, 3).mean(axis=(1, 3))
    image = Image.fromarray(small)
    draw = ImageDraw.Draw(image)
    scale = res * factor
    if show_rejected:
        for points, why in rejected:
            _draw(draw, points, x0, y1, scale, REJECTED, width=1, dots=False)
            if _length(points) >= 5.0:
                mid = _along(points, 1e9)[0]
                draw.text(((mid[0] - x0) / scale + 3, (y1 - mid[1]) / scale), why, fill=REJECTED)
    for n, b in enumerate(barriers):
        _draw(draw, b["points"], x0, y1, scale, COLOURS[b["type"]], width=2, dots=False,
              seen=b.get("seen"), gates=b.get("gates", ()))
    for n, b in enumerate(barriers):
        mid = _along(b["points"], 1e9)[0]
        px, py = (mid[0] - x0) / scale, (y1 - mid[1]) / scale
        draw.text((px + 5, py - 12), str(n), fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    for gx in range(-350, 400, 50):
        px = (gx - x0) / scale
        draw.line([(px, 0), (px, 8)], fill=(255, 255, 255))
        draw.text((px + 2, 0), str(gx), fill=(255, 255, 255))
    for gy in range(250, -300, -50):
        py = (y1 - gy) / scale
        draw.line([(0, py), (8, py)], fill=(255, 255, 255))
        draw.text((10, py - 5), str(gy), fill=(255, 255, 255))
    image.save(path)


def one(photo, grid, barrier, others, path, factor, bounds, title):
    bare, left, top, scale = _crop(photo, grid, *bounds, factor)
    marked = Image.fromarray(bare.copy())
    draw = ImageDraw.Draw(marked)
    for other in others:
        _draw(draw, other["points"], left, top, scale, COLOURS[other["type"]], dots=False, seen=other.get("seen"))
    _draw(draw, barrier["points"], left, top, scale, COLOURS[barrier["type"]],
          seen=barrier.get("seen"), gates=barrier.get("gates", ()))
    draw.text((4, 2), title, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    # A ten-metre bar and the scene coordinates of the crop's corner, to measure against.
    bar = 10.0 / scale
    h = marked.height
    draw.line([(6, h - 8), (6 + bar, h - 8)], fill=(255, 255, 255), width=2)
    draw.text((10 + bar, h - 14), f"10 m   top-left ({left:.1f}, {top:.1f})", fill=(255, 255, 255),
              stroke_width=2, stroke_fill=(0, 0, 0))
    Image.fromarray(_pair(bare, np.asarray(marked))).save(path)


def write(work_dir, out_dir, barriers, rejected=()):
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(out_dir)):          # last run's pictures would mislead
        if name.endswith(".png") and (name[:2].isdigit() or name.startswith("overview")):
            os.remove(os.path.join(out_dir, name))
    grid = _grid(work_dir)
    photo = np.load(os.path.join(work_dir, "photo.npy"), mmap_mode="r")
    overview(photo, grid, barriers, rejected, os.path.join(out_dir, "overview.png"), False)
    overview(photo, grid, barriers, rejected, os.path.join(out_dir, "overview_rejected.png"), True)
    for n, b in enumerate(barriers):
        pts = np.asarray(b["points"], float)
        title = f"{n} {b['type']}  {_length(pts):.1f} m long  {b['height']:.2f} m tall  conf {b['confidence']:.2f}"
        bounds = (pts[:, 0].min() - MARGIN, pts[:, 0].max() + MARGIN,
                  pts[:, 1].min() - MARGIN, pts[:, 1].max() + MARGIN)
        others = [o for o in barriers if o is not b]
        if "seen" in b:                               # the fence: too big for one picture
            half = LOOP_STRETCH / 2
            for k, (cx, cy) in enumerate(_along(pts, LOOP_STRETCH * 0.9)):
                one(photo, grid, b, others, os.path.join(out_dir, f"{n:02d}_fence_p{k:02d}.png"), 2,
                    (cx - half, cx + half, cy - half, cy + half), f"fence, stretch {k}")
            continue
        one(photo, grid, b, others, os.path.join(out_dir, f"{n:02d}_{b['type']}.png"), 2, bounds, title)
        half = min(STRETCH, max(16.0, _length(pts) + 2 * MARGIN)) / 2
        for k, (cx, cy) in enumerate(_along(pts, STRETCH)):
            one(photo, grid, b, others, os.path.join(out_dir, f"{n:02d}_{b['type']}_p{k}.png"), 1,
                (cx - half, cx + half, cy - half, cy + half), f"{title}  stretch {k}")
