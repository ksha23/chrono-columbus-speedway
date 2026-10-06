"""Pictures for judging the paint strokes by eye: the photo, and the photo with the strokes drawn on it.

    crop(photo, raster, strokes, cx, cy, width, height, zoom, path)
                                                             plain crop beside (or above) the same crop
                                                             with every centre line drawn one pixel wide
    overview(photo, raster, strokes, path)                   the whole site, strokes drawn fat
"""
import numpy as np
from PIL import Image, ImageDraw

INK = {"white": (255, 0, 200), "yellow": (0, 90, 255)}     # white paint in magenta, yellow paint in blue
END = (0, 255, 0)

# The windows the brief asks to be read: name, centre x, centre y, width, height (metres), zoom.
WINDOWS = [
    ("straight", -40.0, -70.0, 32.0, 14.0, 3),
    ("parallel_road", -55.0, 5.0, 32.0, 14.0, 3),
    ("pad_a", 146.0, 62.0, 32.0, 14.0, 3),
    ("pad_b", 132.0, 97.0, 32.0, 14.0, 3),
    ("pad_c", 135.0, 68.0, 32.0, 14.0, 3),
    ("pad_d", 172.0, 118.0, 32.0, 14.0, 3),
    ("pad_detail", 131.0, 68.0, 16.0, 7.0, 6),
    ("car_park", -232.0, -150.0, 32.0, 14.0, 3),
    ("car_park_detail", -212.0, -150.5, 16.0, 7.0, 6),
    ("tree_shadow", -15.0, -47.0, 32.0, 14.0, 3),
    ("north_east_loop", 200.0, 190.0, 32.0, 14.0, 3),
    ("north_east_detail", 200.0, 186.2, 8.0, 3.5, 12),
]


def to_pixel(raster, xy):
    """Scene metres to photo pixels (x right, y down, pixel centres at whole numbers)."""
    xy = np.asarray(xy, float)
    return np.stack([(xy[:, 0] - raster["x0"]) / raster["res"] - 0.5, (raster["y1"] - xy[:, 1]) / raster["res"] - 0.5], 1)


def crop(photo, raster, strokes, cx, cy, width, height, zoom, path, ends=True):
    res = raster["res"]
    c0 = int(round((cx - width / 2 - raster["x0"]) / res))
    r0 = int(round((raster["y1"] - cy - height / 2) / res))
    c1, r1 = c0 + int(round(width / res)), r0 + int(round(height / res))
    r0, c0 = max(r0, 0), max(c0, 0)
    plain = Image.fromarray(np.asarray(photo[r0:r1, c0:c1]))
    plain = plain.resize((plain.width * zoom, plain.height * zoom), Image.NEAREST)
    marked = plain.copy()
    draw = ImageDraw.Draw(marked)
    count = 0
    for s in strokes:
        p = (to_pixel(raster, s["points"]) - [c0, r0] + 0.5) * zoom - 0.5
        if p[:, 0].max() < 0 or p[:, 1].max() < 0 or p[:, 0].min() > marked.width or p[:, 1].min() > marked.height:
            continue
        count += 1
        draw.line([tuple(q) for q in p], fill=INK[s["colour"]], width=1)
        if ends:                                           # a dot at each end shows where a dash stops
            for q in (p[0], p[-1]):
                draw.point((q[0], q[1]), fill=END)
    if plain.width > 1000:                                 # wide crops go one above the other
        both = Image.new("RGB", (plain.width, plain.height * 2 + 8), (0, 0, 0))
        both.paste(marked, (0, plain.height + 8))
    else:
        both = Image.new("RGB", (plain.width * 2 + 8, plain.height), (0, 0, 0))
        both.paste(marked, (plain.width + 8, 0))
    both.paste(plain, (0, 0))
    both.save(path)
    return count


def overview(photo, raster, strokes, path, step=8):
    """The whole site at one pixel per `step` photo pixels, with every stroke drawn two pixels wide."""
    rows, cols = photo.shape[:2]
    small = np.zeros((rows // step, cols // step, 3), np.uint8)
    band = 64 * step
    for r in range(0, rows - rows % step, band):               # in bands: never the whole photo at once
        part = np.asarray(photo[r:min(r + band, rows - rows % step), :cols - cols % step])
        part = part.reshape(part.shape[0] // step, step, cols // step, step, 3).mean((1, 3))
        small[r // step:r // step + part.shape[0]] = (part * 0.55).astype(np.uint8)    # dimmed, so strokes stand out
    img = Image.fromarray(small)
    draw = ImageDraw.Draw(img)
    ink = {"white": (255, 255, 255), "yellow": (255, 220, 0)}
    for s in strokes:
        p = (to_pixel(raster, s["points"]) + 0.5) / step
        draw.line([tuple(q) for q in p], fill=ink[s["colour"]], width=2)
    img.save(path)


def write_all(photo, raster, strokes, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for name, cx, cy, w, h, zoom in WINDOWS:
        counts[name] = crop(photo, raster, strokes, cx, cy, w, h, zoom, os.path.join(out_dir, f"{name}.png"))
    overview(photo, raster, strokes, os.path.join(out_dir, "overview.png"))
    return counts
