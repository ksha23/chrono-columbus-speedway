"""Find the traffic cones the drone photographed, so they can be painted out and stood back up.

A cone is a small patch of strong, unnatural colour on pavement: lime green or orange where
concrete has almost no colour at all. The scan flattened each one into a smear a few tens of
centimetres across, but left its shadow on the pavement beside it. The patch says where the
cone stood and what colour it was. The length of the shadow says how tall it was.
"""
import numpy as np
from scipy import ndimage

MARGIN = 1.0          # metres beyond the pavement's edge still searched
MIN_AREA, MAX_AREA = 0.0075, 0.8   # square metres of coloured patch that can be one cone
# Hue range in degrees, then the least saturation and brightness that count. The green cones
# photographed dark and dull on top with only a bright rim, so green is taken more loosely.
# That is safe because a green patch must also throw a shadow to count as a cone.
# Some of the tall cones photographed a dull brick red, far from the orange of the small ones.
# That is taken more loosely too, and on the same condition: it must throw a shadow. A hue
# range that starts above where it ends runs through 360.
KINDS = {"green": (95.0, 165.0, 0.30, 0.30), "orange": (5.0, 40.0, 0.55, 0.45), "red": (345.0, 28.0, 0.33, 0.30)}

LIME, ORANGE, RED = (0.35, 0.85, 0.25), (0.95, 0.35, 0.10), (0.80, 0.24, 0.12)
# Cones come in standard heights. Each is listed with the longest shadow that still counts as
# that size. The flight was near solar noon in mid-September, sun about 49 degrees up, so a
# shadow is about 0.87 of the cone's height, and its thin tip fades out of the photo early.
SIZES = [(0.33, 0.30), (0.48, 0.46), (0.70, 0.71), (9.9, 0.91)]
REACH = 1.5           # metres around a patch searched for its shadow


def hue_sat(rgb):
    """Hue in degrees, saturation and brightness, for a float RGB array in 0..1."""
    mx, mn = rgb.max(-1), rgb.min(-1)
    span = np.maximum(mx - mn, 1e-6)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.where(mx == r, (g - b) / span % 6, np.where(mx == g, (b - r) / span + 2, (r - g) / span + 4)) * 60.0
    return hue, (mx - mn) / np.maximum(mx, 1e-6), mx


def coloured_patches(photo, raster, search_cells, cell):
    """Full-size arrays: where cone-coloured pixels are, and which colour each one is (1, 2, ...)."""
    res = raster["res"]
    k = int(round(cell / res))
    H, W = photo.shape[:2]
    mask = np.zeros((H, W), bool)
    kind = np.zeros((H, W), np.uint8)
    for r0 in range(0, H, 2048):
        r1 = min(r0 + 2048, H)
        a, b = r0 // k, min((r1 + k - 1) // k, search_cells.shape[0])
        if not search_cells[a:b].any():
            continue
        where = np.repeat(np.repeat(search_cells[a:b], k, axis=0), k, axis=1)[r0 - a * k:r1 - a * k, :W]
        where = np.pad(where, ((0, 0), (0, W - where.shape[1])))
        hue, sat, val = hue_sat(np.asarray(photo[r0:r1], dtype=np.float32) / 255)
        for n, (lo, hi, min_sat, min_val) in enumerate(KINDS.values(), start=1):
            in_range = (hue >= lo) & (hue <= hi) if lo <= hi else (hue >= lo) | (hue <= hi)
            hit = where & in_range & (sat > min_sat) & (val > min_val) & (kind[r0:r1] == 0)
            mask[r0:r1] |= hit
            kind[r0:r1][hit] = n
    # A green cone is a bright rim around a dark middle: close the rim and count the middle in.
    return ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2)), kind


def measure(window_rgb, patch, kinds, res):
    """Look at one patch in its surroundings. Returns None if it is not a cone.

    window_rgb is the photo around the patch as floats, patch the patch's own pixels in that
    window, kinds the colour code of every pixel. Returns (shadow length in metres, the cone's
    colour as one of the names in KINDS, the shadow's pixels).
    """
    _, sat, val = hue_sat(window_rgb)
    near = int(round(0.6 / res))
    ring = ndimage.binary_dilation(patch, iterations=near) & ~ndimage.binary_dilation(patch, iterations=int(round(0.15 / res)))
    # A cone stands on pavement, so the ground right around it has no colour. This is what
    # tells a lime cone from a tuft of grass at the pavement's edge.
    if ((sat < 0.16) & (val > 0.3))[ring].mean() < 0.7:
        return None

    # The shadow: dark pavement joined on to the patch. Its far end is the shadow of the tip.
    dark = val < 0.6 * np.median(val[ring])
    joined, _ = ndimage.label(dark | patch)
    rows, cols = np.nonzero(patch)
    shadow = (joined == joined[rows[0], cols[0]]) & dark & ~patch
    length = 0.0
    if shadow.sum() >= 3:
        sr, sc = np.nonzero(shadow)
        length = float(np.hypot(sr - rows.mean(), sc - cols.mean()).max() * res)

    green = int((kinds[patch] == 1).sum())
    orange = int((kinds[patch] == 2).sum())
    red = int((kinds[patch] == 3).sum())
    if orange == 0 and red > green:
        # Brick red on pavement is also a rust stain or a dead leaf. Only a shadow of some
        # length makes it a cone.
        if shadow.sum() < 8 or length < 0.25:
            return None
        return length, "red", shadow
    if orange == 0:
        # Two other things are green patches ringed by pavement: weeds in the joints, and
        # squares painted on the pad to mark where cones go. Neither stands up. A cone throws a
        # shadow, and from above its top shows dark inside the bright rim of its base, where a
        # painted square has plain pavement inside it.
        middle = patch & (kinds == 0)
        dark_middle = middle.sum() >= 2 and np.median(val[middle]) < 0.6 * np.median(val[ring])
        if shadow.sum() < 5 and not dark_middle:
            return None
    # A small cone is an orange dot inside a thin green edge. Anything larger with a wide green
    # rim is a green cone, whatever else is on it: some carry a red pointer.
    return length, "green" if green * res * res >= 0.06 or orange == 0 else "orange", shadow


def find(photo, raster, paved_cells, blocked_cells, cell):
    """Return a list of cones (dicts with x, y, height, colours) and a full-size mask to paint out.

    The mask holds each cone's coloured patch and the shadow found beside it.

    paved_cells and blocked_cells are boolean on the low-resolution grid of size cell: where the
    pavement is, and where something else stands (trees, buildings, vehicles).
    """
    res = raster["res"]
    search = ndimage.binary_dilation(ndimage.binary_closing(paved_cells, iterations=4), iterations=int(round(MARGIN / cell))) & ~blocked_cells
    mask, kind = coloured_patches(photo, raster, search, cell)
    labels, count = ndimage.label(mask)
    H, W = mask.shape
    pad = int(round(REACH / res))
    cones = []
    remove = np.zeros((H, W), bool)
    for n, sl in enumerate(ndimage.find_objects(labels), start=1):
        area = (labels[sl] == n).sum() * res * res
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        r0, r1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        c0, c1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        patch = labels[r0:r1, c0:c1] == n
        seen = measure(np.asarray(photo[r0:r1, c0:c1], dtype=np.float32) / 255, patch, kind[r0:r1, c0:c1], res)
        if seen is None:
            continue
        length, colour, shadow = seen
        green = colour == "green"
        height = next(h for limit, h in SIZES if length <= limit)
        if area < 0.05 and colour != "red":
            # A tall cone has a wide base. A small patch with a long dark streak beside it is a
            # small cone standing next to a crack or a pole's shadow. (A red one shows only
            # part of itself as red, so its patch says nothing about its base.)
            height = min(height, 0.46)
        if length == 0.0 and green:
            height = 0.71     # no shadow found, so go by what the other green ones measured
        rows, cols = np.nonzero(patch)
        small = height <= 0.30
        remove[r0:r1, c0:c1] |= patch | shadow
        cones.append({"x": float(raster["x0"] + (c0 + cols.mean() + 0.5) * res), "y": float(raster["y1"] - (r0 + rows.mean() + 0.5) * res),
                      "height": height, "shadow": round(length, 2),
                      "body": RED if colour == "red" else ORANGE if (small or not green) else LIME,
                      "base": LIME if (small or green or colour == "red") else ORANGE})
    return cones, remove
