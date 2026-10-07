"""Accessible-parking symbols: the blue square with the wheelchair figure, found and redrawn.

The symbol is a square of blue paint a little over a metre on a side, with a white border and
the white figure in the middle. After some years in the sun the blue has faded to a pale
slate, a few grey levels bluer than the concrete, and the figure is a dozen pixels of
slightly whiter grey. Tracing finds the border as a small crooked square and the figure as a
tick or two.

Nothing about the symbol needs tracing. It is a standard sign: once it is known where a
square of faded blue lies and which way it faces, the whole of it can be drawn. So the
squares are found by their colour, and each is replaced by a blue square, a white border and
the figure, built from strokes like the rest of the paint.
"""
import numpy as np
from scipy import ndimage

import parallel
import pavement

BLUER = 9.0          # how much bluer than the pavement around it faded blue paint is, at the least
AS_BRIGHT = 0.86     # and how bright, as a share of that pavement: a shadow is blue too, and dark
AROUND = 4.0         # metres over which "the pavement around it" is taken
SIDE = (0.9, 1.8)    # metres: the sides a square of the symbol can have
FILLED = 0.72        # share of its own bounding square that such a patch fills
COLOURLESS = 0.10    # how far apart its red, green and blue may be, as a share of the brightest
ROOM = 0.8           # metres of plain pavement a symbol has on every side
BLUE = (0.20, 0.36, 0.66)     # the colour drawn: the paint as it was, not as it has faded
BORDER = 0.06        # metres, the white line round the square
FIGURE = 0.07        # metres, the figure's own line width in a square of one metre
RAISE = 0.01         # metres the white lies above the blue, so the two do not fight


def _oriented(rows, cols):
    """The square that best holds a patch: (centre row, centre column, side in cells, angle, fill)."""
    pts = np.stack([cols, rows], axis=1).astype(np.float64)
    best = None
    for angle in np.radians(np.arange(0.0, 90.0, 2.0)):
        u = np.array([np.cos(angle), np.sin(angle)])
        v = np.array([-u[1], u[0]])
        a, b = pts @ u, pts @ v
        wide, tall = a.max() - a.min() + 1.0, b.max() - b.min() + 1.0
        if best is None or wide * tall < best[0]:
            centre = (a.max() + a.min()) / 2 * u + (b.max() + b.min()) / 2 * v
            best = (wide * tall, centre, wide, tall, angle)
    area, centre, wide, tall, angle = best
    return centre[1], centre[0], wide, tall, angle, len(pts) / area


def find(photo, raster, distance, shadow=None, taken=None, unseen=None, log=print):
    """The symbols on the pavement: (list of dicts, full-size mask of the paint to take out).

    distance is the signed distance to the pavement's edge on pavement.RES cells, positive on
    the pavement. shadow is the matte of relit shadows and taken the mask of things already
    accounted for, both at the photo's own size. unseen marks, on the pavement's cells, what
    lies under a crown: leaves in shade are bluer than leaves in sun, and are not paint.
    Each symbol is a dict with x, y, side, and up, the unit vector from the figure's seat to
    its head: toward the kerb the stall ends on.
    """
    res = pavement.RES
    k = int(round(res / raster["res"]))
    paved = distance > 0.3
    h, w = paved.shape
    lum = np.zeros((h, w), np.float32)
    blue = np.zeros((h, w), np.float32)
    grey = np.zeros((h, w), bool)
    dark = np.zeros((h, w), bool) if unseen is None else unseen.copy()

    def band(r0):
        r1 = min(r0 + 512, h)
        block = np.asarray(photo[r0 * k:r1 * k, :w * k, :3], dtype=np.float32).reshape(r1 - r0, k, w, k, 3).mean((1, 3))
        lum[r0:r1] = block.mean(-1)
        blue[r0:r1] = block[..., 2] - (block[..., 0] + block[..., 1]) / 2
        grey[r0:r1] = (block.max(-1) - block.min(-1)) < COLOURLESS * block.max(-1)
        for mask, limit in ((shadow, 32), (taken, 0)):
            if mask is not None:
                dark[r0:r1] |= np.asarray(mask[r0 * k:r1 * k:k, :w * k:k]) > limit

    parallel.each(band, range(0, h, 512))           # each band has its own rows: side by side
    # The pavement around each cell: its usual brightness and blueness, shadows left out.
    plain = paved & ~dark
    weight = ndimage.uniform_filter(plain.astype(np.float32), int(AROUND / res) | 1)
    near = np.maximum(weight, 1e-3)
    usual_lum = ndimage.uniform_filter(lum * plain, int(AROUND / res) | 1) / near
    usual_blue = ndimage.uniform_filter(blue * plain, int(AROUND / res) | 1) / near
    faded = plain & grey & (weight > 0.3) & (blue > usual_blue + BLUER) & (lum > AS_BRIGHT * usual_lum)
    faded = ndimage.binary_opening(ndimage.binary_closing(faded, iterations=2), iterations=2)

    labels, count = ndimage.label(faded)
    symbols = []
    out = np.zeros(photo.shape[:2], bool)
    for n, sl in enumerate(ndimage.find_objects(labels), start=1):
        rows, cols = np.nonzero(labels[sl] == n)
        if not (SIDE[0] ** 2 * 0.6 <= len(rows) * res * res <= SIDE[1] ** 2):
            continue
        cr, cc, wide, tall, angle, fill = _oriented(rows + sl[0].start, cols + sl[1].start)
        side = (wide + tall) / 2 * res
        if not (SIDE[0] <= side <= SIDE[1]) or max(wide, tall) > 1.3 * min(wide, tall) or fill < FILLED:
            continue
        # A symbol is painted in a stall: plain pavement in plain view all round it. A paler
        # slab at the pavement's edge is a square of bluish grey too, with grass beside it.
        ring_r, ring_c = np.ogrid[-int(cr):h - int(cr), -int(cc):w - int(cc)]
        ring = (np.hypot(ring_r, ring_c) * res <= side / 2 + ROOM) & (np.hypot(ring_r, ring_c) * res >= 0.75 * side)
        if distance[int(cr), int(cc)] < side / 2 + ROOM or (plain & grey)[ring].mean() < 0.9:
            continue
        # Which of the square's four sides is the top: the one toward the kerb.
        ways = [np.array([np.cos(angle + q * np.pi / 2), -np.sin(angle + q * np.pi / 2)]) for q in range(4)]      # x east, y north
        reach = [float(ndimage.map_coordinates(distance, [[cr - 3.0 / res * way[1]], [cc + 3.0 / res * way[0]]], order=1, mode="nearest")[0]) for way in ways]
        up = ways[int(np.argmin(reach))]
        x, y = raster["x0"] + (cc + 0.5) * res, raster["y1"] - (cr + 0.5) * res
        symbols.append({"x": round(float(x), 3), "y": round(float(y), 3), "side": round(float(side), 2),
                        "up": [round(float(up[0]), 4), round(float(up[1]), 4)]})
        grow = int(round((side / 2 + 0.25) / raster["res"]))
        pr, pc = int((cr + 0.5) * k), int((cc + 0.5) * k)
        yy, xx = np.mgrid[-grow:grow + 1, -grow:grow + 1]
        # The square itself and a little beyond, turned as the square is.
        along = xx * np.cos(angle) + yy * np.sin(angle)
        across = -xx * np.sin(angle) + yy * np.cos(angle)
        box = (np.abs(along) <= grow * 0.92) & (np.abs(across) <= grow * 0.92)
        r0, c0 = max(pr - grow, 0), max(pc - grow, 0)
        part = box[r0 - (pr - grow):, c0 - (pc - grow):][:out.shape[0] - r0, :out.shape[1] - c0]
        out[r0:r0 + part.shape[0], c0:c0 + part.shape[1]] |= part
    symbols.sort(key=lambda s: (s["x"], s["y"]))
    # A sign is one size, and fading only ever makes its patch measure smaller.
    for symbol in symbols:
        symbol["side"] = max(s["side"] for s in symbols)
    log(f"  {len(symbols)} accessible-parking symbols found by their faded blue")
    return symbols, out


def _figure():
    """The wheelchair figure in a square of side one, as (points, width) with x right and y up."""
    turn = np.radians(np.arange(75.0, 351.0, 15.0))
    wheel = np.stack([-0.03 + 0.21 * np.cos(turn), -0.13 + 0.21 * np.sin(turn)], axis=1)
    ring = np.radians(np.arange(0.0, 361.0, 45.0))
    head = np.stack([0.03 + 0.03 * np.cos(ring), 0.37 + 0.03 * np.sin(ring)], axis=1)       # a ring as wide as its radius: a disc
    return [(head, 0.06), (np.array([[0.03, 0.27], [-0.02, 0.0]]), 0.09), (np.array([[0.0, 0.15], [0.21, 0.13]]), FIGURE),
            (np.array([[-0.02, 0.0], [0.20, 0.0], [0.30, -0.24], [0.40, -0.22]]), 0.08), (wheel, FIGURE)]


def strokes(symbol):
    """The paint of one symbol as strokes for markmodel: the blue square, its border and the figure."""
    centre = np.array([symbol["x"], symbol["y"]])
    up = np.array(symbol["up"])
    right = np.array([up[1], -up[0]])
    side = symbol["side"]

    def place(points, scale):
        return [[round(float(v), 3) for v in centre + scale * (p[0] * right + p[1] * up)] for p in points]

    half = side / 2 - BORDER / 2
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]])
    out = [{"colour": "blue", "width": side, "symbol": "accessible", "points": place(np.array([[-0.5, 0.0], [0.5, 0.0]]), side)},
           {"colour": "white", "width": BORDER, "raise": RAISE, "symbol": "accessible", "points": place(corners, 1.0)}]
    inner = 0.86 * (side - 2 * BORDER)
    for points, width in _figure():
        out.append({"colour": "white", "width": round(width * inner, 3), "raise": RAISE, "symbol": "accessible", "points": place(points, inner)})
    return out


def inside(symbol, points, margin=0.2):
    """Which of some points lie in a symbol's square, or within margin of it."""
    up = np.array(symbol["up"])
    right = np.array([up[1], -up[0]])
    rel = np.asarray(points, float) - np.array([symbol["x"], symbol["y"]])
    reach = symbol["side"] / 2 + margin
    return (np.abs(rel @ up) <= reach) & (np.abs(rel @ right) <= reach)
