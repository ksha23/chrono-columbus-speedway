"""Notches and bumps taken out of the pavement's outline along the roads.

A road's edge runs on. Where the outline traced from the photo steps aside and comes back
within a few metres it is following something else: the shadow of a bush, grass grown over
the kerb, the mouth of a footpath, gravel spilt off a verge. The road's surface is cut along
that outline and a white line is painted beside it, so every one of them shows.

Each such stretch is cut out of the outline and its two ends are joined by a smooth curve
that leaves and arrives along the edge's own direction. The pavement is then made to match:
what the curve takes in becomes pavement, what it leaves out becomes verge.

Only a kink is treated: a sharp turn one way answered by a sharp turn the other way within a
few metres, with the edge heading much the same way after it as before (kinks.py). A
junction's corner is left alone, and so is the nose of the grass between two roads, which is
a notch in the pavement in every way but its size. So is the edge of an apron.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal

import edgelines
import kinks
import pavement
import railmodel

STEP = kinks.STEP
NET = 30.0         # degrees the edge may have turned across a kink. More is a corner, left alone
DEEPEST = 2.5      # metres the outline may stand off the curve that replaces it
CALM = 2.5         # metres over which the outline is evened before turns are measured
WIDE = 12.0        # metres: pavement where a disc of this radius fits is an apron, not a junction
SURE = 1.0         # square metres of ground in plain view that are not argued with
# Where the outline was never seen: under a crown, in a hole in the scan.
BESIDE = 1.25      # metres from unseen ground within which the outline is the outline of that, not of the road
GLIMPSE = 1.0      # metres of edge in plain view that count as seen. Less, between two crowns, is not
BACK = 0.5         # metres of seen edge next to a crown that are not trusted: its fringe overhangs
LOOK = 30.0        # metres of outline either side of an unseen stretch searched for seen edge
ENOUGH = 2.0       # metres of seen edge needed on each side
UNSEEN = 60.0      # metres: the longest stretch carried across
TRUE = 0.35        # metres (rms) the seen edge may stray from the curve fitted through it
ASTRAY = 6.0       # metres: a guess further than this from the curve is a side road, not a guess

def _between(old, new, raster, shape):
    """The cells between two lines with the same ends, as (row slice, column slice, mask)."""
    res = pavement.RES
    ring = np.concatenate([old, new[::-1]])
    cols, rows = (ring[:, 0] - raster["x0"]) / res, (raster["y1"] - ring[:, 1]) / res
    c0, r0 = max(int(cols.min()) - 1, 0), max(int(rows.min()) - 1, 0)
    c1, r1 = min(int(cols.max()) + 2, shape[1]), min(int(rows.max()) + 2, shape[0])
    picture = Image.new("L", (c1 - c0, r1 - r0), 0)
    # A cell is in when its centre is: the picture's pixels are centred on whole numbers.
    ImageDraw.Draw(picture).polygon([(float(c - c0 - 0.5), float(r - r0 - 0.5)) for c, r in zip(cols, rows)], fill=1)
    return slice(r0, r1), slice(c0, c1), np.asarray(picture, dtype=bool)


def _outlines(paved, raster, pad):
    """Every outline of the pavement long enough to matter: (path as traced, the same evened, closed)."""
    res = pavement.RES
    signed = (ndimage.distance_transform_edt(paved) - ndimage.distance_transform_edt(~paved)) * res
    field = ndimage.gaussian_filter(signed.astype(np.float32), 1.0)
    grid = edgelines.GRID
    calm = int(CALM / STEP) | 1
    for line in edgelines.contours(field[::grid, ::grid], 0.0):
        xy = np.stack([raster["x0"] + (line[:, 1] * grid + 0.5) * res, raster["y1"] - (line[:, 0] * grid + 0.5) * res], axis=1)
        if railmodel.length_of(xy) < 4 * kinks.LONGEST:
            continue
        closed = np.allclose(xy[0], xy[-1])
        path = railmodel.resample(xy, STEP)
        if closed:
            # Walk a closed outline once and a bit, so a stretch across the seam is seen whole.
            body = path[:-1]
            reach = min(pad, len(body))
            path = np.concatenate([body[-reach:], body, body[:reach]])
        yield path, signal.savgol_filter(path, calm, 2, axis=0, mode="interp"), closed


def _curve_through(points):
    """A gentle curve through some points of an edge: (origin, axis, coefficients, rms misfit).

    A parabola in the frame of the points' own long axis, fitted again and again without
    the points that stand furthest off it: a bush on the kerb takes a notch out of the seen
    edge, and the curve has to pass it by.
    """
    origin = points.mean(0)
    _, _, axes = np.linalg.svd(points - origin, full_matrices=False)
    axis = axes[0]
    across = np.array([-axis[1], axis[0]])
    x, y = (points - origin) @ axis, (points - origin) @ across
    keep = np.ones(len(x), bool)
    for _ in range(4):
        fit = np.polyfit(x[keep], y[keep], 2)
        off = np.abs(y - np.polyval(fit, x))
        keep = off <= max(2.0 * np.sqrt(np.mean(off[keep] ** 2)), 0.1)
        if keep.mean() < 0.6:
            return origin, axis, fit, np.inf          # no one curve that most of the edge lies on
    return origin, axis, fit, float(np.sqrt(np.mean(off[keep] ** 2)))


def carry_on(paved, raster, unseen):
    """The outline redrawn wherever it was never seen. Returns (new mask, stretches redrawn).

    Under a tree's crown the pavement's edge is a guess, and the guess is a bite out of the
    road where the crown hid it or a bulge into the verge where the road was carried on too
    freely. But the edge is in view before the crown and again after it, if only in glimpses
    between one bush and the next, and a kerb runs on smoothly through all of them. So a
    gentle curve is fitted through what was seen of the edge either side of each unseen
    stretch, and the stretch is redrawn along it. unseen marks, on the pavement's cells,
    ground the photo does not show.
    """
    res = pavement.RES
    # The outline round a crown runs a little clear of it, where the pavement comes back into view.
    near = ndimage.binary_dilation(unseen, iterations=int(round(BESIDE / res)))
    look, back = int(round(LOOK / STEP)), int(round(BACK / STEP))
    pad = int(round((UNSEEN + 2 * LOOK) / STEP))
    flip = np.zeros_like(paved)
    count = 0
    for path, even, closed in _outlines(paved, raster, pad):
        r = np.clip(((raster["y1"] - path[:, 1]) / res).astype(int), 0, paved.shape[0] - 1)
        c = np.clip(((path[:, 0] - raster["x0"]) / res).astype(int), 0, paved.shape[1] - 1)
        # A glimpse of the edge shorter than GLIMPSE is not one, and the ends of each are not trusted.
        seen = ndimage.binary_opening(~near[r, c], structure=np.ones(int(round(GLIMPSE / STEP))))
        trusted = ndimage.binary_erosion(seen, structure=np.ones(2 * back + 1))
        labels, _ = ndimage.label(~seen)
        for sl, in ndimage.find_objects(labels):
            a, b = sl.start - 1, sl.stop
            if a - look < 0 or b + look >= len(path) or (b - a) * STEP > UNSEEN:
                continue
            if closed and not (pad <= sl.start < len(path) - pad):
                continue
            before = np.nonzero(trusted[a - look:a + 1])[0] + a - look
            after = np.nonzero(trusted[b:b + look + 1])[0] + b
            # The nearest seen edge on each side says most: fifteen metres of it at the outside.
            need = int(round(15.0 / STEP))
            before, after = before[-need:], after[:need]
            if min(len(before), len(after)) * STEP < ENOUGH:
                continue
            origin, axis, fit, misfit = _curve_through(even[np.concatenate([before, after])])
            if misfit > TRUE:
                continue                                    # the seen edge is not one smooth edge: a corner
            across = np.array([-axis[1], axis[0]])
            x0, x1 = (even[a] - origin) @ axis, (even[b] - origin) @ axis
            x = np.linspace(x0, x1, max(int(abs(x1 - x0) / STEP), 2) + 1)
            curve = origin + x[:, None] * axis + np.polyval(fit, x)[:, None] * across
            old = path[a:b + 1]
            off = kinks.apart(old, curve)
            if off < res or off > ASTRAY:
                continue
            rows, cols, mask = _between(old, curve, raster, paved.shape)
            flip[rows, cols] ^= mask
            count += 1
    return paved ^ flip, count


def _once(paved, raster, plain=None):
    """One pass over every outline. Returns (new mask, kinks taken out)."""
    res = pavement.RES
    signed = (ndimage.distance_transform_edt(paved) - ndimage.distance_transform_edt(~paved)) * res
    # An apron's edge has real corners in it and is left alone. A junction is not an apron
    # here, wide as it is: only pavement a disc of radius WIDE fits on, and what is round it.
    core = signed >= WIDE
    apron = ndimage.distance_transform_edt(~core) * res <= WIDE + edgelines.STOP if core.any() else np.zeros_like(paved)
    flip = np.zeros_like(paved)
    count = 0
    pad = int(round((kinks.LONGEST + 2 * kinks.LEAD) / STEP))
    for path, even, closed in _outlines(paved, raster, pad):
        for a, b in kinks.spans(even, NET):
            if closed and not (pad <= a < len(path) - pad):
                continue
            r = np.clip(((raster["y1"] - path[a:b + 1, 1]) / res).astype(int), 0, paved.shape[0] - 1)
            c = np.clip(((path[a:b + 1, 0] - raster["x0"]) / res).astype(int), 0, paved.shape[1] - 1)
            if apron[r, c].any():
                continue
            bridge = kinks.bridge(even, a, b)
            old = path[a:b + 1].copy()
            old[0], old[-1] = bridge[0], bridge[-1]
            # A notch is shallow. Anything the curve would stand far off is something else.
            off = kinks.apart(old, bridge)
            if off > DEEPEST or off < res:
                continue
            rows, cols, mask = _between(old, bridge, raster, paved.shape)
            if plain is not None and mask.sum() * res * res >= SURE and plain[rows, cols][mask].mean() >= 0.5:
                continue                # the photo shows this ground plainly, and it is what it looks like
            flip[rows, cols] ^= mask
            count += 1
    return paved ^ flip, count


def unkink(distance, raster, log=print, unseen=None, plain=None):
    """Tidy the roads' outlines. Returns (signed distance, pavement mask).

    Kinks are taken out first. Then, if unseen is given, the stretches nobody saw are redrawn
    from the seen edge either side of them, which is a truer guide with its notches gone.
    Then the kinks that leaves. plain marks ground the photo shows plainly, sunlit and even:
    a wedge of mown grass between two roads is a notch in the pavement in every way but
    that one, and is left as it is.
    """
    res = pavement.RES
    paved = distance > 0
    new, total = _once(paved, raster, plain)
    carried = 0
    if unseen is not None:
        new, carried = carry_on(new, raster, unseen)
    for _ in range(2):          # a notch inside a longer wobble shows only once the notch is gone
        new, count = _once(new, raster, plain)
        total += count
        if count == 0:
            break
    if total + carried == 0:
        return distance, paved
    # Tidy the seams where the new curves meet the old outline, there and nowhere else.
    near = ndimage.binary_dilation(new ^ paved, iterations=int(round(0.5 / res)))
    soft = ndimage.gaussian_filter(new.astype(np.float32), 1.5) > 0.5
    new = np.where(near, soft, new)
    gained, lost = (new & ~paved).sum() * res * res, (paved & ~new).sum() * res * res
    log(f"  outline: {total} kinks taken out, {carried} unseen stretches redrawn from the seen edge, {gained:.0f} m2 made pavement and {lost:.0f} m2 made verge")
    signed = ((ndimage.distance_transform_edt(new) - ndimage.distance_transform_edt(~new)) * res).astype(np.float32)
    return signed, new
