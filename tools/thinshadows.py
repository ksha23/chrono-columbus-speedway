"""Find the thin shadows the main shadow step cannot: light poles, signs, rail posts.

shadows.py predicts shadows from the scan's heights, and a pole is too thin to be in them. What
a pole leaves in the photo is a straight dark line a few metres long and 15 to 35 cm wide,
starting at its foot. This looks for exactly that: ground that is clearly darker than the same
kind of ground half a metre to either side, along a straight run of three metres.

Only lines pointing the way a shadow could are taken. The site was flown between late morning
and mid-afternoon in September, so shadows point between north-north-west and east-north-east.
A joint or a tyre mark running across that is left alone.
"""
import numpy as np
from scipy import ndimage

import parallel
import pavement

RUN = 3.0                    # metres of straight line needed
SIDE = 0.5                   # metres to either side that must be lighter
AZIMUTHS = (-30.0, 72.0)     # where shadows point, degrees clockwise from north
STEP = 6.0                   # degrees between the directions tried
# How much darker than its sides a line must be. On pavement a lower bar (0.12 was tried) also
# takes the tyre marks on the skid pad, in pieces, which looks worse than a faint shadow.
DEPTH = {"paved": 0.18, "land": 0.24}
GROW = 2.5                   # metres a line is followed past the part that clears that bar
TONE = 2.0                   # metres over which the ground's own tone is taken


def depth_map(photo, raster, kinds):
    """How much darker each cell is than ground of its own kind around it, on pavement.RES cells.

    kinds is a list of boolean maps on that grid, one per kind of ground. Returns the depth,
    0 for ground at its local tone and 0.4 for ground 40% darker, and 1 outside every kind.
    """
    k = int(round(pavement.RES / raster["res"]))
    h, w = kinds[0].shape
    lum = np.zeros((h, w), np.float32)

    def band(r0):
        r1 = min(r0 + 512, h)
        block = np.asarray(photo[r0 * k:r1 * k, :w * k], dtype=np.float32).reshape(r1 - r0, k, w, k, 3).mean((1, 3))
        lum[r0:r1] = block @ np.array([0.30, 0.59, 0.11], np.float32)

    parallel.each(band, range(0, h, 512))
    sigma = TONE / pavement.RES
    depth = np.ones((h, w), np.float32)

    def blurred(pictures):
        return parallel.each(lambda picture: ndimage.gaussian_filter(picture, sigma), pictures, most=4)

    weights = [kind.astype(np.float32) for kind in kinds]
    first = blurred([picture for weight in weights for picture in (lum * weight, weight)])
    tones = [first[2 * n] / np.maximum(first[2 * n + 1], 1e-3) for n in range(len(kinds))]
    # Again without the dark cells themselves, so a shadow does not pull its own reference down.
    weights = [(kind & (lum > 0.88 * tone)).astype(np.float32) for kind, tone in zip(kinds, tones)]
    second = blurred([picture for weight in weights for picture in (weight, lum * weight)])
    for n, kind in enumerate(kinds):
        seen = second[2 * n]
        tone = np.where(seen > 0.05, second[2 * n + 1] / np.maximum(seen, 1e-3), tones[n])
        depth[kind] = np.clip(1.0 - lum[kind] / np.maximum(tone[kind], 1.0), -1.0, 1.0)
    return depth


def line_score(depth, azimuths):
    """For every cell, how much darker a straight run through it is than the ground beside it.

    The best over the given directions, in degrees clockwise from north. Cells outside every kind of ground carry
    depth 1, which counts as dark, so a strip of ground squeezed between two things that are
    not ground never looks like a line.
    """
    run = int(round(RUN / pavement.RES))
    side = int(round(SIDE / pavement.RES))
    best = np.full(depth.shape, -1.0, np.float32)

    def one(azimuth):
        # Turn the picture so that a line pointing at this azimuth runs along the rows.
        turned = ndimage.rotate(depth, azimuth - 90.0, order=1, reshape=True, mode="constant", cval=1.0)
        along = ndimage.uniform_filter1d(turned, run, axis=1, mode="constant", cval=1.0)
        beside = ndimage.uniform_filter1d(along, 3, axis=0, mode="constant", cval=1.0)
        score = along - np.maximum(np.roll(beside, side, axis=0), np.roll(beside, -side, axis=0))
        back = ndimage.rotate(score, 90.0 - azimuth, order=1, reshape=True, mode="constant", cval=-1.0)
        r0, c0 = (back.shape[0] - depth.shape[0]) // 2, (back.shape[1] - depth.shape[1]) // 2
        return back[r0:r0 + depth.shape[0], c0:c0 + depth.shape[1]].copy()

    # A direction is a turned copy of the whole map and a few more like it: four at a time is what memory allows.
    parallel.each_into(one, azimuths, lambda score: np.maximum(best, score, out=best), most=4)
    return best


def find(photo, raster, paved, land):
    """Masks of thin shadows on pavement and on other ground, on pavement.RES cells.

    paved and land are boolean maps on that grid: where each kind of open ground is.
    """
    depth = depth_map(photo, raster, [paved, land])
    score = line_score(depth, np.arange(AZIMUTHS[0], AZIMUTHS[1] + 1e-6, STEP))
    masks = {}
    for name, kind in (("paved", paved), ("land", land)):
        seeds = kind & (score > DEPTH[name])
        # A run scores highest in its middle and half as much at its ends, and a shadow fades
        # with distance from the pole. So a line is everything scoring half the bar that joins
        # on to a part that clears it, a cell wider all round to take in its soft edge.
        near = ndimage.distance_transform_edt(~seeds) * pavement.RES < GROW
        line = kind & near & (score > 0.45 * DEPTH[name]) & (depth > 0.04)
        labels, count = ndimage.label(line, structure=np.ones((3, 3)))
        keep = np.zeros(count + 1, bool)
        keep[np.unique(labels[seeds])] = True
        keep[0] = False
        masks[name] = ndimage.binary_dilation(keep[labels], structure=np.ones((3, 3))) & kind
    return masks
