"""Thin shadows the height data cannot predict: light poles, sign posts, masts.

They are found in the picture alone: a dark streak no wider than about half a metre, at
least a metre and a half long, straight, and pointing along a sun azimuth. Cracks, joints
and tyre marks are thin and dark too, but they do not run along the sun.

They are also removed differently from wide shadows. Both sides of a streak are lit ground of
the same material a step away, so the darkening is measured directly as the difference to
those two sides, averaged along the streak so that texture crossing it is left untouched.
"""
import math

import numpy as np
from scipy import ndimage as ndi

WIDTH = 1.0        # metres: anything narrower than this in some direction counts as thin
MIN_LENGTH = 1.5
MAX_WIDTH = 1.0
LEAST = 0.28       # shade amount a streak must stand out by, per pixel
STRONG = 0.45      # and on average over the streak
BLOB = 3.0         # square metres: largest dark patch that may ride along with a streak
ALIGN_DEG = 10.0
SIDE = (0.5, 0.7, 0.9, 1.1, 1.3)    # metres to each side where the lit ground is read
ALONG = 0.4                         # metres either way the darkening is averaged along a streak


def _px(metres, res, least=1):
    return max(least, int(round(metres / res)))


def find(amount, valid, avoid, sun_azimuths, res):
    """Label image of thin sun-aligned streaks, and for each label its direction.

    ``amount`` is the shade amount per pixel, ``avoid`` marks pixels that are not to be
    considered (wide shadow regions, tall things). Returns (labels int32, list of dicts with
    ``label``, ``slice`` and the unit vector ``along`` as (row, col))."""
    a = ndi.gaussian_filter(amount, max(0.6, 0.08 / res))
    n = max(3, _px(WIDTH, res) | 1)
    a -= ndi.grey_opening(a, size=(n, n))            # what is left is narrower than WIDTH
    cand = (a > LEAST) & (amount > 0.3) & valid & ~avoid
    top = a
    cand = ndi.binary_closing(cand, np.ones((3, 3), bool)) & valid & ~avoid
    lab, count = ndi.label(cand, np.ones((3, 3), bool))
    if not count:
        return lab, []
    rr, cc = np.nonzero(cand)
    ll = lab[rr, cc]
    num = np.bincount(ll, minlength=count + 1).astype(np.float64)
    num[0] = 1.0
    mr = np.bincount(ll, weights=rr, minlength=count + 1) / num
    mc = np.bincount(ll, weights=cc, minlength=count + 1) / num
    dr, dc = rr - mr[ll], cc - mc[ll]
    srr = np.bincount(ll, weights=dr * dr, minlength=count + 1) / num
    scc = np.bincount(ll, weights=dc * dc, minlength=count + 1) / num
    src = np.bincount(ll, weights=dr * dc, minlength=count + 1) / num
    half = 0.5 * (srr + scc)
    root = np.sqrt(np.maximum(0.25 * (srr - scc) ** 2 + src ** 2, 0.0))
    length = np.sqrt(12.0 * (half + root)) * res
    width = np.sqrt(12.0 * np.maximum(half - root, 1.0 / 12.0)) * res
    theta = 0.5 * np.arctan2(2.0 * src, srr - scc)   # major axis, angle from the row axis
    bearing = np.degrees(np.arctan2(np.sin(theta), -np.cos(theta))) % 180.0   # compass, mod 180
    strength = np.bincount(ll, weights=top[rr, cc], minlength=count + 1) / num
    del top
    ok = (length >= MIN_LENGTH) & (width <= MAX_WIDTH) & (length >= 5.0 * width) & (strength >= STRONG)
    aligned = np.zeros(count + 1, bool)
    for az in sun_azimuths:
        d = np.abs((bearing - az + 90.0) % 180.0 - 90.0)
        aligned |= d <= ALIGN_DEG
    ok &= aligned
    ok[0] = False
    keep = np.flatnonzero(ok)
    if not len(keep):
        return np.zeros_like(lab), []
    lab = np.where(ok[lab], lab, 0)
    # The lamp at the top of a pole throws a blob a metre across, too wide to count as thin.
    # Any small dark patch touching a streak is taken to be part of it.
    wide, _ = ndi.label((amount > 0.4) & valid & ~avoid, np.ones((3, 3), bool))
    sr, sc = np.nonzero(lab)
    pairs = np.unique(np.stack([wide[sr, sc], lab[sr, sc]], 1), axis=0)
    pairs = pairs[pairs[:, 0] > 0]
    if len(pairs):
        area = np.bincount(wide.ravel(), minlength=int(wide.max()) + 1) * res * res
        owner = np.zeros(len(area), lab.dtype)
        small = area[pairs[:, 0]] < BLOB
        owner[pairs[small, 0]] = pairs[small, 1]
        lab = np.where((lab == 0) & (owner[wide] > 0), owner[wide], lab)
    del wide
    boxes = ndi.find_objects(lab)
    found = [{"label": int(k), "slice": boxes[k - 1], "along": (math.cos(theta[k]), math.sin(theta[k])),
              "length": float(length[k]), "width": float(width[k]), "strength": float(strength[k]),
              "bearing": float(bearing[k])} for k in keep]
    return lab, found


def relight(lp, key, out, alpha, lab, found, valid, avoid, res):
    """Remove each streak in place in ``out`` (uint8) and add it to the matte ``alpha``."""
    if not found:
        return 0
    rows, cols = lab.shape
    grow = _px(0.25, res)
    zone_all = ndi.binary_dilation(lab > 0, iterations=grow)
    clear = valid & ~avoid & ~zone_all          # ground whose colour can be trusted as lit
    pad = _px(SIDE[-1] + ALONG + 0.3, res)
    done = 0
    for item in found:
        sl = item["slice"]
        r0, r1 = max(0, sl[0].start - pad), min(rows, sl[0].stop + pad)
        c0, c1 = max(0, sl[1].start - pad), min(cols, sl[1].stop + pad)
        win = (slice(r0, r1), slice(c0, c1))
        zone = ndi.binary_dilation(lab[win] == item["label"], iterations=grow) & valid[win]
        zr, zc = np.nonzero(zone)
        if not len(zr):
            continue
        ar, ac = item["along"]
        pr, pc = -ac, ar                          # across the streak
        lp_w, key_w, clear_w = lp[win], key[win], clear[win].astype(np.float32)
        k_here = key_w[zr, zc]
        lit = np.zeros((len(zr), 3), np.float32)
        wsum = np.zeros(len(zr), np.float32)
        for t in SIDE:
            for sgn in (-1.0, 1.0):
                pts = np.stack([zr + sgn * t * pr / res, zc + sgn * t * pc / res])
                w = ndi.map_coordinates(clear_w, pts, order=0, mode="constant")
                k = ndi.map_coordinates(key_w, pts, order=0, mode="nearest")
                w = w * np.exp(-((k - k_here) / 0.08) ** 2)      # same material only
                for ch in range(3):
                    lit[:, ch] += w * ndi.map_coordinates(lp_w[..., ch], pts, order=1, mode="nearest")
                wsum += w
        good = wsum > 0.5
        diff = np.zeros(zone.shape + (3,), np.float32)
        diff[zr[good], zc[good]] = lp_w[zr[good], zc[good]] - lit[good] / wsum[good, None]
        have = np.zeros(zone.shape, np.float32)
        have[zr[good], zc[good]] = 1.0
        # average the darkening along the streak: what is left is the shadow, not the texture
        acc = np.zeros((len(zr), 3), np.float32)
        cnt = np.zeros(len(zr), np.float32)
        for u in np.arange(-ALONG, ALONG + res / 2, res):
            pts = np.stack([zr + u * ar / res, zc + u * ac / res])
            w = ndi.map_coordinates(have, pts, order=1, mode="constant")
            for ch in range(3):
                acc[:, ch] += ndi.map_coordinates(diff[..., ch], pts, order=1, mode="constant")
            cnt += w
        dark = acc / np.maximum(cnt, 1e-3)[:, None]
        dark[(cnt < 2.0) | (dark.mean(1) > 0.0)] = 0.0           # only ever a darkening is undone
        taper = np.clip(ndi.distance_transform_edt(zone)[zr, zc] * res / 0.15, 0.0, 1.0).astype(np.float32)
        dark *= taper[:, None]
        fixed = np.exp(lp_w[zr, zc] - dark) * 256.0 - 1.0
        out[r0 + zr, c0 + zc] = np.clip(np.rint(fixed), 0, 255).astype(np.uint8)
        a = np.clip(-dark.mean(1) / 0.5, 0.0, 1.0)
        alpha[r0 + zr, c0 + zc] = np.maximum(alpha[r0 + zr, c0 + zc], a)
        done += 1
    return done
