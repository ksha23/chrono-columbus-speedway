"""Find the light poles the drone photographed, so they can be painted out and stood back up.

A pole is too thin for the scan to have reconstructed. What it leaves is its shadow: a
straight dark line a few metres long and a hand wide, pointing away from the sun, with the pole
standing at its sunward end. The length of the shadow and the height of the sun give the pole's
height. The site was flown at two times of day, so there are two sun directions, and each
shadow is matched to whichever it points along.

Guard rails and fences throw long thin shadows too, but theirs run along something that stands
up over their whole length, and the scan's heights show it. A pole's shadow lies on open ground.
"""
import json
import os

import numpy as np
from scipy import ndimage

import pavement
import thinshadows

BAR = 0.20            # how much darker than its sides a shadow line must be, in the photo as flown
SWING = 6.0           # degrees either side of the sun's line a shadow may point
LOWEST, TALLEST = 5.0, 20.0     # metres: heights that count as a light pole. The skid pad's masts are 15 m
WIDEST = 0.8          # metres: a pole's shadow is narrower than this
JOIN = 2.0            # metres of gap bridged between two pieces of the same shadow
BESIDE = 1.0          # metres either side of a shadow searched for something standing
STANDING = 0.35       # metres above ground that counts as something standing


def _stick(azimuth, length):
    """A structuring element: a one-cell-wide line of the given length in cells, pointing at azimuth."""
    k = int(np.ceil(length / 2))
    out = np.zeros((2 * k + 1, 2 * k + 1), bool)
    t = np.linspace(-length / 2, length / 2, 4 * k + 1)
    out[np.clip(np.round(k - t * np.cos(np.radians(azimuth))).astype(int), 0, 2 * k),
        np.clip(np.round(k + t * np.sin(np.radians(azimuth))).astype(int), 0, 2 * k)] = True
    return out


def suns(work):
    """[(azimuth, elevation)] of the sun for each flight.

    shadows.py finds each sun from the picture and then looks up the nearest position the real
    sun took that day. The real one is used: heights come from the tangent of the elevation,
    and the fitted elevation of the midday flight is 3 degrees more than the sun ever reached.
    """
    with open(os.path.join(work, "shadows", "suns.json")) as f:
        return [(s["real_azimuth"], s["real_elevation"]) for s in json.load(f)]


def find(photo, raster, paved, land, height, cell, sun_list, rejected=None):
    """Return a list of poles: dicts with x, y (the foot), height, and the shadow's far end.

    photo is the photo as flown. paved and land are boolean maps on pavement.RES cells of open
    ground of each kind. height is the scan's height above ground on cells of size cell.
    rejected, if a list is given, collects the lines turned down and why, for tuning.
    """
    res = pavement.RES
    depth = thinshadows.depth_map(photo, raster, [paved, land])
    ground = paved | land
    standing = np.nan_to_num(height, nan=0.0) > STANDING
    poles = []
    for sun_azimuth, elevation in sun_list:
        points_to = (sun_azimuth + 180.0) % 360.0
        score = thinshadows.line_score(depth, points_to + np.linspace(-SWING, SWING, 9))
        seeds = ground & (score > BAR)
        near = ndimage.distance_transform_edt(~seeds) * res < thinshadows.GROW
        line = ground & near & (score > 0.45 * BAR) & (depth > 0.06)
        # A shadow that crosses a kerb or a paint line comes out in pieces. Join pieces that
        # lie end to end along the sun's line, up to JOIN apart.
        line = ndimage.binary_closing(line, structure=_stick(points_to, JOIN / res)) & ground
        labels, count = ndimage.label(line, structure=np.ones((3, 3)))
        seeded = np.zeros(count + 1, bool)
        seeded[np.unique(labels[seeds])] = True
        along = np.array([np.sin(np.radians(points_to)), np.cos(np.radians(points_to))])    # east, north
        for n, sl in enumerate(ndimage.find_objects(labels), start=1):
            if not seeded[n]:
                continue
            rows, cols = np.nonzero(labels[sl] == n)
            x = raster["x0"] + (sl[1].start + cols + 0.5) * res
            y = raster["y1"] - (sl[0].start + rows + 0.5) * res
            pts = np.stack([x, y], 1)
            centre = pts.mean(0)
            evals, evecs = np.linalg.eigh(np.cov((pts - centre).T) + 1e-9 * np.eye(2))
            axis = evecs[:, 1] if evecs[:, 1] @ along > 0 else -evecs[:, 1]
            off = np.degrees(np.arccos(np.clip(axis @ along, -1, 1)))
            t = (pts - centre) @ axis
            length = float(np.ptp(t)) + res
            width = len(rows) * res * res / length
            tall = length * np.tan(np.radians(elevation))
            why = ("points %.0f degrees off the sun's line" % off if off > SWING + 2.0 else
                   "%.2f m wide" % width if width > WIDEST else
                   "would be %.1f m tall" % tall if not (LOWEST <= tall <= TALLEST) else
                   "not straight" if np.sqrt(evals[0]) > 0.35 else None)
            if why:
                if rejected is not None and length > 3.0:
                    rejected.append((float(centre[0]), float(centre[1]), round(length, 1), why))
                continue
            foot, tip = centre + t.min() * axis, centre + t.max() * axis
            # Is something standing along it? Sample the scan's heights in a band around the line.
            steps = np.linspace(0.1, 0.9, 17)[:, None] * (tip - foot) + foot
            side = np.array([-axis[1], axis[0]])
            band = (steps[:, None, :] + np.linspace(-BESIDE, BESIDE, 9)[None, :, None] * side).reshape(-1, 2)
            hr = np.clip(((raster["y1"] - band[:, 1]) / cell).astype(int), 0, height.shape[0] - 1)
            hc = np.clip(((band[:, 0] - raster["x0"]) / cell).astype(int), 0, height.shape[1] - 1)
            beside = standing[hr, hc].reshape(17, 9).any(1).mean()
            if beside > 0.35:
                if rejected is not None:
                    rejected.append((float(centre[0]), float(centre[1]), round(length, 1), "something stands along %.0f%% of it" % (100 * beside)))
                continue
            # The lamp's own shadow makes the far end wider than the rest.
            far = t > t.max() - 1.2
            head = (far.sum() * res * res / 1.2) / max(width, 1e-6)
            poles.append({"x": float(foot[0]), "y": float(foot[1]), "height": round(float(tall), 1), "tip": [float(tip[0]), float(tip[1])],
                          "sun": [sun_azimuth, elevation], "shadow_length": round(length, 1), "shadow_width": round(float(width), 2),
                          "head": round(float(head), 2), "beside": round(float(beside), 2), "depth": round(float(np.median(depth[sl][labels[sl] == n])), 2)})
    # One pole per foot: where two shadow pieces start at the same place, keep the longer.
    poles.sort(key=lambda p: -p["shadow_length"])
    kept = []
    for p in poles:
        if all(np.hypot(p["x"] - q["x"], p["y"] - q["y"]) > 2.0 for q in kept):
            kept.append(p)
    return sorted(kept, key=lambda p: (p["x"], p["y"]))


REACH = 4.5           # metres from a foot searched for the pole's own picture
# The margins below were set by looking at every candidate on this site, about a hundred of
# them: they keep the 23 that are light poles and drop a gate frame, a fence and three stripes.
LONG_ENOUGH = 0.5     # metres of pale line that confirm a pole
VERGE = (-0.2, 4.5)   # metres from the pavement's edge a pole's foot may be, negative on the pavement
DARK = 0.22           # a pole's shadow is at least this much darker than the ground beside it
THIN = 0.45           # metres to either side of that line where there must be no steel


def pole_picture(photo, raster, x, y):
    """Look for the pole itself at the foot of a shadow: (length, azimuth) of its picture.

    The scan could not stand a pole up, so its picture lies on the ground beside its foot: a
    straight line of galvanised steel a couple of metres long, pale and bluer than whatever
    ground it lies on, pointing away from wherever the drone was. A mowing stripe or a crack
    has a shadow-like line and nothing of the kind at its end. Returns (0, 0) if there is none.
    """
    res = raster["res"]
    k = int(round((REACH + 0.5) / res))
    c, r = int((x - raster["x0"]) / res), int((raster["y1"] - y) / res)
    if r - k < 0 or c - k < 0 or r + k + 1 > photo.shape[0] or c + k + 1 > photo.shape[1]:
        return 0.0, 0.0
    rgb = np.asarray(photo[r - k:r + k + 1, c - k:c + k + 1], dtype=np.float32) / 255
    val, low = rgb.max(-1), rgb.min(-1)
    sat = (val - low) / np.maximum(val, 1e-3)
    blue = rgb[..., 2] - rgb[..., 0]
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    far = (xx * xx + yy * yy) * res * res > 4.0
    # Steel in daylight: white with no colour, or, lying on concrete that is itself pale, a
    # touch bluer than the concrete.
    steel = ((sat < 0.14) & (val > 0.72)) | ((blue - np.median(blue[far]) > 0.045) & (sat < 0.20) & (val > 0.62))
    wide = ndimage.maximum_filter(steel, 3)
    steps = np.arange(0.2, REACH, res)
    best = (0.0, 0.0)

    def at(mask, along, across, dx, dy):
        px = k + (along * dx - across * dy) / res
        py = k - (along * dy + across * dx) / res
        return mask[np.clip(py.round().astype(int), 0, 2 * k), np.clip(px.round().astype(int), 0, 2 * k)]

    for azimuth in np.arange(0.0, 360.0, 3.0):
        dx, dy = np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))
        # On the line there is steel, and a step to either side there is not: a pole is thin.
        # A kerb, a footpath or a painted line fails on one side or the other.
        hit = at(wide, steps, 0.0, dx, dy) & ~at(steel, steps, THIN, dx, dy) & ~at(steel, steps, -THIN, dx, dy)
        closed = ndimage.binary_closing(hit, structure=np.ones(int(0.25 / res) + 1))
        labels, count = ndimage.label(closed)
        for n in range(1, count + 1):
            where = np.nonzero(labels == n)[0]
            if steps[where[0]] < 1.0 and len(where) * res > best[0]:
                best = (float(len(where) * res), float(azimuth))
    return best


def confirm(candidates, photo, raster, paved):
    """Keep the shadow lines that are light poles.

    Three things have to hold. The foot stands beside pavement, within VERGE of its edge: every
    pole on this site does, since lighting the pavement is what it is for. The pole's own
    picture lies at the foot. And the shadow is as narrow as a pole's. Mowing stripes, crop
    rows and cracks fail the first two, kerbs and rails the third.
    """
    away = ndimage.distance_transform_edt(~paved) * pavement.RES
    into = ndimage.distance_transform_edt(paved) * pavement.RES
    kept = []
    for p in candidates:
        r = int((raster["y1"] - p["y"]) / pavement.RES)
        c = int((p["x"] - raster["x0"]) / pavement.RES)
        off = float(away[r, c] - into[r, c])
        if not (VERGE[0] <= off <= VERGE[1]) or p["shadow_width"] > (0.45 if p["height"] < 13.0 else 0.6) or p["depth"] < DARK:
            continue
        length, azimuth = pole_picture(photo, raster, p["x"], p["y"])
        if length >= LONG_ENOUGH:
            kept.append({**p, "from_pavement": round(off, 1), "picture_length": round(length, 2), "picture_azimuth": azimuth})
    return kept


def open_ground(valid, trees, buildings, shape, cell):
    """Where a pole's shadow can be looked for: all the scan covers but trees and buildings."""
    cells = valid & ~ndimage.binary_dilation(trees, iterations=2) & ~ndimage.binary_dilation(buildings, iterations=4)
    return pavement.to_fine(cells, shape, cell)


def locate(photo, raster, paved, height, trees, buildings, cell, work):
    """Find and confirm the light poles. photo is the photo as flown, paved on pavement.RES cells."""
    where = open_ground(np.isfinite(height), trees, buildings, paved.shape, cell)
    return confirm(find(photo, raster, paved & where, ~paved & where, height, cell, suns(work)), photo, raster, paved)


def footprint(poles, raster, shape):
    """Full-size mask of what each pole left in the photo: its shadow, its picture, its footing."""
    res = raster["res"]
    mask = np.zeros(shape, bool)

    def capsule(a, b, radius, end=0.0):
        """Mark everything within radius of the segment a-b, and within end of b."""
        reach = max(radius, end) + 0.2
        x0, x1 = min(a[0], b[0]) - reach, max(a[0], b[0]) + reach
        y0, y1 = min(a[1], b[1]) - reach, max(a[1], b[1]) + reach
        c0, c1 = max(int((x0 - raster["x0"]) / res), 0), min(int((x1 - raster["x0"]) / res) + 1, shape[1])
        r0, r1 = max(int((raster["y1"] - y1) / res), 0), min(int((raster["y1"] - y0) / res) + 1, shape[0])
        if r1 <= r0 or c1 <= c0:
            return
        yy, xx = np.mgrid[r0:r1, c0:c1]
        px = raster["x0"] + (xx + 0.5) * res - a[0]
        py = raster["y1"] - (yy + 0.5) * res - a[1]
        d = np.array([b[0] - a[0], b[1] - a[1]])
        t = np.clip((px * d[0] + py * d[1]) / max(d @ d, 1e-9), 0, 1)
        near = np.hypot(px - t * d[0], py - t * d[1]) < radius
        if end:
            near |= np.hypot(px - d[0], py - d[1]) < end
        mask[r0:r1, c0:c1] |= near

    for p in poles:
        foot = (p["x"], p["y"])
        heavy = p["shadow_width"] > 0.3
        capsule(foot, p["tip"], 0.4 if heavy else 0.3, end=1.0 if heavy else 0.8)      # the shadow and the lamp's
        along = np.radians(p["picture_azimuth"])
        reach = p["picture_length"] + 0.8
        capsule(foot, (p["x"] + reach * np.sin(along), p["y"] + reach * np.cos(along)), 0.3, end=0.6)   # the pole lying flat
        capsule(foot, foot, 0.65)                                                        # the footing
    return mask
