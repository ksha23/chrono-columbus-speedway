"""Roads as they are built: strips of even width about their centre lines.

The pavement's outline is first traced from the photo, and along most of a road that is good
to a few centimetres. But where a tree's shadow or its crown lay over the kerb the trace takes
a bite out of the road or bulges into the grass, and a bite a metre deep is plain to see once
the road's surface is cut along it and a white line is painted beside it.

A road is not that shape. It is a strip of one width, laid about a centre line, and the centre
line is known: it is the yellow dashed line, found dash by dash and already fitted to a smooth
curve. So along every centre line the distance to the traced edge is measured on each side,
metre by metre, and the road's half width on that side is taken to be what those measurements
mostly say (their running median). Where the trace disagrees with it, the trace is wrong, and
the outline is redrawn at the fitted width.

Junctions are left as traced. There the edge really does curve away, and there is no centre
line to measure from.
"""
import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import CubicSpline

import markings_regular
import pavement
import railmodel

REACH = 9.0        # metres from the centre line within which a road's edge is looked for
WINDOW = 30.0      # metres either side over which the road's width is judged
FLARE = 10.0       # metres from a junction within which the traced outline is kept
CLEAR = 2.5        # metres beyond the fitted edge that are made grass
SOLID = 20.0       # metres: a yellow stroke this long is a centre line in its own right


def centre_lines(strokes):
    """The roads' centre lines, as (N, 2) arrays of points a metre apart."""
    lines = []
    for chain in markings_regular.runs(strokes):
        if strokes[chain[0]]["colour"] != "yellow" or len(chain) < 4:
            continue
        pts = np.array([np.mean(strokes[n]["points"], axis=0) for n in chain])
        gaps = np.hypot(*np.diff(pts, axis=0).T)
        if np.median(gaps) < markings_regular.ROAD:
            continue
        t = np.concatenate([[0.0], np.cumsum(gaps)])
        spline = CubicSpline(t, pts)
        # Half a spacing beyond the first and the last dash: the line was painted that far.
        s = np.arange(-np.median(gaps) / 2, t[-1] + np.median(gaps) / 2, 1.0)
        lines.append(spline(np.clip(s, 0, t[-1])) + np.where(s[:, None] < 0, (s[:, None]) * spline(0.0, 1) / np.hypot(*spline(0.0, 1)),
                                                               np.where(s[:, None] > t[-1], (s[:, None] - t[-1]) * spline(t[-1], 1) / np.hypot(*spline(t[-1], 1)), 0.0)))
    for s in strokes:
        if s["colour"] == "yellow" and railmodel.length_of(s["points"]) >= SOLID:
            path = railmodel.resample(s["points"], 1.0)
            # One of a pair of solid lines is enough.
            if not any(np.median(np.hypot(*(l[:, None] - path[None]).transpose(2, 0, 1)).min(0)) < 0.6 for l in lines):
                lines.append(path)
    return lines


BEHIND = 8.0       # metres beyond a guard rail within which pavement nobody saw is taken back


def behind_rails(paved, raster, rails, unseen):
    """The pavement mask without what was only guessed at on the far side of a guard rail.

    Where a hedge hides a road's edge the pavement is carried on under it, and how far is a
    guess. A guard rail settles it: the rail stands on the road's edge, so what lies beyond
    it and was never seen is not road. unseen marks, on the pavement's own cells, ground the
    photo does not show.
    """
    res = pavement.RES
    h, w = paved.shape

    def cell(points):
        return (np.clip(((raster["y1"] - points[..., 1]) / res).astype(int), 0, h - 1),
                np.clip(((points[..., 0] - raster["x0"]) / res).astype(int), 0, w - 1))

    taken = np.zeros_like(paved)
    offsets = np.arange(0.4, BEHIND, res / 2)
    for rail in rails:
        line = railmodel.resample(rail["points"], res / 2)
        if len(line) < 8:
            continue
        tangent = np.gradient(line, axis=0)
        tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        share = {side: np.mean([paved[cell(line + side * d * normal)].mean() for d in (1.0, 1.5, 2.0)]) for side in (1.0, -1.0)}
        if abs(share[1.0] - share[-1.0]) < 0.3:
            continue                            # pavement on both sides, or on neither: no telling
        away = -1.0 if share[1.0] > share[-1.0] else 1.0
        r, c = cell(line[:, None, :] + away * offsets[None, :, None] * normal[:, None, :])
        taken[r, c] = True
    taken = ndimage.binary_closing(taken, iterations=1) & unseen & paved
    return paved & ~taken


def _running_median(values, half, least=None):
    """Median of the finite values within half stations either side, NaN where too few."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        near = values[max(i - half, 0):i + half + 1]
        ok = np.isfinite(near)
        if ok.sum() >= (max(8, 0.4 * len(near)) if least is None else least):
            out[i] = np.median(near[ok])
    return out


def straighten(distance, raster, strokes, log=print, doubtful=None):
    """Redraw the pavement's outline along the roads. Returns (signed distance, pavement mask).

    doubtful marks, on the pavement's cells, ground the photo showed badly or not at all:
    under a crown, or in a shadow that was relit. An edge traced there is often not the
    road's: a gravel shoulder in the shade of a tree line passes for concrete. So a side's
    width is judged from the stations where its edge was in plain view, and where there are
    none within reach the road is taken to be as wide on that side as on the other.
    """
    res = pavement.RES
    paved = distance > 0
    h, w = paved.shape

    def cell(points):
        return (np.clip(((raster["y1"] - points[..., 1]) / res).astype(int), 0, h - 1),
                np.clip(((points[..., 0] - raster["x0"]) / res).astype(int), 0, w - 1))

    new = paved.copy()
    steps = np.arange(0.0, REACH, res / 2)
    roads = redrawn = mirrored = 0
    moved = []
    for line in centre_lines(strokes):
        if len(line) < 20:
            continue
        tangent = np.gradient(line, axis=0)
        tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        roads += 1
        # How far from the centre line the traced pavement reaches on each side, at every
        # station, and what that says the road's width is where the edge was in plain view.
        reach, plain, every = {}, {}, {}
        for side in (1.0, -1.0):
            ray = line[:, None, :] + side * steps[None, :, None] * normal[:, None, :]
            on = paved[cell(ray)]
            first_off = np.where(on.all(1), np.nan, steps[np.argmin(on, axis=1)])
            first_off[~on[:, 0]] = np.nan                      # the centre line itself is off the pavement here
            reach[side] = first_off
            every[side] = _running_median(first_off, int(WINDOW))
            if doubtful is None:
                plain[side] = every[side]
            else:
                at_edge = line + side * np.nan_to_num(first_off)[:, None] * normal
                seen = np.isfinite(first_off) & ~doubtful[cell(at_edge)]
                # Only where a fair share of the stations saw the edge. A few that did may all
                # be at one bite, and would say the bite is the road's width.
                plain[side] = _running_median(np.where(seen, first_off, np.nan), int(WINDOW))
        for side in (1.0, -1.0):
            first_off = reach[side]
            usual = np.where(np.isfinite(plain[side]), plain[side], np.where(np.isfinite(plain[-side]), plain[-side], every[side]))
            usual[~np.isfinite(every[side])] = np.nan           # no road on this side here at all
            mirrored += int((~np.isfinite(plain[side]) & np.isfinite(plain[-side]) & np.isfinite(every[side])).sum())
            ok = np.isfinite(usual)
            if ok.sum() < 15:
                continue
            fitted = usual.copy()
            fitted[ok] = signal.savgol_filter(usual[ok], min(21, (ok.sum() - 1) | 1), 2) if ok.sum() > 5 else usual[ok]
            # Near a junction the edge flares out, and that is real. Keep the trace there.
            open_side = ~np.isfinite(first_off)
            near_junction = ndimage.binary_dilation(open_side, iterations=int(FLARE))
            model = ok & ~near_junction
            if not model.any():
                continue
            moved.append(np.abs(first_off[model] - fitted[model]))
            # Stations a twentieth of a metre apart, so the strip is drawn without gaps.
            fine = np.arange(0, len(line) - 1 + 1e-9, res / 2)
            base = np.minimum(fine.astype(int), len(line) - 2)
            frac = (fine - base)[:, None]
            centre = line[base] * (1 - frac) + line[base + 1] * frac
            across = normal[base] * (1 - frac) + normal[base + 1] * frac
            width = np.interp(fine, np.nonzero(ok)[0], fitted[ok])
            traced = np.interp(fine, np.arange(len(line)), np.nan_to_num(first_off, nan=0.0))
            use = np.interp(fine, np.arange(len(line)), model.astype(float)) > 0.999
            grid = centre[use][:, None, :] + side * steps[None, :, None] * across[use][:, None, :]
            inside = steps[None, :] <= width[use][:, None]
            beyond = steps[None, :] <= (width[use] + CLEAR)[:, None]
            r, c = cell(grid)
            if doubtful is not None:
                # Pavement guessed at further out than that, as far as the trace went, goes too.
                beyond |= (steps[None, :] <= (traced[use] + 0.5)[:, None]) & doubtful[r, c]
            new[r[beyond & ~inside], c[beyond & ~inside]] = False
            new[r[inside], c[inside]] = True
            redrawn += int(model.sum())
    if roads == 0:
        return distance, paved
    # Close the pinholes the drawing leaves on the outside of bends, and keep the one network.
    new = ndimage.binary_closing(new, iterations=1) | new
    soft = ndimage.gaussian_filter(new.astype(np.float32), pavement.SMOOTH / res / 2)
    labels, count = ndimage.label(soft > 0.5)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
    new = labels == 1 + int(np.argmax(sizes))
    off = np.concatenate(moved) if moved else np.zeros(1)
    log(f"  roads: {roads} centre lines, outline redrawn along {redrawn} m of edge at the road's own width, {mirrored} m of it taken from the other side."
        f" The trace was within 10 cm of it for {100 * (off < 0.1).mean():.0f}% of that, and over 50 cm out for {100 * (off > 0.5).mean():.1f}%")
    signed = ((ndimage.distance_transform_edt(new) - ndimage.distance_transform_edt(~new)) * res).astype(np.float32)
    return signed, new
