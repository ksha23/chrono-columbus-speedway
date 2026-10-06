"""Sun geometry for the shadow remover: where the sun was, and where a height field puts shade.

Everything here works on a north-up raster (row 0 is the north edge) and takes distances in
metres, so the pixel size only ever enters as ``res``.
"""
import math

import numpy as np
from scipy import ndimage as ndi

LOW = -1.0e4  # stand-in height for cells with no data: never casts a shadow


def solar_position(lat_deg, lon_deg, year, month, day, utc_hours):
    """Sun (azimuth clockwise from north, elevation) in degrees. NOAA low-accuracy formulas,
    good to a few hundredths of a degree, with no refraction."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jd = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045 - 0.5 + utc_hours / 24.0
    t = (jd - 2451545.0) / 36525.0
    l0 = math.radians((280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360.0)
    mean = math.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    centre = math.radians((1.914602 - t * (0.004817 + 0.000014 * t)) * math.sin(mean)
                          + (0.019993 - 0.000101 * t) * math.sin(2 * mean) + 0.000289 * math.sin(3 * mean))
    omega = math.radians(125.04 - 1934.136 * t)
    lam = l0 + centre - math.radians(0.00569 + 0.00478 * math.sin(omega))
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = math.radians(eps0 + 0.00256 * math.cos(omega))
    decl = math.asin(math.sin(eps) * math.sin(lam))
    v = math.tan(eps / 2) ** 2
    eot = 4 * math.degrees(v * math.sin(2 * l0) - 2 * ecc * math.sin(mean)
                           + 4 * ecc * v * math.sin(mean) * math.cos(2 * l0)
                           - 0.5 * v * v * math.sin(4 * l0) - 1.25 * ecc * ecc * math.sin(2 * mean))
    solar_minutes = (utc_hours * 60.0 + eot + 4.0 * lon_deg) % 1440.0
    hour_angle = math.radians(solar_minutes / 4.0 - 180.0)
    lat = math.radians(lat_deg)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    az = math.atan2(-math.sin(hour_angle) * math.cos(decl),
                    math.cos(lat) * math.sin(decl) - math.sin(lat) * math.cos(decl) * math.cos(hour_angle))
    return math.degrees(az) % 360.0, 90.0 - math.degrees(zen)


def closest_solar_time(lat_deg, lon_deg, year, month, day, azimuth, elevation, step_minutes=1.0):
    """The UTC hour on that date when the real sun was nearest the given direction.

    Returns (utc_hours, azimuth, elevation, separation_deg)."""
    target = _unit(azimuth, elevation)
    best = None
    for k in range(int(1440 / step_minutes)):
        h = k * step_minutes / 60.0
        az, el = solar_position(lat_deg, lon_deg, year, month, day, h)
        if el <= 0:
            continue
        sep = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(target, _unit(az, el)))))))
        if best is None or sep < best[3]:
            best = (h, az, el, sep)
    return best


def _unit(az, el):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


def fill_nearest(a, ok=None):
    """Copy of ``a`` with cells outside ``ok`` (default: non-finite ones) taken from the nearest
    cell inside it."""
    if ok is None:
        ok = np.isfinite(a)
    if ok.all() or not ok.any():
        return a
    idx = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
    return a[idx[0], idx[1]]


def cast_shadow(height, res, azimuth, elevation, want_distance=False):
    """Shade predicted by a height field for a sun at (azimuth, elevation) degrees.

    Returns ``depth`` (float32, metres): how far the cell sits below the lowest sun ray that
    clears everything between it and the sun. Positive means shaded, negative means that much
    clearance. With ``want_distance`` also returns the horizontal distance to the thing that
    sets that ray (metres). Cells with no height are filled from their nearest neighbour first.

    The raster is sheared so sun rays run along one axis, which turns the search along every
    ray into a single running maximum.
    """
    az = math.radians(azimuth)
    dc, dr = math.sin(az), -math.cos(az)  # one metre towards the sun, in (col, row)
    h = fill_nearest(np.asarray(height, np.float32))
    h = h - np.float32(np.nanmedian(h[::16, ::16]))
    transposed = abs(dc) > abs(dr)
    if transposed:
        h = h.T
        dc, dr = dr, dc
    rows, cols = h.shape
    hp = np.pad(h, ((0, 0), (1, 2)), mode="edge")  # one spare column each side to interpolate into
    sign = 1.0 if dr >= 0 else -1.0
    slope = dc / abs(dr)  # columns moved per row stepped towards the sun
    step = res * math.hypot(1.0, slope)  # metres along the ray per row
    shift = slope * sign * np.arange(rows, dtype=np.float64)
    extra = int(math.ceil(abs(slope) * (rows - 1))) + 4
    off = (extra if slope * sign > 0 else 0) + 2
    width = cols + extra + 6
    wide = np.full((rows, width), LOW, np.float32)
    inside = np.zeros((rows, width), bool)
    # wide[r, c'] = h[r, c' - off + shift[r]], so every sun ray keeps to one column of wide
    for r in range(rows):
        s = shift[r] - off
        k = int(math.floor(s))
        f = np.float32(s - k)
        lo, hi = -k - 1, cols - k  # c' whose source column lies in [-1, cols)
        seg = hp[r, lo + k + 1:hi + k + 2]
        wide[r, lo:hi] = seg[:-1] * (1 - f) + seg[1:] * f
        inside[r, lo:hi] = True
    row_m = (np.arange(rows) * step).astype(np.float32)
    wide -= (sign * math.tan(math.radians(elevation)) * row_m)[:, None]  # rays are now level
    ahead = np.empty_like(wide)  # the highest thing between each cell and the sun
    if sign > 0:
        ahead[:-1] = np.maximum.accumulate(wide[:0:-1], axis=0)[::-1]
        ahead[-1] = LOW * 10
    else:
        ahead[1:] = np.maximum.accumulate(wide[:-1], axis=0)
        ahead[0] = LOW * 10
    dist_w = None
    if want_distance:
        # the cell that sets the horizon is the nearest one, sunwards, that tops everything beyond it
        rr = np.arange(rows, dtype=np.float32)[:, None]
        if sign > 0:
            idx = np.where(wide > ahead, rr, np.float32(rows))
            dist_w = np.empty_like(wide)
            dist_w[:-1] = np.minimum.accumulate(idx[:0:-1], axis=0)[::-1]
            dist_w[-1] = rows
            dist_w -= rr
        else:
            idx = np.where(wide > ahead, rr, np.float32(-1))
            dist_w = np.empty_like(wide)
            dist_w[1:] = np.maximum.accumulate(idx[:-1], axis=0)
            dist_w[0] = -1
            dist_w = rr - dist_w
        del idx
        dist_w *= np.float32(step)
    ahead -= wide
    del wide
    ahead[~inside] = -1.0
    del inside

    def back(w):
        out = np.empty((rows, cols), np.float32)
        for r in range(rows):
            s = off - shift[r]  # out[r, c] = w[r, c + s]
            k = int(math.floor(s))
            f = np.float32(s - k)
            seg = w[r, k:k + cols + 1]
            out[r] = seg[:-1] * (1 - f) + seg[1:] * f
        return out

    depth = back(ahead)
    del ahead
    np.clip(depth, -50.0, 50.0, out=depth)
    if transposed:
        depth = np.ascontiguousarray(depth.T)
    depth[~np.isfinite(height)] = np.nan
    if not want_distance:
        return depth
    dist = back(dist_w)
    if transposed:
        dist = np.ascontiguousarray(dist.T)
    return depth, dist


def block_reduce(a, factor, how="mean"):
    """Shrink a raster by an integer factor, ignoring NaN. Edge rows that do not fill a block
    are dropped."""
    if factor <= 1:
        return a
    rows, cols = a.shape[0] // factor, a.shape[1] // factor
    v = a[:rows * factor, :cols * factor].reshape(rows, factor, cols, factor, *a.shape[2:])
    with np.errstate(all="ignore"):
        if how == "mean":
            return np.nanmean(v, axis=(1, 3))
        if how == "max":
            return np.nanmax(v, axis=(1, 3))
        if how == "min":
            return np.nanmin(v, axis=(1, 3))
    raise ValueError(how)


def above_ground(height, res, widest=45.0, coarse=1.0):
    """Height of each cell above the bare ground around it (metres), 0 on open ground.

    The ground is found by wearing the surface down with windows that grow up to ``widest``
    metres, accepting a drop only when it is steeper than terrain could be, so buildings and
    crowns come off but banks and slopes stay."""
    f = max(1, int(round(coarse / res)))
    cell = res * f
    with np.errstate(all="ignore"):
        low = block_reduce(height, f, "min")
    ok = np.isfinite(low)
    if not ok.any():
        return np.zeros(height.shape, np.float32)
    idx = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
    ground = low[tuple(idx)].astype(np.float32)
    w, prev = 3.0, 0.0
    while w <= widest:
        n = max(3, int(round(w / cell)) | 1)
        opened = ndi.grey_dilation(ndi.grey_erosion(ground, size=(n, n)), size=(n, n))
        # a bump this much wider than the last window, and this much taller, is not terrain
        allow = min(0.3 + 0.2 * (w - prev), 2.5)
        ground = np.where(ground - opened > allow, opened, ground)
        prev = w
        w *= 1.6
    ground = ndi.gaussian_filter(ground, 1.5 / cell)
    zoom = (height.shape[0] / ground.shape[0], height.shape[1] / ground.shape[1])
    full = ndi.zoom(ground, zoom, order=1, mode="nearest", grid_mode=True)
    full = full[:height.shape[0], :height.shape[1]]
    out = np.nan_to_num(height - full, nan=0.0).astype(np.float32)
    return np.maximum(out, 0.0, out=out)


def crude_shadow_index(rgb):
    """A rough, material-blind shadow test: dark and bluish for how dark it is.

    Skylight is bluer than sunlight, so shade raises the blue share of a pixel while it
    lowers its brightness. Good enough to find the sun with, not good enough to relight by."""
    c = rgb.astype(np.float32)
    total = np.maximum(c.sum(-1), 1.0)
    return c[..., 2] / total - 0.12 * (total / 765.0) > 0.26


def _tile_sums(mask, block, span):
    """Counts of True in overlapping tiles: blocks of ``block`` cells, summed ``span`` at a time."""
    rows, cols = mask.shape[0] // block, mask.shape[1] // block
    s = mask[:rows * block, :cols * block].reshape(rows, block, cols, block).sum((1, 3), dtype=np.float64)
    c = np.cumsum(np.cumsum(np.pad(s, ((1, 0), (1, 0))), 0), 1)
    span_r, span_c = min(span, rows), min(span, cols)
    return c[span_r:, span_c:] - c[:-span_r, span_c:] - c[span_r:, :-span_c] + c[:-span_r, :-span_c]


def find_suns(dark, height, res, valid, cell=0.5, tile=140.0, max_suns=3, log=None):
    """Every sun direction the shadows in the picture agree with.

    A survey flown over several hours is stitched from pictures with different suns, so one
    direction is not always enough. ``dark`` marks pixels that look shaded. The height field
    is cast for a grid of directions and each ~``tile`` metre window votes for the one whose
    predicted shade overlaps its dark pixels best.

    Returns a list of dicts, strongest first, with ``azimuth``, ``elevation``, ``weight`` (share
    of the evidence) and ``score`` (overlap of prediction and picture, 0..1), plus a function
    ``support(shape)`` giving, per sun, a 0..1 map of where that sun explains the shadows.
    """
    f = max(1, int(round(cell / res)))
    c = res * f
    with np.errstate(all="ignore"):
        hc = block_reduce(np.where(valid, height, np.nan), f, "max")
        dk = block_reduce(np.where(valid, dark, np.nan).astype(np.float32), f, "mean") > 0.5
    ok = np.isfinite(hc)
    elev = above_ground(hc, c) > 1.2
    ground = ok & ~ndi.binary_dilation(elev, iterations=1)
    reach = int(round(25.0 / c))
    near = ndi.binary_dilation(elev, iterations=reach) & ground  # shade can only fall near something tall
    dk &= near
    hf = fill_nearest(hc)
    block = max(1, int(round(tile / 4 / c)))

    def score(az, el):
        s = (cast_shadow(hf, c, az, el) > 0.3) & near
        return _tile_sums(s & dk, block, 4), _tile_sums(s | dk, block, 4)

    azs = np.arange(0.0, 360.0, 6.0)
    els = np.arange(20.0, 72.0, 6.0)
    n_dark = _tile_sums(dk, block, 4)
    inter = np.zeros((len(azs), len(els)) + n_dark.shape)
    union = np.ones_like(inter)
    for i, az in enumerate(azs):
        for j, el in enumerate(els):
            inter[i, j], union[i, j] = score(az, el)
    iou = inter / np.maximum(union, 1.0)
    enough = n_dark >= 300.0 * (0.5 / c) ** 2 * 0.25
    suns, claimed = [], np.zeros(n_dark.shape, bool)
    total = float(n_dark[enough].sum()) or 1.0
    for _ in range(max_suns):
        free = enough & ~claimed
        if not free.any():
            break
        # the direction that best explains the windows nobody has claimed yet
        g_iou = inter[:, :, free].sum(-1) / np.maximum(union[:, :, free].sum(-1), 1.0)
        i, j = np.unravel_index(int(g_iou.argmax()), g_iou.shape)
        if g_iou[i, j] < 0.12:
            break
        best = iou.reshape(-1, *n_dark.shape).max(0)
        mine = free & (iou[i, j] >= 0.7 * best) & (iou[i, j] > 0.1)
        if not mine.any():
            break
        az, el = float(azs[i]), float(els[j])
        for step in (2.0, 0.5):  # refine on the windows this sun owns
            cand = [(az + a * step, el + e * step) for a in range(-3, 4) for e in range(-3, 4)]
            vals = []
            for a, e in cand:
                e = min(max(e, 5.0), 85.0)
                n, u = score(a % 360.0, e)
                vals.append(n[mine].sum() / max(u[mine].sum(), 1.0))
            k = int(np.argmax(vals))
            az, el = cand[k][0] % 360.0, min(max(cand[k][1], 5.0), 85.0)
        n, u = score(az, el)
        local = n / np.maximum(u, 1.0)
        suns.append({"azimuth": az, "elevation": el, "weight": float(n_dark[mine].sum()) / total,
                     "score": float(n[mine].sum() / max(u[mine].sum(), 1.0)), "_local": local, "_mine": mine})
        claimed |= mine
        if log:
            log(f"sun {len(suns)}: azimuth {az:.1f}, elevation {el:.1f}, overlap {suns[-1]['score']:.2f}, "
                f"{100 * suns[-1]['weight']:.0f}% of the shadow evidence")
    suns.sort(key=lambda s: -s["weight"])
    tile_step = block * c

    def support(shape):
        """Per sun, how well it explains the shadows around each pixel (0..1, smooth)."""
        out = []
        if not suns:
            return out
        stack = np.stack([np.where(enough, s["_local"], np.nan) for s in suns])
        best = np.nanmax(stack, axis=0)
        for k in range(len(suns)):
            rel = np.where(np.isfinite(best) & (best > 0.05), stack[k] / np.maximum(best, 1e-6), np.nan)
            # windows with no evidence take the answer of the nearest window that has some
            if np.isfinite(rel).any():
                rel = fill_nearest(rel)
            else:
                rel = np.ones_like(best)
            rel = ndi.gaussian_filter(rel.astype(np.float32), 0.7)
            rows = ((np.arange(shape[0]) + 0.5) * res / tile_step - 2.0).clip(0, rel.shape[0] - 1)
            cols = ((np.arange(shape[1]) + 0.5) * res / tile_step - 2.0).clip(0, rel.shape[1] - 1)
            out.append((rel, rows, cols))
        return out

    return suns, support
