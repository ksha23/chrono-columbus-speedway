"""Relighting for the shadow remover: how much to lift each shaded pixel, and how the lift
fades across the soft edge.

The lift is measured, not assumed. Along every shadow edge the same material lies on both
sides, so each edge pixel is paired with the lit colour found by walking out of the shadow
and the shade colour found by walking in. Those pairs, pooled over a few distances, say what
the lit colour of each shaded material is; the drop to undo is the local shade level minus
that lit colour.
"""
import math

import numpy as np
from scipy import ndimage as ndi

import shadow_geom as geom
import shadow_photo as photo

REACH = (0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.0, 3.6, 4.2, 5.0)   # metres searched each side of an edge
MARGIN = 0.8               # metres the fade is tracked beyond where it seems to end
SCALES = (1.5, 5.0, 15.0, 45.0)   # metres over which edge pairs are pooled, nearest first
SHADE_SCALE = 2.0          # metres over which the shade level inside a shadow is averaged
STAT_CELL = 1.0
KS_LO, KS_STEP, KS_BINS = -0.95, 0.1, 16   # shade-side key: log(B/G) of the shaded colour
BIG = 1.0e6


def _px(metres, res, least=1):
    return max(least, int(round(metres / res)))


def _ramp(d, lo, hi):
    t = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ks_coord(v):
    return np.clip((v - KS_LO) / KS_STEP, 0.0, KS_BINS - 1.0).astype(np.float32)


def _shade_key(lp, region, res):
    """log(B/G) of the shaded colour, steadied by a median over about a metre that only looks
    at shaded pixels.

    Inside one shadow everything is under the same light, so plain colour tells materials
    apart there, and it tells shaded water (blue) from shaded grass where the shade-proof
    key cannot. NaN where no shaded pixel is within reach."""
    f = _px(0.2, res)
    v = lp[..., 2] - lp[..., 1]
    with np.errstate(all="ignore"):
        vc = geom.block_reduce(np.where(region, v, np.nan), f)
    rows, cols = vc.shape
    # cells outside the region get +BIG and -BIG in a checkerboard, so they cancel in the median
    board = np.where((np.add.outer(np.arange(rows), np.arange(cols)) & 1) == 0, BIG, -BIG).astype(np.float32)
    filled = np.where(np.isfinite(vc), vc, board).astype(np.float32)
    med = ndi.median_filter(filled, max(3, _px(1.0, res * f) | 1), mode="nearest")
    med[np.abs(med) > BIG / 2] = np.nan
    med = np.where(np.isfinite(med), med, vc)
    out = np.full(region.shape, np.nan, np.float32)
    up = np.repeat(np.repeat(med, f, 0), f, 1)
    out[:up.shape[0], :up.shape[1]] = up[:region.shape[0], :region.shape[1]]
    return out


def _accumulate(values, kc, rr, cc, f, shape, bins=KS_BINS):
    """Per (cell, key bin): sums of the given per-pixel values (n, k) and a count."""
    gr, gc = -(-shape[0] // f), -(-shape[1] // f)
    k = values.shape[1]
    acc = np.zeros((gr * gc * bins, k + 1), np.float32)
    if len(rr):
        idx = ((rr // f) * gc + cc // f) * bins + np.rint(kc).astype(np.int64)
        for c in range(k):
            acc[:, c] = np.bincount(idx, weights=values[:, c], minlength=acc.shape[0])
        acc[:, k] = np.bincount(idx, minlength=acc.shape[0])
    return acc.reshape(gr, gc, bins, k + 1)


def _blur(acc, sig):
    s = ndi.gaussian_filter1d(ndi.gaussian_filter1d(acc, sig, axis=0, mode="constant"), sig, axis=1, mode="constant")
    return ndi.convolve1d(s, np.array([0.25, 0.5, 0.25], np.float32), axis=2, mode="nearest")


def _sample(grid, f, rr, cc, kc):
    """Read a (rows, cols, bins, n) grid of cell size f at pixel positions, smoothly."""
    gr, gc, nb = grid.shape[:3]
    y = np.clip((rr + 0.5) / f - 0.5, 0, gr - 1).astype(np.float32)
    x = np.clip((cc + 0.5) / f - 0.5, 0, gc - 1).astype(np.float32)
    y0 = np.clip(y.astype(np.int64), 0, max(gr - 2, 0))
    x0 = np.clip(x.astype(np.int64), 0, max(gc - 2, 0))
    y1, x1 = np.minimum(y0 + 1, gr - 1), np.minimum(x0 + 1, gc - 1)
    fy, fx = (y - y0)[:, None], (x - x0)[:, None]
    k0 = np.minimum(kc.astype(np.int64), nb - 2)
    fk = (kc - k0)[:, None]
    out = np.zeros((len(rr), grid.shape[-1]), np.float32)
    for yy, wy in ((y0, 1 - fy), (y1, fy)):
        for xx, wx in ((x0, 1 - fx), (x1, fx)):
            out += wy * wx * ((1 - fk) * grid[yy, xx, k0] + fk * grid[yy, xx, k0 + 1])
    return out


def _pooled(acc, f, rr, cc, kc, sig_m, res, need):
    """Sample a pooled statistic and say how far to trust it (0..1 by sample count)."""
    sig = sig_m / (res * f)
    s = _sample(_blur(acc, sig), f, rr, cc, kc)
    n = s[:, -1:]
    return s[:, :-1] / np.maximum(n, 1e-9), np.clip(n * (2 * math.pi * sig * sig) / need, 0.0, 1.0)


def _mean_at(lp, r, c, step):
    """Mean log colour over a small plus-shaped patch at each point."""
    rows, cols = lp.shape[:2]
    out = np.zeros((len(r), 3), np.float32)
    for dr, dc in ((0, 0), (-step, 0), (step, 0), (0, -step), (0, step)):
        out += lp[np.clip(r + dr, 0, rows - 1), np.clip(c + dc, 0, cols - 1)]
    return out / 5.0


def _edge_pairs(lp, inv_key, a_s, region, workable, open_ground, d_in, d_out, res):
    """Walk in and out from every edge pixel to where the shade stops changing.

    A shadow edge can be a hand wide or, where the mosaic blended pictures taken minutes
    apart, a few metres. So the two colours to compare are not taken at a fixed distance:
    from each edge pixel the walk goes inwards until the shade amount has levelled off, and
    outwards until it has gone, and those two spots are the pair.

    Returns edge rows/cols, the pair's lit log colour, the shade spot, and how far each walk
    went (metres)."""
    shape = region.shape
    sd = ndi.gaussian_filter(d_out - d_in, max(1.0, 0.3 / res))
    b_r, b_c = np.nonzero(region & workable & (d_in <= 1.5 * res))
    r0, r1 = np.clip(b_r - 1, 0, shape[0] - 1), np.clip(b_r + 1, 0, shape[0] - 1)
    c0, c1 = np.clip(b_c - 1, 0, shape[1] - 1), np.clip(b_c + 1, 0, shape[1] - 1)
    n_r, n_c = sd[r1, b_c] - sd[r0, b_c], sd[b_r, c1] - sd[b_r, c0]     # points out of the shadow
    del sd
    norm = np.maximum(np.hypot(n_r, n_c), 1e-6)
    n_r, n_c = n_r / norm, n_c / norm
    steps = np.array(REACH, np.float32)
    near = steps <= 3.0

    def walk(sign):
        r = np.rint(b_r[:, None] + sign * n_r[:, None] * steps[None] / res).astype(np.int64)
        c = np.rint(b_c[:, None] + sign * n_c[:, None] * steps[None] / res).astype(np.int64)
        ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        return np.clip(r, 0, shape[0] - 1), np.clip(c, 0, shape[1] - 1), ok

    # inwards: stay in the region and keep getting deeper (stop at the far side of a thin one)
    r, c, ok = walk(-1.0)
    depth = d_in[r, c]
    ok &= region[r, c] & workable[r, c]
    ok[:, 1:] &= depth[:, 1:] > depth[:, :-1] - 0.05
    ok = np.logical_and.accumulate(ok, axis=1)
    a = np.where(ok, a_s[r, c], -np.inf)
    top = a[:, near].max(1)
    i_in = np.argmax(a >= (0.9 * top)[:, None], axis=1)
    has_in = np.isfinite(top) & (top > 0.5)
    pick = np.arange(len(b_r))
    p_r, p_c, w_in = r[pick, i_in], c[pick, i_in], steps[i_in]

    # outwards: stay clear of any shadow region, to where the amount bottoms out
    r, c, ok = walk(1.0)
    ok &= ~region[r, c] & workable[r, c]
    ok = np.logical_and.accumulate(ok, axis=1)
    a = np.where(ok, a_s[r, c], np.inf)
    low = a[:, near].min(1)
    i_out = np.argmax(a <= (low + 0.08)[:, None], axis=1)
    q_r, q_c, w_out = r[pick, i_out], c[pick, i_out], steps[i_out]
    good = has_in & np.isfinite(low) & (low < 0.2) & open_ground[q_r, q_c]
    good &= np.abs(inv_key[q_r, q_c] - inv_key[p_r, p_c]) < 0.15        # same material both sides
    # The lit colour is the average over the two metres beyond that spot, not the spot itself:
    # the spot was chosen for being the least shaded, so alone it would read too bright.
    use = ok & (steps[None] >= w_out[:, None]) & (steps[None] <= w_out[:, None] + 2.0) & open_ground[r, c]
    use &= a <= (low + 0.15)[:, None]
    use[pick, i_out] = True
    n = use.sum(1).astype(np.float32)
    lit = np.stack([(lp[r, c, ch] * use).sum(1) / n for ch in range(3)], 1).astype(np.float32)
    return b_r[good], b_c[good], lit[good], p_r[good], p_c[good], w_in[good], w_out[good]


def _spread(values, rr, cc, shape, res, default):
    """Smooth map of a quantity known only at scattered edge pixels (mean within ~1.5 m)."""
    f = _px(STAT_CELL, res)
    gr, gc = -(-shape[0] // f), -(-shape[1] // f)
    cell = (rr // f) * gc + cc // f
    num = np.bincount(cell, weights=values, minlength=gr * gc).reshape(gr, gc).astype(np.float32)
    den = np.bincount(cell, minlength=gr * gc).reshape(gr, gc).astype(np.float32)
    sig = 1.5 / (res * f)
    num, den = ndi.gaussian_filter(num, sig), ndi.gaussian_filter(den, sig)
    grid = np.where(den > 1e-3, num / np.maximum(den, 1e-6), default).astype(np.float32)
    zoom = (shape[0] / gr, shape[1] / gc)
    full = ndi.zoom(grid, zoom, order=1, mode="nearest", grid_mode=True)
    out = np.full(shape, default, np.float32)
    r, c = min(full.shape[0], shape[0]), min(full.shape[1], shape[1])
    out[:r, :c] = full[:r, :c]
    return out


def measure(lp, inv_key, amount, region, workable, open_ground, guard, ref, table, res):
    """For every pixel in or beside a region: the drop to undo and a first, noisy reading of
    how shaded it is. Returns a dict of per-pixel arrays plus the distance maps."""
    shape = region.shape
    d_in = (ndi.distance_transform_edt(region) * res).astype(np.float32)
    d_out, near = ndi.distance_transform_edt(~region, return_indices=True)
    d_out = (d_out * res).astype(np.float32)
    ks = _shade_key(lp, region, res)
    ks = np.where(np.isfinite(ks), ks, lp[..., 2] - lp[..., 1])
    a_s = ndi.uniform_filter(amount, max(3, _px(0.5, res) | 1))
    f = _px(STAT_CELL, res)

    b_r, b_c, lit_p, p_r, p_c, w_in, w_out = _edge_pairs(
        lp, inv_key, a_s, region, workable, open_ground, d_in, d_out, res)
    del a_s
    acc_pair = _accumulate(lit_p, _ks_coord(ks[p_r, p_c]), b_r, b_c, f, shape)
    fade_in = _spread(w_in, b_r, b_c, shape, res, 0.6)
    fade_out = _spread(w_out, b_r, b_c, shape, res, 0.6)

    # the shade level inside each shadow, by shade key, leaving out the fade at its edges
    c_r, c_c = np.nonzero(region & workable & ~guard & (d_in > fade_in))
    c_k = _ks_coord(ks[c_r, c_c])
    c_v = lp[c_r, c_c]
    acc_sh = _accumulate(c_v, c_k, c_r, c_c, f, shape)
    # second pass without the outliers: a car's own shadow or a dark object inside a wide
    # shadow must not drag the local shade level down
    level, _ = _pooled(acc_sh, f, c_r, c_c, c_k, SHADE_SCALE, res, 60.0)
    fair = np.abs(c_v.mean(1) - level.mean(1)) < 0.25
    acc_sh = _accumulate(c_v[fair], c_k[fair], c_r[fair], c_c[fair], f, shape)
    del c_v, c_k, level, fair
    # plainly lit ground beyond the fade, by its own material: the yardstick for pixels outside
    l_r, l_c = np.nonzero(open_ground & ~region & (d_out > fade_out + 0.2) & (d_out < fade_out + 4.0) & (amount < 0.35))
    acc_own = _accumulate(lp[l_r, l_c], photo.key_coord(inv_key[l_r, l_c]), l_r, l_c, f, shape, photo.KEY_BINS)

    zone = workable & (region | (d_out < fade_out + MARGIN))
    rr, cc = np.nonzero(zone)
    inside = region[rr, cc]
    kc = _ks_coord(ks[near[0][rr, cc], near[1][rr, cc]])   # a region pixel is its own nearest
    del near
    here = lp[rr, cc]
    wide = ref.lookup(0, shape[0], inv_key, zone)          # last resort: the detection reference

    # lit colour of the shaded material, from the edge pairs, nearest pairs winning
    lit = wide.copy()
    for sig_m in sorted(SCALES, reverse=True):
        v, w = _pooled(acc_pair, f, rr, cc, kc, sig_m, res, 12.0)
        lit = w * v + (1 - w) * lit
    # shade level around the pixel. Using this, not the shade at the edge, takes out the slow
    # darkening towards a wall and the wide soft part of a blended edge along with the shadow.
    shade = lit + photo.kappa_at(table, inv_key[rr, cc]).astype(np.float32)
    for sig_m in (SCALES[1], SHADE_SCALE):
        v, w = _pooled(acc_sh, f, rr, cc, kc, sig_m, res, 60.0)
        shade = w * v + (1 - w) * shade
    drop = np.minimum(shade - lit, -0.05)

    # outside a region the pixel is judged against lit ground of its own material
    out_i = np.flatnonzero(~inside)
    judge = lit.copy()
    own = wide[out_i]
    kc_own = photo.key_coord(inv_key[rr[out_i], cc[out_i]])
    for sig_m in sorted(SCALES[:2], reverse=True):
        v, w = _pooled(acc_own, f, rr[out_i], cc[out_i], kc_own, sig_m, res, 60.0)
        own = w * v + (1 - w) * own
    judge[out_i] = own
    raw = ((here - judge) * drop).sum(1) / (drop * drop).sum(1)

    # A lift past about 3x multiplies noise and whatever is not plain matt surface (water,
    # the dark under a car). The more the lift, the less of the pixel's own detail is kept
    # and the more it is simply given the lit colour found at the edge.
    keep = np.clip(1.1 / np.maximum(-drop.min(1), 1e-3), 0.0, 1.0) ** 2
    # (a pixel far off the local shade level is a thing standing in the shadow, a car body
    # say: it is not ground to be filled, so it just takes the plain lift)
    off = (here - shade).mean(1)
    keep = keep + (1.0 - keep) * _ramp(np.abs(off), 0.4, 0.8)
    drop = np.where(inside[:, None], drop + (1.0 - keep[:, None]) * (here - shade), drop)
    # Round a vehicle the height data cannot say where bodywork ends and ground begins. There,
    # whatever is far from the colour of the shaded ground, over half a metre, is bodywork
    # (or the black under it) and is left alone.
    ground = np.ones(len(rr), np.float32)
    g = guard[rr, cc] & inside
    if g.any():
        n = max(3, _px(0.5, res) | 1)
        dense = np.zeros(shape, np.float32)
        dense[rr, cc] = np.where(inside, np.abs(off), 0.0)
        cover = np.zeros(shape, np.float32)
        cover[rr, cc] = inside
        off_s = ndi.uniform_filter(dense, n) / np.maximum(ndi.uniform_filter(cover, n), 1e-3)
        ground[g] = 1.0 - _ramp(off_s[rr[g], cc[g]], 0.3, 0.6)
        del dense, cover, off_s
    return {"rows": rr, "cols": cc, "drop": drop, "raw": np.clip(raw, -0.5, 1.5).astype(np.float32),
            "lit": wide, "ground": ground, "_lit": lit, "_shade": shade, "d_in": d_in, "d_out": d_out, "fade_in": fade_in, "fade_out": fade_out,
            "pairs": len(b_r)}


def matte(m, region, shadelike, res):
    """The soft shadow matte: 1 in full shade, 0 in sun, and across each edge the fade the
    picture actually shows.

    Two kinds of edge occur. A gentle one, metres wide where the mosaic blended pictures
    taken minutes apart, is followed as it is: the raw reading smoothed over a hand's width.
    An abrupt one would get a bright or dark rim from that smoothing, so there the matte is
    a clean step at the place the reading crosses one half, and only the pixels right on the
    step use their own reading.

    Also returns ``hairline``: 1 on an abrupt edge, falling to 0 a hand's width from it."""
    rr, cc, raw = m["rows"], m["cols"], m["raw"]
    d_in, d_out, fade_in, fade_out = m["d_in"], m["d_out"], m["fade_in"], m["fade_out"]
    shape = region.shape
    raw_full = np.zeros(shape, np.float32)
    raw_full[rr, cc] = raw
    full = np.clip(raw_full, 0.0, 1.0)
    known = np.zeros(shape, np.float32)
    known[rr, cc] = 1.0
    sig = max(1.0, 0.3 / res)
    cover = ndi.gaussian_filter(known, sig)
    smooth = np.where(cover > 0.05, ndi.gaussian_filter(full, sig) / np.maximum(cover, 1e-3), region).astype(np.float32)
    # abruptness: slope of the smoothed reading, scaled so a perfect step gives 1
    gy, gx = np.gradient(smooth)
    slope = np.hypot(gy, gx) * np.float32(sig * math.sqrt(2 * math.pi))
    del gy, gx
    slope[cover < 0.9] = 0.0     # beside a wall there is only one side to read: no edge there
    sharp = _ramp(ndi.maximum_filter(slope, max(3, _px(0.7, res) | 1)), 0.55, 0.9)
    del slope, cover
    # (at an abrupt edge the place of the step is read at a finer scale, so that the small
    # separate patches of a tree's shadow, half a metre across, are not smoothed away)
    fine = ndi.gaussian_filter(full, max(0.7, 0.12 / res))
    inside = np.where(sharp > 0.5, fine > 0.5, smooth > 0.5)
    dist = (ndi.distance_transform_edt(inside) - ndi.distance_transform_edt(~inside)).astype(np.float32)
    dist -= np.where(inside, 0.5, -0.5).astype(np.float32)   # measure from the edge, not the far pixel's centre
    dist = np.abs(dist) * res
    on_step = dist < max(0.15, 1.5 * res)
    alpha = sharp * np.where(on_step, full, inside) + (1.0 - sharp) * np.clip((smooth - 0.06) / 0.88, 0.0, 1.0)
    hairline = sharp * np.clip(1.0 - dist / 0.3, 0.0, 1.0)
    del sharp, smooth, dist, inside, on_step, full
    # Deep inside, a building's shade is even and the answer is 1. A tree's is not: light
    # comes through the crown in patches. So inside, the matte is the typical reading over a
    # metre and a half (a median, so that paint and cracks narrower than that do not move it).
    f = _px(0.2, res)
    with np.errstate(all="ignore"):
        level = geom.block_reduce(np.where(known > 0, np.clip(raw_full, 0.0, 1.2), np.nan), f)
    level = ndi.median_filter(geom.fill_nearest(level), max(3, _px(1.4, res * f) | 1))
    zoom = (shape[0] / level.shape[0], shape[1] / level.shape[1])
    deep = np.ones(shape, np.float32)
    up = ndi.zoom(level, zoom, order=1, mode="nearest", grid_mode=True)
    r, c = min(up.shape[0], shape[0]), min(up.shape[1], shape[1])
    deep[:r, :c] = np.clip((up[:r, :c] - 0.06) / 0.84, 0.0, 1.0)
    del level, up, raw_full
    w = _ramp(d_in, fade_in + 0.3, fade_in + MARGIN) * region
    alpha = alpha * (1.0 - w) + deep * w
    del deep, w
    # The loose patches at the fringe of a tree's shadow are half a metre across and only
    # partly dark: smaller than the smoothing above, so it halves them. Inside a region every
    # pixel is lifted at least as far as its own reading, taken over a hand's width. This is
    # the one place texture is given up for the sake of the shadow: a crack inside a shadow
    # is lightened a little along with it.
    np.maximum(alpha, fine * region, out=alpha)
    del fine
    alpha *= 1.0 - _ramp(d_out, fade_out + 0.3, fade_out + MARGIN)
    # Outside a region only a fade may be lifted: at most half shade once clear of the
    # outline's own coarseness, and only where the darkening has the colour of shade.
    # Anything else out there is dark ground, which stays as it is.
    tail = ndi.uniform_filter(shadelike.astype(np.float32), max(3, _px(0.5, res) | 1))
    cap = np.maximum(1.0 - _ramp(d_out, 0.15, 0.45), 0.5 * _ramp(tail, 0.3, 0.7))
    np.minimum(alpha, cap, out=alpha)
    del tail, cap
    alpha[rr, cc] *= m["ground"]
    # Sun comes through a crown in flecks a pixel or two across. A pixel that is already
    # near its lit colour is lifted only as far as that colour (plus a little, for texture).
    reach = np.clip(raw + 0.25 / np.maximum(-m["drop"].mean(1), 0.05), 0.0, 1.0)
    alpha[rr, cc] = np.minimum(alpha[rr, cc], reach)
    alpha[known == 0] = 0.0
    return alpha.astype(np.float32), hairline.astype(np.float32)


def apply(lp, m, alpha, hairline):
    """Corrected 8-bit colours for the measured pixels."""
    rr, cc = m["rows"], m["cols"]
    out = lp[rr, cc] - alpha[rr, cc][:, None] * m["drop"]
    # JPEG stores colour at half resolution, so along an abrupt edge the dark side carries
    # the colour of the bright side. Lifted, that is an orange hairline on grass. Within a
    # hand's width of such an edge the pixel keeps its brightness and takes the lit hue of
    # its own material. Grey surfaces have no hue to bleed and are left alone.
    lit = m["_lit"]          # the lit colour found across this very edge
    vivid = np.clip((lit.max(1) - lit.min(1) - 0.15) / 0.2, 0.0, 1.0)
    e = (hairline[rr, cc] * vivid)[:, None]
    out += e * ((lit - lit.mean(1, keepdims=True)) - (out - out.mean(1, keepdims=True)))
    return np.clip(np.rint(np.exp(out) * 256.0 - 1.0), 0, 255).astype(np.uint8)
