#!/usr/bin/env python3
"""Take the baked-in cast shadows out of a drone orthophoto.

    estimate_sun(rgb, height, res, valid)             -> (azimuth, elevation) in degrees
    remove_shadows(rgb, height, res, valid, sun=None) -> (rgb_out uint8, shadow float32 0..1)

    python -I shadows.py WORK_DIR OUT_DIR [--raster photo|ortho] [--sun AZ EL]... [--full]

How it works, in the order it runs:

1. The lit colour of every material at every place is read off the picture itself
   (shadow_photo.LitReference), and each pixel is scored by how far it has dropped from that
   colour along the direction shade moves colours ("amount") and how much of the drop shade
   cannot explain ("tint").
2. The sun is found by casting the height field (shadow_geom). A mosaic flown over hours has
   more than one sun, so several are looked for. The height field is blobby and misses thin
   things, so the cast is supporting evidence, not the shadow map.
3. Solid dark patches that are shade-coloured, or that the cast predicts, become regions.
4. Each region is relit (shadow_relight) by the drop measured between the same material on
   the two sides of its own edge, with a matte that follows the fade the picture shows.
5. Thin streaks along a sun azimuth, which are light-pole shadows, are found in the picture
   alone and removed against the lit ground either side of them (shadow_thin).

Everything is worked out at about 0.1 m per pixel. A finer picture is shrunk for the analysis
and the resulting gain is applied to it at full resolution in bands, so the texture stays as
sharp as it came and memory stays bounded.
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shadow_geom as geom  # noqa: E402
import shadow_photo as photo  # noqa: E402
import shadow_relight as relight  # noqa: E402
import shadow_thin as thin  # noqa: E402

# Distances, all in metres.
WORK_RES = 0.1        # the analysis is done at about this many metres per pixel
TALL = 1.0            # higher than this above the ground is a tree or a roof: left alone
OBJECT = 0.5          # anything this high that is not vegetation is an object (a car): left alone
REGION_CELL = 0.2     # grid the shadow regions are outlined on
SOLID = 0.5           # a region must hold a disc of this radius somewhere


def _log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


# Places worth looking at, in raster metres (x east, y north, window size).
SPOTS = {
    "building": (-143.0, -20.0, 60.0),
    "building2": (-250.0, -140.0, 60.0),
    "pond": (110.0, -5.0, 60.0),
    "pole": (148.0, 44.0, 24.0),
    "treeline": (-40.0, 90.0, 60.0),
    "road_trees": (-300.0, -213.0, 60.0),
}
SITE = {"lat": 43.2845, "lon": -89.1021, "date": (2024, 9, 15), "utc_offset": -5.0}


def _px(metres, res, least=1):
    return max(least, int(round(metres / res)))


def _disc(radius_px):
    r = int(radius_px)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= radius_px * radius_px + 0.5


def _prepare(rgb, height, valid):
    """Hole-free copies to filter on: every missing cell takes its nearest valid neighbour."""
    if valid.all():
        return rgb, np.asarray(height, np.float32)
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    rgb_f = rgb[idx[0], idx[1]]
    h = np.asarray(height, np.float32)[idx[0], idx[1]]
    if not np.isfinite(h).all():
        h = geom.fill_nearest(h)
    return rgb_f, h


def _material_key(rgb_f, valid, res):
    """Material key per pixel, steadied over about a metre.

    Read pixel by pixel the key is too noisy in deep shade to say what a pixel is made of, and
    a wrong material means a wrong lit colour. A median over a metre keeps material edges
    where they are and removes the noise."""
    f = _px(REGION_CELL, res)
    small = np.stack([geom.block_reduce(rgb_f[..., c].astype(np.float32), f) for c in range(3)], -1)
    key = photo.material_key(photo.log_rgb(small))
    key = ndi.median_filter(key, max(3, _px(1.0, res * f) | 1))
    zoom = (rgb_f.shape[0] / key.shape[0], rgb_f.shape[1] / key.shape[1])
    full = ndi.zoom(key, zoom, order=1, mode="nearest", grid_mode=True)
    out = np.empty(rgb_f.shape[:2], np.float32)
    r, c = min(full.shape[0], out.shape[0]), min(full.shape[1], out.shape[1])
    out[:r, :c] = full[:r, :c]
    out[r:], out[:, c:] = out[r - 1:r], out[:, c - 1:c]
    return out


def _suns(rgb, height, res, valid, log=None):
    f = int(round(WORK_RES / res))
    if f >= 2:
        rgb, height, valid = _shrink(rgb, height, np.asarray(valid, bool), f)
        res = res * f
    dark = geom.crude_shadow_index(rgb) & valid
    return geom.find_suns(dark, height, res, valid, log=log)


def estimate_sun(rgb, height, res, valid):
    """The sun direction that explains most of the shadows: (azimuth clockwise from north,
    elevation), degrees, in the raster's own geometry.

    A mosaic flown over hours has more than one; ``estimate_suns`` returns them all."""
    suns = estimate_suns(rgb, height, res, valid)
    if not suns:
        raise ValueError("no shadow could be matched to the height data")
    return suns[0]["azimuth"], suns[0]["elevation"]


def estimate_suns(rgb, height, res, valid):
    """All sun directions found, strongest first: dicts with azimuth, elevation, weight, score."""
    suns, _ = _suns(rgb, height, res, valid)
    return [{k: v for k, v in s.items() if not k.startswith("_")} for s in suns]


def _cast_all(height_f, res, suns, support, shape):
    """Where the height field says shade falls, for whichever sun explains each area."""
    out = np.zeros(shape, bool)
    maps = support(shape) if support is not None else [None] * len(suns)
    for s, m in zip(suns, maps):
        shade = geom.cast_shadow(height_f, res, s["azimuth"], s["elevation"]) > 0.0
        if m is not None:
            rel, rows, cols = m
            ok = ndi.map_coordinates(rel, np.meshgrid(rows, cols, indexing="ij"), order=1, mode="nearest") > 0.5
            shade &= ok
        out |= shade
    return out


def _amount_maps(lp, key, ref, table, valid):
    """Per pixel: shade amount and leftover tint (see shadow_photo.shade_amount)."""
    rows = lp.shape[0]
    amount = np.zeros(lp.shape[:2], np.float32)
    tint = np.zeros(lp.shape[:2], np.float32)
    step = 256
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        m = valid[r0:r1]
        if not m.any():
            continue
        lit = ref.lookup(r0, r1, key[r0:r1], m)
        a, t = photo.shade_amount(lp[r0:r1][m] - lit, photo.kappa_at(table, key[r0:r1][m]))
        amount[r0:r1][m] = a
        tint[r0:r1][m] = t
    return amount, tint


def _regions(amount, tint, key, table, valid, tall, cast, res):
    """Outline the shadow regions: solid patches carrying a full, shade-coloured drop."""
    f = _px(REGION_CELL, res)
    cell = res * f
    with np.errstate(all="ignore"):
        a = geom.block_reduce(np.where(valid, amount, np.nan), f)
        t = geom.block_reduce(np.where(valid, tint, np.nan), f)
        k = geom.block_reduce(np.where(valid, key, np.nan), f)
    ok = np.isfinite(a)
    a, t, k = np.nan_to_num(a), np.nan_to_num(t), np.nan_to_num(k)
    blocked = geom.block_reduce(tall.astype(np.float32), f) > 0.5
    seen = geom.block_reduce(cast.astype(np.float32), f) > 0.3
    n = max(3, _px(0.6, cell) | 1)
    a_s = ndi.median_filter(a, n)
    t_s = ndi.uniform_filter(t, n)
    spread = photo.kappa_at(table, k)
    spread = spread[..., 2] - spread[..., 0]
    coloured = t_s > -0.6 * np.maximum(a_s, 0.3) * spread  # at least 40% of the blue shift shade gives
    # Where the height field says shade should fall, darkness alone is enough: the colour
    # test is for telling shade from dark ground, and the geometry has already done that.
    # The prediction is a metre or two off, so it is given that much slack.
    near_cast = ndi.binary_dilation(seen, _disc(1.5 / cell))
    strong = np.where(seen, 0.5, 0.65)
    core = ok & ~blocked & (coloured | seen) & (a_s > strong)
    solid = ndi.binary_opening(core, _disc(SOLID / cell))
    solid |= ndi.binary_opening(core & seen, _disc(0.6 * SOLID / cell))
    shadelike = (t_s > -0.8 * np.maximum(a_s, 0.3) * spread) | near_cast
    grow = ok & ~blocked & (a_s > 0.35) & shadelike
    # A tree's shadow breaks up into separate patches at its fringe. Dark patches within a
    # metre of one another are taken together, and a group counts if any of it is solid.
    lab, _ = ndi.label(ndi.binary_dilation(grow, _disc(1.0 / cell)), np.ones((3, 3), bool))
    keep = np.unique(lab[solid & grow])
    region = grow & np.isin(lab, keep[keep > 0])
    # pinholes in the coverage and specks of texture inside a shadow belong to it
    holes = ndi.binary_fill_holes(region) & ~region
    lab, n = ndi.label(holes)
    if n:
        size = ndi.sum(holes, lab, np.arange(1, n + 1)) * cell * cell
        region |= np.isin(lab, 1 + np.flatnonzero(size < 4.0))
    # On pavement the amount can be trusted without the colour test (the surface is even, and
    # shade under leaves is not as blue as shade under open sky). Dark pavement within three
    # metres of a shadow is the loose fringe of that shadow.
    fringe = ok & (np.abs(k) < 0.05) & (a_s > 0.3) & ndi.binary_dilation(region, _disc(3.0 / cell))
    region = (region | fringe) & ~blocked

    def full(mask):
        up = np.repeat(np.repeat(mask, f, 0), f, 1)
        out = np.zeros(amount.shape, bool)
        out[:up.shape[0], :up.shape[1]] = up[:amount.shape[0], :amount.shape[1]]
        return out

    return full(region), full(shadelike | region)


def _shrink(rgb, height, valid, f):
    """Picture, top surface and coverage at 1/f the resolution, made a band at a time."""
    rows, cols = rgb.shape[0] // f, rgb.shape[1] // f
    rgb_c = np.zeros((rows, cols, 3), np.uint8)
    h_c = np.full((rows, cols), np.nan, np.float32)
    valid_c = np.zeros((rows, cols), bool)
    step = 256
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        sl = slice(r0 * f, r1 * f)
        v = valid[sl, :cols * f].reshape(r1 - r0, f, cols, f)
        n = v.sum((1, 3))
        valid_c[r0:r1] = n > 0
        nn = np.maximum(n, 1)
        px = rgb[sl, :cols * f].reshape(r1 - r0, f, cols, f, 3)
        for ch in range(3):
            rgb_c[r0:r1, :, ch] = (np.where(v, px[..., ch], 0).sum((1, 3)) + nn // 2) // nn
        h = height[sl, :cols * f].reshape(r1 - r0, f, cols, f)
        h = np.where(v & np.isfinite(h), h, -np.inf).max((1, 3))      # the top surface: keep the highest
        h_c[r0:r1] = np.where(np.isfinite(h), h, np.nan)
    return rgb_c, h_c, valid_c


def _spread_up(a, f, rows, cols):
    """A coarse band repeated up to fine size and eased over one coarse cell."""
    up = np.repeat(np.repeat(a, f, 0), f, 1)
    if up.shape[0] < rows or up.shape[1] < cols:
        pad = [(0, max(0, rows - up.shape[0])), (0, max(0, cols - up.shape[1]))] + [(0, 0)] * (a.ndim - 2)
        up = np.pad(up, pad, mode="edge")
    size = (f, f) + (1,) * (a.ndim - 2)
    return ndi.uniform_filter(up[:rows, :cols], size=size, mode="nearest")


def remove_shadows(rgb, height, res, valid, sun=None, log=None, debug=None):
    """Relight the cast shadows in ``rgb``.

    ``sun`` may be one (azimuth, elevation) pair, a list of them, or None to find them.
    Returns the corrected picture and the shadow matte (1 = full shade removed there)."""
    valid = np.asarray(valid, bool)
    f = int(round(WORK_RES / res))
    if f < 2:
        return _remove(rgb, height, res, valid, sun, log, debug)
    # finer than needed: analyse a shrunken copy, apply the gain at full size
    rgb_c, h_c, valid_c = _shrink(rgb, height, valid, f)
    out_c, alpha_c = _remove(rgb_c, h_c, res * f, valid_c, sun, log, debug)
    rows, cols = valid.shape
    out = np.empty_like(rgb)
    alpha = np.empty((rows, cols), np.float32)
    step, lap = 256, 2
    for c0 in range(0, out_c.shape[0], step):
        c1 = min(out_c.shape[0], c0 + step)
        a0, a1 = max(0, c0 - lap), min(out_c.shape[0], c1 + lap)
        r0 = c0 * f
        r1 = rows if c1 == out_c.shape[0] else c1 * f
        gain = (photo.log_rgb(out_c[a0:a1]) - photo.log_rgb(rgb_c[a0:a1])).astype(np.float32)
        gain[~valid_c[a0:a1]] = 0.0
        n = (a1 - a0) * f + (rows - out_c.shape[0] * f if a1 == out_c.shape[0] else 0)
        gain = _spread_up(gain, f, n, cols)[(c0 - a0) * f:(c0 - a0) * f + (r1 - r0)]
        band = rgb[r0:r1]
        touched = np.abs(gain).max(-1) > 2e-3
        fixed = np.exp(photo.log_rgb(band) + gain) * 256.0 - 1.0
        out[r0:r1] = np.where(touched[..., None] & valid[r0:r1, :, None],
                              np.clip(np.rint(fixed), 0, 255).astype(np.uint8), band)
        alpha[r0:r1] = _spread_up(alpha_c[a0:a1], f, n, cols)[(c0 - a0) * f:(c0 - a0) * f + (r1 - r0)]
    alpha[~valid] = 0.0
    if log:
        log(f"gain applied at {res:g} m")
    return out, alpha


def _remove(rgb, height, res, valid, sun=None, log=None, debug=None):
    """remove_shadows at the working resolution."""
    log = log or (lambda m: None)
    rgb_f, height_f = _prepare(rgb, height, valid)
    lp = photo.log_rgb(rgb_f)
    key = _material_key(rgb_f, valid, res)
    above = geom.above_ground(height_f, res)
    tall = above > TALL
    log("prepared")

    if sun is None:
        suns, support = _suns(rgb, height, res, valid, log=log)
    else:
        pairs = [sun] if np.isscalar(sun[0]) else list(sun)
        suns, support = [{"azimuth": float(a), "elevation": float(e)} for a, e in pairs], None
    cast = _cast_all(height_f, res, suns, support, valid.shape) if suns else np.zeros(valid.shape, bool)
    log(f"cast {len(suns)} sun(s): {100.0 * (cast & valid).sum() / max(valid.sum(), 1):.1f}% predicted shaded")

    open_ground = valid & ~ndi.binary_dilation(tall, _disc(_px(0.5, res)))
    use = open_ground & ~ndi.binary_dilation(cast, _disc(_px(1.0, res)))
    ref = photo.LitReference(lp, key, use, res)
    table = photo.default_kappa_table()
    amount, tint = _amount_maps(lp, key, ref, table, valid)
    log("scored every pixel against its lit colour")

    # Crowns are blobby in the height data, so shade is followed a metre in under their edge.
    # Anything else that stands up (vehicles, roofs, tanks) has a true outline and is left
    # out whole, with a small margin for its sides.
    green = key < -0.08
    thing = ~green & (above > OBJECT)
    blocked = (green & ndi.binary_erosion(tall, _disc(_px(1.0, res)))) | ndi.binary_dilation(thing & tall, _disc(_px(0.3, res)))
    # the lower parts of a vehicle and the blobby skirt the height data draws round it: ground
    # and bodywork are told apart there by colour, during relighting
    guard = ndi.binary_dilation(thing, _disc(_px(0.5, res))) & ~blocked
    guard &= ~ndi.binary_dilation(green & tall, _disc(_px(2.0, res)))     # not the bare wood of a tree
    region, shadelike = _regions(amount, tint, key, table, valid, blocked, cast, res)
    log(f"regions: {100.0 * region.sum() / max(valid.sum(), 1):.2f}% of the picture")

    workable = valid & ~blocked
    m = relight.measure(lp, key, amount, region, workable, open_ground, guard, ref, table, res)
    log(f"measured the drop across the edges ({m['pairs']} edge pairs)")
    alpha, hairline = relight.matte(m, region, shadelike, res)
    out = rgb.copy()
    out[m["rows"], m["cols"]] = relight.apply(lp, m, alpha, hairline)
    log("relit")
    del m, hairline

    # light poles and the like: thin, straight, along a sun azimuth, invisible to the height data
    avoid = blocked | (alpha > 0.02)
    lab, found = thin.find(amount, valid, avoid, [s["azimuth"] for s in suns], res)
    n_thin = thin.relight(lp, key, out, alpha, lab, found, valid, avoid, res)
    log(f"thin shadows: {n_thin} streaks")
    out[~valid] = rgb[~valid]
    alpha[~valid] = 0.0
    if debug is not None:
        debug.update(amount=amount, tint=tint, region=region, cast=cast, tall=tall, suns=suns, key=key,
                     blocked=blocked, thin=lab > 0, above=above)
    return out, alpha


def _load(work, raster):
    """(rgb, height, meta) for one of the two rasters the pipeline keeps in WORK_DIR."""
    if raster == "photo":
        with open(os.path.join(work, "raster.json")) as f:
            meta = json.load(f)
        return np.load(os.path.join(work, "photo.npy")), np.load(os.path.join(work, "surface.npy")), meta
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with open(os.path.join(work, "ortho_meta.json")) as f:
        meta = json.load(f)
    return np.asarray(Image.open(os.path.join(work, "ortho.png")).convert("RGB")), np.load(os.path.join(work, "dsm.npy")), meta


def _window(meta, x, y, size, shape):
    res = meta["res"]
    half = int(size / res / 2)
    col, row = int((x - meta["x0"]) / res), int((meta["y1"] - y) / res)
    return (slice(max(0, row - half), min(shape[0], row + half)), slice(max(0, col - half), min(shape[1], col + half)))


def _report_sun(azimuth, elevation):
    y, m, d = SITE["date"]
    utc, az, el, sep = geom.closest_solar_time(SITE["lat"], SITE["lon"], y, m, d, azimuth, elevation)
    local = (utc + SITE["utc_offset"]) % 24.0
    return (f"nearest real sun on {y}-{m:02d}-{d:02d}: azimuth {az:.1f}, elevation {el:.1f} at "
            f"{int(local):02d}:{int(round(local % 1 * 60)) % 60:02d} local, {sep:.1f} degrees away")


def main(argv):
    from PIL import Image
    args = list(argv)
    full = "--full" in args
    if full:
        args.remove("--full")
    raster, suns = None, []
    while "--raster" in args:
        i = args.index("--raster")
        raster = args[i + 1]
        del args[i:i + 2]
    while "--sun" in args:
        i = args.index("--sun")
        suns.append((float(args[i + 1]), float(args[i + 2])))
        del args[i:i + 3]
    work, out_dir = args[0], args[1]
    if raster is None:
        raster = "photo" if os.path.isfile(os.path.join(work, "photo.npy")) else "ortho"
    os.makedirs(out_dir, exist_ok=True)
    rgb, height, meta = _load(work, raster)
    res = meta["res"]
    valid = np.isfinite(height)
    for r0 in range(0, rgb.shape[0], 1024):                       # black means no data
        valid[r0:r0 + 1024] &= rgb[r0:r0 + 1024].max(-1) > 0
    _log(f"{raster}: {rgb.shape[1]} x {rgb.shape[0]} at {res:g} m, {100.0 * valid.mean():.1f}% covered")
    info = {}
    out, shadow = remove_shadows(rgb, height, res, valid, sun=suns or None, log=_log, debug=info)
    for s in info.get("suns", []):
        print(f"  sun azimuth {s['azimuth']:.1f} elevation {s['elevation']:.1f}"
              + (f" ({100 * s['weight']:.0f}% of the evidence)" if "weight" in s else "")
              + "; " + _report_sun(s["azimuth"], s["elevation"]))

    # how much was treated, by what it fell on (rough colour classes of the result)
    step = 4
    a = shadow[::step, ::step]
    c = out[::step, ::step].astype(np.int16)
    b = rgb[::step, ::step].astype(np.int16)
    cell = (res * step) ** 2
    hit = a > 0.5
    grey = (c.max(-1) - c.min(-1) < 32) & (c.mean(-1) > 100)
    water = (b[..., 2] > b[..., 1] + 6) & (b[..., 2] > b[..., 0] + 12) & ~grey
    green = ~grey & ~water
    total = valid[::step, ::step].sum() * cell
    print(f"  treated as shadow: {hit.sum() * cell:.0f} m2 of {total:.0f} m2 ({100.0 * hit.sum() * cell / total:.2f}%): "
          f"pavement {(hit & grey).sum() * cell:.0f} m2, grass and other vegetation {(hit & green).sum() * cell:.0f} m2, "
          f"blue in shade (mostly water) {(hit & water).sum() * cell:.0f} m2")

    for name, (x, y, size) in SPOTS.items():
        win = _window(meta, x, y, size, valid.shape)
        if win[0].stop - win[0].start < 8 or win[1].stop - win[1].start < 8:
            continue
        pair = np.concatenate([rgb[win], out[win]], axis=1)
        Image.fromarray(pair).save(os.path.join(out_dir, f"{name}.png"))
    k = max(1, int(round(0.5 / res)))
    Image.fromarray(np.concatenate([rgb[::k, ::k], out[::k, ::k]], axis=1)).save(os.path.join(out_dir, "overview.png"))
    Image.fromarray((shadow[::k, ::k] * 255).astype(np.uint8)).save(os.path.join(out_dir, "overview_matte.png"))
    if full:
        np.save(os.path.join(out_dir, f"{raster}_noshadow.npy"), out)
        np.save(os.path.join(out_dir, f"{raster}_shadow.npy"), np.rint(shadow * 255).astype(np.uint8))
    _log(f"wrote crops to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
