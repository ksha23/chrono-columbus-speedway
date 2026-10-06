"""Repaint the ground where the drone photographed something standing on it.

From straight above, a tree is a picture of its crown and a building is a picture of its roof.
Once those are replaced by 3D models the picture left on the ground is wrong, so it is painted
over: the colour is carried in from the open ground around, and the fine grain is borrowed from
a clean stretch of the same kind of surface so the patch does not look airbrushed.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

GRAIN_TILE = 19.2    # metres, side of the borrowed grain patch
GRAIN_BLUR = 0.6     # metres: grain is the photo minus itself blurred this much
FEATHER = 0.5        # metres over which a repainted patch fades into the photo


def carry_colour(small, known):
    """Fill the unknown cells of a low-resolution picture from the known cells around them.

    Blurs at doubling radii, each normalised by how much known ground it saw. Near an edge the
    narrow blur wins, so local colour is kept, and deep inside a patch the wide ones take over.
    """
    img = small.astype(np.float32)
    w = known.astype(np.float32)
    out = np.zeros_like(img)
    out[:] = (img * w[..., None]).sum((0, 1)) / max(w.sum(), 1.0)
    for sigma in (128, 64, 32, 16, 8, 4, 2):
        ww = ndimage.gaussian_filter(w, sigma)
        blurred = np.stack([ndimage.gaussian_filter(img[..., c] * w, sigma) for c in range(3)], -1) / np.maximum(ww[..., None], 1e-6)
        alpha = np.clip(ww / 0.25, 0, 1)[..., None]
        out = alpha * blurred + (1 - alpha) * out
    return out


def find_grain(photo, raster, usable_cells, cell, want):
    """Cut a square of clean surface from the photo and return its fine grain, mirrored to tile.

    usable_cells marks low-resolution cells that are open ground of the wanted kind. want is
    "grass" or "pavement", used only for the message when nothing is found.
    """
    n = int(round(GRAIN_TILE / cell))
    frac = ndimage.uniform_filter(usable_cells.astype(np.float32), n)
    r, c = np.unravel_index(np.argmax(frac), frac.shape)
    if frac[r, c] < 0.97:
        raise SystemExit(f"no clean {GRAIN_TILE} m square of {want} in the scan (best is {100 * frac[r, c]:.0f}% clean)")
    k = int(round(cell / raster["res"]))
    size = n * k
    r0, c0 = (r - n // 2) * k, (c - n // 2) * k
    patch = np.asarray(photo[r0:r0 + size, c0:c0 + size], dtype=np.float32)
    sigma = GRAIN_BLUR / raster["res"]
    grain = patch - np.stack([ndimage.gaussian_filter(patch[..., ch], sigma, mode="reflect") for ch in range(3)], -1)
    # Mirror into a 2x2 block, which then repeats without a seam.
    top = np.concatenate([grain, grain[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1]], axis=0)


def repaint(photo, raster, mask_cells, known_cells, paved_cells, grains, cell, solid=None):
    """Return the photo with mask_cells repainted as ground.

    The *_cells are boolean on the low-resolution grid of size cell. known_cells is the open
    ground whose colour may be carried into the patches, paved_cells where the pavement is,
    hidden parts included. Pavement is repainted from pavement and everything else from
    everything else, so a road keeps its edge where a crown hung over it. grains is the pair of
    grain tiles from find_grain, grass then pavement. solid, if given, marks cells where the
    photo has nothing at all: there the paint goes on at full strength, with no feathering.
    """
    k = int(round(cell / raster["res"]))
    H, W = photo.shape[:2]
    h, w = mask_cells.shape
    small = np.stack([np.asarray(photo[..., ch], dtype=np.float32)[:h * k, :w * k].reshape(h, k, w, k).mean((1, 3)) for ch in range(3)], -1)
    colour = np.where(paved_cells[..., None], carry_colour(small, known_cells & paved_cells), carry_colour(small, known_cells & ~paved_cells))
    paved_like = paved_cells.astype(np.float32)

    soft = np.clip(ndimage.gaussian_filter(ndimage.binary_dilation(mask_cells, iterations=1).astype(np.float32), FEATHER / cell) * 1.6, 0, 1)
    if solid is not None:
        soft = np.maximum(soft, ndimage.binary_dilation(solid, iterations=1).astype(np.float32))
    out = np.array(photo)
    period = grains[0].shape[0]
    cols = np.arange(W) % period
    for r0 in range(0, H, 1024):
        r1 = min(r0 + 1024, H)
        a, b = r0 // k, min((r1 + k - 1) // k, h)
        if not soft[a:b].any():
            continue
        size = (W, (b - a) * k)
        alpha = np.asarray(Image.fromarray(soft[a:b]).resize(size, Image.BILINEAR))[r0 - a * k:r1 - a * k]
        if not (alpha > 0.002).any():
            continue
        base = np.stack([np.asarray(Image.fromarray(colour[a:b, :, ch]).resize(size, Image.BICUBIC)) for ch in range(3)], -1)[r0 - a * k:r1 - a * k]
        mix = np.asarray(Image.fromarray(paved_like[a:b].astype(np.float32)).resize(size, Image.BILINEAR))[r0 - a * k:r1 - a * k, :, None]
        rows = (np.arange(r0, r1) % period)[:, None]
        grain = grains[0][rows, cols[None, :]] * (1 - mix) + grains[1][rows, cols[None, :]] * mix
        painted = np.clip(base + grain, 0, 255)
        alpha = alpha[..., None]
        out[r0:r1] = (out[r0:r1] * (1 - alpha) + painted * alpha + 0.5).astype(np.uint8)
    return out


def repaint_small(photo, raster, mask, grain, margin=0.15):
    """Paint out small things on pavement in place: cones, and the short shadows beside them.

    mask is a full-size boolean array of what to remove. Each patch is grown by margin metres,
    filled with colour drawn in from its own rim, and given pavement grain.
    """
    res = raster["res"]
    grow = int(round(margin / res))
    labels, count = ndimage.label(mask)
    H, W = mask.shape
    period = grain.shape[0]
    pad = int(round(1.2 / res))
    for n, sl in enumerate(ndimage.find_objects(labels), start=1):
        r0, r1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        c0, c1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        hole = ndimage.binary_dilation(mask[r0:r1, c0:c1], iterations=grow)
        window = photo[r0:r1, c0:c1].astype(np.float32)
        known = (~hole).astype(np.float32)
        fillc = np.zeros_like(window)
        for sigma in (16, 8, 4, 2):   # wide first, then let nearer colour take over where it reaches
            ww = ndimage.gaussian_filter(known, sigma)
            blurred = np.stack([ndimage.gaussian_filter(window[..., ch] * known, sigma) for ch in range(3)], -1) / np.maximum(ww[..., None], 1e-6)
            alpha = np.clip(ww / 0.3, 0, 1)[..., None]
            fillc = alpha * blurred + (1 - alpha) * fillc
        rows = (np.arange(r0, r1) % period)[:, None]
        cols = (np.arange(c0, c1) % period)[None, :]
        painted = np.clip(fillc + grain[rows, cols], 0, 255)
        soft = np.clip(ndimage.gaussian_filter(hole.astype(np.float32), 1.0) * 1.5, 0, 1)[..., None]
        photo[r0:r1, c0:c1] = (window * (1 - soft) + painted * soft + 0.5).astype(np.uint8)
    return count


def _blur_over(values, mask, sigma):
    """Gaussian blur of values taken over mask only: cells outside it neither give nor dilute."""
    w = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
    out = np.stack([ndimage.gaussian_filter(values[..., ch] * mask, sigma) for ch in range(values.shape[-1])], -1)
    return out / np.maximum(w[..., None], 1e-6), w


# Inside a relit shadow, everything coarser than this is replaced by the sunlit tone. Metres, by
# kind of ground: pavement is one even material, grass has texture worth keeping, and a pond
# has none.
FLECK = {"paved": 0.15, "land": 0.30, "water": 0.60}


def _sunlit_grain(photo, raster, cells, cell, fleck):
    """How rough sunlit ground of one kind is at scales finer than fleck: the typical size of its detail."""
    k = int(round(cell / raster["res"]))
    rows, cols = np.nonzero(ndimage.binary_erosion(cells, iterations=int(round(2.0 / cell))))
    if not len(rows):
        return 4.0
    pick = np.random.default_rng(0).choice(len(rows), size=min(40, len(rows)), replace=False)
    sizes = []
    for r, c in zip(rows[pick], cols[pick]):
        patch = np.asarray(photo[r * k - 40:r * k + 40, c * k - 40:c * k + 40], dtype=np.float32)
        if patch.shape[:2] != (80, 80):
            continue
        detail = patch - np.stack([ndimage.gaussian_filter(patch[..., ch], fleck / raster["res"]) for ch in range(3)], -1)
        sizes.append(np.sqrt((detail ** 2).mean()))
    return float(np.median(sizes)) if sizes else 4.0


def _flatten(band, weight, tone, fleck, grain, res, texture=None, keep=1.0):
    """Inside a shadow, swap everything coarser than fleck for the sunlit tone. In place.

    Sun through leaves also leaves a lacework of sharp edges finer than fleck. Where the fine
    detail is rougher than sunlit ground of this kind ever is, it is turned down to that level,
    which fades the lace and leaves an ordinary crack or tuft alone.
    """
    coarse = np.stack([ndimage.gaussian_filter(band[..., ch], fleck / res) for ch in range(3)], -1)
    detail = band - coarse
    rough = np.sqrt(ndimage.gaussian_filter((detail ** 2).mean(-1), 0.4 / res))
    # keep says how much of the fine detail survives at most. On pavement it is none: the
    # shadow of a twig and a crack in the concrete cannot be told apart, and a clean patch is
    # better than the ghost of a tree.
    calm = keep * np.minimum(1.0, 1.4 * grain / np.maximum(rough, 1e-3))
    # Yellow paint is strong fine detail too, and has to stay. Nothing a shadow leaves behind is
    # that colour: red and green well above blue.
    paint = np.clip(((band[..., 0] + band[..., 1]) / 2 - band[..., 2] - 0.12 * band.max(-1)) / (0.08 * np.maximum(band.max(-1), 1.0)), 0, 1)
    calm = np.maximum(calm, ndimage.maximum_filter(paint, 3))[..., None]
    # What is turned down is made up with grain borrowed from clean ground of the same kind, so
    # a levelled patch is not smoother than the ground around it.
    filler = 0.0 if texture is None else texture * (1.0 - calm)
    band += weight[..., None] * (tone + detail * calm + filler - band)


KEEP = {"paved": 0.0, "land": 1.0, "water": 1.0}    # how much fine detail a levelled patch keeps


def even_out(photo, raster, shadow, known_cells, paved_cells, water_cells, cell, paved_fine, grains=None, before=None):
    """Level what is left of a shadow after relighting, and return the result.

    Relighting multiplies a shadow back up, which is right for a shadow with a clean edge and
    leaves a blotchy patch where sun came through leaves. Here each shadowed stretch is brought
    to the tone of the sunlit ground of the same kind around it, the kinds being pavement, water
    and everything else. First its own average over half a metre is swapped for the sunlit
    average carried in from outside. Then a finer pass takes out the dapple and keeps joints,
    cracks, most of a paint line and the grain of grass: see _flatten.

    shadow is the full-size matte, 0 to 255. paved_fine is the pavement map at twice the
    photo's pixel size, which puts the line between the pavement's treatment and the grass's
    exactly on the pavement's edge. grains is the pair of grain tiles from find_grain, grass
    then pavement, used to make up the texture the second pass takes out. before is the photo
    as it was ahead of relighting: relighting also brightens a fringe around each shadow that
    the matte does not cover, and comparing the two shows where.
    """
    k = int(round(cell / raster["res"]))
    res = raster["res"]
    h, w = known_cells.shape

    def cells(a):
        return np.asarray(a, dtype=np.float32)[:h * k, :w * k].reshape((h, k, w, k) + a.shape[2:]).mean((1, 3))

    small = np.stack([cells(photo[..., ch]) for ch in range(3)], -1)
    amount = cells(shadow) / 255.0
    if before is not None:
        was = np.stack([cells(before[..., ch]) for ch in range(3)], -1)
        amount = np.maximum(amount, np.clip(np.abs(small - was).max(-1) / 12.0, 0, 1))
    # The reference is ground well clear of any shadow. Right beside one, relighting leaves a
    # pale rim, and a tone taken from there would make every levelled patch too bright.
    sunlit = known_cells & (ndimage.maximum_filter(amount, 2 * int(round(2.0 / cell)) + 1) < 0.02)
    kinds = {"paved": paved_cells, "water": water_cells & ~paved_cells, "land": ~paved_cells & ~water_cells}
    textures = {"paved": None, "land": None, "water": None} if grains is None else {"paved": grains[1], "land": grains[0], "water": None}
    tones, levels = {}, {}
    ratio = np.ones_like(small)
    for name, kind in kinds.items():
        tones[name] = carry_colour(small, sunlit & kind)
        levels[name] = _sunlit_grain(photo, raster, sunlit & kind, cell, FLECK[name])
        here = known_cells & kind
        local, seen = _blur_over(small, here, 0.5 / cell)
        ok = here & (seen > 0.2)
        ratio[ok] = tones[name][ok] / np.maximum(local[ok], 4.0)
    ratio = np.clip(ratio, 0.5, 2.5)
    # Where sun came through leaves the matte is faint, and that fringe needs the treatment as
    # much as the middle does. So the weight is full wherever there was any real shadow.
    # The treated area reaches three quarters of a metre past it as well: a shadow's last
    # fingers are thinner than the matte can follow.
    reach = ndimage.maximum_filter(np.clip(amount * 4.0, 0, 1), 2 * int(round(0.75 / cell)) + 1)
    weight = np.clip(ndimage.gaussian_filter(reach, 2.0) * 1.5, 0, 1) * known_cells
    gain = 1.0 + weight[..., None] * (ratio - 1.0)

    def enlarge(a, rows, size):
        return np.asarray(Image.fromarray(np.ascontiguousarray(a[rows])).resize(size, Image.BILINEAR))

    # Water and land shade into each other over a couple of metres, as a bank does. The line
    # between pavement and the rest stays sharp: that one comes from paved_fine.
    wet = ndimage.gaussian_filter(kinds["water"].astype(np.float32), 1.0 / cell)
    soft = {"water": wet, "land": (1.0 - wet) * ~paved_cells}

    out = np.array(photo)
    f = 2   # paved_fine cells per photo pixel, each way
    for r0 in range(0, h * k, 1024):
        r1 = min(r0 + 1024, h * k)
        a, b = r0 // k, (r1 + k - 1) // k
        if not (weight[a:b] > 0.01).any():
            continue
        size, rows, cut = (w * k, (b - a) * k), slice(a, b), slice(r0 - a * k, r1 - a * k)
        g = np.stack([enlarge(gain[..., ch], rows, size) for ch in range(3)], -1)[cut]
        band = out[r0:r1, :w * k].astype(np.float32) * g
        on_road = np.repeat(np.repeat(paved_fine[r0 // f:(r1 + f - 1) // f], f, axis=0), f, axis=1)[r0 % f:r0 % f + (r1 - r0), :w * k]
        on_road = np.pad(on_road, ((0, 0), (0, w * k - on_road.shape[1])))
        for name in kinds:
            wt = enlarge(weight, rows, size)[cut] * on_road if name == "paved" else enlarge(weight * soft[name], rows, size)[cut] * ~on_road
            if (wt > 0.01).any():
                tone = np.stack([enlarge(tones[name][..., ch], rows, size) for ch in range(3)], -1)[cut]
                tile = textures[name]
                texture = None if tile is None else tile[(np.arange(r0, r1) % tile.shape[0])[:, None], (np.arange(w * k) % tile.shape[1])[None, :]]
                _flatten(band, wt, tone, FLECK[name], levels[name], res, texture, KEEP[name])
        out[r0:r1, :w * k] = np.clip(band + 0.5, 0, 255).astype(np.uint8)
    return out
