"""Put the ground photo together: the scan where it has data, aerial imagery beyond."""
import numpy as np
from PIL import Image
from scipy import ndimage

import layout

FEATHER = 5.0   # metres over which the scan fades into the aerial picture at its ragged edge
TRIM = 2.0      # metres of the scan's own edge not trusted: its outermost triangles are smeared


def aerial_window(aerial, raster):
    """The part of the scene-wide aerial picture under the raster, as (array, rows, cols) slices."""
    res = layout.REFERENCE_RES
    c0 = int(round((raster["x0"] - layout.SCENE_X0) / res))
    r0 = int(round((layout.SCENE_Y1 - raster["y1"]) / res))
    w = int(round(raster["width"] * raster["res"] / res))
    h = int(round(raster["height"] * raster["res"] / res))
    return aerial[r0:r0 + h, c0:c0 + w]


def coarse(array, raster):
    """A raster-resolution array reduced to the reference resolution by block averaging."""
    f = int(round(layout.REFERENCE_RES / raster["res"]))
    h, w = array.shape[0] // f, array.shape[1] // f
    a = array[:h * f, :w * f].reshape(h, f, w, f, *array.shape[2:])
    return a.mean(axis=(1, 3))


def surround_and_blend(photo, covered, raster, aerial):
    """Return (aerial colour-matched to the scan, photo with every uncovered pixel filled from it).

    The match and the blend weight are worked out at the aerial picture's own resolution and
    then enlarged, which keeps a 180-megapixel job to a few seconds.
    """
    small_photo = coarse(photo, raster)
    small_cov = coarse(covered.astype(np.float32), raster) > 0.999
    window = aerial_window(aerial, raster)
    inside = ndimage.binary_erosion(small_cov, iterations=int(10 / layout.REFERENCE_RES))
    # The two pictures were taken in different seasons: summer for the aerial one, September
    # for the scan, so the same grass is two different greens. Fit one colour transform from
    # aerial to scan over the ground both of them show, by least squares, and apply it to the
    # whole aerial picture. Both are blurred first so a metre of misregistration does not matter.
    def blurred(img):
        return np.stack([ndimage.gaussian_filter(img[..., ch].astype(np.float32), 2.0) for ch in range(3)], -1)

    a = blurred(window)[inside]
    p = blurred(small_photo)[inside]
    terms = np.concatenate([a, a * a / 255.0, np.ones((len(a), 1), np.float32)], axis=1)
    coef, *_ = np.linalg.lstsq(terms, p, rcond=None)
    flat = aerial.reshape(-1, 3).astype(np.float32)
    matched = np.concatenate([flat, flat * flat / 255.0, np.ones((len(flat), 1), np.float32)], axis=1) @ coef
    matched = np.clip(matched, 0, 255).astype(np.uint8).reshape(aerial.shape)

    # Blend weight: 0 at TRIM inside the scan's edge, 1 at TRIM + FEATHER inside.
    depth = ndimage.distance_transform_edt(small_cov) * layout.REFERENCE_RES
    weight = np.clip((depth - TRIM) / FEATHER, 0, 1).astype(np.float32)

    size = (raster["width"], raster["height"])
    big_aerial = np.asarray(Image.fromarray(aerial_window(matched, raster)).resize(size, Image.BICUBIC))
    big_weight = np.asarray(Image.fromarray(weight).resize(size, Image.BILINEAR))
    big_weight = np.where(covered, big_weight, 0.0)[..., None]
    out = np.empty_like(photo)
    for r in range(0, photo.shape[0], 1024):   # in bands, to keep the float copies small
        s = slice(r, r + 1024)
        out[s] = (photo[s] * big_weight[s] + big_aerial[s] * (1 - big_weight[s]) + 0.5).astype(np.uint8)
    return matched, out
