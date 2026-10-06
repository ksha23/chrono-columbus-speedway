"""The photo read across a line: what lies to either side of each point of a path.

Paint is judged by how it stands out from the pavement beside it, so the tests that judge it
(markings_solid.py for yellow lines, groundphoto.py for white ones) all start from the same
thing: the photo sampled along a path and across it.
"""
import numpy as np
from scipy import ndimage

import railmodel

FINE = 0.05        # metres between the points a path is sampled at


def yellowness(rgb):
    """How yellow each pixel is: red and green over blue. Concrete is near 20, yellow paint 50 and more."""
    return (rgb[..., 0] + rgb[..., 1]) / 2 - rgb[..., 2]


def brightness(rgb):
    return rgb.mean(-1)


def leafiness(rgb):
    """1 where a pixel is the green of a leaf, 0 elsewhere."""
    return ((rgb[..., 1] > 1.04 * rgb[..., 0]) & (rgb[..., 1] > 1.15 * rgb[..., 2])).astype(np.float32)


def across(photo, raster, path, offsets, measures):
    """Sample the photo along a path and across it.

    path is resampled every FINE metres. Returns (the resampled path, its unit normals, one
    array per measure of shape (points, offsets)). A positive offset is to the left of the
    path's direction. Each measure is a function of an RGB window, such as yellowness.
    """
    res = raster["res"]
    fine = railmodel.resample(path, FINE)
    tangent = np.gradient(fine, axis=0)
    tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    x = fine[:, None, 0] + offsets[None, :] * normal[:, None, 0]
    y = fine[:, None, 1] + offsets[None, :] * normal[:, None, 1]
    c0, c1 = max(int((x.min() - raster["x0"]) / res) - 2, 0), min(int((x.max() - raster["x0"]) / res) + 3, photo.shape[1])
    r0, r1 = max(int((raster["y1"] - y.max()) / res) - 2, 0), min(int((raster["y1"] - y.min()) / res) + 3, photo.shape[0])
    window = np.asarray(photo[r0:r1, c0:c1, :3], dtype=np.float32)
    at = [(raster["y1"] - y) / res - 0.5 - r0, (x - raster["x0"]) / res - 0.5 - c0]
    return fine, normal, [ndimage.map_coordinates(measure(window), at, order=1, mode="nearest") for measure in measures]


def band(profile, offsets, search, beside, least):
    """The stripe that stands out in a profile across a line: (centre, width, height) or None.

    The stripe's top is looked for within search of the middle, and its height is taken
    above the level further than beside from the middle. It reaches as far to either side as
    the profile stays above three tenths of that height, a second stripe within 0.6 m
    included: two lines side by side are one band. None if the height is under least.
    """
    base = float(np.median(profile[np.abs(offsets) > beside]))
    top = int(np.argmax(np.where(np.abs(offsets) <= search, profile, -np.inf)))
    height = float(profile[top] - base)
    if height < least:
        return None
    labels, count = ndimage.label(profile > base + 0.3 * height)
    keep = [n for n in range(1, count + 1) if abs(offsets[labels == n].mean() - offsets[top]) <= 0.6 and (labels == n).sum() >= 2]
    if not keep:
        return None
    inside = offsets[np.isin(labels, keep)]
    step = float(offsets[1] - offsets[0])
    return float((inside.min() + inside.max()) / 2), float(inside.max() - inside.min() + step), height
