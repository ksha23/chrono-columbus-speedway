"""Find what stands on the ground in the scan: trees, buildings, parked vehicles.

The scan's top surface minus the lidar's bare earth is height above ground. Everything here
works on that at 0.25 m per cell, which is plenty for a tree crown and a hundred times faster
than the photo's own resolution.
"""
import numpy as np
from scipy import ndimage

CELL = 0.25


def block(array, factor, how):
    """Reduce a 2-D array by whole blocks, with nanmax, nanmean or any."""
    h, w = array.shape[0] // factor, array.shape[1] // factor
    a = array[:h * factor, :w * factor].reshape(h, factor, w, factor, *array.shape[2:])
    with np.errstate(all="ignore"):
        if how == "max":
            return np.nanmax(a, axis=(1, 3))
        if how == "mean":
            return np.nanmean(a, axis=(1, 3))
    raise ValueError(how)


def height_above_ground(surface, raster, ref):
    """Height of the scan's top surface above the lidar ground, on CELL-sized cells.

    Returns (height, x, y): height is NaN where the scan has nothing, x and y are the scene
    coordinates of the cell centres.
    """
    factor = int(round(CELL / raster["res"]))
    top = block(surface, factor, "max")
    h, w = top.shape
    x = raster["x0"] + (np.arange(w) + 0.5) * CELL
    y = raster["y1"] - (np.arange(h) + 0.5) * CELL
    gx, gy = np.meshgrid(x, y)
    return (top - ref.elevation(gx, gy)).astype(np.float32), gx, gy


def roughness(height, radius=1.0):
    """Scatter of height about a locally fitted plane, metres. Roofs are smooth, crowns are not."""
    k = max(1, int(round(radius / CELL)))
    size = 2 * k + 1
    filled = np.where(np.isfinite(height), height, 0.0).astype(np.float64)
    # Plane fit in a square window by moments: remove the mean and the two slopes.
    mean = ndimage.uniform_filter(filled, size)
    sq = ndimage.uniform_filter(filled * filled, size)
    ramp = np.arange(-k, k + 1, dtype=np.float64)
    var_r = (ramp**2).mean()
    kx = np.tile(ramp, (size, 1)) / size**2
    cov_x = ndimage.correlate(filled, kx)
    cov_y = ndimage.correlate(filled, kx.T)
    resid = sq - mean**2 - cov_x**2 / var_r - cov_y**2 / var_r
    return np.sqrt(np.maximum(resid, 0)).astype(np.float32)


def classify(height, photo_small):
    """Split what stands above the ground into buildings, trees and vehicles.

    height is height above ground on CELL cells, photo_small the photo averaged onto the same
    cells. Returns a dict of boolean masks.
    """
    valid = np.isfinite(height)
    h = np.where(valid, height, 0.0)
    rough = roughness(h)
    rgb = photo_small.astype(np.float32) / 255
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    green = (rgb[..., 1] >= rgb[..., 0] * 0.95) & (rgb[..., 1] > rgb[..., 2] * 1.08) & (sat > 0.18)

    # Buildings: tall, smooth on top, not green, and big enough to be a roof.
    tall = valid & (h > 2.2)
    smooth = ndimage.binary_opening(tall & (rough < 0.12) & ~green, iterations=2)
    labels, count = ndimage.label(smooth)
    sizes = ndimage.sum(smooth, labels, index=np.arange(1, count + 1)) * CELL**2
    buildings = np.isin(labels, 1 + np.nonzero(sizes > 25.0)[0])
    # Grow each roof back out to its eaves, which the opening above shaved off.
    buildings = ndimage.binary_dilation(buildings, iterations=4) & valid & (h > 1.5) & ~green

    # A building's eaves and whatever leans on its walls stand above the ground too, and from
    # above they carry the roof's colour. Nothing within two metres of a roof is taken for a tree.
    near_building = ndimage.binary_dilation(buildings, iterations=int(round(2.0 / CELL)))
    trees = valid & (h > 1.5) & ~near_building
    trees = ndimage.binary_opening(trees, iterations=1)

    saplings = _saplings(h, valid, rgb, trees | near_building)

    # Vehicles and other clutter on open ground: low, compact, not vegetation.
    low = valid & (h > 0.5) & (h <= 2.6) & ~buildings & ~green & ~ndimage.binary_dilation(trees, iterations=4)
    low = ndimage.binary_opening(low, iterations=1)
    labels, count = ndimage.label(low)
    sizes = ndimage.sum(low, labels, index=np.arange(1, count + 1)) * CELL**2
    vehicles = np.isin(labels, 1 + np.nonzero((sizes > 3.0) & (sizes < 40.0))[0])
    return {"buildings": buildings, "trees": trees | saplings, "vehicles": vehicles, "green": green, "rough": rough}


SAPLING_TOP = 0.7           # metres: the least a young tree or a bush stands
SAPLING_AREA = (0.25, 7.0)  # square metres its crown covers, seen from above


def _saplings(h, valid, rgb, taken):
    """Young trees and bushes standing alone in grass, too low to be counted with the trees.

    A row of planted spruces a metre high is as much in the photo as an oak is: a dark dot
    with a shadow beside it. So anything that rises well clear of the ground on a small
    footprint, is the colour of a plant, and has low green ground all round it is taken for
    one. The last condition leaves out tall weeds, which come in stretches, and anything
    standing on pavement, which is a cone or a post and is found by other means.
    """
    smooth = ndimage.gaussian_filter(h, 1.0)
    rise = valid & (smooth > 0.4) & ~ndimage.binary_dilation(taken, iterations=2)
    labels, count = ndimage.label(rise)
    if count == 0:
        return np.zeros_like(valid)
    index = np.arange(1, count + 1)
    area = ndimage.sum(rise, labels, index) * CELL**2
    top = ndimage.maximum(h, labels, index)
    colour = np.stack([ndimage.mean(rgb[..., ch], labels, index) for ch in range(3)], axis=1)
    plant = ((colour[:, 1] >= 0.92 * colour[:, 0]) & (colour[:, 1] > colour[:, 2])) | (colour.max(1) < 0.42)
    # The ground within a metre and a half of it: low, and green.
    ring = np.where(labels == 0, ndimage.grey_dilation(labels, size=(13, 13)), 0)
    low = ndimage.mean((smooth < 0.3).astype(np.float32), ring, index)
    around = np.stack([ndimage.mean(rgb[..., ch], ring, index) for ch in range(3)], axis=1)
    grassy = (around[:, 1] > 1.1 * around[:, 2]) & (around[:, 1] >= 0.9 * around[:, 0])
    keep = (area >= SAPLING_AREA[0]) & (area <= SAPLING_AREA[1]) & (top >= SAPLING_TOP) & plant & (low > 0.8) & grassy
    return np.isin(labels, index[keep])


def find_trees(height, tree_mask, gx, gy):
    """Split the tree mask into single trees.

    A tree is a local peak of the smoothed crown surface. The taller the tree the wider the
    neighbourhood in which it must be the highest point, which stops one broad crown from
    being counted as several. Crown cells are then given to the nearest peak they connect to
    downhill. Returns a list of dicts with x, y, height and crown radius.
    """
    h = np.where(tree_mask, np.nan_to_num(height, nan=0.0), 0.0)
    smooth = ndimage.gaussian_filter(h, 0.6 / CELL)
    peaks = np.zeros_like(tree_mask)
    # Three size classes of neighbourhood, chosen by the height at the candidate itself.
    for lo, hi, radius in [(0.5, 1.5, 1.0), (1.5, 5.0, 1.5), (5.0, 11.0, 2.6), (11.0, 99.0, 3.8)]:
        k = int(round(radius / CELL))
        yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
        local = ndimage.maximum_filter(smooth, footprint=(xx * xx + yy * yy) <= k * k)
        peaks |= (smooth == local) & (smooth >= lo) & (smooth < hi) & tree_mask
    labels, count = ndimage.label(peaks)
    if count == 0:
        return [], np.zeros(h.shape, np.int32)
    centres = np.array(ndimage.center_of_mass(peaks, labels, index=np.arange(1, count + 1)))
    markers = np.zeros(h.shape, np.int32)
    rows = np.clip(np.round(centres[:, 0]).astype(int), 0, h.shape[0] - 1)
    cols = np.clip(np.round(centres[:, 1]).astype(int), 0, h.shape[1] - 1)
    markers[rows, cols] = np.arange(1, count + 1)

    # Watershed on the inverted crown surface: each cell joins the peak it drains to.
    scaled = np.clip(255 - smooth / max(smooth.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    crowns = ndimage.watershed_ift(scaled, markers)
    crowns[~tree_mask] = 0

    area = ndimage.sum(tree_mask, crowns, index=np.arange(1, count + 1)) * CELL**2
    top = ndimage.maximum(h, crowns, index=np.arange(1, count + 1))
    trees = []
    for n in range(count):
        # A crown's area goes with its height: a metre-high spruce covers a quarter of a
        # square metre, and a patch that small at the top of an oak is a stray branch.
        if top[n] < SAPLING_TOP or area[n] < np.clip(0.15 * top[n] ** 2, SAPLING_AREA[0], 1.0):
            continue
        trees.append({"id": n + 1, "x": float(gx[rows[n], cols[n]]), "y": float(gy[rows[n], cols[n]]),
                      "height": float(top[n]), "radius": float(np.sqrt(area[n] / np.pi))})
    return trees, crowns


def water(ref, gx, gy):
    """Where the lidar ground is dead level over a wide area: the pond.

    Lidar does not return from water, so the survey's makers set each water body to one flat
    level. Nothing else out here is flat to the millimetre over hundreds of square metres.
    """
    step = float(abs(gx[0, 1] - gx[0, 0]))
    z = ref.elevation(gx, gy)
    slope = np.hypot(*np.gradient(z, step))
    flat = ndimage.binary_opening(slope < 2e-4, iterations=4)
    labels, count = ndimage.label(flat)
    sizes = ndimage.sum(flat, labels, index=np.arange(1, count + 1)) * step**2
    pond = np.isin(labels, 1 + np.nonzero(sizes > 300.0)[0])
    # The lidar's posts leave the outline stepped. Round it off.
    return ndimage.gaussian_filter(pond.astype(np.float32), 1.5 / step) > 0.5
