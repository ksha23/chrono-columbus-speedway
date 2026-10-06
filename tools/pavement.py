"""Where the pavement is: a clean map of hard surface against everything else.

The photo decides wherever it can see the ground: concrete, asphalt and gravel have almost no
colour, grass and crops and water do. Where a tree's crown hides the ground the photo cannot
say, so a road is carried across the gap when there is pavement at both ends of a straight line
through it and nothing but canopy in between.

The result is kept as a signed distance to the pavement's edge, in metres, positive on the
pavement. The ground mesh is cut along its zero line.
"""
import numpy as np
from scipy import ndimage

RES = 0.10            # metres per cell of the pavement map
GAP = 45.0            # longest stretch of hidden road carried across, metres
DIRECTIONS = 48       # line directions tried when carrying a road across a gap
SMOOTH = 0.5          # metres of blur that turn the photo's ragged edge into a clean outline
NOTCH = 2.4           # metres: a bite into the pavement narrower than this is filled
BUMP = 2.0            # metres: a tongue of pavement narrower than this is trimmed off


def _disc(radius_cells):
    k = int(np.ceil(radius_cells))
    yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
    return xx * xx + yy * yy <= radius_cells * radius_cells


def _drop_small(mask, min_area, res):
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1)) * res * res
    return np.isin(labels, 1 + np.nonzero(sizes >= min_area)[0])


def to_fine(cells, shape, cell):
    """A map on coarse cells laid over the RES grid, each fine cell taking the coarse cell it is in."""
    rows = np.minimum((np.arange(shape[0]) * RES / cell).astype(int), cells.shape[0] - 1)
    cols = np.minimum((np.arange(shape[1]) * RES / cell).astype(int), cells.shape[1] - 1)
    return cells[rows][:, cols]


def to_cells(fine, shape, cell):
    """A boolean map on the RES grid reduced to coarse cells: true where most of the cell is."""
    small = ndimage.zoom(fine.astype(np.float32), RES / cell, order=1)
    out = np.zeros(shape, bool)
    h, w = min(shape[0], small.shape[0]), min(shape[1], small.shape[1])
    out[:h, :w] = small[:h, :w] > 0.5
    return out


def visible(photo, raster, covered_cells, blocked_cells, cell, shadow=None):
    """Pavement the photo shows, on RES-sized cells: colourless ground that is not a roof.

    shadow is the matte of where shadows were relit in this photo (full size, 0 to 255), if
    they were. Relit concrete keeps some of the blue of the skylight that lit it, so inside a
    relit shadow a second, looser test applies.
    """
    k = int(round(RES / raster["res"]))
    H, W = photo.shape[0] // k, photo.shape[1] // k
    paved = np.zeros((H, W), bool)
    for r0 in range(0, H, 512):
        r1 = min(r0 + 512, H)
        block = np.asarray(photo[r0 * k:r1 * k, :W * k], dtype=np.float32).reshape(r1 - r0, k, W, k, 3).mean((1, 3)) / 255
        mx, mn = block.max(-1), block.min(-1)
        sat = (mx - mn) / np.maximum(mx, 1e-3)
        # Sunlit concrete is bright and grey.
        paved[r0:r1] = (sat < 0.17) & (mx > 0.22)
        if shadow is not None:
            # Relit, shaded concrete comes out pale and a little blue, where relit grass comes
            # out yellow-green. Blue is the largest channel on the one and the smallest on the other.
            shaded = np.asarray(shadow[r0 * k:r1 * k, :W * k], dtype=np.float32).reshape(r1 - r0, k, W, k).mean((1, 3)) > 40
            bluish = (block[..., 2] >= block[..., 1]) & (block[..., 2] >= block[..., 0] * 0.97)
            paved[r0:r1] |= shaded & bluish & (sat < 0.35) & (mx > 0.22)
    paved &= to_fine(covered_cells & ~blocked_cells, paved.shape, cell)
    # Tidy: bridge paint lines and joints, fill what stands on the pavement (cones, weeds, the
    # footprint of a parked car), and drop grey odds and ends that are not part of the network.
    paved = ndimage.binary_closing(paved, structure=_disc(0.3 / RES))
    holes = ndimage.binary_fill_holes(paved) & ~paved
    paved |= holes & ~_drop_small(holes, 40.0, RES)
    paved = ndimage.binary_opening(paved, structure=_disc(0.4 / RES))
    return _drop_small(paved, 60.0, RES)


def carry_across(paved_cells, hidden_cells, cell):
    """Cells of hidden ground that lie on a straight line between two stretches of pavement.

    For each of DIRECTIONS line directions the maps are turned so the lines are rows. In a row,
    a run of hidden cells is filled when it has pavement on both ends and is short enough.
    """
    state = np.where(paved_cells, 1, np.where(hidden_cells, 2, 0)).astype(np.uint8)
    limit = GAP / cell
    filled = np.zeros(state.shape, np.float32)
    for angle in np.arange(DIRECTIONS) * 180.0 / DIRECTIONS:
        rot = ndimage.rotate(state, angle, order=0, reshape=True, mode="constant", cval=0)
        n = rot.shape[1]
        idx = np.arange(n)[None, :]
        solid = rot != 2
        # Position and value of the nearest cell that is not hidden, to the left and to the right.
        left = np.maximum.accumulate(np.where(solid, idx, -1), axis=1)
        right = np.minimum.accumulate(np.where(solid, idx, n)[:, ::-1], axis=1)[:, ::-1]
        rows = np.arange(rot.shape[0])[:, None]
        left_paved = (left >= 0) & (rot[rows, np.clip(left, 0, n - 1)] == 1)
        right_paved = (right < n) & (rot[rows, np.clip(right, 0, n - 1)] == 1)
        hit = (rot == 2) & left_paved & right_paved & ((right - left) <= limit)
        back = ndimage.rotate(hit.astype(np.float32), -angle, order=1, reshape=True, mode="constant", cval=0)
        r0, c0 = (back.shape[0] - state.shape[0]) // 2, (back.shape[1] - state.shape[1]) // 2
        filled = np.maximum(filled, back[r0:r0 + state.shape[0], c0:c0 + state.shape[1]])
    return (filled > 0.5) & hidden_cells


def build(photo, raster, masks, hidden_cells, blocked_cells, shadow=None, log=print):
    """Return (signed distance in metres on RES cells, the pavement mask on RES cells).

    shadow is the matte of the shadows relit in photo, if any were. masks is
    groundphoto.open_ground's result. hidden_cells marks ground the photo cannot see (under
    tree crowns), blocked_cells what can never be pavement (buildings).
    """
    cell = masks["cell"]
    seen = visible(photo, raster, masks["valid"], blocked_cells | hidden_cells, cell, shadow)
    seen_cells = to_cells(seen, hidden_cells.shape, cell)
    carried = carry_across(seen_cells, hidden_cells, cell)
    # Whatever is carried has to make road-sized pavement together with what was seen. That
    # keeps the strip under a crown that overhangs a road's edge, and drops thin stray links
    # between two roads through the woods.
    carried &= ndimage.binary_opening(seen_cells | carried, structure=_disc(1.5 / cell))
    paved = seen | to_fine(carried, seen.shape, cell)
    # A road's edge does not have bites in it. Fill any notch narrower than NOTCH.
    paved = ndimage.binary_closing(paved, structure=_disc(NOTCH / 2 / RES))
    # Nor does it have teeth: a tongue of pavement narrower than BUMP sticking out is trimmed.
    paved = ndimage.binary_opening(paved, structure=_disc(BUMP / 2 / RES))
    # Round the staircase of the carried cells and the pixel noise of the edge into one outline.
    soft = ndimage.gaussian_filter(paved.astype(np.float32), SMOOTH / RES)
    # Keep only the one connected network. Grey patches off on their own are a bare field, glare
    # on the pond or a gravel heap, not somewhere the track goes.
    labels, count = ndimage.label(soft > 0.5)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
    paved = labels == 1 + int(np.argmax(sizes))
    log(f"  pavement: {seen.sum() * RES * RES:.0f} m2 seen in the photo, {carried.sum() * cell * cell:.0f} m2 carried across under trees")
    inside = ndimage.distance_transform_edt(paved)
    outside = ndimage.distance_transform_edt(~paved)
    return ((inside - outside) * RES).astype(np.float32), paved


class Edge:
    """The signed distance field, with lookup at scene coordinates."""

    def __init__(self, distance, raster):
        self.distance = distance
        self.x0, self.y1 = raster["x0"], raster["y1"]

    def at(self, x, y):
        """Signed distance at scene points, metres. Outside the map counts as far off pavement."""
        r = (self.y1 - np.asarray(y, float)) / RES - 0.5
        c = (np.asarray(x, float) - self.x0) / RES - 0.5
        shape = np.shape(r)
        d = ndimage.map_coordinates(self.distance, [np.atleast_1d(r).ravel(), np.atleast_1d(c).ravel()], order=1, mode="constant", cval=-50.0)
        return d.reshape(shape)
