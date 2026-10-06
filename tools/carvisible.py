"""Which triangles of a mesh can be seen from outside it, and from which side.

A parked car is looked at from all round and from above, never from inside and hardly from
below. So the mesh is drawn, without perspective, from a hundred directions into depth
buffers, and each triangle is asked whether it shows in any of them. What never shows is
left out of the model, and what shows only its back is turned over, because the renderer
culls back faces.
"""
import numpy as np

from carmesh import SAMPLES, face_normals, flip


def view_directions(elevations=((-20.0, 8), (-10.0, 12), (-4.0, 12), (2.0, 16), (12.0, 16), (25.0, 12),
                                (40.0, 12), (60.0, 8), (80.0, 4), (90.0, 1))):
    """Unit vectors a viewer looks along: so many around the compass at each elevation of
    the viewer above the horizon, in degrees."""
    out = []
    for elevation, count in elevations:
        e = np.radians(elevation)
        for k in range(count):
            a = 2.0 * np.pi * (k + 0.5 * (count % 3)) / count
            out.append((-np.cos(e) * np.cos(a), -np.cos(e) * np.sin(a), -np.sin(e)))
    return np.array(out)


def _basis(direction):
    d = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, d)
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    return np.stack([u, w, d], axis=1)       # columns: screen x, screen y, depth


def _depth_buffer(S, width, height):
    """Draw triangles S (T, 3, 3), given as (x, y, depth) in pixels. Returns the nearest
    depth at each pixel centre and, for each triangle, the number of centres where it is that."""
    buffer = np.full(width * height, np.inf)
    x, y, z = S[:, :, 0], S[:, :, 1], S[:, :, 2]
    i0 = np.ceil(x.min(axis=1) - 0.5).astype(np.int64)
    i1 = np.floor(x.max(axis=1) - 0.5).astype(np.int64)
    j0 = np.ceil(y.min(axis=1) - 0.5).astype(np.int64)
    j1 = np.floor(y.max(axis=1) - 0.5).astype(np.int64)
    denominator = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    usable = (i1 >= i0) & (j1 >= j0) & (np.abs(denominator) > 1e-9)
    # Triangles are drawn in batches that share a bounding-box size, rounded up to powers of two.
    wide = np.where(usable, np.ceil(np.log2(np.maximum(i1 - i0 + 1, 1))), -1).astype(np.int64)
    tall = np.where(usable, np.ceil(np.log2(np.maximum(j1 - j0 + 1, 1))), -1).astype(np.int64)
    drawn = []
    for a in range(int(wide.max()) + 1):
        for b in range(int(tall.max()) + 1):
            chosen = np.flatnonzero((wide == a) & (tall == b))
            if not len(chosen):
                continue
            w, h = 1 << a, 1 << b
            oy, ox = np.divmod(np.arange(w * h), w)
            step = max(1, 1000000 // (w * h))
            for c0 in range(0, len(chosen), step):
                t = chosen[c0:c0 + step]
                px = i0[t, None] + ox[None, :]
                py = j0[t, None] + oy[None, :]
                cx = px + 0.5 - x[t, 0:1]
                cy = py + 0.5 - y[t, 0:1]
                b1 = (cx * (y[t, 2:3] - y[t, 0:1]) - (x[t, 2:3] - x[t, 0:1]) * cy) / denominator[t, None]
                b2 = ((x[t, 1:2] - x[t, 0:1]) * cy - cx * (y[t, 1:2] - y[t, 0:1])) / denominator[t, None]
                inside = (b1 >= -1e-9) & (b2 >= -1e-9) & (b1 + b2 <= 1.0 + 1e-9)
                inside &= (px >= 0) & (px < width) & (py >= 0) & (py < height)
                depth = (z[t, 0:1] + b1 * (z[t, 1:2] - z[t, 0:1]) + b2 * (z[t, 2:3] - z[t, 0:1]))[inside]
                pixel = (py * width + px)[inside]
                np.minimum.at(buffer, pixel, depth)
                drawn.append((np.broadcast_to(t[:, None], inside.shape)[inside], pixel, depth))
    wins = np.zeros(len(S), dtype=np.int64)
    for triangle, pixel, depth in drawn:
        wins += np.bincount(triangle[depth <= buffer[pixel]], minlength=len(S))
    return buffer.reshape(height, width), wins


class Views:
    """Depth buffers of a set of triangles, drawn without perspective along each direction,
    against which triangles can then be tested for being seen."""

    def __init__(self, triangles, directions, pixel=0.008):
        self.pixel = pixel
        self.views = []
        for direction in directions:
            direction = np.asarray(direction, dtype=np.float64)
            basis = _basis(direction)
            S = (triangles @ basis) / pixel
            low = S.reshape(-1, 3).min(axis=0)
            offset = np.array([1.0 - low[0], 1.0 - low[1], 0.0])
            S += offset
            width = int(S[:, :, 0].max()) + 3
            rows = int(S[:, :, 1].max()) + 3
            buffer, wins = _depth_buffer(S, width, rows)
            self.views.append((direction, basis, offset, width, rows, buffer, wins))

    def seen(self, P, slack=0.003, ground=None, near=2.5, first=None):
        """Count the views in which each triangle shows its front, and those in which it shows
        its back: (front, back), each (T,).

        A triangle shows its front when at one of seven points on it nothing lies in front of
        it by more than a slack of a few millimetres, a little more when it is steeply tilted.
        That test is generous, and at an outline it would also pass the triangles just round
        the corner, facing away. So a triangle shows its back only when it is the nearest
        surface at two pixel centres or more, which needs it to be one of the triangles drawn:
        first says where P starts among them. Without first, backs are not counted.

        With ground given (its z), a view from below the horizon only counts for triangles
        high enough that an eye at least near metres away would still be above the ground.
        """
        fn, _ = face_normals(P)
        height = P[:, :, 2].mean(axis=1)
        front = np.zeros(len(P), dtype=np.int64)
        back = np.zeros(len(P), dtype=np.int64)
        for direction, basis, offset, width, rows, buffer, wins in self.views:
            n = fn @ basis
            facing = np.where(np.abs(n[:, 2]) < 0.05, 0.05, n[:, 2])
            slope = -n[:, :2] / facing[:, None]                     # depth per pixel across the triangle
            allowed = np.minimum(slack / self.pixel + 0.25 * np.abs(slope).sum(axis=1), 0.02 / self.pixel)
            points = np.einsum("sk,tkc->tsc", SAMPLES, (P @ basis) / self.pixel + offset)
            px = np.floor(points[:, :, 0]).astype(np.int64)
            py = np.floor(points[:, :, 1]).astype(np.int64)
            outside = (px < 0) | (px >= width) | (py < 0) | (py >= rows)
            nearest = buffer[np.clip(py, 0, rows - 1), np.clip(px, 0, width - 1)]
            # The buffer holds depths at pixel centres, so the triangle's own plane is followed there.
            depth = (points[:, :, 2] + slope[:, None, 0] * (px + 0.5 - points[:, :, 0])
                     + slope[:, None, 1] * (py + 0.5 - points[:, :, 1]))
            seen = (outside | (depth <= nearest + allowed[:, None])).any(axis=1)
            won = wins[first:first + len(P)] >= 2 if first is not None else np.zeros(len(P), dtype=bool)
            if ground is not None and direction[2] > 0:
                high_enough = height >= ground + 0.1 + near * direction[2]
                seen &= high_enough
                won &= high_enough
            front += seen & (n[:, 2] < -0.08)
            back += won & (n[:, 2] > 0.08)
        return front, back


def seen_from_outside(P, directions, occluders=None, pixel=0.008, ground=None):
    """Views.seen for triangles that also hide one another, with more that only hide."""
    everything = P if occluders is None else np.concatenate([P, occluders])
    return Views(everything, directions, pixel).seen(P, ground=ground, first=0)


def outside_only(P, N, directions, occluders, pixel=0.008, ground=None):
    """Keep the triangles that can be seen, each facing the way it is seen from. One seen
    from both sides, a thin sheet, is kept twice. Returns (P, N, index of each in the input)."""
    front, back = seen_from_outside(P, directions, occluders=occluders, pixel=pixel, ground=ground)
    reversed_too = (back > 0) & ((front == 0) | (back >= np.maximum(3, 0.25 * front)))
    flipped_P, flipped_N = flip(P, N)
    index = np.concatenate([np.flatnonzero(front > 0), np.flatnonzero(reversed_too)])
    return (np.concatenate([P[front > 0], flipped_P[reversed_too]]),
            np.concatenate([N[front > 0], flipped_N[reversed_too]]), index)
