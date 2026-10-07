"""Noise that depends on where a point is in the world, and on nothing else.

The ground is drawn tile by tile, and whatever is added to it must not show where one tile ends
and the next begins. So nothing here is drawn from a random stream, which would depend on the
order things are asked in. The world is covered by a lattice and every lattice point is hashed
from its two whole-number coordinates and a seed. Two tiles that ask about the same point get
the same answer, and the same question gets the same answer on every run.

Everything made from this is synthesized: it is texture of a plausible kind, not a measurement
of the site.
"""
import numpy as np
from scipy import special

_A, _B, _C = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F), np.uint64(0x165667B19E3779F9)
_M1, _M2, _S = np.uint64(0xFF51AFD7ED558CCD), np.uint64(0xC4CEB9FE1A85EC53), np.uint64(33)

# A bell curve by table: 16 bits pick one of 65536 equally likely values of a standard normal.
_NORMAL = special.ndtri((np.arange(65536) + 0.5) / 65536.0).astype(np.float32)
# What a cubic spline through unit noise has left of its spread, on average over where it is asked.
_SPLINE = 0.4793


def _mix(h):
    """Stir 64-bit words so that every input bit reaches every output bit."""
    h = h ^ (h >> _S)
    h = h * _M1
    h = h ^ (h >> _S)
    h = h * _M2
    return h ^ (h >> _S)


def bits(ix, iy, seed):
    """64 random bits for lattice points (ix, iy). The two broadcast, so a row and a column give a grid."""
    with np.errstate(over="ignore"):
        hx = _mix(np.asarray(ix, np.int64).astype(np.uint64) * _A + np.uint64(seed) * _C)
        return _mix(hx + np.asarray(iy, np.int64).astype(np.uint64) * _B)


def uniform(h, k=0):
    """A number in 0..1 from each word. k = 0..3 picks which 16 of its bits, so one word gives four."""
    return (((h >> np.uint64(16 * k)) & np.uint64(0xFFFF)).astype(np.float32) + 0.5) / 65536.0


def normal(h, k=0):
    """A standard normal number from each word. k as in uniform."""
    return _NORMAL[((h >> np.uint64(16 * k)) & np.uint64(0xFFFF)).astype(np.intp)]


def _taps(c, pitch):
    """For coordinates c: the first of the four lattice points a cubic spline reads, and their weights."""
    t = np.asarray(c, np.float64) / pitch
    first = np.floor(t)
    f = (t - first).astype(np.float32)
    weights = np.stack([(1 - f) ** 3, 3 * f ** 3 - 6 * f ** 2 + 4, -3 * f ** 3 + 3 * f ** 2 + 3 * f + 1, f ** 3]) / 6.0
    return first.astype(np.int64) - 1, weights


def smooth_grid(xs, ys, pitch, seed):
    """Smooth unit noise with features about pitch metres across, on the grid of columns xs and rows ys."""
    ix, wx = _taps(xs, pitch)
    iy, wy = _taps(ys, pitch)
    x0, y0 = int(ix.min()), int(iy.min())
    lattice = normal(bits(np.arange(x0, int(ix.max()) + 4)[None, :], np.arange(y0, int(iy.max()) + 4)[:, None], seed))
    rows = sum(wy[k][:, None] * lattice[iy - y0 + k] for k in range(4))
    return sum(rows[:, ix - x0 + k] * wx[k][None, :] for k in range(4)) / _SPLINE


def smooth_at(x, y, pitch, seed):
    """The same noise as smooth_grid, at separate points."""
    ix, wx = _taps(x, pitch)
    iy, wy = _taps(y, pitch)
    total = np.zeros(np.shape(x), np.float32)
    for k in range(4):
        for m in range(4):
            total += wy[k] * wx[m] * normal(bits(ix + m, iy + k, seed))
    return total / _SPLINE
