"""Colour side of the shadow remover: what a surface looks like lit, and how far each pixel
has dropped from that.

Colours are handled as natural logs of the 8-bit values, so a change of light is an offset
per channel. Shade moves a colour along one fixed direction in (log R/G, log B/G), because
skylight is bluer than sunlight whatever it lands on. The coordinate across that direction
(``material_key``) therefore stays put when a shadow passes over a surface, and is used to say
"same material" on both sides of a shadow edge.
"""
import math

import numpy as np
from scipy import ndimage as ndi

SHIFT_DEG = 121.5  # direction of the shade shift in (log R/G, log B/G), measured on this camera
_D = (math.cos(math.radians(SHIFT_DEG)), math.sin(math.radians(SHIFT_DEG)))
_E = (_D[1], -_D[0])  # across the shift: the part of the colour that shade does not change

# material key -> bin coordinate. Bins are narrow near grey, where pavement sits, and wide out
# in the greens.
KEY_SCALE = 0.05
KEY_LO, KEY_STEP, KEY_BINS = -3.15, 0.35, 15
Y_LO, Y_STEP, Y_BINS = -3.6, 0.12, 30

# How far shade drops each channel (log units), by material key. Used until the picture's own
# shadows have been measured, and wherever they are too few to measure.
DEFAULT_KAPPA_KEYS = np.array([-0.9, -0.4, -0.2, -0.1, 0.0, 0.05, 0.3], np.float32)
DEFAULT_KAPPA = np.array([[-1.25, -1.10, -0.62], [-1.19, -0.92, -0.53], [-0.94, -0.75, -0.48],
                          [-0.78, -0.62, -0.37], [-0.62, -0.53, -0.38], [-0.80, -0.70, -0.52],
                          [-0.81, -0.70, -0.52]], np.float32)


def log_rgb(rgb):
    return np.log((rgb.astype(np.float32) + 1.0) * np.float32(1.0 / 256.0))


def material_key(lp):
    """The shade-proof part of a colour, from log RGB."""
    return (lp[..., 0] - lp[..., 1]) * np.float32(_E[0]) + (lp[..., 2] - lp[..., 1]) * np.float32(_E[1])


def key_coord(key):
    """Fractional key-bin index (0 .. KEY_BINS - 1) of a material key."""
    z = np.sign(key) * np.log1p(np.abs(key) * np.float32(1.0 / KEY_SCALE))
    return np.clip((z - KEY_LO) / KEY_STEP, 0.0, KEY_BINS - 1.0).astype(np.float32)


def key_of_bin(b):
    z = KEY_LO + KEY_STEP * np.asarray(b, np.float32)
    return np.sign(z) * np.expm1(np.abs(z)) * KEY_SCALE


def default_kappa_table():
    centres = key_of_bin(np.arange(KEY_BINS))
    return np.stack([np.interp(centres, DEFAULT_KAPPA_KEYS, DEFAULT_KAPPA[:, c]) for c in range(3)], 1).astype(np.float32)


class LitReference:
    """For every place and material key, the colour that material has in full sun there.

    Built from the brighter part of each material's local brightness spread, over pixels the
    caller believes are probably lit, on a grid of ``cell`` metres smoothed over ``radius``."""

    def __init__(self, lp, key, use, res, cell=2.0, radius=10.0, quantile=0.65):
        self.res, self.f = res, max(1, int(round(cell / res)))
        f = self.f
        rows, cols = lp.shape[:2]
        self.gr, self.gc = -(-rows // f), -(-cols // f)
        gr, gc = self.gr, self.gc
        hist = np.zeros(gr * gc * KEY_BINS * Y_BINS, np.float32)
        chunk = f * 16
        for r0 in range(0, rows, chunk):
            sl = slice(r0, min(rows, r0 + chunk))
            m = use[sl]
            if not m.any():
                continue
            rr, cc = np.nonzero(m)
            y = lp[sl][rr, cc].mean(-1)
            kb = np.rint(key_coord(key[sl][rr, cc])).astype(np.int64)
            yb = np.clip(((y - Y_LO) / Y_STEP).astype(np.int64), 0, Y_BINS - 1)
            idx = ((((rr + r0) // f) * gc + cc // f) * KEY_BINS + kb) * Y_BINS + yb
            hist += np.bincount(idx, minlength=hist.size).astype(np.float32)
        hist = hist.reshape(gr, gc, KEY_BINS, Y_BINS)
        self.sig = radius / (res * f)
        y_ref, n = self._levels(hist, self.sig, quantile)
        y_wide, n_wide = self._levels(hist, self.sig * 4.0, quantile)
        g = hist.sum((0, 1))
        y_all, n_all = self._quantile(g, quantile)
        y_wide = np.where(n_wide >= 200.0, y_wide, np.where(n_all >= 50.0, y_all, -0.5)[None, None, :])
        # a small patch can sit wholly in shade, so never let the local level fall far below
        # what the same material shows over the wider area
        w = np.clip(n / 150.0, 0.0, 1.0)
        y_ref = w * np.maximum(y_ref, y_wide - 0.25) + (1.0 - w) * y_wide
        self.y_ref = y_ref.astype(np.float32)
        del hist

        # second pass: the mean colour of the pixels sitting at that level
        acc = np.zeros((gr * gc * KEY_BINS, 4), np.float64)
        for r0 in range(0, rows, chunk):
            sl = slice(r0, min(rows, r0 + chunk))
            m = use[sl]
            if not m.any():
                continue
            rr, cc = np.nonzero(m)
            v = lp[sl][rr, cc]
            kb = np.rint(key_coord(key[sl][rr, cc])).astype(np.int64)
            cell_i = ((rr + r0) // f) * gc + cc // f
            level = self.y_ref.reshape(-1, KEY_BINS)[cell_i, kb]
            near = np.abs(v.mean(-1) - level) < 0.2
            idx = (cell_i * KEY_BINS + kb)[near]
            v = v[near]
            for c in range(3):
                acc[:, c] += np.bincount(idx, weights=v[:, c], minlength=acc.shape[0])
            acc[:, 3] += np.bincount(idx, minlength=acc.shape[0])
        acc = acc.reshape(gr, gc, KEY_BINS, 4).astype(np.float32)
        tot = acc.sum((0, 1))
        col_all = tot[:, :3] / np.maximum(tot[:, 3:], 1.0)
        col = None
        for sig in (self.sig * 4.0, self.sig):
            s = ndi.gaussian_filter1d(ndi.gaussian_filter1d(acc, sig, axis=0, mode="nearest"), sig, axis=1, mode="nearest")
            s = ndi.convolve1d(s, np.array([0.2, 0.6, 0.2], np.float32), axis=2, mode="nearest")
            c_here = s[..., :3] / np.maximum(s[..., 3:], 1e-6)
            w = np.clip(s[..., 3:] * (2 * math.pi * sig * sig) / 150.0, 0.0, 1.0)
            if col is None:
                base = np.where(tot[:, 3:] >= 50.0, col_all, np.float32(-0.5))[None, None]
                col = w * c_here + (1.0 - w) * base
            else:
                col = w * c_here + (1.0 - w) * col
        self.col = np.ascontiguousarray(col, np.float32)  # (gr, gc, KEY_BINS, 3)

    @staticmethod
    def _quantile(h, q):
        cdf = np.cumsum(h, axis=-1)
        total = cdf[..., -1]
        k = np.argmax(cdf >= (q * total)[..., None], axis=-1)
        return (Y_LO + (k + 0.5) * Y_STEP).astype(np.float32), total

    def _levels(self, hist, sig, q):
        s = ndi.gaussian_filter1d(ndi.gaussian_filter1d(hist, sig, axis=0, mode="nearest"), sig, axis=1, mode="nearest")
        s = ndi.convolve1d(s, np.array([0.2, 0.6, 0.2], np.float32), axis=2, mode="nearest")
        y, total = self._quantile(s, q)
        return y, total * (2 * math.pi * sig * sig)  # roughly the pixels under the window

    def lookup(self, r0, r1, key_block, where=None):
        """Lit colour (log RGB) for rows r0:r1, either the whole block or just ``where``."""
        f = self.f
        if where is None:
            rr, cc = np.mgrid[r0:r1, 0:key_block.shape[1]]
            rr, cc, kk = rr.ravel(), cc.ravel(), key_block.ravel()
        else:
            rr, cc = np.nonzero(where)
            kk = key_block[rr, cc]
            rr = rr + r0
        y = np.clip((rr + 0.5) / f - 0.5, 0, self.gr - 1).astype(np.float32)
        x = np.clip((cc + 0.5) / f - 0.5, 0, self.gc - 1).astype(np.float32)
        y0 = np.minimum(y.astype(np.int64), self.gr - 2 if self.gr > 1 else 0)
        x0 = np.minimum(x.astype(np.int64), self.gc - 2 if self.gc > 1 else 0)
        fy, fx = (y - y0)[:, None], (x - x0)[:, None]
        kc = key_coord(kk)
        k0 = np.minimum(kc.astype(np.int64), KEY_BINS - 2)
        fk = (kc - k0)[:, None]
        y1, x1 = np.minimum(y0 + 1, self.gr - 1), np.minimum(x0 + 1, self.gc - 1)
        out = np.zeros((len(rr), 3), np.float32)
        for dy, wy in ((y0, 1 - fy), (y1, fy)):
            for dx, wx in ((x0, 1 - fx), (x1, fx)):
                out += wy * wx * ((1 - fk) * self.col[dy, dx, k0] + fk * self.col[dy, dx, k0 + 1])
        return out


def kappa_at(table, key):
    """Expected shade drop per channel for each material key, read off a (KEY_BINS, 3) table."""
    kc = key_coord(key)
    k0 = np.minimum(kc.astype(np.int64), KEY_BINS - 2)
    fk = (kc - k0)[..., None]
    return (1 - fk) * table[k0] + fk * table[k0 + 1]


def shade_amount(drop, kappa):
    """Split a pixel's drop below its lit colour into "how much shade" and "what else".

    ``amount`` is 0 for lit and 1 for the full expected shade. ``tint`` is the blue-minus-red
    shift left over once that much shade is accounted for: near 0 for a shadow, clearly
    negative for something that is merely dark (rubber, stains, wet concrete), positive for
    something bluer than shade would make it."""
    amount = (drop * kappa).sum(-1) / np.maximum((kappa * kappa).sum(-1), 1e-6)
    tint = (drop[..., 2] - drop[..., 0]) - amount * (kappa[..., 2] - kappa[..., 0])
    return amount.astype(np.float32), tint.astype(np.float32)
