"""Move points between the scan's own coordinates and the scene frame.

The plane part is the polynomial map register_local.py fitted through terrain matches. The
height part removes the scan's dome and tilt, fitted here against the lidar with that plane map
held fixed. Both are empirical corrections to a scan that had no ground control.
"""
import json
import os

import numpy as np

import layout


def _terms(x, y, order):
    x, y = np.asarray(x, float) / 400.0, np.asarray(y, float) / 400.0
    return np.stack([x**i * y**j for i in range(order + 1) for j in range(order + 1 - i)], axis=-1)


class Transform:
    def __init__(self, work):
        w = json.load(open(os.path.join(work, "warp.json")))
        self.order = w["order"]
        self.centre = np.array(w["centre"])
        self.forward = np.array(w["forward"])
        self.inverse = np.array(w["inverse"])
        self.info = w
        path = os.path.join(work, "height_model.json")
        self.height = np.array(json.load(open(path))["coefficients"]) if os.path.isfile(path) else None

    def to_scene(self, x, y):
        """Scan (x, y) to scene (x, y)."""
        en = _terms(x, y, self.order) @ self.forward + self.centre
        return en[..., 0] - layout.ORIGIN_E, en[..., 1] - layout.ORIGIN_N

    def to_scan(self, sx, sy):
        """Scene (x, y) to scan (x, y)."""
        e = np.asarray(sx, float) + layout.ORIGIN_E - self.centre[0]
        n = np.asarray(sy, float) + layout.ORIGIN_N - self.centre[1]
        xy = _terms(e, n, self.order) @ self.inverse
        return xy[..., 0], xy[..., 1]

    def height_terms(self, x, y, z):
        x, y, z = (np.asarray(v, float) for v in (x, y, z))
        u, v = x / 400.0, y / 400.0
        return np.stack([np.ones_like(u), z, u, v, u * u, v * v, u * v, u**3, v**3, u * u * v, u * v * v], axis=-1)

    def to_elevation(self, x, y, z):
        """Scan (x, y, z) to real elevation in metres."""
        return self.height_terms(x, y, z) @ self.height
