"""What colour are a building's walls? Read from the scan's own triangles along each wall.

The drone looked nearly straight down, so walls are thinly covered and part of each is in
shade. The answer is the typical colour of the brighter half of what was seen.
"""
import numpy as np
from PIL import Image

import scanlib

FALLBACK = (0.74, 0.73, 0.70)


def sample(scan_dir, work, found, transform):
    """One sRGB colour (0..1) per building in found."""
    v, vt, tri_v, tri_t, tri_mat, names = scanlib.load_scan(scan_dir, work)
    sx, sy = transform.to_scene(v[:, 0], v[:, 1])
    p = np.stack([sx, sy, transform.to_elevation(v[:, 0], v[:, 1], v[:, 2])], axis=1)
    normal, _ = scanlib.triangle_normals(p, tri_v)
    centre = p[tri_v].mean(1)
    steep = np.abs(normal[:, 2]) < 0.5
    colours = []
    for b in found:
        c, u = np.array(b["centre"]), np.array(b["axis"])
        w = np.array([-u[1], u[0]])
        a, bb = (centre[:, :2] - c) @ u, (centre[:, :2] - c) @ w
        # A band around the outline: just inside the wall line to a metre outside it.
        outer = (np.abs(a) < b["half_length"] + 1.0) & (np.abs(bb) < b["half_width"] + 1.0)
        inner = (np.abs(a) < b["half_length"] - 0.6) & (np.abs(bb) < b["half_width"] - 0.6)
        z = centre[:, 2] - b["ground"]
        sel = np.nonzero(steep & outer & ~inner & (z > 0.6) & (z < b["eave"] - 0.3))[0]
        seen = []
        for m in np.unique(tri_mat[sel]):
            tex = np.asarray(Image.open(scanlib.texture_path(scan_dir, names[m])).convert("RGB"))
            th, tw = tex.shape[:2]
            uv = vt[tri_t[sel[tri_mat[sel] == m]]].mean(1)
            tx = np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1)
            ty = np.clip(((1 - uv[:, 1]) * th).astype(int), 0, th - 1)
            seen.append(tex[ty, tx])
        if not seen or sum(len(s) for s in seen) < 40:
            colours.append(FALLBACK)
            continue
        seen = np.concatenate(seen).astype(np.float32) / 255
        bright = seen.max(1)
        colours.append(tuple(float(x) for x in np.median(seen[bright >= np.median(bright)], axis=0)))
    return colours
