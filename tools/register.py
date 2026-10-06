#!/usr/bin/env python3
"""Fit the scan onto the USGS 3DEP lidar elevation model.

    python -I register.py WORK_DIR REFERENCE_DIR

The scan was georeferenced from the drone's own GPS with no ground control. Measured against
the lidar, which is survey grade and was flown after the track was built, the scan is about
3.5% too large, tilted, and domed: its ground bends away from the truth as a quadratic surface,
metres high at the ends of the site. A rigid or similarity transform cannot remove that, so
the mapping fitted here has two parts:

    east, north = A [x, y] + t                         a plane affine map, 6 numbers
    height      = c0 + c1 z + c2 x + c3 y + c4 x^2 + c5 y^2 + c6 x y    7 numbers

chosen so the scan's open ground (pavement and mown grass) lands on the lidar surface. The fit
starts from the scale, heading and position that georef_local.py found against aerial imagery,
an independent source, and reports how far it moved from them.

Writes WORK_DIR/transform.json. transform.py applies it.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
REF_E0, REF_N1 = 328800.0, 4794900.0   # top-left corner of the reference rasters, 1 m posts


def open_ground(work, spacing=0.5):
    """Scan points on pavement and mown grass: smooth surface, nothing standing on it."""
    meta = json.load(open(os.path.join(work, "ortho_meta.json")))
    step = int(round(spacing / meta["res"]))
    dsm = np.load(os.path.join(work, "dsm.npy"))
    rgb = np.asarray(Image.open(os.path.join(work, "ortho.png")))
    dsm, rgb = dsm[::step, ::step], rgb[::step, ::step].astype(np.float32) / 255
    ok = np.isfinite(dsm)
    filled = np.where(ok, dsm, np.nanmedian(dsm))
    # Rough surface (trees, brush, crops) has height scatter within a couple of metres.
    mean = ndimage.uniform_filter(filled, 5)
    rough = np.sqrt(np.maximum(ndimage.uniform_filter(filled**2, 5) - mean**2, 0))
    # Anything standing well above the lowest ground nearby is a building or a tree.
    low = ndimage.minimum_filter(filled, int(30 / spacing))
    mx, mn = rgb.max(-1), rgb.min(-1)
    paved = (mx > 0.5) & ((mx - mn) / np.maximum(mx, 1e-3) < 0.16)
    keep = ok & (rough < 0.06) & (filled - low < 2.5)
    keep &= ndimage.binary_erosion(ok, iterations=int(6 / spacing))
    rows, cols = np.nonzero(keep)
    x = meta["x0"] + (cols + 0.5) * spacing
    y = meta["y1"] - (rows + 0.5) * spacing
    return np.stack([x, y, dsm[rows, cols]], axis=1).astype(np.float64), paved[rows, cols]


def height_terms(pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack([np.ones_like(x), z, x, y, x * x, y * y, x * y], axis=1)


def main():
    work, ref_dir = sys.argv[1], sys.argv[2]
    dem = np.asarray(Image.open(os.path.join(ref_dir, "dem_utm.tif")), dtype=np.float64)
    pts, paved = open_ground(work)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(pts), size=min(150000, len(pts)), replace=False)
    pts, paved = pts[pick], paved[pick]
    terms = height_terms(pts)
    print(f"{len(pts)} open-ground points from the scan, {100 * paved.mean():.0f}% of them pavement")

    def dem_at(e, n):
        return ndimage.map_coordinates(dem, [REF_N1 - n - 0.5, e - REF_E0 - 0.5], order=1, mode="nearest")

    def plane(h):
        """h = (a, b, c, d, te, tn): east = a x + b y + te, north = c x + d y + tn."""
        return h[0] * pts[:, 0] + h[1] * pts[:, 1] + h[4], h[2] * pts[:, 0] + h[3] * pts[:, 1] + h[5]

    def heights_for(h):
        """Best height coefficients for a given plane map, and the residual they leave."""
        target = dem_at(*plane(h))
        keep = np.ones(len(pts), bool)
        for _ in range(3):
            c, *_ = np.linalg.lstsq(terms[keep], target[keep], rcond=None)
            r = terms @ c - target
            keep = np.abs(r) < max(0.25, 3 * np.median(np.abs(r[keep])))
        return c, r

    def report(label, r):
        print(f"  {label:38s} median |error| {np.median(np.abs(r)):.3f} m   95% within {np.percentile(np.abs(r), 95):.2f} m"
              f"   pavement: {np.median(np.abs(r[paved])):.3f} m")

    guess = json.load(open(os.path.join(work, "georef_local.json")))
    th, s = np.radians(guess["rotation_deg"]), guess["scale"]
    start = np.array([s * np.cos(th), -s * np.sin(th), s * np.sin(th), s * np.cos(th), guess["east"], guess["north"]])

    print("scan ground against the lidar surface:")
    first = json.load(open(os.path.join(work, "georef.json")))
    raw = np.array([1.0, 0, 0, 1.0, first["origin_easting"], first["origin_northing"]])
    r = pts[:, 2] - dem_at(*plane(raw))
    report("as delivered, best height offset", r - np.median(r))
    target = dem_at(*plane(start))
    A = np.stack([np.ones(len(pts)), pts[:, 2], pts[:, 0], pts[:, 1]], 1)
    c4, *_ = np.linalg.lstsq(A, target, rcond=None)
    report("scaled, turned, levelled (no dome)", A @ c4 - target)
    report("with the dome removed", heights_for(start)[1])

    # Now let the terrain itself say where the scan sits. Units: a metre at the far end of the site.
    unit = np.array([2.5e-3, 2.5e-3, 2.5e-3, 2.5e-3, 1.0, 1.0])
    fit = optimize.least_squares(lambda u: heights_for(start + u * unit)[1], np.zeros(6), bounds=(-8, 8),
                                 loss="soft_l1", f_scale=0.1, diff_step=0.05)
    h = start + fit.x * unit
    c, r = heights_for(h)
    report("plane map refined on the terrain", r)

    sx, sy = np.hypot(h[0], h[2]), np.hypot(h[1], h[3])
    heading = np.degrees(np.arctan2(h[2], h[0]))
    skew = np.degrees(np.arctan2(-h[1], h[3])) - heading
    print(f"\n  scale east-west {sx:.4f}, north-south {sy:.4f}   heading {heading:+.3f} deg   skew {skew:+.3f} deg")
    print(f"  aerial imagery said scale {s:.4f}, heading {guess['rotation_deg']:+.3f} deg")
    corners = np.array([[-350, -250], [300, 230], [-150, -30], [200, 100], [0, 0]], float)
    moved = [np.hypot((h[0] - start[0]) * x + (h[1] - start[1]) * y + h[4] - start[4],
                      (h[2] - start[2]) * x + (h[3] - start[3]) * y + h[5] - start[5]) for x, y in corners]
    print(f"  the terrain fit and the imagery fit place the site within {min(moved):.2f} to {max(moved):.2f} m of each other")
    dome = c[4] * 350**2
    print(f"  height: z scale {c[1]:.3f}, tilt {1e3 * c[2]:+.1f} mm/m east {1e3 * c[3]:+.1f} mm/m north,"
          f" dome {1e6 * c[4]:+.1f} / {1e6 * c[5]:+.1f} / {1e6 * c[6]:+.1f} mm per m^2 (x^2, y^2, xy), about {dome:+.1f} m at 350 m")

    e, n = plane(h)
    print("\n  leftover height error on pavement by 120 m block (m), north at the top:")
    for n1 in np.arange(n.max(), n.min(), -120):
        row = []
        for e0 in np.arange(e.min(), e.max(), 120):
            m = (e >= e0) & (e < e0 + 120) & (n <= n1) & (n > n1 - 120) & paved
            row.append(f"{np.median(r[m]):+6.2f}" if m.sum() > 150 else "   .  ")
        print("   ", " ".join(row))

    lat, lon = geo.from_utm(h[4], h[5], 16)
    out = {"plane": [float(v) for v in h], "height": [float(v) for v in c], "utm_zone": 16,
           "origin_latitude": lat, "origin_longitude": lon,
           "median_abs_error_m": float(np.median(np.abs(r))), "p95_abs_error_m": float(np.percentile(np.abs(r), 95))}
    json.dump(out, open(os.path.join(work, "transform.json"), "w"), indent=1)
    print(f"\n  scan (0, 0) is at E {h[4]:.2f} N {h[5]:.2f}, latitude {lat:.6f} longitude {lon:.6f}")


if __name__ == "__main__":
    main()
