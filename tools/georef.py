#!/usr/bin/env python3
"""Find where the scan sits on the map, by matching its orthophoto to USGS aerial imagery.

    python -I georef.py WORK_DIR REFERENCE_DIR

The scan's coordinates are metres east and north of an origin WebODM did not export. This
finds that origin in UTM zone 16N by cross-correlating pavement in the two pictures, then checks
the answer on the two ends of the site separately: if the scan were rotated or mis-scaled, the
ends would disagree. It then compares heights against the USGS 3DEP elevation model.

Writes WORK_DIR/georef.json.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

# The reference pictures: bounds in UTM 16N metres, as requested from the USGS services.
REF_E0, REF_N0, REF_E1, REF_N1 = 328800.0, 4793900.0, 330100.0, 4794900.0
RES = 0.5


def pavement(rgb):
    """A soft score for bright, colourless surface: concrete and gravel, not grass or crops."""
    rgb = rgb.astype(np.float32) / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    score = np.clip((mx - 0.45) / 0.2, 0, 1) * np.clip((0.22 - sat) / 0.1, 0, 1)
    return ndimage.gaussian_filter(score, 1.0)


def best_shift(ref, mov):
    """Offset (rows, cols) of mov's top-left corner inside ref that maximises their correlation."""
    a = ref - ref.mean()
    b = mov - mov.mean()
    corr = signal.fftconvolve(a, b[::-1, ::-1], mode="valid")
    # Normalise by the local energy of the reference so bright fields do not win by size alone.
    energy = signal.fftconvolve(a * a, np.ones_like(b), mode="valid")
    ncc = corr / np.sqrt(np.maximum(energy, 1e-6) * (b * b).sum())
    r, c = np.unravel_index(np.argmax(ncc), ncc.shape)
    # Sub-pixel peak by fitting a parabola through the neighbours.
    def refine(m1, m0, p1):
        d = m1 - 2 * m0 + p1
        return 0.0 if abs(d) < 1e-12 else 0.5 * (m1 - p1) / d
    dr = refine(ncc[r - 1, c], ncc[r, c], ncc[r + 1, c]) if 0 < r < ncc.shape[0] - 1 else 0.0
    dc = refine(ncc[r, c - 1], ncc[r, c], ncc[r, c + 1]) if 0 < c < ncc.shape[1] - 1 else 0.0
    return r + dr, c + dc, float(ncc[r, c])


def main():
    work, ref_dir = sys.argv[1], sys.argv[2]
    with open(os.path.join(work, "ortho_meta.json")) as f:
        meta = json.load(f)
    step = int(round(RES / meta["res"]))
    ortho = Image.open(os.path.join(work, "ortho.png"))
    small = np.asarray(ortho.resize((ortho.width // step, ortho.height // step), Image.BOX))
    dsm = np.load(os.path.join(work, "dsm.npy"))
    covered = np.isfinite(dsm)[: small.shape[0] * step : step, : small.shape[1] * step : step]
    mov = pavement(small) * covered
    naip = np.asarray(Image.open(os.path.join(ref_dir, "naip_utm.png")).convert("RGB"))
    ref = pavement(naip)

    def origin(rows, cols, label):
        r, c, score = best_shift(ref, mov[rows, cols])
        # Pixel (r, c) of the reference is where the window's top-left corner landed.
        east_of_window = REF_E0 + c * RES
        north_of_window = REF_N1 - r * RES
        x_window = meta["x0"] + cols.start * RES
        y_window = meta["y1"] - rows.start * RES
        e0, n0 = east_of_window - x_window, north_of_window - y_window
        print(f"  {label:12s} origin E {e0:11.2f}  N {n0:11.2f}   correlation {score:.3f}")
        return e0, n0

    H, W = mov.shape
    print("scan origin in UTM 16N, from pavement matching:")
    e0, n0 = origin(slice(0, H), slice(0, W), "whole site")
    # The loop at the north-east end and the buildings at the south-west end, matched separately.
    ne = origin(slice(0, H // 2), slice(W // 2, W), "north-east")
    sw = origin(slice(H // 2, H), slice(0, W // 2), "south-west")
    span = np.hypot(W * RES / 2, H * RES / 2)
    disagreement = np.hypot(ne[0] - sw[0], ne[1] - sw[1])
    print(f"  the two ends disagree by {disagreement:.2f} m over about {span:.0f} m"
          f" ({1e3 * disagreement / span:.1f} mm per metre of rotation or scale error at most)")

    lat, lon = geo.from_utm(e0, n0, 16)
    print(f"  scan (0, 0) is at latitude {lat:.6f}, longitude {lon:.6f}")

    # Heights: sample the 3DEP model (1 m posts) at pavement pixels of the scan.
    dem = np.asarray(Image.open(os.path.join(ref_dir, "dem_utm.tif")), dtype=np.float32)
    rows, cols = np.nonzero((mov > 0.8) & covered)
    pick = np.random.default_rng(0).choice(len(rows), size=min(200000, len(rows)), replace=False)
    rows, cols = rows[pick], cols[pick]
    x = meta["x0"] + (cols + 0.5) * RES
    y = meta["y1"] - (rows + 0.5) * RES
    z_scan = dsm[rows * step, cols * step]
    dr = (REF_N1 - (y + n0)) / 1.0 - 0.5
    dc = ((x + e0) - REF_E0) / 1.0 - 0.5
    z_dem = ndimage.map_coordinates(dem, [dr, dc], order=1, mode="nearest")
    diff = z_scan - z_dem
    ok = np.isfinite(diff)
    diff, x, y = diff[ok], x[ok], y[ok]
    med = float(np.median(diff))
    print(f"\nheights on pavement, scan minus USGS 3DEP ({len(diff)} points):")
    print(f"  median {med:+.2f} m, spread (5th to 95th percentile) {np.percentile(diff, 5) - med:+.2f} to {np.percentile(diff, 95) - med:+.2f} m")
    # A tilt of the scan would show as the difference growing across the site.
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, diff, rcond=None)
    resid = diff - A @ coef
    print(f"  fitted tilt: {1e3 * coef[0]:+.2f} mm/m east, {1e3 * coef[1]:+.2f} mm/m north; residual spread after removing it "
          f"{np.percentile(resid, 5):+.2f} to {np.percentile(resid, 95):+.2f} m")
    print(f"  3DEP heights at those points: {np.percentile(z_dem[ok], 1):.1f} to {np.percentile(z_dem[ok], 99):.1f} m")

    out = {
        "utm_zone": 16, "origin_easting": round(e0, 2), "origin_northing": round(n0, 2),
        "origin_latitude": round(lat, 7), "origin_longitude": round(lon, 7),
        "height_minus_3dep_median": round(med, 3),
        "tilt_mm_per_m": [round(1e3 * float(coef[0]), 3), round(1e3 * float(coef[1]), 3)],
        "ends_disagree_m": round(float(disagreement), 2),
    }
    with open(os.path.join(work, "georef.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
