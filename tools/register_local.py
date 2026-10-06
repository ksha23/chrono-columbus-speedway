#!/usr/bin/env python3
"""Pin the scan to the lidar horizontally, window by window, using the shape of the ground.

    python -I register_local.py WORK_DIR REFERENCE_DIR

register.py showed the scan is domed, so its horizontal error is unlikely to be a plain affine
map either. Here every 80 m window of open ground is slid over the lidar surface on its own, up
to 6 m each way, and the slide that makes the two height fields agree best is kept. Within a
window the scan's dome, tilt and height scale are absorbed by a local quadratic, so only real
relief (ditches, banks, road crowns) decides the match. Windows with no relief give a flat
answer and are dropped.

Polynomial maps of rising order are then fitted through the window matches, and the same
windows matched against aerial imagery (georef_local.py) are scored as an outside check.

Writes WORK_DIR/warp.json: the chosen map both ways, scan to true and true to scan.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo  # noqa: E402
import register  # noqa: E402

REF_E0, REF_N1 = register.REF_E0, register.REF_N1
WINDOW, STRIDE, SEARCH, STEP = 80.0, 40.0, 6.0, 0.25


def poly_terms(x, y, order):
    """Monomials x^i y^j with i + j <= order, on coordinates scaled to about +-1."""
    x, y = x / 400.0, y / 400.0
    return np.stack([x**i * y**j for i in range(order + 1) for j in range(order + 1 - i)], axis=1)


def fit_poly(src, dst, order):
    A = poly_terms(src[:, 0], src[:, 1], order)
    coef, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return coef


def apply_poly(coef, order, x, y):
    return poly_terms(np.asarray(x, float), np.asarray(y, float), order) @ coef


def main():
    work, ref_dir = sys.argv[1], sys.argv[2]
    dem = np.asarray(Image.open(os.path.join(ref_dir, "dem_utm.tif")), dtype=np.float64)
    pts, _ = register.open_ground(work, spacing=0.5)
    start = np.array(json.load(open(os.path.join(work, "transform.json")))["plane"])

    # Filter the elevation raster for cubic interpolation once, not on every lookup.
    filtered = ndimage.spline_filter(dem, order=3)

    def dem_at(e, n):
        return ndimage.map_coordinates(filtered, [REF_N1 - n - 0.5, e - REF_E0 - 0.5], order=3, mode="nearest", prefilter=False)

    shifts = np.arange(-SEARCH, SEARCH + 1e-9, STEP)
    dn, de = np.meshgrid(shifts, shifts, indexing="ij")
    src, dst, weight = [], [], []
    print("window centre (scan m)   best slide (m)    rms at best / at zero / worst   points")
    for cx in np.arange(pts[:, 0].min() + WINDOW / 2, pts[:, 0].max(), STRIDE):
        for cy in np.arange(pts[:, 1].min() + WINDOW / 2, pts[:, 1].max(), STRIDE):
            m = (np.abs(pts[:, 0] - cx) < WINDOW / 2) & (np.abs(pts[:, 1] - cy) < WINDOW / 2)
            if m.sum() < 1500:
                continue
            p = pts[m]
            if len(p) > 4000:
                p = p[np.random.default_rng(1).choice(len(p), 4000, replace=False)]
            x, y, z = p[:, 0] - cx, p[:, 1] - cy, p[:, 2]
            e = start[0] * p[:, 0] + start[1] * p[:, 1] + start[4]
            n = start[2] * p[:, 0] + start[3] * p[:, 1] + start[5]
            # Least squares of z on [local quadratic, lidar height at this slide], for every slide
            # at once: project the quadratic out of both, then it is a one-variable regression.
            base, _ = np.linalg.qr(np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], 1))
            zr = z - base @ (base.T @ z)
            d = dem_at(e[None, None, :] + de[:, :, None], n[None, None, :] + dn[:, :, None])
            dr = d - (d @ base) @ base.T
            explained = (dr @ zr) ** 2 / np.maximum((dr * dr).sum(-1), 1e-12)
            rms = np.sqrt(np.maximum(zr @ zr - explained, 0) / len(z))
            i, j = np.unravel_index(np.argmin(rms), rms.shape)
            best, zero, worst = rms[i, j], rms[len(shifts) // 2, len(shifts) // 2], rms.max()
            edge = i in (0, len(shifts) - 1) or j in (0, len(shifts) - 1)
            # A match counts when the misfit rises clearly away from it: at 2 m off in every direction.
            ring = [rms[min(max(i + a, 0), len(shifts) - 1), min(max(j + b, 0), len(shifts) - 1)]
                    for a, b in [(-8, 0), (8, 0), (0, -8), (0, 8)]]
            sharp = min(ring) / best
            ok = (not edge) and sharp > 1.25
            print(f"  ({cx:7.1f}, {cy:7.1f})   ({shifts[j]:+5.2f}, {shifts[i]:+5.2f})   {best:.3f} / {zero:.3f} / {worst:.3f}"
                  f"   {len(p):5d}   sharpness {sharp:.2f} {'' if ok else ' dropped'}")
            if ok:
                src.append((cx, cy))
                dst.append((start[0] * cx + start[1] * cy + start[4] + shifts[j], start[2] * cx + start[3] * cy + start[5] + shifts[i]))
                weight.append(sharp)

    src, dst = np.array(src), np.array(dst)
    print(f"\n{len(src)} windows matched on terrain")
    np.savez(os.path.join(work, "terrain_matches.npz"), src=src, dst=dst)

    print("\npolynomial map through the terrain matches, leave-one-out error (m):")
    results = {}
    for order in (1, 2, 3):
        errs = []
        for k in range(len(src)):
            keep = np.arange(len(src)) != k
            coef = fit_poly(src[keep], dst[keep], order)
            errs.append(np.hypot(*(apply_poly(coef, order, src[k:k + 1, 0], src[k:k + 1, 1])[0] - dst[k])))
        errs = np.array(errs)
        results[order] = errs
        terms = poly_terms(src[:, 0], src[:, 1], order).shape[1]
        print(f"  order {order} ({terms} terms per axis): median {np.median(errs):.2f}, 90th percentile {np.percentile(errs, 90):.2f}, worst {errs.max():.2f}")

    order = min(results, key=lambda o: np.percentile(results[o], 90))
    # Prefer the simpler map unless the richer one is clearly better.
    for o in sorted(results):
        if np.percentile(results[o], 90) < 1.15 * np.percentile(results[order], 90):
            order = o
            break
    forward = fit_poly(src, dst - dst.mean(0), order)
    centre = dst.mean(0)
    inverse = fit_poly(dst - centre, src, order)
    back = apply_poly(inverse, order, *(apply_poly(forward, order, src[:, 0], src[:, 1])).T)
    print(f"\nchosen: order {order}. Forward then inverse returns the window centres within {np.hypot(*(back - src).T).max():.3f} m")

    e0, n0 = apply_poly(forward, order, [0.0], [0.0])[0] + centre
    lat, lon = geo.from_utm(e0, n0, 16)
    json.dump({"order": order, "centre": centre.tolist(), "forward": forward.tolist(), "inverse": inverse.tolist(),
               "windows": len(src), "loo_median_m": float(np.median(results[order])), "loo_p90_m": float(np.percentile(results[order], 90)),
               "utm_zone": 16, "scan_origin_easting": float(e0), "scan_origin_northing": float(n0),
               "scan_origin_latitude": lat, "scan_origin_longitude": lon},
              open(os.path.join(work, "warp.json"), "w"), indent=1)

    # Outside check: the windows matched against aerial imagery, which this fit never saw.
    img = os.path.join(work, "imagery_matches.npz")
    if os.path.isfile(img):
        d = np.load(img)
        pred = apply_poly(forward, order, d["src"][:, 0], d["src"][:, 1]) + centre
        err = np.hypot(*(pred - d["dst"]).T)
        mean = (pred - d["dst"]).mean(0)
        print(f"against {len(err)} aerial-imagery windows: median {np.median(err):.2f} m, worst {err.max():.2f} m, mean offset ({mean[0]:+.2f}, {mean[1]:+.2f}) m")


if __name__ == "__main__":
    main()
