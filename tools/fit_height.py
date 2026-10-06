#!/usr/bin/env python3
"""Fit the scan's height correction against the lidar, with the plane map already fixed.

    python -I fit_height.py WORK_DIR REFERENCE_DIR

Writes WORK_DIR/height_model.json and prints what is left over, overall and by place. Half the
points are held out of the fit and scored separately, so the leftover is not just the fit
admiring itself.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference  # noqa: E402
import register  # noqa: E402
import transform  # noqa: E402


def main():
    work, ref_dir = sys.argv[1], sys.argv[2]
    ref = reference.Reference(ref_dir)
    t = transform.Transform(work)
    pts, paved = register.open_ground(work, spacing=0.5)
    sx, sy = t.to_scene(pts[:, 0], pts[:, 1])
    target = ref.elevation(sx, sy)
    A = t.height_terms(pts[:, 0], pts[:, 1], pts[:, 2])

    # Hold out by 60 m block, not by point: neighbouring points share the same local error.
    block = (np.floor(sx / 60).astype(int) * 7919 + np.floor(sy / 60).astype(int)) % 2 == 0
    labels = ["offset", "z scale", "tilt", "tilt", "dome", "dome", "dome", "cubic", "cubic", "cubic", "cubic"]
    print(f"{len(pts)} open-ground points, {block.sum()} used to fit and {(~block).sum()} held out")
    for n_terms, name in [(4, "offset, z scale, tilt"), (7, "plus dome (quadratic)"), (11, "plus cubic terms")]:
        keep = block.copy()
        for _ in range(3):
            c, *_ = np.linalg.lstsq(A[keep][:, :n_terms], target[keep], rcond=None)
            r = A[:, :n_terms] @ c - target
            keep = block & (np.abs(r) < max(0.25, 3 * np.median(np.abs(r[block]))))
        out = ~block
        print(f"  {name:26s} fitted: median |error| {np.median(np.abs(r[block])):.3f} m   held out: median {np.median(np.abs(r[out])):.3f} m,"
              f" 95% within {np.percentile(np.abs(r[out]), 95):.2f} m, pavement median {np.median(np.abs(r[out & paved])):.3f} m")
        coef = np.zeros(A.shape[1])
        coef[:n_terms] = c
        last = r

    json.dump({"coefficients": coef.tolist(), "terms": labels, "held_out_median_abs_error_m": float(np.median(np.abs(last[~block]))),
               "held_out_p95_abs_error_m": float(np.percentile(np.abs(last[~block]), 95))},
              open(os.path.join(work, "height_model.json"), "w"), indent=1)

    print("\n  leftover on held-out points by 120 m block (m), north at the top:")
    for y1 in np.arange(sy.max(), sy.min(), -120):
        row = []
        for x0 in np.arange(sx.min(), sx.max(), 120):
            m = (sx >= x0) & (sx < x0 + 120) & (sy <= y1) & (sy > y1 - 120) & ~block
            row.append(f"{np.median(last[m]):+6.2f}" if m.sum() > 150 else "   .  ")
        print("   ", " ".join(row))


if __name__ == "__main__":
    main()
