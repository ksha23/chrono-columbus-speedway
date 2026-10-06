#!/usr/bin/env python3
"""Measure the scan's scale and rotation against USGS aerial imagery, window by window.

    python -I georef_local.py WORK_DIR REFERENCE_DIR

Each 120 m window of the scan that holds pavement is matched to the aerial picture on its own,
searching 40 m either way around the whole-site match georef.py found. A similarity transform (scale, rotation, shift)
is then fitted through the window matches. A scan that is true to scale gives scale 1.000.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, signal

Image.MAX_IMAGE_PIXELS = None
REF_E0, REF_N1 = 328800.0, 4794900.0
RES = 0.5


def edges(rgb):
    g = rgb.astype(np.float32).mean(-1)
    g = ndimage.gaussian_filter(g, 1.0)
    return np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))


def main():
    work, ref_dir = sys.argv[1], sys.argv[2]
    meta = json.load(open(os.path.join(work, "ortho_meta.json")))
    first = json.load(open(os.path.join(work, "georef.json")))   # the whole-site match from georef.py
    GUESS = (first["origin_easting"], first["origin_northing"])
    step = int(round(RES / meta["res"]))
    ortho = Image.open(os.path.join(work, "ortho.png"))
    small = np.asarray(ortho.resize((ortho.width // step, ortho.height // step), Image.BOX))
    covered = np.isfinite(np.load(os.path.join(work, "dsm.npy")))[: small.shape[0] * step : step, : small.shape[1] * step : step]
    rgb = small.astype(np.float32) / 255
    mx, mn = rgb.max(-1), rgb.min(-1)
    paved = (mx > 0.55) & ((mx - mn) / np.maximum(mx, 1e-3) < 0.15) & covered
    mov = edges(small) * ndimage.binary_erosion(covered, iterations=4)
    ref = edges(np.asarray(Image.open(os.path.join(ref_dir, "naip_utm.png")).convert("RGB")))

    win, search = int(120 / RES), int(40 / RES)
    src, dst = [], []
    H, W = mov.shape
    for r in range(0, H - win, win // 2):
        for c in range(0, W - win, win // 2):
            if paved[r:r + win, c:c + win].mean() < 0.12:
                continue
            patch = mov[r:r + win, c:c + win]
            # Where this window's corner should be in the reference under the first guess.
            x_w = meta["x0"] + c * RES
            y_w = meta["y1"] - r * RES
            rc = int(round((REF_N1 - (y_w + GUESS[1])) / RES))
            cc = int(round((x_w + GUESS[0] - REF_E0) / RES))
            area = ref[rc - search: rc + win + search, cc - search: cc + win + search]
            if area.shape != (win + 2 * search, win + 2 * search):
                continue
            a, b = area - area.mean(), patch - patch.mean()
            corr = signal.fftconvolve(a, b[::-1, ::-1], mode="valid")
            energy = signal.fftconvolve(a * a, np.ones_like(b), mode="valid")
            ncc = corr / np.sqrt(np.maximum(energy, 1e-6) * (b * b).sum())
            pr, pc = np.unravel_index(np.argmax(ncc), ncc.shape)
            peak = ncc[pr, pc]
            # Reject a peak on the edge of the search area or one that barely stands out.
            second = np.partition(ncc.ravel(), -200)[-200]
            if pr in (0, 2 * search) or pc in (0, 2 * search) or peak < 0.25:
                continue
            centre_scan = (x_w + win * RES / 2, y_w - win * RES / 2)
            e = REF_E0 + (cc - search + pc) * RES + win * RES / 2
            n = REF_N1 - (rc - search + pr) * RES - win * RES / 2
            src.append(centre_scan)
            dst.append((e, n))
            print(f"  window at ({centre_scan[0]:7.1f}, {centre_scan[1]:7.1f})  shift vs guess ({(pc - search) * RES:+6.1f}, {-(pr - search) * RES:+6.1f}) m   ncc {peak:.2f} (200th best {second:.2f})")

    src, dst = np.array(src), np.array(dst)
    # Similarity fit: dst = s R(theta) src + t, solved linearly as [a -b; b a].
    A = np.zeros((2 * len(src), 4))
    A[0::2] = np.stack([src[:, 0], -src[:, 1], np.ones(len(src)), np.zeros(len(src))], 1)
    A[1::2] = np.stack([src[:, 1], src[:, 0], np.zeros(len(src)), np.ones(len(src))], 1)
    b = dst.reshape(-1)
    keep = np.ones(len(src), bool)
    for _ in range(4):  # drop windows that matched the wrong thing
        rows = np.repeat(keep, 2)
        sol, *_ = np.linalg.lstsq(A[rows], b[rows], rcond=None)
        resid = np.hypot(*(A @ sol - b).reshape(-1, 2).T)
        keep = resid < max(3.0, 2.5 * np.median(resid[keep]))
    a_, b_, te, tn = sol
    scale, rot = float(np.hypot(a_, b_)), float(np.degrees(np.arctan2(b_, a_)))
    print(f"\n{keep.sum()} of {len(src)} windows kept")
    print(f"scale {scale:.4f}   rotation {rot:+.3f} deg   origin E {te:.2f} N {tn:.2f}")
    print(f"residual of kept windows: median {np.median(resid[keep]):.2f} m, worst {resid[keep].max():.2f} m")
    # The same data fitted with a shift only, to show what the scale term buys.
    shift = (dst - src)[keep].mean(0)
    r0 = np.hypot(*((dst - src)[keep] - shift).T)
    print(f"shift-only fit: origin E {shift[0]:.2f} N {shift[1]:.2f}, residual median {np.median(r0):.2f} m, worst {r0.max():.2f} m")
    np.savez(os.path.join(work, "imagery_matches.npz"), src=src[keep], dst=dst[keep])
    json.dump({"scale": scale, "rotation_deg": rot, "east": float(te), "north": float(tn),
               "windows": int(keep.sum())}, open(os.path.join(work, "georef_local.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
