"""The ground photo, from the raw redraw of the scan to what gets cut into texture tiles.

In order: take out the photographed shadows, paint over everything that stood on the ground
(tree crowns, roofs, parked vehicles), then fill beyond the scan's edge from aerial imagery.
"""
import json
import os

import numpy as np
from scipy import ndimage

import compose
import cones
import fill
import objects
import pavement


def open_ground(work, raster, photo):
    """Low-resolution masks of what the scan shows: where things stand and what the ground is."""
    obj = np.load(os.path.join(work, "objects.npz"))
    height = obj["height"]
    valid = np.isfinite(height)
    cell = float(obj["cell"])
    k = int(round(cell / raster["res"]))
    h, w = height.shape
    small = np.stack([np.asarray(photo[..., ch], dtype=np.float32)[:h * k, :w * k].reshape(h, k, w, k).mean((1, 3)) for ch in range(3)], -1)

    standing = (ndimage.binary_dilation(obj["trees"], iterations=int(round(0.75 / cell)))
                | ndimage.binary_dilation(obj["buildings"], iterations=int(round(1.0 / cell)))
                | ndimage.binary_dilation(obj["vehicles"], iterations=int(round(0.5 / cell))))
    # Keep a margin from anything standing and from the scan's edge when choosing colour sources.
    known = valid & ~ndimage.binary_dilation(standing, iterations=int(round(0.5 / cell)))
    known &= ndimage.binary_erosion(valid, iterations=int(round(3.0 / cell)))

    rgb = small / 255
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    low = np.nan_to_num(height, nan=9.0) < 0.2
    grass = known & low & (rgb[..., 1] > 0.92 * rgb[..., 0]) & (rgb[..., 1] > 1.15 * rgb[..., 2]) & (sat > 0.25) & (mx > 0.3)
    sunlit_paved = known & low & (sat < 0.14) & (mx > 0.5)
    return {"cell": cell, "standing": standing & valid, "known": known, "grass": grass, "sunlit_paved": sunlit_paved, "valid": valid}


def prepare(work, ref, raster, deshadow=True, log=print):
    """Return (ground photo over the whole raster, aerial picture colour-matched to it, cones, masks, edge).

    masks is what open_ground found, with "paved" replaced by the finished pavement map, for
    the callers that plant things on that ground. edge is the pavement.Edge the mesh is cut on.
    """
    raw = np.load(os.path.join(work, "photo.npy"), mmap_mode="r")
    covered = np.isfinite(np.load(os.path.join(work, "surface.npy"), mmap_mode="r"))
    # What shadows.py leaves when run as: shadows.py WORK_DIR WORK_DIR/shadows --full
    relit, matte = os.path.join(work, "shadows", "photo_noshadow.npy"), os.path.join(work, "shadows", "photo_shadow.npy")
    shadowless = deshadow and os.path.isfile(relit) and os.path.isfile(matte)
    matte = np.load(matte, mmap_mode="r") if shadowless else None
    photo = np.load(relit, mmap_mode="r") if shadowless else raw
    if deshadow and not shadowless:
        log("  shadows.py has not been run on this work directory: shadows are still in the photo")

    g = open_ground(work, raster, photo)
    cell = g["cell"]
    obj = np.load(os.path.join(work, "objects.npz"))
    hidden = ndimage.binary_dilation(obj["trees"], iterations=int(round(0.75 / cell)))
    distance, paved = pavement.build(photo, raster, g, hidden, obj["buildings"], matte, log)
    g["paved"] = pavement.to_cells(paved, g["valid"].shape, cell)
    g["grass"] &= ~g["paved"]
    if shadowless:
        # The pond, as the lidar outlines it and a little beyond: its banks have moved since.
        gy, gx = np.mgrid[0:g["valid"].shape[0], 0:g["valid"].shape[1]]
        wet = objects.water(ref, raster["x0"] + (gx + 0.5) * cell, raster["y1"] - (gy + 0.5) * cell)
        wet = ndimage.binary_dilation(wet, iterations=int(round(1.5 / cell)))
        photo = fill.even_out(photo, raster, matte, g["known"], g["paved"], wet, cell, paved)
        log(f"  shadows relit and levelled over {(np.asarray(matte[::4, ::4]) > 64).mean() * photo.shape[0] * photo.shape[1] * raster['res'] ** 2:.0f} m2")

    grains = (fill.find_grain(photo, raster, g["grass"], cell, "grass"), fill.find_grain(photo, raster, g["sunlit_paved"], cell, "pavement"))
    log(f"  repainting {g['standing'].sum() * cell * cell:.0f} m2 of ground under trees, roofs and vehicles")
    painted = fill.repaint(photo, raster, g["standing"], g["known"], g["paved"], grains, cell)

    # Cones are found by their own shadows, so they are looked for in the photo as flown.
    found, patches = cones.find(raw, raster, g["paved"], g["standing"] & ~hidden, cell)
    fill.repaint_small(painted, raster, patches, grains[1])
    sizes = sorted({c["height"] for c in found})
    log(f"  painted out {len(found)} cones: " + ", ".join(f"{sum(c['height'] == h for c in found)} of {h:g} m" for h in sizes))

    matched, filled = compose.surround_and_blend(painted, covered, raster, ref.aerial())
    return filled, matched, found, g, pavement.Edge(distance, raster)
