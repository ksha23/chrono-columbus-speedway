"""The ground photo, from the raw redraw of the scan to what gets cut into texture tiles.

In order: take out the photographed shadows, the wide ones and then the thin ones, paint over
everything that stood on the ground (tree crowns, roofs, parked vehicles, cones) and the holes
the scan left under trees, then fill beyond the scan's edge from aerial imagery.
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
import thinshadows


def open_ground(work, raster, photo):
    """Low-resolution masks of what the scan shows: where things stand and what the ground is."""
    obj = np.load(os.path.join(work, "objects.npz"))
    height = obj["height"]
    valid = np.isfinite(height)
    cell = float(obj["cell"])
    k = int(round(cell / raster["res"]))
    h, w = height.shape
    small = np.stack([np.asarray(photo[..., ch], dtype=np.float32)[:h * k, :w * k].reshape(h, k, w, k).mean((1, 3)) for ch in range(3)], -1)

    # A parked car is masked wider than a tree: its own shadow lies right against it, and the
    # scan gives its bonnet and boot too little height to be found on their own.
    standing = (ndimage.binary_dilation(obj["trees"], iterations=int(round(0.75 / cell)))
                | ndimage.binary_dilation(obj["buildings"], iterations=int(round(1.0 / cell)))
                | ndimage.binary_dilation(obj["vehicles"], iterations=int(round(1.5 / cell))))
    # The scan has holes where a canopy was too dense to reconstruct. What is under them is
    # ground like any other, so they are painted in with the rest. Inlets from the scan's
    # outer edge up to 6 m wide count as holes too.
    inside = ndimage.binary_fill_holes(ndimage.binary_closing(valid, iterations=int(round(3.0 / cell))))
    holes = inside & ~ndimage.binary_erosion(valid, iterations=1)
    # Keep a margin from anything standing and from the scan's edge when choosing colour sources.
    known = valid & ~ndimage.binary_dilation(standing, iterations=int(round(0.5 / cell)))
    known &= ndimage.binary_erosion(valid, iterations=int(round(3.0 / cell)))
    # A wider margin for the ground whose colour is carried under trees: right beside a crown
    # the photo shows its fringe and the rough grass a mower cannot reach.
    source = known & ~ndimage.binary_dilation(standing, iterations=int(round(2.0 / cell)))

    rgb = small / 255
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    low = np.nan_to_num(height, nan=9.0) < 0.2
    grass = known & low & (rgb[..., 1] > 0.92 * rgb[..., 0]) & (rgb[..., 1] > 1.15 * rgb[..., 2]) & (sat > 0.25) & (mx > 0.3)
    sunlit_paved = known & low & (sat < 0.14) & (mx > 0.5)
    # Anything grey and bright is concrete or gravel, whether or not it is part of the road
    # network: a footpath, a pad under a tank.
    grey = known & (sat < 0.20) & (mx > 0.45)
    return {"cell": cell, "standing": standing & valid, "holes": holes, "inside": inside, "known": known, "source": source,
            "grass": grass, "sunlit_paved": sunlit_paved, "grey": grey, "valid": valid}


def thin_shadows(photo, raster, g, paved, wet, grains, log):
    """Paint out the shadows of poles and posts. photo is changed in place."""
    cell = g["cell"]
    shape = paved.shape
    open_cells = g["known"] & ~wet
    # Stay a step off the pavement's edge on both sides: the photo's own edge is a dark line.
    rim = ndimage.binary_dilation(paved, iterations=2) & ~ndimage.binary_erosion(paved, iterations=2)
    kinds = {"paved": paved & pavement.to_fine(open_cells, shape, cell) & ~rim,
             "land": ~paved & pavement.to_fine(open_cells, shape, cell) & ~rim}
    found = thinshadows.find(photo, raster, kinds["paved"], kinds["land"])
    k = int(round(pavement.RES / raster["res"]))
    for name, grain in (("land", grains[0]), ("paved", grains[1])):
        full = np.zeros(photo.shape[:2], bool)
        big = np.repeat(np.repeat(found[name], k, axis=0), k, axis=1)
        full[:big.shape[0], :big.shape[1]] = big[:full.shape[0], :full.shape[1]]
        lines = fill.repaint_small(photo, raster, full, grain, margin=0.10)
        log(f"  thin shadows on {'pavement' if name == 'paved' else 'other ground'}: {lines} lines, {found[name].sum() * pavement.RES ** 2:.0f} m2")


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
    hidden = ndimage.binary_dilation(obj["trees"], iterations=int(round(0.75 / cell))) | g["holes"]
    distance, paved = pavement.build(photo, raster, g, hidden, obj["buildings"], matte, log)
    g["paved"] = pavement.to_cells(paved, g["valid"].shape, cell)
    g["grass"] &= ~g["paved"]
    grains = (fill.find_grain(photo, raster, g["grass"], cell, "grass"), fill.find_grain(photo, raster, g["sunlit_paved"], cell, "pavement"))
    if shadowless:
        # The pond, as the lidar outlines it and a little beyond: its banks have moved since.
        gy, gx = np.mgrid[0:g["valid"].shape[0], 0:g["valid"].shape[1]]
        wet = objects.water(ref, raster["x0"] + (gx + 0.5) * cell, raster["y1"] - (gy + 0.5) * cell)
        wet = ndimage.binary_dilation(wet, iterations=int(round(1.5 / cell)))
        photo = fill.even_out(photo, raster, matte, g["known"], g["paved"] | g["grey"], wet, cell, paved, grains, before=raw)
        log(f"  shadows relit and levelled over {(np.asarray(matte[::4, ::4]) > 64).mean() * photo.shape[0] * photo.shape[1] * raster['res'] ** 2:.0f} m2")
        thin_shadows(photo, raster, g, paved, wet, grains, log)

    log(f"  repainting {(g['standing'] | g['holes']).sum() * cell * cell:.0f} m2 of ground under trees, roofs and vehicles and in the scan's holes")
    painted = fill.repaint(photo, raster, g["standing"] | g["holes"], g["source"], g["paved"], grains, cell, solid=g["holes"])

    # Cones are found by their own shadows, so they are looked for in the photo as flown.
    found, patches = cones.find(raw, raster, g["paved"], g["standing"] & ~hidden, cell)
    fill.repaint_small(painted, raster, patches, grains[1])
    sizes = sorted({c["height"] for c in found})
    log(f"  painted out {len(found)} cones: " + ", ".join(f"{sum(c['height'] == h for c in found)} of {h:g} m" for h in sizes))

    # Holes are part of the scan's own ground now. Aerial imagery starts at its outer edge only.
    k = int(round(cell / raster["res"]))
    whole = covered.copy()
    big = np.repeat(np.repeat(g["inside"], k, axis=0), k, axis=1)
    whole[:big.shape[0], :big.shape[1]] |= big[:whole.shape[0], :whole.shape[1]]
    matched, filled = compose.surround_and_blend(painted, whole, raster, ref.aerial())
    return filled, matched, found, g, pavement.Edge(distance, raster)
