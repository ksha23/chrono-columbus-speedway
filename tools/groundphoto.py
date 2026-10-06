"""The ground photo, from the raw redraw of the scan to what gets cut into texture tiles.

In order: take out the photographed shadows, the wide ones and then the thin ones, paint over
everything that stood on the ground (tree crowns, roofs, parked vehicles, cones) and the holes
the scan left under trees, then fill beyond the scan's edge from aerial imagery.
"""
import json
import os

import numpy as np
from scipy import ndimage

import barriers
import blocks
import compose
import cones
import edgelines
import fill
import markings
import markings_regular
import markings_stalls
import markings_tidy
import markmodel
import objects
import outline
import pavement
import poles
import railmodel
import roads
import thinshadows
import vehicles


ROUGH = 13.0       # grey levels: how much ground varies over a metre before it stops being even


def open_ground(work, raster, photo):
    """Low-resolution masks of what the scan shows: where things stand and what the ground is."""
    obj = np.load(os.path.join(work, "objects.npz"))
    height = obj["height"]
    valid = np.isfinite(height)
    cell = float(obj["cell"])
    k = int(round(cell / raster["res"]))
    h, w = height.shape
    small = np.stack([np.asarray(photo[..., ch], dtype=np.float32)[:h * k, :w * k].reshape(h, k, w, k).mean((1, 3)) for ch in range(3)], -1)

    # Trees and buildings. Smaller things (cars, cones, poles, rails) are found one by one and
    # painted out at full detail, each with its own shadow.
    standing = (ndimage.binary_dilation(obj["trees"], iterations=int(round(0.75 / cell)))
                | ndimage.binary_dilation(obj["buildings"], iterations=int(round(1.0 / cell))))
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
    # Even ground: mown grass and concrete are, leaves and long weeds are not, whatever their colour.
    lum = small @ np.array([0.30, 0.59, 0.11], np.float32)
    mean = ndimage.uniform_filter(lum, 5)
    smooth = np.sqrt(np.maximum(ndimage.uniform_filter(lum * lum, 5) - mean * mean, 0.0)) < ROUGH
    return {"cell": cell, "standing": standing & valid, "holes": holes, "inside": inside, "known": known, "source": source,
            "grass": grass, "sunlit_paved": sunlit_paved, "grey": grey, "valid": valid, "smooth": smooth}


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


def paint_view(raw, matte, raster, paved):
    """What markings_tidy.carry asks of the photo about the points along a gap in a line.

    Returns view(points) -> (on or beside the pavement, bare pavement in plain view). Beside
    means within four metres: a crown painted flat on the road takes a bite that deep out of
    the pavement's map. Bare means that the half metre around the point is sunlit and all
    the colour of concrete, with no leaf, shadow or paint in it.
    """
    res = raster["res"]
    reach = ndimage.distance_transform_edt(~paved) * pavement.RES <= 4.0
    half = int(round(0.5 / res))

    def view(points):
        points = np.asarray(points, float)
        rows = np.clip(((raster["y1"] - points[:, 1]) / res).astype(int), half, raw.shape[0] - half - 1)
        cols = np.clip(((points[:, 0] - raster["x0"]) / res).astype(int), half, raw.shape[1] - half - 1)
        k = int(round(pavement.RES / res))
        road = reach[np.minimum(rows // k, reach.shape[0] - 1), np.minimum(cols // k, reach.shape[1] - 1)]
        bare = np.zeros(len(points), bool)
        for n, (r, c) in enumerate(zip(rows, cols)):
            rgb = np.asarray(raw[r - half:r + half + 1, c - half:c + half + 1, :3], dtype=np.float32) / 255
            mx, mn = rgb.max(-1), rgb.min(-1)
            plain = ((mx - mn) < 0.2 * np.maximum(mx, 1e-3)) & (mx > 0.45)
            if matte is not None:
                plain &= np.asarray(matte[r - half:r + half + 1, c - half:c + half + 1]) < 32
            bare[n] = plain.mean() > 0.9
        return road, bare

    return view


def _paint_out(photo, raster, left, on_road, grains, taken):
    """Paint a footprint out of the photo, as grass or as pavement by where it lies, and add it
    to the mask of things accounted for."""
    fill.repaint_small(photo, raster, left & ~on_road, grains[0], margin=0.1)
    fill.repaint_small(photo, raster, left & on_road, grains[1], margin=0.1)
    taken |= left


def prepare(work, ref, raster, deshadow=True, without=(), log=print):
    """Return (ground photo over the whole raster, aerial picture colour-matched to it, things, masks, edge).

    things is what was found standing on the ground and painted out, to be put back as objects:
    {"cones": [...], "poles": [...], "vehicles": [...], "blocks": [...], "barriers": [...], "markings": [...], "edge_lines": [...]}. masks is what open_ground found, with "paved" replaced by the finished pavement map, for
    the callers that plant things on that ground. edge is the pavement.Edge the mesh is cut on.
    without names kinds of thing to leave in the photo and not look for.
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
    np.save(os.path.join(work, "pavement.npy"), paved)     # barriers.py and audit_ground.py read it
    grains = (fill.find_grain(photo, raster, g["grass"], cell, "grass"), fill.find_grain(photo, raster, g["sunlit_paved"], cell, "pavement"))
    # The pavement map at the photo's own pixel size, for the steps that paint at full detail.
    k_fine = int(round(pavement.RES / raster["res"]))
    k_cell = int(round(cell / raster["res"]))

    def at_full_size(mask):
        out = np.zeros(raw.shape[:2], bool)
        big = np.repeat(np.repeat(mask, k_fine, axis=0), k_fine, axis=1)
        out[:big.shape[0], :big.shape[1]] = big[:out.shape[0], :out.shape[1]]
        return out

    on_road = at_full_size(paved)
    things = {"poles": [], "vehicles": [], "blocks": [], "barriers": [], "markings": [], "edge_lines": []}
    taken = np.zeros(raw.shape[:2], bool)      # everything found standing, as it lies in the photo
    if shadowless:
        # The pond, as the lidar outlines it and a little beyond: its banks have moved since.
        gy, gx = np.mgrid[0:g["valid"].shape[0], 0:g["valid"].shape[1]]
        wet = objects.water(ref, raster["x0"] + (gx + 0.5) * cell, raster["y1"] - (gy + 0.5) * cell)
        wet = ndimage.binary_dilation(wet, iterations=int(round(1.5 / cell)))
        # Cells that are the road's own concrete and nothing else: well inside the pavement.
        # Not its edge, which is part grass, and not the grey of gravel and bare earth.
        plain = ndimage.binary_erosion(g["paved"], iterations=int(round(0.75 / cell)))
        photo = fill.even_out(photo, raster, matte, g["known"], g["paved"] | g["grey"], wet, cell, paved, grains, before=raw, plain_cells=plain)
        log(f"  shadows relit and levelled over {(np.asarray(matte[::4, ::4]) > 64).mean() * photo.shape[0] * photo.shape[1] * raster['res'] ** 2:.0f} m2")
        thin_shadows(photo, raster, g, paved, wet, grains, log)
        # Light poles: found in the photo as flown, by their shadows, then painted out whole.
        if "poles" not in without:
            things["poles"] = poles.locate(raw, raster, paved, obj["height"], obj["trees"], obj["buildings"], cell, work)
        left = poles.footprint(things["poles"], raster, photo.shape[:2])
        _paint_out(photo, raster, left, on_road, grains, taken)
        log(f"  painted out {len(things['poles'])} light poles with their shadows")
        # Parked vehicles, from the scan's heights and the photo as flown.
        hh, ww = obj["height"].shape
        small_raw = np.stack([np.asarray(raw[..., ch], dtype=np.float32)[:hh * k_cell, :ww * k_cell].reshape(hh, k_cell, ww, k_cell).mean((1, 3)) for ch in range(3)], -1)
        if "vehicles" not in without:
            things["vehicles"] = vehicles.find(obj["height"], small_raw, g["paved"], obj["buildings"], cell, raster["x0"], raster["y1"])
        left = vehicles.footprint(things["vehicles"], raster, photo.shape[:2], raw, poles.suns(work))
        _paint_out(photo, raster, left, on_road, grains, taken)
        log(f"  painted out {len(things['vehicles'])} parked vehicles with their shadows")
        # Tanks, huts, bleachers: small man-made things, given back as blocks.
        if "blocks" not in without:
            things["blocks"] = blocks.find(obj["height"], small_raw, obj["buildings"], things["vehicles"], cell, raster["x0"], raster["y1"])
        left = vehicles.footprint(things["blocks"], raster, photo.shape[:2], raw, poles.suns(work))
        _paint_out(photo, raster, left, on_road, grains, taken)
        log(f"  painted out {len(things['blocks'])} small structures with their shadows")
        # Guard rails and the fence round the property.
        if "barriers" not in without:
            things["barriers"] = barriers.find(work)
        # A guard rail stands on the road's edge: pavement guessed at beyond one is taken back.
        kept = roads.behind_rails(paved, raster, [b for b in things["barriers"] if b["type"] == "guardrail"], pavement.to_fine(hidden, paved.shape, cell))
        if (kept != paved).any():
            log(f"  {(paved & ~kept).sum() * pavement.RES ** 2:.0f} m2 of pavement guessed at beyond a guard rail taken back")
            paved = kept
            distance = ((ndimage.distance_transform_edt(paved) - ndimage.distance_transform_edt(~paved)) * pavement.RES).astype(np.float32)
            g["paved"] = pavement.to_cells(paved, g["valid"].shape, cell)
            on_road = at_full_size(paved)
            np.save(os.path.join(work, "pavement.npy"), paved)
        left = railmodel.footprint(things["barriers"], raster, photo.shape[:2], raw, poles.suns(work))
        _paint_out(photo, raster, left, on_road, grains, taken)
        metres = {kind: sum(railmodel.length_of(run) for b in things["barriers"] if b["type"] == kind for run in railmodel.seen_runs(b)) for kind in ("guardrail", "fence")}
        log(f"  painted out {metres['guardrail']:.0f} m of guard rail and the {metres['fence']:.0f} m of fence that show in the photo")

    log(f"  repainting {(g['standing'] | g['holes']).sum() * cell * cell:.0f} m2 of ground under trees, roofs and vehicles and in the scan's holes")
    painted = fill.repaint(photo, raster, g["standing"] | g["holes"], g["source"], g["paved"], grains, cell, solid=g["holes"])

    # Cones are found by their own shadows, so they are looked for in the photo as flown.
    found, patches = cones.find(raw, raster, g["paved"], g["standing"] & ~hidden, cell)
    fill.repaint_small(painted, raster, patches, grains[1])
    sizes = sorted({c["height"] for c in found})
    log(f"  painted out {len(found)} cones: " + ", ".join(f"{sum(c['height'] == h for c in found)} of {h:g} m" for h in sizes))
    things["cones"] = found
    taken |= ndimage.binary_dilation(patches, iterations=int(round(0.2 / raster["res"])))

    # Road paint: found as strokes, taken out of the photo, and drawn as geometry instead.
    if "markings" not in without:
        def paint_can_be_at(x, y):
            r, c = int((raster["y1"] - y) / raster["res"]), int((x - raster["x0"]) / raster["res"])
            return bool(on_road[r, c]) and not bool(taken[r, c])

        found_paint = markings.find(raw, raster, paved, taken, obj["height"], cell, matte)
        log(f"  {len(found_paint)} strokes of road paint traced")
        # What is painted out is the paint as traced. What is drawn is the paint as fitted.
        traced = markmodel.footprint(found_paint, raster, painted.shape[:2]) & on_road
        beside = ndimage.binary_dilation(obj["trees"], iterations=int(round(2.0 / cell))) | g["holes"]

        def shaded(points):
            """Which points lie in a relit shadow or within two metres of a crown."""
            points = np.asarray(points, float)
            r = np.clip(((raster["y1"] - points[:, 1]) / raster["res"]).astype(int), 0, raw.shape[0] - 1)
            c = np.clip(((points[:, 0] - raster["x0"]) / raster["res"]).astype(int), 0, raw.shape[1] - 1)
            dark = np.array([matte[i, j] for i, j in zip(r, c)]) > 32 if matte is not None else np.zeros(len(points), bool)
            return dark | beside[np.minimum(r // k_cell, beside.shape[0] - 1), np.minimum(c // k_cell, beside.shape[1] - 1)]

        fitted = markings_tidy.tidy(found_paint, log, shaded, paint_view(raw, matte, raster, paved))
        fitted = markings_stalls.regularise(fitted, lambda x, y: bool(on_road[int((raster["y1"] - y) / raster["res"]), int((x - raster["x0"]) / raster["res"])]),
                                            markings_stalls.evidence(raw, raster, taken), log)[0]
        things["markings"], put_back = markings_regular.fill(fitted, paint_can_be_at)
        log(f"  {put_back} dashes put back into runs they were missing from")
        fill.repaint_tiled(painted, raster, traced, grains[1])
        metres = {colour: sum(railmodel.length_of(m["points"]) for m in things["markings"] if m["colour"] == colour) for colour in ("yellow", "white")}
        log(f"  painted out {len(things['markings'])} strokes of road paint: {metres['yellow']:.0f} m yellow, {metres['white']:.0f} m white")

    # Holes are part of the scan's own ground now. Aerial imagery starts at its outer edge only.
    k = int(round(cell / raster["res"]))
    whole = covered.copy()
    big = np.repeat(np.repeat(g["inside"], k, axis=0), k, axis=1)
    whole[:big.shape[0], :big.shape[1]] |= big[:whole.shape[0], :whole.shape[1]]
    # The roads' outline, redrawn at each road's own width about its centre line, and then
    # with the notches and bumps that are left taken out of it. Where that moves the outline
    # the photo has the wrong ground on it, so that ground is painted over to match. One case
    # is left as photographed: pavement in plain view that a smoother outline now leaves
    # outside the road, such as the flare at a footpath's mouth.
    traced = g["paved"]
    unseen = pavement.to_fine(hidden, paved.shape, cell)
    shade = np.zeros_like(paved) if matte is None else (np.asarray(matte[::k_fine, ::k_fine]) > 64)[:paved.shape[0], :paved.shape[1]]
    if things["markings"]:
        distance, paved = roads.straighten(distance, raster, things["markings"], log, unseen | shade)
    modelled = pavement.to_cells(paved, g["valid"].shape, cell)
    plain = pavement.to_fine(g["smooth"] & g["known"], paved.shape, cell) & ~shade
    distance, paved = outline.unkink(distance, raster, log, unseen, plain)
    g["paved"] = pavement.to_cells(paved, g["valid"].shape, cell)
    # What a crown hid has its fringe beside it: flecks of leaf and sun on the pavement
    # round a bite that has just been filled. A metre and a half round it is repainted too.
    gained = g["paved"] & ~traced
    fringe = ndimage.binary_dilation(gained, iterations=int(round(1.5 / cell))) & g["paved"]
    changed = (modelled ^ traced) | gained | fringe | (modelled & ~g["paved"] & (g["standing"] | g["holes"]))
    changed = ndimage.binary_dilation(changed, iterations=1) & g["inside"]
    if changed.any():
        painted = fill.repaint(painted, raster, changed, g["source"] & ~changed, g["paved"], grains, cell, solid=changed)
    np.save(os.path.join(work, "pavement.npy"), paved)

    # Edge lines: not painted on the real track, drawn along every road as an option.
    if "edgelines" not in without:
        scanned = pavement.to_fine(ndimage.binary_erosion(g["valid"], iterations=int(round(3.0 / cell))), paved.shape, cell)
        things["edge_lines"] = edgelines.find(distance, raster, scanned, [b for b in things["barriers"] if b["type"] == "guardrail"], log)

    matched, filled = compose.surround_and_blend(painted, whole, raster, ref.aerial())
    return filled, matched, things, g, pavement.Edge(distance, raster)
