#!/usr/bin/env python3
"""Assemble the published scene from the corrected scan and the USGS reference data.

    python -I build_scene.py WORK_DIR REFERENCE_DIR SCENE_DIR [--levels low,standard]

Reads what rasterize.py left in WORK_DIR and writes into SCENE_DIR:

    speedway_ground.obj     the collision surface, one welded mesh
    speedway_road.obj       the pavement's part of it
    speedway_terrain.obj    the rest of it
    ground/*.obj            the same triangles split by texture tile, for drawing
    textures/<level>/*.jpg  the ground photo, one image per tile, per detail level
    textures/surround.jpg   aerial imagery for the land around the scan
    speedway_scene.json     the manifest
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks  # noqa: E402
import buildings  # noqa: E402
import carmodels  # noqa: E402
import cone_model  # noqa: E402
import cones  # noqa: E402
import forest  # noqa: E402
import ground  # noqa: E402
import groundphoto  # noqa: E402
import landmarks  # noqa: E402
import objects  # noqa: E402
import polemodel  # noqa: E402
import poles  # noqa: E402
import railmodel  # noqa: E402
import layout  # noqa: E402
import markmodel  # noqa: E402
import reference  # noqa: E402
import renderer  # noqa: E402
import rockmodel  # noqa: E402
import tiles  # noqa: E402
import transform  # noqa: E402
import treegen  # noqa: E402
import vehicles  # noqa: E402
import wall_colour  # noqa: E402

# The ground as one collision mesh, and the same triangles as two, for those who want the
# pavement and the rest apart.
COLLISION = {"ground": "speedway_ground.obj", "road": "speedway_road.obj", "terrain": "speedway_terrain.obj"}
MANIFEST = "speedway_scene.json"


PARKED = ("sedan", "hatchback", "suv")   # the models parked cars are drawn from. Chrono's van is a 1970s microbus
TALL_VEHICLE = 1.8    # metres: models at least this tall are the SUVs, vans and pickups
TALL_SEEN = 1.6       # and vehicles the scan measured at least this tall get one of them


def chrono_data(args):
    """Chrono's data directory: as given, or wherever the installed PyChrono keeps it."""
    if args.chrono_data is None:
        import pychrono
        args.chrono_data = pychrono.GetChronoDataPath()
    return args.chrono_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("reference")
    parser.add_argument("scene")
    parser.add_argument("--levels", default="low,standard")
    parser.add_argument("--scan", default=None, help="the scan directory, needed to read wall colours")
    parser.add_argument("--chrono-data", default=None, help="Chrono's data directory, for the cone model (default: ask the installed PyChrono)")
    parser.add_argument("--save-photo", action="store_true", help="also write the finished ground photo and the pavement map to WORK_DIR, for audit_ground.py")
    parser.add_argument("--without", default="", help="comma-separated kinds of thing to leave alone, out of poles, vehicles, blocks, barriers, landmarks, rocks, markings, edgelines")
    parser.add_argument("--keep-shadows", action="store_true", help="publish the photo with its shadows still in")
    args = parser.parse_args()
    start = time.perf_counter()

    raster = json.load(open(os.path.join(args.work, "raster.json")))
    ref = reference.Reference(args.reference)
    t = transform.Transform(args.work)
    os.makedirs(args.scene, exist_ok=True)

    def timed(message):
        """What the photo's preparation reports, with the time each step was reached."""
        print(f"[{time.perf_counter() - start:5.1f} s]{message if message.startswith(' ') else ' ' + message}", flush=True)

    filled, matched, things, masks, edge = groundphoto.prepare(args.work, ref, raster, deshadow=not args.keep_shadows, without=set(filter(None, args.without.split(","))),
                                                               log=timed)
    print(f"[{time.perf_counter() - start:5.1f} s] ground photo ready")
    if args.save_photo:
        np.save(os.path.join(args.work, "ground_photo.npy"), filled)

    os.makedirs(os.path.join(args.scene, "textures"), exist_ok=True)
    tiles.save_texture(matched, os.path.join(args.scene, "textures", "surround.jpg"))
    for level in args.levels.split(","):
        size, with_road = tiles.write_tiles(filled, raster, level, os.path.join(args.scene, "textures", level), edge)
        print(f"[{time.perf_counter() - start:5.1f} s] {level}: {len(raster['tiles'])} tiles of {layout.tile_pixels(layout.LEVELS[level])} px,"
              f" {len(with_road)} of them with a road picture too, {size / 1e6:.0f} MB")

    g = ground.Ground(raster["tiles"], ref, edge)
    g.write_collision(os.path.join(args.scene, COLLISION["ground"]), g.faces, "the whole ground")
    g.write_collision(os.path.join(args.scene, COLLISION["road"]), g.road, "the paved surface")
    g.write_collision(os.path.join(args.scene, COLLISION["terrain"]), g.land, "the ground that is not paved")
    parts = g.write_visual(os.path.join(args.scene, "ground"))
    print(f"[{time.perf_counter() - start:5.1f} s] ground: {len(g.vertices)} vertices, {len(g.road)} road triangles, {len(g.land)} land triangles")

    assets, instances, extra_files = [], [], []
    for name in sorted(parts):
        group, tile = parts[name][0], parts[name][1]
        texture = "textures/surround.jpg" if tile is None else f"textures/<level>/{name}.jpg"
        if group == "Road" and tile not in with_road:
            raise SystemExit(f"{name} has road triangles but its tile has no road picture")
        assets.append({"name": name, "parts": [{"name": name, "mesh": f"ground/{name}.obj", "texture": texture,
                                                 "colour": [1.0, 1.0, 1.0], "ks": [0.0, 0.0, 0.0], "ns": 1.0, "roughness_value": 1.0}]})
        instances.append({"asset": len(assets) - 1, "group": group, "name": name, "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})

    # Buildings, refitted from the scan's height above the lidar ground.
    obj = np.load(os.path.join(args.work, "objects.npz"))
    cell = float(obj["cell"])
    gy, gx = np.mgrid[0:obj["height"].shape[0], 0:obj["height"].shape[1]]
    gx, gy = float(obj["x0"]) + (gx + 0.5) * cell, float(obj["y1"]) - (gy + 0.5) * cell
    found = buildings.measure(obj["buildings"], obj["height"], gx, gy, ref)
    walls = wall_colour.sample(args.scan, args.work, found, t) if args.scan else [wall_colour.FALLBACK] * len(found)
    raw = np.load(os.path.join(args.work, "photo.npy"), mmap_mode="r")
    for asset in buildings.write(found, raw, raster, args.scene, walls):
        b = asset.pop("footprint")
        assets.append(asset)
        instances.append({"asset": len(assets) - 1, "group": "Buildings", "name": asset["name"], "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0],
                          "centre": b["centre"], "length": round(2 * b["half_length"], 2), "width": round(2 * b["half_width"], 2),
                          "eave_height": round(b["eave"], 2), "ridge_height": round(b["ridge"], 2)})
        print(f"  {asset['name']}: {2 * b['half_length']:.1f} x {2 * b['half_width']:.1f} m, eave {b['eave']:.1f} m, ridge {b['ridge']:.1f} m,"
              f" walls {tuple(round(c, 2) for c in walls[len(instances) - len(parts) - 1])}")

    # Trees: generated models planted where the scan measured crowns.
    measured = json.load(open(os.path.join(args.work, "trees.json")))
    k = int(round(cell / raster["res"]))
    h, w = obj["height"].shape
    small = np.stack([np.asarray(raw[..., ch], dtype=np.float32)[:h * k, :w * k].reshape(h, k, w, k).mean((1, 3)) for ch in range(3)], -1)
    # A building is drawn as the rectangle fitted to it, which covers more ground than its roof
    # did where the real plan is an L. Trees keep out of the rectangle, with a metre to spare.
    footprints = obj["buildings"].copy()
    for b in found:
        u = np.array(b["axis"])
        along = (gx - b["centre"][0]) * u[0] + (gy - b["centre"][1]) * u[1]
        across = -(gx - b["centre"][0]) * u[1] + (gy - b["centre"][1]) * u[0]
        footprints |= (np.abs(along) < b["half_length"] + 1.0) & (np.abs(across) < b["half_width"] + 1.0)
    library, placements = forest.plan(measured, obj["crowns"], obj["height"], small, objects.water(ref, gx, gy), masks["paved"], footprints,
                                      (float(obj["x0"]), float(obj["y1"]), cell), ref)
    os.makedirs(os.path.join(args.scene, "trees"), exist_ok=True)
    used = {p["model"] for p in placements}
    index, budget = {}, {}
    for name, kind, height, radius, seed, triangles in library:
        if name not in used:
            continue
        tree = treegen.make_tree(kind, height, radius, seed, triangles=triangles)
        for part in ("wood", "leaves"):
            treegen.write_obj(os.path.join(args.scene, "trees", f"{name}_{part}.obj"), *tree[part])
        budget[name] = len(tree["wood"][2]) + len(tree["leaves"][2])
        index[name] = len(assets)
        assets.append({"name": name, "parts": [
            {"name": "wood", "mesh": f"trees/{name}_wood.obj", "colour": [0.3, 0.25, 0.2], "roughness_value": 0.9},
            # Leaves are single triangles, each meant to be seen from both sides.
            {"name": "leaves", "mesh": f"trees/{name}_leaves.obj", "colour": [0.2, 0.33, 0.1], "roughness_value": 0.9, "double_sided": True}]})
    drawn = 0
    for p in placements:
        model = p.pop("model")
        drawn += budget[model]
        instances.append({"asset": index[model], **p})
    kinds = sorted({p["kind"] for p in placements})
    full = sum(not assets[i["asset"]]["name"].endswith("_light") for i in instances if i["group"] == "Trees")
    print(f"[{time.perf_counter() - start:5.1f} s] {len(placements)} trees from {len(index)} models ({full} full, {len(placements) - full} light), {drawn / 1e6:.2f} M triangles: "
          + ", ".join(f"{sum(p['kind'] == kk for p in placements)} {kk}" for kk in kinds))

    # Light poles, one model per height, each with its arm out over the nearest pavement.
    by_height = {}
    for n, p in enumerate(things["poles"]):
        tall = round(p["height"] * 2) / 2
        if tall not in by_height:
            name, _ = polemodel.write(tall, os.path.join(args.scene, "poles"))
            by_height[tall] = len(assets)
            assets.append({"name": name, "parts": [{"name": part, "mesh": f"poles/{name}_{part}.obj", "roughness_value": 0.9 if part == "footing" else 0.5,
                                                    "colour": [round(v, 3) for v in renderer.colour_for_renderer(polemodel.COLOURS[part])]}
                                                   for part in ("footing", "pole", "lamp")]})
        slope = np.array([edge.at(p["x"] + 0.5, p["y"]) - edge.at(p["x"] - 0.5, p["y"]), edge.at(p["x"], p["y"] + 0.5) - edge.at(p["x"], p["y"] - 0.5)], float)
        yaw = float(np.arctan2(slope[1], slope[0])) if np.hypot(*slope) > 1e-6 else 0.0
        instances.append({"asset": by_height[tall], "group": "Poles", "name": f"pole_{n:02d}", "height": tall,
                          "pos": [round(p["x"], 2), round(p["y"], 2), round(float(ref.elevation(p["x"], p["y"])), 3)],
                          "rot": [round(float(np.cos(yaw / 2)), 5), 0.0, 0.0, round(float(np.sin(yaw / 2)), 5)], "scale": [1.0, 1.0, 1.0]})
    print(f"[{time.perf_counter() - start:5.1f} s] {len(things['poles'])} light poles in {len(by_height)} heights:"
          f" {', '.join(f'{h:g}' for h in sorted(by_height))} m")

    # The pavement's own triangles, for whatever has to lie or stand exactly on them.
    road = markmodel.Road(g.vertices, g.road)

    def ground_height(xs, ys):
        return road.height(xs, ys, ground.surface(ref, xs, ys))

    # Parked vehicles: models made from the meshes Chrono ships, in the colour the drone saw.
    if things["vehicles"]:
        models = [m for m in carmodels.build(chrono_data(args), os.path.join(args.scene, "vehicles")) if m["name"] in PARKED]
        # The scan cannot tell a make. It can tell a tall vehicle from a low one, so the tall
        # ones get the tall models and the low ones the low, taking turns among them. A model
        # is then scaled toward the height measured, by 15% at most: Chrono's SUV is a large
        # off-roader, and what stands in a car park is mostly a size smaller.
        models.sort(key=lambda m: (m["height"], m["name"]))
        low = [m for m in models if m["height"] < TALL_VEHICLE] or models
        tall = [m for m in models if m["height"] >= TALL_VEHICLE] or models
        placed, turn = {}, {"low": 0, "tall": 0}
        for n, car in enumerate(things["vehicles"]):
            kind = "tall" if car["height"] >= TALL_SEEN else "low"
            choices = tall if kind == "tall" else low
            model = choices[turn[kind] % len(choices)]
            turn[kind] += 1
            if model["name"] not in placed:
                placed[model["name"]] = len(assets)
                assets.append({"name": model["name"], "parts": [
                    {"name": p["name"], "mesh": f"vehicles/{p['mesh']}", "colour": [round(float(v), 3) for v in p["colour"]],
                     "roughness_value": p.get("roughness", 0.5), "metallic_value": p.get("metallic", 0.0)} for p in model["parts"]]})
            paint = {p["name"]: [round(v, 3) for v in renderer.colour_for_renderer(car["colour"])] for p in model["parts"] if p.get("paint")}
            size = round(float(np.clip(car["height"] / model["height"], 0.85, 1.05)), 3)
            # Stood on its four wheels: tilted to the ground under them, not level in the air.
            pos, rot = vehicles.seat(car["x"], car["y"], car["yaw"], model["wheel_centres"], size, ground_height)
            instances.append({"asset": placed[model["name"]], "group": "Vehicles", "name": f"vehicle_{n:02d}", "model": model["name"],
                              "pos": [round(pos[0], 2), round(pos[1], 2), round(pos[2], 3)], "rot": [round(v, 5) for v in rot],
                              "scale": [size, size, size], "colours": paint})
        print(f"[{time.perf_counter() - start:5.1f} s] {len(things['vehicles'])} parked vehicles: " + ", ".join(i["model"] for i in instances if i["group"] == "Vehicles"))

    # Small structures, each a block of the size and colour the scan gives it.
    if things["blocks"]:
        os.makedirs(os.path.join(args.scene, "props"), exist_ok=True)
        blocks.write_unit(os.path.join(args.scene, "props", "block.obj"))
        assets.append({"name": "block", "parts": [{"name": "block", "mesh": "props/block.obj", "colour": [0.8, 0.8, 0.8], "roughness_value": 0.8}]})
        for n, b in enumerate(things["blocks"]):
            instances.append({"asset": len(assets) - 1, "group": "Props", "name": f"block_{n:02d}",
                              "pos": [round(b["x"], 2), round(b["y"], 2), round(float(ref.elevation(b["x"], b["y"])), 3)],
                              "rot": [round(float(np.cos(b["yaw"] / 2)), 5), 0.0, 0.0, round(float(np.sin(b["yaw"] / 2)), 5)],
                              "scale": [b["length"], b["width"], b["height"]],
                              "colours": {"block": [round(v, 3) for v in renderer.colour_for_renderer(b["colour"])]}})
        print(f"[{time.perf_counter() - start:5.1f} s] {len(things['blocks'])} small structures as blocks")

    # Guard rails and fence, each built in place along the line it was found on.
    if things["barriers"]:
        os.makedirs(os.path.join(args.scene, "barriers"), exist_ok=True)
        metres, drawn = {"guardrail": 0.0, "fence": 0.0}, 0
        # A guard rail stands at the pavement's edge. The fence stands clear of all pavement.
        runs = [(b, run if b["type"] == "guardrail" else railmodel.fence_line(run, edge.at).tolist())
                for b in things["barriers"] for run in railmodel.standing_runs(b)]
        for n, (b, run) in enumerate(runs):
            pts = np.asarray(run, float)
            k = len(pts) // 2
            step = pts[min(k, len(pts) - 1)] - pts[max(k - 1, 0)]
            left = np.array([-step[1], step[0]]) / max(np.hypot(*step), 1e-9)
            mid = (pts[min(k, len(pts) - 1)] + pts[max(k - 1, 0)]) / 2
            # The pavement is on whichever side of the line is nearer to it.
            road_side = 1.0 if edge.at(*(mid + 1.5 * left)) >= edge.at(*(mid - 1.5 * left)) else -1.0
            parts = railmodel.make(b["type"], run, b["height"], ref.elevation, road_side)
            if parts is None:
                continue
            name = f"{'rail' if b['type'] == 'guardrail' else 'fence'}_{n:02d}"
            asset = {"name": name, "parts": []}
            for part, (v, f) in parts.items():
                drawn += railmodel.write_obj(os.path.join(args.scene, "barriers", f"{name}_{part}.obj"), v, f)
                asset["parts"].append({"name": part, "mesh": f"barriers/{name}_{part}.obj", "roughness_value": 0.45, "double_sided": part == "beam",
                                       "colour": [round(c, 3) for c in renderer.colour_for_renderer(railmodel.COLOURS[part])]})
            assets.append(asset)
            metres[b["type"]] += railmodel.length_of(run)
            seen = sum(railmodel.length_of(r) for r in railmodel.seen_runs(b)) if "seen" in b else railmodel.length_of(run)
            instances.append({"asset": len(assets) - 1, "group": "Barriers", "name": name, "type": b["type"], "length": round(railmodel.length_of(run), 1),
                              "seen_length": round(seen, 1), "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
        print(f"[{time.perf_counter() - start:5.1f} s] barriers: {metres['guardrail']:.0f} m of guard rail, {metres['fence']:.0f} m of fence, {drawn} triangles")

    # Rock piles and boulders, each a heap of stones built in place.
    if things.get("rocks"):
        os.makedirs(os.path.join(args.scene, "rocks"), exist_ok=True)
        stones = drawn = 0
        for n, rock in enumerate(things["rocks"]):
            parts, count = rockmodel.make(rock, ground_height, n)
            if not parts:
                continue
            entry = {"name": f"rocks_{n:02d}", "parts": []}
            for shade, (v, f) in parts.items():
                drawn += railmodel.write_obj(os.path.join(args.scene, "rocks", f"rocks_{n:02d}_{shade}.obj"), v, f)
                colour = np.clip(np.asarray(rock["colour"], float) * rockmodel.SHADES[shade], 0.0, 1.0)
                entry["parts"].append({"name": shade, "mesh": f"rocks/rocks_{n:02d}_{shade}.obj", "colour": [round(float(c), 3) for c in renderer.colour_for_renderer(colour)],
                                       "roughness_value": 0.95})
            assets.append(entry)
            instances.append({"asset": len(assets) - 1, "group": "Rocks", "name": f"rocks_{n:02d}", "kind": rock["kind"], "stones": count,
                              "at": [round(rock["x"], 2), round(rock["y"], 2)], "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
            stones += count
        print(f"[{time.perf_counter() - start:5.1f} s] {len(things['rocks'])} rock piles and boulders: {stones} stones, {drawn} triangles")

    # One-of-a-kind things measured by hand: the lattice tower and the swing gates.
    if things.get("landmarks"):
        added, placed_marks = landmarks.build(things["landmarks"], poles.suns(args.work), ground_height, args.scene, renderer.colour_for_renderer)
        for item in placed_marks:
            item["asset"] += len(assets)
        assets += added
        instances += placed_marks
        print(f"[{time.perf_counter() - start:5.1f} s] landmarks: " + ", ".join(f"{i['name']} ({i['height']:g} m)" for i in placed_marks))

    # Road paint, as ribbons laid on the pavement.
    if things["markings"]:
        os.makedirs(os.path.join(args.scene, "markings"), exist_ok=True)
        drawn = 0
        for colour, (v, f) in markmodel.make(things["markings"], ref, road).items():
            drawn += markmodel.write_obj(os.path.join(args.scene, "markings", f"paint_{colour}.obj"), v, f)
            assets.append({"name": f"paint_{colour}", "parts": [{"name": "paint", "mesh": f"markings/paint_{colour}.obj", "roughness_value": 0.9,
                                                                 "colour": [round(c, 3) for c in renderer.colour_for_renderer(markmodel.COLOURS[colour])]}]})
            instances.append({"asset": len(assets) - 1, "group": "Markings", "name": f"paint_{colour}", "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
        # The strokes themselves travel with the scene: they are the lane geometry, as data.
        with open(os.path.join(args.scene, "markings", "strokes.json"), "w") as f:
            json.dump(things["markings"], f)
        extra_files.append("markings/strokes.json")
        print(f"[{time.perf_counter() - start:5.1f} s] road paint: {len(things['markings'])} strokes, {drawn} triangles")

    # Edge lines along the roads, in a group of their own.
    if things["edge_lines"]:
        os.makedirs(os.path.join(args.scene, "markings"), exist_ok=True)
        v, f = markmodel.make(things["edge_lines"], ref, road)["white"]
        drawn = markmodel.write_obj(os.path.join(args.scene, "markings", "edge_lines.obj"), v, f)
        assets.append({"name": "edge_lines", "parts": [{"name": "paint", "mesh": "markings/edge_lines.obj", "roughness_value": 0.9,
                                                        "colour": [round(c, 3) for c in renderer.colour_for_renderer(markmodel.COLOURS["white"])]}]})
        instances.append({"asset": len(assets) - 1, "group": "EdgeLines", "name": "edge_lines", "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
        with open(os.path.join(args.scene, "markings", "edge_lines.json"), "w") as out:
            json.dump(things["edge_lines"], out)
        extra_files.append("markings/edge_lines.json")
        print(f"[{time.perf_counter() - start:5.1f} s] edge lines: {len(things['edge_lines'])} pieces, {drawn} triangles")

    # Cones, stood back up where the scan flattened them.
    found_cones = things["cones"]
    if found_cones:
        data = chrono_data(args)
        os.makedirs(os.path.join(args.scene, "cones"), exist_ok=True)
        triangles = cone_model.convert(os.path.join(data, cone_model.SOURCE), os.path.join(args.scene, "cones"))
        assets.append({"name": "cone", "parts": [{"name": part, "mesh": f"cones/cone_{part}.obj", "colour": renderer.colour_for_renderer(cones.ORANGE), "roughness_value": 0.5}
                                                  for part in ("body", "base")]})
        for n, c in enumerate(found_cones):
            z = float(ref.elevation(c["x"], c["y"]))
            instances.append({"asset": len(assets) - 1, "group": "Cones", "name": f"cone_{n:03d}", "height": c["height"],
                              "pos": [round(c["x"], 2), round(c["y"], 2), round(z, 3)], "rot": [1.0, 0.0, 0.0, 0.0], "scale": [c["height"]] * 3,
                              "colours": {part: [round(v, 3) for v in renderer.colour_for_renderer(c[part])] for part in ("body", "base")}})
        print(f"[{time.perf_counter() - start:5.1f} s] {len(found_cones)} cones, {triangles} triangles each")

    manifest = {
        "version": 5,
        "name": "Columbus 151 Speedway",
        "frame": {
            "description": "x east, y north, z up, metres. z is elevation above sea level (NAVD88).",
            "utm_zone": layout.UTM_ZONE, "origin_easting": layout.ORIGIN_E, "origin_northing": layout.ORIGIN_N,
            "extent": [layout.SCENE_X0, layout.SCENE_Y0, layout.SCENE_X1, layout.SCENE_Y1],
        },
        "labels": ["Road", "Terrain", "Buildings", "Trees", "Cones", "Poles", "Vehicles", "Barriers", "Props", "Rocks", "Markings", "EdgeLines"],
        "texture_levels": {k: layout.LEVELS[k] for k in args.levels.split(",")},
        "collision": COLLISION,
        "files": extra_files,
        "start": layout.START,
        "assets": assets,
        "instances": instances,
    }
    with open(os.path.join(args.scene, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[{time.perf_counter() - start:5.1f} s] wrote {MANIFEST}: {len(assets)} assets, {len(instances)} placements")


if __name__ == "__main__":
    main()
