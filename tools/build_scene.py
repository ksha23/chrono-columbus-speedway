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
import buildings  # noqa: E402
import cone_model  # noqa: E402
import cones  # noqa: E402
import forest  # noqa: E402
import ground  # noqa: E402
import groundphoto  # noqa: E402
import objects  # noqa: E402
import layout  # noqa: E402
import reference  # noqa: E402
import renderer  # noqa: E402
import tiles  # noqa: E402
import transform  # noqa: E402
import treegen  # noqa: E402
import wall_colour  # noqa: E402

# The ground as one collision mesh, and the same triangles as two, for those who want the
# pavement and the rest apart.
COLLISION = {"ground": "speedway_ground.obj", "road": "speedway_road.obj", "terrain": "speedway_terrain.obj"}
MANIFEST = "speedway_scene.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("reference")
    parser.add_argument("scene")
    parser.add_argument("--levels", default="low,standard")
    parser.add_argument("--scan", default=None, help="the scan directory, needed to read wall colours")
    parser.add_argument("--chrono-data", default=None, help="Chrono's data directory, for the cone model (default: ask the installed PyChrono)")
    parser.add_argument("--save-photo", action="store_true", help="also write the finished ground photo and the pavement map to WORK_DIR, for audit_ground.py")
    parser.add_argument("--keep-shadows", action="store_true", help="publish the photo with its shadows still in")
    args = parser.parse_args()
    start = time.perf_counter()

    raster = json.load(open(os.path.join(args.work, "raster.json")))
    ref = reference.Reference(args.reference)
    t = transform.Transform(args.work)
    os.makedirs(args.scene, exist_ok=True)

    filled, matched, found_cones, masks, edge = groundphoto.prepare(args.work, ref, raster, deshadow=not args.keep_shadows)
    print(f"[{time.perf_counter() - start:5.1f} s] ground photo ready")
    if args.save_photo:
        np.save(os.path.join(args.work, "ground_photo.npy"), filled)
        np.save(os.path.join(args.work, "pavement.npy"), edge.distance > 0)

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

    assets, instances = [], []
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

    # Cones, stood back up where the scan flattened them.
    if found_cones:
        data = args.chrono_data
        if data is None:
            import pychrono
            data = pychrono.GetChronoDataPath()
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
        "version": 1,
        "name": "Columbus 151 Speedway",
        "frame": {
            "description": "x east, y north, z up, metres. z is elevation above sea level (NAVD88).",
            "utm_zone": layout.UTM_ZONE, "origin_easting": layout.ORIGIN_E, "origin_northing": layout.ORIGIN_N,
            "extent": [layout.SCENE_X0, layout.SCENE_Y0, layout.SCENE_X1, layout.SCENE_Y1],
        },
        "labels": ["Road", "Terrain", "Buildings", "Trees", "Cones"],
        "texture_levels": {k: layout.LEVELS[k] for k in args.levels.split(",")},
        "collision": COLLISION,
        "start": layout.START,
        "assets": assets,
        "instances": instances,
    }
    with open(os.path.join(args.scene, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[{time.perf_counter() - start:5.1f} s] wrote {MANIFEST}: {len(assets)} assets, {len(instances)} placements")


if __name__ == "__main__":
    main()
