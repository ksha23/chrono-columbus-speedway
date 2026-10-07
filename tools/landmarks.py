"""One-of-a-kind things: a lattice tower and a pair of swing gates.

Everything else that stands on the site is found by a program, because there are dozens of
each: cones, poles, cars, trees. Of these there is one, or two. A detector written for one
tower would be a description of that tower. So they are measured by hand in the scan's photo,
once, and the measurements are kept in landmarks.json: where each foot or post stands, and
where its shadow ends. Heights are not measured. They follow from the shadows and the sun,
the same way a light pole's does.

What is in the photo of each is painted out like anything else that stood on the ground:
its own flattened picture, and its shadow.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

import gatemodel
import railmodel
import towermodel

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landmarks.json")
ROUND_TOWER = 1.5      # metres painted out beyond a tower's picture
ROUND_GATE = 0.7       # and beyond a gate's tubes and their shadows
NEAR_SHADOW = 9.0      # metres of a tower's shadow next to its feet that the shadow steps leave a smudge of


def load(path=FILE):
    with open(path) as f:
        return json.load(f)


def _cast(sun):
    """How far a shadow reaches on the ground per metre of height, as a vector (east, north)."""
    away = np.radians(sun[0] + 180.0)
    return np.array([np.sin(away), np.cos(away)]) / np.tan(np.radians(sun[1]))


def tower_height(tower, suns):
    """The height of a tower's top, from where its shadow ends."""
    reach = np.hypot(tower["shadow_tip"][0] - tower["x"], tower["shadow_tip"][1] - tower["y"])
    return float(reach * np.tan(np.radians(suns[tower["sun"]][1])))


def gate_height(gate, suns):
    """The height of a gate's rail, from the length of its posts' shadows."""
    return float(gate["post_shadow"] * np.tan(np.radians(suns[gate["sun"]][1])))


def _hull(points, raster, cell, shape, margin):
    """Cells within margin of the convex hull of some points."""
    points = np.asarray(points, float)
    ring = points[ConvexHull(points).vertices]
    picture = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(picture).polygon([((x - raster["x0"]) / cell - 0.5, (raster["y1"] - y) / cell - 0.5) for x, y in ring], fill=1)
    reach = int(np.ceil(margin / cell))
    yy, xx = np.ogrid[-reach:reach + 1, -reach:reach + 1]
    return ndimage.binary_dilation(np.asarray(picture, dtype=bool), structure=(xx * xx + yy * yy) * cell * cell <= margin * margin)


def footprint(marks, raster, cell, shape, suns):
    """What the landmarks left in the photo, on cells of the given size: a boolean map."""
    out = np.zeros(shape, bool)
    for tower in marks.get("towers", []):
        centre = np.array([tower["x"], tower["y"]])
        half = tower["base"] / 2
        feet = centre + half * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        toward = np.asarray(tower["shadow_tip"], float) - centre
        toward /= np.hypot(*toward)
        left = np.array([-toward[1], toward[0]])
        near = [centre + NEAR_SHADOW * toward + side * half * left for side in (-1.0, 1.0)]
        out |= _hull(np.concatenate([feet, [tower["picture_top"]], near]), raster, cell, shape, ROUND_TOWER)
    for gate in marks.get("gates", []):
        shift = gate_height(gate, suns) * _cast(suns[gate["sun"]])
        hinge = np.asarray(gate["hinge"], float)
        for end in (gate["leaf_tip"], gate["wing_end"]):
            end = np.asarray(end, float)
            out |= _hull([hinge, end, hinge + shift, end + shift], raster, cell, shape, ROUND_GATE)
    return out


def tower_shadows(marks, raster, cell, shape):
    """Where the towers' shadows lie, from foot to tip and a little to either side: a boolean map on cells of the given size."""
    out = np.zeros(shape, bool)
    for tower in marks.get("towers", []):
        centre, tip = np.array([tower["x"], tower["y"]]), np.asarray(tower["shadow_tip"], float)
        toward = (tip - centre) / np.hypot(*(tip - centre))
        left = np.array([-toward[1], toward[0]]) * tower["base"] / 2
        out |= _hull([centre + left, centre - left, tip + left, tip - left], raster, cell, shape, ROUND_TOWER)
    return out


def build(marks, suns, ground, scene_dir, colour):
    """Write the landmarks' meshes under scene_dir. Returns (assets, instances) for the manifest.

    ground(xs, ys) gives the ground's height. colour turns a colour as seen into the one the
    renderer needs. An instance's "asset" is its index in the returned assets: the caller
    adds its own offset.
    """
    assets, instances = [], []
    if marks.get("towers"):
        os.makedirs(os.path.join(scene_dir, "props"), exist_ok=True)
    for n, tower in enumerate(marks.get("towers", [])):
        height = tower_height(tower, suns)
        parts = towermodel.make(height, tower["base"], tower["tail_azimuth"])
        for part, (v, f) in parts.items():
            railmodel.write_obj(os.path.join(scene_dir, "props", f"{tower['name']}_{part}.obj"), v, f)
        assets.append({"name": tower["name"], "parts": [
            {"name": part, "mesh": f"props/{tower['name']}_{part}.obj", "colour": [round(c, 3) for c in colour(towermodel.COLOURS[part])],
             "roughness_value": 0.5, "metallic_value": 0.4 if part == "steel" else 0.0} for part in parts]})
        z = float(ground(np.array([tower["x"]]), np.array([tower["y"]]))[0])
        instances.append({"asset": len(assets) - 1, "group": "Props", "name": f"tower_{n:02d}", "kind": "tower", "height": round(height, 1),
                          "pos": [tower["x"], tower["y"], round(z, 3)], "rot": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
    if marks.get("gates"):
        os.makedirs(os.path.join(scene_dir, "barriers"), exist_ok=True)
    for n, gate in enumerate(marks.get("gates", [])):
        height = gate_height(gate, suns)
        v, f = gatemodel.make(gate["hinge"], gate["leaf_tip"], gate["wing_end"], gate["posts"], height, ground)
        railmodel.write_obj(os.path.join(scene_dir, "barriers", f"{gate['name']}.obj"), v, f)
        assets.append({"name": gate["name"], "parts": [{"name": "tubes", "mesh": f"barriers/{gate['name']}.obj", "colour": [round(c, 3) for c in colour(gatemodel.COLOUR)],
                                                         "roughness_value": 0.5, "metallic_value": 0.4}]})
        leaf = float(np.hypot(*(np.subtract(gate["leaf_tip"], gate["hinge"]))))
        wing = float(np.hypot(*(np.subtract(gate["wing_end"], gate["hinge"]))))
        instances.append({"asset": len(assets) - 1, "group": "Barriers", "name": f"gate_{n:02d}", "type": "gate", "height": round(height, 2),
                          "length": round(leaf + wing, 1), "seen_length": round(leaf + wing, 1), "leaf": round(leaf, 2),
                          "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
    return assets, instances
