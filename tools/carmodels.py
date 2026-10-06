#!/usr/bin/env python3
"""Static parked-vehicle models, made from the vehicle meshes that ship with stock PyChrono.

    python carmodels.py CHRONO_DATA_DIR OUT_DIR [NAME ...]

Chrono's data directory has a handful of road vehicles, BSD-3-Clause like the rest of Chrono.
Each one listed in carspecs.py becomes a few plain OBJ files, one per kind of surface (paint,
glass, dark trim, bright parts and lamps, tyres, wheels), and a JSON file that lists them
with a colour each. The scene draws every part as one triangle mesh with one plain material,
back faces culled, and recolours the part marked "paint" per car.

Frame: x forward, y left, z up, metres. The origin is the middle of the footprint, midway
between the bumpers and on the centre line, and z = 0 is the ground under the tyres. The
wheels stand where the Chrono vehicle has them: the spindle positions are read from the
vehicle's own JSON files, and the body is tipped by the fraction of a degree that brings
front and rear tyres down to the ground together. car_wheelcheck.py compares the result with
a vehicle built by PyChrono.

What is done to a chassis mesh, in order: sorted into parts and cut along the borders in its
texture (carparts.py), reduced to what can be seen from outside, each triangle facing the
way it is seen (carvisible.py), and thinned where it is flat enough not to show it
(carthin.py). A wheel is the source tyre's outline turned about the axle, and the outside
face of the source rim.

Colours are linear light, the way an MTL file from Blender has them and the way the renderer
takes them, so they go to SetDiffuseColor as they are. Where the source colours are in a
texture (an sRGB picture), the part's colour is the mean of its triangles' texels, decoded.

Needs numpy, and Pillow for the vehicles whose colours are in textures. The same input gives
the same files, byte for byte.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carmesh  # noqa: E402
import carparts  # noqa: E402
import carpickup  # noqa: E402
import carspecs  # noqa: E402
import carthin  # noqa: E402
import carvisible  # noqa: E402

TYRE_SEGMENTS = 32
BODY_WITHIN = 12.0      # degrees a thinned body triangle may face away from the surface it replaces
RIM_TRIANGLES = 800     # what a rim is thinned to, by its outside face alone
SURFACE = {   # roughness and metallic for each kind of part
    "body": (0.35, 0.1), "top": (0.35, 0.1), "glass": (0.15, 0.0), "trim": (0.7, 0.0), "bright": (0.4, 0.0),
    "taillights": (0.3, 0.0), "tyres": (0.9, 0.0), "wheels": (0.4, 0.3),
}
ORDER = ["body", "top", "glass", "trim", "bright", "taillights", "tyres", "wheels"]


# ---------------------------------------------------------------- wheels

def build_wheel(spec, data_dir, log):
    """One left wheel about the origin, axle along y, outside face towards +y.

    Returns ({part: (P, N, colour)}, tyre radius, tyre half width). The tyre is the source
    tyre's outline turned about the axle, which is all a parked wheel shows of it. The rim is
    the source rim less what the tyre hides and what faces the car.
    """
    folder = os.path.join(data_dir, "vehicle")
    tyre = carmesh.load_obj(os.path.join(folder, spec["tyre"]))
    points = tyre["P"].reshape(-1, 3)
    radial = np.hypot(points[:, 0], points[:, 2])
    hull = carmesh.hull_2d(np.stack([points[:, 1], radial], axis=1))
    # The hull's edge between the two beads lies inside the rim: start the profile after it.
    bead = int(np.argmin([hull[k, 1] + hull[(k + 1) % len(hull), 1] for k in range(len(hull))]))
    profile = carmesh.simplify_closed(np.roll(hull, -(bead + 1), axis=0), spec.get("tyre_profile", 10))
    tyre_P = carmesh.lathe(profile, TYRE_SEGMENTS, closed=False, inside=profile.mean(axis=0))
    radius = float(profile[:, 1].max())
    inner = float(min(profile[0, 1], profile[-1, 1]))
    half_width = float(np.abs(profile[:, 0]).max())
    # A dark disc behind the spokes, so that nothing shows through the wheel from either side.
    back_y = -0.35 * half_width
    backing = np.concatenate([carmesh.disc(back_y, inner + 0.004, TYRE_SEGMENTS, 1),
                              carmesh.disc(back_y, inner + 0.004, TYRE_SEGMENTS, -1)])
    tyre_P = np.concatenate([tyre_P, backing])
    tyre_N = carmesh.smooth_normals(tyre_P, 35.0)
    tyre_colour = carparts.plain_colour(tyre, os.path.join(folder, os.path.dirname(spec["tyre"])), spec.get("tyre_colour", (0.02, 0.02, 0.02)))

    rim = carmesh.load_obj(os.path.join(folder, spec["rim"]))
    rim_folder = os.path.join(folder, os.path.dirname(spec["rim"]))
    rim_colour = carparts.plain_colour(rim, rim_folder, spec.get("rim_colour", (0.5, 0.5, 0.5)))
    rim_P = rim["P"]
    rim_N = rim["N"] if rim["N"] is not None else carmesh.smooth_normals(rim_P)
    hole = np.zeros(len(rim_P), dtype=bool)
    if "rim_pattern" in spec:
        # This rim is a flat disc with its spokes painted on. The dark of the painting is cut
        # out and drawn with the tyres, as the openings between the spokes.
        rule, UV = spec["rim_pattern"], rim["UV"]
        for _ in range(3):        # the disc's triangles are far larger than the pattern
            rim_P, rim_N, UV = carmesh.quarter(rim_P, rim_N, UV)
        rim_N = carmesh.unit(rim_N)
        for _ in range(6):
            before = len(rim_P)
            rim_P, rim_N, UV = carparts.cut_along_borders(rim_P, rim_N, UV, rule, rim_folder)
            if len(rim_P) == before:
                break
        kind, texel = carparts.texel_kinds(np.einsum("sk,tkc->tsc", carmesh.SAMPLES, UV), rule, rim_folder)
        hole = (kind == carparts.DARK).sum(axis=1) >= 4
        _, area = carmesh.face_normals(rim_P)
        light = carparts.srgb_to_linear(texel[~hole]).mean(axis=1)
        rim_colour = [float(x) for x in (light * area[~hole, None]).sum(axis=0) / area[~hole].sum()]
    rim_P, rim_N, _ = carmesh.wind_to_normals(rim_P, rim_N)
    source_count = len(rim["P"])
    directions = carvisible.view_directions(((-50.0, 12), (-25.0, 16), (0.0, 24), (25.0, 16), (50.0, 12)))
    directions = directions[directions[:, 1] < -0.2]          # looking at the outside face
    rim_P, rim_N, index = carvisible.outside_only(rim_P, rim_N, directions, tyre_P, pixel=0.004)
    hole = hole[index]
    # A rim is small and seen from a distance: it is thinned to a count, with looser limits
    # on its shape than a body gets.
    kept, reduced = carthin.decimate(rim_P, hole.astype(np.int64), target=spec.get("rim_triangles", RIM_TRIANGLES), deviation=0.004)
    rim_N = carthin.carry_normals(reduced, rim_P, rim_N)
    rim_P, hole = reduced, hole[kept]
    tyre_P = np.concatenate([tyre_P, rim_P[hole]])
    tyre_N = np.concatenate([tyre_N, rim_N[hole]])
    rim_P, rim_N = rim_P[~hole], rim_N[~hole]
    log(f"  wheel: tyre radius {radius:.4f}, half width {half_width:.3f}, {len(tyre_P)} triangles turned from a "
        f"{len(profile)}-point outline of the source's {len(tyre['P'])}; rim {len(rim_P)} of {source_count}")
    return {"tyres": (tyre_P, tyre_N, tyre_colour), "wheels": (rim_P, rim_N, rim_colour)}, radius, half_width


# ---------------------------------------------------------------- body

def dark_core(body_P, wheels_P, spindles, radius, half_width, log):
    """Dark boxes inside a body that is an open shell, to stand for the floor and the arch
    linings: without them one would look in at a wheel arch and out through the other side.

    Three boxes in a row, behind the rear axle, between the axles and ahead of the front one,
    reach from behind the rear wheels to ahead of the front ones. They are as wide as the gap
    between the tyres and as high as the tyres' tops, and each starts at the lowest edge of
    the body over it. Each is then shrunk until it cannot be seen from above, which is where
    it would show if it stuck out through the bonnet, the boot or a bumper. Returns their
    triangles, without bottoms.
    """
    xs = sorted(set(round(float(c[0]), 6) for c in spindles.values()))
    y = min(abs(float(c[1])) for c in spindles.values()) - half_width - 0.03
    cuts = [xs[0] - radius - 0.10, xs[0], xs[-1], xs[-1] + radius + 0.10]
    above = carvisible.Views(np.concatenate([body_P, wheels_P]), carvisible.view_directions(((80.0, 4), (90.0, 1))), pixel=0.01)
    centre = body_P.mean(axis=1)
    out = []
    for k in range(3):
        over = (centre[:, 0] >= cuts[k]) & (centre[:, 0] <= cuts[k + 1]) & (np.abs(centre[:, 1]) <= y + 0.3)
        floor = max(float(body_P[over][:, :, 2].min()) + 0.01, 0.12) if over.any() else 0.12
        low = np.array([cuts[k], -y, floor])
        high = np.array([cuts[k + 1], y, 2.0 * radius + 0.06])
        faces = "yYZ" + ("x" if k == 0 else "") + ("X" if k == 2 else "") + ("xX" if k == 1 else "")
        for attempt in range(25):
            fine = carmesh.box(low, high, step=0.05, faces=faces)
            front, _ = above.seen(fine)
            if not (front > 0).any():
                break
            shows = fine.mean(axis=1)[front > 0]
            # Pull in whichever faces the visible bits lie on.
            if (shows[:, 2] > high[2] - 0.03).any():
                high[2] -= 0.02
            if k == 2 and (shows[:, 0] > high[0] - 0.06).any():
                high[0] -= 0.03
            if k == 0 and (shows[:, 0] < low[0] + 0.06).any():
                low[0] += 0.03
            if (np.abs(shows[:, 1]) > high[1] - 0.03).any():
                high[1] -= 0.01
                low[1] += 0.01
        else:
            log(f"  core {k}: does not fit inside the body, left out")
            continue
        log(f"  core {k}: x {low[0]:.2f} to {high[0]:.2f}, half width {high[1]:.2f}, z {low[2]:.2f} to {high[2]:.2f},"
            f" {attempt} shrinks")
        out.append(carmesh.box(low, high, faces=faces))
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def build_body(spec, data_dir, to_level, wheels_P, spindles, radius, half_width, log):
    """The chassis mesh in the level frame, cut down to what shows: (P, N, part, colour)."""
    folder = os.path.join(data_dir, "vehicle", os.path.dirname(spec["chassis"]))
    mesh = carmesh.load_obj(os.path.join(data_dir, "vehicle", spec["chassis"]))
    source_count = len(mesh["P"])
    P, N, part, colour = carparts.classify(mesh, spec, folder)
    listed = len(P)

    if "frame" in spec:      # this file is a scaled, shifted copy of the mesh the vehicle JSON uses
        scale, shift = spec["frame"]
        P = (P - np.array(shift)) / scale
    P, N = to_level(P), to_level(N, normal=True)
    P, N, rewound = carmesh.wind_to_normals(P, N)
    once = carmesh.first_of_each(P)
    repeats = len(P) - len(once)
    P, N, part, colour = P[once], N[once], part[once], colour[once]
    if "pickup" in spec:
        rear_axle_x = min(float(c[0]) for c in spindles.values())
        P, N, part, colour = carpickup.convert(P, N, part, colour, rear_axle_x, 2.0 * radius, spec["pickup"], log)

    occluders = wheels_P
    core = np.zeros((0, 3, 3))
    if spec.get("core"):
        core = dark_core(P, wheels_P, spindles, radius, half_width, log)
        occluders = np.concatenate([wheels_P, core])
    P, N, index = carvisible.outside_only(P, N, carvisible.view_directions(), occluders, ground=0.0)
    part, colour = part[index], colour[index]
    once = carmesh.first_of_each(P)          # a face held in both directions and seen in one comes out twice
    P, N, part, colour, index = P[once], N[once], part[once], colour[once], index[once]
    log(f"  body: {source_count} triangles in the source, {listed} in the listed materials once cut along "
        f"the texture's borders, {repeats} of those repeats, "
        f"{len(np.unique(index))} seen from outside, {len(index) - len(np.unique(index))} kept twice as thin sheets, "
        f"{rewound} rewound to their normals")

    # Thin out what is flat enough not to show it. Glass is left alone: its panes are few
    # and large already.
    names = sorted(set(part.tolist()))
    group = np.array([names.index(x) for x in part.tolist()], dtype=np.int64)
    kept, reduced = carthin.decimate(P, group, normals=N, fixed=part == "glass", within_degrees=BODY_WITHIN)
    N = carthin.carry_normals(reduced, P, N)
    log(f"  body: thinned from {len(P)} to {len(reduced)} triangles")
    P, part, colour = reduced, part[kept], colour[kept]

    # Glass is drawn opaque, where the source drew it see-through, and its normals in the
    # source are not fit to be seen: they shade the panes in wedges. Flat panes shade evenly.
    glass = part == "glass"
    if glass.any():
        N[glass] = carmesh.smooth_normals(P[glass], 50.0)

    if len(core):
        dark = colour[part == "trim"]
        tone = dark.mean(axis=0) if len(dark) else np.array([0.03, 0.03, 0.03])
        P = np.concatenate([P, core])
        N = np.concatenate([N, np.repeat(carmesh.face_normals(core)[0][:, None, :], 3, axis=1)])
        part = np.concatenate([part, np.full(len(core), "trim", dtype=object)])
        colour = np.concatenate([colour, np.repeat(tone[None, :], len(core), axis=0)])
    return P, N, part, colour


# ---------------------------------------------------------------- one vehicle

def build_model(spec, data_dir, out_dir, log=print):
    log(f"{spec['name']}: {spec['what']}")
    spindles = carspecs.wheel_positions(data_dir, spec["vehicle"])          # (axle, side) -> chassis frame
    for axle, lift in spec.get("spindle_lift", {}).items():
        for side in (0, 1):
            x, y, z = spindles[(axle, side)]
            spindles[(axle, side)] = (x, y, z + lift)
    wheel, radius, half_width = build_wheel(spec, data_dir, log)

    # The level frame: the chassis frame tipped so that front and rear spindles are equally
    # high, and raised so that they are one tyre radius above z = 0.
    last = max(axle for axle, _ in spindles)
    front_x, _, front_z = spindles[(0, 0)]
    rear_x, _, rear_z = spindles[(last, 0)]
    pitch = float(np.arctan2(front_z - rear_z, front_x - rear_x))
    rise = radius - float(carmesh.rotate_y(np.array([front_x, 0.0, front_z]), pitch)[2])

    def to_level(points, normal=False):
        turned = carmesh.rotate_y(points, pitch)
        return turned if normal else turned + np.array([0.0, 0.0, rise])

    centres = {key: to_level(np.array(c)) for key, c in spindles.items()}
    pieces = {}          # part -> list of (P, N, colour per triangle)
    for (axle, side), centre in sorted(centres.items()):
        turn = 0.0 if side == 0 else np.pi
        for name, (wheel_P, wheel_N, wheel_colour) in wheel.items():
            pieces.setdefault(name, []).append((carmesh.rotate_z(wheel_P, turn) + centre, carmesh.rotate_z(wheel_N, turn),
                                                np.repeat(np.array([wheel_colour], dtype=np.float64), len(wheel_P), axis=0)))
    wheels_P = np.concatenate([p for group in pieces.values() for p, _, _ in group])

    P, N, part, colour = build_body(spec, data_dir, to_level, wheels_P, centres, radius, half_width, log)
    for name in sorted(set(part.tolist())):
        chosen = part == name
        pieces.setdefault(name, []).append((P[chosen], N[chosen], colour[chosen]))

    everything = np.concatenate([p for group in pieces.values() for p, _, _ in group]).reshape(-1, 3)
    low, high = everything.min(axis=0), everything.max(axis=0)
    # The origin goes midway between bumpers and on the vehicle's centre line, which is the
    # middle of its width to within a couple of millimetres (a mirror may reach further on one side).
    shift = np.array([-(low[0] + high[0]) / 2, 0.0, 0.0])

    parts, total = [], 0
    sources = [spec["chassis"], spec["rim"], spec["tyre"], spec["vehicle"]] + list(spec.get("also", []))
    for name in [n for n in ORDER if n in pieces] + sorted(set(pieces) - set(ORDER)):
        part_P = np.concatenate([p for p, _, _ in pieces[name]]) + shift
        part_N = np.concatenate([n for _, n, _ in pieces[name]])
        part_colour = np.concatenate([c for _, _, c in pieces[name]])
        _, area = carmesh.face_normals(part_P)
        mean = (part_colour * area[:, None]).sum(axis=0) / max(area.sum(), 1e-12)
        mesh_name = f"{spec['name']}_{name}.obj"
        written = carmesh.write_obj(os.path.join(out_dir, mesh_name), part_P, part_N, [
            f"{spec['name']} ({spec['what']}): {name}. x forward, y left, z up, metres, tyres on z = 0.",
            "From Project Chrono's data/vehicle/" + ", ".join(sources[:3]) + " (BSD-3-Clause)."])
        total += written
        roughness, metallic = SURFACE.get(name, SURFACE["bright"])
        parts.append({"name": name, "mesh": mesh_name, "colour": [round(float(x), 4) for x in mean],
                      "paint": name == "body", "roughness": roughness, "metallic": metallic, "triangles": written})
        log(f"  {name:10s} {written:6d} triangles, colour {parts[-1]['colour']}")
    model = {
        "name": spec["name"], "description": spec["what"],
        "source": ["vehicle/" + s for s in sources],
        "length": round(float(high[0] - low[0]), 3), "width": round(float(high[1] - low[1]), 3),
        "height": round(float(high[2]), 3),
        "wheelbase": round(float(np.hypot(front_x - rear_x, front_z - rear_z)), 3), "triangles": total,
        "colour_space": "linear, as SetDiffuseColor takes it",
        "tyre_radius": round(radius, 4),
        "wheel_centres": [[round(float(v), 4) for v in c + shift] for _, c in sorted(centres.items())],
        "body_pitch_degrees": round(float(np.degrees(pitch)), 3),
        "parts": parts,
    }
    # One key to a line, and one part to a line, so that the file reads as a table.
    lines = []
    for key, value in model.items():
        if key in ("source", "parts"):
            lines.append(f' {json.dumps(key)}: [\n' + ",\n".join("  " + json.dumps(item) for item in value) + "\n ]")
        else:
            lines.append(f" {json.dumps(key)}: {json.dumps(value)}")
    with open(os.path.join(out_dir, spec["name"] + ".json"), "w", newline="\n") as f:
        f.write("{\n" + ",\n".join(lines) + "\n}\n")
    log(f"  {model['length']} x {model['width']} x {model['height']} m, wheelbase {model['wheelbase']}, {total} triangles")
    return model


def build(chrono_data_dir, out_dir, only=None, log=print):
    """Write every model into out_dir. Returns their JSON descriptions, in order."""
    os.makedirs(out_dir, exist_ok=True)
    return [build_model(spec, chrono_data_dir, out_dir, log) for spec in carspecs.MODELS if not only or spec["name"] in only]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2], only=sys.argv[3:])
