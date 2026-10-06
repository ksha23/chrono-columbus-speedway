#!/usr/bin/env python3
"""Find the guard rails and fences in the drone scan and say where they run.

    python -I barriers.py WORK_DIR OUT_DIR

writes WORK_DIR/barriers.json and overlay pictures to OUT_DIR. From Python, find(WORK_DIR)
returns the same list of dicts:

    {"type": "guardrail" | "fence", "points": [[x, y], ...], "height": m, "confidence": 0..1}

How it works. Galvanised steel seen from above is a thin line that is bluer than whatever it
lies on: warm concrete, grass, leaves. So the photo gives the lines (barriers_lines.py): a
thin-line filter on blue minus red, then a test at every candidate pixel that the line carries
on for metres in some direction. What survives is traced into pieces. Here each piece is
measured against the scan and the photo and the ones that cannot be a barrier are dropped:
shadows are dark, branches are up in the canopy, roof edges are on buildings, cars stand out
on the pavement. Pieces that are one barrier seen with gaps are joined, and each barrier must
then be long enough and have something standing along it. A rail hugs the pavement and stands
under a metre tall, and a fence does neither, which is how the two are told apart.

Guard rails are reported one by one. The fence is one fence around the whole property, so the
stretches of it that were seen are joined into a single closed line (barriers_fence.py). Its
dict carries two more keys: "seen", one flag per segment of "points" saying whether that
segment was seen or only inferred, and "gates", pairs of posts where the site road passes
through. The segment between a pair of gate posts is part of the line and is not fence.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import barriers_fence  # noqa: E402
import barriers_lines as lines  # noqa: E402
from barriers_lines import RES, length_of  # noqa: E402

LINE_WEAK = 6.0           # thin-line contrast that still counts inside a gap
EDGE = 1.0                # metres either side of the pavement's edge that count as on it
GAP_FILLED = 0.85         # fraction of a gap that must be covered by something

# What a piece of barrier must be.
BRIGHT = 90               # brightness (0..255) below which a line is in shade
SHADED_WALL = 1.0         # metres: a shaded line on a thin thing this tall is an overgrown fence
VEHICLE = 8.0             # metres, longer than a car
VEHICLE_TOP = 2.2         # metres, taller than any rail
THIN_TALL = 3.0           # metres: a thin thing taller than this is a trunk or a pole
CANOPY = 2.0              # metres: a line with nothing lower than this beside it is in a tree
BUILDING_MARGIN = 4.5     # metres from a roof inside which lines are the building's own
ON_ROAD = 1.0             # metres inside the pavement beyond which a thing is parked on it

# What a whole barrier must be.
LENGTH_MIN = 5.0          # metres, for a guard rail
FENCE_MIN = 8.0           # metres: a fence is a weaker sight than a rail and must be longer
STANDS = 0.35             # metres above ground that counts as something standing
STANDS_FRACTION = 0.6     # of the seen length, within 0.4 m of the line
RAIL_PAVEMENT = 1.5       # metres from pavement within which a barrier lines the road
RAIL_STANDS = 0.4         # metres that must stand on the very line of a barrier lining the road
RAIL_TOP = 1.1            # metres, a guard rail is not taller than this
SIMPLIFY = 0.12           # metres, tolerance of a guard rail's reported polyline


def context_fields(rasters):
    """Rasters derived once and looked up along every piece."""
    height = np.nan_to_num(rasters["height"], nan=0.0)
    near = 2 * int(round(0.4 / RES)) + 1
    wide = 2 * int(round(0.75 / RES)) + 1
    beside = 2 * int(round(0.5 / RES)) + 1
    step = 2

    def distance(mask):
        """Metres from every cell to the nearest set cell of a mask, worked out at half detail."""
        coarse = ndimage.distance_transform_edt(~mask[::step, ::step]).astype(np.float32) * (RES * step)
        full = np.repeat(np.repeat(coarse, step, 0), step, 1)
        out = np.zeros(mask.shape, np.float32)
        out[:full.shape[0], :full.shape[1]] = full[:mask.shape[0], :mask.shape[1]]
        return out

    return {
        "near_height": ndimage.maximum_filter(height, size=near),
        "on_height": ndimage.maximum_filter(height, size=3),
        "low_beside": ndimage.minimum_filter(height, size=beside),
        # Height of a thin thing above whatever is beside it: what an opening 1.5 m wide removes.
        "ridge": ndimage.maximum_filter(height - ndimage.grey_opening(height, size=(wide, wide)), size=5),
        "near_line": ndimage.maximum_filter(rasters["line"], size=5),
        "from_building": distance(rasters["buildings"]),
        "from_pavement": distance(rasters["pavement"]),
        "into_pavement": distance(~rasters["pavement"]),
    }


def measure(points, seen_length, rasters, fields):
    """What the scan and the photo say along a run of points."""
    row, col = lines.cells(rasters["grid"], rasters["height"].shape, points)
    top = fields["near_height"][row, col]
    return {
        "seen": seen_length,
        "stands": float((top >= STANDS).mean()),
        "top": float(np.percentile(top, 75)),
        "on": float(np.median(fields["on_height"][row, col])),
        "ridge": float(np.percentile(fields["ridge"][row, col], 75)),
        "canopy": float((fields["low_beside"][row, col] > CANOPY).mean()),
        "building": float((fields["from_building"][row, col] < BUILDING_MARGIN).mean()),
        "pavement": float(np.median(fields["from_pavement"][row, col])),
        "on_road": float(np.median(fields["into_pavement"][row, col])),
        "contrast": float(np.median(fields["near_line"][row, col])),
        "lum": float(np.median(rasters["lum"][row, col])),
    }


def piece_fault(m):
    """Why a piece cannot be part of a barrier, or None if it can."""
    if m["canopy"] > 0.5:
        return "canopy"
    if m["ridge"] > THIN_TALL:
        return "tall"
    if m["building"] > 0.3:
        return "building"
    if m["on_road"] > ON_ROAD and (m["seen"] < VEHICLE or m["on"] > VEHICLE_TOP):
        return "on road"
    if m["lum"] < BRIGHT:
        # In shade a blue line is usually the shadow itself: of a pole, or along the edge of
        # long grass. It is a barrier only where the scan shows one standing in it.
        wall = m["ridge"] >= SHADED_WALL
        rail = m["pavement"] <= RAIL_PAVEMENT and m["on"] >= RAIL_STANDS
        if not (wall or rail):
            return "dark"
    return None


def judge(m):
    """Decide whether a measured chain is a barrier, and which kind: (dict, None) or (None, why)."""
    if m["length"] < LENGTH_MIN or m["seen"] < 0.8 * LENGTH_MIN:
        return None, "short"
    if m["stands"] < STANDS_FRACTION:
        return None, "flat"
    lines_road = m["pavement"] <= RAIL_PAVEMENT
    if lines_road and m["on"] < RAIL_STANDS:
        return None, "kerb"
    if lines_road and m["on"] <= RAIL_TOP:
        kind, height = "guardrail", float(np.clip(m["on"], 0.55, 0.85))
    else:
        kind, height = "fence", float(np.clip(m["top"], 1.2, 2.0))
        if m["length"] < FENCE_MIN:
            return None, "short"
    seen = m["seen"] / max(m["length"], 1e-6)
    confidence = 0.35 + 0.25 * min(1.0, (m["stands"] - STANDS_FRACTION) / (1 - STANDS_FRACTION)) \
        + 0.2 * seen + 0.2 * min(1.0, m["seen"] / 20.0) - (0.2 if m["lum"] < BRIGHT else 0.0)
    return {"type": kind, "height": round(height, 2),
            "confidence": round(float(np.clip(confidence, 0, 1)), 2)}, None


def search(work_dir):
    """The whole search. Returns (barriers, rejected). Each barrier carries its measurements
    under "_measured". rejected is a list of (points, why) for pieces and chains turned down,
    which the overlays show so that misses can be hunted."""
    rasters = lines.build_rasters(work_dir)
    fields = context_fields(rasters)
    grid, shape = rasters["grid"], rasters["height"].shape
    road_core = fields["into_pavement"] > 0.5

    def covered(a, b):
        """Is the stretch from a to b hidden rather than open? Something stands along it or
        a faint line shows, and it does not cross two metres of bare road."""
        n = max(2, int(np.hypot(*(b - a)) / RES) + 1)
        t = np.linspace(0.0, 1.0, n)[1:-1, None]
        if len(t) == 0:
            return True
        row, col = lines.cells(grid, shape, a + (b - a) * t)
        stands = fields["near_height"][row, col] >= STANDS - 0.05
        if (stands | (fields["near_line"][row, col] >= LINE_WEAK)).mean() < GAP_FILLED:
            return False
        run = longest = 0
        for bare in road_core[row, col] & ~stands:
            run = run + 1 if bare else 0
            longest = max(longest, run)
        return longest * RES < 2.0

    def guided(a, b):
        """Does the stretch from a to b keep to the pavement's edge, with something standing
        along it? A guard rail does, and can then be followed round a bend on scraps."""
        n = max(2, int(np.hypot(*(b - a)) / RES) + 1)
        row, col = lines.cells(grid, shape, a + (b - a) * np.linspace(0.0, 1.0, n)[:, None])
        at_edge = (fields["from_pavement"][row, col] <= EDGE) & (fields["into_pavement"][row, col] <= EDGE)
        return bool(at_edge.all() and (fields["near_height"][row, col] >= STANDS).mean() >= GAP_FILLED)

    mask = lines.line_pixels(rasters["line"])
    pieces, rejected = [], []
    for piece in lines.trace_pieces(mask, rasters["line"], grid):
        fault = piece_fault(measure(piece, length_of(piece), rasters, fields))
        if fault is None:
            pieces.append(piece)
        else:
            rejected.append((piece, fault))
    pieces = lines.drop_twins(pieces)

    barriers, stretches = [], []
    for chain in lines.join_pieces(pieces, covered, guided):
        seen = np.concatenate([p for p, shown in chain if shown])
        whole = np.concatenate([p for p, _ in chain])
        m = measure(seen, sum(length_of(p) for p, shown in chain if shown), rasters, fields)
        m["length"] = length_of(whole)
        verdict, why = judge(m)
        if verdict is None:
            rejected.append((whole, why))
        elif verdict["type"] == "fence":
            stretches.append((whole, m))
        else:
            pts = lines.fit_polyline(whole, SIMPLIFY)
            # West end first, so that the same rail is always written the same way round.
            if (pts[0, 0], pts[0, 1]) > (pts[-1, 0], pts[-1, 1]):
                pts = pts[::-1]
            verdict["points"] = [[round(float(x), 2), round(float(y), 2)] for x, y in pts]
            verdict["_measured"] = m
            barriers.append(verdict)
    barriers.sort(key=lambda b: (b["points"][0][0], b["points"][0][1]))

    fence = barriers_fence.loop([p for p, _ in stretches], [m["seen"] for _, m in stretches], rasters, fields)
    if fence is not None:
        seen, total = barriers_fence.seen_length(fence)
        tops = np.array([np.clip(m["top"], 1.2, 2.0) for _, m in stretches])
        weight = np.array([m["seen"] for _, m in stretches])
        height = float(tops[np.argsort(tops)][np.searchsorted(np.cumsum(weight[np.argsort(tops)]), weight.sum() / 2)])
        barriers.append({
            "type": "fence",
            "points": [[round(float(x), 2), round(float(y), 2)] for x, y in fence["points"]],
            "seen": fence["seen"],
            "gates": [[[round(float(x), 2), round(float(y), 2)] for x, y in gate] for gate in fence["gates"]],
            "height": round(height, 2),
            "confidence": round(0.3 + 0.6 * seen / max(total, 1e-6), 2),
            "_measured": {"seen": seen, "length": total, "stretches": len(stretches)},
        })
    return barriers, rejected


def plain(barriers):
    keys = ("type", "points", "seen", "gates", "height", "confidence")
    return [{k: b[k] for k in keys if k in b} for b in barriers]


def find(work_dir):
    """Guard rails and the fence found in the scan under work_dir, as a list of dicts."""
    return plain(search(work_dir)[0])


def main():
    work_dir, out_dir = sys.argv[1], sys.argv[2]
    barriers, rejected = search(work_dir)
    with open(os.path.join(work_dir, "barriers.json"), "w") as f:
        json.dump(plain(barriers), f, indent=1)
    for n, b in enumerate(barriers):
        m, p = b["_measured"], b["points"]
        if "seen" in b:
            print(f"{n:3d} fence     loop of {len(p) - 1} segments, {m['length']:.0f} m, {m['seen']:.0f} m of it seen "
                  f"in {m['stretches']} stretches, gates {b['gates']}, conf {b['confidence']:.2f}")
            continue
        print(f"{n:3d} {b['type']:9s} ({p[0][0]:7.1f},{p[0][1]:7.1f}) to ({p[-1][0]:7.1f},{p[-1][1]:7.1f}) "
              f"{m['length']:6.1f} m, seen {m['seen']:6.1f}, stands {m['stands']:.2f}, on {m['on']:.2f}, "
              f"ridge {m['ridge']:.2f}, top {m['top']:.2f}, pavement {m['pavement']:4.1f}, "
              f"contrast {m['contrast']:4.1f}, lum {m['lum']:3.0f}, {len(p)} vertices, conf {b['confidence']:.2f}")
    rails = [b for b in barriers if b["type"] == "guardrail"]
    print(f"guardrail: {len(rails)}, {sum(length_of(np.array(b['points'])) for b in rails):.0f} m")
    import barriers_overlay
    barriers_overlay.write(work_dir, out_dir, barriers, rejected)


if __name__ == "__main__":
    main()
