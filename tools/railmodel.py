"""Guard rails and fences as meshes, built along the lines they were found on.

A guard rail is a W-beam, the corrugated steel strip, on posts every 1.9 m. A fence is a line
of posts with a top rail and two tension wires. The chain link itself is not drawn: stock
Chrono::VSG does draw a see-through panel, but the panel throws a solid shadow, two metres of
it along every stretch of fence that stands across the sun.

Each barrier is built in place, in scene coordinates, following the ground under it. Metres,
Z up. Two parts each, so they can be coloured apart: "beam" or "rails", and "posts".
"""
import numpy as np
from scipy import ndimage

STEP = 1.0             # metres between the points a barrier is built through
POST_EVERY = {"guardrail": 1.9, "fence": 3.0}
BURIED = 0.3           # how far posts carry on below the ground
COLOURS = {"beam": (0.70, 0.72, 0.73), "rails": (0.62, 0.64, 0.65), "posts": (0.52, 0.53, 0.54)}

# The W-beam's section: (out toward the road, up), metres, from its lower edge to its upper.
W_SECTION = [(0.00, -0.155), (0.045, -0.115), (0.045, -0.045), (0.00, 0.00), (0.045, 0.045), (0.045, 0.115), (0.00, 0.155)]
BEAM_CENTRE = 0.53     # height of the beam's middle above the ground
BEAM_OUT = 0.10        # how far the beam's back stands out from the posts' centre line


def length_of(points):
    """Length of a polyline in metres."""
    pts = np.asarray(points, float)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def seen_runs(barrier):
    """The stretches of a barrier that show in the photo, as a list of polylines.

    A guard rail shows along its whole length. The fence round the property was seen in places
    and carried on between them where trees hide it: its "seen" list says which segments.
    """
    pts = barrier["points"]
    seen = barrier.get("seen")
    if seen is None:
        return [pts]
    runs, run = [], []
    for i, visible in enumerate(seen):
        if visible:
            run = run or [pts[i]]
            run.append(pts[i + 1])
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def standing_runs(barrier):
    """Where a barrier is to be built, as a list of polylines: all of it but its gates.

    A gate is a pair of points on the line, its two posts. What lies between them is left open.
    """
    pts = np.array(barrier["points"], float)
    closed = len(pts) > 2 and np.allclose(pts[0], pts[-1])
    count = len(pts) - 1                      # segments, and for a closed line distinct vertices too
    open_segment = np.zeros(count, bool)
    for gate in barrier.get("gates") or []:
        i, j = sorted(int(np.hypot(*(pts[:count + (0 if closed else 1)] - np.array(g)).T).argmin()) for g in gate)
        if closed and j - i > count / 2:      # the short way between the posts is across the seam
            open_segment[j:] = True
            open_segment[:i] = True
        else:
            open_segment[i:j] = True
    runs, run = [], []
    for k in range(count):
        if open_segment[k]:
            if run:
                runs.append(run)
            run = []
        else:
            run = run or [pts[k].tolist()]
            run.append(pts[k + 1].tolist())
    if run:
        runs.append(run)
    # A closed loop cut by a gate is one run, not two: the last piece carries on into the first.
    if closed and len(runs) > 1 and not open_segment[0] and not open_segment[-1]:
        runs[0] = runs.pop() + runs[0][1:]
    return runs


def keep_off(points, distance, margin=1.0):
    """Move a line off the pavement: every point at least margin metres outside its edge.

    distance(x, y) is the signed distance to the pavement's edge, positive on the pavement.
    The fence was carried on between the stretches where it shows, in straight lines, and a
    straight line past a bend in the road can clip the road.
    """
    path = resample(points, 1.0)
    for _ in range(40):
        d = distance(path[:, 0], path[:, 1])
        inside = d > -margin
        if not inside.any():
            break
        gx = (distance(path[:, 0] + 0.3, path[:, 1]) - distance(path[:, 0] - 0.3, path[:, 1])) / 0.6
        gy = (distance(path[:, 0], path[:, 1] + 0.3) - distance(path[:, 0], path[:, 1] - 0.3)) / 0.6
        size = np.maximum(np.hypot(gx, gy), 1e-6)
        path[inside] -= 0.25 * np.stack([gx / size, gy / size], axis=1)[inside]
    return path.tolist()


def resample(points, step=STEP):
    """Points every step metres along a polyline, ends included. Returns an (N, 2) array."""
    pts = np.asarray(points, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(seg)])
    if along[-1] < 1e-6:
        return pts[:1]
    count = max(int(np.ceil(along[-1] / step)), 1)
    t = np.linspace(0.0, along[-1], count + 1)
    return np.stack([np.interp(t, along, pts[:, 0]), np.interp(t, along, pts[:, 1])], axis=1)


def _smooth(values, reach):
    """Even out a line of points or of heights: each becomes the average of those within
    reach of it on either side, fewer toward the ends, which stay where they are."""
    values = np.asarray(values, float)
    out = values.copy()
    for i in range(1, len(values) - 1):
        k = min(reach, i, len(values) - 1 - i)
        out[i] = values[i - k:i + k + 1].mean(0)
    return out


def _grade(heights, reach):
    """A line of heights evened out to a steady grade: a running average over reach points
    either side, taken twice, holding the end values past the ends."""
    z = np.asarray(heights, float)
    for _ in range(2):
        z = ndimage.uniform_filter1d(z, 2 * reach + 1, mode="nearest")
    return z


def simplify(points, tolerance):
    """The polyline with every vertex dropped that lies within tolerance of the line through
    its neighbours that are kept (Douglas and Peucker). Straightens a run found in pieces."""
    pts = np.asarray(points, float)
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        d = pts[j] - pts[i]
        length = np.hypot(*d)
        rel = pts[i + 1:j] - pts[i]
        off = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / length if length > 1e-9 else np.hypot(*rel.T)
        k = int(off.argmax())
        if off[k] > tolerance:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


def fence_line(points, distance):
    """Where a fence found in the scan is built: straight between its corners, clear of pavement.

    A fence is strung in straight runs from corner post to corner post. What was found is a
    string of short pieces that wander by a few tenths of a metre, so the line is first reduced
    to its corners, then stood off any pavement it would cross, and the place where it was
    pushed is rounded so that it does not show as a notch.
    """
    path = np.array(keep_off(simplify(points, 0.6), distance))
    smooth = _smooth(path, 3)
    clear = distance(smooth[:, 0], smooth[:, 1]) < -0.5
    path[clear] = smooth[clear]
    return path


def _frames(path):
    """Unit direction and left-hand normal at each point of a 2D path."""
    d = np.gradient(path, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return d, np.stack([-d[:, 1], d[:, 0]], axis=1)


def _box(centre, along, across, half_along, half_across, z0, z1):
    """An upright box. along and across are unit vectors in plan. Returns (vertices, faces)."""
    corners = [centre + sa * half_along * along + sb * half_across * across for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    v = np.array([[c[0], c[1], z0] for c in corners] + [[c[0], c[1], z1] for c in corners])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


def _strip(path, ground, normal, section):
    """A sheet swept along the path: section is a list of (out along normal, up) offsets."""
    rows = [np.column_stack([path + out * normal, ground + up]) for out, up in section]
    v = np.concatenate(rows)
    n = len(path)
    f = []
    for k in range(len(section) - 1):
        for i in range(n - 1):
            a, b = k * n + i, k * n + i + 1
            f += [(a, b, b + n), (a, b + n, a + n)]
    return v, np.array(f)


def _merge(pieces):
    v, f, base = [], [], 0
    for pv, pf in pieces:
        v.append(pv)
        f.append(pf + base)
        base += len(pv)
    return np.concatenate(v), np.concatenate(f)


def _posts(path, level, actual, direction, normal, every, half, top):
    """Posts along the path: each from below the ground as it is, up to top above the level."""
    seg = np.hypot(*np.diff(path, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(seg)])
    count = max(int(round(along[-1] / every)), 1)
    pieces = []
    for t in np.linspace(0.0, along[-1], count + 1):
        i = int(np.clip(np.searchsorted(along, t), 0, len(path) - 1))
        centre = np.array([np.interp(t, along, path[:, 0]), np.interp(t, along, path[:, 1])])
        z = float(np.interp(t, along, level))
        foot = min(z, float(np.interp(t, along, actual)))
        pieces.append(_box(centre, direction[i], normal[i], half[0], half[1], foot - BURIED, z + top))
    return _merge(pieces)


def make(kind, points, height, elevation, road_side=1.0):
    """Meshes for one barrier: {"beam" or "rails": (vertices, faces), "posts": (vertices, faces)}.

    points is the line it stands on, elevation(x, y) the ground height. road_side is +1 if the
    pavement lies to the left of the line as it runs, -1 if to the right: a guard rail's beam
    faces the road.

    Nothing here follows the ground bump for bump. A guard rail is set to the road's grade, read
    just inside the pavement's edge and evened out along the rail. A fence's top rail is set to
    the ground's grade evened out over some fifteen metres. Posts reach down to the ground as it
    is, so where it dips the posts are longer and the rail stays level.
    """
    path = resample(points)
    if len(path) < 2:
        return None
    if kind == "guardrail":
        path = _smooth(_smooth(path, 6), 6)
    direction, normal = _frames(path)
    actual = np.asarray(elevation(path[:, 0], path[:, 1]), float)
    if kind == "guardrail":
        face = road_side * normal
        inside = path + 0.6 * face
        level = _grade(elevation(inside[:, 0], inside[:, 1]), 5)
        beam = _strip(path + BEAM_OUT * face, level + BEAM_CENTRE, face, W_SECTION)
        posts = _posts(path, level, actual, direction, normal, POST_EVERY[kind], (0.05, 0.075), BEAM_CENTRE + 0.16)
        return {"beam": beam, "posts": posts}
    tall = float(np.clip(height, 1.0, 2.4))
    level = _grade(actual, 8)

    def tube(z, half):
        return _strip(path, level + z, normal, [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)])

    # A top rail, and the wires the mesh is tied to at the middle and the foot.
    rails = _merge([tube(tall, 0.025), tube(tall / 2, 0.012), tube(0.1, 0.012)])
    posts = _posts(path, level, actual, direction, normal, POST_EVERY[kind], (0.035, 0.035), tall + 0.05)
    return {"rails": rails, "posts": posts}


def write_obj(path, vertices, faces):
    """Flat-shaded OBJ: every triangle gets its own vertices and normal."""
    lines, count = [], 0
    for a, b, c in faces:
        n = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        length = np.linalg.norm(n)
        if length < 1e-12:
            continue
        n /= length
        for p in (a, b, c):
            lines.append(f"v {vertices[p][0]:.3f} {vertices[p][1]:.3f} {vertices[p][2]:.3f}\nvn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        count += 1
    with open(path, "w") as out:
        out.write("".join(lines))
        out.write("".join(f"f {3 * t + 1}//{3 * t + 1} {3 * t + 2}//{3 * t + 2} {3 * t + 3}//{3 * t + 3}\n" for t in range(count)))
    return count


def footprint(barriers, raster, shape, photo, sun_list):
    """Full-size mask of what each barrier left in the photo: itself and its shadow.

    As with vehicles, the shadow is the barrier's line carried away from the sun, tried for
    each of the two flights' suns, and the way that lands on darker ground is kept.
    """
    res = raster["res"]
    mask = np.zeros(shape, bool)
    for barrier, points in ((b, run) for b in barriers for run in seen_runs(b)):
        path = resample(points, 0.25)
        own_reach = 0.45 if barrier["type"] == "guardrail" else 0.3
        offsets = [(0.0, 0.0)]
        best = None
        for azimuth, elevation in sun_list:
            throw = barrier["height"] / np.tan(np.radians(elevation)) + 0.2
            away = np.radians(azimuth + 180.0)
            shifted = path + throw * np.array([np.sin(away), np.cos(away)])
            cols = np.clip(((shifted[:, 0] - raster["x0"]) / res).astype(int), 0, shape[1] - 1)
            rows = np.clip(((raster["y1"] - shifted[:, 1]) / res).astype(int), 0, shape[0] - 1)
            tone = float(np.asarray(photo[rows, cols], dtype=np.float32).max(-1).mean())
            if best is None or tone < best[0]:
                best = (tone, [(t * np.sin(away), t * np.cos(away)) for t in np.linspace(0.0, throw, 5)[1:]])
        offsets += best[1] if best else []
        for dx, dy in offsets:
            for x, y in path + np.array([dx, dy]):
                c = int((x - raster["x0"]) / res)
                r = int((raster["y1"] - y) / res)
                k = int(round(own_reach / res))
                mask[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1] = True
    return mask
