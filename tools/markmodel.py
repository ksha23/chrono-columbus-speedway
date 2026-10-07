"""Road paint as geometry: each painted line a flat ribbon laid on the pavement.

In the ground's photo a lane line is two texels wide, and seen from a car, at a shallow angle,
texture filtering smears it into the concrete. So the paint is taken out of the photo and drawn
as its own mesh instead, a couple of centimetres above the road, where it stays sharp at any
distance. markings.py finds the paint as strokes: a centre line, a width and a colour.

One mesh per colour, in scene coordinates. Metres, Z up.
"""
import numpy as np
from scipy.spatial import cKDTree

import ground

LIFT = 0.02        # metres above the road: enough to win the depth test, too little to see
STEP = 0.5         # metres between the points a ribbon follows the ground through
COLOURS = {"yellow": (0.86, 0.70, 0.16), "white": (0.93, 0.93, 0.90), "blue": (0.20, 0.36, 0.66)}


def _densify(points, step=STEP):
    """The polyline with extra points so no segment is longer than step. Corners are kept."""
    pts = np.asarray(points, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.ceil(np.hypot(*(b - a)) / step)), 1)
        out += [a + (b - a) * t / n for t in range(1, n + 1)]
    return np.array(out)


def _offsets(path, closed):
    """For each point, the vector to the ribbon's left edge for a half width of one.

    At a corner the two edges meet in a mitre, so a painted rectangle keeps square corners.
    """
    seg = np.diff(path, axis=0)
    seg /= np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-seg[:, 1], seg[:, 0]], axis=1)
    before = np.concatenate([normal[-1:] if closed else normal[:1], normal])
    after = np.concatenate([normal, normal[:1] if closed else normal[-1:]])
    mid = before + after
    mid /= np.maximum(np.linalg.norm(mid, axis=1, keepdims=True), 1e-9)
    # Never longer than three half widths: a hairpin would otherwise throw a spike.
    return mid / np.clip((mid * before).sum(1), 1 / 3, 1.0)[:, None]


class Road:
    """The road's own triangles, for laying paint on them and not on the surface they came from.

    Along the pavement's edge the ground mesh is cut, and a cut triangle is not the triangle
    ground.surface describes. On a bank the two differ by a couple of centimetres, which is
    all the clearance paint has.
    """

    def __init__(self, vertices, faces):
        self.corners = np.asarray(vertices, float)[np.asarray(faces)]
        centre = self.corners[:, :, :2].mean(1)
        self.reach = float(np.hypot(*(self.corners[:, :, :2] - centre[:, None, :]).transpose(2, 0, 1)).max()) + 1e-6
        self.tree = cKDTree(centre)

    def height(self, x, y, otherwise):
        """Height of the road under each point, or the matching value of otherwise where there is none."""
        out = np.array(otherwise, float)
        for n, near in enumerate(self.tree.query_ball_point(np.column_stack([x, y]), self.reach)):
            for j in near:
                a, b, c = self.corners[j]
                det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if abs(det) < 1e-12:
                    continue
                u = ((b[1] - c[1]) * (x[n] - c[0]) + (c[0] - b[0]) * (y[n] - c[1])) / det
                v = ((c[1] - a[1]) * (x[n] - c[0]) + (a[0] - c[0]) * (y[n] - c[1])) / det
                if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
                    out[n] = u * a[2] + v * b[2] + (1 - u - v) * c[2]
                    break
        return out


def make(strokes, ref, road=None):
    """{colour: (vertices, faces)} for a list of strokes, laid on the ground mesh.

    road, a Road, gives the height of the pavement's own triangles where there are any.
    """
    out = {}
    for colour in COLOURS:
        vertices, faces, base = [], [], 0
        for stroke in strokes:
            if stroke["colour"] != colour or len(stroke["points"]) < 2:
                continue
            pts = np.asarray(stroke["points"], float)
            closed = len(pts) > 3 and np.allclose(pts[0], pts[-1])
            path = _densify(pts)
            if len(path) < 2:
                continue
            side = _offsets(path, closed) * stroke["width"] / 2
            left, right = path + side, path - side
            edge = np.concatenate([left, right])
            z = ground.surface(ref, edge[:, 0], edge[:, 1])
            if road is not None:
                z = road.height(edge[:, 0], edge[:, 1], z)
            z = z + LIFT + stroke.get("raise", 0.0)       # paint on paint lies a little above it
            vertices.append(np.column_stack([edge, z]))
            n = len(path)
            for i in range(n - 1):
                a, b, c, d = base + i, base + i + 1, base + n + i + 1, base + n + i      # left i, left i+1, right i+1, right i
                faces += [(d, c, b), (d, b, a)]                                           # wound to face up
            base += 2 * n
        if vertices:
            out[colour] = (np.concatenate(vertices), np.array(faces))
    return out


def write_obj(path, vertices, faces):
    with open(path, "w") as out:
        out.write("".join(f"v {x:.3f} {y:.3f} {z:.4f}\n" for x, y, z in vertices))
        out.write("vn 0 0 1\n")
        out.write("".join(f"f {a}//1 {b}//1 {c}//1\n" for a, b, c in faces + 1))
    return len(faces)


def footprint(strokes, raster, shape, margin=0.04):
    """Full-size mask of the paint the strokes stand for, each a little wider than drawn."""
    res = raster["res"]
    mask = np.zeros(shape, bool)
    for stroke in strokes:
        half = stroke["width"] / 2 + margin
        path = _densify(stroke["points"], res)
        k = int(np.ceil(half / res))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disc = (xx * xx + yy * yy) * res * res <= half * half
        cols = ((path[:, 0] - raster["x0"]) / res).astype(int)
        rows = ((raster["y1"] - path[:, 1]) / res).astype(int)
        ok = (rows >= k) & (rows < shape[0] - k) & (cols >= k) & (cols < shape[1] - k)
        for r, c in zip(rows[ok], cols[ok]):
            mask[r - k:r + k + 1, c - k:c + k + 1] |= disc
    return mask
