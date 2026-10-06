"""The perimeter fence as one loop, from the stretches of it that can be seen.

The site has a single fence that wraps the whole property. barriers.py finds the stretches
that show in the photo. Here they are put in order around the site and joined into one closed
line. Between two seen stretches the fence is carried straight on, and where a straight line
would run onto the pavement it is taken round the pavement's edge instead, because the fence
stays outside everything paved. The one place it must cross pavement is where the site road
passes through it, and that crossing is reported as a gate.
"""
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

import barriers_lines as lines
from barriers_lines import RES, length_of

CELL = 1.0                # metres, the grid inferred stretches are routed on
CLEAR = 3.0               # metres an inferred stretch keeps from the pavement where it can
PAVED_COST = 30.0         # what a metre across pavement costs, in metres of detour
NEAR_COST = 3.0           # extra cost of a metre right at the pavement's edge
BLANK_COST = 3.0          # cost of a metre where the scan has nothing
ROAD = 2.0                # metres of pavement crossed that make a gate
INSIDE = 1.0              # metres inside the pavement's mapped edge that count as on it
POST_SNAP = 3.0           # metres within which a gate post is the end of a seen stretch
SEEN_WITHIN = 0.6         # metres from seen fence that makes a point of the loop seen
FIT = 0.25                # metres, tolerance of the reported polyline along seen stretches
CORNER_TURN = 15.0        # degrees between two stretches that make a hidden corner worth placing
CORNER_REACH = 1.5        # a stretch is carried on to a corner at most this many times the gap
ROUTE_FIT = 1.0           # metres, tolerance along routed stretches


def whitening(pavement, grid):
    """Centre and transform that make the long, thin site roughly round, from where the
    pavement lies. Angles around the centre then put things in order around the site."""
    rows, cols = np.nonzero(pavement[::10, ::10])
    x, y = lines.to_scene(grid, rows * 10, cols * 10)
    pts = np.stack([x, y], axis=1)
    centre = pts.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((pts - centre).T))
    return centre, (vectors / np.sqrt(values)).T


def in_order(pieces, weights, centre, transform):
    """Sort seen stretches by angle around the site and turn each to run the same way round.

    Where two stretches cover the same angles they are the same fence seen twice, and the one
    with less of it seen is dropped. Returns the kept stretches, counter-clockwise.
    """
    def angle(p):
        q = (p - centre) @ transform.T
        return np.unwrap(np.arctan2(q[:, 1], q[:, 0]))

    spans = []
    for k, p in enumerate(pieces):
        a = angle(p)
        if a[-1] < a[0]:
            p, a = p[::-1], a[::-1]
        mid = (a[0] + a[-1]) / 2
        shift = np.round(mid / (2 * np.pi)) * 2 * np.pi
        spans.append((mid - shift, a[0] - shift, a[-1] - shift, weights[k], k, p))
    spans.sort(key=lambda s: (s[0], s[4]))
    kept = []
    for span in spans:
        while kept:
            last = kept[-1]
            overlap = min(span[2], last[2]) - max(span[1], last[1])
            if overlap <= 0.5 * min(span[2] - span[1], last[2] - last[1]):
                break
            if span[3] <= last[3]:
                span = None
                break
            kept.pop()
        if span is not None:
            kept.append(span)
    ordered = [s[5] for s in kept]
    # A stretch that points at the centre has no way round of its own: turn it whichever way
    # makes the joins to its neighbours shorter.
    n = len(ordered)
    for k in range(n):
        if n < 3:
            break
        before, after, p = ordered[k - 1][-1], ordered[(k + 1) % n][0], ordered[k]
        as_is = np.hypot(*(p[0] - before)) + np.hypot(*(after - p[-1]))
        turned = np.hypot(*(p[-1] - before)) + np.hypot(*(after - p[0]))
        if turned < as_is - 1e-9:
            ordered[k] = p[::-1]
    return ordered


class Router:
    """Cheapest way between two points for a fence nobody can see: straight where the ground
    is open, round the edge of the pavement where it is not."""

    def __init__(self, rasters, fields):
        self.grid = rasters["grid"]
        k = int(round(CELL / RES))
        h, w = rasters["pavement"].shape[0] // k, rasters["pavement"].shape[1] // k
        block = lambda a: a[:h * k, :w * k].reshape(h, k, w, k)
        paved = block(rasters["pavement"]).any(axis=(1, 3))
        blank = block(np.isnan(rasters["height"])).all(axis=(1, 3))
        near = np.clip(1 - block(fields["from_pavement"]).min(axis=(1, 3)) / CLEAR, 0, 1)
        self.cost = 1 + PAVED_COST * paved + NEAR_COST * near * ~paved + (BLANK_COST - 1) * (blank & ~paved)
        self.into = fields["into_pavement"]
        self.shape = rasters["pavement"].shape

    def cell(self, point):
        row, col = lines.to_cell(self.grid, point[0], point[1])
        k = CELL / RES
        return (int(np.clip((row + 0.5) // k, 0, self.cost.shape[0] - 1)),
                int(np.clip((col + 0.5) // k, 0, self.cost.shape[1] - 1)))

    def paved_runs(self, points):
        """Stretches of a polyline that lie on pavement: list of (start, end) scene points."""
        dense = lines.resample(points, RES)
        row, col = lines.cells(self.grid, self.shape, dense)
        on = self.into[row, col] > INSIDE         # the mapped edge is a metre uncertain
        edges = np.flatnonzero(np.diff(np.concatenate([[False], on, [False]]).astype(int)))
        back = int(round(INSIDE / RES))               # step back out to the edge itself
        return [(dense[max(a - back, 0)], dense[min(b - 1 + back, len(dense) - 1)])
                for a, b in zip(edges[::2], edges[1::2]) if (b - a) * RES >= ROAD]

    def path(self, a, b, out_of_a=None, out_of_b=None):
        """Polyline from a to b for a stretch of fence nobody saw.

        Straight if that keeps off the pavement. Where the seen stretches either side run in
        different directions (out_of_a, out_of_b: unit vectors pointing out of their ends) the
        two are carried on to where they meet, since a fence runs straight and turns at a
        post. Otherwise the cheapest way round the pavement.
        """
        straight = np.array([a, b])
        if not self.paved_runs(straight):
            if out_of_a is None or out_of_b is None:
                return straight
            gap = float(np.hypot(*(b - a)))
            cross = out_of_a[0] * out_of_b[1] - out_of_a[1] * out_of_b[0]
            turn = np.degrees(np.arccos(float(np.clip(-np.dot(out_of_a, out_of_b), -1, 1))))
            if turn < CORNER_TURN or abs(cross) < 1e-6:
                return straight
            along_a = ((b - a)[0] * out_of_b[1] - (b - a)[1] * out_of_b[0]) / cross
            along_b = ((b - a)[0] * out_of_a[1] - (b - a)[1] * out_of_a[0]) / cross
            reach = CORNER_REACH * gap
            if not (0 < along_a <= reach and 0 < along_b <= reach):
                return straight
            cornered = np.array([a, a + out_of_a * along_a, b])
            return straight if self.paved_runs(cornered) else cornered
        (r0, c0), (r1, c1) = self.cell(a), self.cell(b)
        margin = int(max(60.0, 0.5 * np.hypot(*(b - a))) / CELL)
        top, left = max(0, min(r0, r1) - margin), max(0, min(c0, c1) - margin)
        cost = self.cost[top:max(r0, r1) + margin + 1, left:max(c0, c1) + margin + 1]
        h, w = cost.shape
        index = np.arange(h * w).reshape(h, w)
        src, dst, weight = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            one = index[:h - dr, max(0, -dc):w - max(0, dc)]
            two = index[dr:, max(0, dc):w - max(0, -dc)]
            step = np.hypot(dr, dc) * CELL
            src.append(one.ravel())
            dst.append(two.ravel())
            weight.append((cost.ravel()[one.ravel()] + cost.ravel()[two.ravel()]) / 2 * step)
        graph = sparse.coo_matrix((np.concatenate(weight), (np.concatenate(src), np.concatenate(dst))),
                                  shape=(h * w, h * w)).tocsr()
        start, end = index[r0 - top, c0 - left], index[r1 - top, c1 - left]
        _, before = csgraph.dijkstra(graph, directed=False, indices=start, return_predecessors=True)
        cells = [end]
        while cells[-1] != start and before[cells[-1]] >= 0:
            cells.append(before[cells[-1]])
        cells = np.array(cells[::-1])
        k = CELL / RES
        x, y = lines.to_scene(self.grid, (cells // w + top + 0.5) * k - 0.5, (cells % w + left + 0.5) * k - 0.5)
        route = np.concatenate([[a], np.stack([x, y], axis=1)[1:-1], [b]])
        return lines.simplify(route, ROUTE_FIT)


def loop(pieces, weights, rasters, fields):
    """One closed fence line through the seen stretches.

    pieces are the seen stretches as point runs, weights how much of each was seen. Returns
    {"points": closed polyline, "seen": one flag per segment, "gates": pairs of posts}, or
    None if there is too little to go on.
    """
    if len(pieces) < 3:
        return None
    centre, transform = whitening(rasters["pavement"], rasters["grid"])
    ordered = in_order(pieces, weights, centre, transform)
    if len(ordered) < 3:
        return None
    router = Router(rasters, fields)
    evidence = cKDTree(np.concatenate([lines.resample(p, 0.25) for p in ordered]))

    points, gates = [], []
    for k, piece in enumerate(ordered):
        following = ordered[(k + 1) % len(ordered)]
        fitted = lines.fit_polyline(piece, FIT)
        after = lines.fit_polyline(following, FIT)
        link = router.path(fitted[-1], after[0], lines.end_direction(piece, True, span=10.0),
                           lines.end_direction(following, False, span=10.0))
        for first, last in router.paved_runs(link):
            posts = []
            for post, end in ((first, fitted[-1]), (last, after[0])):
                posts.append(end if np.hypot(*(post - end)) <= POST_SNAP else post)
            gates.append(posts)
        points.extend(fitted)
        points.extend(link[1:-1])
    points.append(points[0])
    points = np.array(points)
    keep = np.concatenate([[True], np.hypot(*np.diff(points, axis=0).T) > 1e-6])
    points = points[keep]

    seen = []
    for a, b in zip(points[:-1], points[1:]):
        along = lines.resample(np.array([a, b]), 0.25)
        distance, _ = evidence.query(along)
        seen.append(bool((distance <= SEEN_WITHIN).mean() >= 0.5))
    return {"points": points, "seen": seen, "gates": gates}


def seen_length(fence):
    """(metres seen, metres in all) of a loop returned by loop()."""
    seg = np.hypot(*np.diff(np.asarray(fence["points"], float), axis=0).T)
    return float(seg[np.asarray(fence["seen"], bool)].sum()), float(seg.sum())
