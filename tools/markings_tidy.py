"""Turn the paint as traced from the photo into paint as it was laid: straights, arcs, corners.

markings.py follows each painted line through the photo, and what it returns is true to the
photo: a line worn through in the middle comes back as two, a rectangle's sides stop a few
centimetres short of its corners, a curve carries the wobble of its pixels, and the dashes of
a dashed curve each point their own way. Paint is laid with a machine along a string or round
a pin, so every stroke here is refitted as what it must have been:

  join     pieces of one line that a worn gap separates become one stroke again
  fit      each stroke becomes straight segments and circular arcs meeting at sharp corners
  meet     ends that stop just short of a corner or of another line are carried on to it
  dashes   the dashes of a run are laid on one smooth curve through them, at one length

Drawn as crisp geometry, a line that is nearly straight looks worse than one that is.
"""
import numpy as np
from scipy.interpolate import UnivariateSpline

import markings_regular
import railmodel

STEP = 0.1            # metres between the points a stroke is examined at
CORNER = 40.0         # degrees of turn within 0.8 m that make a corner
STRAIGHT = 0.04       # metres a straight piece may stray from its line
ROUND = 0.035         # metres (rms) an arc may stray from its circle
GAP = 3.0             # metres: the longest worn gap joined over
GAP_SHARE = 0.35      # and no more than this share of the shorter piece, so dashes stay dashes
ALIGNED = 10.0        # degrees two pieces of one line may differ in direction
REACH = 0.45          # metres an end is carried on to meet a corner or another line
NUDGE = 0.12          # metres a dash may be moved to sit on its run's curve
LONG = 5.0            # metres: a stroke this long is a line, and may run on under a tree
UNDER = 30.0          # metres: the longest stretch of a line carried on where the photo could not show it
BARE = 2.0            # metres of bare road in plain view that such a stretch may have in it
SPECK = 1.5           # metres: a stroke shorter than this, alone in a tree's dapple, is a fleck of sun
ROAD_LINE = 2.0       # metres: on a road, a white stroke shorter than this is not a line


def _unit(v):
    return v / max(float(np.hypot(*v)), 1e-12)


def _angle(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(u), _unit(v)), -1.0, 1.0))))


def _line(points):
    """Best line through points: (centre, unit direction, largest distance from it)."""
    centre = points.mean(0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    u = vt[0]
    off = np.abs((points - centre) @ np.array([-u[1], u[0]]))
    return centre, u, float(off.max())


def _circle(points):
    """Best circle through points: (centre, radius, rms distance from it), or None if flat."""
    mid = points.mean(0)
    p = points - mid
    A = np.column_stack([2 * p, np.ones(len(p))])
    sol, *_ = np.linalg.lstsq(A, (p * p).sum(1), rcond=None)
    r2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
    if r2 <= 0 or r2 > 1e8:
        return None
    centre, r = sol[:2] + mid, float(np.sqrt(r2))
    return centre, r, float(np.sqrt(np.mean((np.hypot(*(points - centre).T) - r) ** 2)))


def _corners(path):
    """Indices of the sharp corners of a path sampled every STEP."""
    k, wide = int(round(0.4 / STEP)), int(round(1.2 / STEP))
    n = len(path)
    turn = np.zeros(n)
    for i in range(k, n - k):
        turn[i] = _angle(path[i] - path[i - k], path[i + k] - path[i])
    found = []
    for i in range(k, n - k):
        if turn[i] < CORNER or turn[i] < turn[max(i - k, 0):i + k + 1].max():
            continue
        # A corner turns all at once. An arc of small radius keeps turning: over a baseline
        # three times as long it has turned three times as far.
        a, b = max(i - wide, 0), min(i + wide, n - 1)
        if _angle(path[i] - path[a], path[b] - path[i]) < 1.7 * turn[i] and (not found or i - found[-1] > k):
            found.append(i)
    return found


def _piece(points):
    """One piece between corners as a primitive: ("line", a, b), ("arc", points) or ("free", points)."""
    centre, u, off = _line(points)
    if off <= STRAIGHT or len(points) < 6:
        along = (points - centre) @ u
        return ("line", centre + along[0] * u, centre + along[-1] * u)
    fit = _circle(points)
    if fit is not None and fit[2] <= ROUND:
        c, r, _ = fit
        angle = np.unwrap(np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0]))
        # Fine enough that the chords stay within 3 mm of the circle.
        count = max(int(np.ceil(abs(angle[-1] - angle[0]) / (2 * np.arccos(max(1 - 0.003 / r, -1.0))))), 2)
        t = np.linspace(angle[0], angle[-1], count + 1)
        return ("arc", c + r * np.stack([np.cos(t), np.sin(t)], axis=1))
    # Neither: an S or a spiral. Keep its shape and take out the wobble.
    t = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    smooth = np.stack([UnivariateSpline(t, points[:, d], k=3, s=len(points) * 0.02 ** 2)(t) for d in (0, 1)], axis=1)
    smooth[[0, -1]] = points[[0, -1]]
    return ("free", railmodel.simplify(smooth, 0.004))


def _cross(a0, a1, b0, b1):
    """Where the lines through a0-a1 and b0-b1 cross, or None if they are nearly parallel."""
    d1, d2 = a1 - a0, b1 - b0
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 0.3 * np.hypot(*d1) * np.hypot(*d2):
        return None
    t = ((b0[0] - a0[0]) * d2[1] - (b0[1] - a0[1]) * d2[0]) / det
    return a0 + t * d1


def fit(points):
    """A traced stroke as straight segments and arcs: the new list of points."""
    pts = np.asarray(points, float)
    length = railmodel.length_of(pts)
    if length < 0.3:
        return pts
    closed = length > 2.0 and np.hypot(*(pts[0] - pts[-1])) < 0.3
    path = railmodel.resample(pts, STEP)
    cuts = _corners(path)
    bounds = [0] + cuts + [len(path) - 1]
    pieces = [_piece(path[a:b + 1]) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]
    if not pieces:
        return pts

    def start(p):
        return p[1] if p[0] == "line" else p[1][0]

    def end(p):
        return p[2] if p[0] == "line" else p[1][-1]

    def body(p):
        return np.array([p[1], p[2]]) if p[0] == "line" else p[1]

    out = [body(pieces[0])]
    for prev, cur in zip(pieces[:-1], pieces[1:]):
        corner = (end(prev) + start(cur)) / 2
        if prev[0] == "line" and cur[0] == "line":
            x = _cross(prev[1], prev[2], cur[1], cur[2])
            if x is not None and np.hypot(*(x - corner)) < 0.5:
                corner = x
        out[-1][-1] = corner
        nxt = body(cur).copy()
        nxt[0] = corner
        out.append(nxt)
    if closed:
        first, last = pieces[0], pieces[-1]
        x = _cross(first[1], first[2], last[1], last[2]) if first[0] == "line" and last[0] == "line" else None
        corner = x if x is not None and np.hypot(*(x - out[0][0])) < 0.5 else (out[0][0] + out[-1][-1]) / 2
        out[0][0] = out[-1][-1] = corner
    line = np.concatenate([out[0]] + [o[1:] for o in out[1:]])
    keep = np.concatenate([[True], np.hypot(*np.diff(line, axis=0).T) > 1e-6])
    return line[keep]


def _ends(stroke):
    """The two ends of a stroke: (point, unit vector pointing out of the stroke), first then last."""
    p = np.asarray(stroke["points"], float)
    path = railmodel.resample(p, STEP) if len(p) > 2 else p
    k = min(int(round(1.0 / STEP)), len(path) - 1)
    return (path[0], _unit(path[0] - path[k])), (path[-1], _unit(path[-1] - path[-1 - k]))


def join(strokes):
    """Join pieces of one line across worn gaps. Returns (strokes, how many joins were made)."""
    strokes = [dict(s, points=[list(map(float, p)) for p in s["points"]]) for s in strokes]
    made = 0
    while True:
        ends = [(n, which, *e) for n, s in enumerate(strokes) for which, e in enumerate(_ends(s))]
        lengths = [railmodel.length_of(s["points"]) for s in strokes]
        best = None
        pts = np.array([e[2] for e in ends])
        for i, (n, wi, p, t) in enumerate(ends):
            gap = np.hypot(*(pts - p).T)
            for j in np.nonzero((gap < GAP) & (gap > 1e-9))[0]:
                m, wj, q, s = ends[j]
                if m <= n or strokes[m]["colour"] != strokes[n]["colour"] or strokes[m]["width"] != strokes[n]["width"]:
                    continue
                if gap[j] > max(0.12, GAP_SHARE * min(lengths[n], lengths[m])):
                    continue
                across = q - p
                if _angle(t, across) > ALIGNED or _angle(s, -across) > ALIGNED or _angle(t, -s) > ALIGNED:
                    continue
                if best is None or gap[j] < best[0]:
                    best = (gap[j], n, wi, m, wj)
        if best is None:
            return strokes, made
        _, n, wi, m, wj = best
        a = strokes[n]["points"] if wi == 1 else strokes[n]["points"][::-1]      # ends at the gap
        b = strokes[m]["points"] if wj == 0 else strokes[m]["points"][::-1]      # starts at the gap
        strokes[n] = dict(strokes[n], points=a + b)
        del strokes[m]
        made += 1


def carry(strokes, view):
    """Join the pieces of a line across what the photo could not show. Returns (strokes, joins made).

    A solid line that runs under a tree comes out of the trace in two pieces, one either side
    of the crown and its shadow. Two long strokes of one colour that point at each other
    across a gap are one line, unless the photo shows the road between them bare: a line
    that stops at a junction has clean concrete after it. view says, for an array of points,
    which are on or beside the pavement, and which of them are bare pavement in plain view.
    """
    strokes = [dict(s) for s in strokes]
    made = 0
    while True:
        lengths = [railmodel.length_of(s["points"]) for s in strokes]
        ends = [(n, which, *e) for n, s in enumerate(strokes) if lengths[n] >= LONG for which, e in enumerate(_ends(s))]
        best = None
        for i, (n, wi, p, t) in enumerate(ends):
            for m, wj, q, u in ends[i + 1:]:
                if m == n or strokes[m]["colour"] != strokes[n]["colour"] or strokes[m]["width"] != strokes[n]["width"]:
                    continue
                across = q - p
                gap = float(np.hypot(*across))
                if not (GAP < gap <= UNDER) or (best is not None and gap >= best[0]):
                    continue
                if _angle(t, across) > ALIGNED or _angle(u, -across) > ALIGNED or _angle(t, -u) > 1.5 * ALIGNED:
                    continue
                count = max(int(gap / 0.25), 2)
                road, bare = view(p + np.linspace(0.0, 1.0, count)[:, None] * across)
                if road.mean() < 0.95 or bare.sum() * gap / count > BARE:
                    continue
                best = (gap, n, wi, m, wj)
        if best is None:
            return strokes, made
        _, n, wi, m, wj = best
        a = list(strokes[n]["points"]) if wi == 1 else list(strokes[n]["points"])[::-1]      # ends at the gap
        b = list(strokes[m]["points"]) if wj == 0 else list(strokes[m]["points"])[::-1]      # starts at the gap
        strokes[n] = dict(strokes[n], points=a + b)
        del strokes[m]
        made += 1


def weed(strokes, unseen):
    """Drop the flecks of sun that were traced as paint. Returns (strokes, how many were dropped).

    Under a tree the pavement is a dapple of bright flecks, and the trace takes some of them
    for short strokes. A short stroke that lies in a relit shadow and is not one of a run of
    dashes is dropped. The same stroke out in the open is a tick mark, and stays.
    """
    in_run = {n for chain in markings_regular.runs(strokes, shortest=0.25) if len(chain) >= 3 for n in chain}
    kept = []
    for n, s in enumerate(strokes):
        p = np.asarray(s["points"], float)
        if n not in in_run and railmodel.length_of(p) < SPECK and unseen(railmodel.resample(p, 0.1)).mean() > 0.5:
            continue
        kept.append(s)
    return kept, len(strokes) - len(kept)


def roadworthy(strokes, where, crisp):
    """Drop the white strokes that are not paint from the roads. Returns (strokes, how many).

    On the skid pad and in the car parks white paint is everywhere and comes in every shape.
    On a road it comes as a line: something long and crisp. What the trace finds there
    besides is the joint down the middle of the slab, the pale rim of a stain, sand washed
    onto the edge, a dead branch. So away from the aprons a white stroke stays only if it is
    one of a run of dashes, or is ROAD_LINE long and passes crisp, a second and harder look
    at the photo. A stroke that is mostly off the pavement goes wherever it is.
    where(points) gives the share of some points that is on pavement and the share that is
    on an apron.
    """
    in_run = {n for chain in markings_regular.runs(strokes, shortest=0.25) if len(chain) >= 3 for n in chain}
    kept = []
    for n, s in enumerate(strokes):
        if s["colour"] == "white":
            pts = np.asarray(s["points"], float)
            paved, apron = where(railmodel.resample(pts, 0.1))
            if paved < 0.9:
                continue
            if apron <= 0.5 and n not in in_run and not (railmodel.length_of(pts) >= ROAD_LINE and crisp(pts)):
                continue
        kept.append(s)
    return kept, len(strokes) - len(kept)


def meet(strokes):
    """Carry ends on to the corner or the line they stop just short of. In place. Returns how many."""
    segs = []      # every segment of every stroke: (stroke, a, b)
    for n, s in enumerate(strokes):
        p = np.asarray(s["points"], float)
        segs += [(n, p[i], p[i + 1]) for i in range(len(p) - 1)]
    owner = np.array([s[0] for s in segs])
    same = {n: np.array([strokes[o]["colour"] == s["colour"] for o in owner]) for n, s in enumerate(strokes)}
    a = np.array([s[1] for s in segs])
    b = np.array([s[2] for s in segs])
    moved = 0
    for n, s in enumerate(strokes):
        p = np.asarray(s["points"], float)
        # Not closed outlines, which have no ends, and not dashes and ticks, which end where
        # they end.
        if len(p) < 2 or np.allclose(p[0], p[-1]) or railmodel.length_of(p) < 1.0:
            continue
        for at, inner in ((0, 1), (-1, -2)):
            tip, t = p[at], _unit(p[at] - p[inner])
            # Where the ray from the tip, straight on, meets a segment of another stroke.
            d = b - a
            det = t[0] * (-d[:, 1]) - t[1] * (-d[:, 0])
            ok = (owner != n) & same[n] & (np.abs(det) > 0.5 * np.hypot(*d.T))      # another line of this colour, at a fair angle
            with np.errstate(divide="ignore", invalid="ignore"):
                r = a - tip
                along = (r[:, 0] * (-d[:, 1]) - r[:, 1] * (-d[:, 0])) / det
                where = (t[0] * r[:, 1] - t[1] * r[:, 0]) / det
            slack = REACH / np.maximum(np.hypot(*d.T), 1e-9)
            ok &= (along > -0.05) & (along < REACH) & (where > -slack) & (where < 1 + slack)
            if not ok.any():
                continue
            k = int(np.where(ok, along, np.inf).argmin())
            # On to the other line's centre and half its width more, so the corner is filled.
            reach = along[k] + strokes[owner[k]]["width"] / 2
            s["points"][at] = [float(v) for v in tip + reach * t]
            moved += 1
    return moved


def dashes(strokes):
    """Lay the dashes of each run on one smooth curve, at one length. In place. Returns how many."""
    moved = 0
    for chain in markings_regular.runs(strokes, shortest=0.25):
        if len(chain) < 4:
            continue
        seg = np.array([strokes[n]["points"] for n in chain], float)
        centre = seg.mean(1)
        length = np.hypot(*(seg[:, 1] - seg[:, 0]).T)
        t = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(centre, axis=0).T))])
        c0, u, off = _line(centre)
        if off <= 0.03:
            new = c0 + ((centre - c0) @ u)[:, None] * u
            tangent = np.tile(u, (len(chain), 1))
        else:
            spline = [UnivariateSpline(t, centre[:, d], k=3, s=len(chain) * 0.025 ** 2) for d in (0, 1)]
            new = np.stack([sp(t) for sp in spline], axis=1)
            tangent = np.stack([sp.derivative()(t) for sp in spline], axis=1)
            tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
        usual = float(np.median(length))
        for i, n in enumerate(chain):
            if np.hypot(*(new[i] - centre[i])) > NUDGE:
                continue
            half = (usual if abs(length[i] - usual) <= 0.35 * usual else length[i]) / 2
            way = tangent[i] if tangent[i] @ (seg[i, 1] - seg[i, 0]) >= 0 else -tangent[i]
            strokes[n]["points"] = [[float(v) for v in new[i] - half * way], [float(v) for v in new[i] + half * way]]
            moved += 1
    return moved


def tidy(strokes, log=print, unseen=None, view=None, road=None):
    """Join, fit, meet and regularise. Returns the new list of strokes.

    unseen, if given, says which of an array of points lie in a tree's shadow or beside its
    crown: see weed. view is what carry needs to join a line across a gap. road is the pair
    (where, crisp) that roadworthy needs.
    """
    joined, made = join(strokes)
    under = flecks = stray = 0
    if road is not None:
        joined, stray = roadworthy(joined, *road)
    if unseen is not None:
        joined, flecks = weed(joined, unseen)
    if view is not None:
        joined, under = carry(joined, view)
    out = []
    for s in joined:
        pts = fit(s["points"])
        if len(pts) >= 2 and railmodel.length_of(pts) >= 0.2:
            out.append(dict(s, points=[[round(float(x), 3), round(float(y), 3)] for x, y in pts]))
    laid = dashes(out)
    carried = meet(out)
    for s in out:
        s["points"] = [[round(float(x), 3), round(float(y), 3)] for x, y in s["points"]]
    log(f"  road paint tidied: {made} worn gaps joined, {under} lines carried on under trees, {flecks} flecks of sun dropped,"
        f" {stray} white strokes on roads that are not paint dropped,"
        f" {laid} dashes laid on their run's curve, {carried} ends carried on to meet a line")
    return out
