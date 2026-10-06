"""Put back the dashes of a dashed line that the search did not find.

A dashed line is painted by a machine: equal dashes at equal spacing along a line or an even
curve. markings.py finds each dash from the photo on its own, and misses one now and then,
under a tree's shadow or where the paint has worn. In a run of dashes that are there, a gap
of twice or three times the run's own spacing is one or two dashes missing, and they are put
back where the run says they are. Only lane lines are treated this way. The patterns on the
skid pad are laid out by hand for particular exercises, and a gap in one may be meant.
"""
import numpy as np
from scipy.interpolate import CubicSpline

DASH = (0.6, 4.5)      # metres: straight strokes of this length count as dashes
AHEAD = 40.0           # metres searched ahead of a dash for the next one of its line
SIDEWAYS = 0.35        # metres the next dash may sit off the line of this one, plus 3% of the distance
TURN = 6.0             # degrees the next dash may be turned against this one, plus half a degree per metre
ROAD = 4.0             # metres: only runs spaced at least this far apart are lane lines
MULTIPLES = (2, 3)     # gaps of this many spacings are filled
SLACK = 0.15           # how far from a whole multiple a gap may be, in spacings


def _dashes(strokes, shortest=None):
    """Index, centre, unit direction and length of every stroke that is a straight dash."""
    shortest = DASH[0] if shortest is None else shortest
    out = []
    for n, s in enumerate(strokes):
        p = np.asarray(s["points"], float)
        if len(p) != 2:
            continue
        d = p[1] - p[0]
        length = float(np.hypot(*d))
        if shortest <= length <= DASH[1]:
            out.append((n, p.mean(0), d / length, length))
    return out


def runs(strokes, shortest=None):
    """Chains of dashes that follow one another along a line: lists of stroke indices, in order.

    shortest is the least length that counts as a dash, DASH[0] if not given.
    """
    dashes = _dashes(strokes, shortest)
    if not dashes:
        return []
    centre = np.array([d[1] for d in dashes])
    heading = np.array([d[2] for d in dashes])
    colour = [strokes[d[0]]["colour"] for d in dashes]
    # A dash's direction has no sense, so each is searched from in both directions.
    chains, used = [], set()
    nexts = {}
    for i in range(len(dashes)):
        for sign in (1.0, -1.0):
            rel = centre - centre[i]
            along = rel @ (sign * heading[i])
            across = np.abs(rel @ np.array([-heading[i][1], heading[i][0]]))
            turned = np.degrees(np.arccos(np.clip(np.abs(heading @ heading[i]), 0, 1)))
            ok = (along > 1.0) & (along < AHEAD) & (across < SIDEWAYS + 0.03 * along) & (turned < TURN + 0.5 * along)
            ok &= np.array([c == colour[i] for c in colour])
            if ok.any():
                nexts.setdefault(i, []).append(int(np.where(ok, along, np.inf).argmin()))
    # Two dashes are neighbours when each is the other's nearest along the line.
    links = {i: [j for j in js if i in nexts.get(j, [])] for i, js in nexts.items()}
    for start in sorted(links):
        if start in used or len(links[start]) != 1:
            continue                      # start from an end of a run
        chain, prev, cur = [start], None, start
        used.add(start)
        while True:
            onward = [j for j in links.get(cur, []) if j != prev and j not in used]
            if not onward:
                break
            prev, cur = cur, onward[0]
            chain.append(cur)
            used.add(cur)
        if len(chain) >= 3:
            chains.append([dashes[k][0] for k in chain])
    return chains


def fill(strokes, allowed):
    """Return (strokes with the missing dashes of each run added, how many were added).

    allowed(x, y) says whether paint can be at a point: on the pavement, and not under something
    already accounted for.
    """
    added = []
    for chain in runs(strokes):
        pts = np.array([np.mean(strokes[n]["points"], axis=0) for n in chain])
        lengths = [float(np.hypot(*np.subtract(strokes[n]["points"][1], strokes[n]["points"][0]))) for n in chain]
        gaps = np.hypot(*np.diff(pts, axis=0).T)
        spacing = float(np.median(gaps[gaps <= 1.3 * gaps.min()]))
        if spacing < ROAD:
            continue        # a pattern painted on the skid pad, where a gap may be meant
        where = np.concatenate([[0.0], np.cumsum(gaps)])
        line = CubicSpline(where, pts)
        for k, gap in enumerate(gaps):
            ratio = gap / spacing
            count = int(round(ratio))
            if count not in MULTIPLES or abs(ratio - count) > SLACK:
                continue
            for m in range(1, count):
                at = where[k] + gap * m / count
                mid, along = line(at), line(at, 1)
                along = along / max(np.hypot(*along), 1e-9)
                half = 0.5 * float(np.median(lengths)) * along
                a, b = mid - half, mid + half
                if allowed(*mid) and allowed(*a) and allowed(*b):
                    added.append({"colour": strokes[chain[k]]["colour"], "width": strokes[chain[k]]["width"],
                                  "points": [[round(float(a[0]), 3), round(float(a[1]), 3)], [round(float(b[0]), 3), round(float(b[1]), 3)]], "inferred": True})
    return strokes + added, len(added)
