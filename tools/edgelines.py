"""White lines along both edges of every road. They are not on the real track: an option.

The line follows the pavement's outline, a little way in from it, wherever the pavement is a
road and not an apron (the skid pad, a car park). Round a junction it follows the kerb from one
road into the other, as a painted edge line does. Where a guard rail stands on the pavement's
edge the line runs inside the rail.

The pavement's outline comes from a photograph and wanders by a few tenths of a metre. Paint
drawn as crisp geometry shows every wobble, so each line is smoothed over fifteen metres with
a filter that leaves straights straight and arcs round, and what is left of any notch or bump
in the outline is then bridged, or failing that left out of the line (kinks.py).
"""
import numpy as np
from scipy import ndimage, signal

import kinks
import pavement
import railmodel

WIDTH = 0.10        # metres, the painted width
INSET = 0.30        # metres from the pavement's edge to the line's centre
APRON = 7.0         # metres: pavement where a disc of this radius fits is an apron, not a road
STOP = 3.0          # metres short of an apron at which a line ends
RAIL = 0.45         # metres of pavement a guard rail takes up on either side of its own line
WINDOW = 15.0       # metres over which a line is smoothed
SPACING = 0.5       # metres between the points a line is smoothed through
SHORTEST = 8.0      # metres: shorter pieces are dropped
RAGGED = 0.15       # metres (rms) an outline may stray from its own smoothing and still be a kerb
GRID = 2            # the outline is traced on every GRID-th cell of the pavement map
CORNER = 75.0       # degrees a line may turn across a kink that is taken out of it
END = 5.0           # metres from a line's end within which a kink ends the line instead
BREAK = 15.0        # degrees: a kink this sharp that cannot be bridged is left out of the line
CLEAR = 2.0         # metres of line left out either side of it


def contours(field, level):
    """Lines along which field equals level, as a list of (N, 2) arrays of (row, column).

    Marching squares. A closed line repeats its first point at the end.
    """
    h, w = field.shape
    above = field > level
    code = (above[:-1, :-1] * 8 + above[:-1, 1:] * 4 + above[1:, 1:] * 2 + above[1:, :-1] * 1).astype(np.uint8)
    rows, cols = np.nonzero((code > 0) & (code < 15))
    kind = code[rows, cols]
    # A cell's four sides, each named by the grid edge it lies on: across edges first, then down.
    top, bottom = rows * w + cols, (rows + 1) * w + cols
    left, right = h * w + rows * w + cols, h * w + rows * w + cols + 1
    sides = {"t": top, "b": bottom, "l": left, "r": right}
    table = {1: ["lb"], 2: ["br"], 3: ["lr"], 4: ["tr"], 5: ["tl", "br"], 6: ["tb"], 7: ["tl"],
             8: ["tl"], 9: ["tb"], 10: ["tr", "lb"], 11: ["tr"], 12: ["lr"], 13: ["br"], 14: ["lb"]}
    a, b = [], []
    for value, pairs in table.items():
        sel = kind == value
        for pair in pairs:
            a.append(sides[pair[0]][sel])
            b.append(sides[pair[1]][sel])
    a, b = np.concatenate(a), np.concatenate(b)

    links = {}
    for p, q in zip(a.tolist(), b.tolist()):
        links.setdefault(p, []).append(q)
        links.setdefault(q, []).append(p)

    def point(edge):
        edge = np.asarray(edge)
        down = edge >= h * w
        e = np.where(down, edge - h * w, edge)
        i, j = e // w, e % w
        i2, j2 = np.where(down, i + 1, i), np.where(down, j, j + 1)
        f0, f1 = field[i, j], field[i2, j2]
        t = (level - f0) / np.where(f1 == f0, 1.0, f1 - f0)
        return np.stack([i + t * (i2 - i), j + t * (j2 - j)], axis=1)

    lines, used = [], set()
    ends = [e for e, n in links.items() if len(n) == 1]
    for start in ends + list(links):
        if start in used:
            continue
        chain, prev, cur = [start], None, start
        used.add(start)
        while True:
            onward = [n for n in links[cur] if n != prev and n not in used]
            if not onward:
                break
            prev, cur = cur, onward[0]
            chain.append(cur)
            used.add(cur)
        closed = len(chain) > 2 and chain[0] in links[chain[-1]]
        pts = point(chain)
        lines.append(np.concatenate([pts, pts[:1]]) if closed else pts)
    return lines


def _smooth(path, closed):
    """Even a line out over WINDOW, keeping straights straight and arcs round.

    A cubic fitted through the points of each window: a straight or an arc passes through it
    unchanged, and wobbles shorter than about half the window are taken out.
    """
    body = path[:-1] if closed else path
    count = min(int(WINDOW / SPACING) | 1, len(body) if len(body) % 2 else len(body) - 1)
    if count < 5:
        return path
    if closed:
        body = signal.savgol_filter(body, count, 3, axis=0, mode="wrap")
        return np.concatenate([body, body[:1]])
    out = signal.savgol_filter(body, count, 3, axis=0, mode="interp")
    out[[0, -1]] = path[[0, -1]]
    return out


def _run_on(run, clearance, whole):
    """Take the kinks out of a line. Returns the line as a list of pieces.

    The flare at a footpath's mouth, a jog where a kerb was patched, gravel spilt off a
    verge: the pavement's outline really does that, and a painted line does not follow it.
    Each kink is replaced by a curve across it, provided the curve stays clear of the
    pavement's edge. Where it would not, the line is broken instead: a gap in an edge line
    is an ordinary sight and a zigzag is not. So is a kink too near a line's end to bridge.
    clearance gives the distance in from the edge at each of some points.
    """
    path = run
    for _ in range(2):
        path = railmodel.resample(path, kinks.STEP)
        pieces, start = [], 0
        for a, b in kinks.spans(path, CORNER):
            if a < start:
                continue
            curve = kinks.bridge(path, a, b)
            if clearance(curve).min() < 0.5 * INSET or kinks.apart(path[a:b + 1], curve) > 2.5:
                continue
            pieces += [path[start:a], curve]
            start = b + 1
        if not pieces:
            break
        path = np.concatenate(pieces + [path[start:]])
    reach, margin = int(END / kinks.STEP), int(CLEAR / kinks.STEP)
    out = np.zeros(len(path), bool)
    for first, last, sharp in kinks.pairs(path):
        near_end = not whole and (first < reach or last >= len(path) - reach)
        if sharp >= BREAK or near_end:
            out[max(first - margin, 0):last + margin + 1] = True
    if not out.any():
        return [path]
    if whole:
        # Start a closed line inside a break, so that no piece runs across the seam.
        shift = int(np.argmax(out))
        path, out = np.concatenate([path[shift:-1], path[:shift + 1]]), np.concatenate([out[shift:-1], out[:shift + 1]])
    labels, count = ndimage.label(~out)
    return [path[labels == n] for n in range(1, count + 1)]


def find(distance, raster, scanned, rails, log=print):
    """The edge lines, as strokes: {"colour": "white", "width", "points"}.

    distance is the signed distance to the pavement's edge on pavement.RES cells, positive on
    the pavement. scanned marks, on the same cells, ground well inside what the scan covers:
    where the pavement runs off the scan its end is not an edge. rails is the list of guard
    rails found.
    """
    res = pavement.RES
    paved = distance > 0
    # Aprons: wherever a disc of radius APRON fits on the pavement, and everything it covers.
    core = distance >= APRON
    apron = paved & (ndimage.distance_transform_edt(~core) * res <= APRON + STOP) if core.any() else np.zeros_like(paved)

    usable = paved.copy()
    for rail in rails:
        path = railmodel.resample(rail["points"], res)
        r = np.clip(((raster["y1"] - path[:, 1]) / res).astype(int), 0, paved.shape[0] - 1)
        c = np.clip(((path[:, 0] - raster["x0"]) / res).astype(int), 0, paved.shape[1] - 1)
        line = np.zeros_like(paved)
        line[r, c] = True
        usable &= ndimage.distance_transform_edt(~line) * res > RAIL
    field = ((ndimage.distance_transform_edt(usable) - ndimage.distance_transform_edt(~usable)) * res).astype(np.float32)
    field = ndimage.gaussian_filter(field, 1.0)

    def at(grid, pts, order=1):
        r = (raster["y1"] - pts[:, 1]) / res - 0.5
        c = (pts[:, 0] - raster["x0"]) / res - 0.5
        return ndimage.map_coordinates(grid, [r, c], order=order, mode="nearest")

    wanted = (~apron & scanned).astype(np.float32)
    strokes = []
    for line in contours(field[::GRID, ::GRID], INSET):
        xy = np.stack([raster["x0"] + (line[:, 1] * GRID + 0.5) * res, raster["y1"] - (line[:, 0] * GRID + 0.5) * res], axis=1)
        closed = len(xy) > 3 and np.allclose(xy[0], xy[-1])
        if railmodel.length_of(xy) < SHORTEST:
            continue
        path = railmodel.resample(xy, SPACING)
        keep = at(wanted, path, order=0) > 0.5
        if closed and not keep.all() and keep.any():
            # Start a closed line at a point that is dropped, so no kept run straddles the seam.
            shift = int(np.argmin(keep))
            path = np.concatenate([path[shift:-1], path[:shift + 1]])
            keep = np.concatenate([keep[shift:-1], keep[:shift + 1]])
            closed = False
        labels, count = ndimage.label(keep)
        for n in range(1, count + 1):
            run = path[labels == n]
            whole = closed and count == 1 and keep.all()
            if railmodel.length_of(run) < SHORTEST:
                continue
            traced = run
            run = _smooth(run, whole)
            # A kerb is smooth. Where the outline as traced strays far from its own smoothing
            # it is the ragged edge of gravel or of worn-out pavement, and gets no line.
            # A tight corner strays too, for a few metres. Raggedness is judged twenty metres
            # at a time, and a piece is dropped when most of it is ragged.
            off2 = np.sum((run - traced) ** 2, axis=1)
            span = min(int(20.0 / SPACING), len(off2))
            local = np.sqrt(np.convolve(off2, np.ones(span) / span, mode="same"))
            if (local > RAGGED).mean() > 0.5:
                continue
            # Smoothing can carry a line toward the grass on the inside of a bend. Bring any
            # part that came too near the edge back in.
            for _ in range(3):
                d = at(field, run)
                near = d < 0.6 * INSET
                if not near.any():
                    break
                gx = (at(field, run + [0.2, 0.0]) - at(field, run - [0.2, 0.0])) / 0.4
                gy = (at(field, run + [0.0, 0.2]) - at(field, run - [0.0, 0.2])) / 0.4
                size = np.maximum(np.hypot(gx, gy), 1e-6)
                run[near] += ((INSET - d) / size)[near, None] * np.stack([gx / size, gy / size], axis=1)[near]
            pieces = _run_on(run, lambda p: at(field, p), whole)
            for piece in pieces:
                if len(piece) < 2 or railmodel.length_of(piece) < SHORTEST:
                    continue
                pts = railmodel.simplify(piece, 0.02)
                if whole and len(pieces) == 1:
                    pts = np.concatenate([pts[:-1], pts[:1]]) if np.allclose(pts[0], pts[-1]) else np.concatenate([pts, pts[:1]])
                strokes.append({"colour": "white", "width": WIDTH, "points": [[round(float(x), 3), round(float(y), 3)] for x, y in pts]})
    log(f"  edge lines: {len(strokes)} pieces, {sum(railmodel.length_of(s['points']) for s in strokes):.0f} m, along the roads only")
    return strokes
