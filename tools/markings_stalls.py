"""Parking stalls: rows of equal lines at equal spacing, restored where the paint has faded.

Stall lines are the faintest paint on the site, white on concrete that the sun has bleached
nearly as white. Tracing finds some of each row, and often only part of a line. But a row of
stalls is the most regular thing there is: parallel lines, one pitch apart, all starting on one
line and ending on another. So each row is fitted as a row, and then the photo is asked a much
easier question than "where is there paint": "is there paint along this particular line".

A row takes in every traced line that lies on its lattice, however long the gap, and every
line between its first traced line and its last is drawn, at the row's full length: a row of
stalls has no line missing from its middle. Beyond the last line found, the row is followed
on for as long as the photo keeps showing lines.
"""
import numpy as np
from scipy import ndimage

LENGTH = (1.2, 7.0)       # metres: straight white strokes this long may be stall lines
PITCH = (2.3, 3.6)        # metres between neighbouring stall lines
PARALLEL = 4.0            # degrees two lines of a row may differ in direction
BESIDE = 3.5              # metres two neighbours' middles may be apart along their length
SHOWS = 0.30              # share of a line's visible length along which paint must show
BRIGHTER = 3.0            # grey levels by which paint must outshine the concrete beside it
BEYOND = 12               # stalls a row is followed past its last traced line, at most
FULL = 5.8                # metres: no stall line is carried on past this length
ON_LATTICE = 0.35         # metres off a row's lattice within which a traced line elsewhere along it is the row's
FAR = 30                  # stalls: the furthest such a line may be beyond the row's end
PITCH_ERROR = 0.03        # how far out a pitch fitted to three lines may be, as a share of itself
EDGE = 0.6                # metres of pavement a line added past a row's end must have on either side


def evidence(photo, raster, taken=None):
    """A function that asks the photo whether there is white paint along a segment.

    Returns shows(a, b) -> (share of the visible length along which paint shows, visible length
    in metres). Parts of the segment under something already accounted for (a parked car) are
    not visible. Paint shows where the line is brighter than the concrete 30 to 45 cm to either
    side of it, on average over half a metre.
    """
    res = raster["res"]

    def shows(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = float(np.hypot(*(b - a)))
        if length < 0.5:
            return 0.0, 0.0
        u = (b - a) / length
        normal = np.array([-u[1], u[0]])
        pts = a + np.linspace(0.0, 1.0, max(int(length / res), 4))[:, None] * (b - a)
        c0 = int((min(a[0], b[0]) - 1.0 - raster["x0"]) / res)
        c1 = int((max(a[0], b[0]) + 1.0 - raster["x0"]) / res) + 1
        r0 = int((raster["y1"] - max(a[1], b[1]) - 1.0) / res)
        r1 = int((raster["y1"] - min(a[1], b[1]) + 1.0) / res) + 1
        if r0 < 0 or c0 < 0 or r1 > photo.shape[0] or c1 > photo.shape[1]:
            return 0.0, 0.0
        window = np.asarray(photo[r0:r1, c0:c1], dtype=np.float32).mean(-1)

        def at(grid, p, order=1):
            return ndimage.map_coordinates(grid, [(raster["y1"] - p[:, 1]) / res - 0.5 - r0, (p[:, 0] - raster["x0"]) / res - 0.5 - c0], order=order)

        beside = np.stack([at(window, pts + off * normal) for off in (-0.45, -0.3, 0.3, 0.45)]).mean(0)
        contrast = ndimage.uniform_filter1d(at(window, pts) - beside, max(int(0.5 / res), 1), mode="nearest")
        visible = np.ones(len(pts), bool)
        if taken is not None:
            visible = at(np.asarray(taken[r0:r1, c0:c1], dtype=np.float32), pts, order=0) < 0.5
        if visible.sum() < 2:
            return 0.0, 0.0
        return float((contrast[visible] > BRIGHTER).mean()), float(visible.mean() * length)

    return shows


def _candidates(strokes):
    out = []
    for n, s in enumerate(strokes):
        p = np.asarray(s["points"], float)
        if s["colour"] != "white" or len(p) != 2:
            continue
        d = p[1] - p[0]
        length = float(np.hypot(*d))
        if LENGTH[0] <= length <= LENGTH[1]:
            u = d / length
            out.append((n, p.mean(0), u if (u[0], u[1]) > (0, 0) else -u, length))
    return out


def rows(strokes):
    """Groups of stroke indices that stand side by side like the lines of a row of stalls."""
    cand = _candidates(strokes)
    parent = list(range(len(cand)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (_, ci, ui, _) in enumerate(cand):
        for j in range(i + 1, len(cand)):
            _, cj, uj, _ = cand[j]
            if abs(ui @ uj) < np.cos(np.radians(PARALLEL)):
                continue
            rel = cj - ci
            across, along = abs(rel @ np.array([-ui[1], ui[0]])), abs(rel @ ui)
            if 0.3 * PITCH[0] < across < 3.2 * PITCH[1] and along < BESIDE:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(len(cand)):
        groups.setdefault(root(i), []).append(cand[i][0])
    return [g for g in groups.values() if len(g) >= 3]


def _fit_row(seg):
    """A row's direction, pitch and extent from its traced lines, or None if it is not regular."""
    d = seg[:, 1] - seg[:, 0]
    d[d @ d[0] < 0] *= -1
    u = d.sum(0) / np.hypot(*d.sum(0))
    n = np.array([-u[1], u[0]])
    s = seg.mean(1) @ n
    order = np.argsort(s)
    gaps = np.diff(s[order])
    single = [g / round(g / 2.75) for g in gaps if round(g / 2.75) >= 1 and PITCH[0] <= g / round(g / 2.75) <= PITCH[1]]
    if not single:
        return None
    pitch = float(np.median(single))
    k = np.round((s - s.min()) / pitch)
    fit = np.polyfit(k, s, 1) if len(set(k)) > 1 else None
    if fit is None or not (PITCH[0] <= fit[0] <= PITCH[1]):
        return None
    pitch, start = float(fit[0]), float(fit[1])
    off = np.abs(s - (start + k * pitch))
    if np.median(off) > 0.2:
        return None
    lo, hi = np.minimum(seg[:, 0] @ u, seg[:, 1] @ u), np.maximum(seg[:, 0] @ u, seg[:, 1] @ u)
    # The longest lines are the ones that have not faded: they say where the row starts and ends.
    full = (hi - lo) >= 0.8 * (hi - lo).max()
    return {"u": u, "n": n, "pitch": pitch, "start": start, "k": k.astype(int), "ok": off <= 0.4,
            "lo": float(np.median(lo[full])), "hi": float(np.median(hi[full]))}


def _refit(seg, pitch):
    """One lattice through all the traced lines of a row, however far apart: see _fit_row.

    The pitch is known roughly. Over a gap of ten stalls an error of three centimetres in it
    puts a line a foot out, so the pitch is searched for within four per cent of the rough
    one, and the one that leaves the lines nearest their places is taken.
    """
    d = seg[:, 1] - seg[:, 0]
    d[d @ d[int(np.argmax(np.hypot(*d.T)))] < 0] *= -1
    u = d.sum(0) / np.hypot(*d.sum(0))
    n = np.array([-u[1], u[0]])
    s = seg.mean(1) @ n
    best = None
    for p in np.linspace(0.96 * pitch, 1.04 * pitch, 81):
        k = np.round((s - s.min()) / p)
        off = s - s.min() - k * p
        # Every line counts, none for more than a foot: one stray stroke must not set the pitch,
        # and nor may a pitch be chosen that suits most lines and leaves the rest between places.
        miss = float(np.minimum(np.abs(off - np.median(off)), 0.3).mean())
        if best is None or miss < best[0] - 1e-6:
            best = (miss, p, k)
    _, pitch, k = best
    if len(set(k)) > 1:
        pitch, start = (float(v) for v in np.polyfit(k, s, 1))
    else:
        start = float(s.mean())
    lo, hi = np.minimum(seg[:, 0] @ u, seg[:, 1] @ u), np.maximum(seg[:, 0] @ u, seg[:, 1] @ u)
    full = (hi - lo) >= 0.8 * (hi - lo).max()
    return {"u": u, "n": n, "pitch": pitch, "start": start, "k": k.astype(int), "off": np.abs(s - (start + k * pitch)),
            "lo": float(np.median(lo[full])), "hi": float(np.median(hi[full]))}


def regularise(strokes, on_pavement, shows, log=print):
    """Refit every row of stalls. Returns (strokes, rows found, lines drawn, lines traced in them).

    A row is first found where three traced lines stand side by side. It then takes in every
    other traced line that lies on its lattice, however long the gap: one row of stalls along
    one kerb is one row, and the trace finds it in pieces with whole stretches missing where
    the paint has faded or a bush threw its shadow. Between the first traced line and the
    last, every line of the row is drawn: a row of stalls has no line missing from its
    middle. Beyond them the row is followed for as long as the photo keeps showing lines.
    """
    cand = {n: (centre, u) for n, centre, u, _ in _candidates(strokes)}
    found_rows = []
    for group in rows(strokes):
        row = _fit_row(np.array([strokes[n]["points"] for n in group], float))
        if row is None:
            continue
        members = [n for n, ok in zip(group, row["ok"]) if ok]
        if len(members) >= 3:
            found_rows.append((row["pitch"], members))
    found_rows.sort(key=lambda r: -len(r[1]))

    out, gone = [], set()
    found = drawn = traced = 0
    for pitch, members in found_rows:
        if any(n in gone for n in members):
            continue                          # already part of a longer row
        members = list(members)
        for _ in range(3):                    # each line taken in sharpens the lattice for the next
            row = _refit(np.array([strokes[n]["points"] for n in members], float), pitch)
            first, last = row["k"].min(), row["k"].max()
            more = []
            for n, (centre, u) in cand.items():
                if n in gone or n in members or abs(u @ row["u"]) < np.cos(np.radians(PARALLEL)):
                    continue
                if not (row["lo"] - 1.0 <= centre @ row["u"] <= row["hi"] + 1.0):
                    continue
                at = (centre @ row["n"] - row["start"]) / row["pitch"]
                # The pitch is known to a few per cent, so the further along, the more leeway.
                beyond = max(first - round(at), round(at) - last, 0)
                leeway = min(ON_LATTICE + PITCH_ERROR * row["pitch"] * beyond, 0.45 * row["pitch"])
                if abs(at - round(at)) * row["pitch"] <= leeway and beyond <= FAR:
                    more.append(n)
            if not more:
                break
            members += more
        keep_members = [n for n, off in zip(members, row["off"]) if off <= 0.4]
        have = sorted({int(k) for k, off in zip(row["k"], row["off"]) if off <= 0.4})
        if len(have) < 3:
            continue

        def at(k, a0, a1):
            s = row["start"] + k * row["pitch"]
            return s * row["n"] + a0 * row["u"], s * row["n"] + a1 * row["u"]

        # A whole row can have faded at one end. Carry both ends on, half a metre at a time,
        # while the photo shows paint on that half metre of at least a third of the row's
        # lines, up to the length of a stall.
        for end, way in (("hi", 1.0), ("lo", -1.0)):
            while row["hi"] - row["lo"] < FULL:
                edge = row[end]
                seen = [shows(*at(k, min(edge, edge + way * 1.0), max(edge, edge + way * 1.0))) for k in have]
                seen = [share for share, visible in seen if visible >= 0.5]
                if len(seen) < 2 or np.mean([share >= 0.5 for share in seen]) < 0.34:
                    break
                row[end] = edge + way * 0.5

        def line(k):
            return at(k, row["lo"], row["hi"])

        def paved(k):
            # A stall line ends at the kerb, and the kerb is where the pavement map is least
            # sure of itself: the line is asked for a little way in from each end.
            a, b = line(k)
            return all(on_pavement(*(a + t * (b - a))) for t in (0.15, 0.5, 0.85))

        def clear(k):
            # Past the last traced line a kerb passes for paint: bright, straight, and
            # parallel to the stalls. A line added there must have pavement on both sides.
            a, b = line(k)
            return all(on_pavement(*(a + t * (b - a) + side * row["n"])) for t in (0.15, 0.5, 0.85) for side in (-EDGE, 0.0, EDGE))

        keep = {k for k in range(have[0], have[-1] + 1) if paved(k)}
        for way in (1, -1):
            k, missed, unsure = (have[-1] if way > 0 else have[0]) + way, 0, []
            for _ in range(BEYOND):
                if not clear(k):
                    break                                   # the pavement has ended, and the row with it
                share, visible = shows(*line(k))
                if visible < 1.0:
                    unsure.append(k)                        # under a parked car: no telling yet
                elif share >= SHOWS:
                    keep.update(unsure + [k])
                    missed, unsure = 0, []
                else:
                    missed += 1
                    if missed == 2:
                        break
                k += way
        width = float(np.median([strokes[n]["width"] for n in keep_members]))
        for k in sorted(keep):
            a, b = line(k)
            out.append({"colour": "white", "width": width, "points": [[round(float(a[0]), 3), round(float(a[1]), 3)], [round(float(b[0]), 3), round(float(b[1]), 3)]]})
        gone.update(members)
        found += 1
        drawn += len(keep)
        traced += len(have)
    kept = [s for n, s in enumerate(strokes) if n not in gone]
    log(f"  parking stalls: {found} rows, {drawn} lines drawn where {traced} were traced, each at its row's full length")
    return kept + out, found, drawn, traced
