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

import photoprofile
import railmodel

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
    side of it, on average over half a metre. shows.along(a, b) gives the same thing point by
    point: (distance along the segment, whether paint shows there, whether it is visible, how
    bright the concrete beside it is).
    """
    res = raster["res"]

    def along(a, b, smooth=0.5, brighter=BRIGHTER):
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = float(np.hypot(*(b - a)))
        empty = np.zeros(0), np.zeros(0, bool), np.zeros(0, bool), np.zeros(0)
        if length < 0.5:
            return empty
        u = (b - a) / length
        normal = np.array([-u[1], u[0]])
        count = max(int(length / res), 4)
        pts = a + np.linspace(0.0, 1.0, count)[:, None] * (b - a)
        c0 = int((min(a[0], b[0]) - 1.0 - raster["x0"]) / res)
        c1 = int((max(a[0], b[0]) + 1.0 - raster["x0"]) / res) + 1
        r0 = int((raster["y1"] - max(a[1], b[1]) - 1.0) / res)
        r1 = int((raster["y1"] - min(a[1], b[1]) + 1.0) / res) + 1
        if r0 < 0 or c0 < 0 or r1 > photo.shape[0] or c1 > photo.shape[1]:
            return empty
        window = np.asarray(photo[r0:r1, c0:c1], dtype=np.float32).mean(-1)

        def at(grid, p, order=1):
            return ndimage.map_coordinates(grid, [(raster["y1"] - p[:, 1]) / res - 0.5 - r0, (p[:, 0] - raster["x0"]) / res - 0.5 - c0], order=order)

        beside = np.stack([at(window, pts + off * normal) for off in (-0.45, -0.3, 0.3, 0.45)]).mean(0)
        contrast = ndimage.uniform_filter1d(at(window, pts) - beside, max(int(smooth / res), 1), mode="nearest")
        visible = np.ones(len(pts), bool)
        if taken is not None:
            visible = at(np.asarray(taken[r0:r1, c0:c1], dtype=np.float32), pts, order=0) < 0.5
        return np.linspace(0.0, length, count), contrast > brighter, visible, beside

    def shows(a, b):
        where, painted, visible, _ = along(a, b)
        if visible.sum() < 2:
            return 0.0, 0.0
        return float(painted[visible].mean()), float(visible.mean() * where[-1])

    def crisp(a, b, least=2 * BRIGHTER):
        """Is there a painted line along the segment: a narrow stripe brighter than what lies on BOTH sides of it?

        The lip of a kerb is brighter than the grass beside it and no brighter than the
        concrete, and along such an edge the test above says paint. This one does not.
        """
        offsets = np.arange(-0.8, 0.8 + 1e-9, 0.05)
        profile = photoprofile.across(photo, raster, np.stack([np.asarray(a, float), np.asarray(b, float)]), offsets, [photoprofile.brightness])[2][0].mean(0)
        found = photoprofile.band(profile, offsets, 0.1, 0.3, least)
        if found is None or found[1] > 0.35:
            return False
        top = profile[np.abs(offsets) <= 0.1].max()
        return bool(top - max(np.median(profile[offsets < -0.3]), np.median(profile[offsets > 0.3])) >= least)

    shows.along = along
    shows.crisp = crisp
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


def _fit_row(seg, pitches=None, usual=2.75):
    """A row's direction, pitch and extent from its traced lines, or None if it is not regular.

    pitches is the range the pitch may lie in, PITCH if not given, and usual a typical one.
    """
    pitches = pitches or PITCH
    d = seg[:, 1] - seg[:, 0]
    d[d @ d[0] < 0] *= -1
    u = d.sum(0) / np.hypot(*d.sum(0))
    n = np.array([-u[1], u[0]])
    s = seg.mean(1) @ n
    order = np.argsort(s)
    gaps = np.diff(s[order])
    single = [g / round(g / usual) for g in gaps if round(g / usual) >= 1 and pitches[0] <= g / round(g / usual) <= pitches[1]]
    if not single:
        return None
    pitch = float(np.median(single))
    k = np.round((s - s.min()) / pitch)
    fit = np.polyfit(k, s, 1) if len(set(k)) > 1 else None
    if fit is None or not (pitches[0] <= fit[0] <= pitches[1]):
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
            out.append({"colour": "white", "width": width, "stall": True, "points": [[round(float(a[0]), 3), round(float(a[1]), 3)], [round(float(b[0]), 3), round(float(b[1]), 3)]]})
        gone.update(members)
        found += 1
        drawn += len(keep)
        traced += len(have)
    kept = [s for n, s in enumerate(strokes) if n not in gone]
    log(f"  parking stalls: {found} rows, {drawn} lines drawn where {traced} were traced, each at its row's full length")
    return kept + out, found, drawn, traced


HATCH = (0.6, 1.6)        # metres between the stripes of a hatched area
STRIPE = 0.8              # metres: the shortest stripe
GAP = 0.3                 # metres of a stripe's line with no paint showing across which it is still carried on
NEAR_STALLS = 12.0        # metres from a stall line within which regular stripes are a hatched area
SAME_TONE = 15.0          # grey levels the concrete under a stripe may differ along it
SNAP = 0.3                # metres short of another line at which a stripe is carried on to it
OVER = 1.0                # metres across another line at which a stripe is cut back to it
OUTSIDE = 3               # stripes a hatched area is followed past its last traced one, at most


def _painted_run(shows, on_pavement, a, b, want, tone=None):
    """Where along the line from a to b the paint is: (start, end) in metres from a, or None.

    want is the stretch (start, end) that was traced, or None. Faded paint shows in pieces.
    Within the traced stretch that is taken on trust. Beyond it the stripe is carried on only
    while paint shows with no break longer than GAP: bright specks on bare concrete are a
    metre apart, and would carry a stripe across the whole car park. Off the pavement nothing
    is paint, and nor is anything on concrete of another tone than the stripes lie on: the
    footpath beside a car park is concrete too, and far paler. With nothing traced, the
    answer is the longest unbroken stretch there is. Returns (start, end, tone under it).
    """
    where, painted, visible, beside = shows.along(a, b, smooth=0.3)
    if len(where) < 4:
        return None
    step = where[1] - where[0]
    way = (np.asarray(b, float) - np.asarray(a, float)) / where[-1]
    paved = np.array([on_pavement(*(np.asarray(a, float) + t * way)) for t in where])
    if want is not None:
        tone = float(np.median(beside[(where >= want[0]) & (where <= want[1])]))
    if tone is not None:
        paved &= np.abs(beside - tone) < SAME_TONE
    solid = ndimage.binary_closing(painted & visible & paved, structure=np.ones(int(GAP / step) | 1))
    solid = ndimage.binary_opening(solid, structure=np.ones(max(int(0.15 / step), 1)))
    if want is not None:
        solid |= (where >= want[0]) & (where <= want[1])
    labels, count = ndimage.label(solid)
    best = None
    for n in range(1, count + 1):
        index = np.nonzero(labels == n)[0]
        start, end = where[index[0]], where[index[-1]]
        if end - start < STRIPE:
            continue
        score = min(end, want[1]) - max(start, want[0]) if want else end - start
        if score > 0 and (best is None or score > best[0]):
            best = (score, float(start), float(end), tone)
    return None if best is None else best[1:]


def hatching(strokes, on_pavement, shows, log=print):
    """Refit every hatched area: equal stripes at one pitch, each as long as its paint, and the edges they end on.

    The hatching beside an accessible stall and in the corner of a car park is the faintest
    paint of all and the most regular: parallel stripes a metre apart. Tracing finds a few of
    them, each broken off somewhere along its length. So the stripes are fitted as a lattice,
    like a row of stalls, and then each stripe's two ends are read from the photo along its
    own line, where the question is only how far the paint goes. An end that lands beside
    another painted line is carried onto it. Where the ends of neighbouring stripes line up
    and the photo shows paint between them, that is the edge of the hatched area, and it is
    drawn too. Hatching belongs to a car park: only stripes within NEAR_STALLS of a stall line
    are taken for it. The skid pad has rows of short parallel lines too, and they are rulers.
    Returns (strokes, areas found, stripes drawn).
    """
    out, gone = [], set()
    areas = drawn = 0
    stalls = np.array([np.mean(s["points"], axis=0) for s in strokes if s.get("stall")]).reshape(-1, 2)
    # The stall lines themselves are not stripes, though three stripes would fit between two of them.
    loose = [n for n, s in enumerate(strokes) if not s.get("stall")]
    for group in ([loose[i] for i in found] for found in rows([strokes[n] for n in loose])):
        seg = np.array([strokes[n]["points"] for n in group], float)
        row = _fit_row(seg, HATCH, 1.0)
        if row is None or not len(stalls) or np.hypot(*(stalls - seg.mean((0, 1))).T).min() > NEAR_STALLS:
            continue
        members = [n for n, ok in zip(group, row["ok"]) if ok]
        have = sorted({int(k) for k, ok in zip(row["k"], row["ok"]) if ok})
        if len(have) < 3 or np.diff(have).min() > 1:
            continue                    # stripes stand next to one another: two of them at least
        u, n_hat = row["u"], row["n"]
        traced = {}
        for n, k, ok in zip(group, row["k"], row["ok"]):
            if ok:
                p = np.asarray(strokes[n]["points"], float) @ u
                lo, hi = traced.get(int(k), (np.inf, -np.inf))
                traced[int(k)] = (min(lo, p.min()), max(hi, p.max()))
        low = min(lo for lo, _ in traced.values()) - 3.0
        high = max(hi for _, hi in traced.values()) + 3.0

        tones = []
        rest = [railmodel.resample(s["points"], 0.1) for m, s in enumerate(strokes) if m not in group and s["colour"] == "white" and not s.get("stall")]

        def along_line(k):
            """Does some other traced stroke run along the stripe at place k for half a metre?"""
            for path in rest:
                if len(path) < 3:
                    continue
                way = np.gradient(path, axis=0)
                way /= np.maximum(np.hypot(*way.T), 1e-9)[:, None]
                on = (np.abs(path @ n_hat - (row["start"] + k * row["pitch"])) < 0.2) & (np.abs(way @ u) > 0.9)
                if on.sum() * 0.1 >= 0.5:
                    return True
            return False

        def stripe(k):
            """The stripe at place k, as its two ends, or None if the photo shows none there."""
            base = (row["start"] + k * row["pitch"]) * n_hat
            want = traced.get(k)
            found = _painted_run(shows, on_pavement, base + low * u, base + high * u, None if want is None else (want[0] - low, want[1] - low),
                                 float(np.median(tones)) if tones else None)
            if found is None:
                return None if want is None else [base + want[0] * u, base + want[1] * u]
            if want is not None:
                tones.append(found[2])
            return [base + (low + found[0]) * u, base + (low + found[1]) * u]

        stripes = {k: stripe(k) for k in have}                  # the traced ones first: they say what tone the concrete is
        stripes.update({k: stripe(k) for k in range(have[0], have[-1] + 1) if k not in stripes})
        for way in (1, -1):
            k = (have[-1] if way > 0 else have[0]) + way
            for _ in range(OUTSIDE):
                found, before = stripe(k), stripes[k - way]
                # The photo alone is believed while the stripes keep their length. In a corner
                # they shorten fast, and there a piece of the trace has to lie along the line.
                if found is None or before is None:
                    break
                if np.hypot(*(found[1] - found[0])) < 0.6 * np.hypot(*(before[1] - before[0])) and not along_line(k):
                    break
                stripes[k] = found
                k += way
        # A hatched area has no stripe missing from its middle. One the photo does not show is
        # drawn as long as its neighbours say.
        known = sorted(k for k, v in stripes.items() if v is not None)
        for k in range(known[0], known[-1] + 1):
            if stripes.get(k) is None:
                before, after = max(j for j in known if j < k), min(j for j in known if j > k)
                t = (k - before) / (after - before)
                base = (row["start"] + k * row["pitch"]) * n_hat
                ends = [(1 - t) * (np.asarray(stripes[before][w]) @ u) + t * (np.asarray(stripes[after][w]) @ u) for w in (0, 1)]
                stripes[k] = [base + ends[0] * u, base + ends[1] * u]
        stripes = {k: v for k, v in stripes.items() if v is not None}
        if len(stripes) < 3:
            continue

        # A hatched area is a convex shape, so across the stripes their ends trace a curve
        # that never turns inward. A stripe that falls short of that curve has faded at its
        # end, and is carried on to it.
        order = sorted(stripes)
        for which, outward in ((0, -1.0), (1, 1.0)):
            reach = np.array([outward * (stripes[k][which] @ u) for k in order])
            hull = _over(np.array(order, float), reach)
            for k, old, new in zip(order, reach, hull):
                if new - old > 0.15:
                    stripes[k][which] = stripes[k][which] + outward * (new - old) * u

        # Another painted line beside a stripe's end is the line the stripe ends on.
        others = []
        for m, s in enumerate(strokes):
            if m in group or s["colour"] != "white":
                continue
            p = np.asarray(s["points"], float)
            others += [(p[i], p[i + 1]) for i in range(len(p) - 1) if np.hypot(*(p[i + 1] - p[i])) > 0.5]
        free = {}
        for k, ends in stripes.items():
            for which, outward in ((0, -u), (1, u)):
                move = _stop(ends[which], outward, np.hypot(*(ends[1] - ends[0])), others)
                if move is None:
                    free[(k, which)] = ends[which]
                else:
                    ends[which] = ends[which] + move * outward

        # Edges nobody traced: where the free ends of neighbouring stripes line up, and paint shows between them.
        width = float(np.median([strokes[n]["width"] for n in members]))
        edges = []
        order = sorted(stripes)
        for which in (0, 1):
            run = []
            for k, nxt in zip(order[:-1], order[1:]):
                a, b = free.get((k, which)), free.get((nxt, which))
                joined = a is not None and b is not None and nxt == k + 1
                if joined:
                    # The ends overshoot the edge they stop at by a little, or fall short of
                    # it. The edge is looked for a step at a time along the stripes.
                    reach = (b - a) / max(np.hypot(*(b - a)), 1e-9)
                    best = (0.0, 0.0)
                    for slide in np.arange(-0.6, 0.31, 0.05):
                        where, painted, visible, beside = shows.along(a - 0.2 * reach + slide * u, b + 0.2 * reach + slide * u, brighter=2 * BRIGHTER)
                        if len(where) and visible.mean() > 0.8 and tones and np.median(np.abs(beside - np.median(tones))) < SAME_TONE / 2:
                            best = max(best, (float(painted.mean()), float(slide)))
                    joined = best[0] >= 0.8 and shows.crisp(a + best[1] * u, b + best[1] * u)
                    if joined:
                        a, b = a + best[1] * u, b + best[1] * u
                if joined and run and abs(_turn(run[-1] - run[-2], b - a)) < 12.0:
                    run.append(b)
                else:
                    if len(run) >= 2:
                        edges.append(np.array(run))
                    run = [a, b] if joined else []
            if len(run) >= 2:
                edges.append(np.array(run))
        for edge in edges:
            # One straight line through the ends it joins, and each of those stripes carried onto it.
            centre = edge.mean(0)
            _, _, axes = np.linalg.svd(edge - centre, full_matrices=False)
            way = axes[0]
            reach = (edge - centre) @ way
            # The edge runs on past the stripes it was found between, to the corners of the area.
            far = 4.0
            found = _painted_run(shows, on_pavement, centre + (reach.min() - far) * way, centre + (reach.max() + far) * way, (far, far + reach.max() - reach.min()))
            lo, hi = (reach.min(), reach.max()) if found is None else (reach.min() - far + found[0], reach.min() - far + found[1])
            a, b = centre + (lo - width / 2) * way, centre + (hi + width / 2) * way
            for which, outward in ((0, -way), (1, way)):
                move = _stop(a if which == 0 else b, outward, np.hypot(*(b - a)), others)
                if move is not None:
                    a, b = (a + move * outward, b) if which == 0 else (a, b + move * outward)
            out.append({"colour": "white", "width": width, "points": [[round(float(v), 3) for v in a], [round(float(v), 3) for v in b]]})
            cross = u[0] * way[1] - u[1] * way[0]
            for (k, which), end in free.items():
                if np.hypot(*(centre + ((end - centre) @ way) * way - end)) <= SNAP and abs(cross) > 0.2:
                    along = ((centre - end)[0] * way[1] - (centre - end)[1] * way[0]) / cross
                    stripes[k][which] = end + along * u

        lines = []
        for k in sorted(stripes):
            a, b = stripes[k]
            if np.hypot(*(b - a)) < STRIPE / 2:
                continue
            lines.append((a, b))
            out.append({"colour": "white", "width": width, "points": [[round(float(v), 3) for v in a], [round(float(v), 3) for v in b]]})
        gone.update(members)
        # A stroke that mostly runs along one of the stripes is that stripe, traced crooked.
        for m, s in enumerate(strokes):
            if m in gone or m in group or s["colour"] != "white":
                continue
            path = railmodel.resample(s["points"], 0.1)
            if not (0.5 <= railmodel.length_of(path) <= 8.0):
                continue
            near = np.zeros(len(path), bool)
            for a, b in lines:
                d = b - a
                t = np.clip(((path - a) @ d) / (d @ d), 0.0, 1.0)
                near |= np.hypot(*(a + t[:, None] * d - path).T) < 0.25
            if near.mean() >= 0.4:
                gone.add(m)
        areas += 1
        drawn += len(lines)
    kept = [s for n, s in enumerate(strokes) if n not in gone]
    log(f"  hatched areas: {areas}, {drawn} stripes drawn, each as long as its paint, with {len(out) - drawn} edges nobody traced")
    return kept + out, areas, drawn


def _stop(end, outward, length, others):
    """How far to move a line's end along outward so that it stops on the line it ends at, or None.

    others is a list of segments (a, b). Short of one by a little, the line is carried on to
    it. Across one by as much as OVER, it is cut back: a stripe does not cross its edge. A
    segment that runs the line's own way is no edge to it.
    """
    best = None
    for a, b in others:
        d = b - a
        span = np.hypot(*d)
        if abs(d @ outward) > 0.94 * span:
            continue
        cross = outward[0] * d[1] - outward[1] * d[0]
        move = ((a - end)[0] * d[1] - (a - end)[1] * d[0]) / cross
        t = ((end + move * outward - a) @ d) / span
        if not (-SNAP <= t <= span + SNAP) or not (-OVER <= move <= SNAP) or -move > 0.6 * length:
            continue
        if best is None or abs(move) < abs(best):
            best = float(move)
    return best


def _over(x, y):
    """The least curve over the points (x, y) that never turns upward: their upper hull, at each x."""
    hull = []
    for point in sorted(zip(x.tolist(), y.tolist())):
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (point[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (point[0] - hull[-2][0]) >= 0:
            hull.pop()
        hull.append(point)
    hx, hy = np.array(hull).T
    return np.interp(x, hx, hy)


def _turn(a, b):
    """The angle in degrees from direction a to direction b."""
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))

