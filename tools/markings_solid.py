"""Solid yellow lines, one line or a pair, each drawn from a single smooth curve.

A double yellow line is two lines a hand apart. At 5 cm a pixel the photo shows the pair as
one band of yellow about 40 cm wide where a single line shows about 20, and the trace follows
one line somewhere in that band, or both for a while, or neither for its whole length. Drawn
from the trace, a double line comes out single for most of its length, with a second line
that starts and stops beside it and is not quite parallel.

So a double line is not drawn from its trace. The trace says where to look. Every half metre
along it the photo is asked where the middle of the yellow is and how wide it is. The width
says whether the line is one or two, a stretch at a time, since paint does not change its
mind every metre. The middles give one smooth curve, and the line is drawn on that curve, or
the pair is drawn at equal distances either side of it: parallel because they are the same
curve.
"""
import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree

import markings_regular
import photoprofile
import railmodel

LONG = 5.0          # metres: a yellow stroke this long that is not one of a run of dashes is a solid line
BESIDE = 0.7        # metres to the side within which another yellow stroke is the same marking
ALIGNED = 20.0      # degrees it may differ in direction
STATION = 0.5       # metres between the places the photo is asked
ACROSS = 0.9        # metres either side that are looked at
FAINT = 5.0         # yellowness over the concrete beside it that still counts as paint
PAIR = 0.30         # metres: a band wider than this is two lines. Single lines measure 0.15 to 0.25, pairs 0.35 to 0.50
SEEN_LINE = 0.16    # metres one line measures, so a pair's lines are the band's width less this apart
SHARE = 0.35        # of the stations within WINDOW that must show a pair for the stretch to be one
WINDOW = 10.0       # metres either side over which that is judged
SHORTEST = 6.0      # metres: a shorter stretch takes the kind of the line around it
TRUE = 0.04         # metres (rms) the curve may leave the middles it is fitted through
STIFF = 0.10        # the same for the first, stiffer curve that shows which middles are wrong
WRONG = 0.08        # metres off that curve at which a middle is not believed
LEAVES = 0.08       # share of what lies across a station that may be leaf before it is not believed
WIDTH = 0.10        # metres, the drawn width of each line
APART = (0.18, 0.50)   # metres between the two lines of a pair, at the least and at the most
REACH = 60.0        # metres a line is followed past the end of its trace
LOST = 3.0          # metres of bare road in plain view at which it has ended
BLIND = 30.0        # metres it is carried on where the photo cannot show it


def _tangents(path):
    t = np.gradient(path, axis=0)
    return t / np.maximum(np.hypot(*t.T), 1e-9)[:, None]


def corridors(strokes):
    """The solid yellow lines, each with every traced piece of it: [(path, stroke indices)].

    The longest piece is the start. Any other yellow stroke that runs beside it for two
    metres, or for half its own length, is part of the same marking, and whatever of that
    stroke runs on past the path's end lengthens the path.
    """
    dashed = {n for chain in markings_regular.runs(strokes) if len(chain) >= 3 for n in chain}
    length = [railmodel.length_of(s["points"]) for s in strokes]
    yellow = [n for n, s in enumerate(strokes) if s["colour"] == "yellow" and n not in dashed and length[n] >= 1.0]
    used, out = set(), []
    for seed in sorted((n for n in yellow if length[n] >= LONG), key=lambda n: -length[n]):
        if seed in used:
            continue
        path = railmodel.resample(strokes[seed]["points"], 0.25)
        members = [seed]
        used.add(seed)
        grew = True
        while grew:
            grew = False
            tree, along = cKDTree(path), _tangents(path)
            for n in yellow:
                if n in used:
                    continue
                piece = railmodel.resample(strokes[n]["points"], 0.25)
                if len(piece) < 3:
                    continue
                gap, at = tree.query(piece)
                same_way = np.abs((_tangents(piece) * along[at]).sum(1)) > np.cos(np.radians(ALIGNED))
                beside = (gap < BESIDE) & same_way
                if beside.sum() * 0.25 < min(2.0, 0.5 * length[n]):
                    continue
                members.append(n)
                used.add(n)
                for end in (0, len(path) - 1):
                    past = np.nonzero(~beside & (at == end))[0]
                    # What runs on is one end of the piece, all of it together.
                    if len(past) * 0.25 < 0.5 or (np.diff(past) != 1).any() or (past[0] != 0 and past[-1] != len(piece) - 1):
                        continue
                    extra = piece[past]
                    if np.hypot(*(extra[0] - path[end])) > np.hypot(*(extra[-1] - path[end])):
                        extra = extra[::-1]
                    path = np.concatenate([extra[::-1], path]) if end == 0 else np.concatenate([path, extra])
                    grew = True
                    break
                if grew:
                    break       # the path has changed: start the search over it again
        out.append((railmodel.resample(path, 0.25), members))
    return out


def measure(path, photo, raster, hidden):
    """What the photo shows at stations along a path.

    Returns (stations, unit normals, middle of the yellow as an offset along the normal, its
    width, whether the photo showed it). hidden says which of some points the photo cannot
    show. A station under leaves is not shown either: leaves are yellow too.
    """
    offsets = np.arange(-ACROSS, ACROSS + 1e-9, 0.025)
    fine, normal, (yellow, leaf) = photoprofile.across(photo, raster, path, offsets, [photoprofile.yellowness, photoprofile.leafiness])
    step = int(round(STATION / photoprofile.FINE))
    index = np.arange(0, len(fine), step)
    middle, width = np.zeros(len(index)), np.zeros(len(index))
    known = ~np.asarray(hidden(fine[index]), bool)
    for k, i in enumerate(index):
        a, b = max(i - step, 0), min(i + step + 1, len(fine))
        found = photoprofile.band(yellow[a:b].mean(0), offsets, 0.5, 0.65, FAINT)
        if found is None or leaf[a:b].mean() > LEAVES:
            known[k] = False
        else:
            middle[k], width[k] = found[0], found[1]
    return fine[index], normal[index], middle, width, known


def follow(path, photo, raster, hidden):
    """The line beyond the last point of path, as far as the photo shows paint: points or None.

    A worn line drops out of the trace long before it drops out of the photo. From the end
    of the trace the line is followed half a metre at a time: where the road's own curve
    says the next point is, the photo is asked for yellow within a foot of it. It ends where
    the photo shows bare road for LOST metres. Under a crown it is carried on unseen.
    """
    trail = [np.asarray(p, float) for p in railmodel.resample(path[-int(6.0 / 0.25):], STATION)]
    start = len(trail)
    offsets = np.arange(-0.6, 0.6 + 1e-9, 0.025)
    sure = start            # how much of the trail has been confirmed by paint
    lost = blind = 0.0
    while (len(trail) - start) * STATION < REACH:
        recent = np.array(trail[-9:])
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(recent, axis=0).T))])
        ahead = [np.polyval(np.polyfit(along, recent[:, d], 2 if len(recent) >= 6 else 1), along[-1] + np.array([STATION, STATION + 0.1])) for d in (0, 1)]
        nxt = np.array([ahead[0][0], ahead[1][0]])
        way = np.array([ahead[0][1], ahead[1][1]]) - nxt
        way /= max(np.hypot(*way), 1e-9)
        left = np.array([-way[1], way[0]])
        segment = np.stack([nxt - 0.25 * way, nxt + 0.25 * way])
        try:
            _, _, (yellow, leaf) = photoprofile.across(photo, raster, segment, offsets, [photoprofile.yellowness, photoprofile.leafiness])
        except ValueError:
            break               # off the edge of the photo
        if bool(np.asarray(hidden(nxt[None]))[0]) or leaf.mean() > LEAVES:
            blind += STATION
            if blind > BLIND:
                break
            trail.append(nxt)
            continue
        found = photoprofile.band(yellow.mean(0), offsets, 0.3, 0.45, FAINT)
        if found is None:
            lost += STATION
            if lost > LOST:
                break
            trail.append(nxt)
            continue
        trail.append(nxt + found[0] * left)
        sure = len(trail)
        lost = blind = 0.0
    if (sure - start) * STATION < 2.0:
        return None
    return np.array(trail[:sure])


def _kinds(width, known):
    """Station by station, whether the line is a pair: judged a stretch at a time."""
    if not known.any():
        return np.zeros(len(width), bool)
    pair = (known & (width >= PAIR)).astype(np.float32)
    size = 2 * int(WINDOW / STATION) + 1
    seen = ndimage.uniform_filter1d(known.astype(np.float32), size, mode="constant")
    share = ndimage.uniform_filter1d(pair, size, mode="constant") / np.maximum(seen, 1e-6)
    # Where nothing was seen for the whole window, the nearest place that was says.
    at = np.nonzero(seen > 0)[0]
    kind = np.interp(np.arange(len(width)), at, share[at]) >= SHARE
    least = int(SHORTEST / STATION)
    for _ in range(4):
        runs = np.split(np.arange(len(kind)), np.nonzero(kind[1:] != kind[:-1])[0] + 1)
        short = [run for run in runs if len(run) < least]
        if len(runs) < 2 or not short:
            break
        run = min(short, key=len)
        kind[run] = not kind[run[0]]
    return kind


def _curve(points, true=TRUE):
    """A smooth curve through points half a metre apart: (points on it, unit normals)."""
    if len(points) < 6:
        line = np.linspace(points[0], points[-1], len(points))
        t = (points[-1] - points[0]) / max(np.hypot(*(points[-1] - points[0])), 1e-9)
        return line, np.tile([-t[1], t[0]], (len(points), 1))
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    fits = [UnivariateSpline(along, points[:, d], k=3, s=len(points) * true ** 2) for d in (0, 1)]
    curve = np.stack([f(along) for f in fits], axis=1)
    tangent = np.stack([f.derivative()(along) for f in fits], axis=1)
    tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
    return curve, np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


def redraw(strokes, photo, raster, hidden, log=print):
    """Replace the traced pieces of every double yellow line with the line as the photo shows it.

    A solid line that is single from end to end is left as traced and fitted.

    Returns (strokes, bands). bands are the stripes of the photo the lines were read from,
    as strokes with the width to paint out: wider than the trace knew, for a pair.
    """
    # Every double line is first followed past both ends of its trace. What is found is one
    # more piece of the line, like the traced ones, and may join two lines into one.
    followed, traced_length = [], 0.0
    for path, _ in corridors(strokes):
        _, _, _, width, known = measure(path, photo, raster, hidden)
        if known.sum() < 4 or not _kinds(width, known).any():
            continue
        traced_length += railmodel.length_of(path)
        for way in (path, path[::-1]):
            more = follow(way, photo, raster, hidden)
            if more is not None:
                followed.append({"colour": "yellow", "width": WIDTH, "followed": True, "points": more.tolist()})
    strokes = list(strokes) + followed

    gone, drawn, bands = set(), [], []
    metres = {False: 0.0, True: 0.0}
    for path, members in corridors(strokes):
        station, normal, middle, width, known = measure(path, photo, raster, hidden)
        if known.sum() < 4:
            continue                    # nothing to go on: the trace stays as it is
        kind = _kinds(width, known)
        if not kind.any():
            continue                    # one line throughout: the trace has it, corners and all
        for run in np.split(np.arange(len(kind)), np.nonzero(kind[1:] != kind[:-1])[0] + 1):
            pair = bool(kind[run[0]])
            # The middle of a pair is read where both lines show. Where one has worn away
            # the yellow that is left is one line, half the pair's width to the side.
            good = run[known[run] & ((width[run] >= PAIR) == pair)]
            if len(good) < 2:
                good = run[known[run]]
            if len(good) < 2:
                continue
            # A stiff curve first. A middle that stands off it was read from something else:
            # a leaf, the lip of a patch. The curve proper goes through the rest.
            offset = np.interp(run, good, middle[good])
            stiff, _ = _curve(station[run] + offset[:, None] * normal[run], STIFF)
            off = np.hypot(*(station[good] + middle[good][:, None] * normal[good] - stiff[good - run[0]]).T)
            if (off <= WRONG).sum() >= 2:
                good = good[off <= WRONG]
            offset = np.interp(run, good, middle[good])
            curve, across = _curve(station[run] + offset[:, None] * normal[run])
            if railmodel.length_of(curve) < 1.0:
                continue
            apart = float(np.clip(np.median(width[good]) - SEEN_LINE, *APART)) if pair else 0.0
            for side in ((0.5, -0.5) if pair else (0.0,)):
                line = railmodel.simplify(curve + side * apart * across, 0.005)
                drawn.append({"colour": "yellow", "width": WIDTH, "points": [[round(float(x), 3), round(float(y), 3)] for x, y in line]})
            bands.append({"colour": "yellow", "width": apart + SEEN_LINE + 0.12, "points": [[round(float(x), 3), round(float(y), 3)] for x, y in curve]})
            metres[pair] += railmodel.length_of(curve)
        gone.update(members)
    kept = [s for n, s in enumerate(strokes) if n not in gone and not s.get("followed")]
    lines = sum(1 for b in bands if b["width"] > SEEN_LINE + 0.12)
    log(f"  double yellow lines read from the photo: {metres[True]:.0f} m as pairs in {lines} lines and {metres[False]:.0f} m of the same lines single,"
        f" in place of {len(gone) - len(followed)} traced pieces. {metres[True] + metres[False] - traced_length:.0f} m of that is worn paint followed past the end of a trace")
    return kept + drawn, bands
