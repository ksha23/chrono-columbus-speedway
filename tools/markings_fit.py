"""Turn a rough line (a chain of ridge points) into a measured stroke: centre line, ends, width, colour.

Both fits work the same way. Lay a strip along the current guess, read where the paint's crest
sits in each row and where the paint starts and stops, move the guess onto the crests, and
repeat until it no longer moves. A straight stroke is a line fitted through the crests, each
row weighted by how much paint it shows. A curved one is a smoothing spline through them,
grown from the rough line outwards a few pixels at a time for as long as the paint goes on.
"""
import numpy as np
from scipy.interpolate import splev, splprep

from markings_strip import measure, profile, runs, sample, weighted_line


class Ground:
    """The photo, and where on it the ground is visible pavement."""

    def __init__(self, photo, paved, taken, height, scale, shadow=None):
        self.photo, self.paved, self.taken, self.height, self.scale = photo, paved, taken, height, scale
        self.shadow = shadow
        self.hide = None                                 # pixels under paint of the other colour
        self.used = None                                 # pixels already described by a stroke

    def _cells(self, xy, shape):
        c = np.rint(xy[:, 0]).astype(np.int64)
        r = np.rint(xy[:, 1]).astype(np.int64)
        inside = (r >= 0) & (c >= 0) & (r < shape[0]) & (c < shape[1])
        return np.where(inside, r, 0), np.where(inside, c, 0), inside

    def known(self, xy, standing=True):
        r, c, inside = self._cells(xy, self.photo.shape)
        pr = np.minimum(r // 2, self.paved.shape[0] - 1)
        pc = np.minimum(c // 2, self.paved.shape[1] - 1)
        hr = np.minimum(r // self.scale, self.height.shape[0] - 1)
        hc = np.minimum(c // self.scale, self.height.shape[1] - 1)
        ok = inside & np.asarray(self.paved[pr, pc]) & ~np.asarray(self.taken[r, c])
        if self.hide is not None:
            ok &= ~self.hide[r, c]
        return ok & ~(np.asarray(self.height[hr, hc]) > 0.4) if standing else ok

    def free(self, xy):
        """Rows of a strip that no stroke has claimed yet."""
        if self.used is None:
            return np.ones(len(xy), bool)
        r, c, inside = self._cells(xy, self.used.shape)
        return ~(self.used[r, c] & inside)

    def shade(self, xy):
        """How much of a line lies in cast shadow, 0 to 1."""
        if self.shadow is None:
            return 0.0
        r, c, inside = self._cells(xy, self.shadow.shape)
        return float((np.asarray(self.shadow[r, c])[inside] > 40).mean()) if inside.any() else 0.0


def cover(mask, line, reach):
    """Mark the pixels within `reach` of a stroke's centre line."""
    step = np.hypot(*np.diff(line, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(step)])
    fine = np.arange(0.0, t[-1] + 0.5, 0.5)
    x, y = np.interp(fine, t, line[:, 0]), np.interp(fine, t, line[:, 1])
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            if dr * dr + dc * dc > reach * reach + 1:
                continue
            r = np.clip(np.rint(y).astype(int) + dr, 0, mask.shape[0] - 1)
            c = np.clip(np.rint(x).astype(int) + dc, 0, mask.shape[1] - 1)
            mask[r, c] = True


def _pick(spans, lo, hi):
    """The run of paint that shares most with the rows lo..hi."""
    best, share = None, 0.0
    for a, b in spans:
        s = min(b, hi) - max(a, lo)
        if s > share:
            best, share = (a, b), s
    return best


def _stats(ground, m, rows, kind, line):
    _, width = profile(m["excess"], rows, m["shift"])
    amount = m["amount"][rows]
    return {"kind": kind, "width": width, "rel": m["rel"], "tilt": m["tilt"], "level": m["level"], "ground": m["ground"], "clear": m["clear"],
            "even": float(np.std(amount) / max(np.mean(amount), 1e-6)), "shade": ground.shade(line)}


def fit_line(ground, mid, along, half, kind):
    """A straight stroke from a rough one. Returns a dict with ends "a" and "b", or None."""
    mid = np.asarray(mid, float)
    along = np.asarray(along, float) / np.hypot(*along)
    margin = 16.0
    for _ in range(16):
        across = np.array([-along[1], along[0]])
        reach = np.ceil(half + margin)
        s = np.arange(-reach, reach + 1.0)
        centres = mid[None, :] + s[:, None] * along[None, :]
        known = ground.known(centres)
        strip = sample(ground.photo, centres, np.repeat(across[None, :], len(s), 0))
        m = measure(strip, kind, (np.abs(s) <= max(half * 0.8, 2.0)) & known, ground.shade(centres[np.abs(s) <= half]) > 0.25)
        if m is None or m["clear"] < 2.0:                # nothing here stands out of the ground's grain
            return None
        m["amount"] = np.where(ground.free(centres), m["amount"], 0.0)       # another stroke's paint ends this one
        span = _pick(runs(m["amount"], m["level"], known), reach - half, reach + half)
        if span is None:
            return None
        lo, hi = span[0] - reach, span[1] - reach
        rows = (s >= lo) & (s <= hi) & known
        if rows.sum() < 4:
            return None
        weight = np.clip(m["amount"][rows], 0, 1.5 * m["level"]) + 1e-6
        a, b, rms = weighted_line(s[rows], m["shift"][rows], weight)
        b = float(np.clip(b, -0.08, 0.08))                # turn a little at a time
        centre = 0.5 * (lo + hi)
        moved = abs(a + b * centre) + abs(b) * 0.5 * (hi - lo)
        touching = (span[0] <= 2 and known[0]) or (span[1] >= len(s) - 3 and known[-1])
        mid = mid + centre * along + (a + b * centre) * across
        along = along + b * across
        along /= np.hypot(*along)
        half = 0.5 * (hi - lo)
        if touching:
            margin = min(margin * 2, 800.0)
        elif moved < 0.03:
            ends = np.stack([mid - half * along, mid + half * along])
            out = _stats(ground, m, rows, kind, centres[rows])
            out.update(a=ends[0], b=ends[1], rms=rms, length=2 * half)
            return out
    return None


def dash_at(ground, mid, along, size, kind, level):
    """A dash looked for on a line its neighbours give: the line is kept, only the paint's ends are read.

    For the dash a fit of its own would refuse: one lying along the edge of a patch, in deep
    shade, or under the rim of a tree's crown, where the height map says something stands
    but the photo shows the road. It must show paint of about the right length, at least a
    third as strong as its neighbours, on concrete.
    """
    along = along / np.hypot(*along)
    across = np.array([-along[1], along[0]])
    reach = np.ceil(size)
    s = np.arange(-reach, reach + 1.0)
    centres = mid[None, :] + s[:, None] * along[None, :]
    known = ground.known(centres, standing=False)
    strip = sample(ground.photo, centres, np.repeat(across[None, :], len(s), 0))
    m = measure(strip, kind, (np.abs(s) <= 0.3 * size) & known, ground.shade(centres) > 0.25)
    if m is None or m["level"] < 0.35 * level or m["clear"] < 2.5:
        return None
    amount = np.where(ground.free(centres), m["amount"], 0.0)
    span = _pick(runs(amount, m["level"], known), reach - 0.5 * size, reach + 0.5 * size)
    if span is None or not 0.6 * size <= span[1] - span[0] <= 1.5 * size:
        return None
    red, green, blue = m["rel"]
    soil = m["ground"]
    if (soil.max() - soil.min()) / max(soil.mean(), 1.0) > 0.30 or soil[1] - max(soil[0], soil[2]) > 8:
        return None
    if (red - blue < 0.03) if kind == "yellow" else (min(red, green, blue) < 0.02 or 0.5 * (red + green) - blue > 0.04):
        return None
    rows = (s >= span[0] - reach) & (s <= span[1] - reach) & known
    off = float(np.average(m["shift"][rows], weights=np.clip(m["amount"][rows], 1e-6, None)))
    if abs(off) > 1.2:
        return None
    a, b = (mid + (v - reach) * along + off * across for v in span)
    out = _stats(ground, m, rows, kind, centres[rows])
    out.update(a=a, b=b, rms=0.0, length=float(span[1] - span[0]))
    return out


def _path(points, weight=None, spacing=1.0, rough=0.2):
    """A smooth curve through points, resampled at even spacing along its length."""
    step = np.hypot(*np.diff(points, axis=0).T)
    keep = np.concatenate([[True], step > 1e-6])
    points = points[keep]
    weight = None if weight is None else weight[keep]
    t = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    if len(points) < 6 or t[-1] < 4:
        return None
    w = np.full(len(points), 1.0 / rough) if weight is None else weight / rough
    try:
        tck, _ = splprep([points[:, 0], points[:, 1]], w=w, u=t, k=3, s=float(len(points)))
    except (ValueError, TypeError):
        return None
    fine = np.linspace(0, t[-1], max(int(t[-1] * 8), 8))
    xy = np.stack(splev(fine, tck), 1)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    even = np.arange(0.0, arc[-1] + 1e-9, spacing)
    return np.stack([np.interp(even, arc, xy[:, 0]), np.interp(even, arc, xy[:, 1])], 1)


def _onward(tail, count):
    """Points that carry a curve on past its last point, keeping the bend it has there."""
    tail = tail[-40:]
    origin = tail[-1]
    along = tail[-1] - tail[max(len(tail) - 12, 0)]
    along /= max(np.hypot(*along), 1e-9)
    across = np.array([-along[1], along[0]])
    u, v = (tail - origin) @ along, (tail - origin) @ across
    bend = float(np.clip(np.polyfit(u, v, 2)[0], -0.01, 0.01)) if len(tail) >= 12 else 0.0
    step = np.arange(1.0, count + 1.0)
    return origin + step[:, None] * along + (bend * step ** 2)[:, None] * across


def _frames(path):
    d = np.gradient(path, axis=0)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    return np.stack([-d[:, 1], d[:, 0]], 1)


def _read(ground, path, kind, seed):
    across = _frames(path)
    known = ground.known(path)
    m = measure(sample(ground.photo, path, across), kind, seed & known, ground.shade(path[seed]) > 0.25)
    if m is not None:
        m["amount"] = np.where(ground.free(path), m["amount"], 0.0)
    return m, known, across


def _grow(ground, path, kind, step=16, tail=48):
    """Follow the paint on past the end of a path. Returns the longer path and whether it met its own start."""
    while len(path) < 60000:
        nb = min(tail, len(path))
        ext = np.concatenate([path[-nb:], _onward(path, step)])
        seed = np.arange(len(ext)) < nb
        for _ in range(2):                                   # pull the new rows onto the crest
            m, known, across = _read(ground, ext, kind, seed)
            if m is None:
                return path, False
            pull = np.clip(m["shift"], -1.5, 1.5) * (m["amount"] > 0.3 * m["level"]) * ~seed
            ext = ext + pull[:, None] * across
        span = _pick(runs(m["amount"], m["level"], known), 0, nb - 1)
        if span is None or span[1] < len(ext) - 3:           # the paint stops within the new rows
            keep = nb if span is None else max(int(np.floor(span[1])) + 1, nb)
            return np.concatenate([path[:-nb], ext[:keep]]), False
        if len(path) > 80:                                   # a ring comes back to where it began
            d = np.hypot(*(ext[nb:, None, :] - path[None, :12, :]).transpose(2, 0, 1)).min(1)
            if d.min() < 1.5:
                return np.concatenate([path[:-nb], ext[:nb + int(np.argmin(d)) + 1]]), True
        path = np.concatenate([path[:-nb], ext])
    return path, False


def fit_curve(ground, points, kind):
    """A curved stroke from a chain of rough points. Returns a dict with "path" (a point a pixel), or None."""
    path = _path(np.asarray(points, float))
    if path is None:
        return None
    for _ in range(2):                                       # settle onto the crests where the rough line is
        m, known, across = _read(ground, path, kind, np.ones(len(path), bool))
        if m is None or m["clear"] < 2.0:
            return None
        rows = known & (m["amount"] > 0.3 * m["level"])
        if rows.sum() < 8:
            return None
        path = _path(path[rows] + m["shift"][rows, None] * across[rows], np.clip(m["amount"][rows] / m["level"], 0.05, 1.0))
        if path is None:
            return None
    path, ring = _grow(ground, path, kind)
    if not ring:
        path = _grow(ground, path[::-1], kind)[0][::-1]
    lead = 0 if ring else 8
    for _ in range(3):
        if lead:
            path = np.concatenate([_onward(path[::-1], lead)[::-1], path, _onward(path, lead)])
        seed = np.ones(len(path), bool)
        seed[:lead] = seed[len(path) - lead:] = False
        m, known, across = _read(ground, path, kind, seed)
        if m is None:
            return None
        span = _pick(runs(m["amount"], m["level"], known), lead, len(path) - 1 - lead)
        if span is None:
            return None
        lo, hi = span
        ring = ring and lo <= 2 and hi >= len(path) - 3      # a ring with a piece missing is an arc
        rows = np.zeros(len(path), bool)
        rows[int(np.ceil(lo)):int(np.floor(hi)) + 1] = True
        rows &= known
        if rows.sum() < 8:
            return None
        weight = np.clip(m["amount"] / m["level"], 0.05, 1.0)
        target = path[rows] + m["shift"][rows, None] * across[rows]
        dense = _path(target, weight[rows], spacing=0.2)
        if dense is None:
            return None
        path = dense[::5] if (len(dense) - 1) % 5 == 0 else np.concatenate([dense[::5], dense[-1:]])
    resid = _apart(target, dense)
    out = _stats(ground, m, rows, kind, path)
    if ring:
        path = np.concatenate([path, path[:1]])
    else:
        head = path[0] + (np.ceil(lo) - lo) * _unit(path[0] - path[1])
        tail = path[-1] + (hi - np.floor(hi)) * _unit(path[-1] - path[-2])
        path = np.concatenate([head[None], path, tail[None]])
    out.update(path=path, ring=ring, length=float(hi - lo), rms=float(np.sqrt(np.average(resid ** 2, weights=weight[rows]))))
    out["bow"] = bow(path)[0]
    return out


def _unit(v):
    return v / max(np.hypot(*v), 1e-9)


def _apart(points, path):
    """Distance from each point to the nearest vertex of a densely sampled path that runs alongside."""
    out = np.empty(len(points))
    for a in range(0, len(points), 512):
        b = path[max(a * 5 - 400, 0):a * 5 + 512 * 5 + 400]
        d = points[a:a + 512, None, :] - b[None, :, :]
        out[a:a + 512] = np.sqrt((d ** 2).sum(2).min(1))
    return out


def bow(path):
    """How far a path strays from the straight line that fits it best, and that line (middle, direction, half length)."""
    mid = path.mean(0)
    _, _, vt = np.linalg.svd(path - mid, full_matrices=False)
    off = (path - mid) @ np.array([-vt[0][1], vt[0][0]])
    t = (path - mid) @ vt[0]
    return float(np.abs(off).max()), mid + 0.5 * (t.max() + t.min()) * vt[0], vt[0], 0.5 * float(np.ptp(t))


def judge(st, lenient=False):
    """Is a measured stroke paint? Returns "" for yes, or the reason it is not.

    `lenient` is for a dash looked for where its neighbours say one should be: there the
    place itself is evidence, and fainter paint is believed.
    """
    ease = 0.5 if lenient else 1.0
    red, green, blue = st["rel"]
    ground = st["ground"]
    shaded = st["shade"] > 0.25
    if (ground.max() - ground.min()) / max(ground.mean(), 1.0) > 0.30 or ground[1] - max(ground[0], ground[2]) > 8:
        return "ground"                                  # grass, soil, a roof: not concrete either side
    if st["tilt"] > 0.12 / ease and not (shaded and st["kind"] == "yellow"):
        return "edge"                                    # one side lighter than the other: an edge, not a line
    wide = 3.9 if st["clear"] < 8.0 else (5.5 if st["kind"] == "yellow" else 4.6)       # worn yellow reads broad: its blue is coarse
    if not 1.3 <= st["width"] <= wide:
        return "width"
    if st["clear"] < 4.0 * (0.65 if lenient else 1.0):
        return "grain"                                   # no clearer than the streaks the pavement has anyway
    if st["kind"] == "white":
        if min(red, green, blue) < 0.025 * (0.8 if lenient else 1.0) or 0.5 * (red + green) - blue > 0.04:
            return "colour"
        if max(red, green, blue) > 0.85:
            return "glare"
        if min(red, green, blue) < 0.05 and not lenient and (st.get("bow", 0.0) > 1.0 or st["length"] < 20 or st["tilt"] > 0.05):
            return "faint"                               # paint this worn has to be a clean, all but straight line
        if min(red, green, blue) < 0.12 and not lenient and st["clear"] < 8.0:
            return "grain"                               # and to stand well clear: asphalt has pale streaks of its own
        if shaded and ("path" in st or st["length"] < 16 or st["rms"] > 0.12 or st["even"] > 0.30):
            return "sun fleck"                           # under trees only a clean straight line is believed
    else:
        if red - blue < 0.08 * ease or red < -0.03 or green > red + 0.05 or blue > 0.6 * red + 0.02:
            return "colour"
        if shaded and (red - blue < 0.15 * ease or blue > 0.4 * red + 0.02):
            return "sun fleck"                           # sunlight is yellower than shade, but not as yellow as paint
    if st["rms"] > (0.22 if "a" in st else 0.30):
        return "wobble"
    if st["even"] > 0.45:
        return "uneven"
    if st["length"] < 5:
        return "short"
    return ""
