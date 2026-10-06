"""Make measured strokes agree with each other.

Three things, in order:
  meet     an end that stops at another stroke is moved onto that stroke's centre line, so a
           corner is one point and a T has no gap
  outline  straight strokes of one colour that meet end to end become one many-pointed stroke,
           closed when they come back round (a painted rectangle)
  runs     dashes that follow one another are chained, and each stretch of a chain that lies
           on one straight line or one circle is put exactly on it

Positions are photo pixels (x, y). A stroke here is a dict with "kind", "pts" (N x 2), "width"
(pixels) and "form": "line" (two points), "curve" (a point a pixel) or "outline".
"""
import numpy as np

import markings_fit as fit

BACK = 3.5             # pixels an end may be pulled back to meet another stroke: paint runs on to the far edge of a corner
ON = {True: 1.6, False: 2.6}   # and pushed on: to a stroke of its own colour, or under one of the other colour
OFF = 0.6              # pixels a dash may be moved sideways onto the common line or circle


def items(strokes):
    out = []
    for st in strokes:
        line = "a" in st
        out.append({"kind": st["kind"], "form": "line" if line else "curve", "closed": bool(st.get("ring")),
                    "pts": np.stack([st["a"], st["b"]]) if line else np.array(st["path"]),
                    "width": st["width"], "level": st["level"], "snapped": False})
    return out


def _end(pts, last):
    """An end point and the unit direction the stroke leaves it in."""
    p = pts[::-1] if last else pts
    tip, back = p[0], p[min(6, len(p) - 1)]
    d = tip - back
    return tip, d / max(np.hypot(*d), 1e-9)


def _hit(tip, out, a, b, slack, on):
    """Where the line through an end crosses segments a-b: distance along `out`, or nan."""
    d = b - a
    length = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    cross = out[0] * d[:, 1] - out[1] * d[:, 0]
    w = a - tip
    ok = np.abs(cross) / length > 0.34                               # not running alongside: 20 degrees or more
    safe = np.where(ok, cross, 1.0)
    t = (w[:, 0] * d[:, 1] - w[:, 1] * d[:, 0]) / safe               # along the end's own direction
    s = (w[:, 0] * out[1] - w[:, 1] * out[0]) / safe                 # along the segment, 0 to 1
    ok &= (t >= -BACK) & (t <= on) & (s >= -slack[:, 0] / length) & (s <= 1 + slack[:, 1] / length)
    return np.where(ok, t, np.nan)


def _corners(its):
    """Pairs of ends of one colour that stop within a few pixels of each other at an angle: both go to where their lines cross."""
    ends = [(i, last) + _end(it["pts"], last) for i, it in enumerate(its) if not it["closed"] for last in (False, True)]
    if not ends:
        return [], set()
    tips = np.array([e[2] for e in ends])
    outs = np.array([e[3] for e in ends])
    white = np.array([its[e[0]]["kind"] == "white" for e in ends])
    owner = np.array([e[0] for e in ends])
    best = {}
    for k in range(len(ends)):
        d = np.hypot(*(tips - tips[k]).T)
        cross = outs[k, 0] * outs[:, 1] - outs[k, 1] * outs[:, 0]
        ok = (d < 6.0) & (owner != owner[k]) & (white == white[k]) & (np.abs(cross) > 0.34)
        if ok.any():
            best[k] = int(np.nonzero(ok)[0][np.argmin(d[ok])])
    moves, done = [], set()
    for k, j in sorted(best.items()):
        if best.get(j) != k or j < k:
            continue
        cross = outs[k, 0] * outs[j, 1] - outs[k, 1] * outs[j, 0]
        w = tips[j] - tips[k]
        tk = (w[0] * outs[j, 1] - w[1] * outs[j, 0]) / cross
        tj = (w[0] * outs[k, 1] - w[1] * outs[k, 0]) / cross
        if abs(tk) > 5.0 or abs(tj) > 5.0:
            continue
        point = tips[k] + tk * outs[k]
        for e, t in ((k, tk), (j, tj)):
            moves.append((ends[e][0], ends[e][1], float(t), point - t * outs[e], outs[e]))     # so that tip + t * out is the one point
            done.add((ends[e][0], ends[e][1]))
    return moves, done


def meet(its):
    """Move every end that stops at another stroke onto that stroke's centre line."""
    seg_a, seg_b, owner, slack = [], [], [], []
    for i, it in enumerate(its):
        p = it["pts"]
        n = len(p) - 1
        seg_a.append(p[:-1])
        seg_b.append(p[1:])
        owner.append(np.full(n, i))
        sl = np.zeros((n, 2))
        if not it["closed"]:
            sl[0, 0] = sl[-1, 1] = 2.0                               # the other stroke may itself stop a little short
        slack.append(sl)
    seg_a, seg_b, owner, slack = map(np.concatenate, (seg_a, seg_b, owner, slack))
    white = np.array([it["kind"] == "white" for it in its])[owner]
    lo, hi = np.minimum(seg_a, seg_b) - 2 * BACK, np.maximum(seg_a, seg_b) + 2 * BACK
    moves, done = _corners(its)
    for i, it in enumerate(its):
        if it["closed"]:
            continue
        for last in (False, True):
            if (i, last) in done:
                continue
            tip, out = _end(it["pts"], last)
            near = (owner != i) & (lo[:, 0] <= tip[0]) & (tip[0] <= hi[:, 0]) & (lo[:, 1] <= tip[1]) & (tip[1] <= hi[:, 1])
            if not near.any():
                continue
            on = np.where(white[near] == (it["kind"] == "white"), ON[True], ON[False])
            t = _hit(tip, out, seg_a[near], seg_b[near], slack[near], on)
            if np.isnan(t).all():
                continue
            moves.append((i, last, float(t[np.nanargmin(np.abs(t))]), tip, out))
    for i, last, t, tip, out in moves:
        p = its[i]["pts"]
        new = tip + t * out
        if its[i]["form"] == "line":
            p[-1 if last else 0] = new
            continue
        q = p[::-1] if last else p                                   # a curve: drop what lies past the new end
        keep = ((q - new) @ out) < -0.5
        q = np.concatenate([new[None], q[keep]])
        its[i]["pts"] = q[::-1] if last else q
    return its


def outline(its):
    """Strokes of one colour that share end points become one stroke: a corner is then a point of it."""
    open_ = [i for i, it in enumerate(its) if not it["closed"]]
    cells = {}
    for i in open_:
        for last in (0, 1):
            p = its[i]["pts"][-last]
            cells.setdefault((int(np.floor(p[0])), int(np.floor(p[1]))), []).append((i, last))
    link = {}
    for i in open_:
        for last in (0, 1):
            p = its[i]["pts"][-last]
            cx, cy = int(np.floor(p[0])), int(np.floor(p[1]))
            same = [e for dx in (-1, 0, 1) for dy in (-1, 0, 1) for e in cells.get((cx + dx, cy + dy), [])
                    if e != (i, last) and np.hypot(*(its[e[0]]["pts"][-e[1]] - p)) < 0.05]
            if len(same) == 1 and same[0][0] != i and its[same[0][0]]["kind"] == its[i]["kind"]:
                link[(i, last)] = same[0]
    used, out = set(), []
    for i in open_:
        if i in used or ((i, 0) in link and (i, 1) in link):
            continue                                                 # start chains from a free end
        chain = _walk(its, link, i, 0 if (i, 1) in link else 1, used)
        out.append(chain if chain is not None else its[i])
    for i in open_:
        if i not in used:                                            # what is left is closed loops
            chain = _walk(its, link, i, 0, used)
            out.append(chain if chain is not None else its[i])
    return out + [it for it in its if it["closed"]]


def _walk(its, link, i, start, used):
    """Follow links from stroke i, entering at end `start`. Returns the joined stroke, or None if i stands alone."""
    first = i
    pts, widths, levels = [], [], []
    while True:
        used.add(i)
        p = its[i]["pts"][::-1] if start else its[i]["pts"]
        pts.append(p if not pts else p[1:])
        widths.append(its[i]["width"])
        levels.append(its[i]["level"])
        nxt = link.get((i, 1 - start))
        if nxt is None or nxt[0] == first or nxt[0] in used:
            closed = nxt is not None and nxt[0] == first
            break
        i, start = nxt
    if len(pts) == 1:
        return None
    pts = np.concatenate(pts)
    if closed:
        pts[-1] = pts[0]
    return {"kind": its[first]["kind"], "form": "outline", "closed": closed, "pts": pts,
            "width": float(np.median(widths)), "level": float(np.median(levels)), "snapped": False}


def fit_straight(p):
    mid = p.mean(0)
    _, _, vt = np.linalg.svd(p - mid, full_matrices=False)
    across = np.array([-vt[0][1], vt[0][0]])
    return ("line", mid, vt[0]), (p - mid) @ across


def fit_circle(p):
    """Least-squares circle: algebraic start, then a few steps on the true distances."""
    mid = p.mean(0)
    q = p - mid
    scale = max(np.abs(q).max(), 1e-9)
    q = q / scale
    A = np.column_stack([2 * q, np.ones(len(q))])
    sol, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
    c = sol[:2]
    r = np.sqrt(max(sol[2] + c @ c, 1e-12))
    for _ in range(8):
        d = q - c
        dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)
        J = np.column_stack([-d / dist[:, None], -np.ones(len(q))])
        step, *_ = np.linalg.lstsq(J, -(dist - r), rcond=None)
        c, r = c + step[:2], r + step[2]
    centre, radius = mid + c * scale, r * scale
    return ("circle", centre, radius), np.hypot(*(p - centre).T) - radius


def _onto(model, p):
    if model[0] == "line":
        _, mid, along = model
        return mid + ((p - mid) @ along)[:, None] * along
    _, centre, radius = model
    d = p - centre
    return centre + d * (radius / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9))[:, None]


def chains(its):
    """Chain dashes that follow one another. Returns lists of (index, flipped) in order along each chain."""
    dash = [i for i, it in enumerate(its) if it["form"] == "line"]
    if not dash:
        return []
    a = np.array([its[i]["pts"][0] for i in dash])
    b = np.array([its[i]["pts"][-1] for i in dash])
    kind = np.array([its[i]["kind"] == "white" for i in dash])
    mid, half = 0.5 * (a + b), 0.5 * np.hypot(*(b - a).T)
    along = (b - a) / (2 * half[:, None])
    nxt = {}
    for k in range(len(dash)):
        for way in (1, -1):
            u = along[k] * way
            d = mid - mid[k]
            ahead = d @ u
            side = d[:, 0] * u[1] - d[:, 1] * u[0]
            gap = ahead - half[k] - half
            dot = along @ u
            turn1 = np.arctan2(side, ahead)                           # from this dash to the way across
            v = along * np.sign(dot)[:, None]
            turn2 = np.arctan2(d[:, 0] * v[:, 1] - d[:, 1] * v[:, 0], (d * v).sum(1))   # from the way across to that dash
            ok = (kind == kind[k]) & (gap > 2) & (gap <= 16 * 0.5 * (half[k] + half) * 2) & (np.abs(side) <= 1.5 + 0.15 * ahead)
            ok &= (half / half[k] > 0.4) & (half / half[k] < 2.5) & (np.abs(turn1) < 0.21) & (np.abs(turn2) < 0.21) & (np.abs(turn1 + turn2) < 0.07 + 2.0 / np.maximum(ahead, 1))
            ok[k] = False
            if ok.any():
                j = int(np.nonzero(ok)[0][np.argmin(ahead[ok])])
                nxt[(k, way)] = (j, int(np.sign(dot[j])) or 1)        # which way along that dash we are then going
    pair = {}
    for (k, way), (j, wj) in nxt.items():                             # keep a link only if it holds both ways
        back = nxt.get((j, -wj))
        if back is not None and back[0] == k:
            pair[(k, way)] = (j, wj)
    seen, out = set(), []
    for ring in (False, True):                                        # open chains from a loose end, then closed rings
        for k in range(len(dash)):
            if k in seen or (not ring and (k, 1) in pair and (k, -1) in pair):
                continue
            way = 1 if (k, 1) in pair else -1
            chain, cur = [], (k, way)
            while cur is not None and cur[0] not in seen:
                seen.add(cur[0])
                chain.append((dash[cur[0]], cur[1] < 0))
                cur = pair.get(cur)
            if len(chain) >= 3:
                out.append(chain)
    return out


def _ends(its, chain):
    """End points of the dashes of a chain, in travelling order: (n, 2, 2)."""
    return np.array([its[i]["pts"][[-1, 0]] if flip else its[i]["pts"][[0, -1]] for i, flip in chain])


def snap(its, chain):
    """Put each stretch of a chain that lies on one line or one circle exactly on it. Returns how many dashes moved."""
    ends = _ends(its, chain)
    n, k, moved = len(chain), 0, 0
    while k + 2 < n:
        best = None
        for e in range(k + 2, n):
            p = ends[k:e + 1].reshape(-1, 2)
            for fit in (fit_straight, fit_circle):
                model, left = fit(p)
                if model[0] == "circle" and model[2] < 40:
                    continue
                worst = np.abs(left).reshape(-1, 2).max(1)
                if np.median(worst) <= 0.25 and (worst <= OFF).mean() >= 0.85 and worst.max() <= 2 * OFF:
                    best = (e, model, worst)
                    break
            else:
                break
        if best is None:
            k += 1
            continue
        e, model, worst = best
        for j in range(k, e + 1):
            if worst[j - k] > OFF:
                continue
            i, flip = chain[j]
            p = _onto(model, ends[j])
            if model[0] == "circle" and np.hypot(*(p[1] - p[0])) ** 2 / (8 * model[2]) > 0.2:
                p = np.stack([p[0], _onto(model, 0.5 * (p[0] + p[1])[None])[0], p[1]])     # a dash long enough to show the bend
            its[i]["pts"] = p[::-1] if flip else p
            its[i]["snapped"] = True
            moved += 1
        k = e + 1
    width = float(np.median([its[i]["width"] for i, _ in chain]))
    for i, _ in chain:
        its[i]["width"] = width
    return moved


def _item(st):
    return {"kind": st["kind"], "form": "line", "closed": False, "pts": np.stack([st["a"], st["b"]]),
            "width": st["width"], "level": st["level"], "snapped": False}


def _look(ground, mid, along, size, kind, level):
    """Is there a dash of about this size, here, pointing this way? The measured stroke, or None."""
    st = fit.fit_line(ground, mid, along, 0.5 * size, kind)
    if st is not None and not fit.judge(st, lenient=True) and 0.5 * size <= st["length"] <= 1.6 * size:
        d = st["b"] - st["a"]
        off = 0.5 * (st["a"] + st["b"]) - mid
        if (abs(off[0] * along[1] - off[1] * along[0]) <= 2.0 and abs(off @ along) <= 0.6 * size
                and abs(d[0] * along[1] - d[1] * along[0]) / np.hypot(*d) <= 0.09):       # five degrees
            return st
    return fit.dash_at(ground, mid, along, size, kind, level)


def recover(ground, its, chain):
    """Look harder where a chain of dashes says there should be one: in its gaps, and on past its ends."""
    ends = _ends(its, chain)
    kind = its[chain[0][0]]["kind"]
    mid = ends.mean(1)
    size = float(np.median(np.hypot(*(ends[:, 1] - ends[:, 0]).T)))
    level = float(np.median([its[i]["level"] for i, _ in chain]))
    hop = np.hypot(*np.diff(mid, axis=0).T)
    pitch = float(np.median(hop[hop <= 1.5 * np.percentile(hop, 25)]))
    found = []
    for k in range(len(chain) - 1):                                   # gaps: a smooth path from one dash to the next
        count = int(round(hop[k] / pitch))
        if count < 2 or abs(hop[k] / pitch - count) > 0.3:
            continue
        p0, p1 = mid[k], mid[k + 1]
        t0 = (ends[k, 1] - ends[k, 0]) / np.hypot(*(ends[k, 1] - ends[k, 0])) * hop[k]
        t1 = (ends[k + 1, 1] - ends[k + 1, 0]) / np.hypot(*(ends[k + 1, 1] - ends[k + 1, 0])) * hop[k]
        for j in range(1, count):
            u = j / count
            at = (2 * u ** 3 - 3 * u ** 2 + 1) * p0 + (u ** 3 - 2 * u ** 2 + u) * t0 + (-2 * u ** 3 + 3 * u ** 2) * p1 + (u ** 3 - u ** 2) * t1
            way = (6 * u ** 2 - 6 * u) * p0 + (3 * u ** 2 - 4 * u + 1) * t0 + (-6 * u ** 2 + 6 * u) * p1 + (3 * u ** 2 - 2 * u) * t1
            st = _look(ground, at, way / np.hypot(*way), size, kind, level)
            if st is not None:
                found.append(st)
    for flip in (False, True):                                        # ends: carry on with the bend the last three have
        tail = mid[::-1][-3:] if flip else mid[-3:]
        last = ends[0, ::-1] if flip else ends[-1]
        for _ in range(40):
            a, b, c = tail
            way = (last[1] - last[0]) / np.hypot(*(last[1] - last[0]))
            turn = np.arctan2((b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0], (b - a) @ (c - b))
            turn *= pitch / max(0.5 * (np.hypot(*(b - a)) + np.hypot(*(c - b))), 1e-9)       # turning per dash
            rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
            half = np.array([[np.cos(turn / 2), -np.sin(turn / 2)], [np.sin(turn / 2), np.cos(turn / 2)]])
            st = _look(ground, c + pitch * (half @ way), rot @ way, size, kind, level)
            if st is None:
                break
            found.append(st)
            last = np.stack([st["a"], st["b"]])
            if (last[1] - last[0]) @ way < 0:
                last = last[::-1]
            tail = np.stack([b, c, last.mean(0)])
    out = []
    for st in found:
        fit.cover(ground.used, np.stack([st["a"], st["b"]]), 1)
        out.append(_item(st))
    return out


def tidy(ground, strokes, log=None):
    """Measured strokes to finished ones: ends meeting, outlines joined, dashes on their common lines."""
    its = outline(meet(items(strokes)))
    ground.used = np.zeros(ground.photo.shape[:2], bool)
    white = np.zeros(ground.photo.shape[:2], bool)
    for it in its:
        fit.cover(ground.used, it["pts"], 1)
        if it["kind"] == "white":
            fit.cover(white, it["pts"], 2)
    extra = 0
    for _ in range(2):                                                # a dash found may link two chains: look again
        new = []
        for chain in chains(its):
            ground.hide = white if its[chain[0][0]]["kind"] == "yellow" else None
            new += recover(ground, its, chain)
        its += new
        extra += len(new)
        if not new:
            break
    ground.used = ground.hide = None
    runs = chains(its)
    moved = sum(snap(its, chain) for chain in runs)
    yellow = [it for it in its if it["kind"] == "yellow"]
    if yellow:                         # yellow reads broad or narrow with its wear, not its width: one width for all of it
        width = np.array([it["width"] for it in yellow])
        length = np.array([np.hypot(*np.diff(it["pts"], axis=0).T).sum() for it in yellow])
        order = np.argsort(width, kind="stable")
        middle = width[order][np.searchsorted(np.cumsum(length[order]), 0.5 * length.sum())]
        for it in yellow:
            it["width"] = float(middle)
    if log is not None:
        log.update({"runs": len(runs), "dashes in runs": sum(len(c) for c in runs), "dashes snapped": moved,
                    "dashes recovered": extra, "outlines": sum(it["form"] == "outline" for it in its)})
    return its
