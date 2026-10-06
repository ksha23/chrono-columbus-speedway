"""Long thin lines in the drone photo, traced into ordered runs of points.

This is the geometry half of barriers.py and knows nothing about what a barrier is. It turns
the photo into a thin-line raster, keeps the pixels that carry on in a straight run, traces
them into smooth pieces, and chains pieces end to end across gaps that a caller-supplied test
accepts.
"""
import json
import os
import warnings

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

RES = 0.1                 # working cell, metres

# Thin blue line in the photo.
LINE_FINE = 0.075         # metres, the line's own scale
LINE_COARSE = 0.4         # metres, its surroundings
LINE_MIN = 12.0           # blue-minus-red contrast (0..255 scale) that counts as line

# A candidate must carry on in a straight line.
RUN_HALF = 2.5            # metres either side of the pixel
RUN_ANGLES = 36           # directions tried, over half a turn
RUN_CORE = 0.7            # filled fraction of the run that makes a line
RUN_EDGE = 0.45           # filled fraction that still belongs to a line it touches

# Pieces.
HOP = 0.35                # metres, pixels of one piece are at most this far apart
PIECE_MIN = 1.5           # metres, shorter pieces are dropped
STEP = 0.25               # metres between samples along a piece
SMOOTH = 1.5              # metres, half window of the local straight-line fit
CORNER = 35.0             # degrees of turn within two metres that splits a piece

# Joins.
GAP_FREE = 1.2            # metres, a gap this short needs no explanation
GAP_CURVE = 12.0          # metres, up to here a gap may follow a bend in the road
GAP_MAX = 40.0            # metres, longest gap joined across
TURN_RADIUS = 20.0        # metres, the tightest bend a short gap may hide
CORNER_GAP = 1.0          # metres, ends this close may meet at an angle
JOG_TURN = 20.0           # degrees: ends that close and this nearly parallel are one line, stepped
JOG_BLEND = 2.0           # metres over which such a step is smoothed away
CORNER_TURN = 100.0       # degrees, the sharpest corner one barrier turns
TWIN = 0.8                # metres within which a parallel line is the same thing seen twice


# ----------------------------------------------------------------------------- rasters

def load_grid(work_dir):
    with open(os.path.join(work_dir, "raster.json")) as f:
        raster = json.load(f)
    return {"x0": float(raster["x0"]), "y1": float(raster["y1"]), "res": float(raster["res"])}


def to_cell(grid, x, y):
    """Scene metres to fractional (row, col) on the RES grid, cell centres at whole numbers."""
    return (grid["y1"] - np.asarray(y)) / RES - 0.5, (np.asarray(x) - grid["x0"]) / RES - 0.5


def to_scene(grid, row, col):
    return grid["x0"] + (np.asarray(col) + 0.5) * RES, grid["y1"] - (np.asarray(row) + 0.5) * RES


def cells(grid, shape, points):
    """Whole (row, col) of the cells under scene points, clipped to the raster."""
    row, col = to_cell(grid, points[:, 0], points[:, 1])
    return (np.clip(np.round(row).astype(int), 0, shape[0] - 1),
            np.clip(np.round(col).astype(int), 0, shape[1] - 1))


def build_rasters(work_dir):
    """Everything the search needs, on RES cells.

    Returns a dict: "height" above ground (NaN where no scan), "line" thin-blue-line contrast,
    "lum" brightness, "buildings" and "pavement" masks, and the grid.
    """
    cache = work_dir     # the work directory holds the rasters itself
    grid = load_grid(work_dir)
    surface = np.load(os.path.join(cache, "surface.npy"), mmap_mode="r")
    photo = np.load(os.path.join(cache, "photo.npy"), mmap_mode="r")
    objects = np.load(os.path.join(cache, "objects.npz"))
    f = int(round(RES / grid["res"]))
    k = int(round(float(objects["cell"]) / grid["res"]))
    rows, cols = surface.shape
    h, w = rows // f, cols // f

    top = np.full((h, w), np.nan, np.float32)
    top_coarse = np.full((rows // k, cols // k), np.nan, np.float32)
    line = np.zeros((h, w), np.float32)
    lum = np.zeros((h, w), np.uint8)
    fine, coarse = LINE_FINE / grid["res"], LINE_COARSE / grid["res"]
    pad = int(np.ceil(4 * coarse)) + 2
    pad += pad % 2
    strip = 160 * f * k                       # whole blocks of both sizes
    for r0 in range(0, rows, strip):
        r1 = min(rows, r0 + strip)
        z = np.asarray(surface[r0:r1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # blocks with no scan at all
            n = (r1 - r0) // f
            top[r0 // f:r0 // f + n] = np.nanmax(z[:n * f, :w * f].reshape(n, f, w, f), axis=(1, 3))
            n = (r1 - r0) // k
            top_coarse[r0 // k:r0 // k + n] = np.nanmax(
                z[:n * k, :(cols // k) * k].reshape(n, k, cols // k, k), axis=(1, 3))
        del z

        a, b = max(0, r0 - pad), min(rows, r1 + pad)
        rgb = np.asarray(photo[a:b])
        seen = (rgb.max(-1) > 0).astype(np.float32)
        blue = rgb[..., 2].astype(np.float32) - rgb[..., 0].astype(np.float32)
        # Averages over seen pixels only, so the scan's black border is not an edge.
        near = ndimage.gaussian_filter(blue * seen, fine) / np.maximum(ndimage.gaussian_filter(seen, fine), 1e-3)
        far = ndimage.gaussian_filter(blue * seen, coarse) / np.maximum(ndimage.gaussian_filter(seen, coarse), 1e-3)
        contrast = ((near - far) * seen)[r0 - a:r0 - a + (r1 - r0)]
        n = (r1 - r0) // f
        line[r0 // f:r0 // f + n] = contrast[:n * f, :w * f].reshape(n, f, w, f).max(axis=(1, 3))
        grey = rgb[r0 - a:r0 - a + n * f, :w * f].reshape(n, f, w, f, 3).mean(axis=(1, 3, 4))
        lum[r0 // f:r0 // f + n] = grey.astype(np.uint8)
        del rgb, seen, blue, near, far, contrast

    # Ground under the scan: the coarse top surface less the coarse height above ground.
    ground = top_coarse - objects["height"][:top_coarse.shape[0], :top_coarse.shape[1]]
    known = np.isfinite(ground)
    nearest = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    ground = ground[nearest[0], nearest[1]]
    scale = RES / float(objects["cell"])
    rr = (np.arange(h) + 0.5) * scale - 0.5
    cc = (np.arange(w) + 0.5) * scale - 0.5
    ground = ndimage.map_coordinates(ground, np.meshgrid(rr, cc, indexing="ij"), order=1, mode="nearest")
    height = top - ground.astype(np.float32)

    def onto_grid(mask, cell):
        """A mask on coarser cells of the same origin, repeated onto the RES grid."""
        up, down = int(round(cell / RES * 2)), 2
        big = np.repeat(np.repeat(mask, up, 0), up, 1)[::down, ::down]
        out = np.zeros((h, w), bool)
        out[:min(h, big.shape[0]), :min(w, big.shape[1])] = big[:h, :w]
        return out

    pavement = np.load(os.path.join(cache, "pavement.npy"), mmap_mode="r")
    pavement_cell = RES * round(h / pavement.shape[0])
    return {"grid": grid, "height": height, "line": line, "lum": lum,
            "buildings": onto_grid(objects["buildings"], float(objects["cell"])),
            "pavement": onto_grid(np.asarray(pavement), pavement_cell)}


# ----------------------------------------------------------------------------- lines

def straight_runs(mask):
    """For every set pixel, the largest filled fraction of a straight run through it.

    The run reaches RUN_HALF either side and is tried in RUN_ANGLES directions. Zero off the
    mask.
    """
    reach = int(round(RUN_HALF / RES))
    padded = np.pad(mask, reach + 2).astype(np.uint8)
    rows, cols = np.nonzero(mask)
    rows += reach + 2
    cols += reach + 2
    steps = np.arange(-reach, reach + 1)
    best = np.zeros(rows.size, np.float32)
    for i in range(RUN_ANGLES):
        angle = np.pi * i / RUN_ANGLES
        dr = np.round(-steps * np.sin(angle)).astype(int)
        dc = np.round(steps * np.cos(angle)).astype(int)
        count = np.zeros(rows.size, np.uint16)
        for a, b in zip(dr, dc):
            count += padded[rows + a, cols + b]
        np.maximum(best, count / np.float32(steps.size), out=best)
    fill = np.zeros(mask.shape, np.float32)
    fill[rows - reach - 2, cols - reach - 2] = best
    return fill


def line_pixels(line):
    """Mask of the pixels that lie on long thin lines."""
    candidate = line > LINE_MIN
    fill = straight_runs(candidate)
    # A run centred on the last metre of a line is half empty. Keep such pixels where they
    # touch a pixel that is plainly on a line.
    edge = candidate & (fill >= RUN_EDGE)
    labels, _ = ndimage.label(edge, structure=np.ones((3, 3), int))
    core = np.unique(labels[fill >= RUN_CORE])
    return np.isin(labels, core[core > 0])


def local_fit(points, half):
    """Smooth an ordered run of points by fitting a straight line around each one.

    Each point is replaced by the value at its own position of a line fitted, by distance
    along the run, to the points within `half` metres. A straight run stays straight to its
    very ends, a curve is followed.
    """
    if len(points) < 3:
        return points.copy()
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    out = np.empty_like(points)
    lo = np.searchsorted(s, s - half, "left")
    hi = np.searchsorted(s, s + half, "right")
    for i in range(len(points)):
        a, b = lo[i], hi[i]
        if b - a < 3:
            out[i] = points[i]
            continue
        t = s[a:b] - s[i]
        wt = (1 - np.minimum(np.abs(t) / half, 1.0) ** 3) ** 3 + 1e-3
        sw, st, stt = wt.sum(), (wt * t).sum(), (wt * t * t).sum()
        det = sw * stt - st * st
        for axis in (0, 1):
            v = points[a:b, axis]
            sv, stv = (wt * v).sum(), (wt * t * v).sum()
            out[i, axis] = (stt * sv - st * stv) / det if abs(det) > 1e-9 else sv / sw
    return out


def resample(points, step):
    """Points every `step` metres along a polyline, both ends kept."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return points[:1].copy()
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])], axis=1)


def length_of(points):
    return float(np.hypot(*np.diff(points, axis=0).T).sum()) if len(points) > 1 else 0.0


def longest_path(graph):
    """The longest of the shortest paths through a connected graph: (node order, length)."""
    d0 = csgraph.dijkstra(graph, directed=False, indices=0)
    start = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    d1, before = csgraph.dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    end = int(np.argmax(np.where(np.isfinite(d1), d1, -1)))
    path = [end]
    while path[-1] != start:
        path.append(int(before[path[-1]]))
    return np.array(path), float(d1[end])


def split_at_corners(points):
    """Cut an ordered run where it turns sharply, so that each part is one straight or
    gently curving stretch. The parts share their corner point."""
    reach = int(round(1.0 / STEP))
    n = len(points)
    if n < 2 * reach + 3:
        return [points]
    i = np.arange(reach, n - reach)
    back = points[i] - points[i - reach]
    ahead = points[i + reach] - points[i]
    cosine = (back * ahead).sum(1) / np.maximum(np.hypot(*back.T) * np.hypot(*ahead.T), 1e-9)
    turn = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    cuts = []
    while True:
        k = int(np.argmax(turn))
        if turn[k] < CORNER:
            break
        cuts.append(int(i[k]))
        turn[max(0, k - 2 * reach):k + 2 * reach + 1] = 0.0     # one cut per corner
    parts, last = [], 0
    for cut in sorted(cuts) + [n - 1]:
        if cut > last:
            parts.append(points[last:cut + 1])
        last = cut
    return parts


def onto_crest(points, line, grid):
    """Move each point of a run sideways onto the strongest contrast across the line."""
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    offsets = np.arange(-3, 4) * 0.5 * RES
    px = points[:, None, 0] + normal[:, None, 0] * offsets
    py = points[:, None, 1] + normal[:, None, 1] * offsets
    profile = ndimage.map_coordinates(line, list(to_cell(grid, px, py)), order=1, mode="nearest")
    peak = np.clip(profile.argmax(axis=1), 1, len(offsets) - 2)
    i = np.arange(len(points))
    a, b, c = profile[i, peak - 1], profile[i, peak], profile[i, peak + 1]
    bend = a - 2 * b + c
    shift = np.where(bend < -1e-6, 0.5 * (a - c) / np.where(bend < -1e-6, bend, -1.0), 0.0)
    across = offsets[peak] + np.clip(shift, -1, 1) * 0.5 * RES
    return points + normal * across[:, None]


def trace_pieces(mask, line, grid):
    """Turn the line pixels into ordered, smoothed runs of scene points.

    Pixels within HOP of each other hang together. The centre line of such a set is the
    longest shortest path through it. Whatever is left once that path and the pixels beside it
    are taken away, such as a second line crossing the first, is traced the same way. Each
    path is cut at sharp corners, moved onto the crest of the line across its width and
    smoothed along its length.
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return []
    xy = np.stack(to_scene(grid, rows, cols), axis=1)
    pairs = cKDTree(xy).query_pairs(HOP, output_type="ndarray")
    dist = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    n = len(xy)
    graph = sparse.coo_matrix((dist, (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    graph = graph + graph.T
    count, member = csgraph.connected_components(graph, directed=False)
    order = np.argsort(member, kind="stable")
    bounds = np.searchsorted(member[order], np.arange(count + 1))
    few = PIECE_MIN / RES * 0.5
    work = [order[bounds[c]:bounds[c + 1]] for c in range(count) if bounds[c + 1] - bounds[c] >= few]

    pieces = []
    while work:
        nodes = work.pop()
        sub = graph[nodes][:, nodes]
        path, length = longest_path(sub)
        if length < PIECE_MIN:
            continue
        rough = resample(local_fit(xy[nodes[path]], 0.6), STEP)
        for part in split_at_corners(rough):
            if length_of(part) < PIECE_MIN:
                continue
            part = resample(local_fit(onto_crest(part, line, grid), SMOOTH), STEP)
            if length_of(part) >= PIECE_MIN:
                pieces.append(part)
        away, _ = cKDTree(xy[nodes[path]]).query(xy[nodes])
        rest = nodes[away > 0.3]
        if rest.size < few:
            continue
        parts, which = csgraph.connected_components(graph[rest][:, rest], directed=False)
        for c in range(parts):
            if (which == c).sum() >= few:
                work.append(rest[which == c])
    # A fixed order, whatever order the labelling came in.
    pieces.sort(key=lambda p: (round(float(p[0, 0]), 3), round(float(p[0, 1]), 3), len(p)))
    return pieces


# ----------------------------------------------------------------------------- joining

def end_direction(points, at_end, span=6.0):
    """Unit vector pointing out of a piece at one of its ends, from its last `span` metres."""
    pts = points if at_end else points[::-1]
    n = max(2, min(len(pts), int(round(span / STEP)) + 1))
    tail = pts[-n:]
    _, _, vt = np.linalg.svd(tail - tail.mean(axis=0), full_matrices=False)
    direction = vt[0]
    if np.dot(direction, tail[-1] - tail[0]) < 0:
        direction = -direction
    return direction


def join_pieces(pieces, covered, guided=None):
    """Chain pieces that are one barrier seen with gaps.

    Two ends are joined when each points at the other, and, for a gap longer than GAP_FREE,
    when covered(a, b) says the stretch between them is hidden rather than open. Where
    guided(a, b) says the gap runs along something that leads it, such as the edge of the
    pavement, short pieces are trusted and the gap may bend for twice as far. Ends that all
    but touch may also meet at a corner. Shortest gaps are joined first and an end joins
    once. Returns a list of chains, each a list of (points, seen) with seen False for a gap.
    """
    if not pieces:
        return []
    ends, dirs, lens = [], [], []
    for p in pieces:
        for at_end in (False, True):
            ends.append(p[-1] if at_end else p[0])
            dirs.append(end_direction(p, at_end))
            lens.append(length_of(p))
    ends, dirs, lens = np.array(ends), np.array(dirs), np.array(lens)
    pairs = cKDTree(ends).query_pairs(GAP_MAX, output_type="ndarray")
    candidates = []
    for i, j in pairs:
        if i // 2 == j // 2:
            continue
        gap = ends[j] - ends[i]
        dist = float(np.hypot(*gap))
        turn = np.degrees(np.arccos(float(np.clip(-np.dot(dirs[i], dirs[j]), -1, 1))))
        if dist <= CORNER_GAP and turn <= CORNER_TURN:
            candidates.append((dist, turn, int(i), int(j)))
            continue
        if dist <= CORNER_GAP:
            continue
        unit = gap / dist
        # Split the misfit of the two ends into a sideways offset, which is never allowed to
        # be large, and a turn, which a short gap may hide where the road bends. A long gap
        # is only followed blind in a straight line.
        back = -dirs[j]
        alpha = np.arctan2(dirs[i][0] * unit[1] - dirs[i][1] * unit[0], float(np.dot(dirs[i], unit)))
        beta = np.arctan2(unit[0] * back[1] - unit[1] * back[0], float(np.dot(unit, back)))
        noise = np.arctan2(0.25, min(lens[i], lens[j], 6.0))
        led = guided is not None and dist > GAP_FREE and dist <= 2 * GAP_CURVE and guided(ends[i], ends[j])
        short = dist <= GAP_CURVE or led
        if abs(alpha - beta) / 2 > np.radians(4.0 if short else 2.5) + np.arctan2(0.3, dist) + noise:
            continue
        if abs(alpha + beta) > np.radians(6.0) + noise + ((dist + 6.0) / TURN_RADIUS if short else 0.0):
            continue
        if dist > GAP_FREE and not led:
            if min(lens[i], lens[j]) < min(dist * 0.4, 10.0):
                continue                      # a scrap does not vouch for a long gap
            if not covered(ends[i], ends[j]):
                continue
        candidates.append((dist, turn, int(i), int(j)))
    candidates.sort()

    link = {}
    parent = list(range(len(pieces)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, _, i, j in candidates:
        if i in link or j in link or root(i // 2) == root(j // 2):
            continue
        link[i], link[j] = j, i
        parent[root(i // 2)] = root(j // 2)

    # Every chain is a path with two free ends. Start each from its lower-numbered end piece.
    chains, used = [], set()
    for piece in range(len(pieces)):
        if piece in used:
            continue
        if 2 * piece not in link:
            cur = 2 * piece
        elif 2 * piece + 1 not in link:
            cur = 2 * piece + 1
        else:
            continue                          # an inner piece, reached from its chain's end
        chain, carry = [], None
        while True:
            used.add(cur // 2)
            pts = (pieces[cur // 2] if cur % 2 == 0 else pieces[cur // 2][::-1]).copy()
            if carry is not None:
                along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
                pts += np.clip(1 - along / JOG_BLEND, 0, 1)[:, None] * carry
                carry = None
            far = cur ^ 1
            if far not in link:
                chain.append((pts, True))
                break
            nxt = link[far]
            jump = ends[nxt] - ends[far]
            turn = np.degrees(np.arccos(float(np.clip(-np.dot(dirs[far], dirs[nxt]), -1, 1))))
            if np.hypot(*jump) <= CORNER_GAP and turn <= JOG_TURN:
                # One line seen as two, a step apart: lead both ends to the middle of the step.
                along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts[::-1], axis=0).T))])[::-1]
                pts += np.clip(1 - along / JOG_BLEND, 0, 1)[:, None] * (jump / 2)
                carry = -jump / 2
                chain.append((pts, True))
            else:
                chain.append((pts, True))
                if np.hypot(*jump) > 1e-6:
                    chain.append((np.array([pts[-1], ends[nxt]]), False))
            cur = nxt
        chains.append(chain)
    return chains


def drop_twins(pieces):
    """Remove the stretches of each piece that run alongside a longer one.

    A rail seen at a slant shows two edges, and a fence shows its top bar and its fabric. Of
    two parallel lines within TWIN of each other the longer is kept. What is left of the
    shorter is kept too, where it is long enough to stand as a piece.
    """
    if not pieces:
        return []
    xy = np.concatenate(pieces)
    owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(pieces)])
    tangent = np.concatenate([np.gradient(p, axis=0) for p in pieces])
    tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
    rank = np.argsort(np.argsort([-length_of(p) for p in pieces], kind="stable"), kind="stable")
    tree = cKDTree(xy)
    out, start = [], 0
    for k, p in enumerate(pieces):
        mine = np.arange(start, start + len(p))
        start += len(p)
        twin = np.zeros(len(p), bool)
        for n, near in enumerate(tree.query_ball_point(p, TWIN)):
            near = np.array(near)
            near = near[rank[owner[near]] < rank[k]]
            if near.size and np.abs(tangent[near] @ tangent[mine[n]]).max() > 0.87:   # within 30 degrees
                twin[n] = True
        edges = np.flatnonzero(np.diff(np.concatenate([[True], twin, [True]]).astype(int)))
        for a, b in zip(edges[::2], edges[1::2]):
            if length_of(p[a:b]) >= PIECE_MIN:
                out.append(p[a:b])
    return out


def fit_polyline(points, tolerance):
    """The polyline of fewest vertices that stays within about `tolerance` of the points.

    Douglas-Peucker says where the corners are. Each stretch between corners is then the
    least-squares line through its own points, and the vertices are where those lines meet,
    so that a straight fence comes out straight rather than strung between two noisy points.
    """
    corners = np.flatnonzero(simplify(points, tolerance, mask=True))
    if len(points) < 3:
        return points.copy()
    centres, directions = [], []
    for a, b in zip(corners[:-1], corners[1:]):
        part = points[a:b + 1]
        centre = part.mean(axis=0)
        if len(part) >= 3:
            direction = np.linalg.svd(part - centre, full_matrices=False)[2][0]
        else:
            direction = (part[-1] - part[0]) / max(np.hypot(*(part[-1] - part[0])), 1e-9)
        centres.append(centre)
        directions.append(direction)

    def onto(k, point):
        return centres[k] + directions[k] * np.dot(point - centres[k], directions[k])

    vertices = [onto(0, points[0])]
    for k in range(1, len(centres)):
        corner = points[corners[k]]
        d0, d1 = directions[k - 1], directions[k]
        cross = d0[0] * d1[1] - d0[1] * d1[0]
        meet = None
        if abs(cross) > 0.05:
            gap = centres[k] - centres[k - 1]
            meet = centres[k - 1] + d0 * ((gap[0] * d1[1] - gap[1] * d1[0]) / cross)
        if meet is None or np.hypot(*(meet - corner)) > 2 * tolerance:
            meet = (onto(k - 1, corner) + onto(k, corner)) / 2
        vertices.append(meet)
    vertices.append(onto(len(centres) - 1, points[-1]))
    return np.array(vertices)


def simplify(points, tolerance, mask=False):
    """Douglas-Peucker: the fewest of the points that stay within tolerance of all of them.
    With mask=True the boolean choice is returned in place of the chosen points."""
    keep = np.zeros(len(points), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        chord = points[b] - points[a]
        norm = np.hypot(*chord)
        rel = points[a + 1:b] - points[a]
        if norm < 1e-9:
            off = np.hypot(*rel.T)
        else:
            off = np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0]) / norm
        worst = int(np.argmax(off))
        if off[worst] > tolerance:
            keep[a + 1 + worst] = True
            stack.append((a, a + 1 + worst))
            stack.append((a + 1 + worst, b))
    return keep if mask else points[keep]
