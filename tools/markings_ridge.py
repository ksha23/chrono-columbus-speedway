"""Where thin bright or yellow lines run in the photo: ridge points to sub-pixel, linked into chains.

Paint is two or three pixels wide, so it is found as a ridge: a pixel where the brightness
(or the yellowness) falls away on both sides across the line. At each pixel the second
derivatives of the smoothed image give the direction across the line and how sharp the
ridge is, and the first derivative says how far from the pixel centre the crest lies
(Steger's line detector). Both maps are logarithms, so a line in shade scores like the
same line in sun.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SIGMA = 1.0                                   # pixels: tuned to a 10 to 15 cm line at 5 cm per pixel
SEED = {"white": 0.012, "yellow": 0.014}      # ridge strength a chain point needs
STRONG = {"white": 0.018, "yellow": 0.022}    # and what the best tenth of a chain must reach
TILE = 1024
MARGIN = 12


def maps(rgb):
    """The two pictures lines are looked for in: how white, how yellow. Logarithms, float32."""
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    white = np.log(np.minimum(np.minimum(r, g), b) + 16.0)
    yellow = np.log(((r + g) * 0.5 + 16.0) / (b + 16.0))
    return {"white": white, "yellow": yellow}


def ridge(m, sigma=SIGMA):
    """Strength of the bright ridge through each pixel, and the crest's offset from the pixel centre."""
    rx = ndi.gaussian_filter(m, sigma, order=(0, 1))
    ry = ndi.gaussian_filter(m, sigma, order=(1, 0))
    rxx = ndi.gaussian_filter(m, sigma, order=(0, 2))
    ryy = ndi.gaussian_filter(m, sigma, order=(2, 0))
    rxy = ndi.gaussian_filter(m, sigma, order=(1, 1))
    half = (rxx + ryy) * 0.5
    root = np.sqrt(((rxx - ryy) * 0.5) ** 2 + rxy ** 2)
    low = half - root                                    # the sharper curvature: across the line
    ax, ay = rxy, low - rxx                              # its direction, from whichever row of
    bx, by = low - ryy, rxy                              # (H - low I) is better conditioned
    pick = (bx * bx + by * by) > (ax * ax + ay * ay)
    nx = np.where(pick, bx, ax)
    ny = np.where(pick, by, ay)
    norm = np.hypot(nx, ny) + 1e-12
    nx /= norm
    ny /= norm
    step = -(rx * nx + ry * ny) / np.where(low < 0, low, -1e-12)
    dx, dy = step * nx, step * ny
    on = (low < 0) & (np.abs(dx) <= 0.55) & (np.abs(dy) <= 0.55)
    on &= (half + root) > 0.5 * low                      # a blob, or the round end of a line, is not a ridge
    return -low * sigma * sigma, dx, dy, on


def allowed(paved, taken, height, scale, r0, r1, c0, c1):
    """Pixels of a window where paint could be: on pavement, not taken, nothing standing."""
    ok = _upsample(paved, 2, r0, r1, c0, c1)
    ok = ndi.binary_erosion(ok, iterations=2, border_value=1)       # keep off the grass edge
    ok &= ~np.asarray(taken[r0:r1, c0:c1])
    ok &= ~(_upsample(height, scale, r0, r1, c0, c1) > 0.4)
    return ok


def _upsample(grid, scale, r0, r1, c0, c1):
    rows = np.minimum(np.arange(r0, r1) // scale, grid.shape[0] - 1)
    cols = np.minimum(np.arange(c0, c1) // scale, grid.shape[1] - 1)
    return np.asarray(grid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1])[np.ix_(rows - rows[0], cols - cols[0])]


def points(photo, paved, taken, height, scale, window=None):
    """Ridge points over the whole pavement, tile by tile.

    Returns {"white": (rows, cols, y, x, strength), "yellow": ...} with y, x the crest position
    in pixel units (pixel centres at whole numbers).
    """
    rows, cols = photo.shape[:2]
    wr0, wr1, wc0, wc1 = window or (0, rows, 0, cols)
    found = {k: [] for k in SEED}
    for tr in range(wr0, wr1, TILE):
        for tc in range(wc0, wc1, TILE):
            r0, r1 = max(tr - MARGIN, 0), min(tr + TILE + MARGIN, rows)
            c0, c1 = max(tc - MARGIN, 0), min(tc + TILE + MARGIN, cols)
            if not paved[r0 // 2:(r1 + 1) // 2, c0 // 2:(c1 + 1) // 2].any():
                continue
            ok = allowed(paved, taken, height, scale, r0, r1, c0, c1)
            inner = np.zeros(ok.shape, bool)                       # each pixel belongs to one tile
            inner[tr - r0:min(tr + TILE, wr1) - r0, tc - c0:min(tc + TILE, wc1) - c0] = True
            ok &= inner
            if not ok.any():
                continue
            for kind, m in maps(np.asarray(photo[r0:r1, c0:c1])).items():
                strength, dx, dy, on = ridge(m)
                rr, cc = np.nonzero(on & ok & (strength >= SEED[kind]))
                found[kind].append((rr + r0, cc + c0, rr + r0 + dy[rr, cc], cc + c0 + dx[rr, cc], strength[rr, cc]))
    return {k: tuple(np.concatenate([t[i] for t in v]) if v else np.zeros(0) for i in range(5)) for k, v in found.items()}


def chains(pts, kind, width, min_points=5):
    """Link touching ridge points into ordered runs. Each is (xy, strength), xy as (x, y) pixels."""
    rr, cc, y, x, strength = pts
    if len(rr) == 0:
        return []
    key = rr.astype(np.int64) * width + cc.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key, y, x, strength = key[order], y[order], x[order], strength[order]
    col = key % width
    src, dst = [], []
    for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
        want = key + dr * width + dc
        at = np.minimum(np.searchsorted(key, want), len(key) - 1)
        hit = (key[at] == want) & (col + dc >= 0) & (col + dc < width)
        src.append(np.nonzero(hit)[0])
        dst.append(at[hit])
    src, dst = np.concatenate(src), np.concatenate(dst)
    n = len(key)
    graph = coo_matrix((np.ones(len(src) * 2, np.int8), (np.concatenate([src, dst]), np.concatenate([dst, src]))), shape=(n, n)).tocsr()
    count, label = connected_components(graph, directed=False)
    size = np.bincount(label, minlength=count)
    best = np.zeros(count)
    np.maximum.at(best, label, strength)
    keep = (size >= min_points) & (best >= STRONG[kind])
    members = np.argsort(label, kind="stable")
    starts = np.concatenate([[0], np.cumsum(size)])
    indptr, indices = graph.indptr, graph.indices
    out = []
    for comp in np.nonzero(keep)[0]:
        nodes = members[starts[comp]:starts[comp + 1]]
        for path in _paths(nodes.tolist(), indptr, indices, min_points):
            path = np.asarray(path)
            s = strength[path]
            if np.percentile(s, 90) >= STRONG[kind]:
                out.append((np.stack([x[path], y[path]], 1), s))
    return out


def _paths(nodes, indptr, indices, min_points):
    """Peel a component into simple paths: the longest way through it first, then what is left."""
    alive = set(nodes)
    todo = [nodes]
    while todo:
        group = [n for n in todo.pop() if n in alive]
        if len(group) < min_points:
            continue
        far, _ = _sweep(group[0], alive, indptr, indices)
        end, back = _sweep(far[-1], alive, indptr, indices)
        path = [end[-1]]
        while back[path[-1]] >= 0:
            path.append(back[path[-1]])
        reached = end
        if len(path) >= min_points:
            yield path
        for n in path:                                    # the path and what touches it are used up
            alive.discard(n)
            for m in indices[indptr[n]:indptr[n + 1]].tolist():
                alive.discard(m)
        rest = [n for n in reached if n in alive]
        seen = set()
        for n in rest:                                    # what remains falls into separate pieces
            if n in seen:
                continue
            piece, _ = _sweep(n, alive, indptr, indices)
            seen.update(piece)
            todo.append(piece)


def _sweep(start, alive, indptr, indices):
    """Breadth-first from a point: the points reached in order (farthest last), and the way back."""
    back = {start: -1}
    queue = [start]
    for n in queue:
        for m in indices[indptr[n]:indptr[n + 1]].tolist():
            if m in alive and m not in back:
                back[m] = n
                queue.append(m)
    return queue, back


def simplify(xy, tolerance):
    """Douglas-Peucker: indices of the points that keep a polyline within `tolerance` of the original."""
    keep = np.zeros(len(xy), bool)
    keep[[0, -1]] = True
    stack = [(0, len(xy) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        d = xy[b] - xy[a]
        length = np.hypot(*d)
        rel = xy[a + 1:b] - xy[a]
        off = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / length if length > 1e-9 else np.hypot(rel[:, 0], rel[:, 1])
        k = int(np.argmax(off))
        if off[k] > tolerance:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return np.nonzero(keep)[0]


def pieces(xy, min_points=4, corner=0.75):
    """Cut a chain where it turns a corner. Returns (start, stop, straight) index ranges."""
    knots = simplify(xy, 0.7)
    cuts = [0]
    for i in range(1, len(knots) - 1):
        a, b, c = xy[knots[i - 1]], xy[knots[i]], xy[knots[i + 1]]
        turn = np.arctan2((b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0], (b - a) @ (c - b))
        if abs(turn) > corner:
            cuts.append(int(knots[i]))
    cuts.append(len(xy) - 1)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        a2 = a + (1 if a > 0 else 0)                     # the corner itself belongs to neither side
        b2 = b - (1 if b < len(xy) - 1 else 0)
        if b2 - a2 + 1 >= min_points:
            inside = knots[(knots > a2) & (knots < b2)]
            out.append((a2, b2 + 1, len(inside) == 0))
    return out
