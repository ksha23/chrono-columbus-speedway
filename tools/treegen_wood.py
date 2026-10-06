"""Wood for treegen: grow a branch skeleton toward target points, size it, and skin it with tubes.

The skeleton is a list of branches. Each branch is a finely sampled polyline that starts on a
point of its parent, so limbs can fork anywhere along an older limb. Tubes are skinned from a
coarser resampling of the same polylines, which is what the triangle budget controls.
"""
import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _tangents(points):
    t = np.gradient(points, axis=0) if len(points) > 2 else np.repeat(points[-1:] - points[:1], len(points), 0)
    return unit(t)


def curve(a, tan_a, b, rng, step, fork=0.45, lift=0.4, wiggle=0.05):
    """A limb from a (where the parent heads along tan_a) to b: a cubic Bezier plus a slow wander.

    fork blends the departure direction between the parent's heading (1) and the straight line
    to b (0). lift bends the far end upward (negative lets it droop).
    """
    chord = b - a
    length = float(np.linalg.norm(chord))
    u = chord / max(length, 1e-9)
    d0 = unit(fork * tan_a + (1.0 - fork) * u)
    d1 = unit(u + lift * UP)
    n = max(3, int(np.ceil(length / step)) + 1)
    s = np.linspace(0.0, 1.0, n)[:, None]
    p1 = a + d0 * length * 0.35
    p2 = b - d1 * length * 0.35
    pts = (1 - s) ** 3 * a + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s ** 2 * p2 + s ** 3 * b
    e1 = np.cross(u, UP)
    e1 = unit(e1) if np.linalg.norm(e1) > 0.05 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(u, e1)
    f1, f2 = rng.uniform(0.6, 1.7, 2)
    ph1, ph2 = rng.uniform(0.0, 2 * np.pi, 2)
    amp = np.sin(np.pi * s) * wiggle * length
    return pts + amp * (np.sin(2 * np.pi * f1 * s + ph1) * e1 + np.sin(2 * np.pi * f2 * s + ph2) * e2)


class Skeleton:
    def __init__(self, step):
        self.step = step
        self.points = []     # per branch, (n, 3). Empty once a branch has been merged away.
        self.parent = []     # per branch, (parent branch, point index on it) or (-1, 0) for a root
        self.open_from = []  # per branch, first point index that a child may start from
        self.path0 = []      # per branch, path length from the ground to its first point
        self.extended = []   # per branch, True once a limb has been continued from its tip

    def add(self, points, parent=(-1, 0), open_from=0):
        pb, pi = parent
        self.points.append(np.asarray(points, dtype=float))
        self.parent.append(parent)
        self.open_from.append(open_from)
        self.path0.append(self.path0[pb] + self._arc(pb)[pi] if pb >= 0 else 0.0)
        self.extended.append(False)

    def _arc(self, b):
        p = self.points[b]
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])

    def _nodes(self):
        """Every point a new limb may start from. The last stretch of a branch is closed (its tube
        is tapering to nothing there) except the very tip, which continues the branch."""
        pos, tan, path, ref = [], [], [], []
        for b, p in enumerate(self.points):
            arc = self._arc(b)
            ok = (arc[-1] - arc > 4.0 * self.step) & (np.arange(len(p)) >= self.open_from[b])
            ok[-1] = True
            pos.append(p[ok])
            tan.append(_tangents(p)[ok])
            path.append(self.path0[b] + arc[ok])
            ref.append(np.stack([np.full(ok.sum(), b), np.flatnonzero(ok)], axis=1))
        return np.concatenate(pos), np.concatenate(tan), np.concatenate(path), np.concatenate(ref)

    def grow(self, targets, rng, bf=0.55, cos_min=0.2, fork=0.45, lift=0.4, wiggle=0.05):
        """Join every target to the skeleton, cheapest first.

        Cost of joining a target at a node is the new wood needed plus bf times the path from the
        ground to that node, plus a penalty when the new limb would leave its parent at more than
        acos(cos_min). The trade between the first two is what makes limbs share a scaffold
        instead of each running back to the trunk.
        """
        remaining = list(range(len(targets)))
        while remaining:
            pos, tan, path, ref = self._nodes()
            t = targets[remaining]
            d = t[:, None, :] - pos[None, :, :]
            dist = np.linalg.norm(d, axis=2)
            cosang = (d * tan[None]).sum(axis=2) / np.maximum(dist, 1e-9)
            cost = dist + bf * path[None] + 3.0 * dist * np.clip(cos_min - cosang, 0.0, None)
            cost = cost + np.where(dist < 2.0 * self.step, 1e6, 0.0)
            ti, ni = np.unravel_index(np.argmin(cost), cost.shape)
            pts = curve(pos[ni], tan[ni], t[ti], rng, self.step, fork, lift, wiggle)
            pb, pi = int(ref[ni, 0]), int(ref[ni, 1])
            if pi == len(self.points[pb]) - 1:  # from a tip: the same limb carries on
                self.points[pb] = np.concatenate([self.points[pb], pts[1:]])
                self.extended[pb] = True
            else:
                self.add(pts, parent=(pb, pi))
            remaining.pop(ti)

    def finish_roots(self):
        """A trunk or stem that no limb continued turns into its topmost limb, so it never ends
        in a stump. Roots that carry nothing are dropped."""
        for b in range(len(self.points)):
            if self.parent[b][0] >= 0 or self.extended[b]:
                continue
            kids = [(self.parent[c][1], c) for c in range(len(self.points)) if self.parent[c][0] == b]
            if not kids:
                self.points[b] = np.zeros((0, 3))
                continue
            pi, c = max(kids)
            self.points[b] = np.concatenate([self.points[b][: pi + 1], self.points[c][1:]])
            for g in range(len(self.points)):
                if self.parent[g][0] == c:
                    self.parent[g] = (b, pi + self.parent[g][1])
            self.points[c], self.parent[c] = np.zeros((0, 3)), (-1, 0)

    def radii(self, r_base, r_tip, power=0.5):
        """Radius from the length of wood each point carries (a pipe model)."""
        n = len(self.points)
        arcs = [self._arc(b) if len(self.points[b]) else np.zeros(0) for b in range(n)]
        sub = np.array([a[-1] if len(a) else 0.0 for a in arcs])
        down = [a[-1] - a if len(a) else a for a in arcs]
        for b in range(n - 1, -1, -1):  # children always come after their parent
            pb, pi = self.parent[b]
            if pb >= 0:
                sub[pb] += sub[b]
                down[pb][: pi + 1] += sub[b]
        total = max(down[b][0] for b in range(n) if self.parent[b][0] < 0 and len(down[b]))
        return [np.maximum(r_base * (d / total) ** power, r_tip) for d in down]


def plan_rings(skel, radii, ds, sides_max):
    """Choose ring positions, ring radii and side counts per branch, and count the triangles."""
    plan, count = [], 0
    for b, p in enumerate(skel.points):
        if len(p) < 2:
            plan.append(None)
            continue
        m = max(2, int(round(skel._arc(b)[-1] / ds)) + 1)
        idx = np.unique(np.round(np.linspace(0, len(p) - 1, m)).astype(int))
        # A limb thins just past each fork. Holding the radius until the next ring keeps the
        # tube thick where the child limb starts, so the child's open end stays buried.
        r = np.array([radii[b][0]] + [radii[b][i0 + 1: i1 + 1].max() for i0, i1 in zip(idx[:-1], idx[1:])])
        sides = int(np.clip(3 + (r[0] > 0.03) + (r[0] > 0.065) + (r[0] > 0.12), 3, sides_max))
        plan.append((idx, r, sides))
        count += 2 * sides * (len(idx) - 2) + sides
    return plan, count


def skin(skel, plan, rng, flare_height=0.0, flare=0.0):
    """Tapered tubes with smooth radial normals. Each tube closes to a point at its tip."""
    verts, norms, faces, base = [], [], [], 0
    for b, item in enumerate(plan):
        if item is None:
            continue
        idx, r, s = item
        p = skel.points[b][idx]
        if skel.parent[b][0] < 0 and flare > 0.0:  # root flare, held constant below the ground
            r = r * (1.0 + flare * np.exp(-np.clip(p[:, 2], 0.0, None) / flare_height))
        t = _tangents(skel.points[b])[idx]
        m = len(p)
        frames = [unit(np.cross(t[0], UP if abs(t[0, 2]) < 0.95 else np.array([1.0, 0.0, 0.0])))]
        for i in range(1, m):  # carry the frame along the limb without twisting it
            frames.append(unit(frames[-1] - np.dot(frames[-1], t[i]) * t[i]))
        nrm = np.array(frames)
        bin_ = np.cross(t, nrm)
        ang = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(s) / s
        radial = np.cos(ang)[None, :, None] * nrm[:, None, :] + np.sin(ang)[None, :, None] * bin_[:, None, :]
        verts += [(p[:-1, None, :] + r[:-1, None, None] * radial[:-1]).reshape(-1, 3), p[-1:]]
        norms += [radial[:-1].reshape(-1, 3), t[-1:]]
        j = np.arange(s)
        jn = (j + 1) % s
        for i in range(m - 2):
            a, c = base + i * s, base + (i + 1) * s
            faces += [np.stack([a + j, a + jn, c + jn], 1), np.stack([a + j, c + jn, c + j], 1)]
        a, apex = base + (m - 2) * s, base + (m - 1) * s
        faces.append(np.stack([a + j, a + jn, np.full(s, apex)], 1))
        base = apex + 1
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.concatenate(verts), np.concatenate(norms), np.concatenate(faces).astype(np.int64)
