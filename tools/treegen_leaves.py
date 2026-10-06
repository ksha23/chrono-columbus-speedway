"""Foliage for treegen: where leaves go, how they are turned, and the polygons themselves.

Leaves sit on the outer shells of foliage clumps (one clump per limb end) and face mostly outward
from their clump. The renderer culls back faces, so a single-faced leaf is only seen from the
side it faces. Shells that face outward put every leaf where some viewpoint sees it.
"""
import numpy as np

from treegen_wood import unit, UP

GOLDEN = np.pi * (3.0 - np.sqrt(5.0))


def _spread_dirs(n, rng):
    """n well spread unit vectors: a Fibonacci spiral under a random rotation."""
    i = np.arange(n)
    z = 1.0 - (2.0 * i + 1.0) / max(n, 1)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    d = np.stack([r * np.cos(i * GOLDEN), r * np.sin(i * GOLDEN), z], axis=1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return d @ q.T


def clump_points(centres, radii, n, rng, crown_out=None, depth=0.35, bury=0.9, inner_keep=1.0, z_floor=-1e9,
                 density=None):
    """n leaf positions on the clump shells, skipping the parts buried inside a neighbour.

    crown_out(points) gives the outward direction of the whole crown. Shell points that face
    back into the crown are thinned to inner_keep of their number: from outside they are hidden
    behind the outer leaves, so most of that budget is better spent where it shows.
    density(points) is an optional relative leaf density, for packing more, smaller leaves low
    in the crown where a viewer on the ground gets close.
    Returns positions, the outward direction of the owning clump at each, and the shell area
    the kept leaves have to cover (used to size them).
    """
    k = len(centres)
    weight = radii[:, 0] * (radii[:, 0] + 2.0 * radii[:, 2])
    counts = np.maximum(3, np.round(2.5 * n * weight / weight.sum()).astype(int))
    pts, out, owner = [], [], []
    for c in range(k):
        d = _spread_dirs(counts[c], rng)
        rho = 1.0 - depth * rng.uniform(0.0, 1.0, counts[c]) ** 1.5
        pts.append(centres[c] + d * radii[c] * rho[:, None])
        out.append(unit(d / radii[c]))
        owner.append(np.full(counts[c], c))
    pts, out, owner = np.concatenate(pts), np.concatenate(out), np.concatenate(owner)
    q = np.linalg.norm((pts[:, None, :] - centres[None]) / radii[None], axis=2)
    q[np.arange(len(pts)), owner] = 9.0
    score = np.minimum(q.min(axis=1), bury)  # below bury means inside some other clump
    score = np.where(pts[:, 2] < z_floor, 0.0, score)
    share = np.where(score >= bury, 1.0, 0.0)
    if crown_out is not None and inner_keep < 1.0:
        inner = (out * crown_out(pts)).sum(axis=1) < -0.15
        drop = inner & (rng.uniform(size=len(pts)) > inner_keep)
        score = np.where(drop, 0.5 * score, score)
        share = np.where(inner, inner_keep, 1.0) * share
    shell_area = 4.0 * np.pi * weight / 3.0
    area = float((shell_area * np.bincount(owner, share, k) / counts).sum())
    key = rng.uniform(1e-9, 1.0, len(pts))
    if density is not None:  # weighted sampling without replacement
        key = key ** (1.0 / density(pts))
    order = np.argsort(-np.where(score >= bury, 10.0 + key, score), kind="stable")
    keep = np.sort(order[:n])
    return pts[keep], out[keep], area


def strand_points(starts, z_end, n, spacing, rng, sway=0.05):
    """Leaf positions strung down hanging strands (willow), at most n of them."""
    drop = np.maximum(starts[:, 2] - z_end, spacing)
    per = np.maximum(1, (drop / spacing).astype(int))
    take = np.cumsum(per) <= n
    take[0] = True
    starts, per = starts[take], per[take]
    sid = np.repeat(np.arange(len(starts)), per)
    step = np.concatenate([np.arange(m) for m in per]) + rng.uniform(0.2, 0.8, len(sid))
    lean = rng.normal(0.0, sway, (len(starts), 2))
    pts = starts[sid].copy()
    pts[:, 2] -= step * spacing
    pts[:, :2] += lean[sid] * (step * spacing)[:, None] + rng.normal(0.0, 0.25 * spacing, (len(sid), 2))
    return pts[:n]


def orient(out, rng, jitter=0.5, droop=0.5, tilt_up=0.15, flatten=1.0):
    """Facing direction and long axis per leaf: facing out of the clump, tips hanging down.

    flatten below 1 turns the faces toward the horizon, which is where a viewer on the ground
    is: leaves that face straight up or down are seen edge on from there.
    """
    n = len(out)
    facing = unit(out * np.array([1.0, 1.0, flatten]) + jitter * rng.normal(size=(n, 3)) + tilt_up * UP)
    want = rng.normal(size=(n, 3)) + out - 2.0 * droop * UP
    axis = want - (want * facing).sum(axis=1, keepdims=True) * facing
    return facing, unit(axis)


def leaf_shapes(n, rng, corners, aspect, widest=0.3):
    """Leaf outlines in the leaf's own plane, as (along the leaf, across it), base at the origin.

    corners=3 is a triangle with its point at the tip, corners=4 a teardrop kite. Each outline
    is scaled so its longest dimension is exactly 1. Returns (n, corners, 2) and the areas.
    """
    hw = 0.5 * aspect * rng.uniform(0.8, 1.0, n)
    skew = rng.uniform(-0.1, 0.1, n)
    zero, one = np.zeros(n), np.ones(n)
    if corners == 3:
        lop = rng.uniform(-0.22, 0.22, n)  # one base corner sits further along, so no two leaves match
        pts = np.stack([np.stack([np.maximum(lop, 0), -hw], 1), np.stack([one, skew], 1),
                        np.stack([np.maximum(-lop, 0), hw], 1)], axis=1)
    else:
        w = np.clip(widest + rng.uniform(-0.08, 0.08, n), 0.15, 0.6)
        pts = np.stack([np.stack([zero, zero], 1), np.stack([w + skew, -hw], 1),
                        np.stack([one, zero], 1), np.stack([w - skew, hw], 1)], axis=1)
    longest = np.linalg.norm(pts[:, :, None] - pts[:, None], axis=3).max(axis=(1, 2))
    pts = pts / longest[:, None, None]
    x, y = pts[:, :, 0], pts[:, :, 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1))
    return pts, area


def leaf_mesh(centre, facing, axis, length, shapes, smooth, rng, face_weight=0.25, bend=0.5, cup=0.35,
              double_sided=False):
    """Leaf polygons, counter-clockwise seen from the facing side.

    A kite is creased across its widest line and the tip half hangs back by a random angle, so
    it does not read as a flat card. smooth is the part of the lighting normal shared with the
    neighbours (it rounds the crown). Each corner adds face_weight of the true normal there,
    and the two side corners lean their normals apart by cup.
    """
    n, corners = shapes.shape[:2]
    side = np.cross(facing, axis)
    theta = rng.uniform(-0.3, 1.0, n)[:, None] * bend
    tip_axis = axis * np.cos(theta) - facing * np.sin(theta)
    tip_face = facing * np.cos(theta) + axis * np.sin(theta)
    u, v = shapes[:, :, 0, None] * length[:, None, None], shapes[:, :, 1, None] * length[:, None, None]
    origin = (centre - axis * 0.5 * length[:, None])[:, None, :]
    verts = origin + axis[:, None, :] * u + side[:, None, :] * v
    tip = corners - 2  # the tip is corner 1 of a triangle, corner 2 of a kite
    if corners == 4:
        crease = 0.5 * (u[:, 1] + u[:, 3])
        verts[:, tip] = origin[:, 0] + axis * crease + tip_axis * (u[:, tip] - crease)
    both = unit(facing + tip_face)
    lean = cup * side
    corner_normals = [both - lean, tip_face, both + lean] if corners == 3 else [facing, both - lean, tip_face, both + lean]
    norms = np.stack([unit(smooth + face_weight * c) for c in corner_normals], axis=1)
    i = corners * np.arange(n)[:, None]
    faces = i + np.array([0, 1, 2]) if corners == 3 else np.concatenate([i + np.array([0, 1, 3]), i + np.array([1, 2, 3])])
    if double_sided:
        faces = np.concatenate([faces, faces[:, ::-1]])
    return verts.reshape(-1, 3), norms.reshape(-1, 3), faces.astype(np.int64)
