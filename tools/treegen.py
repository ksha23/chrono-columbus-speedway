"""Procedural low-polygon trees for stock Chrono::VSG.

    make_tree(kind, height, crown_radius, seed, triangles=7000) -> {"wood": (V, N, F), "leaves": (V, N, F), "info": {...}}
    write_obj(path, vertices, normals, faces)

Z up, metres, trunk base on the origin, trunk carried 0.3 m below z = 0. Geometry only: the caller
picks the colours. Everything is opaque triangles, and no leaf is longer than 5% of the tree height.

Back faces: Chrono::VSG culls them, so a leaf is drawn from one side only. Each leaf is wound to
face out of its foliage clump, which keeps the near side of the crown visible from anywhere, but
the far side then never shows through the gaps and the crown looks thin. The cheap fix is on the
shape, not in the mesh: call SetDoubleFaced(True) on the leaf ChVisualShapeTriangleMesh and both
sides are drawn for the same triangles (SetBackfaceCull(False) does nothing in this build). The
kinds below are tuned for that. double_sided=True is the fallback that needs no flag: it writes
every leaf twice, so the same budget buys half as many leaves.

How a tree is put together:
  1. a crown outline for the kind, scaled to the measured height and crown radius
  2. foliage clumps spread over that outline (each is a flattened ball of leaves at a limb end)
  3. a trunk and limbs grown to reach every clump, thickened by how much wood each part carries
  4. tubes skinned at the best ring spacing the wood share of the triangle budget affords
  5. leaves on the clump shells with every triangle that is left
"""
import numpy as np

from treegen_wood import Skeleton, curve, plan_rings, skin, unit, UP
from treegen_leaves import clump_points, strand_points, orient, leaf_shapes, leaf_mesh

MAX_LEAF = 0.049  # longest leaf dimension over tree height. The hard limit is 0.05.
# Leaf lighting normal: weight of "outward from the crown", "outward from its clump", "its own face".
# The first two make the crown and each clump shade as rounded volumes instead of as loose cards.
SHADE = (0.35, 0.40, 0.25)

# Crown outline
#   crown_base   bottom of the crown as a fraction of height
#   peak         where the crown is widest, 0 at its base and 1 at its top; p_lo and p_hi shape the
#                outline below and above that (2 is an ellipse, lower is more pointed, higher boxier)
#   lobes        how far the outline departs from round, seen from above
# Clumps
#   t_min        lowest part of the outline that carries a clump (the rest is open underneath)
#   clumps, flat number of clumps for a crown of ordinary width, and clump height over width
#   clump_size   clump radius over the spacing between clumps (0.5 just touching, more overlaps)
# Wood
#   stems        1 for a trunk, more for a shrub's ring of stems
#   trunk_top    where the trunk stops, as a fraction of height; attach_min is the lowest limb
#   bf           0 makes every limb run back toward the trunk, 1 chains limbs end to end
#   cos_min      limbs leaving their parent at more than acos(cos_min) are penalised
#   fork, lift   how much a limb follows its parent at first, and how much its end turns upward
#   wiggle       sideways wander of a limb as a share of its length
#   wood, girth  most of the triangle budget the wood may take, and a scale on the trunk radius
# Leaves
#   corners      3 for triangle leaves, 4 for creased kites (half as many for the same budget)
#   aspect       leaf width over length
#   cover        leaf area over the clump area it has to cover; leaves shrink below MAX_LEAF when
#                the budget covers more than this
#   depth        how far into a clump its leaves reach (0 a thin shell, 1 to the centre)
#   inner_keep   share kept of the leaves that face back into the crown
#   jitter       scatter of the leaf facing direction
#   flatten      below 1 turns leaf faces toward the horizon, where the viewer is
#   droop        how strongly leaf tips point down
#   grade        leaf size falls by this share toward the crown base and rises by it toward the top
#   strands      share of the leaves strung on hanging strands (willow), with their own aspect
_BASE = dict(lobes=1.0, clump_size=0.62, stems=1, wiggle=0.05, girth=1.0, corners=3, cover=1.0, depth=0.35,
             inner_keep=0.4, jitter=0.5, flatten=0.7, droop=1.0, grade=0.0, strands=0.0, strand_aspect=0.5)
KINDS = {
    "broadleaf": dict(_BASE, crown_base=0.26, peak=0.42, p_lo=2.0, p_hi=2.2, t_min=0.10, clumps=16, flat=0.80,
                      trunk_top=0.47, attach_min=0.24, bf=0.55, cos_min=0.2, fork=0.45, lift=0.5,
                      wood=0.18, aspect=0.85, grade=0.25),
    "upright": dict(_BASE, crown_base=0.20, peak=0.36, p_lo=1.8, p_hi=1.5, t_min=0.08, clumps=15, flat=1.15,
                    trunk_top=0.93, attach_min=0.18, bf=0.35, cos_min=0.72, fork=0.5, lift=1.2,
                    wood=0.18, aspect=0.8, wiggle=0.03, girth=0.9, grade=0.25),
    "willow": dict(_BASE, crown_base=0.10, peak=0.55, p_lo=3.0, p_hi=2.0, t_min=0.50, clumps=12, flat=0.65,
                   trunk_top=0.40, attach_min=0.20, bf=0.5, cos_min=0.1, fork=0.5, lift=0.0,
                   wood=0.17, aspect=0.5, wiggle=0.06, girth=1.3, jitter=0.4, droop=2.0, strands=0.55,
                   strand_aspect=0.45, cover=1.1, clump_size=0.72),
    "shrub": dict(_BASE, crown_base=0.02, peak=0.5, p_lo=2.6, p_hi=2.5, t_min=0.12, clumps=10, flat=0.9,
                  trunk_top=0.08, attach_min=0.03, bf=0.6, cos_min=0.1, fork=0.6, lift=0.6,
                  wood=0.10, aspect=1.0, stems=6, girth=0.75, jitter=0.35, droop=0.6, cover=1.3, lobes=1.8,
                  inner_keep=0.5, clump_size=0.8, depth=0.75, flatten=0.4),
}

# (ring spacing in metres, most sides on a tube), best first. The first one the wood budget affords is used.
WOOD_QUALITY = [(0.30, 6), (0.33, 6), (0.36, 6), (0.40, 5), (0.44, 5), (0.50, 5), (0.56, 4), (0.64, 4), (0.75, 4),
                (0.9, 3), (1.1, 3), (1.4, 3), (2.0, 3), (3.0, 3)]


class Crown:
    """The crown outline: radius by height and bearing, with a couple of random lobes."""

    def __init__(self, k, height, radius, rng):
        self.zb, self.zt, self.r = k["crown_base"] * height, float(height), float(radius)
        self.peak, self.p_lo, self.p_hi = k["peak"], k["p_lo"], k["p_hi"]
        self.lobe_amp = rng.uniform(0.04, 0.13, 2) * k["lobes"]
        self.lobe_phase = rng.uniform(0.0, 2 * np.pi, 2)

    def surface(self, t, az):
        lo = np.clip((self.peak - t) / self.peak, 0, 1) ** self.p_lo
        hi = np.clip((t - self.peak) / (1 - self.peak), 0, 1) ** self.p_hi
        f = np.where(t < self.peak, (1 - lo) ** (1 / self.p_lo), (1 - hi) ** (1 / self.p_hi))
        f = f * (1 + self.lobe_amp[0] * np.cos(2 * az + self.lobe_phase[0])
                 + self.lobe_amp[1] * np.cos(3 * az + self.lobe_phase[1]))
        return np.stack([self.r * f * np.cos(az), self.r * f * np.sin(az), self.zb + t * (self.zt - self.zb)], 1)

    def outward(self, p):
        zc = self.zb + self.peak * (self.zt - self.zb)
        half = np.where(p[:, 2] > zc, self.zt - zc, zc - self.zb)
        return unit(np.stack([p[:, 0], p[:, 1], (p[:, 2] - zc) * (self.r / half) ** 2], 1))


def _layout_clumps(crown, k, count, rng):
    """Clump centres and radii: evenly spread points on the outline, each pulled a little inside."""
    m = 40 * count
    cand = crown.surface(rng.uniform(k["t_min"], 0.97, m), rng.uniform(0, 2 * np.pi, m))
    pick = [int(np.argmax(cand[:, 2]))]
    d = np.linalg.norm(cand - cand[pick[0]], axis=1)
    for _ in range(count - 1):  # farthest point sampling
        pick.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(cand - cand[pick[-1]], axis=1))
    pts = cand[pick]
    gap = np.sort(np.linalg.norm(pts[:, None] - pts[None], axis=2), axis=1)[:, 1:3].mean(axis=1)
    size = k["clump_size"] * gap * rng.uniform(0.8, 1.2, count)
    centres = pts - crown.outward(pts) * (0.55 * size)[:, None]
    return centres, np.stack([size, size, size * k["flat"]], axis=1)


def _fit(crown, centres, radii, height, crown_radius):
    """Stretch the layout so the foliage spans the measured height and crown radius.

    Leaf centres sit on the clump shells, so a little is held back for the leaves themselves.
    """
    margin = 0.25 * MAX_LEAF * height
    sx = (crown_radius - min(margin, 0.3 * crown_radius)) / (np.linalg.norm(centres[:, :2], axis=1) + radii[:, 0]).max()
    sz = (height - margin) / (centres[:, 2] + radii[:, 2]).max()
    s = np.array([sx, sx, sz])
    crown.r, crown.zb, crown.zt = crown.r * sx, crown.zb * sz, crown.zt * sz
    return centres * s, radii * s


def _roots(skel, k, height, crown_radius, rng):
    """One trunk, or a ring of stems for a shrub."""
    z_top, z_open = k["trunk_top"] * height, k["attach_min"] * height
    if k["stems"] == 1:
        lean = rng.normal(0.0, 0.015 * height, 2)
        bases, tops = np.array([[0.0, 0.0, -0.3]]), np.array([[lean[0], lean[1], z_top]])
    else:
        az = rng.uniform(0, 2 * np.pi) + 2 * np.pi * (np.arange(k["stems"]) + rng.uniform(-0.3, 0.3, k["stems"])) / k["stems"]
        ring = np.stack([np.cos(az), np.sin(az)], 1) * crown_radius * rng.uniform(0.05, 0.10, (k["stems"], 1))
        bases = np.concatenate([ring, np.full((len(az), 1), -0.3)], 1)
        tops = np.concatenate([ring * 2.2, np.full((len(az), 1), z_top)], 1)
    for a, b in zip(bases, tops):
        pts = curve(a, UP, b, rng, skel.step, fork=1.0, lift=1.0, wiggle=0.012)
        skel.add(pts, open_from=int(np.searchsorted(pts[:, 2], z_open)))


def make_tree(kind, height, crown_radius, seed, triangles=7000, double_sided=False, leaf_corners=None):
    """Build one tree. See the module docstring for the conventions.

    kind           "broadleaf", "upright", "willow" or "shrub"
    height         metres, ground to the top of the crown
    crown_radius   metres, the measured half-width of the crown
    seed           int; the same arguments always give the same tree
    triangles      total budget for wood plus leaves, met to within a triangle or two
    double_sided   write every leaf with both windings (see the module docstring)
    leaf_corners   3 or 4 to override the kind's leaf outline (triangle or creased kite)

    The crown shape and the limbs depend on kind, height, crown_radius and seed only, so the same
    tree at a smaller budget is the same tree with coarser limbs and fewer leaves.
    "info" reports leaves, longest_leaf (metres), clumps and ring_spacing (metres).
    """
    k = dict(KINDS[kind], corners=leaf_corners or KINDS[kind]["corners"])
    code = sorted(KINDS).index(kind)
    rng_shape, rng_wood, rng_leaf = (np.random.default_rng([int(seed), code, i]) for i in range(3))

    crown = Crown(k, height, crown_radius, rng_shape)
    count = int(np.clip(round(k["clumps"] * np.sqrt(crown_radius / (0.36 * height))), 6, 26))
    centres, radii = _fit(crown, *_layout_clumps(crown, k, count, rng_shape), height, crown_radius)

    skel = Skeleton(step=float(np.clip(0.012 * height, 0.06, 0.18)))
    _roots(skel, k, height, crown_radius, rng_shape)
    tips = centres + crown.outward(centres) * 0.35 * radii
    skel.grow(tips, rng_shape, bf=k["bf"], cos_min=k["cos_min"], fork=k["fork"], lift=k["lift"], wiggle=k["wiggle"])
    if k["trunk_top"] < 0.9:
        skel.finish_roots()

    r_base = 0.0105 * height ** 1.2 * k["girth"] * float(np.clip(np.sqrt(crown_radius / (0.33 * height)), 0.8, 1.25))
    limb_radii = skel.radii(r_base, r_tip=0.005 + 0.0018 * height)
    for ds, sides in WOOD_QUALITY:  # the best ring spacing the wood share of the budget affords
        plan, wood_tris = plan_rings(skel, limb_radii, ds, sides)
        if wood_tris <= k["wood"] * triangles:
            break
    wood = skin(skel, plan, rng_wood, flare_height=0.05 * height, flare=0.5 if k["stems"] == 1 else 0.0)

    per_leaf = (k["corners"] - 2) * (2 if double_sided else 1)
    n = max(1, (triangles - wood_tris) // per_leaf)
    leaves, longest = _foliage(k, crown, centres, radii, n, height, rng_leaf, double_sided)
    info = dict(leaves=n, longest_leaf=longest, clumps=count, ring_spacing=ds)
    return {"wood": wood, "leaves": leaves, "info": info}


def _foliage(k, crown, centres, radii, n, height, rng, double_sided):
    """Place n leaves, size them to cover the clumps, turn them, and build the polygons."""
    n_strand = int(k["strands"] * n)
    pts_s = np.zeros((0, 3))
    if n_strand:  # willow: leaves strung down strands that hang from the rim of the dome
        cand, out, _ = clump_points(centres, radii, 4 * len(centres) * 12, rng, depth=0.1)
        rim = cand[(out[:, 2] < 0.35) & (out[:, 2] > -0.6)]
        z_end = crown.zb + rng.uniform(0.0, 0.3, len(rim)) * (rim[:, 2] - crown.zb)
        spacing = 0.55 * MAX_LEAF * height
        want = int(np.ceil(n_strand / max(1.0, ((rim[:, 2] - z_end) / spacing).mean())))
        sel = rng.choice(len(rim), size=want, replace=want > len(rim))
        pts_s = strand_points(rim[sel], z_end[sel], n_strand, spacing, rng)
        reach = np.linalg.norm(pts_s[:, :2], axis=1, keepdims=True)
        pts_s[:, :2] *= np.minimum(1.0, crown.r / np.maximum(reach, 1e-9))  # strands sway, but not past the crown

    def grade(p):  # leaf size by height in the crown: smaller low down, larger at the top
        return 1.0 + k["grade"] * (2.0 * np.clip((p[:, 2] - crown.zb) / (crown.zt - crown.zb), 0.0, 1.0) - 1.0)

    pts_c, out_c, area = clump_points(centres, radii, n - len(pts_s), rng, crown.outward, inner_keep=k["inner_keep"],
                                      z_floor=0.045 * height, density=lambda p: grade(p) ** -2, depth=k["depth"])
    if n_strand:
        area += 2 * np.pi * crown.r * (crown.zt - crown.zb) * 0.5
    pts = np.concatenate([pts_c, pts_s])

    aspect = np.where(np.arange(n) < len(pts_c), k["aspect"], k["strand_aspect"])
    shapes, unit_area = leaf_shapes(n, rng, k["corners"], aspect)
    size, graded = rng.uniform(0.72, 1.0, n), grade(pts)
    length = np.sqrt(k["cover"] * area / (unit_area * (size * graded) ** 2).sum())
    length = np.minimum(length * graded, MAX_LEAF * height) * size

    radial = unit(pts_s * np.array([1.0, 1.0, 0.0]))
    out = np.concatenate([out_c, radial])
    facing, axis = orient(out, rng, jitter=k["jitter"], droop=k["droop"], flatten=k["flatten"])
    if n_strand:
        f_s, a_s = orient(radial, rng, jitter=0.45, droop=4.0, tilt_up=0.0)
        facing[len(pts_c):], axis[len(pts_c):] = f_s, a_s
    round_out = crown.outward(pts)
    round_out[len(pts_c):] = unit(radial + 0.35 * UP)
    w_round, w_clump, w_face = SHADE
    smooth = unit(w_round * round_out + w_clump * out) * (1.0 - w_face)
    mesh = leaf_mesh(pts, facing, axis, length, shapes, smooth, rng, face_weight=w_face, double_sided=double_sided)
    return mesh, float(length.max())


def write_obj(path, vertices, normals, faces):
    """Plain Wavefront OBJ: v and vn lines, faces as v//vn, no materials."""
    f = np.asarray(faces) + 1
    with open(path, "w") as out:
        out.write("".join("v %.4f %.4f %.4f\n" % tuple(v) for v in np.asarray(vertices)))
        out.write("".join("vn %.4f %.4f %.4f\n" % tuple(v) for v in np.asarray(normals)))
        out.write("".join("f %d//%d %d//%d %d//%d\n" % (a, a, b, b, c, c) for a, b, c in f))
