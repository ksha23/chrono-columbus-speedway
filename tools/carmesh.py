"""Triangle-mesh helpers for the parked-vehicle models: read, orient, cut, make simple shapes, write.

A mesh here is triangle soup: P is (T, 3, 3), the three corner positions of T triangles, and
N is (T, 3, 3), a unit normal for each corner. Corners that agree on both become one vertex
when the mesh is written. Everything is numpy, and nothing depends on set or dictionary order,
so the same input gives the same bytes.
"""
import numpy as np


# ---------------------------------------------------------------- reading

def load_obj(path):
    """A Wavefront OBJ as triangle soup.

    Returns a dictionary: P (T, 3, 3), N (T, 3, 3) or None when the file has no normals,
    UV (T, 3, 2) or None, material (T,) as an index into materials (a list of names), and
    mtllib, the material file it names, or None.
    Polygons are fanned from their first corner.
    """
    v, vn, vt = [], [], []
    corners, material_of = [], []
    materials, current, library = [], -1, None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("mtllib"):
                library = line.split(None, 1)[1].strip()
            elif line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("vn "):
                vn.append(line.split()[1:4])
            elif line.startswith("vt "):
                vt.append(line.split()[1:3])
            elif line.startswith("usemtl"):
                name = line.split(None, 1)[1].strip()
                if name not in materials:
                    materials.append(name)
                current = materials.index(name)
            elif line.startswith("f "):
                ring = []
                for token in line.split()[1:]:
                    part = token.split("/")
                    a = int(part[0])
                    b = int(part[1]) if len(part) > 1 and part[1] else 0
                    c = int(part[2]) if len(part) > 2 and part[2] else 0
                    ring.append((a + len(v) + 1 if a < 0 else a,
                                 b + len(vt) + 1 if b < 0 else b,
                                 c + len(vn) + 1 if c < 0 else c))
                for k in range(1, len(ring) - 1):
                    corners.append((ring[0], ring[k], ring[k + 1]))
                    material_of.append(current)
    if not materials:
        materials = [""]
    v = np.array(v, dtype=np.float64).reshape(-1, 3)
    index = np.array(corners, dtype=np.int64).reshape(-1, 3, 3)
    mesh = {"P": v[index[:, :, 0] - 1], "N": None, "UV": None,
            "material": np.maximum(np.array(material_of, dtype=np.int64), 0), "materials": materials,
            "mtllib": library}
    if vn and (index[:, :, 2] > 0).all():
        mesh["N"] = unit(np.array(vn, dtype=np.float64)[index[:, :, 2] - 1])
    if vt and (index[:, :, 1] > 0).all():
        mesh["UV"] = np.array(vt, dtype=np.float64)[index[:, :, 1] - 1]
    return mesh


def load_mtl(path):
    """The materials of an MTL file: name to a dictionary of its statements, numbers parsed."""
    out, current = {}, None
    with open(path, errors="replace") as f:
        for line in f:
            word = line.split()
            if not word or word[0].startswith("#"):
                continue
            if word[0] == "newmtl":
                current = out.setdefault(line.split(None, 1)[1].strip(), {})
            elif current is not None:
                try:
                    current[word[0]] = [float(x) for x in word[1:]]
                except ValueError:
                    current[word[0]] = line.split(None, 1)[1].strip()
    return out


# ---------------------------------------------------------------- basics

# Seven points on a triangle, as weights of its corners: the middle, three towards the corners and three near them.
SAMPLES = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
                     [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])


def unit(a):
    length = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(length, 1e-20)


def face_normals(P):
    """Unit normal by winding, and area, of each triangle."""
    cross = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    length = np.linalg.norm(cross, axis=1)
    return cross / np.maximum(length, 1e-20)[:, None], 0.5 * length


def weld(P, tolerance=2e-5):
    """An integer id for each corner, equal for corners at the same place. Returns (ids (T, 3), count)."""
    key = np.round(P.reshape(-1, 3) / tolerance).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return inverse.reshape(-1, 3), int(inverse.max()) + 1 if len(inverse) else 0


def first_of_each(P):
    """Indices of the triangles left when exact repeats, same corners and same facing, are
    dropped. Some source files hold every face twice."""
    ids, _ = weld(P)
    turn = np.argmin(ids, axis=1)
    key = np.stack([ids[np.arange(len(ids)), (turn + k) % 3] for k in range(3)], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    return np.sort(first)


def quarter(*arrays):
    """Each triangle into four, through the middles of its edges. Every array is (T, 3, n)
    and is cut the same way."""
    out = []
    for a in arrays:
        m01, m12, m20 = (a[:, 0] + a[:, 1]) / 2, (a[:, 1] + a[:, 2]) / 2, (a[:, 2] + a[:, 0]) / 2
        out.append(np.concatenate([np.stack([a[:, 0], m01, m20], axis=1), np.stack([m01, a[:, 1], m12], axis=1),
                                   np.stack([m20, m12, a[:, 2]], axis=1), np.stack([m01, m12, m20], axis=1)]))
    return out


def clip(P, N, axis, value, keep_above=True):
    """The parts of triangles on one side of the plane where coordinate axis equals value.

    Returns (P, N, index), index giving the triangle each piece came from. New corners lie
    exactly on the plane.
    """
    distance = (P[:, :, axis] - value) * (1.0 if keep_above else -1.0)
    inside = distance >= 0
    count = inside.sum(axis=1)
    out_P, out_N, out_index = [P[count == 3]], [N[count == 3]], [np.flatnonzero(count == 3)]
    for wanted in (1, 2):
        chosen = np.flatnonzero(count == wanted)
        if not len(chosen):
            continue
        # Turn each triangle so that its odd corner, the one alone on its side, comes first.
        odd = np.argmax(inside[chosen], axis=1) if wanted == 1 else np.argmin(inside[chosen], axis=1)
        order = (odd[:, None] + np.arange(3)[None, :]) % 3
        rows = chosen[:, None]
        p, n, d = P[rows, order], N[rows, order], distance[rows, order]
        t_ab = (d[:, 0] / (d[:, 0] - d[:, 1]))[:, None]
        t_ac = (d[:, 0] / (d[:, 0] - d[:, 2]))[:, None]
        p_ab, p_ac = p[:, 0] + t_ab * (p[:, 1] - p[:, 0]), p[:, 0] + t_ac * (p[:, 2] - p[:, 0])
        n_ab, n_ac = n[:, 0] + t_ab * (n[:, 1] - n[:, 0]), n[:, 0] + t_ac * (n[:, 2] - n[:, 0])
        p_ab[:, axis] = value
        p_ac[:, axis] = value
        if wanted == 1:
            out_P.append(np.stack([p[:, 0], p_ab, p_ac], axis=1))
            out_N.append(np.stack([n[:, 0], n_ab, n_ac], axis=1))
            out_index.append(chosen)
        else:
            out_P += [np.stack([p_ab, p[:, 1], p[:, 2]], axis=1), np.stack([p_ab, p[:, 2], p_ac], axis=1)]
            out_N += [np.stack([n_ab, n[:, 1], n[:, 2]], axis=1), np.stack([n_ab, n[:, 2], n_ac], axis=1)]
            out_index += [chosen, chosen]
    return np.concatenate(out_P), unit(np.concatenate(out_N)), np.concatenate(out_index)


def fan(outline, centre=None):
    """A flat polygon as triangles about its middle. outline is (n, 3), in order."""
    outline = np.asarray(outline, dtype=np.float64)
    middle = outline.mean(axis=0) if centre is None else np.asarray(centre, dtype=np.float64)
    return np.stack([np.repeat(middle[None, :], len(outline), axis=0), outline, np.roll(outline, -1, axis=0)], axis=1)


def facing(P, direction):
    """The same triangles, each wound to face along direction ((3,) or one per triangle)."""
    fn, _ = face_normals(P)
    wrong = (fn * np.asarray(direction, dtype=np.float64)).sum(axis=-1) < 0
    P = P.copy()
    P[wrong] = P[wrong][:, ::-1]
    return P


def flip(P, N):
    """The same triangles facing the other way."""
    return P[:, ::-1].copy(), -N[:, ::-1]


def wind_to_normals(P, N):
    """Reverse the winding of triangles whose corners' normals say they face the other way."""
    geometric, _ = face_normals(P)
    wrong = (geometric * N.sum(axis=1)).sum(axis=1) < 0
    P = P.copy()
    N = N.copy()
    P[wrong] = P[wrong][:, ::-1]
    N[wrong] = N[wrong][:, ::-1]
    return P, N, int(wrong.sum())


def corner_angles(P):
    a = unit(np.roll(P, -1, axis=1) - P)
    b = unit(np.roll(P, 1, axis=1) - P)
    return np.arccos(np.clip((a * b).sum(axis=-1), -1.0, 1.0))


def smooth_normals(P, crease_degrees=40.0):
    """Corner normals from the geometry alone: faces around a point are blended when they
    meet at less than the crease angle, and kept apart otherwise."""
    fn, _ = face_normals(P)
    ids, _ = weld(P)
    ids = ids.reshape(-1)
    weight = corner_angles(P).reshape(-1)
    face = np.repeat(np.arange(len(P)), 3)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    start = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    size = np.diff(np.r_[start, len(order)])
    limit = np.cos(np.radians(crease_degrees))
    out = np.zeros((len(order), 3))
    for g in np.unique(size):
        first = start[size == g]
        for c0 in range(0, len(first), max(1, 400000 // (int(g) * int(g)))):
            block = first[c0:c0 + max(1, 400000 // (int(g) * int(g)))]
            member = order[block[:, None] + np.arange(g)[None, :]]            # (G, g) corner numbers
            n = fn[face[member]]                                              # (G, g, 3)
            near = np.einsum("gik,gjk->gij", n, n) >= limit
            out[member] = np.einsum("gij,gj,gjk->gik", near, weight[member], n)
    out = out.reshape(-1, 3, 3)
    flat = np.linalg.norm(out, axis=-1) < 1e-12
    out[flat] = np.repeat(fn[:, None, :], 3, axis=1)[flat]
    return unit(out)


def rotate_z(P, angle):
    c, s = np.cos(angle), np.sin(angle)
    return P @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def rotate_y(P, angle):
    """Rotate about y by angle: a point ahead of the origin drops when the angle is positive."""
    c, s = np.cos(angle), np.sin(angle)
    return P @ np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def box(low, high, step=None, faces="xXyYzZ"):
    """The faces of an axis-aligned box as outward-facing triangles: x and X are the low and
    high x faces, and so on. With step, each face is cut into squares about that size."""
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    out = []
    for name in faces:
        axis = "xyz".index(name.lower())
        a, b = [k for k in range(3) if k != axis]
        na = 1 if step is None else max(1, int(np.ceil((high[a] - low[a]) / step)))
        nb = 1 if step is None else max(1, int(np.ceil((high[b] - low[b]) / step)))
        ua = np.linspace(low[a], high[a], na + 1)
        ub = np.linspace(low[b], high[b], nb + 1)
        for i in range(na):
            for j in range(nb):
                corner = np.zeros((4, 3))
                corner[:, axis] = high[axis] if name.isupper() else low[axis]
                corner[:, a] = [ua[i], ua[i + 1], ua[i + 1], ua[i]]
                corner[:, b] = [ub[j], ub[j], ub[j + 1], ub[j + 1]]
                out += [corner[[0, 1, 2]], corner[[0, 2, 3]]]
    P = np.array(out)
    fn, _ = face_normals(P)
    centre = (low + high) / 2
    wrong = (fn * (P.mean(axis=1) - centre)).sum(axis=1) < 0
    P[wrong] = P[wrong][:, ::-1]
    return P


# ---------------------------------------------------------------- shapes

def hull_2d(points):
    """Convex hull of 2-D points, counter-clockwise, starting from the lowest of the leftmost."""
    pts = np.unique(np.round(np.asarray(points, dtype=np.float64), 6), axis=0)
    pts = [tuple(p) for p in pts]

    def half(sequence):
        out = []
        for p in sequence:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def simplify_closed(points, count):
    """Drop the corners of a closed polygon that matter least until count are left."""
    pts = [tuple(p) for p in points]
    while len(pts) > count:
        n = len(pts)
        loss = []
        for k in range(n):
            a, b, c = pts[k - 1], pts[k], pts[(k + 1) % n]
            loss.append(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))
        del pts[int(np.argmin(loss))]
    return np.array(pts)


def lathe(profile, segments, closed=True, inside=None):
    """Turn a profile of (y, r) points about the y axis. The surface faces away from the
    point inside, a (y, r) pair, which is the middle of the profile unless given."""
    profile = np.asarray(profile, dtype=np.float64)
    angle = 2.0 * np.pi * np.arange(segments + 1) / segments
    angle[-1] = 0.0
    ring = np.stack([profile[:, 1, None] * np.cos(angle)[None, :],
                     np.repeat(profile[:, 0, None], segments + 1, axis=1),
                     profile[:, 1, None] * np.sin(angle)[None, :]], axis=-1)       # (points, segments + 1, 3)
    pairs = [(k, (k + 1) % len(profile)) for k in range(len(profile) if closed else len(profile) - 1)]
    triangles = []
    for k0, k1 in pairs:
        a, b = ring[k0, :-1], ring[k0, 1:]
        c, d = ring[k1, :-1], ring[k1, 1:]
        if profile[k0, 1] > 1e-9:
            triangles.append(np.stack([a, b, c], axis=1))
        if profile[k1, 1] > 1e-9:
            triangles.append(np.stack([b, d, c], axis=1))
    P = np.concatenate(triangles)
    middle = profile.mean(axis=0) if inside is None else np.asarray(inside, dtype=np.float64)
    centre = P.mean(axis=1)
    radius = np.hypot(centre[:, 0], centre[:, 2])
    outward = np.stack([(radius - middle[1]) * centre[:, 0] / np.maximum(radius, 1e-12),
                        centre[:, 1] - middle[0],
                        (radius - middle[1]) * centre[:, 2] / np.maximum(radius, 1e-12)], axis=1)
    fn, _ = face_normals(P)
    wrong = (fn * outward).sum(axis=1) < 0
    P[wrong] = P[wrong][:, ::-1]
    return P


def disc(y, radius, segments, facing):
    """A flat disc across the y axis, facing +y (facing = 1) or -y (facing = -1)."""
    angle = 2.0 * np.pi * np.arange(segments + 1) / segments
    angle[-1] = 0.0
    rim = np.stack([radius * np.cos(angle), np.full(segments + 1, y), radius * np.sin(angle)], axis=1)
    centre = np.repeat(np.array([[0.0, y, 0.0]]), segments, axis=0)
    P = np.stack([centre, rim[1:], rim[:-1]], axis=1)
    fn, _ = face_normals(P)
    if (fn[0, 1] > 0) != (facing > 0):
        P = P[:, ::-1].copy()
    return P


# ---------------------------------------------------------------- writing

def write_obj(path, P, N, comments):
    """Write one mesh as v, vn and f a//a b//b c//c lines. Returns the triangle count written.

    Positions are kept to 0.01 mm and normals to four places. Corners equal in both become
    one vertex, numbered in order of first use. Triangles that collapse to a line are dropped.
    """
    key = np.concatenate([np.round(P.reshape(-1, 3) * 1e5), np.round(unit(N.reshape(-1, 3)) * 1e4)], axis=1).astype(np.int64)
    unique, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    unique = unique[order]
    faces = rank[inverse].reshape(-1, 3)
    place = np.round(P.reshape(-1, 3) * 1e5).astype(np.int64).reshape(-1, 3, 3)
    solid = ~((place[:, 0] == place[:, 1]).all(axis=1) | (place[:, 1] == place[:, 2]).all(axis=1)
              | (place[:, 0] == place[:, 2]).all(axis=1))
    faces = faces[solid]
    lines = ["# " + c for c in comments]
    lines += ["v %.5f %.5f %.5f" % (x / 1e5, y / 1e5, z / 1e5) for x, y, z in unique[:, :3].tolist()]
    lines += ["vn %.4f %.4f %.4f" % (x / 1e4, y / 1e4, z / 1e4) for x, y, z in unique[:, 3:].tolist()]
    lines += ["f %d//%d %d//%d %d//%d" % (a, a, b, b, c, c) for a, b, c in (faces + 1).tolist()]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return int(len(faces))
