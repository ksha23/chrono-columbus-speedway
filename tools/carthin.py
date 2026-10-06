"""Fewer triangles for a mesh, without letting it show."""
import heapq

import numpy as np

from carmesh import face_normals, unit, weld


def decimate(P, group, normals=None, fixed=None, target=0, deviation=0.002, seam_deviation=0.002, within_degrees=12.0):
    """Fewer triangles by collapsing edges, cheapest first (quadric error, Garland and Heckbert).

    A collapse moves one end of an edge onto the other, so every remaining vertex is one of
    the original ones. The cost is the summed squared distance, weighted by area, to the
    planes of the original triangles the moved vertex stands for; a collapse is refused when
    that comes to more than deviation (metres, root mean square). With normals (T, 3, 3), the
    mesh's own corner normals, it is also refused when a triangle it reshapes would face more
    than within_degrees away from the original normal at any of its corners: a triangle may
    then only span as much curve as that. Flat and gently curved panels thin out, and the
    count stops falling where the shape would start to show it, or at target if that is reached
    first.

    group (T,) keeps kinds of surface apart. A seam is an edge where two groups meet or where
    the surface ends. Car bodies are made of many separate panels, and letting their edges
    drift opens cracks between them. So a seam vertex only moves along its seam, onto a
    neighbour, and only if it lies within seam_deviation of the straight line that replaces
    it; where seams branch or end it stays put. Triangles marked in fixed (T,) are left
    exactly as they are.

    Returns the indices of the kept triangles and, for each, its three corner positions:
    (keep (K,), P (K, 3, 3)).
    """
    ids, count = weld(P)
    position = np.zeros((count, 3))
    position[ids.reshape(-1)] = P.reshape(-1, 3)
    fn, area = face_normals(P)

    plane = np.concatenate([fn, -(fn * P[:, 0]).sum(axis=1)[:, None]], axis=1)
    face_q = area[:, None, None] * plane[:, :, None] * plane[:, None, :]
    Q = np.zeros((count, 4, 4))
    W = np.zeros(count)
    for k in range(3):
        np.add.at(Q, ids[:, k], face_q)
        np.add.at(W, ids[:, k], area)

    # Seams: edges used by one triangle, by more than two, or by triangles of two groups.
    a = ids[:, [0, 1, 2]].reshape(-1)
    b = ids[:, [1, 2, 0]].reshape(-1)
    face = np.repeat(np.arange(len(P)), 3)
    low, high = np.minimum(a, b), np.maximum(a, b)
    key = low * count + high
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    run = np.cumsum(np.r_[True, sorted_key[1:] != sorted_key[:-1]]) - 1
    uses = np.bincount(run)
    group_low = np.full(run.max() + 1, np.iinfo(np.int64).max)
    group_high = np.full(run.max() + 1, -1)
    np.minimum.at(group_low, run, group[face[order]])
    np.maximum.at(group_high, run, group[face[order]])
    seam = order[(uses[run] != 2) | (group_low[run] != group_high[run])]
    along_seam = [[] for _ in range(count)]            # for each vertex, its neighbours along seams
    for x, y in sorted(set(zip(low[seam].tolist(), high[seam].tolist()))):
        if x != y:
            along_seam[x].append(y)
            along_seam[y].append(x)
    pinned = np.zeros(count, dtype=bool)
    if fixed is not None:
        pinned[ids[fixed].reshape(-1)] = True
    pinned = pinned.tolist()

    triangles = [tuple(row) for row in ids.tolist()]
    alive = [t[0] != t[1] and t[1] != t[2] and t[0] != t[2] for t in triangles]
    around = [[] for _ in range(count)]
    for t, tri in enumerate(triangles):
        if alive[t]:
            for vertex in tri:
                around[vertex].append(t)
    position = [tuple(row) for row in position.tolist()]
    Q = [q for q in Q]
    W = W.tolist()
    seam_error = [0.0] * count        # how far the seam at a vertex has already strayed

    def cost(source, onto):
        """The error of moving source onto onto, or None when that is not allowed."""
        if pinned[source]:
            return None
        if along_seam[source]:
            if len(along_seam[source]) != 2 or onto not in along_seam[source]:
                return None
            other = along_seam[source][0] if along_seam[source][1] == onto else along_seam[source][1]
            p, q0, q1 = position[source], position[onto], position[other]
            ex, ey, ez = q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]
            dx, dy, dz = p[0] - q0[0], p[1] - q0[1], p[2] - q0[2]
            length = ex * ex + ey * ey + ez * ez
            along = (dx * ex + dy * ey + dz * ez) / length if length > 0 else -1.0
            if along <= 0.0 or along >= 1.0:
                return None
            off = ((dx - along * ex) ** 2 + (dy - along * ey) ** 2 + (dz - along * ez) ** 2) ** 0.5
            if off + seam_error[source] > seam_deviation:
                return None
        x, y, z = position[onto]
        h = np.array([x, y, z, 1.0])
        error = float(h @ (Q[source] + Q[onto]) @ h)
        if error > deviation ** 2 * (W[source] + W[onto]):
            return None
        return error

    def normal_of(p0, p1, p2):
        ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
        return uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx

    def spoils(source, onto):
        """Would a triangle around source turn over or nearly vanish, or the surface pinch,
        if source moved onto onto?"""
        target = position[onto]
        shared = 0
        for t in around[source]:
            tri = triangles[t]
            if onto in tri:
                shared += 1
                continue
            before = normal_of(*(position[x] for x in tri))
            after = normal_of(*(target if x == source else position[x] for x in tri))
            dot = before[0] * after[0] + before[1] * after[1] + before[2] * after[2]
            size_before = before[0] ** 2 + before[1] ** 2 + before[2] ** 2
            size_after = after[0] ** 2 + after[1] ** 2 + after[2] ** 2
            if dot <= 0.5 * (size_before * size_after) ** 0.5 or size_after < 1e-4 * size_before:
                return True
            for x in tri:
                vertex = onto if x == source else x
                if original[vertex]:
                    best = max(after[0] * n[0] + after[1] * n[1] + after[2] * n[2] for n in original[vertex])
                    if best < closest * size_after ** 0.5:
                        return True
        # The two ends may share only the vertices opposite their common edge.
        near_source = set(x for t in around[source] for x in triangles[t])
        near_onto = set(x for t in around[onto] for x in triangles[t])
        return len((near_source & near_onto) - {source, onto}) > shared

    closest = float(np.cos(np.radians(within_degrees)))
    original = [[] for _ in range(count)]              # the normals the mesh had at each vertex
    if normals is not None:
        seen = sorted(set(zip(ids.reshape(-1).tolist(), map(tuple, np.round(normals.reshape(-1, 3), 3).tolist()))))
        for vertex, normal in seen:
            original[vertex].append(normal)
    heap = []
    stamp = [0] * count

    def push(u, w):
        for source, onto in ((u, w), (w, u)):
            c = cost(source, onto)
            if c is not None:
                heapq.heappush(heap, (c, source, onto, stamp[source], stamp[onto]))

    for x, y in sorted(set(zip(low.tolist(), high.tolist()))):
        if x != y:
            push(x, y)

    remaining = sum(alive)
    while remaining > target and heap:
        _, source, onto, s0, s1 = heapq.heappop(heap)
        if stamp[source] != s0 or stamp[onto] != s1 or not around[source] or not around[onto]:
            continue
        if spoils(source, onto):
            continue
        Q[onto] = Q[onto] + Q[source]
        W[onto] += W[source]
        touched = set()
        for t in around[source]:
            tri = triangles[t]
            if onto in tri:
                alive[t] = False
                remaining -= 1
                for vertex in tri:
                    if vertex != source:
                        around[vertex] = [x for x in around[vertex] if x != t]
            else:
                tri = tuple(onto if x == source else x for x in tri)
                triangles[t] = tri
                around[onto].append(t)
        around[source] = []
        refresh = [onto]
        if along_seam[source]:
            other = along_seam[source][0] if along_seam[source][1] == onto else along_seam[source][1]
            p, q0, q1 = position[source], position[onto], position[other]
            ex, ey, ez = q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]
            dx, dy, dz = p[0] - q0[0], p[1] - q0[1], p[2] - q0[2]
            along = (dx * ex + dy * ey + dz * ez) / (ex * ex + ey * ey + ez * ez)
            off = ((dx - along * ex) ** 2 + (dy - along * ey) ** 2 + (dz - along * ez) ** 2) ** 0.5
            strayed = seam_error[source] + off
            seam_error[onto] = max(seam_error[onto], strayed)
            seam_error[other] = max(seam_error[other], strayed)
            along_seam[onto] = [other if x == source else x for x in along_seam[onto]]
            along_seam[other] = [onto if x == source else x for x in along_seam[other]]
            along_seam[source] = []
            refresh.append(other)
        for vertex in refresh:
            stamp[vertex] += 1
        for vertex in refresh:
            for other in sorted(set(x for t in around[vertex] for x in triangles[t]) - {vertex}):
                push(vertex, other)

    keep = np.array([t for t in range(len(triangles)) if alive[t]], dtype=np.int64)
    position = np.array(position)
    return keep, position[np.array([triangles[t] for t in keep], dtype=np.int64).reshape(-1, 3)]


def carry_normals(P_new, P_old, N_old):
    """Corner normals for a reduced mesh, taken from the original.

    Every corner of the reduced mesh sits on an original vertex, which had one normal or, at
    a crease, several. Each corner takes the one closest to the way its new triangle faces.
    """
    old_ids, count = weld(np.concatenate([P_old, P_new]))
    new_ids = old_ids[len(P_old):].reshape(-1)
    old_ids = old_ids[:len(P_old)].reshape(-1)
    fn, _ = face_normals(P_new)
    want = np.repeat(fn, 3, axis=0)
    order = np.argsort(old_ids, kind="stable")
    start = np.searchsorted(old_ids[order], np.arange(count))
    stop = np.searchsorted(old_ids[order], np.arange(count), side="right")
    normals = N_old.reshape(-1, 3)[order]
    best = np.full(len(new_ids), -2.0)
    out = want.copy()
    for k in range(int((stop - start).max()) if count else 0):
        has = (stop - start)[new_ids] > k
        candidate = normals[np.minimum(start[new_ids] + k, len(normals) - 1)]
        score = (candidate * want).sum(axis=1)
        better = has & (score > best)
        out[better] = candidate[better]
        best[better] = score[better]
    out[best < 0.3] = want[best < 0.3]
    return unit(out).reshape(-1, 3, 3)
