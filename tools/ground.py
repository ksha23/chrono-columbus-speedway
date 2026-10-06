"""The ground, as triangles on the lidar surface: what is drawn and what the wheels touch.

Texture tiles the scan reaches get a 1 m grid. Everything else in the scene rectangle gets 8 m
cells. Where a coarse cell borders a fine tile it is drawn as a fan through the fine vertices
on that edge, so the two resolutions meet without a crack.

The fine grid is cut along the pavement's edge, so every triangle is either road or not. A
triangle the edge passes through is split there, with the new vertices placed on the edge
itself. Road and the rest are then separate meshes that share their boundary vertices exactly.

The same vertices and triangles are written twice: as two welded meshes for collision (road,
and everything else), and split by texture tile, with texture coordinates and normals, for
drawing. So what the car is seen driving on is exactly what it is driving on.
"""
import os

import numpy as np

import layout

SNAP = 0.02   # metres: the edge is never taken closer to a grid vertex than this, to avoid slivers


class Ground:
    def __init__(self, tiles, ref, edge):
        self.tiles = sorted(tuple(t) for t in tiles)
        self.ref = ref
        self.edge = edge
        self.nx = int(layout.SCENE_X1 - layout.SCENE_X0) + 1   # lattice of whole metres
        self.ny = int(layout.SCENE_Y1 - layout.SCENE_Y0) + 1
        self.index = np.full((self.ny, self.nx), -1, np.int64)
        self.count = 0
        self.cuts = {}        # (vertex a, vertex b) -> id of the vertex where the edge crosses a-b
        self.cut_xy = {}      # that vertex id -> (x, y)
        self.road_faces, self.land_faces = {}, {}
        self._build_tiles()
        self.surround_faces = self._build_surround()
        self._finish()

    def _ids(self, ix, iy):
        """Vertex numbers for lattice points, handing out new ones where needed."""
        ix, iy = np.asarray(ix), np.asarray(iy)
        flat = self.index.reshape(-1)
        key = iy * self.nx + ix
        new = np.unique(key[flat[key] < 0])
        flat[new] = self.count + np.arange(len(new))
        self.count += len(new)
        return flat[key]

    def _lattice(self, x, y):
        return int(round(x - layout.SCENE_X0)), int(round(y - layout.SCENE_Y0))

    def _cut(self, a, b, pa, pb, da, db):
        """The vertex where the pavement edge crosses the segment a-b, made once and shared."""
        key = (a, b) if a < b else (b, a)
        v = self.cuts.get(key)
        if v is None:
            t = da / (da - db)
            v = self.count
            self.count += 1
            self.cuts[key] = v
            self.cut_xy[v] = (pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]))
        return v

    def _split(self, ids, pts, d, road, land):
        """Cut one triangle along the edge and hand the pieces to road or land."""
        inside = [x > 0 for x in d]
        if all(inside):
            road.append(ids)
            return
        if not any(inside):
            land.append(ids)
            return
        # One corner is alone on its side. Rotate it to the front.
        lone = inside.index(True) if sum(inside) == 1 else inside.index(False)
        a, b, c = (lone, (lone + 1) % 3, (lone + 2) % 3)
        ab = self._cut(ids[a], ids[b], pts[a], pts[b], d[a], d[b])
        ac = self._cut(ids[a], ids[c], pts[a], pts[c], d[a], d[c])
        tip, rest = (road, land) if inside[lone] else (land, road)
        tip.append((ids[a], ab, ac))
        rest.append((ab, ids[b], ids[c]))
        rest.append((ab, ids[c], ac))

    def _build_tiles(self):
        n = int(layout.TILE / layout.FINE)
        for (i, j) in self.tiles:
            x0, y0, x1, y1 = layout.tile_bounds(i, j)
            if x0 < layout.SCENE_X0 or x1 > layout.SCENE_X1 or y0 < layout.SCENE_Y0 or y1 > layout.SCENE_Y1:
                raise SystemExit(f"tile {(i, j)} sticks out of the scene rectangle")
            ox, oy = self._lattice(x0, y0)
            gx, gy = np.meshgrid(ox + np.arange(n + 1), oy + np.arange(n + 1))
            ids = self._ids(gx, gy)
            px, py = gx + layout.SCENE_X0, gy + layout.SCENE_Y0
            d = self.edge.at(px, py)
            # Keep the edge a little away from grid vertices: a crossing right at a vertex
            # would make a triangle with no area.
            d = np.where(np.abs(d) < SNAP, np.where(d > 0, SNAP, -SNAP), d)
            a, b, c, e = ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]
            pos = d > 0
            corners = pos[:-1, :-1].astype(int) + pos[:-1, 1:] + pos[1:, 1:] + pos[1:, :-1]
            both = np.concatenate([np.stack([a, b, c], -1)[None], np.stack([a, c, e], -1)[None]])   # 2 x n x n x 3
            road = [t for t in both[:, corners == 4].reshape(-1, 3)]
            land = [t for t in both[:, corners == 0].reshape(-1, 3)]
            for r, col in zip(*np.nonzero((corners > 0) & (corners < 4))):
                quad = [(r, col), (r, col + 1), (r + 1, col + 1), (r + 1, col)]
                for tri in ((0, 1, 2), (0, 2, 3)):
                    rc = [quad[k] for k in tri]
                    self._split(tuple(int(ids[q]) for q in rc), [(px[q], py[q]) for q in rc], [float(d[q]) for q in rc], road, land)
            self.road_faces[(i, j)] = np.array(road, np.int64).reshape(-1, 3)
            self.land_faces[(i, j)] = np.array(land, np.int64).reshape(-1, 3)

    def _build_surround(self):
        step = int(layout.COARSE)
        per_tile = int(layout.TILE)
        fine = set(self.tiles)

        def is_fine(lx, ly):
            """Whether the coarse cell whose low corner is lattice (lx, ly) lies in a fine tile."""
            x, y = lx + layout.SCENE_X0, ly + layout.SCENE_Y0
            return (int(np.floor(x / per_tile)), int(np.floor(y / per_tile))) in fine

        faces = []
        for ly in range(0, self.ny - 1, step):
            for lx in range(0, self.nx - 1, step):
                if is_fine(lx, ly):
                    continue
                sides = [is_fine(lx, ly - step), is_fine(lx + step, ly), is_fine(lx, ly + step), is_fine(lx - step, ly)]
                corners = [(lx, ly), (lx + step, ly), (lx + step, ly + step), (lx, ly + step)]
                if not any(sides):
                    ids = self._ids([c[0] for c in corners], [c[1] for c in corners])
                    faces += [(ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])]
                    continue
                ring = []
                for k in range(4):
                    (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
                    steps = step if sides[k] else 1
                    for s in range(steps):
                        ring.append((ax + (bx - ax) * s // steps, ay + (by - ay) * s // steps))
                ids = self._ids([p[0] for p in ring], [p[1] for p in ring])
                centre = self._ids([lx + step // 2], [ly + step // 2])[0]
                for k in range(len(ids)):
                    faces.append((centre, ids[k], ids[(k + 1) % len(ids)]))
        return np.array(faces, np.int64)

    def _finish(self):
        iy, ix = np.nonzero(self.index >= 0)
        order = self.index[iy, ix]
        x = np.empty(self.count)
        y = np.empty(self.count)
        x[order], y[order] = ix + layout.SCENE_X0, iy + layout.SCENE_Y0
        for v, (cx, cy) in self.cut_xy.items():
            x[v], y[v] = cx, cy
        z = self.ref.elevation(x, y)
        self.vertices = np.stack([x, y, z], axis=1)
        d = 1.0
        gx = (self.ref.elevation(x + d, y) - self.ref.elevation(x - d, y)) / (2 * d)
        gy = (self.ref.elevation(x, y + d) - self.ref.elevation(x, y - d)) / (2 * d)
        n = np.stack([-gx, -gy, np.ones_like(gx)], axis=1)
        self.normals = n / np.linalg.norm(n, axis=1, keepdims=True)

    @property
    def road(self):
        return np.concatenate(list(self.road_faces.values()))

    @property
    def land(self):
        return np.concatenate(list(self.land_faces.values()) + [self.surround_faces])

    @property
    def faces(self):
        return np.concatenate([self.road, self.land])

    def write_collision(self, path, faces, what):
        """One welded mesh, positions only."""
        used, local = np.unique(faces, return_inverse=True)
        local = local.reshape(faces.shape) + 1
        with open(path, "w") as f:
            f.write(f"# Columbus 151 Speedway, {what}. Metres, Z up. Collision surface.\n")
            f.write("".join(f"v {x:.3f} {y:.3f} {z:.4f}\n" for x, y, z in self.vertices[used]))
            f.write("".join(f"f {a} {b} {c}\n" for a, b, c in local))

    def _write_part(self, path, faces, uv):
        used, local = np.unique(faces, return_inverse=True)
        local = local.reshape(faces.shape) + 1
        with open(path, "w") as f:
            f.write("".join(f"v {x:.3f} {y:.3f} {z:.4f}\n" for x, y, z in self.vertices[used]))
            f.write("".join(f"vt {u:.6f} {v:.6f}\n" for u, v in uv[used]))
            f.write("".join(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in self.normals[used]))
            f.write("".join(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" for a, b, c in local))
        return len(used), len(faces)

    def write_visual(self, directory):
        """Textured meshes: road and land per tile, plus the surround.

        Returns {name: (group, tile name or None, vertices, triangles)}.
        """
        os.makedirs(directory, exist_ok=True)
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        out = {}
        for group, suffix, parts in (("Road", "_road", self.road_faces), ("Terrain", "", self.land_faces)):
            for (i, j), faces in parts.items():
                if not len(faces):
                    continue
                uv = np.stack(layout.tile_uv(i, j, x, y), axis=1)
                name = tile_name(i, j) + suffix
                out[name] = (group, tile_name(i, j)) + self._write_part(os.path.join(directory, name + ".obj"), faces, uv)
        uv = np.stack([(x - layout.SCENE_X0) / (layout.SCENE_X1 - layout.SCENE_X0),
                       (y - layout.SCENE_Y0) / (layout.SCENE_Y1 - layout.SCENE_Y0)], axis=1)
        out["surround"] = ("Terrain", None) + self._write_part(os.path.join(directory, "surround.obj"), self.surround_faces, uv)
        return out


def tile_name(i, j):
    """File stem for tile (i, j). Indices can be negative, so they are spelled out."""
    return f"tile_{'e' if i >= 0 else 'w'}{abs(i)}_{'n' if j >= 0 else 's'}{abs(j)}"
