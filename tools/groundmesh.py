"""The ground's meshes, read back from the scene: where a picture lies, how high the land is.

The tools that add detail to a finished scene have nothing but the scene to go by. What they
need to know about the ground is all in its meshes. A tile's texture coordinates say what
piece of the world its picture covers. The land's triangles say how high the ground is under
any point, and whether there is land there at all. The pavement's triangles say where its
edge runs: it is the edges that only one triangle has.

Also here, because every one of those tools has to mind it: the count of pictures a scene
names. Chrono::Sensor's Metal ray tracer takes METAL_TAKES of them and draws whatever comes
after in plain colour, without a word.
"""
import numpy as np

METAL_TAKES = 64     # pictures in one scene that Chrono::Sensor's Metal ray tracer takes


def pictures(doc):
    """The different texture pictures a manifest's placed assets name."""
    placed = {inst["asset"] for inst in doc["instances"]}
    return {part["texture"] for n, asset in enumerate(doc["assets"]) if n in placed for part in asset["parts"] if part.get("texture")}


def count_pictures(doc, log, who):
    """Say how many pictures the scene names now that who is done, and stop if the renderer cannot take them."""
    count = len(pictures(doc))
    log(f"{who}: the scene names {count} pictures, of the {METAL_TAKES} the renderer takes")
    if count > METAL_TAKES:
        raise SystemExit(f"{who}: the scene names {count} pictures and Chrono::Sensor's Metal ray tracer takes {METAL_TAKES}. "
                         "It would draw the rest in plain colour: a card of blades as a solid slab")


def read(path):
    """An OBJ file as (vertices, texture coordinates, faces, faces' texture indices), indices from 0.

    The last two are (n, 3). A file without texture coordinates gives None for them.
    """
    vertices, coords, faces, tex = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append(line.split()[1:4])
            elif line.startswith("vt "):
                coords.append(line.split()[1:3])
            elif line.startswith("f "):
                corners = [c.split("/") for c in line.split()[1:4]]
                faces.append([c[0] for c in corners])
                if coords:
                    tex.append([c[1] for c in corners])
    faces = np.array(faces, np.int64).reshape(-1, 3) - 1
    if not coords:
        return np.array(vertices, np.float64), None, faces, None
    return np.array(vertices, np.float64), np.array(coords, np.float64), faces, np.array(tex, np.int64).reshape(-1, 3) - 1


def frame(vertices, coords, faces, tex):
    """Where a mesh's picture lies in the world: (x0, y0, width, height), in metres.

    Texture coordinate (u, v) is the point (x0 + u * width, y0 + v * height). Fitted to every
    corner of every triangle, east against u and north against v. A picture that is turned or
    sheared against the world's axes does not fit that way, and neither does one that is not a
    picture of the ground from above. For those the answer is None: the tools that use this
    work along rows and columns.
    """
    if coords is None or not len(faces):
        return None
    uv = coords[tex.reshape(-1)]
    xy = vertices[faces.reshape(-1), :2]
    found, miss = [], 0.0
    for k in range(2):
        design = np.column_stack([uv[:, k], np.ones(len(uv))])
        (size, start), *_ = np.linalg.lstsq(design, xy[:, k], rcond=None)
        miss = max(miss, float(np.abs(design @ [size, start] - xy[:, k]).max()))
        found.append((float(start), float(size)))
    if miss > 0.005:
        return None
    return found[0][0], found[1][0], found[0][1], found[1][1]


def outline(vertices, faces):
    """The edges of a welded mesh that only one triangle has, as (n, 2, 2): from and to, in plan.

    The scene's triangles are wound anticlockwise seen from above, so the mesh lies to the left
    of each edge and what is outside it lies to the right.
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    low, high = edges.min(1), edges.max(1)
    _, where, count = np.unique(low * (int(faces.max()) + 1) + high, return_inverse=True, return_counts=True)
    alone = edges[count[where.reshape(-1)] == 1]
    return vertices[alone][:, :, :2]


class Surface:
    """A mesh to stand things on: which triangle is under a point, and how high it is there."""

    def __init__(self, vertices, faces, cell=1.0):
        self.vertices, self.faces, self.cell = vertices, faces, cell
        xy = vertices[faces][:, :, :2]
        self.x0, self.y0 = xy[..., 0].min(), xy[..., 1].min()
        # Every triangle is listed under each square of a grid that its bounding box touches.
        low = np.floor((xy.min(1) - [self.x0, self.y0]) / cell + 1e-9).astype(np.int64)
        high = np.floor((xy.max(1) - [self.x0, self.y0]) / cell - 1e-9).astype(np.int64)
        high = np.maximum(high, low)
        wide, tall = high[:, 0] - low[:, 0] + 1, high[:, 1] - low[:, 1] + 1
        self.nx = int(high[:, 0].max()) + 1
        ny = int(high[:, 1].max()) + 1
        which = np.repeat(np.arange(len(faces)), wide * tall)
        step = np.arange(len(which)) - np.repeat(np.cumsum(wide * tall) - wide * tall, wide * tall)
        cx = low[which, 0] + step % wide[which]
        cy = low[which, 1] + step // wide[which]
        key = cy * self.nx + cx
        order = np.argsort(key, kind="stable")
        self.listed = which[order]
        self.start = np.searchsorted(key[order], np.arange(self.nx * ny + 1))
        self.most = int(np.diff(self.start).max())

    def under(self, x, y):
        """For points: (height, triangle index). The index is -1 and the height NaN where no triangle is."""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        cx = np.floor((x - self.x0) / self.cell).astype(np.int64)
        cy = np.floor((y - self.y0) / self.cell).astype(np.int64)
        inside = (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy * self.nx + cx < len(self.start) - 1)
        key = np.where(inside, cy * self.nx + cx, 0)
        first, count = self.start[key], np.where(inside, self.start[key + 1] - self.start[key], 0)
        height = np.full(x.shape, np.nan)
        found = np.full(x.shape, -1, np.int64)
        for k in range(self.most):
            todo = np.nonzero((count > k) & (found < 0))[0]
            if not len(todo):
                break
            tri = self.listed[first[todo] + k]
            a, b, c = (self.vertices[self.faces[tri, n]] for n in range(3))
            px, py = x[todo], y[todo]
            area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            area = np.where(np.abs(area) < 1e-12, 1e-12, area)
            wb = ((px - a[:, 0]) * (c[:, 1] - a[:, 1]) - (py - a[:, 1]) * (c[:, 0] - a[:, 0])) / area
            wc = ((b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (b[:, 1] - a[:, 1]) * (px - a[:, 0])) / area
            hit = (wb >= -1e-9) & (wc >= -1e-9) & (wb + wc <= 1 + 1e-9)
            height[todo[hit]] = (a[:, 2] + wb * (b[:, 2] - a[:, 2]) + wc * (c[:, 2] - a[:, 2]))[hit]
            found[todo[hit]] = tri[hit]
        return height, found

    def normals(self, triangles):
        """Unit normals of triangles, pointing up."""
        a, b, c = (self.vertices[self.faces[triangles, n]] for n in range(3))
        n = np.cross(b - a, c - a)
        n *= np.sign(n[:, 2:3]) + (n[:, 2:3] == 0)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
