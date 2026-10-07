"""Bark for the ray-traced camera's trees: a furrowed picture, and where on it each bit of wood lies.

    picture(colour) -> (SIZE, SIZE, 3) uint8
    wrap(vertices, faces) -> (texture coordinates, their indices per face), or None

Neither is measured. The scan sees a tree from above and gives no trunk at all, so the bark
is drawn from numbers: plates that are long the way the limb runs, dark furrows where they
meet, each plate a shade of its own, and fine streaks of grain. It repeats in both
directions without a seam, and its mean colour is the colour the wood had.

treegen skins a limb as a tube: rings of a few vertices and a point at the tip. wrap finds the
tubes again in the mesh and lays the picture round each one a whole number of times, so the
picture meets itself where the tube closes. Along the limb it advances in step with the
limb's girth at every ring, so the bark is stretched nowhere: a limb half as thick carries
plates half as large, which is roughly what wood does. Every tube starts at its own place in
the picture, so that the ridges of two limbs do not line up.
"""
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

SIZE = 512           # pixels along a side of the picture
PLATES = (7, 3)      # plates of bark round the picture, and along it
ACROSS = 0.5         # metres round a limb that one picture covers, about: thicker limbs take it twice or more
ALONG = 2.0          # and it covers this many times as far along the limb: a plate is longer than it is wide
GAMMA = 2.2          # the renderer's own: it raises a picture's values to this to get light
SEED = 20261007


def picture(colour):
    """Bark whose mean colour, as light, is colour. Columns go round the limb and rows along it."""
    rng = np.random.default_rng(SEED)

    def noise(across, along):
        """Smooth noise that repeats at the picture's edges, mean 0 and spread 1."""
        n = gaussian_filter(rng.normal(size=(SIZE, SIZE)), (along, across), mode="wrap")
        return (n - n.mean()) / n.std()

    # Plates: one about each of a scatter of points, long the way the limb runs. A furrow is where two plates meet.
    cols, rows = PLATES
    points = (np.stack(np.meshgrid(np.arange(cols), np.arange(rows)), -1).reshape(-1, 2) + rng.uniform(0.1, 0.9, (cols * rows, 2))) / [cols, rows]
    shifts = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1]), -1).reshape(-1, 1, 2)
    tree = cKDTree(((points[None] + shifts) * [cols, rows]).reshape(-1, 2))      # repeated all round, so the plates repeat too
    u = (np.arange(SIZE) + 0.5) / SIZE
    round_, along = np.meshgrid(u, u)
    # Their edges wander: a plate's outline is not a straight cut.
    where = np.stack([(round_ + 0.035 * noise(SIZE / 40, SIZE / 14)) * cols, (along + 0.05 * noise(SIZE / 30, SIZE / 30)) * rows], -1)
    near, which = tree.query(where.reshape(-1, 2), k=2)
    edge = (near[:, 1] - near[:, 0]).reshape(SIZE, SIZE)                     # 0 in the middle of a furrow
    plate = (which[:, 0] % (cols * rows)).reshape(SIZE, SIZE)
    furrow = np.clip(0.42 + 0.14 * noise(SIZE / 20, SIZE / 8), 0.2, 0.8)    # how far a furrow's slope reaches into the plate
    height = np.clip(edge / furrow, 0.0, 1.0) ** 0.9
    tone = rng.uniform(0.86, 1.14, cols * rows)[plate]                       # a plate is a little lighter or darker than the next
    grain = 0.09 * noise(0.9, 6.0) + 0.04 * noise(0.8, 0.8)                  # fine streaks along the limb
    value = np.clip(0.26 + 0.86 * height * tone * (1.0 + 0.08 * noise(SIZE / 9, SIZE / 9)) + grain, 0.12, None)
    # The furrows keep the wood's brown. The plates are weathered, and so greyer. Nothing stands out on one
    # plate, a patch of moss say: it would come round again with every repeat and give the repeat away.
    grey = (0.35 * height)[..., None]
    light = value[..., None] * ((1 - grey) * np.asarray(colour, dtype=float) + grey * np.mean(colour))
    light *= np.asarray(colour, dtype=float) / light.reshape(-1, 3).mean(0)
    return (np.clip(light, 0, 1) ** (1 / GAMMA) * 255.0 + 0.5).astype(np.uint8)


def wrap(vertices, faces):
    """Texture coordinates for a mesh of treegen's tubes, and for each face the three it uses.

    None if the mesh is not made that way: then it is better left plain than given a smear.
    """
    count = len(vertices)
    a, b = faces[:, [0, 1, 2]].ravel(), faces[:, [1, 2, 0]].ravel()
    tubes, tube = connected_components(sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(count, count)), directed=False)
    rng = np.random.default_rng(SEED)
    sides = np.zeros(count, int)       # per vertex: its tube's sides, its ring and its place in the ring, and whether it is the tip
    ring, place, tip = np.zeros(count, int), np.zeros(count, int), np.zeros(count, bool)
    first = np.zeros(count, int)       # per vertex: where its tube's coordinates start in the list
    coords = []
    uses = np.bincount(faces.ravel(), minlength=count)
    for t in range(tubes):
        own = np.flatnonzero(tube == t)
        lo, hi = own[0], own[-1]
        s = int(uses[hi])              # the tip is the tube's last vertex, and one triangle of each side meets it
        if hi - lo + 1 != len(own) or s < 3 or (len(own) - 1) % s or len(own) < s + 1:
            return None
        rings = (len(own) - 1) // s
        body = vertices[lo:hi].reshape(rings, s, 3)
        middle = np.concatenate([body.mean(1), vertices[hi:hi + 1]])
        radius = np.linalg.norm(body - middle[:-1, None], axis=2).mean(1)
        girth = 2 * s * np.sin(np.pi / s) * radius                                  # round a ring, along its flat sides
        between = np.maximum(0.5 * (girth + np.append(girth[1:], 0.0)), 1e-3)       # from one ring to the next, and to the tip
        times = max(1, int(round(girth[0] / ACROSS)))
        # At every ring the picture is as large along the limb as round it, but for ALONG: the bark is stretched nowhere.
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(middle, axis=0), axis=1) * times / (between * ALONG))])
        u0, v0 = rng.uniform(0.0, 1.0, 2)
        first[own] = sum(len(c) for c in coords)
        local = np.arange(len(own) - 1)
        sides[own], ring[own[:-1]], place[own[:-1]], tip[hi] = s, local // s, local % s, True
        ring[hi] = rings
        col = np.arange(s + 1) * times / s
        coords.append(np.stack([np.tile(u0 + col, rings), np.repeat(v0 + along[:-1], s + 1)], 1))
        coords.append(np.stack([u0 + (np.arange(s) + 0.5) * times / s, np.full(s, v0 + along[-1])], 1))

    # A face lies on one side of its tube. On the side that closes the tube, the ring's first vertex takes the
    # coordinate one whole turn on, which is the same place in the picture.
    s, column = sides[faces], np.where(tip[faces], -1, place[faces])
    closes = ((column == 0).any(1) & (column == s - 1).any(1))[:, None]
    column = np.where(closes & (column == 0), s, column)
    side = np.where(column < 0, 99, column).min(1, keepdims=True)
    index = first[faces] + np.where(tip[faces], ring[faces] * (s + 1) + side, ring[faces] * (s + 1) + column)
    return np.concatenate(coords), index
