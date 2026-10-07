"""Rock piles and boulders, as heaps of rough stones.

The scan shows a pile of stones as a flat patch of grey with dark gaps in it. What stands in
for it is a heap: stones of mixed sizes dropped over the patch the pile covers, more of them
toward its middle, each settled into the ground or into the ones under it. Each stone is a rough lump with flat faces, the hull of a few
points thrown at random. No two piles are the same, but the same pile is always built the
same way: every choice comes from a number fixed by the pile's place in the list.

A pile is built in place, in scene coordinates, with each stone set on the ground under it.
Its stones come in three shades of the one colour the photo gives, so that the heap does not
read as a single grey mass.
"""
import numpy as np
from scipy.spatial import ConvexHull

SHADES = {"light": 1.12, "mid": 1.0, "dark": 0.82}
STONE = (0.35, 0.85)     # metres across, the smallest and the largest stone of a pile
COVER = 0.3              # square metres of ground one stone accounts for, in a pile
MOST = 350               # stones in the largest pile
LOWEST = 0.4             # metres: a pile the scan shows flat is still one stone high
HIGHEST = 1.3            # metres, and none is built higher than this
SUNK = 0.33              # how much of a stone's height is in the ground, or in the stones under it
TIP = 0.12               # radians a stone may be tipped from lying flat


def _stone(rng, size):
    """One stone about the origin: (vertices, faces), a rough lump about size across.

    A stone lying on the ground has settled onto its broadest side. So the lump is flatter
    below its middle than above, and it is turned any way round but tipped only a little.
    """
    points = rng.normal(size=(12, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= rng.uniform(0.72, 1.0, size=(12, 1))
    points *= np.array([size, size * rng.uniform(0.6, 1.0), size * rng.uniform(0.4, 0.7)]) / 2
    points[points[:, 2] < 0, 2] *= 0.45
    turn, lean, roll = rng.uniform(0, 2 * np.pi), rng.uniform(-TIP, TIP), rng.uniform(-TIP, TIP)
    cz, sz, cy, sy, cx, sx = np.cos(turn), np.sin(turn), np.cos(lean), np.sin(lean), np.cos(roll), np.sin(roll)
    spin = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    points = points @ spin.T
    faces = ConvexHull(points).simplices.copy()
    # Wound to face out: away from the stone's own middle.
    a, b, c = (points[faces[:, k]] for k in range(3))
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), a) < 0
    faces[inward] = faces[inward][:, ::-1]
    return points, faces


def _boulder(rng, size, tall):
    """One boulder about the origin: (vertices, faces), a rounded lump size across and tall high.

    A boulder is as wide one way as the other and rounded all over, where a stone of a pile
    is a flat lump. So it is cut from a ball, a little dented, and not flattened underneath.
    """
    points = rng.normal(size=(20, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= rng.uniform(0.86, 1.0, size=(20, 1))
    points *= np.array([size, size * rng.uniform(0.8, 1.0), tall]) / 2
    turn = rng.uniform(0, 2 * np.pi)
    points[:, :2] = points[:, :2] @ np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]).T
    faces = ConvexHull(points).simplices.copy()
    a, b, c = (points[faces[:, k]] for k in range(3))
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), a) < 0
    faces[inward] = faces[inward][:, ::-1]
    return points, faces


def _inside(outline, points):
    """Which of some points lie inside a closed outline, and how far each is from its edge."""
    ring = np.asarray(outline, float)
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    a, b = ring, np.roll(ring, -1, axis=0)
    x, y = points[:, :1], points[:, 1:]
    crosses = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / np.where(b[:, 1] == a[:, 1], 1e-12, b[:, 1] - a[:, 1]) + a[:, 0])
    d = b - a
    t = np.clip(((points[:, None, :] - a[None]) * d[None]).sum(-1) / np.maximum((d * d).sum(-1), 1e-12)[None], 0.0, 1.0)
    edge = np.hypot(*(a[None] + t[..., None] * d[None] - points[:, None, :]).transpose(2, 0, 1)).min(1)
    return crosses.sum(1) % 2 == 1, edge


def _area(outline):
    """The area a closed outline encloses."""
    ring = np.asarray(outline, float)
    return float(abs(np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])) / 2)


def make(rock, ground, seed):
    """({"light": (vertices, faces), "mid": ..., "dark": ...}, how many stones) for one pile or one boulder.

    rock is a dict as rocks.find gives it, with "height" added: how high the scan shows the
    pile standing. ground(xs, ys) gives the ground's height. seed is a whole number that
    fixes every random choice.

    Stones are dropped over the pile's patch one by one, the large ones first. Each comes to
    rest at the level of what lies under its middle, sunk into it by a third, so that a stone
    that lands on the shoulder of another leans into it and none is left standing on a point
    with air beneath. A pile is never built higher than the scan shows it: most of the piles
    here are one layer of stones lying on the grass, and only the largest is a heap.
    """
    rng = np.random.default_rng(1000 + seed)
    names = list(SHADES)
    parts = {name: ([], [], 0) for name in SHADES}

    def add(v, f, where):
        name = names[int(rng.choice(3, p=[0.3, 0.4, 0.3]))]
        vertices, faces, base = parts[name]
        vertices.append(v + where)
        faces.append(f + base)
        parts[name] = (vertices, faces, base + len(v))

    count = 0
    if rock["kind"] == "boulder":
        size = float(np.clip(max(rock["length"], rock["width"]), 0.5, 2.0))
        # A boulder stands as high out of the ground as the scan shows it, within what its width allows.
        shown = float(np.clip(rock.get("height", 0.0), 0.3 * size, 0.45 * size))
        v, f = _boulder(rng, size, shown / (1.0 - SUNK))
        tall = v[:, 2].max() - v[:, 2].min()
        level = float(ground(np.array([rock["x"]]), np.array([rock["y"]]))[0])
        add(v, f, np.array([rock["x"], rock["y"], level - v[:, 2].min() - SUNK * tall]))
        count = 1
    else:
        ring = np.asarray(rock["outline"], float)
        if not np.allclose(ring[0], ring[-1]):
            ring = np.concatenate([ring, ring[:1]])
        lo, hi = ring.min(0), ring.max(0)
        highest = float(np.clip(rock.get("height", 0.0), LOWEST, HIGHEST))
        wanted = int(np.clip((1.0 + 2.0 * highest) * _area(ring) / COVER, 6, MOST))
        tries = rng.uniform(lo, hi, size=(60 * wanted, 2))
        inside, edge = _inside(ring, tries)
        deepest = max(float(edge[inside].max()), 1e-6) if inside.any() else 1.0
        keep = inside & (rng.uniform(size=len(tries)) < 0.3 + 0.7 * edge / deepest)
        spots = tries[keep][:wanted]
        sizes = np.sort(rng.uniform(*STONE, size=len(spots)) * (1.25 if _area(ring) > 40 else 1.0))[::-1]       # the big ones first, at the bottom
        if not len(spots):
            return {}, 0
        level = np.asarray(ground(spots[:, 0], spots[:, 1]), float)
        grain = 0.1                                         # metres, the cells the heap's own height is kept on
        heap = np.zeros((int((hi[1] - lo[1]) / grain) + 5, int((hi[0] - lo[0]) / grain) + 5))
        yy, xx = np.mgrid[0:heap.shape[0], 0:heap.shape[1]]
        for (x, y), size, z in zip(spots, sizes, level):
            v, f = _stone(rng, size)
            tall = v[:, 2].max() - v[:, 2].min()
            away = np.hypot(xx - (x - lo[0]) / grain - 2, yy - (y - lo[1]) / grain - 2) * grain
            middle = away <= 0.2 * size
            rest = float(np.median(heap[middle])) if middle.any() else 0.0
            bottom = rest - SUNK * tall                     # sunk a third into what it rests on
            if bottom + tall > highest:
                continue                                    # the pile is as high here as the scan shows it
            add(v, f, np.array([x, y, z + bottom - v[:, 2].min()]))
            # What it leaves for the next to rest on: its top in the middle, falling away to its rim.
            under = away <= 0.5 * size
            dome = bottom + tall * (1.0 - 0.8 * (away[under] / (0.5 * size)) ** 2)
            heap[under] = np.maximum(heap[under], dome)
            count += 1
    return {name: (np.concatenate(v), np.concatenate(f)) for name, (v, f, _) in parts.items() if v}, count
