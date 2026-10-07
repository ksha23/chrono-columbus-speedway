"""Sprays of small leaves for the ray-traced camera's trees, all in one picture with holes in it.

    build(pairs) -> (picture, places, cover)

A tree's leaf is one triangle, 0.1 to 0.6 m long. Here it becomes a spray: a twig with a few
dozen small leaves on and about it, drawn inside the triangle and cut out of it by the
picture's alpha. Nothing of this is measured. The scan gives a tree its place, its size and
one colour, and no leaf: the leaf shapes, their number, how they sit on the twig and how
their colours differ are all made up here, to look like foliage from a car's distance.

pairs lists (kind, colour): every kind of tree with every leaf colour it comes in. Each pair
gets its own few sprays, drawn in that colour, because the renderer has no way to tint a
picture. They all go into ONE picture, since the renderer takes 64 pictures in all and the
ground has most of them. The picture is cut into square cells, as large as will hold every
spray, and each cell into two triangles with a spray in each.

places[(kind, colour)] is a list, one per spray, of where the triangle's corners go in the
picture: (3, 2) texture coordinates of the base's first end, the tip and the base's other
end, v upward as in an OBJ file. cover[kind] is the share of the triangle its sprays fill:
about half, which is why foliage.py enlarges the triangles.

A spray is drawn in the shape of its kind's leaf triangle, so that leaves keep their shape
when the triangle is laid on it. Every leaf lies wholly inside the triangle and stops short
of its edges by a margin of its own: the outline of a spray is then leaf tips, and no
straight edge shows. A willow's leaves are long and narrow and hang the way its strand
does. The others' stand off the twig to both sides.

Leaves differ in brightness and in how yellow they are, the ones underneath are darker, and
one in fifty has turned yellow. The differences are kept moderate on purpose: the renderer
does not average a picture over the area a pixel sees, so at a distance each pixel shows one
leaf's colour, and strong differences between leaves would turn into noise there. A spray's
mean colour is exactly the tree's.

The same pairs always give the same picture: every choice comes from fixed numbers.
"""
import numpy as np

SIZE = 1024          # pixels along a side of the picture
GUTTER = 1.5         # pixels between a triangle and the edge of its cell, so neighbours do not bleed
SUB = 3              # a pixel is leaf if most of its SUB x SUB points are
SEED = 20261007
GAMMA = 2.2          # the renderer's own: it raises a picture's values to this to get light
TWIG = (0.075, 0.055, 0.04)        # a twig's colour, as light
YELLOWED = 0.02      # the share of leaves that have turned
DEEP = 0.55          # how bright the undermost leaf is, against the topmost
START = 0.02         # where a twig starts, along the triangle from its base
THICK = 0.011        # half a twig's thickness at its start, over the triangle's length

# half     half the triangle's base over its length: treegen's leaf for the kind
# sprays   how many different sprays the kind has
# turns    how many of the triangle's corners a spray's tip may be laid on (1: the tip only)
# twig, loose, under    leaves on the twig, lying loose on top, and underneath
# long     a leaf's length over the triangle's, shortest and longest
# wide     its width over its length;  widest: where along it that is;  point: above 1 narrows the ends
# turn     radians a leaf stands off its twig, least and most
KINDS = {
    "broadleaf": dict(half=0.41, sprays=4, turns=3, twig=10, loose=14, under=30, long=(0.14, 0.21), wide=0.52, widest=0.40, point=0.85, turn=(0.5, 1.2)),
    "upright": dict(half=0.385, sprays=4, turns=3, twig=10, loose=14, under=30, long=(0.13, 0.19), wide=0.74, widest=0.36, point=0.75, turn=(0.5, 1.3)),
    "shrub": dict(half=0.48, sprays=4, turns=3, twig=5, loose=7, under=14, long=(0.26, 0.36), wide=0.58, widest=0.45, point=0.8, turn=(0.5, 1.3)),
    "willow": dict(half=0.235, sprays=4, turns=1, twig=9, loose=10, under=18, long=(0.24, 0.34), wide=0.20, widest=0.42, point=1.0, turn=(0.12, 0.4)),
}
OTHER = "broadleaf"  # what a kind not listed above is drawn as


def kind_of(kind):
    return kind if kind in KINDS else OTHER


def _outline(leaf, k, n=9):
    """Points round a leaf: n along each side."""
    foot, angle, length = leaf[:2], leaf[2], leaf[3]
    s = np.linspace(0.0, 1.0, n)
    along, across = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
    half = _half_width(s, length, k)
    spine = foot + np.outer(s * length, along) + np.outer(leaf[4] * length * s * (1 - s), across)
    return np.concatenate([spine + half[:, None] * across, spine - half[:, None] * across])


def _half_width(s, length, k):
    bulge = np.sin(np.pi * np.clip(s, 0.0, 1.0) ** (np.log(0.5) / np.log(k["widest"]))) ** k["point"]
    return 0.5 * k["wide"] * length * bulge


def _fits(points, half, margin):
    """Whether points lie inside the triangle (0, -half), (1, 0), (0, half), margin away from its edges."""
    x, y = points[:, 0], points[:, 1]
    return bool((x.min() >= margin) and ((half * (1 - x) - np.abs(y)) / np.hypot(1.0, half)).min() >= margin)


def _twig(twig, x):
    """A twig's y and slope at x. It runs from the middle of the base to (twig[0], twig[1]), bowed by twig[2]."""
    t = np.clip((x - START) / (twig[0] - START), 0.0, 1.0)
    return twig[1] * t + twig[2] * np.sin(np.pi * t), (twig[1] + twig[2] * np.pi * np.cos(np.pi * t)) / (twig[0] - START)


def _spray(kind, number):
    """One spray: its twig (see _twig), and its leaves from the undermost up.

    A leaf is (foot x, foot y, angle, length, curl, depth, brightness, yellow): depth runs from
    0 for the undermost to 1 on top, and yellow from about -0.3 (bluer) through 0 to 0.9 (turned).
    """
    k = KINDS[kind]
    rng = np.random.default_rng([SEED, sorted(KINDS).index(kind), number])
    half = k["half"]
    twig = np.array([rng.uniform(0.55, 0.68), rng.uniform(-0.12, 0.12) * half, rng.uniform(-0.05, 0.05)])

    def settle(foot, angle, length, depth, loose):
        """The leaf as asked for or as near as will go in: a loose one drawn back along itself, one on the twig made shorter. None if neither does it."""
        leaf = np.array([foot[0], foot[1], angle, length, rng.uniform(-0.25, 0.25), depth, 0.0, 0.0])
        margin = rng.uniform(0.012, 0.06)
        for _ in range(14 if loose else 5):
            if _fits(_outline(leaf, k), half, margin):
                return leaf
            if loose:
                leaf[:2] -= 0.025 * np.array([np.cos(angle), np.sin(angle)])
            else:
                leaf[3] *= 0.85
        return None

    leaves = []
    side = rng.choice([-1.0, 1.0])
    for i in range(k["twig"]):          # along the twig, a side each in turn, the last one at its end
        last = i == k["twig"] - 1
        x = twig[0] if last else START + (twig[0] - START) * np.clip(0.06 + 0.86 * (i + rng.uniform(-0.25, 0.25)) / (k["twig"] - 1), 0.0, 1.0)
        y, slope = _twig(twig, x)
        off = rng.uniform(-0.25, 0.25) if last else side * rng.uniform(*k["turn"])
        leaf = settle(np.array([x, y]), np.arctan(slope) + off, rng.uniform(*k["long"]) * (0.85 if last else 1.0), rng.uniform(0.55, 1.0), False)
        if leaf is not None:
            leaves.append(leaf)
        side = -side
    for layer, count in ((1, k["loose"]), (0, k["under"])):
        for _ in range(count * 8):      # the rest, wherever they go in: on top, and underneath
            if count == 0:
                break
            x = 1.0 - np.sqrt(rng.uniform(0.0, 1.0))
            tip = np.array([x, rng.uniform(-1.0, 1.0) * half * (1 - x)])
            # It points away from the middle of the triangle, or the way the strand hangs on a willow.
            angle = np.arctan2(tip[1], tip[0] - 0.33) + rng.uniform(-0.8, 0.8) if k["turns"] > 1 else rng.uniform(-0.45, 0.45)
            length = rng.uniform(*k["long"]) * rng.uniform(0.6, 1.0)        # some small, which go into the corners
            foot = tip - length * np.array([np.cos(angle), np.sin(angle)])
            leaf = settle(foot, angle, length, rng.uniform(0.55, 1.0) if layer else rng.uniform(0.0, 0.45), True)
            if leaf is not None:
                leaves.append(leaf)
                count -= 1
    leaves = np.array(sorted(leaves, key=lambda leaf: leaf[5]))
    leaves[:, 6] = (DEEP + (1 - DEEP) * leaves[:, 5] ** 0.8) * np.exp(rng.normal(0.0, 0.11, len(leaves)))
    leaves[:, 7] = rng.normal(0.0, 0.16, len(leaves))
    turned = (rng.uniform(size=len(leaves)) < YELLOWED) & (leaves[:, 5] > 0.3)
    leaves[turned, 7] = rng.uniform(0.55, 0.9, turned.sum())
    return twig, leaves


def _window(points, shape):
    """Pixel rows and columns that cover points, and the SUB x SUB places inside each pixel: (rows, cols, SUB * SUB, 2) as x, y."""
    x0, y0 = np.maximum(np.floor(points.min(0)).astype(int) - 1, 0)
    x1, y1 = np.minimum(np.ceil(points.max(0)).astype(int) + 2, [shape[1], shape[0]])
    step = (np.arange(SUB) + 0.5) / SUB
    inside = np.stack(np.meshgrid(step, step), -1).reshape(-1, 2)
    corner = np.stack(np.meshgrid(np.arange(x0, x1), np.arange(y0, y1)), -1)
    return (slice(y0, y1), slice(x0, x1)), corner[:, :, None, :] + inside[None, None]


def _draw(layers, number, corners, kind, spray):
    """Draw one spray into the triangle with pixel corners (base, tip, base)."""
    k = KINDS[kind]
    alpha, shade, yellow, wood, owner = layers
    flat = np.array([[0.0, -k["half"]], [1.0, 0.0], [0.0, k["half"]]])
    lay = np.linalg.solve(flat[1:] - flat[0], corners[1:] - corners[0])      # a row vector in the spray's own frame to pixels
    back = np.linalg.inv(lay)
    twig, leaves = spray

    def stamp(points, test):
        where, pixels = _window((points - flat[0]) @ lay + corners[0], alpha.shape)
        own = (pixels - corners[0]) @ back + flat[0]
        hit, values = test(own)
        most = hit.mean(2) >= 0.5
        return where, most, [v[:, :, (SUB * SUB) // 2] for v in values]

    def twig_test(own):
        x, y = own[..., 0], own[..., 1]
        return (x > START) & (x < twig[0]) & (np.abs(y - _twig(twig, x)[0]) < THICK * (1.0 - 0.5 * x / twig[0])), []

    along_twig = np.linspace(START, twig[0], 9)
    twig_box = np.concatenate([np.stack([along_twig, _twig(twig, along_twig)[0] + d], 1) for d in (-THICK, THICK)])

    drawn = False
    for leaf in leaves:
        if leaf[5] > 0.5 and not drawn:              # the twig lies over the leaves underneath and under the rest
            where, most, _ = stamp(twig_box, twig_test)
            alpha[where] |= most
            wood[where] |= most
            owner[where][most] = number
            drawn = True
        foot, angle, length, curl = leaf[:2], leaf[2], leaf[3], leaf[4]
        along, across = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])

        def leaf_test(own):
            s = (own - foot) @ along / length
            t = (own - foot) @ across - curl * length * s * (1 - s)
            half = _half_width(s, length, k)
            return (s > 0) & (s < 1) & (np.abs(t) < half), [s, t / np.maximum(half, 1e-9)]

        where, most, (s, t) = stamp(_outline(leaf, k), leaf_test)
        # Within a leaf: a pale midrib, one half a little darker as if folded along it, paler toward the tip.
        within = (0.93 + 0.14 * np.clip(s, 0, 1)) * np.where(t > 0, 0.93, 1.05) * np.where(np.abs(t) < 0.09, 1.16, 1.0)
        alpha[where] |= most
        wood[where] &= ~most
        shade[where][most] = (leaf[6] * within)[most]
        yellow[where][most] = leaf[7]
        owner[where][most] = number


def _cells(count):
    """Pixel corners (base, tip, base) of count triangles: two to a square cell, in the fewest, largest cells that hold them."""
    side = int(np.ceil(np.sqrt(count / 2.0)))
    cell = SIZE // side
    g, far = GUTTER, cell - 2.42 * GUTTER
    out = []
    for n in range(count):
        row, col = divmod(n // 2, side)
        corner = np.array([col * cell, row * cell], dtype=float)
        # The tip goes to a sharp corner of the half cell, which is the nearest to its own shape.
        tri = np.array([[g, g], [far, g], [g, far]]) if n % 2 == 0 else cell - np.array([[g, g], [far, g], [g, far]])
        out.append(corner + tri)
    return out


def build(pairs):
    """The picture as (SIZE, SIZE, 4) uint8 with alpha 0 or 255, each pair's sprays' places, and each kind's cover."""
    kinds = sorted({kind_of(kind) for kind, _ in pairs})
    sprays = {kind: [_spray(kind, n) for n in range(KINDS[kind]["sprays"])] for kind in kinds}
    wanted = [(pair, n) for pair in pairs for n in range(KINDS[kind_of(pair[0])]["sprays"])]
    corners = _cells(len(wanted))
    alpha, wood = np.zeros((SIZE, SIZE), bool), np.zeros((SIZE, SIZE), bool)
    shade, yellow = np.ones((SIZE, SIZE), np.float32), np.zeros((SIZE, SIZE), np.float32)
    owner = np.full((SIZE, SIZE), -1, np.int32)
    places, filled = {}, {kind: [] for kind in kinds}
    for number, ((pair, n), tri) in enumerate(zip(wanted, corners)):
        kind = kind_of(pair[0])
        _draw((alpha, shade, yellow, wood, owner), number, tri, kind, sprays[kind][n])
        places.setdefault(pair, []).append(np.stack([tri[:, 0] / SIZE, 1.0 - tri[:, 1] / SIZE], 1))
        (ax, ay), (bx, by) = tri[1] - tri[0], tri[2] - tri[0]
        area = 0.5 * abs(ax * by - ay * bx)
        filled[kind].append((owner == number).sum() / area)

    # Colour, as light. A leaf's yellow moves red up and blue down, and a turned leaf is brighter too.
    tint = np.array([pair[1] for pair, _ in wanted], dtype=np.float32)
    which = np.maximum(owner, 0)
    y = yellow[..., None]
    y = np.where(y < 0, 0.5 * y, y)
    light = tint[which] * shade[..., None] * np.clip(1.0 + y * np.array([0.75, 0.22, -0.55], np.float32), 0.05, None) * (1.0 + 0.25 * np.clip(y, 0, 1))
    # Each spray's mean colour over what shows of it is its tree's colour: the tints stay what they were.
    leaf = alpha & ~wood
    for c in range(3):
        mean = np.bincount(owner[leaf], light[..., c][leaf], len(wanted)) / np.maximum(np.bincount(owner[leaf], minlength=len(wanted)), 1)
        light[..., c] *= (tint[:, c] / np.maximum(mean, 1e-6))[which]
    light[wood] = TWIG
    picture = np.zeros((SIZE, SIZE, 4), np.uint8)
    picture[..., :3] = (np.clip(light, 0, 1) ** (1 / GAMMA) * 255.0 + 0.5).astype(np.uint8)
    picture[~alpha, :3] = (np.clip(tint.mean(0), 0, 1) ** (1 / GAMMA) * 255.0 + 0.5).astype(np.uint8)   # under the holes, where nothing shows
    picture[..., 3] = np.where(alpha, 255, 0)
    return picture, places, {kind: float(np.mean(filled[kind])) for kind in kinds}
