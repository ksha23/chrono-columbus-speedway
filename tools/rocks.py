"""Find the rubble and loose boulders lying on grass, so they can be painted out and set back as rocks.

Stone has almost no colour, where grass is green and dry grass and dirt are yellow. Concrete
has none either, but concrete is one even tone and a heap of stones is not: every stone is lit
differently from the next, the lit faces differ among themselves, and there is a dark gap
between them. So a pile is a patch of colourless ground a few stones across whose lightness
varies from stone to stone, on the light side as well as the dark.

A boulder on its own is too smooth to be told from a concrete lid that way. But it stands up:
it is shaded from light to dark across its top and throws a hard shadow about its own size
right beside it. A lid throws none, a lamp post throws a long thin one, and a white plastic
well cap is whiter than any stone.

Other things are left out by what they are not. Galvanised steel (the lattice tower) is
colourless but bluish where stones in a heap are neutral or warm. Dead branches, wires and
fence rails are colourless but thin, and nothing narrower than a quarter of a metre is kept.
A guard rail seen with the gravel beside it is wide enough, but all its changes in lightness
run one way, and a brush pile or a machine stands steeper than loose stones can. Pavement
comes from the pavement map, with a margin, because sand and gravel collect along its edge.
That map is not believed in two places where it takes in ground that is not pavement: see
RAISED and EDGE below.

A long line of stones (a rock border) is given back as a run of thin piles, each a few metres
long and nearly straight, so that each has a length, a width and a heading that mean something.
"""
import numpy as np
from scipy import ndimage

import edgelines
import railmodel

CLEAR = 0.5           # metres kept clear of the pavement's edge
TALL = 2.5            # metres: what the scan shows standing higher than this is a tree or a mast
TORN = -1.0           # metres: where the scan lies this far under the ground it is torn, and so is the photo
RIM = 4.0             # metres along the scan's outer edge that are left alone: the photo is smeared and bleached there
WALL = 5.0            # metres kept clear of a building: steps, ramps and machinery stand against it
# The pavement map takes in a heap of field stones under a dead tree at the site entrance.
# Pavement is flat, so a patch of the map that the scan shows standing this far above the
# ground, and that reaches the map's own edge, is searched like any other ground.
RAISED = 0.35         # metres above the ground
RAISED_AREA = 4.0     # square metres such a patch must cover
# It also takes in the dirt apron of the entrance, where five boulders stand. A boulder is
# known by its shadow and not by the ground it is on, so boulders (not piles) are looked for
# this far inside the map's edge as well. Set it to zero to keep strictly off the map.
EDGE = 2.5            # metres

SOFT = 1.0            # pixels of blur before a pixel's colour is judged
GREY_SAT = 0.12       # saturation below which a pixel has no colour to speak of
GREY_LIGHT = 0.30     # how light (0..1) a colourless pixel must be: darker is shadow or tar
COOL = 0.02           # how far blue may exceed red in the grey of a pile. Beyond that it is steel
TAN_SAT = 0.22        # saturation up to which a warm pixel is tan: concrete rubble, pink granite, dirt
TAN_LIGHT = 0.45      # how light a tan pixel must be
# Tan is the colour of dirt. A tan pixel counts as stone only where the tan pixels around it
# differ in lightness, as the faces of broken slabs do and a dirt track does not, and only
# inside a pile already found by its grey stones.
WINDOW = 13           # pixels across which lightness is compared
TAN_ROUGH = 0.055     # standard deviation of lightness (0..1) among the tan pixels of one window
THIN = 2              # pixels: half the width of the thinnest thing kept (branches, rails and wires go)

STRIP = 1024          # rows of the photo read at a time
PAD = 64              # rows and columns read beyond a strip, so that a boulder on its edge is seen whole
JOIN = 0.5            # metres between stones that are looked at together
REACH = 1.5           # metres around a group of stones that are looked at with it

GRAIN = 2.0           # pixels of blur that take out texture finer than a stone (grass blades, gravel)
PILE_ROUGH = 0.035    # standard deviation of lightness from stone to stone, one window at a time
PLAIN = 0.5           # square metres of one unbroken grey patch that must be rough in itself. An even one is a lid or a footing
FACETS = 0.08         # how far the lightest tenth of a pile's stone lies above its middle. Concrete is one tone
STREAKED = 0.75       # share of a heap's changes in lightness that run one way. Stones have no grain, a rail's edge has
STEEP = 0.35          # height over width a heap of loose stones can stand. Brush, fences and machines stand steeper
PILE_STONE = 0.8      # square metres of stone seen that make a pile
PILE_JOIN = 0.4       # metres: half the gap between rubble that is still one pile (weeds grow through)
RUN = 8.0             # metres: a longer line of stones is given back in stretches, each nearly straight
SLIM = 1.0            # square metres of stone per metre of length under which a pile is a line of stones

DARK = 0.27           # lightness below which a pixel is in hard shadow
YELLOW = 0.03         # how far the lesser of red and green may exceed blue in a boulder. Grass and dirt are yellower
GREEN = 0.06          # how far green may exceed red in a boulder. A utility box is greener
BOULDER_SAT = 0.30    # saturation up to which a pixel can be a boulder: sky light turns a shaded side blue
PINK = 0.05           # red over green beyond which a stone is pink granite, taken apart from the grey around it
CORE = 3              # pixels: half the width of the narrowest boulder
BOULDER_AREA = (0.15, 2.5)   # square metres of one boulder's lit top
ROUND = 2.0           # length over width a boulder may have
FILL = 0.55           # share of its bounding rectangle a boulder must fill
BOULDER_LIGHT = (0.45, 0.80)  # the middle lightness of a boulder. Whiter is paint or plastic
BOULDER_GLINT = 0.65  # how light the lightest tenth of a boulder must be
BOULDER_ROUGH = 0.07  # standard deviation of lightness across a boulder: it is shaded, a lid is not
SHADOW = (0.15, 3.0)  # a boulder's shadow, as a multiple of its own area. More is a tree's shadow it sits in
SHADOW_LONG = 2.5     # a boulder's shadow is at most this many times as long as the boulder...
SLENDER = 3.0         # ...and at most this many times as long as it is wide. A post's is longer and thinner
DIM = 0.7             # a post's shadow fades with distance: it is followed down to this share of the ground's lightness
POST = 2.5            # metres: a shadow that runs on this far, however faint, is a post's
CROWD = 0.5           # share of the ground just around a boulder that may be more of the same or shadow
BOULDER_TALL = 1.0    # metres the scan may show a boulder standing
LEVEL = 0.3           # metres the scan may show the ground around a boulder standing. Higher is brush
BURIED = 0.5          # share of a boulder's rim that is the stone of a pile before it is counted in the pile

GROW = 0.15           # metres the patch to paint out is grown beyond the stones and their shadows
OUTLINE = 0.05        # metres an outline may be simplified by


def _onto(grid, shape, r0, r1, c0, c1):
    """A coarser map of the same ground, read out cell for cell over a window of a finer grid."""
    rows = np.arange(r0, r1) * grid.shape[0] // shape[0]
    cols = np.arange(c0, c1) * grid.shape[1] // shape[1]
    return grid[rows[:, None], cols[None, :]]


def _disc(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius + radius


def _placed(small, row, col, r0, r1, c0, c1):
    """A small boolean array that sits at (row, col) of the photo, as it shows in the window r0:r1, c0:c1."""
    out = np.zeros((r1 - r0, c1 - c0), bool)
    a0, a1 = max(row, r0), min(row + small.shape[0], r1)
    b0, b1 = max(col, c0), min(col + small.shape[1], c1)
    if a0 < a1 and b0 < b1:
        out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = small[a0 - row:a1 - row, b0 - col:b1 - col]
    return out


def _window(sl, pad, shape):
    """A pair of slices grown by pad on every side, kept inside shape."""
    return (slice(max(sl[0].start - pad, 0), min(sl[0].stop + pad, shape[0])),
            slice(max(sl[1].start - pad, 0), min(sl[1].stop + pad, shape[1])))


def open_ground(paved, height, trees, buildings, cell, fine):
    """Where to look, on the pavement map's grid (cells of size fine): for piles, and for boulders.

    Piles are looked for off the pavement and clear of its edge. Boulders are looked for there
    and on the outer EDGE of the pavement map as well.
    """
    seen = np.isfinite(height)
    h = np.nan_to_num(height, nan=0.0)
    shape = paved.shape
    whole = (0, shape[0], 0, shape[1])
    # Patches of the pavement map that stand above the ground and touch unpaved ground.
    raised = paved & _onto(ndimage.binary_opening(h > RAISED), shape, *whole)
    labels, count = ndimage.label(raised)
    if count:
        index = np.arange(1, count + 1)
        area = ndimage.sum(raised, labels, index) * fine * fine
        at_edge = ndimage.maximum(ndimage.binary_dilation(~paved), labels, index) > 0
        raised = np.concatenate([[False], (area >= RAISED_AREA) & at_edge])[labels]
    del labels
    inside = ndimage.binary_erosion(ndimage.binary_fill_holes(seen), iterations=int(round(RIM / cell)))
    blocked = ~seen | ~inside | (h > TALL) | (h < TORN) | trees | ndimage.binary_dilation(buildings, iterations=int(round(WALL / cell)))
    clear = ~_onto(blocked, shape, *whole)
    flat = paved & ~raised
    piles = clear & ~ndimage.binary_dilation(flat, iterations=int(round(CLEAR / fine)))
    rim = flat & ~ndimage.binary_erosion(flat, iterations=int(round(EDGE / fine))) if EDGE > 0 else np.zeros_like(flat)
    return piles, clear & (piles | rim)


def contrast(light, support, size=WINDOW):
    """How much lightness varies among the supporting pixels around each pixel.

    A standard deviation taken over the pixels of `support` only, so that the edge of a pale
    patch against dark grass does not count as variation within it. Zero where fewer than a
    quarter of the window's pixels support it.
    """
    weight = ndimage.uniform_filter(support.astype(np.float32), size)
    safe = np.maximum(weight, 1e-3)
    mean = ndimage.uniform_filter(np.where(support, light, 0.0).astype(np.float32), size) / safe
    square = ndimage.uniform_filter(np.where(support, light * light, 0.0).astype(np.float32), size) / safe
    return np.where(weight > 0.25, np.sqrt(np.maximum(square - mean * mean, 0.0)), 0.0)


def soften(rgb):
    """The photo lightly blurred, with each pixel's saturation and lightness."""
    soft = ndimage.gaussian_filter(rgb, (SOFT, SOFT, 0))
    mx, mn = soft.max(-1), soft.min(-1)
    return soft, (mx - mn) / np.maximum(mx, 1e-6), soft.mean(-1)


def pile_colours(soft, sat, light):
    """What each pixel could be in a pile: 0 nothing, 1 grey stone, 2 rough tan."""
    r, b = soft[..., 0], soft[..., 2]
    tan = (sat >= GREY_SAT) & (sat < TAN_SAT) & (r >= soft.max(-1) - 0.01) & (light > TAN_LIGHT)
    tan &= contrast(light, tan) > TAN_ROUGH
    kind = np.zeros(light.shape, np.uint8)
    kind[tan] = 2
    kind[(sat < GREY_SAT) & (light > GREY_LIGHT) & (r - b > -COOL)] = 1
    return kind


def boulder_colours(soft, sat, light):
    """What each pixel could be of a boulder: 0 nothing, 1 grey or bluish stone, 2 pink stone."""
    r, g, b = soft[..., 0], soft[..., 1], soft[..., 2]
    plain = (np.minimum(r, g) - b < YELLOW) & (g - r < GREEN) & (sat < BOULDER_SAT) & (light >= DARK)
    return plain.astype(np.uint8) + (plain & (r - g > PINK))


def shape_of(part, res):
    """Centre (row, column), length, width and heading of a patch, from its principal axes.

    The heading is the direction of the long axis in radians from east, counter-clockwise,
    between -pi/2 and pi/2. Length and width are the patch's extent along and across it.
    """
    rows, cols = np.nonzero(part)
    if rows.size < 3:
        return float(rows.mean()), float(cols.mean()), res, res, 0.0
    x, y = (cols - cols.mean()) * res, -(rows - rows.mean()) * res
    _, vectors = np.linalg.eigh(np.cov(np.stack([x, y])))
    ax, ay = vectors[:, 1]
    if ax < 0 or (ax == 0 and ay < 0):
        ax, ay = -ax, -ay
    along, across = x * ax + y * ay, -x * ay + y * ax
    return float(rows.mean()), float(cols.mean()), float(np.ptp(along) + res), float(np.ptp(across) + res), float(np.arctan2(ay, ax))


def boulder_shadow(part, shadows, length, res):
    """The hard shadow thrown by one compact thing, or None if what is beside it is not its own shadow.

    shadows is the labelled map of hard shadow around it. The window it comes in reaches
    REACH beyond the thing, so a shadow that runs to the window's edge is something larger's.
    """
    beside = np.unique(shadows[ndimage.binary_dilation(part, iterations=4) & (shadows > 0)])
    if beside.size == 0:
        return None
    shadow = np.isin(shadows, beside)
    if shadow[0].any() or shadow[-1].any() or shadow[:, 0].any() or shadow[:, -1].any():
        return None
    share = shadow.sum() / part.sum()
    _, _, long, wide, _ = shape_of(shadow, res)
    if not (SHADOW[0] <= share <= SHADOW[1]) or long > SHADOW_LONG * length or long > SLENDER * wide:
        return None
    return shadow


def post_shadow(part, light, ground, res):
    """Whether the shadow beside a thing runs on like a post's, followed while it is fainter than hard shadow."""
    labels, _ = ndimage.label((light < DIM * ground) & (light > 0))
    beside = np.unique(labels[ndimage.binary_dilation(part, iterations=4) & (labels > 0)])
    if beside.size == 0:
        return False
    faint = np.isin(labels, beside)
    return bool(faint[0].any() or faint[-1].any() or faint[:, 0].any() or faint[:, -1].any()) or shape_of(faint, res)[2] > POST


def boulder_at(part, light, stone, shadows, tall, res):
    """Judge one compact patch of boulder-coloured pixels.

    Returns (name, shadow). The name is "boulder", or "thing" for something that stands and
    throws a shadow as a boulder does but is not one (a well cap, the foot of a lamp post),
    or None for what does not stand at all, when the shadow is None too.

    light, stone (every boulder-coloured patch), shadows (hard shadow, labelled) and tall (the
    scan's height) cover the same window as part.
    """
    area = part.sum() * res * res
    _, _, length, width, _ = shape_of(part, res)
    if length > ROUND * width or area < FILL * length * width:
        return None, None
    shadow = boulder_shadow(part, shadows, length, res)
    if shadow is None:
        return None, None
    # A boulder stands on plain ground. In brush and in a field of dry stalks every pale scrap
    # has a dark gap beside it, and more scraps and gaps all round.
    both = part | shadow
    around = ndimage.binary_dilation(both, iterations=16) & ~ndimage.binary_dilation(both, iterations=4)
    busy = (stone & ~part) | (shadows > 0)
    if busy[around].mean() > CROWD or np.nanmedian(tall[around]) > LEVEL:
        return None, None
    low, middle, high = np.percentile(light[part], [10, 50, 90])
    stony = BOULDER_LIGHT[0] <= middle <= BOULDER_LIGHT[1] and high >= BOULDER_GLINT and contrast(light, part)[part].mean() >= BOULDER_ROUGH
    if not stony or np.nanmax(tall[part]) > BOULDER_TALL or post_shadow(part, light, float(np.median(light[~busy & ~part])), res):
        return "thing", shadow
    return "boulder", shadow


def boulders_in(soft, sat, light, where, tall, rows, res):
    """What stands alone in one strip of the photo: a list of (name, row, column, pixels, shadow pixels).

    The name is "boulder" or "thing" (see boulder_at). Row and column are where the two small
    arrays sit in the strip. Only what has its middle row within `rows` (first, last) is
    returned, so that nothing is found twice where strips overlap.
    """
    kind = boulder_colours(soft, sat, light)
    kind[~where] = 0
    grey, pink = ndimage.binary_opening(kind == 1, _disc(CORE)), ndimage.binary_opening(kind == 2, _disc(CORE))
    stone = grey | pink
    shadows, _ = ndimage.label((light < DARK) & (soft.max(-1) > 0))
    pad = int(round(REACH / res))
    found = []
    for solid in (grey, pink):
        labels, count = ndimage.label(solid)
        if not count:
            continue
        area = ndimage.sum(solid, labels, np.arange(1, count + 1)) * res * res
        for n, sl in enumerate(ndimage.find_objects(labels), start=1):
            middle = (sl[0].start + sl[0].stop) / 2
            if not (BOULDER_AREA[0] <= area[n - 1] <= BOULDER_AREA[1]) or not (rows[0] <= middle < rows[1]):
                continue
            w = _window(sl, pad, labels.shape)
            part = labels[w] == n
            name, shadow = boulder_at(part, light[w], stone[w], shadows[w], tall[w], res)
            if name:
                found.append((name, w[0].start, w[1].start, part, shadow))
    return found


def survey(photo, piles_ground, boulder_ground, height, res):
    """One pass over the photo, strip by strip.

    Returns a full-size array of pile colours on open ground, thin things taken out, and what
    stands alone as a list of (name, row, column, pixels, shadow pixels) placed in the photo.
    """
    H, W = photo.shape[:2]
    kinds = np.zeros((H, W), np.uint8)
    boulders = []
    for r0 in range(0, H, STRIP):
        r1 = min(r0 + STRIP, H)
        a, b = max(r0 - PAD, 0), min(r1 + PAD, H)
        where = _onto(boulder_ground, (H, W), a, b, 0, W)
        cols = np.flatnonzero(where.any(axis=0))
        if cols.size == 0:
            continue
        c0, c1 = max(cols[0] - PAD, 0), min(cols[-1] + 1 + PAD, W)
        soft, sat, light = soften(np.asarray(photo[a:b, c0:c1, :3], dtype=np.float32) / 255)
        kind = pile_colours(soft, sat, light)
        kind[~_onto(piles_ground, (H, W), a, b, c0, c1)] = 0
        kind[~ndimage.binary_opening(kind > 0, _disc(THIN))] = 0
        kinds[r0:r1, c0:c1] = kind[r0 - a:r1 - a]
        tall = _onto(height, (H, W), a, b, c0, c1)
        for name, row, col, part, shadow in boulders_in(soft, sat, light, where[:, c0:c1], tall, (r0 - a, r1 - a), res):
            boulders.append((name, a + row, c0 + col, part, shadow))
    return kinds, boulders


def groups(kinds, res, cell):
    """Grey stones that lie within JOIN of each other, as windows of the photo.

    Yields (r0, r1, c0, c1, mine): the window, grown by REACH, and a boolean array over it that
    is true on the ground of this group.
    """
    k = int(round(cell / res))
    H, W = kinds.shape
    h, w = H // k, W // k
    count = (kinds[:h * k, :w * k] == 1).reshape(h, k, w, k).sum(axis=(1, 3))
    near = ndimage.binary_dilation(count > 0, iterations=max(1, int(round(JOIN / 2 / cell))))
    labels, total = ndimage.label(near, structure=np.ones((3, 3), bool))
    if not total:
        return
    stone = ndimage.sum(count, labels, np.arange(1, total + 1)) * res * res
    pad = int(round(REACH / res))
    for n, sl in enumerate(ndimage.find_objects(labels), start=1):
        if stone[n - 1] < PILE_STONE / 2:
            continue
        r0, r1 = max(sl[0].start * k - pad, 0), min(sl[0].stop * k + pad, H)
        c0, c1 = max(sl[1].start * k - pad, 0), min(sl[1].stop * k + pad, W)
        mine = np.zeros((r1 - r0, c1 - c0), bool)
        inside = _onto(labels, (h * k, w * k), r0, min(r1, h * k), c0, min(c1, w * k)) == n
        mine[:inside.shape[0], :inside.shape[1]] = inside
        yield r0, r1, c0, c1, mine


def joined(stones, reach):
    """Stones and the gaps between them, up to twice reach wide, as one body.

    Grown by reach and shrunk back by a little less, so that the neck between two small
    stones holds where an exact closing would cut it.
    """
    grown = ndimage.binary_dilation(np.pad(stones, reach), _disc(reach))
    return ndimage.binary_fill_holes(ndimage.binary_erosion(grown, _disc(max(reach - 3, 1))))[reach:-reach, reach:-reach] | stones


def within(light, support, sigma):
    """Lightness blurred among the supporting pixels only, so that nothing from outside them leaks in."""
    weight = ndimage.gaussian_filter(support.astype(np.float32), sigma)
    return ndimage.gaussian_filter(np.where(support, light, 0.0).astype(np.float32), sigma) / np.maximum(weight, 1e-3)


def streaks(light, stone):
    """How far the changes in lightness inside the stone run one way, around each pixel: 0 for none, 1 for all.

    Among stones lightness changes in every direction. Along a guard rail, a kerb or a bundle
    of branches it changes across and hardly at all along.
    """
    down, across = np.gradient(light)
    inside = ndimage.binary_erosion(stone, iterations=2)

    def nearby(values):
        return ndimage.uniform_filter(np.where(inside, values, 0.0).astype(np.float32), 21)

    xx, yy, xy = nearby(across * across), nearby(down * down), nearby(across * down)
    return np.sqrt((xx - yy) ** 2 + 4 * xy ** 2), xx + yy


def piles_in(kind, light, tall, mine, res):
    """The piles among the stone pixels of one group: a list of boolean arrays, one pile's stones each."""
    stone = (kind > 0) & mine
    grey = (kind == 1) & mine
    # Lightness from stone to stone: texture finer than a stone is blurred away first.
    even = within(light, stone, GRAIN)
    rough = contrast(even, stone)
    one_way, all_ways = streaks(even, stone)
    # An unbroken grey patch of some size that is not rough in itself is concrete: a lid, a
    # footing, a kerb. It is taken out before it is counted in with the stones beside it.
    labels, count = ndimage.label(grey)
    if count:
        index = np.arange(1, count + 1)
        plain = (ndimage.sum(grey, labels, index) * res * res >= PLAIN) & (ndimage.mean(rough, labels, index) < PILE_ROUGH)
        plain = np.concatenate([[False], plain])[labels]
        grey &= ~plain
        stone &= ~ndimage.binary_dilation(plain, iterations=2)
    # Stones within reach of each other are one heap, with the rough tan faces among them.
    heaps, total = ndimage.label(joined(grey, int(round(PILE_JOIN / res))))
    found = []
    for h, sl in enumerate(ndimage.find_objects(heaps), start=1):
        heap = heaps[sl] == h
        seen = grey[sl] & heap
        area = seen.sum() * res * res
        if area < PILE_STONE or rough[sl][seen].mean() < PILE_ROUGH:
            continue
        if one_way[sl][seen].sum() > STREAKED * all_ways[sl][seen].sum() or np.nanmax(tall[sl][seen]) > STEEP * np.sqrt(area):
            continue
        stones = heap & stone[sl] | seen
        low, middle, high = np.percentile(light[sl][stones], [10, 50, 90])
        if high - middle < FACETS:
            continue
        whole = np.zeros(stone.shape, bool)
        whole[sl] = stones
        found.append(whole)
    return found


def stretches(stones, res):
    """A long line of stones cut across into stretches of equal stone, each at most about RUN long."""
    row, col, length, _, angle = shape_of(stones, res)
    if length <= RUN or stones.sum() * res * res > SLIM * length:
        return [stones]
    rows, cols = np.nonzero(stones)
    along = (cols - col) * np.cos(angle) - (rows - row) * np.sin(angle)
    cuts = np.quantile(along, np.linspace(0.0, 1.0, int(np.ceil(length / RUN)) + 1))
    cuts[-1] += 1.0
    pieces = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        inside = (along >= a) & (along < b)
        piece = np.zeros(stones.shape, bool)
        piece[rows[inside], cols[inside]] = True
        pieces.append(piece)
    return pieces


def settle(stones, grey, boulders, buried, window):
    """Sort out what stands alone in or against a pile. Returns the pile's stones without it.

    A boulder half of whose rim is the pile's grey stone is one of the pile's stones: it is marked
    in `buried` and counted in. One that only stands against the pile stays a boulder and is
    cut out of it. A thing that is no boulder (a well cap on a patch of dirt) is cut out too.
    """
    for i, (name, row, col, part, shadow) in enumerate(boulders):
        here = _placed(part, row, col, *window)
        if not here.any():
            continue
        rim = ndimage.binary_dilation(here, iterations=8) & ~ndimage.binary_dilation(here, iterations=2) & ~_placed(shadow, row, col, *window)
        if name == "boulder" and (rim & stones & grey).sum() >= BURIED * rim.sum():
            buried[i] = True
            stones = stones | here
        else:
            stones = stones & ~ndimage.binary_dilation(here, iterations=2)
    return stones


def shadow_of(stones, dark, res):
    """The hard shadow joined on to a pile: dark pixels that touch it, as far as REACH / 2."""
    labels, _ = ndimage.label(dark)
    touching = np.unique(labels[ndimage.binary_dilation(stones, iterations=2) & dark])
    shadow = np.isin(labels, touching[touching > 0]) & ~stones
    return shadow & ndimage.binary_dilation(stones, iterations=int(round(REACH / 2 / res)))


def outline_of(region, res, x_left, y_top):
    """A closed polygon round a patch, in scene metres, as a list of [x, y]."""
    field = ndimage.gaussian_filter(np.pad(region, 3).astype(np.float32), 1.0)
    line = max(edgelines.contours(field, 0.5), key=len)
    xy = np.stack([x_left + (line[:, 1] - 3 + 0.5) * res, y_top - (line[:, 0] - 3 + 0.5) * res], axis=1)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in railmodel.simplify(xy, OUTLINE)]


def describe(stones, shadow, name, rgb, res, x_left, y_top):
    """The record of one find and the pixels to paint out for it."""
    # The outline goes round everything that belongs to the find: a pile's stones are joined
    # across the gaps between them first. Its size and heading are those of what the outline holds.
    whole = joined(stones, int(round(PILE_JOIN / res))) if name == "pile" else ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(stones, 2), _disc(2)))[2:-2, 2:-2] | stones
    labels, count = ndimage.label(whole)
    if count > 1:
        whole = labels == 1 + int(np.argmax(ndimage.sum(whole, labels, np.arange(1, count + 1))))
    row, col, length, width, angle = shape_of(whole, res)
    rock = {"x": round(float(x_left + (col + 0.5) * res), 3), "y": round(float(y_top - (row + 0.5) * res), 3), "kind": name,
            "length": round(length, 3), "width": round(width, 3), "angle": round(angle, 4), "area": round(float(stones.sum() * res * res), 3),
            "colour": [round(float(v), 3) for v in rgb[stones].mean(axis=0)], "outline": outline_of(whole, res, x_left, y_top)}
    return rock, ndimage.binary_dilation(whole | stones | shadow, iterations=int(round(GROW / res)))


def find(photo, raster, paved, height, trees, buildings, cell, log=print):
    """Returns (rocks, mask).

    rocks is a list of dicts, sorted by x and then y:
        x, y      scene metres, the middle of the stone seen
        kind      "pile" or "boulder"
        length, width, angle   extent along and across the long axis, and its direction in
                  radians from east, counter-clockwise
        area      square metres of stone seen
        colour    [r, g, b] in 0..1, the mean colour of the stone
        outline   [[x, y], ...] a closed polygon round the stones
    mask is a full-size boolean array of what to paint out: the stones and their shadows, grown
    a little.

    photo is the orthophoto, paved the pavement map on its own finer grid, height, trees and
    buildings the scan's maps on cells of size cell.
    """
    res = raster["res"]
    H, W = photo.shape[:2]
    piles_ground, boulder_ground = open_ground(paved, height, trees, buildings, cell, res * H / paved.shape[0])
    kinds, alone = survey(photo, piles_ground, boulder_ground, height, res)
    log(f"  rocks: {(kinds == 1).sum() * res * res:.0f} m2 of colourless ground and {len(alone)} things standing alone to look at")
    rocks = []
    mask = np.zeros((H, W), bool)
    buried = np.zeros(len(alone), bool)
    for r0, r1, c0, c1, mine in groups(kinds, res, cell):
        rgb = np.asarray(photo[r0:r1, c0:c1, :3], dtype=np.float32) / 255
        light = ndimage.gaussian_filter(rgb, (SOFT, SOFT, 0)).mean(-1)
        dark = (light < DARK) & (rgb.max(-1) > 0)
        grey = kinds[r0:r1, c0:c1] == 1
        for stones in piles_in(kinds[r0:r1, c0:c1], light, _onto(height, (H, W), r0, r1, c0, c1), mine, res):
            stones = settle(stones, grey, alone, buried, (r0, r1, c0, c1))
            if (stones & grey).sum() * res * res < PILE_STONE:
                continue
            for stretch in stretches(stones, res):
                rock, patch = describe(stretch, shadow_of(stretch, dark, res), "pile", rgb, res, raster["x0"] + c0 * res, raster["y1"] - r0 * res)
                rocks.append(rock)
                mask[r0:r1, c0:c1] |= patch
    for (name, row, col, part, shadow), inside in zip(alone, buried):
        if inside or name != "boulder":
            continue
        rgb = np.asarray(photo[row:row + part.shape[0], col:col + part.shape[1], :3], dtype=np.float32) / 255
        rock, patch = describe(part, shadow, "boulder", rgb, res, raster["x0"] + col * res, raster["y1"] - row * res)
        rocks.append(rock)
        mask[row:row + part.shape[0], col:col + part.shape[1]] |= patch
    rocks.sort(key=lambda rock: (rock["x"], rock["y"]))
    piles = sum(rock["kind"] == "pile" for rock in rocks)
    log(f"  rocks: {piles} piles and {len(rocks) - piles} boulders, {mask.sum() * res * res:.0f} m2 to paint out")
    return rocks, mask
