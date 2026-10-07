"""The shapes of the plainer small structures: a shed, a cabinet, a concrete pad, a rack of pipes.

The scan gives each of these as a melted lump with a sharp top. What can be read from it is
where the thing stands, how big it is and what colour, and what kind of thing it is. The
shape is then the ordinary shape of that kind of thing, at the size read.

Everything is built where it stands, in scene coordinates, on a Site: a spot on the ground
with a heading. So a thing on a slope has each foot on the ground it is over, and feet, walls
and posts carry on BURIED below it. Every piece is closed and wound to face outward. The
tanks are in tankmodel.py, the deck in deckmodel.py and the plant yard in yardmodel.py.
"""
import numpy as np

import tubes

BURIED = 0.3          # metres that anything standing on the ground carries on below it
UP = np.array([0.0, 0.0, 1.0])
COLOURS = {
    "concrete": (0.74, 0.73, 0.70),
    "walls": (0.52, 0.28, 0.33),        # the shed's maroon
    "roof": (0.93, 0.94, 0.93),
    "trim": (0.92, 0.92, 0.90),
    "door": (0.80, 0.80, 0.78),
    "body": (0.80, 0.81, 0.80),         # a cabinet's sheet metal, unless its entry or its block says otherwise
    "pipe": (0.90, 0.91, 0.92),
    "stand": (0.36, 0.33, 0.30),
}


class Site:
    """A spot on the ground with a heading. s runs along the heading and d across it.

    along and out are directions on the ground. out is squared to along, so only the side it
    points to matters. ground(xs, ys) gives the ground's height.
    """

    def __init__(self, origin, along, out, ground):
        self.origin = np.asarray(origin, float)[:2]
        along = np.asarray(along, float)[:2]
        self.along = along / np.hypot(*along)
        out = np.asarray(out, float)[:2]
        out = out - (out @ self.along) * self.along
        self.out = out / np.hypot(*out)
        self._ground = ground

    @classmethod
    def heading(cls, x, y, yaw, ground):
        """A site at (x, y) with s along yaw, in radians from east, and d to its left."""
        return cls((x, y), (np.cos(yaw), np.sin(yaw)), (-np.sin(yaw), np.cos(yaw)), ground)

    def xy(self, s, d):
        return self.origin + s * self.along + d * self.out

    def ground(self, s, d):
        x, y = self.xy(s, d)
        return float(self._ground(np.array([x]), np.array([y]))[0])

    def span(self, s, d):
        """The lowest and the highest ground at the corners of a rectangle."""
        heights = [self.ground(a, b) for a in s for b in d]
        return min(heights), max(heights)

    def at(self, s, d, z):
        x, y = self.xy(s, d)
        return np.array([x, y, z])

    def box(self, s, d, z):
        """A box between two values of each of s, d and z."""
        centre = self.at((s[0] + s[1]) / 2, (d[0] + d[1]) / 2, (z[0] + z[1]) / 2)
        return tubes.box(centre, ((s[1] - s[0]) / 2 * np.append(self.along, 0.0), (d[1] - d[0]) / 2 * np.append(self.out, 0.0), (z[1] - z[0]) / 2 * UP))


def solid(bottom, top):
    """A block with four corners below and the matching four above, each in order round the outline."""
    vertices = np.concatenate([np.asarray(bottom, float), np.asarray(top, float)])
    faces = np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    return (vertices, faces) if tubes.volume(vertices, faces) > 0 else (vertices, faces[:, ::-1])


def lathe(a, b, profile, sides):
    """A shape turned about the line from a to b.

    profile is a list of (fraction of the way from a to b, radius), in order. A radius of 0
    closes the shape to a point. An end left open by its radius gets a flat cap.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    along = (b - a) / np.linalg.norm(b - a)
    side = np.cross(along, UP if abs(along[2]) < 0.9 else [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    other = np.cross(along, side)
    turn = (np.arange(sides) + 0.5) * 2 * np.pi / sides
    circle = np.cos(turn)[:, None] * side + np.sin(turn)[:, None] * other
    if profile[0][1] > 0:
        profile = [(profile[0][0], 0.0)] + list(profile)
    if profile[-1][1] > 0:
        profile = list(profile) + [(profile[-1][0], 0.0)]
    vertices, rings = [], []                  # a ring is (index of its first vertex, how many)
    for t, r in profile:
        centre = a + t * (b - a)
        rings.append((len(vertices), sides if r > 0 else 1))
        vertices += list(centre + r * circle) if r > 0 else [centre]
    faces = []
    for (p, np_), (q, nq) in zip(rings[:-1], rings[1:]):
        for i in range(sides):
            j = (i + 1) % sides
            if np_ == 1 and nq > 1:
                faces.append((p, q + j, q + i))
            elif np_ > 1 and nq == 1:
                faces.append((p + i, p + j, q))
            elif np_ > 1 and nq > 1:
                faces += [(p + i, p + j, q + j), (p + i, q + j, q + i)]
    vertices, faces = np.array(vertices), np.array(faces)
    return (vertices, faces) if tubes.volume(vertices, faces) > 0 else (vertices, faces[:, ::-1])


def pad(site, length, width, kerb=0.0, lip=0.15, proud=0.05):
    """A level concrete slab about the site, proud of the highest ground under it.

    With a kerb it is a basin: a rim that wide and lip tall round its edge. A basin on a slope
    is cut into it, as the one on this site is: its floor is proud of the middle ground, and
    on the high side the rim holds the ground back. Returns the slab and the level of its top.
    """
    s, d = (-length / 2, length / 2), (-width / 2, width / 2)
    low, high = site.span(s, d)
    if not kerb:
        return site.box(s, d, (low - BURIED, high + proud)), high + proud
    top = max(np.mean([site.ground(a, b) for a in s for b in d]) + proud, high - lip + 0.03)
    inner_s, inner_d = (s[0] + kerb - 0.01, s[1] - kerb + 0.01), (d[0] + kerb - 0.01, d[1] - kerb + 0.01)
    z = (low - BURIED, top + lip)
    pieces = [site.box(inner_s, inner_d, (low - BURIED, top)),
              site.box(s, (d[0], d[0] + kerb), z), site.box(s, (d[1] - kerb, d[1]), z),
              site.box((s[0], s[0] + kerb), (d[0] + kerb, d[1] - kerb), z), site.box((s[1] - kerb, s[1]), (d[0] + kerb, d[1] - kerb), z)]
    return tubes.merge(pieces), top


def cabinet(site, length, width, height, base):
    """An equipment cabinet standing on the level base: a box under a lid that overhangs it,
    with a plinth, and on each long side two doors shown by the raised strips round them."""
    half_l, half_w = length / 2, width / 2
    foot = min(site.span((-half_l, half_l), (-half_w, half_w))[0] - BURIED, base - 0.02)
    pieces = [site.box((-half_l, half_l), (-half_w, half_w), (base + 0.08, base + height - 0.04)),
              site.box((-half_l + 0.03, half_l - 0.03), (-half_w + 0.03, half_w - 0.03), (foot, base + 0.09)),
              site.box((-half_l - 0.04, half_l + 0.04), (-half_w - 0.04, half_w + 0.04), (base + height - 0.06, base + height))]
    low, high = base + 0.16, base + height - 0.14
    for side in (-1.0, 1.0):
        face = sorted((side * (half_w - 0.01), side * (half_w + 0.02)))
        for s in (-half_l + 0.06, 0.0, half_l - 0.06):
            pieces.append(site.box((s - 0.025, s + 0.025), face, (low, high)))
        for z in (low, high):
            pieces.append(site.box((-half_l + 0.035, half_l - 0.035), face, (z - 0.025, z + 0.025)))
    return tubes.merge(pieces)


def shed(site, length, width, high, low, overhang=0.15, thick=0.12, door=(1.8, 2.0), fascia=0.5):
    """A shed with a roof of one slope, its length along s and its high end at +s.

    length and width are the roof's, high and low its upper face at the two ends above the
    highest ground under the shed. The walls stand overhang inside the roof's edge. The door
    is in the high end, under a band of trim.
    Returns {"walls": ..., "roof": ..., "trim": ..., "door": ...}.
    """
    half_l, half_w = length / 2 - overhang, width / 2 - overhang
    ground_low, floor = site.span((-half_l, half_l), (-half_w, half_w))
    foot = ground_low - BURIED

    def under(s):
        """The roof's lower face over s."""
        return floor + low + (high - low) * (s + length / 2) / length - thick

    ring = [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)]
    walls = solid([site.at(s, d, foot) for s, d in ring], [site.at(s, d, under(s) + 0.02) for s, d in ring])
    rise = (high - low) / 2
    roof = tubes.box(site.at(0.0, 0.0, floor + (high + low) / 2 - thick / 2),
                     (length / 2 * np.append(site.along, 0.0) + rise * UP, width / 2 * np.append(site.out, 0.0), thick / 2 * UP))
    board, proud = 0.11, 0.025
    trim = []
    for s, d in ring:                          # a board down each corner, proud of both walls
        ss = sorted((s + np.sign(s) * proud, s - np.sign(s) * (board - proud)))
        dd = sorted((d + np.sign(d) * proud, d - np.sign(d) * (board - proud)))
        trim.append(site.box(ss, dd, (foot, under(min(ss)) + 0.03)))
    front = (half_l - 0.01, half_l + proud)
    trim.append(site.box(front, (-half_w + board - proud, half_w - board + proud), (under(half_l) - fascia, under(half_l) + 0.03)))
    wide, tall = min(door[0], 1.1 * half_w), min(door[1], under(half_l) - fascia - floor - 0.15)
    frame = 0.09
    for d in (-wide / 2 - frame, wide / 2):    # the door's frame, and the line where its two leaves meet
        trim.append(site.box((half_l - 0.01, half_l + 0.04), (d, d + frame), (floor - 0.02, floor + tall + frame)))
    trim.append(site.box((half_l - 0.01, half_l + 0.04), (-wide / 2, wide / 2), (floor + tall, floor + tall + frame)))
    trim.append(site.box((half_l - 0.01, half_l + 0.045), (-0.025, 0.025), (floor + 0.02, floor + tall)))
    leaves = site.box((half_l - 0.01, half_l + 0.02), (-wide / 2, wide / 2), (floor - 0.02, floor + tall))
    return {"walls": walls, "roof": roof, "trim": tubes.merge(trim), "door": leaves}


def pipes(runs, diameter, rest, ground, sides=12, stands=4):
    """Long pipes lying side by side on trestles. runs is a list of (one end, the other end),
    rest is how far a pipe's underside is above the ground. Returns {"pipe": ..., "stand": ...}."""
    runs = [(np.asarray(a, float)[:2], np.asarray(b, float)[:2]) for a, b in runs]

    def lift(p, above):
        return np.array([p[0], p[1], float(ground(p[:1], p[1:2])[0]) + above])

    pipe = [tubes.strut(lift(a, rest + diameter / 2), lift(b, rest + diameter / 2), diameter / 2, sides=sides) for a, b in runs]
    # The trestles stand square to the longest pipe and reach under all of them.
    a, b = max(runs, key=lambda r: np.hypot(*(r[1] - r[0])))
    along = (b - a) / np.hypot(*(b - a))
    across = np.array([-along[1], along[0]])
    offsets = [float((p - a) @ across) for run in runs for p in run]
    near, far = min(offsets) - diameter / 2 - 0.15, max(offsets) + diameter / 2 + 0.15
    stand = []
    for t in np.linspace(0.08, 0.92, stands):
        middle = a + t * (b - a)
        ends = [middle + near * across, middle + far * across]
        legs = [middle + (near + 0.1) * across, middle + (far - 0.1) * across]
        level = max(lift(p, 0.0)[2] for p in ends) + rest - 0.06       # the beam's middle: its top is just under the pipes
        stand.append(tubes.strut([*ends[0], level], [*ends[1], level], 0.07))
        stand += [tubes.strut(lift(p, -BURIED), [*p, level], 0.06) for p in legs]
    return {"pipe": tubes.merge(pipe), "stand": tubes.merge(stand)}
