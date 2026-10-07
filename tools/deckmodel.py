"""A timber deck against a building's wall, with its ramps, its steps and its steel handrails.

The scan shows such a thing well from above and badly from the side: every rail has melted
into a tent. So its plan is read by hand, as rectangles in the wall's own frame, and so are
the levels of its floors, which the scan does get right. s runs along the wall and d out from
it. A platform is a level rectangle. A ramp is a rectangle that slopes along s from one level
to another, and steps are a ramp cut into risers. Where a level is not given, that end
stands on the ground. A handrail is a line of points on the plan: it takes its height from
whatever floor is under each point.

Where the wall itself stands is the building's business and not known here to better than a
few tenths of a metre: the roof hides it from above. So whatever is given as reaching the wall
(d of 0) carries on BEHIND it, platforms and rails alike, and is hidden inside the building
wherever the wall is drawn.

Built where it stands, in scene coordinates. Timber is drawn as slabs and posts and every rail
as a top tube, a tube halfway and one at the foot, on posts that go down into the ground: the
pickets between them are thinner than the renderer can draw without shimmer.
"""
import numpy as np

import tubes
from propmodel import BURIED, UP, solid

THICK = 0.14          # boards and the joists under them, drawn as one slab
BEHIND = 0.8          # metres that whatever reaches the wall line carries on behind it
TIMBER = 0.06         # from the middle of a timber post to its corner
TUBE = 0.032          # and of a rail
POST = 0.038          # and of a rail's post
RAILS = (0.95, 0.53, 0.12)   # the tubes' heights above the floor
SPAN = 1.4            # the most metres between a rail's posts
RISER = 0.19          # about the height of one step
COLOURS = {"wood": (0.73, 0.71, 0.67), "steel": (0.72, 0.74, 0.75)}


def _inside(box, s, d):
    (s0, s1), (d0, d1) = sorted(box["s"]), sorted(box["d"])
    return s0 - 1e-6 <= s <= s1 + 1e-6 and d0 - 1e-6 <= d <= d1 + 1e-6


def make(site, spec):
    """{"wood": ..., "steel": ...} for the deck that spec describes, on site: s along the wall, d out from it."""
    platforms, ramps, steps = spec.get("platforms", []), spec.get("ramps", []), spec.get("steps", [])

    def ends(run, d):
        """The levels of a ramp's or a flight's two ends, at d across it."""
        levels = run.get("levels", [run.get("level"), None])
        return [site.ground(s, d) if z is None else z for s, z in zip(run["s"], levels)]

    def floor(s, d):
        """The level of whatever is underfoot at (s, d). Behind the wall line it is what is at the wall."""
        d = max(d, 0.0)
        for p in platforms:
            if _inside(p, s, d):
                return p["level"]
        for run in ramps + steps:
            if _inside(run, s, d):
                z0, z1 = ends(run, d)
                return z0 + (s - run["s"][0]) / (run["s"][1] - run["s"][0]) * (z1 - z0)
        return site.ground(s, d)

    wood, steel = [], []
    for p in platforms:
        s, d = sorted(p["s"]), sorted(p["d"])
        wood.append(site.box(s, (d[0] - BEHIND if d[0] <= 0 else d[0], d[1]), (p["level"] - THICK, p["level"])))
        # A post under each corner that is out in the open and far enough off the ground to need one.
        for a in (s[0] + 0.1, s[1] - 0.1):
            for b in (d[0] + 0.1, d[1] - 0.1):
                if b > 0.3 and p["level"] - THICK - site.ground(a, b) > 0.25:
                    wood.append(tubes.strut(site.at(a, b, site.ground(a, b) - BURIED), site.at(a, b, p["level"] - 0.02), TIMBER))
    for run in ramps:
        ring = [(run["s"][0], run["d"][0]), (run["s"][1], run["d"][0]), (run["s"][1], run["d"][1]), (run["s"][0], run["d"][1])]
        top = [site.at(s, d, ends(run, d)[k]) for (s, d), k in zip(ring, (0, 1, 1, 0))]
        wood.append(solid([p - THICK * UP for p in top], top))
    for run in steps:
        d = sorted(run["d"])
        head = run["level"]
        foot = min(ends(run, d[0])[1], ends(run, d[1])[1])
        count = max(int(round((head - foot) / RISER)), 2)          # risers: one tread fewer
        tread = (run["s"][1] - run["s"][0]) / (count - 1)
        for k in range(1, count):
            s = (run["s"][0] + (k - 1) * tread, run["s"][0] + k * tread)
            wood.append(site.box(s, d, (site.span(s, d)[0] - BURIED, head - k * (head - foot) / count)))

    stood = set()                                                   # where a rail post already stands

    def post(s, d, z):
        key = (round(s, 2), round(d, 2))
        if key not in stood:
            stood.add(key)
            steel.append(tubes.strut(site.at(s, d, min(site.ground(s, d), z) - BURIED), site.at(s, d, z + RAILS[0] + 0.03), POST))

    for line in spec.get("rails", []):
        points = [(s, d - BEHIND if d <= 0 else d, floor(s, d)) for s, d in line]
        for (s0, d0, z0), (s1, d1, z1) in zip(points[:-1], points[1:]):
            count = max(int(np.ceil(np.hypot(s1 - s0, d1 - d0) / SPAN)), 1)
            for t in np.linspace(0.0, 1.0, count + 1):
                post(s0 + t * (s1 - s0), d0 + t * (d1 - d0), z0 + t * (z1 - z0))
            for high in RAILS:
                steel.append(tubes.strut(site.at(s0, d0, z0 + high), site.at(s1, d1, z1 + high), TUBE))
    return {"wood": tubes.merge(wood), "steel": tubes.merge(steel)}
