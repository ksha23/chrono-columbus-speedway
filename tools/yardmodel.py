"""A yard of air-handling plant against a building's wall: the units, their ducts, the screens round them.

The units are the kind made to stand on a roof, here stood on the ground: a long sheet-metal
box on a base rail, with one or two fans let into its lid, and a duct from it to a riser that
goes up the building's wall. Screen walls of ribbed sheet on posts stand round each one, as
tall as the unit, so from the road it is the screens that are seen and the fans over them.

Read by hand from the scan in the wall's own frame: s along the wall and d out from it. Built
where it stands, in scene coordinates. Each unit and each enclosure is level, on the highest
ground under it, and carries on down to the lowest.
"""
import numpy as np

import tubes
from propmodel import BURIED

BEHIND = 1.0          # metres that screens and risers carry on behind the wall line, to be sure of reaching the wall
PANEL = 0.03          # half the thickness of a screen
POST = 0.05           # and of one of its posts
BAY = 2.4             # the most metres between posts
RAIL = 0.14           # the height of a unit's base rail
COLOURS = {"unit": (0.86, 0.87, 0.87), "fan": (0.20, 0.22, 0.25), "base": (0.30, 0.31, 0.33), "screen": (0.54, 0.58, 0.64), "duct": (0.74, 0.76, 0.78)}


def _unit(site, unit):
    """One unit: (body, fans, base)."""
    s, d = sorted(unit["s"]), sorted(unit["d"])
    low, high = site.span(s, d)
    top = high + unit["height"]
    body = [site.box(s, d, (high + RAIL, top - 0.05)),
            site.box((s[0] - 0.03, s[1] + 0.03), (d[0] - 0.03, d[1] + 0.03), (top - 0.06, top))]       # the lid overhangs
    # The seams between its panels, as raised strips on the two long sides.
    long_s = s[1] - s[0] >= d[1] - d[0]
    a, b = (s, d) if long_s else (d, s)
    for t in (1 / 3, 2 / 3):
        x = a[0] + t * (a[1] - a[0])
        for face in ((b[0] - 0.012, b[0] + 0.01), (b[1] - 0.01, b[1] + 0.012)):
            strip = ((x - 0.025, x + 0.025), face)
            body.append(site.box(*(strip if long_s else strip[::-1]), (high + RAIL + 0.03, top - 0.08)))
    fans = []
    for fs, fd in unit.get("fans", []):
        fans.append(tubes.strut(site.at(fs, fd, top - 0.02), site.at(fs, fd, top + 0.10), unit.get("fan_diameter", 0.8) / 2, sides=16))
        body.append(tubes.strut(site.at(fs, fd, top - 0.02), site.at(fs, fd, top + 0.14), 0.11, sides=8))   # the motor's cap
    base = site.box((s[0] + 0.04, s[1] - 0.04), (d[0] + 0.04, d[1] - 0.04), (low - BURIED, high + RAIL + 0.01))
    return body, fans, base


def _screens(site, yard, height):
    """The three screen walls of one enclosure: the building is the fourth."""
    s, depth = sorted(yard["s"]), yard["depth"]
    low, high = site.span(s, (0.0, depth))
    foot, top = low - BURIED, high + height
    pieces = [site.box((x - PANEL, x + PANEL), (-BEHIND, depth), (foot, top)) for x in s]
    pieces.append(site.box(s, (depth - PANEL, depth + PANEL), (foot, top)))
    stand = [(x, y) for x in s for y in np.linspace(0.0, depth, int(np.ceil(depth / BAY)) + 1)]
    stand += [(x, depth) for x in np.linspace(s[0], s[1], int(np.ceil((s[1] - s[0]) / BAY)) + 1)[1:-1]]
    pieces += [site.box((x - POST, x + POST), (y - POST, y + POST), (foot, top + 0.06)) for x, y in stand]
    return pieces


def make(site, spec):
    """{"unit": ..., "fan": ..., "base": ..., "screen": ..., "duct": ...} for the yard that spec describes."""
    unit, fan, base, duct = [], [], [], []
    for u in spec.get("units", []):
        body, fans, rail = _unit(site, u)
        unit += body
        fan += fans
        base.append(rail)
    screen = [piece for yard in spec.get("enclosures", []) for piece in _screens(site, yard, spec.get("screen_height", 2.4))]
    for run in spec.get("ducts", []):
        s, d = sorted(run["s"]), sorted(run["d"])
        d = (-BEHIND if d[0] <= 0 else d[0], d[1])
        low, high = site.span(s, (max(d[0], 0.0), d[1]))
        duct.append(site.box(s, d, (low - BURIED, high + run["height"])))
    parts = {"unit": unit, "fan": fan, "base": base, "screen": screen, "duct": duct}
    return {name: tubes.merge(pieces) for name, pieces in parts.items() if pieces}
