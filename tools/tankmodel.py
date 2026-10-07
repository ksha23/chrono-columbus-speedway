"""Tanks: a propane tank on its saddles, a fuel tank on its skid, and what stands round them.

A horizontal pressure tank is a cylinder with domed ends, whichever tank it is, so the scan
has only to give its length, its diameter and how high its crown is. The scan's own shape for
one is a loaf: the underside is never seen from the air, and the photographs fill it in down
to the grass.

Built where it stands, in scene coordinates, on a propmodel.Site with the tank's length along s.
"""
import numpy as np

import tubes
from propmodel import BURIED, UP, lathe

COLOURS = {
    "shell": (0.93, 0.94, 0.94),        # white paint
    "feet": (0.42, 0.43, 0.44),         # saddles and skids
    "steel": (0.70, 0.72, 0.73),        # galvanised tube
    "concrete": (0.74, 0.73, 0.70),
    "bollard": (0.90, 0.72, 0.16),
    "cap": (0.62, 0.14, 0.12),          # the red fittings on a fuel tank
}


def tank(site, length, diameter, top, head=0.25, sides=24, stand=None, skid=False):
    """A horizontal tank on two saddles, its middle over the site and its length along s.

    top is how far its crown is above what it stands on: the ground under its middle, or the
    level stand if one is given. head is how far each end bulges, as a fraction of the
    diameter: 0.25 for a pressure tank, next to nothing for a fuel tank. A fuel tank sits on
    a skid: two runners under its saddles.
    Returns {"shell": ..., "feet": ...}.
    """
    radius, foot = diameter / 2, 0.28 * length
    base = site.ground(0.0, 0.0) if stand is None else stand
    axis = base + top - radius
    bulge = head * diameter / length
    arc = [(bulge * (1 - np.cos(t)), radius * np.sin(t)) for t in np.radians([30.0, 60.0, 90.0])]
    profile = [(0.0, 0.0)] + arc + [(1 - t, r) for t, r in reversed(arc)] + [(1.0, 0.0)]
    shell = [lathe(site.at(-length / 2, 0.0, axis), site.at(length / 2, 0.0, axis), profile, sides)]
    # The fittings stand in a low dome on the crown.
    crown = site.at(0.0, 0.0, axis + radius)
    shell.append(tubes.strut(crown - 0.06 * UP, crown + 0.15 * UP, 0.14, sides=12))
    feet = []
    reach = 0.36 * diameter                    # a saddle's half width: where its top corners just meet the shell
    for s in (-foot, foot):
        under = base if stand is not None else site.span((s - 0.1, s + 0.1), (-reach, reach))[0] - BURIED
        feet.append(site.box((s - 0.1, s + 0.1), (-reach, reach), (under, axis - reach)))
    if skid:
        for d in (-reach + 0.06, reach - 0.06):
            feet.append(site.box((-0.42 * length, 0.42 * length), (d - 0.05, d + 0.05), (base - 0.01, base + 0.10)))
    return {"shell": tubes.merge(shell), "feet": tubes.merge(feet)}


def fuel_fittings(site, length, diameter, top, stand):
    """What stands on a fuel tank's crown: a fill cap and an emergency vent, both red."""
    crown = stand + top
    pieces = [tubes.strut(site.at(s, 0.0, crown - 0.08), site.at(s, 0.0, crown + high), r, sides=8)
              for s, r, high in ((-0.3 * length, 0.07, 0.12), (0.28 * length, 0.05, 0.20))]
    return tubes.merge(pieces)


def guard(a, b, height, ground, radius=0.03):
    """A hoop of tube standing from a to b: two legs, a rail over them and a rail halfway up."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    za, zb = (float(ground(p[:1], p[1:2])[0]) for p in (a, b))
    pieces = [tubes.strut([*a, za - BURIED], [*a, za + height], radius), tubes.strut([*b, zb - BURIED], [*b, zb + height], radius)]
    for part in (1.0, 0.5):
        pieces.append(tubes.strut([*a, za + part * height - radius], [*b, zb + part * height - radius], radius))
    return tubes.merge(pieces)


def bollards(points, height, ground, radius=0.085):
    """Round posts at the given points, each on its own ground."""
    pieces = []
    for p in np.asarray(points, float):
        z = float(ground(p[:1], p[1:2])[0])
        pieces.append(tubes.strut([p[0], p[1], z - BURIED], [p[0], p[1], z + height], radius, sides=8))
    return tubes.merge(pieces)
