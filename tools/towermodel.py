"""A lattice tower with a small wind turbine at its top.

Four legs that lean in toward the top, tied at every panel by a ring of horizontals and
crossed on each face by two diagonals: the galvanised tower that carries a small turbine or
an aerial. From above the scan shows only a foreshortened picture of it lying on the grass.
Its shadow is the true record: the length gives the height, and what the shadow carries at
its tip shows there is a machine at the top with a tail.

The model stands about the origin, z up from the ground at its feet, its sides square to x
and y. Lengths in metres.
"""
import numpy as np

import tubes

LEG = 0.08           # from the middle of a leg to its corner
BRACE = 0.035        # and of a horizontal or a diagonal
PANEL = 2.6          # metres between rings of horizontals, about
TOP = 0.55           # metres across the tower where the turbine sits
BURIED = 0.6         # how far the legs carry on below z = 0, for sloping ground
BLADE = 3.2          # metres from the hub to a blade's tip
TAIL = 2.8           # metres of boom behind the machine
COLOURS = {"steel": (0.70, 0.72, 0.73), "turbine": (0.86, 0.87, 0.88)}


def make(height, base, tail_azimuth=90.0):
    """{"steel": (vertices, faces), "turbine": (vertices, faces)} for a tower with its hub at height.

    base is the distance between neighbouring feet. tail_azimuth is the way the turbine's
    tail points, in degrees clockwise from north: the rotor faces the other way.
    """
    top = height - 0.5                       # the steel stops under the machine
    count = max(int(round(top / PANEL)), 2)
    levels = np.linspace(0.0, top, count + 1)

    def corners(z):
        half = (base + (TOP - base) * z / top) / 2
        return np.array([[half, half, z], [-half, half, z], [-half, -half, z], [half, -half, z]])

    steel = []
    foot, head = corners(-BURIED), corners(top)
    steel += [tubes.strut(foot[k], head[k], LEG) for k in range(4)]
    for lower, upper in zip(levels[:-1], levels[1:]):
        low, high = corners(lower), corners(upper)
        for k in range(4):
            j = (k + 1) % 4
            steel += [tubes.strut(high[k], high[j], BRACE), tubes.strut(low[k], high[j], BRACE), tubes.strut(low[j], high[k], BRACE)]

    away = np.radians(tail_azimuth)
    tail = np.array([np.sin(away), np.cos(away), 0.0])
    front, side, up = -tail, np.array([np.cos(away), -np.sin(away), 0.0]), np.array([0.0, 0.0, 1.0])
    hub = np.array([0.0, 0.0, height])
    turbine = [tubes.box(hub, (0.7 * front, 0.24 * side, 0.24 * up)),
               tubes.strut(hub + 0.7 * tail, hub + (0.7 + TAIL) * tail, 0.04),
               tubes.box(hub + (0.7 + TAIL) * tail, (0.55 * tail, 0.012 * side, 0.42 * up)),
               tubes.strut(hub + 0.7 * front, hub + 0.95 * front, 0.16, sides=6),
               tubes.strut(np.array([0.0, 0.0, top]), hub - 0.24 * up, 0.09, sides=6)]
    for angle in np.radians([90.0, 210.0, 330.0]):
        out = np.cos(angle) * side + np.sin(angle) * up
        chord = np.cross(front, out)
        turbine.append(tubes.box(hub + 0.85 * front + BLADE / 2 * out, (BLADE / 2 * out, 0.11 * chord, 0.02 * front)))
    return {"steel": tubes.merge(steel), "turbine": tubes.merge(turbine)}
