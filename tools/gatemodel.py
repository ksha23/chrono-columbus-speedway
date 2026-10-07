"""A swing gate across a road: a tubular leaf on a hinge post, and the rail that runs off from it.

The gates here are in pairs, one each side of a road, each leaf half the road's width. Open,
a leaf lies back along the verge. From its hinge post a single rail on posts runs away from
the road, so that nobody drives round the gate. All of it is galvanised tube.

A gate is built in place, in scene coordinates, like the guard rails.
"""
import numpy as np

import tubes

POST = 0.045         # half the thickness of a post
RAIL = 0.04          # and of a rail or one of the leaf's tubes
HINGE = 0.06         # and of the hinge post, which is stouter
BURIED = 0.3         # how far posts carry on below the ground
COLOUR = (0.70, 0.72, 0.73)


def make(hinge, tip, wing_end, posts, height, ground):
    """(vertices, faces) for one gate.

    hinge is where the hinge post stands, tip where the open leaf ends, wing_end where the
    rail off the hinge ends, each as (x, y). posts is how many posts the rail has, the hinge
    post counted. height is the rail's height above the ground. ground(xs, ys) gives the
    ground's height.
    """
    hinge, tip, wing_end = (np.asarray(p, float) for p in (hinge, tip, wing_end))

    def at(point, above):
        return np.array([point[0], point[1], float(ground(point[:1], point[1:2])[0]) + above])

    pieces = []
    stand = [hinge + (wing_end - hinge) * t for t in np.linspace(0.0, 1.0, posts)]
    for n, point in enumerate(stand):
        pieces.append(tubes.strut(at(point, -BURIED), at(point, height + (0.15 if n == 0 else 0.0)), HINGE if n == 0 else POST, sides=6))
    for a, b in zip(stand[:-1], stand[1:]):
        pieces.append(tubes.strut(at(a, height - RAIL), at(b, height - RAIL), RAIL, sides=6))
    # The leaf hangs level from its hinge: a top tube, a tube that rises from the hinge's foot
    # to meet it at the tip, and an upright between the two.
    level = float(ground(hinge[:1], hinge[1:2])[0])
    top_a, top_b = np.array([*hinge, level + height - 0.1]), np.array([*tip, level + height - 0.1])
    low_a = np.array([*hinge, level + 0.35])
    pieces += [tubes.strut(top_a, top_b, RAIL, sides=6), tubes.strut(low_a, top_b, RAIL, sides=6)]
    middle = 0.45
    pieces.append(tubes.strut(low_a + middle * (top_b - low_a), top_a + middle * (top_b - top_a), 0.03, sides=6))
    return tubes.merge(pieces)
