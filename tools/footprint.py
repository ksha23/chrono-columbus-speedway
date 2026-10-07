"""The plan of a building as its roof shows it: a rectangle with square steps in its long sides.

A roof seen from above is the one part of a building the scan gets right, and its edge is the
building's plan. Three of the four on this site are plain rectangles. The fourth has a wing
that stands three metres out from one side, and drawn as its bounding rectangle it grows a
corner it does not have.

The plan is kept in the building's own frame, u along the ridge and w across it. Walking along
u, the roof's edge on either side is a level that now and then steps to another: so each side
is fitted as a few level runs, and the plan is the strips between them. A step has to be
worth having. The mask's edge is ragged by a cell or two and the frame is never quite square
to it, and neither of those is a wing.
"""
import numpy as np

BIN = 0.25         # metres along the ridge that one reading of the edge covers
STEP = 0.8         # metres an edge must move by to count as a step
RUN = 1.5          # metres a level must last
BREAK = 6.0        # what a step costs, in bins of one metre's misfit
FINE = 0.05        # metres between the points an edge is looked for at, across the ridge
REACH = 2.0        # metres beyond the fitted rectangle that a wing is looked for


def _levels(values):
    """A row of readings as the fewest level runs that fit it: a list of (first, last + 1, level).

    Least absolute misfit plus BREAK for every run, by dynamic programming. The median of a
    run is its level, so a few wild readings at a ragged corner do not move it.
    """
    n = len(values)
    least = int(round(RUN / BIN))
    cost = np.full(n + 1, np.inf)
    cost[0] = 0.0
    back = np.zeros(n + 1, int)
    for end in range(least, n + 1):
        for start in range(0, end - least + 1):
            if not np.isfinite(cost[start]):
                continue
            part = values[start:end]
            total = cost[start] + np.abs(part - np.median(part)).sum() + BREAK
            if total < cost[end]:
                cost[end], back[end] = total, start
    runs, end = [], n
    while end > 0:
        start = back[end]
        runs.append([start, end, float(np.median(values[start:end]))])
        end = start
    runs.reverse()
    # Runs whose levels differ by less than a step are one run: a frame a little off square
    # shows as a slow drift, which the fit above cuts into a staircase.
    merged = [runs[0]]
    for run in runs[1:]:
        last = merged[-1]
        if abs(run[2] - last[2]) < STEP:
            last[2] = float(np.median(values[last[0]:run[1]]))
            last[1] = run[1]
        else:
            merged.append(run)
    return merged


def _same(runs):
    """Level runs with levels that are within half a step of each other given one level between them.

    The wall on either side of a wing is one wall, and read twice it comes out a few
    centimetres apart.
    """
    order = sorted(runs, key=lambda run: run[2])
    groups = [[order[0]]]
    for run in order[1:]:
        if run[2] - groups[-1][-1][2] < STEP / 2:
            groups[-1].append(run)
        else:
            groups.append([run])
    for group in groups:
        level = sum(run[2] * (run[1] - run[0]) for run in group) / sum(run[1] - run[0] for run in group)
        for run in group:
            run[2] = level
    return runs


def strips(covered, half_length, half_width):
    """The plan as strips across the ridge: a list of (u0, u1, w_low, w_high), in order of u.

    covered(u, w) says for arrays of frame coordinates whether the roof covers each point.
    half_length and half_width are those of the rectangle fitted round the roof.
    """
    count = max(int(round(2 * half_length / BIN)), 2)
    edges = np.linspace(-half_length, half_length, count + 1)
    across = np.arange(-half_width - REACH, half_width + REACH, FINE)
    low, high = np.full(count, np.nan), np.full(count, np.nan)
    for n in range(count):
        here = across[covered(np.full(len(across), (edges[n] + edges[n + 1]) / 2), across)]
        if len(here) * FINE >= 1.0:
            low[n], high[n] = here[0] - FINE / 2, here[-1] + FINE / 2
    known = np.isfinite(low)
    low = np.interp(np.arange(count), np.nonzero(known)[0], low[known])
    high = np.interp(np.arange(count), np.nonzero(known)[0], high[known])
    runs = {"low": {run[0]: run[2] for run in _same(_levels(low))}, "high": {run[0]: run[2] for run in _same(_levels(high))}}
    cuts = sorted(set(runs["low"]) | set(runs["high"]) | {count})
    out, level = [], {}
    for first, last in zip(cuts[:-1], cuts[1:]):
        for side in ("low", "high"):
            level[side] = runs[side].get(first, level.get(side))
        out.append((float(edges[first]), float(edges[last]), level["low"], level["high"]))
    return out


def polygon(plan):
    """The outline of a plan, anticlockwise: an array of (u, w) corners."""
    lower, upper = [], []
    for u0, u1, low, high in plan:
        lower += [(u0, low), (u1, low)]
        upper += [(u0, high), (u1, high)]
    ring = np.array(lower + upper[::-1])
    # Only the corners: a point part of the way along a straight edge is not one.
    apart = np.hypot(*(ring - np.roll(ring, 1, axis=0)).T) > 1e-6
    ring = ring[apart]
    before, after = ring - np.roll(ring, 1, axis=0), np.roll(ring, -1, axis=0) - ring
    return ring[np.abs(before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]) > 1e-9]


def inset(ring, by):
    """An anticlockwise outline of square corners moved inward by a distance: the walls under a roof's overhang."""
    ahead = np.roll(ring, -1, axis=0) - ring
    ahead /= np.hypot(*ahead.T)[:, None]
    inward = np.stack([-ahead[:, 1], ahead[:, 0]], axis=1)       # to the left of the way round
    return ring + by * (inward + np.roll(inward, 1, axis=0))


def covers(plan, along, across, margin=0.0):
    """Which points lie in a plan, or within margin of it."""
    inside = np.zeros(np.shape(along), bool)
    for u0, u1, low, high in plan:
        inside |= (along >= u0 - margin) & (along <= u1 + margin) & (across >= low - margin) & (across <= high + margin)
    return inside
