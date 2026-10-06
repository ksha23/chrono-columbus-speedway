"""Kinks in a line that should run on: where they are, and a smooth curve across each.

A kink is a sharp turn one way answered within a few metres by a sharp turn the other way: a
notch, a bump or a step. A corner turns one way only and is not one. This is what the road
outline (outline.py) and the edge lines (edgelines.py) both look for.
"""
import numpy as np

STEP = 0.25        # metres between the points of a line handed to these functions
CHORD = 1.5        # metres: headings are taken over chords this long
TURN = 8.0         # degrees between neighbouring chords that make a sharp turn
WITHIN = 6.0       # metres within which a sharp turn must be answered by one the other way
MARGIN = 1.5       # metres of line replaced beyond the first and the last sharp turn
LEAD = 3.0         # metres of line beyond that from which the line's direction is taken
LONGEST = 16.0     # metres: nothing longer is a kink


def pairs(path):
    """[first, last, sharpness] of each turn-and-answer: indices into path with no margin
    added, and the lesser of its two opposite turns in degrees."""
    n = int(round(CHORD / STEP))
    if len(path) < 4 * n:
        return []
    chord = path[n:] - path[:-n]
    heading = np.unwrap(np.arctan2(chord[:, 1], chord[:, 0]))
    turn = np.degrees(heading[n:] - heading[:-n])             # the turn at path[n + i]
    sign = np.where(turn > TURN, 1, np.where(turn < -TURN, -1, 0))
    change = np.nonzero(np.diff(np.concatenate([[0], sign, [0]])) != 0)[0]
    runs = [(first, last) for first, last in zip(change[:-1], change[1:]) if sign[first] != 0]      # last is one past the end
    reach = int(round(WITHIN / STEP))
    # Each run of sharp turns gives no more of itself than lies within reach of the run that
    # answers it: the rest of a long corner is corner.
    found = []
    for (s0, e0), (s1, e1) in zip(runs[:-1], runs[1:]):
        if sign[s0] != sign[s1] and s1 - e0 <= reach:
            first, last = max(s0, s1 - reach) + n, min(e1 - 1, e0 + reach) + n
            sharp = float(min(np.abs(turn[s0:e0]).max(), np.abs(turn[s1:e1]).max()))
            if found and first <= found[-1][1]:
                found[-1][1] = max(found[-1][1], last)      # two kinks that share a run are one
                found[-1][2] = max(found[-1][2], sharp)
            else:
                found.append([first, last, sharp])
    return found


def spans(path, net):
    """(first, last) indices into path of each stretch to replace, ends included.

    net is how many degrees the line may have turned from one side of a kink to the other. A
    small figure keeps to notches in an edge that runs straight on, a large one also takes a
    notch out of a corner.
    """
    lead, margin = int(round(LEAD / STEP)), int(round(MARGIN / STEP))
    out = []
    for first, last, _ in pairs(path):
        a, b = first - margin, last + margin
        if a - lead < 0 or b + lead >= len(path) or (b - a) * STEP > LONGEST:
            continue
        before, after = path[a] - path[a - lead], path[b + lead] - path[b]
        turned = np.degrees(np.arctan2(before[0] * after[1] - before[1] * after[0], before @ after))
        if abs(turned) <= net:
            out.append((a, b))
    return out


def bridge(path, a, b, lead=LEAD):
    """A curve from path[a] to path[b] that leaves and arrives along the line's own direction,
    taken over lead metres of the line beyond each end."""
    lead = int(round(lead / STEP))
    p0, p1 = path[a], path[b]
    reach = np.hypot(*(p1 - p0))
    m0 = path[a] - path[a - lead]
    m1 = path[b + lead] - path[b]
    m0, m1 = reach * m0 / np.hypot(*m0), reach * m1 / np.hypot(*m1)
    t = np.linspace(0.0, 1.0, max(int(reach / STEP), 2) + 1)[:, None]
    return (2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * m0 + (-2 * t**3 + 3 * t**2) * p1 + (t**3 - t**2) * m1


def apart(one, other):
    """The furthest any point of either line lies from the other line, metres."""
    d = np.hypot(*(one[:, None, :] - other[None, :, :]).transpose(2, 0, 1))
    return float(max(d.min(1).max(), d.min(0).max()))
