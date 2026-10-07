"""How worn each stretch of road paint is, read from the photo.

Paint is drawn as ribbons of one colour, as bright as the day it was laid. No painted line on
a track looks like that for long: tyres scrub it on the racing line, and the sun fades it
everywhere else. The photo knows which stretches have gone. A fresh yellow line stands far
above the concrete in yellowness and a worn one barely does, and the same goes for a white
line in brightness.

So each half metre of each stroke is given a level: how much of the paint's colour it keeps,
the rest being pavement showing through. Only a build that asks for the track as found
(--worn) uses it. The levels are few, since each is a mesh and a material of its own.

Two cautions that are built in. The drone exposed the concrete near white, and a white line
on concrete that is already at 235 cannot stand 40 levels above it however fresh it is: white
paint is judged against the room there was above the pavement, not against a fixed step. And
nothing the tracer found is taken away altogether: the lowest level still shows.
"""
import numpy as np
from scipy import ndimage

import markmodel
import photoprofile

LEVELS = (0.45, 0.6, 0.75, 0.9, 1.0)   # the share of its own colour a stretch of paint keeps
ALONG = 0.05        # metres between samples along a stretch
ACROSS = 0.025      # and across it
BESIDE = (0.3, 0.5)  # metres from the line's middle between which the pavement beside it is read
FRESH = 0.9         # the share of a colour's stretches that are less yellow than fresh paint
ROOM = 250.0        # the brightest a pixel of paint can be in the photo
LEAST_ROOM = 20.0   # grey levels of room above the pavement below which a white line cannot be judged
FADED = 0           # the level of a symbol's paint: the photo shows them faded to pale slate
PIECES = 120        # segments of a stroke read from the photo at a time


def _contrast(photo, raster, path, width, measure):
    """For each segment of a path, how far the line stands above the pavement beside it, and that pavement's level."""
    res = raster["res"]
    a, b = path[:-1], path[1:]
    length = np.hypot(*(b - a).T)
    along = np.maximum(np.ceil(length.max() / ALONG).astype(int), 1)
    t = (np.arange(along) + 0.5) / along
    offsets = np.arange(-BESIDE[1], BESIDE[1] + 1e-9, ACROSS)
    ahead = (b - a) / np.maximum(length, 1e-9)[:, None]
    normal = np.stack([-ahead[:, 1], ahead[:, 0]], axis=1)
    # Points as (segment, along, across).
    centre = a[:, None, :] + (b - a)[:, None, :] * t[None, :, None]
    x = centre[:, :, None, 0] + offsets[None, None, :] * normal[:, None, None, 0]
    y = centre[:, :, None, 1] + offsets[None, None, :] * normal[:, None, None, 1]
    c0, c1 = max(int((x.min() - raster["x0"]) / res) - 2, 0), min(int((x.max() - raster["x0"]) / res) + 3, photo.shape[1])
    r0, r1 = max(int((raster["y1"] - y.max()) / res) - 2, 0), min(int((raster["y1"] - y.min()) / res) + 3, photo.shape[0])
    window = measure(np.asarray(photo[r0:r1, c0:c1, :3], dtype=np.float32))
    values = ndimage.map_coordinates(window, [(raster["y1"] - y) / res - 0.5 - r0, (x - raster["x0"]) / res - 0.5 - c0], order=1, mode="nearest")
    profile = values.mean(1)                                                     # (segment, across)
    on = np.abs(offsets) <= width / 2 + ACROSS
    beside = (np.abs(offsets) >= BESIDE[0]) & (np.abs(offsets) <= BESIDE[1])
    base = np.median(profile[:, beside], axis=1)
    return profile[:, on].max(1) - base, base


def measure(strokes, photo, raster, unseen, log=print):
    """Give every stroke "wear": one index into LEVELS for each segment of the path its ribbon follows.

    photo is the ground photo with its shadows relit and the paint still in it. unseen(points)
    says which points lie under a crown or in a hole of the scan, where the photo has no paint
    to read: such a stretch takes the level its stroke usually has.
    """
    read = {}        # index of a stroke -> (share of full strength per segment, which segments were seen)
    fresh = {}
    for colour, gauge in (("yellow", photoprofile.yellowness), ("white", photoprofile.brightness)):
        stands = []
        for n, stroke in enumerate(strokes):
            if stroke["colour"] != colour or stroke.get("symbol") or len(stroke["points"]) < 2:
                continue
            path = markmodel._densify(np.asarray(stroke["points"], float))
            if len(path) < 2:
                continue
            # A long line in pieces: the photo is read over the box round each piece.
            parts = [_contrast(photo, raster, path[k:k + PIECES + 1], stroke["width"], gauge) for k in range(0, len(path) - 1, PIECES)]
            above, base = (np.concatenate(side) for side in zip(*parts))
            if colour == "white":
                # Against the room there was: on concrete near white a fresh line has little.
                above = above / np.maximum(ROOM - base, LEAST_ROOM)
            seen = ~np.asarray(unseen((path[:-1] + path[1:]) / 2), bool)
            if stroke.get("inferred"):
                seen[:] = False              # drawn where a line had to be: the photo did not show it
            read[n] = (above, seen)
            stands.append(above[seen])
        if stands and sum(len(s) for s in stands):
            fresh[colour] = float(np.percentile(np.concatenate(stands), 100 * FRESH))
    levels = np.array(LEVELS)
    usual = {}
    for n, (above, seen) in read.items():
        colour = strokes[n]["colour"]
        share = np.clip(above / max(fresh.get(colour, 1.0), 1e-6), 0.0, 1.0)
        level = np.abs(share[:, None] - levels[None, :]).argmin(1)
        if seen.sum() >= 3:
            level[seen] = ndimage.median_filter(level[seen], 3, mode="nearest")     # wear changes over metres, not from one half metre to the next
        if seen.any():
            level[~seen] = int(np.median(level[seen]))
        read[n] = (level, seen)
        usual.setdefault(colour, []).append(level[seen])
    counts = {}
    for n, stroke in enumerate(strokes):
        if stroke.get("symbol"):
            segments = max(len(markmodel._densify(np.asarray(stroke["points"], float))) - 1, 0)
            stroke["wear"] = [FADED] * segments
        elif n in read:
            level, seen = read[n]
            if not seen.any():
                known = np.concatenate(usual.get(stroke["colour"], [np.array([len(LEVELS) - 1])]))
                level[:] = int(np.median(known)) if len(known) else len(LEVELS) - 1
            stroke["wear"] = [int(v) for v in level]
        else:
            continue
        for v in stroke["wear"]:
            counts[v] = counts.get(v, 0) + 1
    total = max(sum(counts.values()), 1)
    log("  paint wear: " + ", ".join(f"{100 * counts.get(k, 0) / total:.0f}% at {LEVELS[k]:g}" for k in range(len(LEVELS)))
        + f" of full colour, fresh yellow stands {fresh.get('yellow', 0):.0f} above the concrete")


def colour(name, level, pavement):
    """The colour of a paint at a level of wear, as seen: its own with the pavement's showing through."""
    return tuple(LEVELS[level] * p + (1.0 - LEVELS[level]) * g for p, g in zip(markmodel.COLOURS[name], pavement))
