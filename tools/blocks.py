"""Find the small man-made things that stand about the site: tanks, a shed, a deck.

What can be said of each from the scan is where it stands, its footprint, its height and its
colour. What kind of thing it is cannot be, and structures.py takes that from a list read by
hand. One the list does not name is given back as a block of the size and colour found here.

Telling them from brush is the hard part, since a pile of dead branches is as tall and as grey.
Man-made things are colourless and even on top, and brush is neither.
"""
import numpy as np
from scipy import ndimage

import objects

LOW, HIGH = 0.7, 3.5      # metres tall
SMALLEST = 2.5            # square metres
LONGEST = 3.6             # length over width: anything thinner is a rail or a fence
# Colourless and even on top, in one of two degrees: quite colourless and fairly even, or
# fairly colourless and very even. (saturation at most, roughness at most)
PLAIN = ((0.07, 0.70), (0.12, 0.08))


def find(height, small_photo, buildings, taken, cell, x0, y1):
    """Return a list of blocks: dicts with x, y, yaw, length, width, height and colour.

    taken is a list of things already accounted for (vehicles), as dicts with x and y: nothing
    within 3.5 m of one is reported again.
    """
    h = np.nan_to_num(height, nan=0.0)
    rgb = small_photo.astype(np.float32) / 255
    top, low = rgb.max(-1), rgb.min(-1)
    sat = (top - low) / np.maximum(top, 1e-3)
    rough = objects.roughness(h)
    solid = (h > 0.6) & ~ndimage.binary_dilation(buildings, iterations=int(round(1.0 / cell))) & (sat < 0.2)
    solid = ndimage.binary_opening(solid, iterations=1)
    labels, _ = ndimage.label(solid)
    blocks = []
    for n, sl in enumerate(ndimage.find_objects(labels), start=1):
        rows, cols = np.nonzero(labels[sl] == n)
        if len(rows) * cell * cell < SMALLEST:
            continue
        tall = float(np.percentile(h[sl][rows, cols], 90))
        s, r = float(np.median(sat[sl][rows, cols])), float(np.median(rough[sl][rows, cols]))
        # Nothing nearly black either: that is the lip of a hole in the scan, not a thing.
        if not (LOW <= tall <= HIGH) or not any(s <= a and r <= b for a, b in PLAIN) or np.median(top[sl][rows, cols]) < 0.2:
            continue
        pts = np.stack([x0 + (sl[1].start + cols + 0.5) * cell, y1 - (sl[0].start + rows + 0.5) * cell], 1)
        centre = pts.mean(0)
        if any(np.hypot(centre[0] - t["x"], centre[1] - t["y"]) < 3.5 for t in taken):
            continue
        _, evecs = np.linalg.eigh(np.cov((pts - centre).T) + 1e-9 * np.eye(2))
        axis, side = evecs[:, 1], evecs[:, 0]
        a, b = (pts - centre) @ axis, (pts - centre) @ side
        a0, a1 = np.percentile(a, [3, 97])
        b0, b1 = np.percentile(b, [3, 97])
        length, width = float(a1 - a0 + cell), float(b1 - b0 + cell)
        if length / max(width, 1e-6) > LONGEST or len(rows) * cell * cell < 0.6 * length * width:
            continue
        centre = centre + axis * (a0 + a1) / 2 + side * (b0 + b1) / 2
        colour = np.median(rgb[sl][rows, cols], axis=0)
        blocks.append({"x": float(centre[0]), "y": float(centre[1]), "yaw": float(np.arctan2(axis[1], axis[0])),
                       "length": round(length, 2), "width": round(width, 2), "height": round(tall, 2),
                       "colour": [round(float(v), 3) for v in colour]})
    return sorted(blocks, key=lambda b: (b["x"], b["y"]))


def write_unit(path):
    """A block one metre each way, standing on z = 0 and centred on the origin, carried on
    0.3 m below the ground for slopes. A placement scales it to size."""
    z0 = -0.3
    v = [(-0.5, -0.5, z0), (0.5, -0.5, z0), (0.5, 0.5, z0), (-0.5, 0.5, z0), (-0.5, -0.5, 1.0), (0.5, -0.5, 1.0), (0.5, 0.5, 1.0), (-0.5, 0.5, 1.0)]
    faces = [((4, 5, 6), (4, 6, 7), (0, 0, 1)), ((0, 1, 5), (0, 5, 4), (0, -1, 0)), ((1, 2, 6), (1, 6, 5), (1, 0, 0)),
             ((2, 3, 7), (2, 7, 6), (0, 1, 0)), ((3, 0, 4), (3, 4, 7), (-1, 0, 0))]
    with open(path, "w") as out:
        count = 0
        for a, b, normal in faces:
            for tri in (a, b):
                for p in tri:
                    out.write("v %.3f %.3f %.3f\nvn %d %d %d\n" % (*v[p], *normal))
                count += 1
        for t in range(count):
            out.write(f"f {3 * t + 1}//{3 * t + 1} {3 * t + 2}//{3 * t + 2} {3 * t + 3}//{3 * t + 3}\n")
