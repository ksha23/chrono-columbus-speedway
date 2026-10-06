"""Find the vehicles parked on the pavement, so they can be painted out and put back as models.

A parked car is the one thing on pavement that is a metre and a half tall, the size of a car,
and no taller. The scan's height above ground finds the roof. The body around the roof gives
the footprint, and so where the car is and which way it points. The photo gives its colour.
"""
import numpy as np
from scipy import ndimage

# Metres above ground that only a vehicle's roof reaches, of the things on pavement. Guard rails,
# bleachers and brush piles stop at 1.2 m. It also has to clear the saddle the scan leaves
# between two cars parked side by side, which is about 1.2 m.
ROOF = 1.35
BODY = 0.45           # metres above ground that counts as part of the vehicle
TALLEST = 2.5         # anything reaching higher is a tree or a building
LENGTH = (3.4, 6.6)   # metres: footprints outside this are not a car
WIDTH = (1.4, 3.1)     # the scan smears a car's sides outward by a few tenths of a metre
ON_PAVEMENT = 0.4     # share of the footprint that must be pavement
MARGIN = 0.35         # metres added round a footprint when it is painted out


def find(height, small_photo, paved_cells, buildings, cell, x0, y1):
    """Return a list of vehicles: dicts with x, y, yaw, length, width, height and colour.

    Everything is on the grid of size cell: height above ground, the photo averaged onto it,
    the pavement and the buildings. yaw is the long axis as an angle from east, in radians. The
    front and the back of a car look alike from here, so which end leads is arbitrary.
    """
    h = np.nan_to_num(height, nan=0.0)
    near_pavement = ndimage.binary_dilation(paved_cells, iterations=int(round(1.0 / cell)))
    body = (h > BODY) & near_pavement & ~ndimage.binary_dilation(buildings, iterations=int(round(1.0 / cell)))
    roofs = ndimage.binary_opening(body & (h > ROOF), iterations=1)
    labels, count = ndimage.label(roofs)
    if count == 0:
        return []
    # Each body cell goes to the nearest roof, if one is within a car's half length of it.
    distance, (near_r, near_c) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    owner = np.where(body & (distance * cell < 2.6), labels[near_r, near_c], 0)

    cars, rows_of_cars = [], []
    rgb = small_photo.astype(np.float32) / 255
    for n, sl in enumerate(ndimage.find_objects(owner), start=1):
        if sl is None:
            continue
        rows, cols = np.nonzero(owner[sl] == n)
        tops = h[sl][rows, cols]
        area = len(rows) * cell * cell
        if area < 4.0 or tops.max() > TALLEST or paved_cells[sl][rows, cols].mean() < ON_PAVEMENT:
            continue
        colour = rgb[sl][rows, cols]
        if _is_plant(np.median(colour, axis=0)):
            continue
        pts = np.stack([x0 + (sl[1].start + cols + 0.5) * cell, y1 - (sl[0].start + rows + 0.5) * cell], 1)
        _, evecs = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
        car = _car(pts, tops, colour, evecs[:, 1], cell)
        if _car_sized(car):
            cars.append(car)
        elif 14.0 <= area <= 50.0:
            rows_of_cars.append((pts, tops, colour))

    # Cars parked side by side come out of the scan as one block. Its heading is not its own
    # long axis, so it is taken from the nearest car found on its own, which stands in the
    # same row of stalls, and the block is cut across into as many cars as fit.
    singles = list(cars)
    for pts, tops, colour in rows_of_cars:
        centre = pts.mean(0)
        near = [c for c in singles if np.hypot(c["x"] - centre[0], c["y"] - centre[1]) < 12.0]
        if not near:
            continue
        guide = min(near, key=lambda c: np.hypot(c["x"] - centre[0], c["y"] - centre[1]))
        axis = np.array([np.cos(guide["yaw"]), np.sin(guide["yaw"])])
        across = (pts - centre) @ np.array([-axis[1], axis[0]])
        b0, b1 = np.percentile(across, [2, 98])
        count = int(round((b1 - b0 + cell) / 2.6))
        if not (2 <= count <= 4):
            continue
        edges = np.linspace(b0, b1 + 1e-6, count + 1)
        parts = [_car(pts[m], tops[m], colour[m], axis, cell) for m in ((across >= edges[k]) & (across < edges[k + 1]) for k in range(count)) if m.sum() * cell * cell >= 4.0]
        if len(parts) == count and all(_car_sized(c) for c in parts):
            cars += parts
    return sorted(cars, key=lambda c: (c["x"], c["y"]))


def _car_sized(car):
    """A car's length and width, tall enough to be one, and solid: a brush or rock pile of the
    same size leaves gaps in its outline and is lower."""
    return (LENGTH[0] <= car["length"] <= LENGTH[1] and WIDTH[0] <= car["width"] <= WIDTH[1]
            and car["height"] >= 1.4 and car["fill"] >= 0.72)


def _car(pts, tops, colour, axis, cell):
    """One vehicle from the cells it covers: where it is, how big, which way, what colour."""
    side = np.array([-axis[1], axis[0]])
    centre = pts.mean(0)
    a, b = (pts - centre) @ axis, (pts - centre) @ side
    a0, a1 = np.percentile(a, [2, 98])
    b0, b1 = np.percentile(b, [2, 98])
    centre = centre + axis * (a0 + a1) / 2 + side * (b0 + b1) / 2
    # Which end is the front: the lower one. A bonnet is lower than a boot or a tailgate.
    span = a1 - a0
    ahead, behind = tops[a > a1 - 0.3 * span], tops[a < a0 + 0.3 * span]
    if len(ahead) and len(behind) and ahead.mean() - behind.mean() > 0.15:
        axis = -axis
    paint = _paint(colour, (np.abs(a - (a0 + a1) / 2) < 0.42 * (a1 - a0)) & (np.abs(b - (b0 + b1) / 2) < 0.38 * (b1 - b0)))
    return {"x": float(centre[0]), "y": float(centre[1]), "yaw": float(np.arctan2(axis[1], axis[0])),
            "length": round(float(a1 - a0 + cell), 2), "width": round(float(b1 - b0 + cell), 2),
            "height": round(float(np.percentile(tops, 95)), 2), "colour": [round(float(v), 3) for v in paint],
            "fill": round(float(len(pts) * cell * cell / max((a1 - a0 + cell) * (b1 - b0 + cell), 1e-6)), 2)}


def _paint(colour, inner):
    """A vehicle's paint colour from the cells it covers, as seen from above.

    colour is the photo at each cell and inner marks the cells away from the footprint's rim,
    which is smeared with pavement and with whatever is parked alongside.
    """
    bright = colour.max(1)
    sat = (bright - colour.min(1)) / np.maximum(bright, 1e-3)
    # A strong colour anywhere is the paint: a red pickup is mostly dark bed and glass from
    # above, with its red on the bonnet and along the sides. Pavement is never that colour.
    vivid = (sat > 0.45) & (bright > 0.45)
    if vivid.mean() > 0.10:
        # The more strongly coloured half of it: the rest is mixed with whatever is beside it.
        return colour[vivid & (sat >= np.median(sat[vivid]))].mean(0)
    if inner.sum() >= 8:
        colour, bright = colour[inner], bright[inner]
    if np.median(bright) < 0.35:
        # A dark car. Its brighter parts are glints and the sky in its glass, not its paint.
        paint = np.median(colour[bright <= np.percentile(bright, 60)], axis=0)
    else:
        # A light car. The darker part of what shows is glass.
        lo, hi = np.percentile(bright, [60, 95])
        paint = colour[(bright >= lo) & (bright <= hi)].mean(0)
    # Silver and white paint photographs blue under an open sky. Take half of that back out.
    return paint + 0.5 * (paint.mean() - paint) if (paint.max() - paint.min()) / max(paint.max(), 1e-3) < 0.25 else paint


def _is_plant(rgb):
    """Green or straw: a shrub that leans over the pavement, not a car."""
    r, g, b = (float(v) for v in rgb)
    top = max(r, g, b)
    return top > 0.05 and (top - min(r, g, b)) / top > 0.14 and g >= r and b < 0.9 * g


def footprint(cars, raster, shape, photo, sun_list):
    """Full-size mask of what each vehicle left in the photo: itself and its shadow.

    The shadow is the vehicle's outline carried away from the sun by its height over the tangent
    of the sun's elevation. The site was flown at two times of day, so the outline is carried
    both ways in turn and the way that lands on the darker ground is kept.
    """
    res = raster["res"]
    mask = np.zeros(shape, bool)
    for car in cars:
        axis = np.array([np.cos(car["yaw"]), np.sin(car["yaw"])])
        side = np.array([-axis[1], axis[0]])
        half = np.array([car["length"] / 2 + MARGIN, car["width"] / 2 + MARGIN])
        reach = float(np.hypot(*half)) + 6.0
        c0, c1 = max(int((car["x"] - reach - raster["x0"]) / res), 0), min(int((car["x"] + reach - raster["x0"]) / res) + 1, shape[1])
        r0, r1 = max(int((raster["y1"] - car["y"] - reach) / res), 0), min(int((raster["y1"] - car["y"] + reach) / res) + 1, shape[0])
        yy, xx = np.mgrid[r0:r1, c0:c1]
        px = raster["x0"] + (xx + 0.5) * res - car["x"]
        py = raster["y1"] - (yy + 0.5) * res - car["y"]

        def outline(dx=0.0, dy=0.0):
            return (np.abs((px - dx) * axis[0] + (py - dy) * axis[1]) < half[0]) & (np.abs((px - dx) * side[0] + (py - dy) * side[1]) < half[1])

        own = outline()
        window = np.asarray(photo[r0:r1, c0:c1], dtype=np.float32).max(-1)
        best, darkest = own, np.inf
        for azimuth, elevation in sun_list:
            throw = car["height"] / np.tan(np.radians(elevation)) + 0.3
            away = np.radians(azimuth + 180.0)
            swept = own.copy()
            for t in np.linspace(0.0, throw, 8)[1:]:
                swept |= outline(t * np.sin(away), t * np.cos(away))
            shade = swept & ~own
            tone = float(window[shade].mean()) if shade.any() else np.inf
            if tone < darkest:
                best, darkest = swept, tone
        mask[r0:r1, c0:c1] |= best
    return mask
