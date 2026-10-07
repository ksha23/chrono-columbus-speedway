"""Cut the finished ground photo into texture tiles.

Road and land are separate meshes that meet along the pavement's edge, and each gets its own
picture of a tile. In the road's picture the pavement is carried a little way out past its edge,
and in the land's picture the grass is carried a little way in. The mesh edge is a straight cut
through a boundary that in the photo is soft and wanders by a few centimetres, so without this
each mesh would show a fringe of the other's colour along it.
"""
import os

import numpy as np
from PIL import Image
from scipy import ndimage

import ground
import layout
import pavement
import renderer

Image.MAX_IMAGE_PIXELS = None
JPEG_QUALITY = 90
SOLID = 0.10    # metres inside its own side of the edge before a pixel is trusted as pure


def save_texture(picture, path, as_photographed=False):
    """Save an sRGB picture as a JPEG texture, converted to what the renderer expects.

    The conversion comes after any resizing, so averaging is done on the photo's own values.
    as_photographed leaves the conversion out: the picture as the drone took it, for a
    renderer that takes a texture as an ordinary picture and has no ceiling to stay under.
    """
    # Full-resolution colour: a paint line is two pixels wide, and halved colour would fade it.
    Image.fromarray(picture if as_photographed else renderer.for_renderer(picture)).save(path, quality=JPEG_QUALITY, subsampling=0, optimize=True)


def spread(window, source):
    """Every pixel replaced by the nearest pixel of source. Pixels in source keep their own colour."""
    _, (rows, cols) = ndimage.distance_transform_edt(~source, return_indices=True)
    return window[rows, cols]


def write_tiles(photo, raster, level, directory, edge, as_photographed=False):
    """Write the JPEGs of one detail level. photo is the full raster at raster["res"].

    edge is the pavement.Edge. Returns (total bytes, names of the tiles that got a road picture).
    as_photographed is save_texture's.
    """
    os.makedirs(directory, exist_ok=True)
    res = layout.LEVELS[level]
    factor = res / raster["res"]
    if abs(factor - round(factor)) > 1e-9 or factor < 1:
        raise SystemExit(f"level {level} ({res} m) is not a whole multiple of the raster's {raster['res']} m")
    factor = int(round(factor))
    size = layout.tile_pixels(res)
    total, with_road = 0, []
    for i, j in raster["tiles"]:
        x0, y0, x1, y1 = layout.tile_bounds(i, j)
        c0 = int(round((x0 - layout.GUTTER - raster["x0"]) / raster["res"]))
        r0 = int(round((raster["y1"] - (y1 + layout.GUTTER)) / raster["res"]))
        n = size * factor
        if c0 < 0 or r0 < 0 or c0 + n > raster["width"] or r0 + n > raster["height"]:
            raise SystemExit(f"tile {(i, j)} with its gutter reaches outside the raster")
        window = np.asarray(photo[r0:r0 + n, c0:c0 + n])
        # Distance to the pavement edge at every pixel of the window.
        xs = raster["x0"] + (c0 + np.arange(n) + 0.5) * raster["res"]
        ys = raster["y1"] - (r0 + np.arange(n) + 0.5) * raster["res"]
        step = max(1, int(round(pavement.RES / raster["res"])))
        coarse = edge.at(*np.meshgrid(xs[::step], ys[::step]))
        distance = np.repeat(np.repeat(coarse, step, axis=0), step, axis=1)[:n, :n]
        versions = {"": distance < -SOLID}
        if (distance > SOLID).any():
            versions["_road"] = distance > SOLID
            with_road.append(ground.tile_name(i, j))
        for suffix, source in versions.items():
            picture = spread(window, source) if not source.all() else window
            img = Image.fromarray(picture)
            if factor > 1:
                img = img.resize((size, size), Image.BOX)
            path = os.path.join(directory, ground.tile_name(i, j) + suffix + ".jpg")
            save_texture(np.asarray(img), path, as_photographed)
            total += os.path.getsize(path)
    return total, with_road
