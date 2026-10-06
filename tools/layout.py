"""Where the scene sits and how it is cut up. Every other tool takes its numbers from here.

The scene frame is metres east (x) and north (y) of ORIGIN in UTM zone 16N, with z the real
elevation above sea level (NAVD88) that the USGS lidar reports. Nothing is recentred in height,
so the track is near z = 295 m.
"""
UTM_ZONE = 16
ORIGIN_E, ORIGIN_N = 329450.0, 4794560.0

# The whole scene: the facility with fields and woods around it, so the view has a horizon.
SCENE_X0, SCENE_X1 = -640.0, 640.0
SCENE_Y0, SCENE_Y1 = -512.0, 512.0

# The reference rasters downloaded for that rectangle (see reference/), and their post spacing.
REFERENCE_RES = 0.5

# Where the demo car starts: on the long south-east straight, facing north-east up it toward the
# loop. x, y in metres and yaw in radians, found from the pavement in the photo.
START = {"x": -49.95, "y": -79.23, "yaw": 0.72}

# Ground mesh: fine cells where the scan has coverage, coarse cells beyond.
FINE, COARSE = 1.0, 8.0

# Photo texture: square tiles over the part the scan covers, anchored at the scene origin. A
# tile is a whole number of coarse ground cells, so the mesh changes resolution on tile edges.
# Around the picture of its own ground a tile image carries a gutter that repeats its
# neighbours, so filtering at a tile edge blends into the right colours instead of wrapping.
TILE = 96.0
GUTTER = 0.4

# What one texture pixel covers at each published detail level, metres.
LEVELS = {"low": 0.1, "standard": 0.05, "full": 0.025}


def tile_pixels(res):
    """Width of a tile image in pixels, gutter included."""
    return int(round((TILE + 2 * GUTTER) / res))


def tile_bounds(i, j):
    """(x0, y0, x1, y1) of tile (i, j), gutter not included."""
    return i * TILE, j * TILE, (i + 1) * TILE, (j + 1) * TILE


def tile_uv(i, j, x, y):
    """Texture coordinates of scene point (x, y) in tile (i, j). v runs up, as in an OBJ file."""
    x0, y0, _, _ = tile_bounds(i, j)
    span = TILE + 2 * GUTTER
    return (x - x0 + GUTTER) / span, (y - y0 + GUTTER) / span
