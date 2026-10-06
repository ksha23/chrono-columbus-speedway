"""The USGS rasters downloaded for the scene rectangle: lidar elevation and aerial imagery.

Both are in the scene frame already (UTM 16N, north up), at layout.REFERENCE_RES per pixel.
"""
import os

import numpy as np
from PIL import Image
from scipy import ndimage

import layout

Image.MAX_IMAGE_PIXELS = None


class Reference:
    def __init__(self, ref_dir):
        self.dem = np.asarray(Image.open(os.path.join(ref_dir, "dem_scene.tif")), dtype=np.float64)
        self.res = layout.REFERENCE_RES
        expected = (int(round((layout.SCENE_Y1 - layout.SCENE_Y0) / self.res)), int(round((layout.SCENE_X1 - layout.SCENE_X0) / self.res)))
        if self.dem.shape != expected:
            raise SystemExit(f"dem_scene.tif is {self.dem.shape}, expected {expected} for the scene rectangle")
        self._naip = os.path.join(ref_dir, "naip_scene.png")

    def pixel(self, x, y):
        """Fractional (row, col) of scene point (x, y). Pixel centres sit half a pixel in."""
        return (layout.SCENE_Y1 - np.asarray(y, float)) / self.res - 0.5, (np.asarray(x, float) - layout.SCENE_X0) / self.res - 0.5

    def elevation(self, x, y, order=1):
        """Bare-earth elevation at scene points, metres."""
        r, c = self.pixel(x, y)
        shape = np.shape(r)
        z = ndimage.map_coordinates(self.dem, [np.atleast_1d(r), np.atleast_1d(c)], order=order, mode="nearest")
        return z.reshape(shape)

    def aerial(self):
        """The aerial picture as an HxWx3 uint8 array on the same grid as the elevation."""
        return np.asarray(Image.open(self._naip).convert("RGB"))
