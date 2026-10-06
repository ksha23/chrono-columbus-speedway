"""How stock Chrono::VSG turns a texture into what is on screen, and the inverse of it.

Measured on PyChrono 10.0.0 build 1187 with grey test cards (see the README): the renderer
takes texture values as linear light, lights them, and encodes the result as sRGB for the
display. A photograph is already sRGB, so shown as it is it comes out pale and flat. Flat ground
under the script's default light (intensity 1, 45 degrees up) receives 0.70 of its texture
value, of which 0.17 is ambient, so a cast shadow is 0.24 as bright as the sunlit ground.

for_renderer() prepares a photo so that sunlit flat ground appears on screen as it does in the
photo. The same applies to plain material colours: colour_for_renderer().
"""
import numpy as np

LIT = 0.70       # texture value to screen value, linear, flat ground in the default light
AMBIENT = 0.17   # the part of that which reaches shadowed ground too


def decode(srgb):
    """sRGB values in 0..1 to linear light."""
    srgb = np.asarray(srgb, dtype=np.float32)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def encode(linear):
    """Linear light in 0..1 to sRGB values."""
    linear = np.clip(np.asarray(linear, dtype=np.float32), 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


_TABLE = np.clip(decode(np.arange(256) / 255.0) / LIT * 255.0 + 0.5, 0, 255).astype(np.uint8)


def for_renderer(photo):
    """An sRGB uint8 picture as the texture that makes the renderer show that picture."""
    return _TABLE[photo]


def colour_for_renderer(rgb):
    """An sRGB colour in 0..1 as the material colour that shows as that colour when sunlit."""
    return [float(min(1.0, v)) for v in decode(np.asarray(rgb, dtype=np.float32)) / LIT]
