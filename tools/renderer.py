"""How stock Chrono::VSG turns a texture into what is on screen, and the inverse of it.

Measured on PyChrono 10.0.0 build 1187 with grey test cards (see the README): the renderer
takes texture values as linear light, lights them, and encodes the result as sRGB for the
display. A photograph is already sRGB, so shown as it is it comes out pale and flat. Flat ground
under the script's light (45 degrees up, intensity 1, which is the most it accepts) receives
0.70 of its texture value, of which 0.17 is ambient, so a cast shadow is 0.24 as bright as the
sunlit ground.

That 0.70 is also a ceiling: nothing lying on the ground can show brighter than sRGB 218. The
drone exposed the concrete at about 200, with paint above it at 240 to 250. Stored as they
are, paint and concrete would both hit the ceiling and a white line would vanish into the
pavement, while a yellow one would lose its red and turn pale. So the scene is shown a little
darker than the photo, by EXPOSURE, which puts the concrete at about 180 and leaves the paint
room above it. What still goes over the ceiling is scaled down as a whole colour, so it keeps
its hue.

for_renderer() prepares a photo this way. The same applies to plain material colours:
colour_for_renderer().
"""
import numpy as np

LIT = 0.70        # texture value to screen value, linear, flat ground in the script's light
AMBIENT = 0.17    # the part of that which reaches shadowed ground too
EXPOSURE = 0.80   # the scene's brightness against the photo's, in linear light


def decode(srgb):
    """sRGB values in 0..1 to linear light."""
    srgb = np.asarray(srgb, dtype=np.float32)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def encode(linear):
    """Linear light in 0..1 to sRGB values."""
    linear = np.clip(np.asarray(linear, dtype=np.float32), 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


_TABLE = (decode(np.arange(256) / 255.0) * (EXPOSURE / LIT)).astype(np.float32)


def for_renderer(photo):
    """An sRGB uint8 picture as the texture that makes the renderer show that picture."""
    out = np.empty(photo.shape, np.uint8)
    for r0 in range(0, photo.shape[0], 1024):     # in bands, to keep the float copy small
        value = _TABLE[photo[r0:r0 + 1024]]
        over = np.maximum(value.max(-1, keepdims=True), 1.0)
        out[r0:r0 + 1024] = (value / over * 255.0 + 0.5).astype(np.uint8)
    return out


def shown(texture):
    """What the renderer puts on screen for a texture made by for_renderer, sunlit: sRGB uint8."""
    return (encode(texture.astype(np.float32) / 255.0 * LIT) * 255.0 + 0.5).astype(np.uint8)


def colour_for_renderer(rgb):
    """An sRGB colour in 0..1 as the material colour that shows as that colour when sunlit."""
    value = decode(np.asarray(rgb, dtype=np.float32)) * (EXPOSURE / LIT)
    return [float(v) for v in value / max(float(value.max()), 1.0)]
