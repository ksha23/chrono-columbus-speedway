"""What a camera does to the light that reaches it: develop a rendered picture as a sensor would.

A renderer hands back the light itself: every pixel exactly its colour, sharp to the corner,
exposed by nobody. No camera does that, and software that is to be tested on pictures from a
real one should not be shown it. This takes a rendered picture and puts it through what
stands between the scene and the file a camera writes, in the order it happens:

  the lens       falls off toward the corners and blurs a little
  the exposure   is set from the picture's own brightness, and what is brighter than the
                 sensor can hold is lost
  the sensor     counts photons, which come at random, and adds noise of its own
  the processor  applies its tone curve and sharpens

It works on a plain array, so it serves a picture from look_sensor.py and a camera buffer in
a running simulation alike. It needs numpy only.

The numbers for each camera below are of two kinds, and each says which it is. Resolution and
field of view are the maker's. Everything else is typical of a sensor and lens of that kind and
is NOT measured on a real unit: fall-off, blur, well depth, read noise, sharpening. With one
frame from the real camera of a known scene they can be fitted, and until then they are a
reasonable camera, not that camera.

The picture is assumed rectified, as a stereo camera's own software delivers it: lens distortion
is not put in, because the software under test never sees it.
"""
import numpy as np

GAMMA = 2.2          # the renderer's: what it writes is light to the power of 1 / 2.2
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

CAMERAS = {
    "zedx_2mm": {
        "about": "Stereolabs ZED X One GS with the 2.2 mm lens, in its 960 x 600 binned mode",
        "size": (960, 600), "hfov": 110.0,                  # the maker's
        "falloff": 0.35,        # assumed: share of the light lost in the corners
        "blur": 0.6,            # assumed: the lens's blur as a Gaussian, in pixels
        "well": 40000.0,        # assumed: electrons a pixel holds, four 3 micron pixels binned
        "read": 6.0,            # assumed: electrons of noise per readout
        "bits": 12,
        "sharpen": (0.5, 1.0),  # assumed: the processor's unsharp mask, amount and radius in pixels
    },
    "zedx_4mm": {
        "about": "Stereolabs ZED X One GS with the 4.6 mm lens, in its 960 x 600 binned mode",
        "size": (960, 600), "hfov": 73.0,
        "falloff": 0.25, "blur": 0.6, "well": 40000.0, "read": 6.0, "bits": 12, "sharpen": (0.5, 1.0),
    },
}
EXPOSE_TO = 0.18         # the mean light an auto exposure aims for, as a share of what the sensor holds
EXPOSURES = (0.25, 4.0)  # how far it may turn the picture down or up


def _blurred(picture, sigma):
    """A picture blurred by a Gaussian of that many pixels, its edge carried on outward."""
    reach = max(int(np.ceil(3 * sigma)), 1)
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) / sigma) ** 2).astype(np.float32)
    kernel /= kernel.sum()
    for axis in (0, 1):
        pad = [(0, 0)] * picture.ndim
        pad[axis] = (reach, reach)
        wide = np.pad(picture, pad, mode="edge")
        picture = sum(k * np.take(wide, range(n, n + picture.shape[axis]), axis=axis) for n, k in enumerate(kernel))
    return picture


def vertical_fov(camera):
    """The vertical field of view in degrees of a camera whose picture is rectified."""
    wide, tall = camera["size"]
    return float(np.degrees(2 * np.arctan(np.tan(np.radians(camera["hfov"]) / 2) * tall / wide)))


def exposure(light, camera):
    """The gain an auto exposure settles on for a picture of linear light: its middle weighs double."""
    tall, wide = light.shape[:2]
    weight = np.ones((tall, wide), np.float32)
    weight[tall // 4:3 * tall // 4, wide // 4:3 * wide // 4] = 2.0
    mean = float(((light @ LUMA) * weight).sum() / weight.sum())
    return float(np.clip(EXPOSE_TO / max(mean, 1e-6), *EXPOSURES))


def develop(picture, camera, rng, gain=None):
    """A rendered picture (height, width, 3, uint8) as the camera would have written it. Same shape and type.

    rng is a numpy Generator: the noise is drawn from it, so a seed gives the same picture
    again. gain fixes the exposure, for a run of pictures that must not each choose their own.
    """
    light = (picture[..., :3].astype(np.float32) / 255.0) ** GAMMA
    tall, wide = light.shape[:2]
    # The lens. Light falls off with the square of the distance from the middle, to the corner's share.
    y, x = np.mgrid[0:tall, 0:wide].astype(np.float32)
    off = ((x - (wide - 1) / 2) ** 2 + (y - (tall - 1) / 2) ** 2) / (((wide - 1) / 2) ** 2 + ((tall - 1) / 2) ** 2)
    light *= (1.0 - camera["falloff"] * off)[..., None]
    if camera["blur"] > 0:
        light = _blurred(light, camera["blur"])
    # The exposure, and the sensor's ceiling.
    if gain is None:
        gain = exposure(light, camera)
    held = np.clip(light * gain, 0.0, 1.0)
    # The sensor. Photons arrive at random: of n expected, the square root of n more or fewer do.
    electrons = held * camera["well"]
    electrons = electrons + rng.standard_normal(electrons.shape, dtype=np.float32) * np.sqrt(electrons + camera["read"] ** 2)
    steps = 2 ** camera["bits"] - 1
    raw = np.clip(np.round(electrons / camera["well"] * steps), 0, steps) / steps
    # The processor: the tone curve, then sharpening, which is done on the picture as seen.
    seen = raw ** (1.0 / GAMMA)
    amount, radius = camera["sharpen"]
    if amount > 0:
        seen = seen + amount * (seen - _blurred(seen, radius))
    return np.clip(seen * 255.0 + 0.5, 0, 255).astype(np.uint8)
