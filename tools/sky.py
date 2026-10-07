"""Turn a sky picture so that its sun stands where the scene's light comes from.

Chrono::Sensor shows a sky from an equirectangular picture in Radiance .hdr format, and lights
the scene with a lamp that knows nothing of the picture. If the two disagree the shadows fall
toward the sun, which the eye notices at once. The renderer cannot turn the picture. So a copy
of it is made with its columns shifted round, which is the same thing: a column is a compass
direction.

Only what that needs is here: reading the .hdr format (run-length coded or plain), finding the
sun, and writing the file back. Pixels stay as the four bytes they are stored as, red, green,
blue and a shared exponent, so nothing is lost in between.
"""
import numpy as np


def read(path):
    """A Radiance .hdr picture as (height, width, 4) bytes: red, green, blue, exponent."""
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"\n\n") + 2
    size = data[end:data.index(b"\n", end)].split()
    if len(size) != 4 or size[0] != b"-Y" or size[2] != b"+X":
        raise ValueError(f"{path}: only pictures stored top row first are read, and this one says {size}")
    height, width = int(size[1]), int(size[3])
    at = data.index(b"\n", end) + 1
    raw = np.frombuffer(data, np.uint8)
    out = np.empty((height, width, 4), np.uint8)
    for row in range(height):
        if not (8 <= width < 32768 and raw[at] == 2 and raw[at + 1] == 2 and (int(raw[at + 2]) << 8 | int(raw[at + 3])) == width):
            out[row:] = raw[at:at + (height - row) * width * 4].reshape(height - row, width, 4)      # plain, to the end
            break
        at += 4
        for channel in range(4):
            col = 0
            while col < width:
                count = int(raw[at])
                if count > 128:                       # a run: the next byte, this many times
                    out[row, col:col + count - 128, channel] = raw[at + 1]
                    col, at = col + count - 128, at + 2
                else:                                 # this many bytes as they are
                    out[row, col:col + count, channel] = raw[at + 1:at + 1 + count]
                    col, at = col + count, at + 1 + count
    return out


def write(path, picture):
    """Write (height, width, 4) bytes as a Radiance .hdr picture, run-length framed and not compressed."""
    height, width, _ = picture.shape
    # Each row: a marker with the width, then each of the four channels in pieces of 128 bytes
    # at most, every piece led by its length.
    starts = list(range(0, width, 128))
    framed = np.empty((height, 4, width + len(starts)), np.uint8)
    channels = picture.transpose(0, 2, 1)
    for n, start in enumerate(starts):
        count = min(128, width - start)
        framed[:, :, start + n] = count
        framed[:, :, start + n + 1:start + n + 1 + count] = channels[:, :, start:start + count]
    marker = np.array([2, 2, width >> 8, width & 255], np.uint8)
    rows = np.concatenate([np.broadcast_to(marker, (height, 4)), framed.reshape(height, -1)], axis=1)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {height} +X {width}\n".encode())
        f.write(rows.tobytes())


def brightness(picture):
    """How bright each pixel is, as a float picture."""
    return picture[..., :3].astype(np.float32).mean(-1) * np.exp2(picture[..., 3].astype(np.float32) - 136.0)


def sun(picture):
    """Where the sun is: (column, row), the middle of the brightest part of the upper half."""
    light = brightness(picture[:picture.shape[0] // 2])
    # The sun's disc is clipped at a value thousands of times the sky's. Its middle is the
    # mean place of what is within a tenth of the brightest, taken round the join at the edge.
    rows, cols = np.nonzero(light >= 0.1 * light.max())
    angle = cols * 2 * np.pi / light.shape[1]
    middle = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()) % (2 * np.pi)
    return float(middle * light.shape[1] / (2 * np.pi)), float(rows.mean())


def grounded(picture, colour, below=0.3, over=1.2):
    """The picture with one colour for ground: everything lower than a little below the horizon.

    A sky picture is taken somewhere, and what lies under its horizon is that place's ground:
    a field of sunflowers, a quarry. From the air it would show round the scene as if it were
    there. below is how many degrees under the horizon the colour starts and over how many
    more it takes over, so that what stands on the picture's horizon stays.
    """
    height = picture.shape[0]
    under = (np.arange(height) + 0.5) * 180.0 / height - 90.0            # degrees below the horizon, by row
    share = np.clip((under - below) / over, 0.0, 1.0)[:, None, None]
    light = picture[..., :3].astype(np.float32) * np.exp2(picture[..., 3:].astype(np.float32) - 136.0)
    mixed = light * (1.0 - share) + np.asarray(colour, np.float32) * share
    out = picture.copy()
    rows = np.nonzero(share[:, 0, 0] > 0)[0]
    top = np.maximum(mixed[rows].max(-1), 1e-32)
    mantissa, exponent = np.frexp(top)
    out[rows, :, :3] = np.clip(mixed[rows] * (mantissa * 256.0 / top)[..., None], 0, 255).astype(np.uint8)
    out[rows, :, 3] = (exponent + 128).astype(np.uint8)
    return out


def turned(picture, columns):
    """The picture with everything moved that many columns to the right, round the join."""
    return np.roll(picture, int(round(columns)), axis=1)
