"""Tufts of grass as the renderer needs them: a sheet of blade pictures, and cards that wear them.

A tuft is two upright cards that cross, so it shows blades from any side. A card is a
rectangle whose picture is a row of blades with nothing between them: where the picture is
clear the camera sees through it, and so does the sun. All of it is synthesized. The blades
are drawn, not photographed, and no tuft is one that grows on the site.

One sheet holds every picture: a row for each colour the grass comes in and a column for each
shape of tuft. Chrono::Sensor's Metal renderer has room for 64 textures and this scene uses 61
before any detail is added, so a picture per colour is not to be had. A card picks its colour and its shape by
where its corners point into the sheet.

A card is lit by the normals its corners carry, and those are the ground's own, not the
card's: standing grass then takes the sun as the lawn under it does, and a tuft comes out the
colour of the ground it stands on, a little darker where its neighbours shade it.
"""
import numpy as np
from PIL import Image, ImageDraw

import worldnoise

GAMMA = 2.2          # the camera's light from a texture value: Chrono::Sensor raises it to this power
CELL = (160, 64)     # pixels of one tuft's picture: about 2 mm of blade to a pixel on a card of common size
PAD = 4              # clear pixels around each picture, so no picture's edge is blended with its neighbour's
SHAPES = 6           # different tufts drawn
OVER = 4             # drawn this many times finer and averaged down, for soft edges
BLADES = 30          # tall blades in a tuft's picture
THATCH = 44          # short ones along its foot, so the foot is closed
STRAW = 0.10         # the share of blades that are dry
SEED = 1510


def _draws(count, shape, what):
    """count numbers in 0..1 for one shape of tuft: a different set for every what."""
    return worldnoise.uniform(worldnoise.bits(np.arange(count), shape * 1000 + what, SEED))


def shape(n):
    """Tuft n as three pictures of CELL: how much of each pixel is blade, how bright the blade, how dry."""
    wide, tall = CELL[0] * OVER, CELL[1] * OVER
    cover, shade, dry = (Image.new("L", (wide, tall), fill) for fill in (0, 128, 0))
    pens = [ImageDraw.Draw(layer) for layer in (cover, shade, dry)]
    count = BLADES + THATCH
    foot = (np.arange(count) % BLADES + _draws(count, n, 0)) / BLADES
    foot[BLADES:] = _draws(THATCH, n, 1)
    height = np.where(np.arange(count) < BLADES, 1.0 - 0.6 * _draws(count, n, 2) ** 1.6, 0.12 + 0.22 * _draws(count, n, 3))
    lean = (_draws(count, n, 4) - 0.5) * 0.22 + (_draws(1, n, 5)[0] - 0.5) * 0.10
    bend = (_draws(count, n, 6) - 0.5) * 0.10
    width = np.where(_draws(count, n, 7) < 0.7, 6.0 + 4.0 * _draws(count, n, 8), 3.0 + 1.5 * _draws(count, n, 8)) * OVER
    bright = 0.78 + 0.44 * _draws(count, n, 9)
    straw = _draws(count, n, 10) < STRAW
    order = np.argsort(-height, kind="stable")                    # tall ones first: the short ones stand in front
    steps = np.linspace(0.0, 1.0, 7)
    for k in order:
        base = np.array([foot[k] * wide, tall])
        tip = np.array([(foot[k] + lean[k] * height[k]) * wide, tall * (1.0 - height[k])])
        knee = (base + tip) / 2 + [bend[k] * wide, 0.0]
        spine = (1 - steps)[:, None] ** 2 * base + 2 * ((1 - steps) * steps)[:, None] * knee + steps[:, None] ** 2 * tip
        half = np.maximum(width[k] / 2 * (1 - steps) ** 0.7, OVER * 0.45)
        for s in range(len(steps) - 1):
            quad = [(spine[s, 0] - half[s], spine[s, 1]), (spine[s, 0] + half[s], spine[s, 1]),
                    (spine[s + 1, 0] + half[s + 1], spine[s + 1, 1]), (spine[s + 1, 0] - half[s + 1], spine[s + 1, 1])]
            pens[0].polygon(quad, fill=255)
            # A blade is a little darker at its foot and lighter toward its tip. Only a little: the renderer
            # shades the foot of a tuft itself, and tips drawn much lighter stand out pale against the lawn.
            pens[1].polygon(quad, fill=int(np.clip(128 * bright[k] * (0.90 + 0.18 * steps[s]), 0, 255)))
            pens[2].polygon(quad, fill=255 if straw[k] else 0)
    return [np.asarray(layer.resize(CELL, Image.BOX), np.float32) / 255.0 for layer in (cover, shade, dry)]


def sheet(colours):
    """The sheet of pictures for these colours (light, 0..1, one row each): an RGBA uint8 picture.

    Each picture's blades average, in light, to exactly its row's colour: brighter and darker
    blades, and dry ones, are spread about that.
    """
    colours = np.asarray(colours, np.float32)
    step = (CELL[0] + 2 * PAD, CELL[1] + 2 * PAD)
    out = np.zeros((len(colours) * step[1], SHAPES * step[0], 4), np.uint8)
    shapes = [shape(n) for n in range(SHAPES)]
    for row, colour in enumerate(colours):
        # A dry blade: brighter, and from green toward straw.
        dried = colour * np.array([1.30, 1.18, 0.88], np.float32)
        out[row * step[1]:(row + 1) * step[1], :, :3] = (colour ** (1 / GAMMA) * 255 + 0.5).astype(np.uint8)
        for col, (cover, shade, dry) in enumerate(shapes):
            light = (colour + (dried - colour) * dry[..., None]) * (shade[..., None] * (255.0 / 128.0))
            weight = cover[..., None]
            light *= colour / np.maximum((light * weight).sum((0, 1)) / weight.sum(), 1e-6)
            # On a pale lawn the brightest blade would pass white: pull them all toward the row's colour, which keeps the average.
            light = colour + (light - colour) * np.minimum(1.0, (1.0 - colour) / np.maximum((light * (weight > 0)).max((0, 1)) - colour, 1e-6))
            y, x = row * step[1] + PAD, col * step[0] + PAD
            out[y:y + CELL[1], x:x + CELL[0], :3] = (np.clip(light, 0, 1) ** (1 / GAMMA) * 255 + 0.5).astype(np.uint8)
            out[y:y + CELL[1], x:x + CELL[0], 3] = (cover * 255 + 0.5).astype(np.uint8)
    return out


def corners(rows):
    """Texture coordinates of every picture's four corners on a sheet of that many rows: (rows * SHAPES * 4, 2).

    In the order foot left, foot right, top right, top left, for row * SHAPES + shape.
    """
    step = (CELL[0] + 2 * PAD, CELL[1] + 2 * PAD)
    wide, tall = SHAPES * step[0], rows * step[1]
    found = []
    for row in range(rows):
        for col in range(SHAPES):
            u0, u1 = (col * step[0] + PAD + 0.5) / wide, (col * step[0] + PAD + CELL[0] - 0.5) / wide
            v0, v1 = 1 - (row * step[1] + PAD + CELL[1]) / tall, 1 - (row * step[1] + PAD + 0.5) / tall
            found += [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    return np.array(found)


def write(path, foot_a, foot_b, top_a, top_b, picture, flipped, normals, tuft):
    """Write cards as an OBJ file. Returns the number of triangles.

    foot_a, foot_b, top_a, top_b: (n, 3) corners of each card. picture: row * SHAPES + shape for
    each. flipped: whether a card wears its picture mirrored. normals: (tufts, 3), and tuft says
    which of them each card's corners carry. Positions, texture coordinates and normals are
    listed separately and a face names one of each, which keeps the file a third the size.
    """
    rows = int(picture.max()) // SHAPES + 1 if len(picture) else 1
    with open(path, "w") as f:
        f.write("# Synthesized grass tufts, made by tools/verge.py. Metres, Z up, scene coordinates.\n")
        corner = np.stack([foot_a, foot_b, top_b, top_a], axis=1).reshape(-1, 3)
        f.write("".join(f"v {x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in corner))
        f.write("".join(f"vt {u:.5f} {v:.5f}\n" for u, v in corners(rows)))
        f.write("".join(f"vn {x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in normals))
        order = np.where(flipped[:, None], [1, 0, 3, 2], [0, 1, 2, 3]) + picture[:, None] * 4 + 1
        first = np.arange(len(picture)) * 4 + 1
        n = tuft + 1
        f.write("".join(f"f {v}/{t[0]}/{k} {v + 1}/{t[1]}/{k} {v + 2}/{t[2]}/{k}\nf {v}/{t[0]}/{k} {v + 2}/{t[2]}/{k} {v + 3}/{t[3]}/{k}\n"
                        for v, t, k in zip(first.tolist(), order.tolist(), n.tolist())))
    return 2 * len(picture)
