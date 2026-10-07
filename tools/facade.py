"""Paint a wall: its cladding, and the doors and windows that are known to be in it.

The drone looked down, so what the scan has of a wall is a smear. But a smear still says what
the wall is clad in, what colour it is, and where the bright rectangle of a door or the dark
one of a window sits. Those few facts are read off by hand (facades.json), and the wall is
painted from them: a clean picture of the right kind of wall, not a photograph of this one.

A wall is painted as a sheet, s along it from left to right as seen from outside and z up
from the ground, at PIX metres a pixel. Colours are as the eye would see them in the sun.
"""
import numpy as np
from scipy import ndimage

PIX = 0.02          # metres a pixel of a wall covers
RIB = 0.30          # metres between the ribs of metal cladding
BOARD = 0.15        # metres a lap board shows
GROOVE = 0.40       # metres between the grooves of a panelled wall
COURSE = 0.20       # metres a course of stone is high
FOOTING = 0.15      # metres of concrete that show at the foot of a wall
CORNER = 0.10       # metres a corner trim is wide
EAVE_SHADE = 0.35   # metres below the roof that a wall is in the roof's own shade
WHITE = (236, 236, 232)
CONCRETE = (176, 174, 168)
GLASS = (74, 112, 122)


class Sheet:
    """A wall's picture being painted: float colours, rows from the top down."""

    def __init__(self, length, top, sunk):
        self.top, self.sunk = top, sunk
        self.rows, self.cols = int(np.ceil((top + sunk) / PIX)), max(int(np.ceil(length / PIX)), 2)
        self.s = (np.arange(self.cols) + 0.5) * PIX
        self.z = top - (np.arange(self.rows) + 0.5) * PIX
        self.rgb = np.zeros((self.rows, self.cols, 3), np.float32)

    def box(self, s0, s1, z0, z1):
        """The pixels of a rectangle given in metres, as a pair of slices."""
        c0, c1 = int(round(s0 / PIX)), int(round(s1 / PIX))
        r0, r1 = int(round((self.top - z1) / PIX)), int(round((self.top - z0) / PIX))
        return slice(max(r0, 0), max(min(r1, self.rows), 0)), slice(max(c0, 0), max(min(c1, self.cols), 0))

    def fill(self, s0, s1, z0, z1, colour):
        self.rgb[self.box(s0, s1, z0, z1)] = colour

    def shade(self, s0, s1, z0, z1, by):
        self.rgb[self.box(s0, s1, z0, z1)] *= by

    def frame(self, s0, s1, z0, z1, width, colour):
        """The border of a rectangle, drawn inward."""
        for box in ((s0, s1, z1 - width, z1), (s0, s1, z0, z0 + width), (s0, s0 + width, z0, z1), (s1 - width, s1, z0, z1)):
            self.fill(*box, colour)


def cladding(kind, colour, s, z, rng):
    """A stretch of wall in one cladding: an array (len(z), len(s), 3) of colours.

    s and z are the metres of each column and row.
    """
    shade = np.ones((len(z), len(s)), np.float32)
    if kind == "ribbed":
        along = np.mod(s, RIB)
        shade *= np.where(along < 0.03, 0.80, np.where(along < 0.05, 1.07, 1.0))[None, :]
        shade *= (1.0 + 0.03 * rng.standard_normal(int(s[-1] / 0.9) + 2))[(s / 0.9).astype(int)][None, :]      # sheet by sheet
    elif kind == "lap":
        down = np.mod(-z, BOARD) / BOARD
        shade *= np.where(down > 0.88, 0.80, 1.03 - 0.06 * down)[:, None]
    elif kind == "panel":
        shade *= np.where(np.mod(s, GROOVE) < PIX, 0.86, 1.0)[None, :]
    elif kind == "stone":
        course = np.floor(z / COURSE).astype(int)
        for c in np.unique(course):
            rows = course == c
            ends = np.cumsum(rng.uniform(0.25, 0.6, int(s[-1] / 0.25) + 3)) - rng.uniform(0.0, 0.4)
            block = np.searchsorted(ends, s)
            tone = rng.uniform(0.84, 1.08, block.max() + 1)[block]
            tone[np.diff(block, prepend=block[0]) != 0] = 0.74          # the joint between two stones
            shade[rows] *= tone[None, :]
        shade[np.mod(z, COURSE) < PIX] *= 0.76                          # and the one between two courses
    # Nothing built is one even tone: a little slow unevenness over the whole wall.
    cloud = ndimage.gaussian_filter(rng.standard_normal((len(z) // 8 + 2, len(s) // 8 + 2)), 3.0)
    cloud = np.kron(cloud, np.ones((8, 8), np.float32))[:len(z), :len(s)]
    shade *= 1.0 + 0.35 * cloud
    return shade[..., None] * np.asarray(colour, np.float32)


def _glass(sheet, s0, s1, z0, z1, colour):
    rows, cols = sheet.box(s0, s1, z0, z1)
    if rows.stop <= rows.start or cols.stop <= cols.start:
        return
    down = np.linspace(1.18, 0.78, rows.stop - rows.start, dtype=np.float32)       # the sky above, the ground below
    sheet.rgb[rows, cols] = down[:, None, None] * np.asarray(colour, np.float32)


def window(sheet, f):
    s0, s1, z0, z1 = f["s"] - f["wide"] / 2, f["s"] + f["wide"] / 2, f["sill"], f["sill"] + f["tall"]
    frame = f.get("frame", WHITE)
    sheet.shade(s0 - 0.03, s1 + 0.03, z0 - 0.07, z1 + 0.03, 0.72)              # the reveal's shadow
    sheet.fill(s0, s1, z0, z1, frame)
    _glass(sheet, s0 + 0.06, s1 - 0.06, z0 + 0.06, z1 - 0.06, f.get("glass", GLASS))
    panes = f.get("panes", 2 if f["wide"] > 1.2 else 1)
    for n in range(1, panes):
        at = s0 + n * f["wide"] / panes
        sheet.fill(at - 0.025, at + 0.025, z0, z1, frame)
    for n in range(1, f.get("bars", 1)):
        at = z0 + n * f["tall"] / f.get("bars", 1)
        sheet.fill(s0, s1, at - 0.025, at + 0.025, frame)
    sheet.fill(s0 - 0.04, s1 + 0.04, z0 - 0.04, z0, frame)                       # the sill


def door(sheet, f):
    s0, s1, z0, z1 = f["s"] - f["wide"] / 2, f["s"] + f["wide"] / 2, f.get("sill", 0.0), f.get("sill", 0.0) + f["tall"]
    colour = np.asarray(f.get("colour", WHITE), np.float32)
    sheet.fill(s0 - 0.05, s1 + 0.05, z0, z1 + 0.05, f.get("frame", WHITE))
    sheet.fill(s0, s1, z0, z1, colour)
    for low, high in ((0.12, 0.44), (0.52, 0.90)):                               # two pressed panels
        sheet.frame(s0 + 0.12, s1 - 0.12, z0 + low * f["tall"], z0 + high * f["tall"], PIX, colour * 0.84)
    sheet.fill(s1 - 0.14, s1 - 0.08, z0 + 0.95, z0 + 1.07, (60, 60, 62))         # the handle
    sheet.fill(s0, s1, z0, z0 + 0.03, (120, 120, 118))                           # the threshold


def glass_door(sheet, f):
    s0, s1, z0, z1 = f["s"] - f["wide"] / 2, f["s"] + f["wide"] / 2, f.get("sill", 0.0), f.get("sill", 0.0) + f["tall"]
    frame = f.get("frame", (150, 152, 155))
    sheet.fill(s0, s1, z0, z1, frame)
    leaves = f.get("leaves", 2)
    for n in range(leaves):
        a, b = s0 + n * f["wide"] / leaves, s0 + (n + 1) * f["wide"] / leaves
        _glass(sheet, a + 0.07, b - 0.07, z0 + 0.20, z1 - 0.07, f.get("glass", (52, 74, 84)))
        sheet.fill(b - 0.16 if n == 0 else a + 0.11, b - 0.11 if n == 0 else a + 0.16, z0 + 0.85, z0 + 1.25, frame)       # the pull


def roll_up(sheet, f):
    s0, s1, z0, z1 = f["s"] - f["wide"] / 2, f["s"] + f["wide"] / 2, 0.0, f["tall"]
    colour = np.asarray(f.get("colour", WHITE), np.float32)
    sheet.fill(s0 - 0.10, s1 + 0.10, z0, z1 + 0.25, colour * 0.93)               # the guides and the drum's hood
    rows, cols = sheet.box(s0, s1, z0, z1)
    slat = np.where(np.mod(sheet.z[rows], 0.08) < PIX, 0.88, 1.0).astype(np.float32)
    sheet.rgb[rows, cols] = slat[:, None, None] * colour
    sheet.fill(s0, s1, z0, z0 + 0.06, (70, 70, 72))                              # the rubber at the foot


def overhead(sheet, f):
    """A sectional garage door: four sections of pressed panels."""
    s0, s1, z0, z1 = f["s"] - f["wide"] / 2, f["s"] + f["wide"] / 2, 0.0, f["tall"]
    colour = np.asarray(f.get("colour", WHITE), np.float32)
    sheet.fill(s0 - 0.08, s1 + 0.08, z0, z1 + 0.08, f.get("frame", WHITE))
    sheet.shade(s0, s1, z0, z1 + 0.02, 0.80)
    sheet.fill(s0, s1, z0, z1, colour)
    across = max(int(round(f["wide"] / 0.75)), 2)
    for row in range(4):
        low = z0 + row * f["tall"] / 4
        sheet.shade(s0, s1, low, low + PIX, 0.80)
        for n in range(across):
            a = s0 + n * f["wide"] / across
            sheet.frame(a + 0.08, a + f["wide"] / across - 0.08, low + 0.10, low + f["tall"] / 4 - 0.08, PIX, colour * 0.88)


def louvre(sheet, f):
    s0, s1, z0, z1 = f["s"] - f["wide"] / 2, f["s"] + f["wide"] / 2, f["sill"], f["sill"] + f["tall"]
    colour = np.asarray(f.get("colour", WHITE), np.float32)
    sheet.fill(s0, s1, z0, z1, colour)
    rows, cols = sheet.box(s0 + 0.05, s1 - 0.05, z0 + 0.05, z1 - 0.05)
    slat = np.where(np.mod(sheet.z[rows], 0.08) < 0.04, 0.62, 0.94).astype(np.float32)
    sheet.rgb[rows, cols] = slat[:, None, None] * colour


def strip(sheet, f):
    """A run of translucent sheets let into the cladding, as a shed has for daylight."""
    s0, s1, z0, z1 = f["s"] - f["wide"] / 2, f["s"] + f["wide"] / 2, f["sill"], f["sill"] + f["tall"]
    colour = np.asarray(f.get("colour", (196, 208, 216)), np.float32)
    rows, cols = sheet.box(s0, s1, z0, z1)
    rib = np.where(np.mod(sheet.s[cols], RIB) < 0.03, 0.90, 1.0).astype(np.float32)
    sheet.rgb[rows, cols] = rib[None, :, None] * colour
    for n in range(int(round(f["wide"] / 1.2)) + 1):
        at = s0 + n * f["wide"] / max(int(round(f["wide"] / 1.2)), 1)
        sheet.fill(at - 0.02, at + 0.02, z0, z1, colour * 0.72)
    sheet.frame(s0, s1, z0, z1, 0.04, colour * 0.72)


def plain_box(sheet, f):
    s0, s1, z0, z1 = f["s"] - f["wide"] / 2, f["s"] + f["wide"] / 2, f["sill"], f["sill"] + f["tall"]
    sheet.fill(s0, s1, z0, z1, f["colour"])
    sheet.frame(s0, s1, z0, z1, PIX, np.asarray(f["colour"], np.float32) * 0.7)


DRAW = {"window": window, "door": door, "glass_door": glass_door, "roll_up": roll_up, "overhead": overhead,
        "louvre": louvre, "strip": strip, "box": plain_box}


def paint(wall, style, features, sunk, seed):
    """One wall's picture: uint8, rows from the wall's highest point down to sunk below the ground.

    wall has length, tops (the height of its top edge at points along it, as (s, z) pairs) and
    gable (whether the ridge runs into it). style has the cladding and its colours. features
    are the things in this wall, each with s, the metres along it of its middle.
    """
    rng = np.random.default_rng(seed)
    tops = np.array(wall["tops"])
    sheet = Sheet(wall["length"], float(tops[:, 1].max()) + PIX, sunk)
    sheet.rgb[:] = cladding(style.get("cladding", "plain"), style.get("colour", (190, 188, 182)), sheet.s, sheet.z, rng)
    upper = style.get("upper")
    if upper and (wall["gable"] or not upper.get("gables_only", True)):
        rows, cols = sheet.box(0.0, wall["length"], upper["from"], sheet.top)
        sheet.rgb[rows, cols] = cladding(upper.get("cladding", style.get("cladding", "plain")), upper["colour"], sheet.s[cols], sheet.z[rows], rng)
        sheet.fill(0.0, wall["length"], upper["from"] - 0.03, upper["from"] + 0.03, np.asarray(upper["colour"], np.float32) * 0.86)
    skirt = style.get("skirt")
    if skirt:
        rows, cols = sheet.box(0.0, wall["length"], -sunk, skirt["to"])
        sheet.rgb[rows, cols] = cladding(skirt.get("cladding", "panel"), skirt["colour"], sheet.s[cols], sheet.z[rows], rng)
    for f in features:
        if f["kind"] == "cladding":           # a stretch of the wall in something else, or with no width given the whole of it
            along = (f["s"] - f["wide"] / 2, f["s"] + f["wide"] / 2) if "wide" in f else (0.0, wall["length"])
            rows, cols = sheet.box(*along, f.get("sill", -sunk), f.get("sill", -sunk) + f["tall"] if "tall" in f else sheet.top)
            sheet.rgb[rows, cols] = cladding(f["cladding"], f["colour"], sheet.s[cols], sheet.z[rows], rng)
    trim = style.get("trim", WHITE)
    sheet.fill(0.0, CORNER, -sunk, sheet.top, trim)
    sheet.fill(wall["length"] - CORNER, wall["length"], -sunk, sheet.top, trim)
    sheet.fill(0.0, wall["length"], -sunk, FOOTING, CONCRETE)
    for f in features:
        if f["kind"] in DRAW:
            DRAW[f["kind"]](sheet, f)
    # The roof shades the top of the wall under it: deepest right under the eave.
    edge = np.interp(sheet.s, tops[:, 0], tops[:, 1])
    below = (edge[None, :] - sheet.z[:, None]) / EAVE_SHADE
    sheet.rgb *= np.clip(0.72 + 0.28 * below, 0.72, 1.0)[..., None]
    # And rain leaves the foot of it a little darker.
    sheet.rgb *= np.clip(0.90 + 0.10 * (sheet.z[:, None, None] - FOOTING) / 0.5, 0.90, 1.0)
    return np.clip(sheet.rgb + 0.5, 0, 255).astype(np.uint8)


def atlas(pictures, gap=4):
    """Several walls' pictures as one: (the picture, where each one's top left corner is as (row, column)).

    Each is stood on the one before with a few pixels of its own edge repeated between, so
    that one wall's colours do not bleed into the next when the picture is shrunk.
    """
    wide = max(p.shape[1] for p in pictures) + 2 * gap
    tall = sum(p.shape[0] + 2 * gap for p in pictures)
    out = np.zeros((tall, wide, 3), np.uint8)
    places, row = [], 0
    for p in pictures:
        out[row:row + p.shape[0] + 2 * gap, :p.shape[1] + 2 * gap] = np.pad(p, ((gap, gap), (gap, gap), (0, 0)), mode="edge")
        out[row:row + p.shape[0] + 2 * gap, p.shape[1] + 2 * gap:] = out[row:row + p.shape[0] + 2 * gap, p.shape[1] + 2 * gap - 1:p.shape[1] + 2 * gap]
        places.append((row + gap, gap))
        row += p.shape[0] + 2 * gap
    return out, places
