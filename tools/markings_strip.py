"""Measure a painted line in the photo by straightening it out.

A strip is the photo resampled along a centre line: one row per pixel of length, one column
per half pixel of sideways offset. In a strip the paint is a bright column down the middle,
and everything about a stroke is read from it: what colour it is against the concrete on
both sides, where its crest lies in each row (so the centre line can be moved onto it),
where it starts and stops, and how wide it is.

Positions are photo pixels as (x, y), x to the right, y down, pixel centres at whole numbers.
"""
import numpy as np
from scipy import ndimage as ndi

OFFSETS = np.arange(-9.0, 9.01, 0.5)                 # sideways sample positions, pixels
LEFT = OFFSETS <= -3.0                               # the concrete either side of the line
RIGHT = OFFSETS >= 3.0
CORE = np.abs(OFFSETS) <= 2.5                        # where the paint itself is
CENTRE = np.abs(OFFSETS) <= 0.5
WINDOW = np.exp(-0.5 * (OFFSETS / 1.2) ** 2) * (np.abs(OFFSETS) <= 2.5)    # narrow: a second line 20 cm away must not pull
RAMP = np.clip((OFFSETS + 6.0) / 12.0, 0.0, 1.0)


def sample(photo, centres, normals):
    """The strip along a centre line: float32 (rows, offsets, 3)."""
    out = np.empty((len(centres), len(OFFSETS), 3), np.float32)
    for a in range(0, len(centres), 400):                       # in pieces, so each window stays small
        c, n = centres[a:a + 400], normals[a:a + 400]
        x = c[:, :1] + n[:, :1] * OFFSETS[None, :]
        y = c[:, 1:] + n[:, 1:] * OFFSETS[None, :]
        r0, c0 = max(int(np.floor(y.min())) - 1, 0), max(int(np.floor(x.min())) - 1, 0)
        r1, c1 = min(int(np.ceil(y.max())) + 2, photo.shape[0]), min(int(np.ceil(x.max())) + 2, photo.shape[1])
        if r1 - r0 < 2 or c1 - c0 < 2:
            out[a:a + 400] = 0
            continue
        win = np.asarray(photo[r0:r1, c0:c1]).astype(np.float32)
        for ch in range(3):
            out[a:a + 400, :, ch] = ndi.map_coordinates(win[..., ch], [y - r0, x - c0], order=1, mode="nearest")
    return out


def _sides(values, span):
    """The ground either side of the line in each row, steadied along the line."""
    left = ndi.median_filter(np.median(values[:, LEFT], axis=1), size=span, mode="nearest")
    right = ndi.median_filter(np.median(values[:, RIGHT], axis=1), size=span, mode="nearest")
    return left, right


def measure(strip, kind, seed, shaded=False):
    """Read a strip. `seed` marks the rows already believed to be paint.

    Returns None when the middle column does not stand out, else a dict:
      amount   per row, how much paint the row shows: the excess over the concrete, as a
               fraction of it, summed across the line. For yellow it is the excess in hue
               (yellow against blue), which a patch of sun or shade on the line does not change.
      shift    per row, how far the crest sits from the centre column (pixels)
      level    the amount where the paint is at its usual strength
      rel      per channel, the line's contrast against the concrete as a fraction
      ground   per channel, the concrete's colour; tilt: how much the two sides differ
      excess   what the crest and the width are read from (rows, offsets)
      clear    the line's height over the ground, in units of the ground's own grain
    """
    smooth = ndi.uniform_filter1d(strip, 5, axis=0, mode="nearest")
    span = (min(15, 2 * (len(strip) // 2) - 1), 1)                  # a line crossing this one must not count as ground
    left, right = _sides(smooth, span)
    back = left[:, None, :] + (right - left)[:, None, :] * RAMP[None, :, None]
    over = (smooth - back) / np.maximum(back, 8.0)                   # as a fraction of the ground: the same in shade as in sun
    mid = over[:, CENTRE].mean(1)                                    # rows x 3
    hue = None
    if kind == "yellow":
        tone = np.log(((smooth[..., 0] + smooth[..., 1]) * 0.5 + 16.0) / (smooth[..., 2] + 16.0))
        tl, tr = _sides(tone[..., None], span)
        hue = tone - (tl + (tr - tl) * RAMP[None, :])
        rough = hue[:, CORE].sum(1)
    else:
        rough = over[:, CORE].mean(2).sum(1)
    if not seed.any():
        return None
    base = np.median(rough[seed])
    if base <= 0:
        return None
    rows = seed & (rough > 0.5 * base)
    if rows.sum() < 3:
        return None
    contrast = np.median(mid[rows], axis=0)
    ground = np.median((left[rows] + right[rows]) * 0.5, axis=0)
    total = np.abs(contrast).sum()
    if total < 0.05:
        return None
    lean = contrast.copy()                                           # lean on the channels that carry this line
    if kind == "yellow" and abs(lean[0]) + abs(lean[1]) >= 0.04:
        lean[2] = 0.0                # blue is the photo's coarsest channel: it sits up to a pixel off the other two
    excess = over @ (lean / np.abs(lean).sum())
    amount = (hue if hue is not None else excess)[:, CORE].sum(1) * 0.5
    if hue is not None and shaded:
        excess = hue                                                 # under trees brightness says sun fleck, hue says paint
    plus = np.clip(excess, 0, None) * WINDOW[None, :]
    shift = (plus * OFFSETS[None, :]).sum(1) / np.maximum(plus.sum(1), 1e-6)
    level = float(np.percentile(amount[rows], 85))                   # the paint at its usual strength, not its worn parts
    if level <= 0:
        return None
    sides = np.abs(left[rows] - right[rows]).mean(1) / np.maximum(((left[rows] + right[rows]) * 0.5).mean(1), 1.0)
    signal = hue if hue is not None else excess                      # how far the line stands above the ground's own grain
    grain = np.median(np.minimum(signal[rows][:, LEFT].std(1), signal[rows][:, RIGHT].std(1)))   # the quieter side: the other may hold a second line
    clear = float(np.median(signal[rows][:, CENTRE].mean(1)) / max(grain, 1e-4))
    return {"amount": amount, "shift": shift, "level": level, "rel": contrast, "ground": ground,
            "tilt": float(np.median(sides)), "excess": excess, "rows": rows, "clear": clear}


def runs(amount, level, known):
    """Stretches of rows where there is paint: (start, stop) to a fraction of a row.

    A row counts when its amount is over half the level. A dip does not break a run if it lasts
    two rows or less, or is short and stays over a quarter, or lies under something that hides
    the ground. Each end is put where the amount crosses half of what the run has near that end.
    """
    n = len(amount)
    on = amount > 0.5 * level
    soft = amount > 0.25 * level
    out = []
    i = 0
    while i < n:
        if not (on[i] and known[i]):
            i += 1
            continue
        j = i
        while j + 1 < n:
            k = j + 1
            if on[k] and known[k]:
                j = k
                continue
            end = k                                             # look across the gap
            while end < n and not (on[end] and known[end]):
                end += 1
            if end >= n:
                break
            gap = slice(k, end)
            hidden = ~known[gap]
            if (end - k) <= 2 or ((soft[gap] | hidden).all() and hidden.sum() <= 12 and (~hidden).sum() <= 6):
                j = end
                continue
            break
        out.append((_edge(amount, known, i, j, -1), _edge(amount, known, i, j, +1)))
        i = j + 1
    return out


def _edge(amount, known, i, j, way):
    """Where a run's amount falls through half its local level, going outwards from one end."""
    n = len(amount)
    inner = amount[i:j + 1]
    near = inner[:10] if way < 0 else inner[-10:]
    half = 0.5 * float(np.median(near))
    k = i if way < 0 else j
    while 0 <= k + way < n and known[k + way] and amount[k + way] > half:      # the run may reach a little further
        k += way
    while i <= k <= j and amount[k] < half:                                    # or stop a little short
        k -= way
    nxt = k + way
    if nxt < 0 or nxt >= n or not known[nxt]:
        return float(k)
    a, b = amount[k], amount[nxt]
    return float(k + way * (a - half) / max(a - b, 1e-6)) if a > b else float(k)


def profile(excess, rows, shift):
    """The paint's average cross-section, each row slid so its crest is at zero, and its width at half height."""
    grid = OFFSETS
    acc = np.zeros(len(grid))
    for r in np.nonzero(rows)[0]:
        acc += np.interp(grid + shift[r], grid, excess[r])
    acc /= max(int(rows.sum()), 1)
    top = acc[CENTRE].max()
    if top <= 0:
        return acc, 0.0
    fine = np.arange(-6.0, 6.001, 0.05)
    p = np.interp(fine, grid, acc)
    c = int(np.argmin(np.abs(fine)))
    lo = c
    while lo > 0 and p[lo] > 0.5 * top:
        lo -= 1
    hi = c
    while hi < len(fine) - 1 and p[hi] > 0.5 * top:
        hi += 1
    return acc, float(fine[hi] - fine[lo])


def weighted_line(s, shift, weight):
    """Least-squares shift = a + b s, fitted twice so that a few wild rows do not steer it.

    Returns a, b and the weighted rms of what is left.
    """
    w = weight / weight.sum()
    for _ in range(2):
        sm = (w * s).sum()
        var = (w * (s - sm) ** 2).sum()
        b = (w * (s - sm) * shift).sum() / var if var > 1e-9 else 0.0
        a = (w * shift).sum() - b * sm
        left = shift - a - b * s
        rms = float(np.sqrt((w * left ** 2).sum()))
        w = weight / (1.0 + (left / 0.4) ** 2)
        w = w / w.sum()
    return float(a), float(b), rms
