"""Turn an SUV body into a crew-cab pickup: Chrono's data has no pickup of its own.

The body is cut off behind the rear doors, above the line of the window sills, and what that
opens is closed with flat pieces made here: a back wall for the cab with a window in it, and
a load bed with a rim, sides and a floor. Everything below the sills is the SUV's own metal,
so the wheels, arches, tailgate and lamps stay where Chrono has them.
"""
import numpy as np

import carmesh

RIM = 0.05        # width of the bed's top edge, metres


def convert(P, N, part, colour, rear_axle_x, tyre_top, options, log):
    """P, N, part, colour of the body in the level frame (z = 0 the ground), made into a pickup.

    options: "cab_back", how far ahead of the rear axle the cab ends; "belt", the height of
    the cut along the sills; "roof", a height above which everything goes (the roof rack).
    """
    wall_x = rear_axle_x + options["cab_back"]
    belt = options["belt"]
    paint = colour[part == "body"].mean(axis=0)
    dark = colour[part == "trim"].mean(axis=0)
    pane = colour[part == "glass"].mean(axis=0)

    pieces = []
    cab_P, cab_N, index = carmesh.clip(P, N, 0, wall_x, keep_above=True)
    pieces.append((cab_P, cab_N, part[index], colour[index]))
    rear_P, rear_N, index = carmesh.clip(P, N, 0, wall_x, keep_above=False)
    rear_part, rear_colour = part[index], colour[index]
    rear_P, rear_N, index = carmesh.clip(rear_P, rear_N, 2, belt, keep_above=False)
    pieces.append((rear_P, rear_N, rear_part[index], rear_colour[index]))

    def add(triangles, name, tone):
        normals = np.repeat(carmesh.face_normals(triangles)[0][:, None, :], 3, axis=1)
        pieces.append((triangles, normals, np.full(len(triangles), name, dtype=object),
                       np.repeat(np.asarray(tone, dtype=np.float64)[None, :], len(triangles), axis=0)))

    # The cab's back wall: the outline of the cut through the cab, from the sills up.
    points = cab_P.reshape(-1, 3)
    cut = points[(np.abs(points[:, 0] - wall_x) < 1e-9) & (points[:, 2] >= belt - 1e-9)]
    outline = carmesh.hull_2d(cut[:, 1:])
    wall = np.column_stack([np.full(len(outline), wall_x), outline])
    add(carmesh.facing(carmesh.fan(wall), (-1.0, 0.0, 0.0)), "body", paint)
    top, half = float(outline[:, 1].max()), float(np.abs(outline[:, 0]).max())
    low, high = belt + 0.10, top - 0.16
    window = np.array([[wall_x - 0.004, y, z] for y, z in ((-0.62 * half, low), (0.62 * half, low), (0.56 * half, high), (-0.56 * half, high))])
    add(carmesh.facing(carmesh.fan(window), (-1.0, 0.0, 0.0)), "glass", pane)

    # The bed: the outline of the cut along the sills, a rim inside it, sides down to a floor.
    points = rear_P.reshape(-1, 3)
    cut = points[np.abs(points[:, 2] - belt) < 1e-9]
    outline = carmesh.hull_2d(cut[:, :2])
    middle = (outline.min(axis=0) + outline.max(axis=0)) / 2
    size = outline.max(axis=0) - outline.min(axis=0)
    inner = middle + (outline - middle) * (1.0 - 2.0 * RIM / size)
    floor = tyre_top + 0.03
    count = len(outline)
    # Whatever of the SUV is left standing in the bed, above its floor, goes: lamp housings, door linings.
    centre = rear_P.mean(axis=1)
    edge = np.roll(inner, -1, axis=0) - inner
    to_centre = centre[:, None, :2] - inner[None, :, :]
    within = (edge[None, :, 0] * to_centre[:, :, 1] - edge[None, :, 1] * to_centre[:, :, 0] > 0.01 * np.linalg.norm(edge, axis=1)[None, :]).all(axis=1)
    keep = ~(within & (centre[:, 2] > floor))
    pieces[1] = tuple(array[keep] for array in pieces[1])
    upper = np.column_stack([outline, np.full(count, belt)])
    upper_in = np.column_stack([inner, np.full(count, belt)])
    lower_in = np.column_stack([inner, np.full(count, floor)])
    nxt = np.roll(np.arange(count), -1)
    rim = np.concatenate([np.stack([upper, upper[nxt], upper_in[nxt]], axis=1), np.stack([upper, upper_in[nxt], upper_in], axis=1)])
    add(carmesh.facing(rim, (0.0, 0.0, 1.0)), "body", paint)
    sides = np.concatenate([np.stack([upper_in, upper_in[nxt], lower_in[nxt]], axis=1), np.stack([upper_in, lower_in[nxt], lower_in], axis=1)])
    inward = np.append(middle, (belt + floor) / 2) - sides.mean(axis=1)
    inward[:, 2] = 0.0
    add(carmesh.facing(sides, inward), "trim", dark)
    add(carmesh.facing(carmesh.fan(lower_in), (0.0, 0.0, 1.0)), "trim", dark)
    log(f"  pickup: cab ends at x = {wall_x:.3f} (chassis frame), sills at {belt:.2f}, bed "
        f"{size[0]:.2f} x {size[1]:.2f} m and {belt - floor:.2f} deep, floor at {floor:.2f}")

    P, N, part, colour = (np.concatenate([piece[k] for piece in pieces]) for k in range(4))
    if "roof" in options:
        P, N, index = carmesh.clip(P, N, 2, options["roof"], keep_above=False)
        part, colour = part[index], colour[index]
    return P, N, part, colour
