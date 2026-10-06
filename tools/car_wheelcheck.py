#!/usr/bin/env python3
"""Check the parked-vehicle models' wheels against the vehicles PyChrono builds.

    python car_wheelcheck.py CHRONO_DATA_DIR MODELS_DIR

carmodels.py reads the spindle positions out of the vehicles' JSON files itself, so that
building the models needs no PyChrono. This builds each vehicle with PyChrono's own
WheeledVehicle, initialises it, asks it where its spindles are, and compares:

  1. the positions read from the JSON files with PyChrono's, in the chassis frame;
  2. the wheel centres written into the model with PyChrono's, as distances between wheels,
     which do not depend on the frame (a model whose spec lifts an axle differs there, by
     that much, and says so);
  3. the model's tyres with the ground: every wheel centre one tyre radius up, the lowest
     tyre vertex on z = 0, the two sides mirror images, and the footprint centred.

Exits with status 1 if anything is off by more than a tenth of a millimetre.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carmesh  # noqa: E402
import carspecs  # noqa: E402

TOLERANCE = 1e-4


def chrono_spindles(data_dir, vehicle_json):
    """{(axle, side): (x, y, z)} in the chassis frame, from a vehicle PyChrono has built."""
    import pychrono as chrono
    import pychrono.vehicle as veh

    veh.SetVehicleDataPath(os.path.join(data_dir, "vehicle", ""))
    vehicle = veh.WheeledVehicle(veh.GetVehicleDataFile(vehicle_json), chrono.ChContactMethod_NSC)
    vehicle.Initialize(chrono.ChCoordsysd(chrono.ChVector3d(3.0, -2.0, 1.5), chrono.QuatFromAngleZ(0.4)))
    frame = vehicle.GetChassisBody().GetFrameRefToAbs()
    out = {}
    for axle in range(vehicle.GetNumberAxles()):
        for side, which in ((0, veh.LEFT), (1, veh.RIGHT)):
            p = frame.TransformPointParentToLocal(vehicle.GetSpindlePos(axle, which))
            out[(axle, side)] = (p.x, p.y, p.z)
    return out


def spans(points):
    """Distances between every pair of points, in a fixed order."""
    return np.array([np.linalg.norm(np.subtract(a, b)) for a, b in itertools.combinations(points, 2)])


def main():
    data_dir, models_dir = sys.argv[1], sys.argv[2]
    bad = 0
    for spec in carspecs.MODELS:
        print(spec["name"])
        from_json = carspecs.wheel_positions(data_dir, spec["vehicle"])
        from_chrono = chrono_spindles(data_dir, spec["vehicle"])
        keys = sorted(from_chrono)
        worst = max(np.abs(np.subtract(from_json[k], from_chrono[k])).max() for k in keys) if sorted(from_json) == keys else np.inf
        print(f"  spindles read from JSON against PyChrono's, chassis frame: largest difference {worst * 1000:.4f} mm")
        for k in keys:
            print(f"    axle {k[0]} {'left ' if k[1] == 0 else 'right'}  JSON {np.round(from_json[k], 4)}  PyChrono {np.round(from_chrono[k], 4)}")
        bad += worst > TOLERANCE

        with open(os.path.join(models_dir, spec["name"] + ".json")) as f:
            model = json.load(f)
        centres = np.array(model["wheel_centres"])
        lifted = dict(from_chrono)
        for axle, lift in spec.get("spindle_lift", {}).items():
            for side in (0, 1):
                x, y, z = lifted[(axle, side)]
                lifted[(axle, side)] = (x, y, z + lift)
        off = np.abs(spans(centres) - spans([lifted[k] for k in keys])).max()
        note = f" (with axle lift {spec['spindle_lift']} applied, a deliberate departure from Chrono)" if "spindle_lift" in spec else ""
        print(f"  model's wheel centres against PyChrono's, as distances between wheels: largest difference {off * 1000:.4f} mm{note}")
        bad += off > 2 * TOLERANCE          # the model's numbers are rounded to 0.1 mm

        radius = model["tyre_radius"]
        tyres = [p for p in model["parts"] if p["name"] == "tyres"][0]
        P = carmesh.load_obj(os.path.join(models_dir, tyres["mesh"]))["P"].reshape(-1, 3)
        height = np.abs(centres[:, 2] - radius).max()
        mirror = max(np.abs(centres[0::2, 1] + centres[1::2, 1]).max(), np.abs(centres[0::2, [0, 2]] - centres[1::2, [0, 2]]).max())
        print(f"  tyre radius {radius:.4f}: wheel centres off that height by {height * 1000:.4f} mm, "
              f"lowest tyre vertex at z = {P[:, 2].min() * 1000:.4f} mm, left and right differ by {mirror * 1000:.4f} mm")
        bad += height > TOLERANCE or abs(P[:, 2].min()) > TOLERANCE or mirror > 2 * TOLERANCE
        everything = np.concatenate([carmesh.load_obj(os.path.join(models_dir, p["mesh"]))["P"].reshape(-1, 3) for p in model["parts"]])
        low, high = everything.min(axis=0), everything.max(axis=0)
        print(f"  footprint centre ({(low[0] + high[0]) / 2 * 1000:.3f}, {(low[1] + high[1]) / 2 * 1000:.3f}) mm, "
              f"size {high[0] - low[0]:.3f} x {high[1] - low[1]:.3f} x {high[2]:.3f}, lowest point z = {low[2] * 1000:.3f} mm")
        # In y the origin is the centre line, which a mirror can put a few millimetres off the middle.
        bad += abs(low[0] + high[0]) > 4 * TOLERANCE or abs(low[1] + high[1]) > 0.01 or low[2] < -TOLERANCE
    print("all checks passed" if not bad else f"{bad} CHECKS FAILED")
    sys.stdout.flush()
    os._exit(1 if bad else 0)


if __name__ == "__main__":
    main()
