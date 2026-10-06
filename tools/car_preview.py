#!/usr/bin/env python3
"""Look at the parked-vehicle models: line them up beside a 4.5 x 1.8 x 1.4 m box and save pictures.

    python car_preview.py MODELS_DIR OUT_DIR            the line-up, from four sides and above
    python car_preview.py MODELS_DIR OUT_DIR NAME ...   close-ups of the named models, one after another

Runs on stock PyChrono and draws each part the way the scene does: one triangle mesh, one
plain material, back faces culled. Each body gets a different paint. One window, closed
when the pictures are written.
"""
import json
import os
import sys

import pychrono as chrono
import pychrono.vsg3d as vsg

PAINT = [(0.55, 0.03, 0.03), (0.75, 0.75, 0.75), (0.03, 0.12, 0.45), (0.8, 0.45, 0.05), (0.05, 0.25, 0.1),
         (0.02, 0.02, 0.02), (0.3, 0.3, 0.32)]
SPACING = 3.6
BOX = (4.5, 1.8, 1.4)


def add_part(body, path, colour, roughness, metallic, x, y, yaw):
    mesh = chrono.ChTriangleMeshConnected.CreateFromWavefrontFile(path, True, True)
    shape = chrono.ChVisualShapeTriangleMesh()
    shape.SetMesh(mesh, False)
    shape.SetMutable(False)
    material = chrono.ChVisualMaterial()
    material.SetDiffuseColor(chrono.ChColor(*colour))
    material.SetRoughness(roughness)
    material.SetMetallic(metallic)
    shape.AddMaterial(material)
    body.AddVisualShape(shape, chrono.ChFramed(chrono.ChVector3d(x, y, 0.0), chrono.QuatFromAngleZ(yaw)))


def add_model(body, folder, model, paint, x, y, yaw=0.0):
    for part in model["parts"]:
        colour = paint if part["paint"] else part["colour"]
        add_part(body, os.path.join(folder, part["mesh"]), colour, part["roughness"], part["metallic"], x, y, yaw)


def add_box(body, size, colour, x, y, z):
    shape = chrono.ChVisualShapeBox(*size)
    material = chrono.ChVisualMaterial()
    material.SetDiffuseColor(chrono.ChColor(*colour))
    material.SetRoughness(0.9)
    shape.AddMaterial(material)
    body.AddVisualShape(shape, chrono.ChFramed(chrono.ChVector3d(x, y, z), chrono.QUNIT))


def shoot(system, views, fov=30.0):
    """Open one window and save a picture from each (path, eye, target)."""
    vis = vsg.ChVisualSystemVSG()
    vis.AttachSystem(system)
    vis.SetWindowSize(1600, 900)
    vis.SetWindowTitle("car preview")
    vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
    # The camera starts out looking level. Its up vector is fixed when it is added, square to
    # the line of sight, and a later change of view keeps it: starting from a tilted view
    # would roll every picture after the first.
    eye, target = views[0][1], views[0][2]
    vis.AddCamera(chrono.ChVector3d(eye[0], eye[1], target[2]), chrono.ChVector3d(*target))
    vis.SetCameraAngleDeg(fov)
    vis.SetLightIntensity(1.0)
    vis.SetLightDirection(1.5 * chrono.CH_PI_2, chrono.CH_PI_4)
    vis.EnableShadows()
    vis.SetBaseGuiVisibility(False)
    vis.HideLogo()
    vis.Initialize()
    vis.UpdateCamera(chrono.ChVector3d(*eye), chrono.ChVector3d(*target))
    frame, index = 0, 0
    while vis.Run() and index < len(views):
        vis.BeginScene()
        vis.Render()
        vis.EndScene()
        frame += 1
        if frame == 10:
            vis.WriteImageToFile(views[index][0])
        if frame == 12:
            frame, index = 0, index + 1
            if index < len(views):
                vis.UpdateCamera(chrono.ChVector3d(*views[index][1]), chrono.ChVector3d(*views[index][2]))


def main():
    folder, out = sys.argv[1], sys.argv[2]
    wanted = sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    models = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            with open(os.path.join(folder, name)) as f:
                models.append(json.load(f))
    models.sort(key=lambda m: (m["length"] * m["height"], m["name"]))

    system = chrono.ChSystemNSC()
    body = chrono.ChBody()
    body.SetFixed(True)
    body.EnableCollision(False)
    system.Add(body)

    views = []
    if wanted:
        # One model at a time would need a window each, so they stand far apart in one scene.
        chosen = [m for m in models if m["name"] in wanted]
        add_box(body, (30.0 * len(chosen) + 120.0, 120.0, 0.2), (0.45, 0.45, 0.45), 15.0 * (len(chosen) - 1), 0.0, -0.1)
        for k, model in enumerate(chosen):
            cx = 30.0 * k
            add_model(body, folder, model, PAINT[models.index(model) % len(PAINT)], cx, 0.0)
            add_box(body, BOX, (0.6, 0.6, 0.2), cx, 3.2, BOX[2] / 2)
            stem = os.path.join(out, model["name"])
            views += [
                (stem + "_side.png", (cx, -11.0, 1.0), (cx, 0.0, 0.75)),
                (stem + "_front.png", (cx + 6.5, -5.0, 2.0), (cx, 0.3, 0.6)),
                (stem + "_rear.png", (cx - 6.5, -5.0, 2.0), (cx, 0.3, 0.6)),
                (stem + "_low.png", (cx + 4.2, -5.5, 0.45), (cx, 0.0, 0.55)),
                (stem + "_lowrear.png", (cx - 4.6, -4.4, 0.35), (cx, 0.0, 0.6)),
                (stem + "_top.png", (cx + 0.01, -0.6, 13.0), (cx, 0.6, 0.0)),
                (stem + "_wheel.png", (cx + model["wheelbase"] / 2 + 0.9, -3.2, 0.7), (cx + model["wheelbase"] / 2, -0.6, 0.4)),
                (stem + "_shadowside.png", (cx - 5.0, 6.0, 1.6), (cx, -0.6, 0.6)),
            ]
    else:
        add_box(body, (160.0, 160.0, 0.2), (0.45, 0.45, 0.45), 0.0, 0.0, -0.1)
        add_box(body, BOX, (0.6, 0.6, 0.2), 0.0, 0.0, BOX[2] / 2)
        for k, model in enumerate(models):
            add_model(body, folder, model, PAINT[k % len(PAINT)], 0.0, -SPACING * (k + 1))
        span = SPACING * len(models)
        mid = -span / 2
        views = [
            (os.path.join(out, "lineup_front.png"), (1.15 * span, mid - 0.5 * span, 0.4 * span), (0.0, mid, -0.5)),
            (os.path.join(out, "lineup_rear.png"), (-1.15 * span, mid - 0.5 * span, 0.4 * span), (0.0, mid, -0.5)),
            (os.path.join(out, "lineup_nose.png"), (1.4 * span, mid, 3.0), (0.0, mid, -1.5)),
            (os.path.join(out, "lineup_tail.png"), (-1.4 * span, mid, 3.0), (0.0, mid, -1.5)),
            (os.path.join(out, "lineup_low.png"), (0.75 * span, -span - 0.55 * span, 1.4), (0.0, mid - 1.0, 0.8)),
            (os.path.join(out, "lineup_top.png"), (0.01, mid, 1.45 * span), (0.0, mid, 0.0)),
        ]
    shoot(system, views)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
