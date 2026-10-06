#!/usr/bin/env python3
"""Render the scene from a fixed camera and save a picture. For checking a build by eye.

    python look.py SCENE_DIR OUT.png  EYE_X EYE_Y EYE_ABOVE_GROUND  TARGET_X TARGET_Y TARGET_ABOVE_GROUND [options]

Heights are metres above the ground at that spot. Runs on stock PyChrono, through the same
loader the published script uses.
"""
import argparse
import os
import sys

# speedway.py sits one directory up in the published repository.
here = os.path.dirname(os.path.abspath(__file__))
for candidate in (os.path.join(here, ".."), os.path.join(here, "..", "repo")):
    if os.path.isfile(os.path.join(candidate, "speedway.py")):
        sys.path.insert(0, candidate)
import speedway  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("out")
    parser.add_argument("pose", type=float, nargs=6)
    parser.add_argument("--textures", default="standard")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--groups", default=None)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--sun", type=float, nargs=2, default=None, metavar=("AZIMUTH", "ELEVATION"), help="light direction in degrees")
    args = parser.parse_args()

    import pychrono as chrono
    import pychrono.vsg3d as vsg

    system = chrono.ChSystemNSC()
    speedway.add_scenery(system, args.scene, args.textures, groups=args.groups.split(",") if args.groups else None, verbose=False)
    ex, ey, eh, tx, ty, th = args.pose
    ez = (speedway.ground_height(args.scene, ex, ey, radius=8.0) or 295.0) + eh
    tz = (speedway.ground_height(args.scene, tx, ty, radius=8.0) or 295.0) + th

    vis = vsg.ChVisualSystemVSG()
    vis.AttachSystem(system)
    vis.SetWindowSize(1600, 900)
    vis.SetWindowTitle("look")
    vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
    vis.AddCamera(chrono.ChVector3d(ex, ey, ez), chrono.ChVector3d(tx, ty, tz))
    if args.fov:
        vis.SetCameraAngleDeg(args.fov)
    vis.SetLightIntensity(1.0)
    if args.sun:
        vis.SetLightDirection(chrono.CH_DEG_TO_RAD * args.sun[0], chrono.CH_DEG_TO_RAD * args.sun[1])
    else:
        vis.SetLightDirection(1.5 * chrono.CH_PI_2, chrono.CH_PI_4)
    if not args.no_shadows:
        vis.EnableShadows()
    vis.EnableSkyTexture()
    vis.Initialize()
    frames = 0
    while vis.Run() and frames < 12:
        vis.BeginScene()
        vis.Render()
        vis.EndScene()
        system.DoStepDynamics(1e-3)
        frames += 1
        if frames == 10:
            vis.WriteImageToFile(args.out)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
