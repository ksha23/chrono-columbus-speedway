#!/usr/bin/env python3
"""Time the renderer on the scene from fixed cameras: frames per second with nothing else going on.

    python fps_probe.py SCENE_DIR [--no-shadows] [--groups A,B] [--single-sided]

Prints one line per camera. Runs on stock PyChrono through the published loader.
"""
import argparse
import os
import sys
import time

# speedway.py sits one directory up in the published repository.
here = os.path.dirname(os.path.abspath(__file__))
for candidate in (os.path.join(here, ".."), os.path.join(here, "..", "repo")):
    if os.path.isfile(os.path.join(candidate, "speedway.py")):
        sys.path.insert(0, candidate)
import speedway  # noqa: E402

CAMERAS = {
    "start pose, along the straight": ((-56, -85, 2.2), (40, -5, 1.0)),
    "skid pad, facing the open pad": ((120, 50, 2.2), (200, 130, 1.0)),
    "skid pad, facing the woods": ((150, 80, 2.2), (60, -60, 1.0)),
    "overview from the south": ((60, -330, 200), (60, 20, 0)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--groups", default=None)
    parser.add_argument("--single-sided", action="store_true", help="draw leaves from one side only, to see what both sides cost")
    parser.add_argument("--only", type=int, default=None, help="index of the one camera to time")
    args = parser.parse_args()

    import pychrono as chrono
    import pychrono.vsg3d as vsg

    if args.single_sided:
        real = chrono.ChVisualShapeTriangleMesh.SetDoubleFaced
        chrono.ChVisualShapeTriangleMesh.SetDoubleFaced = lambda self, on: real(self, False)

    names = list(CAMERAS)
    if args.only is not None:
        names = [names[args.only]]
    name = names[0]
    system = chrono.ChSystemNSC()
    speedway.add_scenery(system, args.scene, groups=args.groups.split(",") if args.groups else None, verbose=False)
    (ex, ey, eh), (tx, ty, th) = CAMERAS[name]
    ground = lambda x, y: speedway.ground_height(args.scene, x, y, radius=8.0) or 295.0
    vis = vsg.ChVisualSystemVSG()
    vis.AttachSystem(system)
    vis.SetWindowSize(1600, 900)
    vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
    vis.AddCamera(chrono.ChVector3d(ex, ey, ground(ex, ey) + eh), chrono.ChVector3d(tx, ty, ground(tx, ty) + th))
    vis.SetLightIntensity(1.0)
    vis.SetLightDirection(1.5 * chrono.CH_PI_2, chrono.CH_PI_4)
    if not args.no_shadows:
        vis.EnableShadows()
    vis.EnableSkyTexture()
    vis.Initialize()
    frames, start = 0, None
    while vis.Run() and frames < 160:
        vis.BeginScene()
        vis.Render()
        vis.EndScene()
        frames += 1
        if frames == 60:
            start = time.perf_counter()
    rate = 100 / (time.perf_counter() - start)
    print(f"  {name:34s} {rate:6.1f} frames/s  ({1000 / rate:.1f} ms a frame)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
