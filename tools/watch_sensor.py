#!/usr/bin/env python3
"""Watch the scene through Chrono::Sensor's camera, in a window, with the camera moving.

    python watch_sensor.py SENSOR_SCENE [--views gallery_car.json] [--only NAME ...] [--no-edge-lines]

A window opens and shows what a camera at car height sees as it runs forward along each view
in turn, at the speed of a car in a car park, and then starts again. Ctrl-C in the terminal
stops it. Nothing is written.

It is for looking at the world in motion: whether leaves and grain shimmer, how the paint and
the pavement read as they come toward the camera. SENSOR_SCENE is a copy of the scene made by
sensor_scene.py, and what this needs of PyChrono is what look_sensor.py needs.

The window shows the ray tracer's own picture. The camera model (cameramodel.py) works on
pictures after they are drawn, and look_sensor.py --camera applies it to the ones it writes.
Here only the camera's size and field of view are taken from it.

The ray tracer takes the scene in before it shows anything: about 10 s for a plain copy and
40 s for one made with --detail. The window opens after that.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import cameramodel  # noqa: E402
import sensorstage  # noqa: E402

RATE = 30.0          # pictures a second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--views", default=os.path.join(here, "gallery_car.json"), help="a file of named views: each is where a run starts and which way it goes")
    parser.add_argument("--only", nargs="+", default=None, help="names of the views to run along, out of the file's")
    parser.add_argument("--camera", choices=sorted(cameramodel.CAMERAS), default="zedx_2mm", help="the camera whose size and field of view the window has")
    parser.add_argument("--speed", type=float, default=5.0, help="metres a second the camera runs forward at")
    parser.add_argument("--seconds", type=float, default=6.0, help="how long it runs along each view")
    parser.add_argument("--samples", type=int, default=1, help="rays per pixel, each way. A real camera takes one")
    parser.add_argument("--sun", type=float, default=180.0, help="compass bearing the sun stands at, in degrees")
    parser.add_argument("--sky", default="sunflowers_4k", help="one of Chrono's sky pictures by name, or the path of a .hdr file")
    parser.add_argument("--no-edge-lines", action="store_true", help="leave out the white edge lines, which the real track does not have")
    args = parser.parse_args()

    chrono, sens = sensorstage.modules()
    with open(args.views) as f:
        views = {name: view["pose"] for name, view in json.load(f)["views"].items() if args.only is None or name in args.only}
    if not views:
        sys.exit("no views to run along")
    start = time.perf_counter()
    folder = tempfile.mkdtemp(prefix="watch_sensor_")
    turned = os.path.join(folder, "sky.hdr")
    system, holder, manager, _ = sensorstage.stage(args.scene, turned, args.sun, args.sky, edge_lines=not args.no_edge_lines)
    ground = sensorstage.Ground(args.scene)

    def pose(view, run):
        """The camera run metres along a view: both its eye and what it looks at move forward, each staying its own height above the ground."""
        ex, ey, eh, tx, ty, th = view
        reach = math.hypot(tx - ex, ty - ey)
        dx, dy = (tx - ex) / reach * run, (ty - ey) / reach * run
        eye = [ex + dx, ey + dy, (ground.height(ex + dx, ey + dy) or 295.0) + eh]
        return sensorstage.frame(chrono, eye, [tx + dx, ty + dy, (ground.height(tx + dx, ty + dy) or eye[2] - eh) + th])

    model = cameramodel.CAMERAS[args.camera]
    wide, tall = model["size"]
    first = next(iter(views.values()))
    camera = sens.ChCameraSensor(holder, RATE, pose(first, 0.0), wide, tall, math.radians(model["hfov"]), args.samples)
    camera.SetLag(0)
    camera.SetCollectionWindow(0)
    camera.PushFilter(sens.ChFilterVisualize(wide, tall, "Columbus 151 Speedway through Chrono::Sensor"))
    manager.AddSensor(camera)
    print(f"[{time.perf_counter() - start:5.1f} s] scene loaded. The ray tracer now takes it in, which shows nothing for 10 to 40 s."
          " Then the window opens. Ctrl-C here stops it.", flush=True)

    shown = False
    try:
        while True:
            for name, view in views.items():
                began = time.perf_counter()
                tick = 0
                while tick / RATE < args.seconds:
                    camera.SetOffsetPose(pose(view, args.speed * tick / RATE))
                    manager.Update()
                    system.DoStepDynamics(1.0 / RATE)
                    if not shown:
                        shown = True
                        began = time.perf_counter()        # the first picture took as long as the scene did
                        print(f"[{time.perf_counter() - start:5.1f} s] the window is open", flush=True)
                    tick += 1
                    ahead = began + tick / RATE - time.perf_counter()
                    if ahead > 0:
                        time.sleep(ahead)                   # no faster than the clock
                took = time.perf_counter() - began
                print(f"        {name}: {args.seconds * args.speed:.0f} m at {args.speed:g} m/s, {tick / took:.0f} pictures a second", flush=True)
    except KeyboardInterrupt:
        print("stopped", flush=True)
    os.remove(turned)
    os.rmdir(folder)
    os._exit(0)       # the renderer's threads do not let the interpreter leave on its own


if __name__ == "__main__":
    main()
