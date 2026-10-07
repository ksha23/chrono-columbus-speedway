#!/usr/bin/env python3
"""Draw the scene with Chrono::Sensor's ray-traced camera, as an alternative to Chrono::VSG.

    python look_sensor.py SENSOR_SCENE OUT_DIR [--views gallery.json] [--only NAME ...]
    python look_sensor.py SENSOR_SCENE OUT_DIR --view NAME EYE_X EYE_Y EYE_ABOVE_GROUND TARGET_X TARGET_Y TARGET_ABOVE_GROUND

SENSOR_SCENE is a copy of the scene made by sensor_scene.py: the scene as downloaded is stored
for Chrono::VSG and comes out far too dark here. Views are those of look.py: an eye and a
target, each a place and a height above the ground there. One PNG is written per view.

This does not run on the PyChrono the demo runs on. It needs one built with the Sensor module
and a ray tracing backend, and the conda package for macOS has neither. The pictures in the
README were drawn on an M4 Pro with the Metal backend, through Python bindings for its camera
that are not in Chrono yet.

The light is one sun and an even ambient light, set so that flat ground in the sun comes out
about as bright as Chrono::VSG shows it. The sky is one of the pictures Chrono ships, turned so that its
sun stands where the light comes from (sky.py). The sun's height is the picture's own.
"""
import argparse
import json
import math
import os
import struct
import sys
import time
import zlib

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
for candidate in (os.path.join(here, ".."), os.path.join(here, "..", "repo")):
    if os.path.isfile(os.path.join(candidate, "speedway.py")):
        sys.path.insert(0, candidate)
sys.path.insert(0, here)
import sky  # noqa: E402
import speedway  # noqa: E402

SUN = 1.10          # the sun's strength and the ambient light's, set by comparing pictures: sunlit
AMBIENT = 0.30      # ground then shows within 6% of what Chrono::VSG shows, and shadow at about a third of it
VERTICAL_FOV = 40.0  # degrees, what Chrono::VSG's camera has unless told otherwise
RATE = 20.0         # pictures a second the camera is asked for. Only the last one of a view is kept
GAMMA = 2.2         # the camera's own: what it shows is the light it gathered to the power of 1 / 2.2


def write_png(path, rgb):
    """An RGB picture as a PNG, with nothing but the standard library."""
    height, width, _ = rgb.shape
    rows = b"".join(b"\x00" + rgb[r].tobytes() for r in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("out")
    parser.add_argument("--views", default=os.path.join(here, "gallery.json"), help="a file of named views")
    parser.add_argument("--only", nargs="+", default=None, help="names of the views to draw, out of the file's")
    parser.add_argument("--view", nargs=7, default=None, metavar=("NAME", "EX", "EY", "EH", "TX", "TY", "TH"), help="one view given here, and not the file's")
    parser.add_argument("--fov", type=float, default=None, help="vertical field of view in degrees, for --view")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--samples", type=int, default=3, help="rays per pixel, each way")
    parser.add_argument("--sun", type=float, default=180.0, help="compass bearing the sun stands at, in degrees")
    parser.add_argument("--sky", default="sunflowers_4k", help="one of Chrono's sky pictures by name, or the path of a .hdr file")
    parser.add_argument("--light", type=float, nargs=2, default=(SUN, AMBIENT), metavar=("SUN", "AMBIENT"), help="strength of the sun and of the ambient light")
    args = parser.parse_args()

    import pychrono as chrono
    try:
        import pychrono.sensor as sens
        sens.ChCameraSensor
    except (ImportError, AttributeError):
        sys.exit("This PyChrono has no Chrono::Sensor camera. See the top of this file for what is needed.")

    if args.view:
        views = {args.view[0]: {"pose": [float(x) for x in args.view[1:]], **({"fov": args.fov} if args.fov else {})}}
    else:
        with open(args.views) as f:
            views = {name: view for name, view in json.load(f)["views"].items() if args.only is None or name in args.only}
    if not views:
        sys.exit("no views to draw")
    os.makedirs(args.out, exist_ok=True)
    start = time.perf_counter()

    system = chrono.ChSystemNSC()
    speedway.add_scenery(system, args.scene, "standard", verbose=False)
    holder = chrono.ChBody()
    holder.SetFixed(True)
    system.Add(holder)

    def pose(ex, ey, eh, tx, ty, th):
        eye = np.array([ex, ey, (speedway.ground_height(args.scene, ex, ey, radius=8.0) or 295.0) + eh])
        target = np.array([tx, ty, (speedway.ground_height(args.scene, tx, ty, radius=8.0) or 295.0) + th])
        ahead = (target - eye) / np.linalg.norm(target - eye)
        turn = chrono.QuatFromAngleZ(math.atan2(ahead[1], ahead[0])) * chrono.QuatFromAngleY(-math.asin(ahead[2]))
        return chrono.ChFramed(chrono.ChVector3d(*eye), turn)

    # The sky, turned. A column of the picture is a direction: the middle column is east, and
    # directions go round anticlockwise seen from above as columns go right.
    source = args.sky if os.path.isfile(args.sky) else os.path.join(chrono.GetChronoDataPath(), "sensor", "textures", args.sky + ".hdr")
    picture = sky.read(source)
    column, row = sky.sun(picture)
    height, width = picture.shape[:2]
    toward = (90.0 - args.sun) % 360.0                           # degrees anticlockwise from east
    elevation = 90.0 - (row + 0.5) * 180.0 / height
    turned = os.path.join(args.out, "_sky.hdr")
    picture = sky.turned(picture, (toward + 180.0) / 360.0 * width - column)
    # Under its horizon the picture has the ground of wherever it was taken. Ours takes its place.
    # The scene gives that colour as seen, and the picture holds light: hence the power.
    beyond = speedway.manifest(args.scene).get("beyond")
    if beyond:
        picture = sky.grounded(picture, [c ** GAMMA for c in beyond])
    sky.write(turned, picture)

    manager = sens.ChSensorManager(system)
    sun, ambient = args.light
    manager.scene.AddDirectionalLight(chrono.ChColor(sun, sun, sun), math.radians(elevation), math.radians(toward))
    manager.scene.SetAmbientLight(chrono.ChVector3f(ambient, ambient, ambient))
    background = sens.Background()
    background.mode = sens.BackgroundMode_ENVIRONMENT_MAP
    background.env_tex = turned
    manager.scene.SetBackground(background)

    # A camera's field of view is fixed when it is made: one camera for each that is asked for.
    wide, tall = args.size
    cameras = {}
    for view in views.values():
        fov = float(view.get("fov", VERTICAL_FOV))
        if fov not in cameras:
            across = 2 * math.atan(math.tan(math.radians(fov) / 2) * wide / tall)
            camera = sens.ChCameraSensor(holder, RATE, pose(*view["pose"]), wide, tall, across, args.samples)
            camera.SetLag(0)
            camera.SetCollectionWindow(0)
            camera.PushFilter(sens.ChFilterRGBA8Access())
            manager.AddSensor(camera)
            cameras[fov] = camera
    print(f"[{time.perf_counter() - start:5.1f} s] scene loaded, sun at bearing {args.sun:.0f} and {elevation:.0f} degrees up", flush=True)

    failed = []
    for name, view in views.items():
        camera = cameras[float(view.get("fov", VERTICAL_FOV))]
        camera.SetOffsetPose(pose(*view["pose"]))
        moved, image = system.GetChTime(), None
        for _ in range(600):
            manager.Update()
            system.DoStepDynamics(1.0 / RATE)
            buffer = camera.GetMostRecentRGBA8Buffer()
            # A picture started at least two ticks after the camera moved: none from before it.
            if buffer.HasData() and buffer.TimeStamp > moved + 2.0 / RATE:
                image = np.ascontiguousarray(np.flipud(buffer.GetRGBA8Data())[..., :3])      # the buffer's first row is the bottom one
                break
            time.sleep(0.02)
        if image is None:
            failed.append(name)
            print(f"[{time.perf_counter() - start:5.1f} s] {name}: no picture came", flush=True)
            continue
        write_png(os.path.join(args.out, name + ".png"), image)
        print(f"[{time.perf_counter() - start:5.1f} s] {name}", flush=True)
    os.remove(turned)
    sys.stdout.flush()
    os._exit(1 if failed else 0)       # the renderer's threads do not let the interpreter leave on its own


if __name__ == "__main__":
    main()
