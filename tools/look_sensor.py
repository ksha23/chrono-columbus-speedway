#!/usr/bin/env python3
"""Draw the scene with Chrono::Sensor's ray-traced camera, as an alternative to Chrono::VSG.

    python look_sensor.py SENSOR_SCENE OUT_DIR [--views gallery.json] [--only NAME ...]
    python look_sensor.py SENSOR_SCENE OUT_DIR --view NAME EYE_X EYE_Y EYE_ABOVE_GROUND TARGET_X TARGET_Y TARGET_ABOVE_GROUND

It writes pictures and opens no window: one PNG per view, into OUT_DIR. watch_sensor.py
opens a window and moves the camera.

SENSOR_SCENE is a copy of the scene made by sensor_scene.py: the scene as downloaded is stored
for Chrono::VSG and comes out far too dark here. Views are those of look.py: an eye and a
target, each a place and a height above the ground there.

This does not run on the PyChrono the demo runs on. It needs one built with the Sensor module
and a ray tracing backend, and the conda package for macOS has neither. The pictures in the
README were drawn on an M4 Pro with the Metal backend, through Python bindings for its camera
that are not in Chrono yet.

With --camera the picture is what a camera of that kind would have written, not the light
itself: see cameramodel.py. --no-edge-lines leaves out the edge lines the real track lacks.
The light and the sky are sensorstage.py's.
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
sys.path.insert(0, here)
import cameramodel  # noqa: E402
import sensorstage  # noqa: E402
from sensorstage import speedway  # noqa: E402

VERTICAL_FOV = 40.0  # degrees, what Chrono::VSG's camera has unless told otherwise
RATE = 20.0         # pictures a second the camera is asked for. Only the last one of a view is kept
HEADROOM = 0.6      # the share of the light a picture is drawn in when a camera model is to expose it


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
    parser.add_argument("--light", type=float, nargs=2, default=(sensorstage.SUN, sensorstage.AMBIENT), metavar=("SUN", "AMBIENT"), help="strength of the sun and of the ambient light")
    parser.add_argument("--camera", choices=sorted(cameramodel.CAMERAS), default=None,
                        help="draw as this camera would: its size and field of view, and its lens, exposure, noise and sharpening (cameramodel.py)")
    parser.add_argument("--seed", type=int, default=0, help="for the camera's noise")
    parser.add_argument("--no-edge-lines", action="store_true", help="leave out the white edge lines, which the real track does not have")
    args = parser.parse_args()

    chrono, sens = sensorstage.modules()
    if args.view:
        views = {args.view[0]: {"pose": [float(x) for x in args.view[1:]], **({"fov": args.fov} if args.fov else {})}}
    else:
        with open(args.views) as f:
            views = {name: view for name, view in json.load(f)["views"].items() if args.only is None or name in args.only}
    if not views:
        sys.exit("no views to draw")
    os.makedirs(args.out, exist_ok=True)
    start = time.perf_counter()

    camera_model = cameramodel.CAMERAS[args.camera] if args.camera else None
    sun, ambient = args.light
    if camera_model:
        args.size = camera_model["size"]
        # A camera sets its own exposure. What it is given must not have run out of range
        # before that, and sunlit concrete at full light does.
        sun, ambient = sun * HEADROOM, ambient * HEADROOM
    turned = os.path.join(args.out, "_sky.hdr")
    system, holder, manager, elevation = sensorstage.stage(args.scene, turned, args.sun, args.sky, (sun, ambient), not args.no_edge_lines)

    def pose(ex, ey, eh, tx, ty, th):
        return sensorstage.frame(chrono, [ex, ey, (speedway.ground_height(args.scene, ex, ey, radius=8.0) or 295.0) + eh],
                                 [tx, ty, (speedway.ground_height(args.scene, tx, ty, radius=8.0) or 295.0) + th])

    # A camera's field of view is fixed when it is made: one camera for each that is asked for.
    wide, tall = args.size
    cameras = {}
    for view in views.values():
        # A camera has its own field of view, whatever the view asks for.
        view["fov"] = cameramodel.vertical_fov(camera_model) if camera_model else float(view.get("fov", VERTICAL_FOV))
        fov = view["fov"]
        if fov not in cameras:
            across = 2 * math.atan(math.tan(math.radians(fov) / 2) * wide / tall)
            camera = sens.ChCameraSensor(holder, RATE, pose(*view["pose"]), wide, tall, across, args.samples)
            camera.SetLag(0)
            camera.SetCollectionWindow(0)
            camera.PushFilter(sens.ChFilterRGBA8Access())
            manager.AddSensor(camera)
            cameras[fov] = camera
    print(f"[{time.perf_counter() - start:5.1f} s] scene loaded, sun at bearing {args.sun:.0f} and {elevation:.0f} degrees up", flush=True)
    print(f"        the ray tracer now takes the scene in, which shows nothing for 10 to 40 s. Then {len(views)} picture{'s' if len(views) != 1 else ''}"
          f" {'is' if len(views) == 1 else 'are'} written to {args.out}. No window opens: watch_sensor.py has one.", flush=True)

    noise = np.random.default_rng(args.seed)
    try:
        failed = draw(views, cameras, pose, manager, system, camera_model, noise, args.out, start)
    except KeyboardInterrupt:
        print("stopped", flush=True)
        failed = list(views)
    os.remove(turned)
    print(f"{len(views) - len(failed)} of {len(views)} pictures written to {args.out}", flush=True)
    os._exit(1 if failed else 0)       # the renderer's threads do not let the interpreter leave on its own


def draw(views, cameras, pose, manager, system, camera_model, noise, out, start):
    """Draw every view and write it. Returns the names of those no picture came for."""
    failed = []
    for name, view in views.items():
        camera = cameras[view["fov"]]
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
        if camera_model:
            image = cameramodel.develop(image, camera_model, noise)
        write_png(os.path.join(out, name + ".png"), image)
        print(f"[{time.perf_counter() - start:5.1f} s] {name}", flush=True)
    return failed


if __name__ == "__main__":
    main()
