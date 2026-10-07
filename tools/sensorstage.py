"""The scene set up for Chrono::Sensor's camera: loaded, lit, and under its sky.

What look_sensor.py, which writes pictures, and watch_sensor.py, which shows a window, both
start from.

The light is one sun and an even ambient light, set so that flat ground in the sun comes out
about as bright as Chrono::VSG shows it. The sky is one of the pictures Chrono ships, turned so
that its sun stands where the light comes from (sky.py). The sun's height is the picture's own.
"""
import math
import os
import sys

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
SUN_TINT = (1.04, 1.00, 0.92)   # sunlight is a little warm and the sky's light is blue, so a shadow is
SKY_TINT = (0.90, 1.00, 1.18)   # bluer than the ground beside it. Together, in the sun, they come out neutral
GAMMA = 2.2         # the camera's own: what it shows is the light it gathered to the power of 1 / 2.2
NEEDS = ("This PyChrono has no Chrono::Sensor camera. It takes one built from source with the Sensor module and a ray"
         " tracing backend, and Python bindings for its camera: see the top of look_sensor.py.")


def modules():
    """(pychrono, pychrono.sensor), or leave with a word on what is missing."""
    import pychrono as chrono
    try:
        import pychrono.sensor as sens
        sens.ChCameraSensor
    except (ImportError, AttributeError):
        sys.exit(NEEDS)
    return chrono, sens


def frame(chrono, eye, target):
    """Where a camera is and which way it looks, from the point it is at and a point it looks at."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    ahead = (target - eye) / np.linalg.norm(target - eye)
    turn = chrono.QuatFromAngleZ(math.atan2(ahead[1], ahead[0])) * chrono.QuatFromAngleY(-math.asin(ahead[2]))
    return chrono.ChFramed(chrono.ChVector3d(*eye), turn)


def stage(scene, sky_file, bearing=180.0, sky_name="sunflowers_4k", light=(SUN, AMBIENT), edge_lines=True):
    """Load a sensor copy of the scene and light it: (system, a fixed body to hang cameras on, sensor manager, the sun's height in degrees).

    sky_file is where the turned sky picture is written: the caller removes it when done.
    bearing is the compass bearing the sun stands at.
    """
    chrono, sens = modules()
    system = chrono.ChSystemNSC()
    groups = None if edge_lines else [group for group in speedway.manifest(scene)["labels"] if group != "EdgeLines"]
    speedway.add_scenery(system, scene, "standard", groups=groups, verbose=False)
    holder = chrono.ChBody()
    holder.SetFixed(True)
    system.Add(holder)

    # The sky, turned. A column of the picture is a direction: the middle column is east, and
    # directions go round anticlockwise seen from above as columns go right.
    source = sky_name if os.path.isfile(sky_name) else os.path.join(chrono.GetChronoDataPath(), "sensor", "textures", sky_name + ".hdr")
    picture = sky.read(source)
    column, row = sky.sun(picture)
    height, width = picture.shape[:2]
    toward = (90.0 - bearing) % 360.0                            # degrees anticlockwise from east
    elevation = 90.0 - (row + 0.5) * 180.0 / height
    picture = sky.turned(picture, (toward + 180.0) / 360.0 * width - column)
    # Under its horizon the picture has the ground of wherever it was taken. Ours takes its place.
    # The scene gives that colour as seen, and the picture holds light: hence the power.
    beyond = speedway.manifest(scene).get("beyond")
    if beyond:
        picture = sky.grounded(picture, [c ** GAMMA for c in beyond])
    sky.write(sky_file, picture)

    manager = sens.ChSensorManager(system)
    sun, ambient = light
    manager.scene.AddDirectionalLight(chrono.ChColor(*(sun * c for c in SUN_TINT)), math.radians(elevation), math.radians(toward))
    manager.scene.SetAmbientLight(chrono.ChVector3f(*(ambient * c for c in SKY_TINT)))
    background = sens.Background()
    background.mode = sens.BackgroundMode_ENVIRONMENT_MAP
    background.env_tex = sky_file
    manager.scene.SetBackground(background)
    return system, holder, manager, elevation


class Ground:
    """The ground's height anywhere on the scan, quickly: for a camera that moves.

    Under the scan the ground mesh has a vertex at every whole metre. They are read once, and
    a height is found between the four round a point.
    """

    def __init__(self, scene):
        self.z = {}
        with open(os.path.join(scene, speedway.GROUND)) as f:
            for line in f:
                if line.startswith("v "):
                    x, y, z = (float(v) for v in line.split()[1:4])
                    if x == int(x) and y == int(y):
                        self.z[(int(x), int(y))] = z

    def height(self, x, y):
        """The ground's height at a point, or None off the metre lattice."""
        i, j = math.floor(x), math.floor(y)
        corners = [self.z.get((i + a, j + b)) for a in (0, 1) for b in (0, 1)]
        if None in corners:
            return None
        u, v = x - i, y - j
        return (corners[0] * (1 - u) * (1 - v) + corners[1] * (1 - u) * v + corners[2] * u * (1 - v) + corners[3] * u * v)
