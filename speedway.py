#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Drive an Audi around Columbus 151 Speedway, a vehicle test track in Wisconsin, in PyChrono.

    conda install projectchrono::pychrono -c conda-forge    # once, from the projectchrono channel
    python speedway.py

The first run downloads the scene into scene/ beside this file and checks its hash. After that
it starts straight away. Nothing is converted and Chrono is not modified: this needs only a
stock PyChrono with the vehicle and VSG modules.

Controls: hold W to accelerate and S to brake, hold A or D to steer. Let go and the car coasts
and the wheel centres, as in a driving game. --keys step gives Chrono's older behaviour, where
each press nudges an input and it stays there.

To put the track in a simulation of your own, import this file:

    import speedway
    scene = speedway.fetch()                      # scene directory, downloaded if it is not there
    speedway.add_scenery(system, scene)           # what you see: visual shapes, no collision
    terrain = speedway.add_ground(system, scene)  # what the wheels touch: one collision mesh
    z = speedway.ground_height(scene, x, y)       # ground height, usable before the first step

The ground is the USGS 3DEP lidar survey of the site. Its picture, and where the trees and
buildings stand, come from a drone scan. See the README for how they were put together.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request

# The published scene: one pinned archive, checked against its hash before anything is unpacked,
# so a changed or truncated download fails here and not later as a half-loaded scene.
RELEASE = "https://github.com/ksha23/chrono-columbus-speedway/releases/download/v5/"
SCENE_URL = RELEASE + "speedway_scene_base.tar.gz"
SCENE_SHA256 = "fb71f9537b78a1459f7ef0499b8f1c58f43eee791c196ca8f0f0933cbc466b02"
SCENE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene")

MANIFEST = "speedway_scene.json"
# The collision surface, and the same triangles as two meshes that share their boundary: the
# pavement and everything else.
GROUND = "speedway_ground.obj"
ROAD = "speedway_road.obj"
TERRAIN = "speedway_terrain.obj"
MARKER = ".speedway-scene"

# Ground photo detail: what one texture pixel covers, in metres. Both levels come with the
# scene. The drone photographed the site at about 2 cm.
TEXTURES = {"low": 0.10, "standard": 0.05}

# What the scenery is made of. add_scenery(groups=...) takes any of these.
GROUPS = ["Road", "Terrain", "Buildings", "Trees", "Cones", "Poles", "Vehicles", "Barriers", "Props", "Rocks", "Markings", "EdgeLines"]

# The groups the demo can leave out, each with a --no-... option: option, group, what it is.
OPTIONAL = [
    ("trees", "Trees", "the trees"),
    ("cones", "Cones", "the traffic cones"),
    ("poles", "Poles", "the light poles"),
    ("vehicles", "Vehicles", "the parked vehicles"),
    ("barriers", "Barriers", "the guard rails, the fence and the swing gates"),
    ("props", "Props", "the small structures: tanks, a hut, bleachers, the lattice tower"),
    ("rocks", "Rocks", "the rock piles and boulders"),
    ("markings", "Markings", "the road paint, for bare concrete to lay out lanes of your own"),
    ("edge-lines", "EdgeLines", "the white lines along the road edges. They are an addition: the real track has none"),
]

# The scene format this script expects. fetch() replaces an older scene it installed itself.
SCENE_VERSION = 5

# Where the car starts if the scene does not say: x, y in metres and yaw in radians. The site
# keeps its real elevation, so the ground is near z = 295 m and not z = 0.
START_X, START_Y, START_YAW = 146.0, 75.7, 0.70


# --------------------------------------------------------------------------------------------
# Getting the scene
# --------------------------------------------------------------------------------------------


def fetch(scene_dir=SCENE_DIR):
    """Return a directory holding the scene, downloading and unpacking it if needed."""
    scene_dir = os.path.abspath(scene_dir)

    have = _has(scene_dir, MANIFEST) and _has(scene_dir, GROUND)
    if have and _version(scene_dir) < SCENE_VERSION:
        if _has(scene_dir, MARKER):
            # An older scene that an earlier version of this script put here. It is a download
            # cache and nothing else, so it is set aside and fetched again.
            old = f"{scene_dir}.v{_version(scene_dir)}"
            print(f"The scene in {scene_dir} is an older version. Moving it to {old} and fetching the current one.")
            if os.path.exists(old):
                raise SystemExit(f"{old} is in the way. Remove it and run again.")
            os.replace(scene_dir, old)
            have = False
        else:
            print(f"Note: {scene_dir} holds an older scene (version {_version(scene_dir)}, this script expects "
                  f"{SCENE_VERSION}). It was not installed by this script, so it is used as it is.")

    if not have:
        if os.path.isdir(scene_dir) and os.listdir(scene_dir):
            raise SystemExit(f"{scene_dir} exists but holds no speedway scene. Remove it, or pass another --data directory.")
        print(f"Speedway scene not found in {scene_dir}")
        partial = _download_and_unpack(os.environ.get("SPEEDWAY_SCENE_URL", SCENE_URL), SCENE_SHA256, scene_dir)
        with open(os.path.join(partial, MARKER), "w") as f:
            f.write(f"installed by speedway.py, scene version {SCENE_VERSION}\n")
        if os.path.isdir(scene_dir):
            os.rmdir(scene_dir)
        os.replace(partial, scene_dir)
        print(f"  scene ready in {scene_dir}")

    return scene_dir


def _version(scene_dir):
    """The scene format version the manifest declares."""
    try:
        with open(os.path.join(scene_dir, MANIFEST)) as f:
            return int(json.load(f).get("version", 1))
    except (OSError, ValueError):
        return 0


def _has(scene_dir, name):
    return os.path.isfile(os.path.join(scene_dir, name))


def _download_and_unpack(url, sha256, scene_dir):
    """Download one archive, check it, and unpack it beside scene_dir. Returns the unpacked path.

    Unpacking beside the target and moving into place afterwards means an interrupted run cannot
    leave behind something that looks like a scene but is missing half its files.
    """
    os.makedirs(os.path.dirname(scene_dir), exist_ok=True)
    archive = scene_dir + ".download"
    partial = scene_dir + ".partial"

    print(f"  downloading {url}")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as src, open(archive, "wb") as dst:
            total = int(src.headers.get("Content-Length") or 0)
            done, shown = 0, -1
            while True:
                block = src.read(1 << 20)
                if not block:
                    break
                dst.write(block)
                digest.update(block)
                done += len(block)
                percent = 100 * done // total if total else 0
                if percent != shown and sys.stdout.isatty():
                    print(f"\r  {done >> 20} MB  {percent:3d}%", end="", flush=True)
                    shown = percent
        if sys.stdout.isatty():
            print()
    except Exception as err:
        _remove(archive)
        raise SystemExit(f"  download failed: {err}\n  Fetch it by hand (see the README) and pass --data DIR.")

    if digest.hexdigest() != sha256:
        _remove(archive)
        raise SystemExit(f"  checksum mismatch\n    expected {sha256}\n    got      {digest.hexdigest()}")

    print("  unpacking")
    shutil.rmtree(partial, ignore_errors=True)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(partial, filter="data")
        except TypeError:  # Python older than 3.12 has no extraction filters
            tar.extractall(partial)
    _remove(archive)
    return partial


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------------------------
# Loading it into a Chrono system
# --------------------------------------------------------------------------------------------


def add_scenery(system, scene_dir, textures="standard", groups=None, verbose=True):
    """Add everything you see: the road and the land around it, and all that stands on them.

    The manifest lists meshes and the placements they appear at. Each mesh becomes one visual
    shape, added to a body once per placement, so its triangles are loaded once however often it
    appears. The bodies are fixed and carry no collision geometry. Scenery never reaches the
    solver, and the driving surface is a separate object: see add_ground.

    textures picks the ground photo's detail, one of TEXTURES. groups, if given, keeps only
    those manifest groups, out of GROUPS.

    Every part's material also gets a class id for Chrono::Sensor's segmentation camera.
    labels(scene_dir) lists what the ids mean.

    Returns the bodies, one per group.
    """
    import pychrono as chrono

    doc = manifest(scene_dir)
    assets = doc["assets"]
    class_ids = {name: n for n, name in enumerate(doc.get("labels", []), start=1)}

    meshes = {}     # mesh path -> ChTriangleMeshConnected, or None if unusable
    materials = {}  # (mesh path, colour) -> ChVisualMaterial
    shapes = {}     # (asset index, scale, colours) -> the asset's parts as visual shapes
    bodies = {}     # group -> ChBody
    placed = {}     # group -> number of placements
    textured = set()
    skipped = 0

    for inst in doc["instances"]:
        group = inst.get("group", "default")
        if groups is not None and group not in groups:
            continue

        # A shape is shared by every placement of the same asset at the same scale and colours.
        # Scale lives on the shape and not on the placement frame, so differing scales need
        # their own shape. Rounding keeps near-identical scales together.
        scale = inst.get("scale", [1.0, 1.0, 1.0])
        colours = inst.get("colours", {})
        key = (inst["asset"], tuple(round(s * 1000) for s in scale), tuple(sorted((k, tuple(v)) for k, v in colours.items())))
        parts = shapes.get(key)
        if parts is None:
            parts = []
            for part in assets[inst["asset"]]["parts"]:
                path = os.path.join(scene_dir, part["mesh"])
                if path not in meshes:
                    mesh = None
                    if os.path.isfile(path):
                        # Texture coordinates are not loaded by default and a texture needs them.
                        mesh = chrono.ChTriangleMeshConnected.CreateFromWavefrontFile(path, True, True)
                        if mesh is not None and mesh.GetNumTriangles() == 0:
                            mesh = None
                    meshes[path] = mesh
                mesh = meshes[path]
                if mesh is None:
                    continue

                # One material per part and colour, shared by every scale of the asset.
                colour = tuple(colours.get(part["name"], part.get("colour", [0.62, 0.62, 0.64])))
                material = materials.get((path, colour))
                if material is None:
                    material = chrono.ChVisualMaterial()
                    material.SetDiffuseColor(chrono.ChColor(*colour))
                    material.SetSpecularColor(chrono.ChColor(*part.get("ks", [0.0, 0.0, 0.0])))
                    material.SetSpecularExponent(float(part.get("ns", 1.0)))
                    material.SetRoughness(part.get("roughness_value", 1.0))
                    material.SetMetallic(part.get("metallic_value", 0.0))
                    texture = _existing(scene_dir, (part.get("texture") or "").replace("<level>", textures))
                    if texture:
                        material.SetKdTexture(texture)
                        textured.add(path)
                    label = inst.get("label", group)
                    if label in class_ids:
                        material.SetClassID(class_ids[label])
                    materials[(path, colour)] = material

                shape = chrono.ChVisualShapeTriangleMesh()
                # load_materials=False matters. The default re-parses the OBJ on every call to
                # look for an MTL, which costs minutes over a scene this size and finds nothing,
                # because the material is attached explicitly below.
                shape.SetMesh(mesh, False)
                shape.SetScale(chrono.ChVector3d(*scale))
                shape.SetMutable(False)
                if part.get("double_sided"):
                    # Leaves are single triangles. Chrono::VSG draws one side of a triangle
                    # unless told otherwise, which leaves half of every crown missing.
                    shape.SetDoubleFaced(True)
                shape.AddMaterial(material)
                parts.append(shape)
            shapes[key] = parts
        if not parts:
            skipped += 1
            continue

        body = bodies.get(group)
        if body is None:
            body = chrono.ChBody()
            body.SetName("scenery_" + group)
            body.SetFixed(True)
            body.EnableCollision(False)
            system.Add(body)
            bodies[group] = body

        rot = chrono.ChQuaterniond(*inst.get("rot", [1.0, 0.0, 0.0, 0.0]))
        rot.Normalize()
        frame = chrono.ChFramed(chrono.ChVector3d(*inst.get("pos", [0.0, 0.0, 0.0])), rot)
        for shape in parts:
            body.AddVisualShape(shape, frame)
        placed[group] = placed.get(group, 0) + 1

    if verbose:
        used = sum(1 for m in meshes.values() if m is not None)
        print(f"  scenery: {sum(placed.values())} placements of {used} meshes, {len(textured)} textured")
        for group in sorted(placed):
            print(f"    {placed[group]:5d}  {group}")
        if skipped:
            print(f"    ({skipped} placements skipped: their meshes are missing)")
    return list(bodies.values())


def manifest(scene_dir):
    """The scene manifest as a dict, for the parts of it that are data and not geometry.

    "frame" says where the scene is on Earth. "instances" name every placement: each tree has
    its position, height and crown width as measured. "start" is a pose on the track.
    """
    with open(os.path.join(scene_dir, MANIFEST)) as f:
        return json.load(f)


def labels(scene_dir):
    """class id -> name for the class ids add_scenery puts on materials."""
    return dict(enumerate(manifest(scene_dir).get("labels", []), start=1))


def start_pose(scene_dir):
    """(x, y, yaw) of a pose on the track, facing along it."""
    start = manifest(scene_dir).get("start")
    return (start["x"], start["y"], start["yaw"]) if start else (START_X, START_Y, START_YAW)


def _existing(scene_dir, relative):
    if not relative:
        return None
    path = os.path.join(scene_dir, relative)
    return path if os.path.isfile(path) else None


def add_ground(system, scene_dir, friction=0.9, restitution=0.01, young_modulus=2e7, offroad_friction=None):
    """Add the driving surface and return it as an initialized RigidTerrain.

    One mesh over the whole site: pavement, grass, banks and the fields around. These are the
    same triangles add_scenery draws, so what you see and what the wheels touch are the same
    surface. It is collision only, since drawing it a second time would z-fight.

    Give offroad_friction to have the pavement and the rest as two patches with their own
    friction. They share their boundary, so the surface is the same. It costs speed: measured on
    PyChrono build 1187, a step of the demo takes 0.45 ms that way against 0.31 ms with one mesh.

    Keep the returned object alive for as long as the system uses it.
    """
    import pychrono as chrono
    import pychrono.vehicle as veh

    patches = [(GROUND, friction)] if offroad_friction is None else [(ROAD, friction), (TERRAIN, offroad_friction)]
    terrain = veh.RigidTerrain(system)
    for name, mu in patches:
        info = chrono.ChContactMaterialData()
        info.mu = mu
        info.cr = restitution
        info.Y = young_modulus
        material = info.CreateMaterial(system.GetContactMethod())
        # connected_mesh=True is the one to use. Measured on PyChrono build 1187, the
        # triangle-soup alternative takes many times longer to enter the collision system, and
        # its height queries land centimetres below the real surface.
        terrain.AddPatch(material, chrono.CSYSNORM, os.path.join(scene_dir, name), True, 0.0, False)
    terrain.Initialize()
    return terrain


def ground_height(scene_dir, x, y, radius=2.0):
    """Height of the ground near (x, y), read straight from the ground mesh.

    RigidTerrain.GetHeight raycasts the collision system and returns 0 on a miss. Asked during
    setup, before the first step, it misses, and 0 is 295 m below this track. Reading the mesh
    works at any time. Returns None if no vertex lies within radius of the point.
    """
    best = None
    with open(os.path.join(scene_dir, GROUND)) as f:
        for line in f:
            if line.startswith("v "):
                vx, vy, vz = map(float, line.split()[1:4])
                if abs(vx - x) < radius and abs(vy - y) < radius and (best is None or vz > best):
                    best = vz
    return best


# --------------------------------------------------------------------------------------------
# The demo
# --------------------------------------------------------------------------------------------

TEXTURES_HELP = """ground photo detail (default: standard)
  low       10 cm per pixel, for a GPU short of memory
  standard  5 cm per pixel"""

TIRES = {"pac02": "audi/json/audi_Pac02Tire.json", "tmeasy": "audi/json/audi_TMeasyTire.json", "rigid": "audi/json/audi_RigidTire.json"}


def main():
    parser = argparse.ArgumentParser(
        description="Drive an Audi around Columbus 151 Speedway. Hold W/S for throttle and brake, A/D to steer.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data", metavar="DIR", default=SCENE_DIR, help="scene directory, downloaded into if empty (default: scene/ beside this file)")
    parser.add_argument("--textures", choices=list(TEXTURES), default="standard", help=TEXTURES_HELP)
    for flag, _, what in OPTIONAL:
        parser.add_argument(f"--no-{flag}", action="store_true", help=f"leave out {what}")
    parser.add_argument("--no-sky", action="store_true", help="plain background instead of the sky dome")
    parser.add_argument("--no-shadows", action="store_true", help="do not draw shadows. Worth trying on a slow GPU: stock Chrono redraws the scene for every shadow map")
    parser.add_argument("--offroad-friction", metavar="MU", type=float, default=None, help="friction off the pavement. The pavement has 0.9.\nGiving this splits the ground into two collision meshes, which slows the physics (default: 0.9 everywhere)")
    parser.add_argument("--tire", choices=sorted(TIRES), default="pac02", help="tire model (default: pac02)")
    parser.add_argument("--tire-step", metavar="S", type=float, default=1e-4, help="tire internal step in seconds (default: 1e-4)")
    parser.add_argument("--keys", choices=["held", "step"], default="held", help="held: inputs follow the keys held down, as in a driving game (default)\nstep: each press nudges an input, which then stays put")
    parser.add_argument("--speed-limit", metavar="V", type=float, default=20.0, help="with --keys step, the speed the throttle steps are scaled toward, m/s (default: 20)")
    parser.add_argument("--duration", metavar="S", type=float, default=None, help="stop after this many simulated seconds (default: run until the window closes)")
    parser.add_argument("--headless", action="store_true", help="no window: simulate with the vehicle parked and print where it is")
    parser.add_argument("--snapshot", metavar="FILE", default=None, help="save a picture of the window to FILE two seconds in, then carry on")
    args = parser.parse_args()

    try:
        import pychrono as chrono
        import pychrono.vehicle as veh
        if not args.headless:
            import pychrono.vsg3d  # noqa: F401, checked here so a missing module fails before the download
    except ImportError as err:
        raise SystemExit(
            f"This needs PyChrono with the vehicle and VSG modules ({err}).\n"
            "The projectchrono conda channel ships them. The conda-forge pychrono package does not.\n"
            "  conda install projectchrono::pychrono -c conda-forge"
        )

    scene = fetch(args.data)
    boot = time.perf_counter()

    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    # The default solver under-solves a vehicle's suspension constraints, which shows up as the
    # suspension juddering for no visible reason. These are the settings Chrono's own road demos use.
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.GetSolver().AsIterative().SetMaxIterations(150)
    system.SetMaxPenetrationRecoverySpeed(4.0)

    if not args.headless:
        leave_out = {group for flag, group, _ in OPTIONAL if getattr(args, "no_" + flag.replace("-", "_"))}
        add_scenery(system, scene, args.textures, groups=[g for g in GROUPS if g not in leave_out])
    terrain = add_ground(system, scene, offroad_friction=args.offroad_friction)

    x, y, yaw = start_pose(scene)
    z = ground_height(scene, x, y)
    if z is None:
        raise SystemExit(f"no ground found under the start pose in {scene}")

    audi = veh.WheeledVehicle(system, veh.GetVehicleDataFile("audi/json/audi_Vehicle.json"))
    audi.Initialize(chrono.ChCoordsysd(chrono.ChVector3d(x, y, z + 0.5), chrono.QuatFromAngleZ(yaw)))
    audi.SetChassisVisualizationType(chrono.VisualizationType_MESH)
    audi.SetSuspensionVisualizationType(chrono.VisualizationType_MESH)
    audi.SetSteeringVisualizationType(chrono.VisualizationType_MESH)
    audi.SetWheelVisualizationType(chrono.VisualizationType_MESH)

    engine = veh.ReadEngineJSON(veh.GetVehicleDataFile("audi/json/audi_EngineSimpleMap.json"))
    transmission = veh.ReadTransmissionJSON(veh.GetVehicleDataFile("audi/json/audi_AutomaticTransmissionSimpleMap.json"))
    audi.InitializePowertrain(veh.ChPowertrainAssembly(engine, transmission))

    for axle in audi.GetAxles():
        for wheel in axle.GetWheels():
            tire = veh.ReadTireJSON(veh.GetVehicleDataFile(TIRES[args.tire]))
            tire.SetStepsize(args.tire_step)
            audi.InitializeTire(tire, wheel, chrono.VisualizationType_MESH)

    step = 1e-3
    if args.headless:
        run_headless(system, audi, terrain, veh, step, args.duration if args.duration is not None else 5.0, z)
    else:
        run_window(system, audi, terrain, chrono, veh, step, args, boot)

    # Tearing the visual system down from Python can crash on the way out, after everything has
    # already worked. Leave without running destructors.
    sys.stdout.flush()
    os._exit(0)


def run_window(system, audi, terrain, chrono, veh, step, args, boot):
    driver = veh.ChInteractiveDriver(audi)
    held = args.keys == "held"
    if held and not hasattr(veh.ChInteractiveDriver, "KeyboardMode_HELD"):
        print("  This PyChrono has no held-key driving, so falling back to --keys step.")
        held = False
    if held:
        # Pedals and wheel follow the keys that are down, and ease toward them: Chrono's driver
        # approaches each target with a time constant of a quarter of a second.
        driver.SetKeyboardMode(veh.ChInteractiveDriver.KeyboardMode_HELD)
    else:
        driver.SetSteeringDelta(0.04)
        driver.SetThrottleDelta(1.0 / max(1.0, args.speed_limit))
        driver.SetBrakingDelta(0.3)
    driver.Initialize()

    vis = veh.ChWheeledVehicleVisualSystemVSG()
    if held and hasattr(vis, "SetKeyboardMode"):
        vis.SetKeyboardMode(veh.ChInteractiveDriver.KeyboardMode_HELD)
    vis.SetWindowTitle("Columbus 151 Speedway")
    vis.SetWindowSize(1600, 900)
    vis.AttachVehicle(audi)
    vis.SetChaseCamera(chrono.ChVector3d(0.0, 0.0, 1.75), 7.0, 0.6)
    vis.SetLightIntensity(1.0)
    vis.SetLightDirection(1.5 * chrono.CH_PI_2, chrono.CH_PI_4)
    if not args.no_shadows:
        vis.EnableShadows()
    if not args.no_sky:
        vis.EnableSkyTexture()
    vis.AttachDriver(driver)
    vis.Initialize()

    print(f"\n  [{time.perf_counter() - boot:.1f} s to build the scene and open the window]")
    if held:
        print("Hold W to accelerate, S to brake, A or D to steer. Let go to coast and centre.\n")
    else:
        print("W/S step the throttle and brake, A/D step the steering.\n")

    render_step = 1.0 / 50  # physics wants 1 kHz, the display does not
    next_render = next_report = 0.0
    last_render = -1.0
    frames = 0
    clock = time.perf_counter()  # the wall-clock instant that simulated time 0 is held against
    mark_wall, mark_time, mark_frames = clock, 0.0, 0
    snapshot = args.snapshot

    while vis.Run():
        now = system.GetChTime()
        if args.duration is not None and now >= args.duration:
            break

        # How far the simulation has fallen behind the wall clock. A stall that long is the
        # window being dragged or the machine being busy, and is written off instead of chased.
        lag = (time.perf_counter() - clock) - now
        if lag > 0.5:
            clock += lag - 0.05
            lag = 0.05

        # Draw only when the simulation is keeping up. A heavy scene can take longer to draw
        # than a 50 Hz frame lasts, and drawing every frame regardless turns that into slow
        # motion. Skipping frames keeps the car at real time and lets the frame rate drop
        # instead, down to a floor of five a second.
        if now >= next_render and (lag < render_step or now - last_render >= 0.2):
            vis.BeginScene()
            vis.Render()
            vis.EndScene()
            frames += 1
            last_render = now
            next_render = now + render_step
            if snapshot and now >= 2.0:
                vis.WriteImageToFile(snapshot)
                print(f"  picture saved to {snapshot}")
                snapshot = None

        if now >= next_report:
            # Rates over the stretch since the last line, so the slow first seconds, when the
            # textures are still being loaded, do not drag every later figure down.
            wall = time.perf_counter()
            if wall > mark_wall:
                print(f"  t={now:5.1f} s   {(now - mark_time) / (wall - mark_wall):.2f}x real time   "
                      f"{(frames - mark_frames) / (wall - mark_wall):4.1f} frames/s   {audi.GetSpeed():5.1f} m/s")
            mark_wall, mark_time, mark_frames = wall, now, frames
            next_report += 2.0

        inputs = driver.GetInputs()
        driver.Synchronize(now)
        terrain.Synchronize(now)
        audi.Synchronize(now, inputs, terrain)
        vis.Synchronize(now, inputs)

        driver.Advance(step)
        terrain.Advance(step)
        audi.Advance(step)
        vis.Advance(step)
        system.DoStepDynamics(step)

        # Physics alone runs several times faster than real time, so wait for the clock.
        ahead = (now + step) - (time.perf_counter() - clock)
        if ahead > 0.002:
            time.sleep(ahead - 0.001)


def run_headless(system, audi, terrain, veh, step, duration, ground_z):
    inputs = veh.DriverInputs()
    inputs.m_braking = 1.0
    print(f"  headless: {duration:g} s with the brakes on, ground at z = {ground_z:.2f} m")
    next_report = 0.0
    while system.GetChTime() < duration:
        now = system.GetChTime()
        if now >= next_report:
            pos = audi.GetPos()
            print(f"  t={now:4.1f} s   x={pos.x:8.2f}  y={pos.y:8.2f}  z={pos.z:7.2f}   {audi.GetSpeed():5.2f} m/s")
            next_report += 1.0
        terrain.Synchronize(now)
        audi.Synchronize(now, inputs, terrain)
        terrain.Advance(step)
        audi.Advance(step)
        system.DoStepDynamics(step)
    pos = audi.GetPos()
    sunk = ground_z - pos.z
    if sunk > 0.5 or abs(pos.z - ground_z) > 2.0:
        raise SystemExit(f"  FAILED: the vehicle is at z = {pos.z:.2f} m, the ground is at {ground_z:.2f} m")
    print(f"  ok: resting {pos.z - ground_z:.2f} m above the ground (chassis reference height)")


if __name__ == "__main__":
    main()
