# chrono-columbus-speedway

Columbus 151 Speedway, a vehicle test track at W2140 Krause Road, Columbus, Wisconsin, as a
drivable [PyChrono](https://projectchrono.org) scene in one Python script. It runs on stock
PyChrono. Nothing is converted and Chrono is not modified.

![An Audi on the track in the Chrono VSG window](docs/speedway.jpg)

## Run it

```sh
conda install projectchrono::pychrono -c conda-forge
curl -LO https://raw.githubusercontent.com/ksha23/chrono-columbus-speedway/main/speedway.py
python speedway.py
```

The first run downloads the scene (89 MB) into `scene/` beside the script, checks its
SHA-256 and unpacks it. Later runs start in a few seconds.

Hold **W** to accelerate and **S** to brake, and hold **A** or **D** to steer. Let go and the car
coasts and the wheel centres, as in a driving game.

| Option | |
| --- | --- |
| `--no-cones` | leave the traffic cones out |
| `--no-trees` | leave the trees out |
| `--textures low\|standard` | ground photo at 10 cm or 5 cm per pixel, default `standard` |
| `--offroad-friction MU` | friction off the pavement, which has 0.9. Default: 0.9 everywhere. See below for what it costs |
| `--no-shadows` | do not draw shadows. Worth trying on a slow GPU |
| `--no-sky` | plain background instead of the sky dome |
| `--keys held\|step` | `held`, the default, follows the keys held down. `step` is Chrono's older behaviour: each press nudges an input and it stays there |
| `--data DIR` | keep the scene somewhere else |
| `--tire pac02\|tmeasy\|rigid` | tire model, default `pac02` |
| `--tire-step S` | tire internal step in seconds, default `1e-4` |
| `--speed-limit V` | with `--keys step`, the speed the throttle steps are scaled toward, default 20 m/s |
| `--duration S` | stop after S simulated seconds |
| `--snapshot FILE` | save a picture of the window two seconds in |
| `--headless` | no window. Sets the car down with the brakes on and checks that it rests on the ground |

PyChrono has to come from the `projectchrono` conda channel. The `conda-forge` package of the
same name has no vehicle or VSG module.

The simulation holds real time. When a frame takes too long to draw, the script skips frames
instead of slowing the car down.

## Use the track in your own simulation

`speedway.py` is also a module. Put it next to your script:

```python
import pychrono as chrono
import pychrono.vehicle as veh
import speedway

system = chrono.ChSystemNSC()
system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

scene = speedway.fetch()                      # scene directory, downloaded if it is not there
speedway.add_scenery(system, scene)           # what you see: visual shapes, no collision
terrain = speedway.add_ground(system, scene)  # what the wheels touch: one collision mesh

x, y, yaw = speedway.start_pose(scene)        # a pose on the long straight
z = speedway.ground_height(scene, x, y)       # ground height there
```

- `add_scenery` puts every placement on a few fixed bodies as visual shapes. Pass
  `groups=["Road", "Terrain"]` to load only some of `Road`, `Terrain`, `Buildings`, `Trees` and
  `Cones`. Cones are their own group so they can be left out and replaced with your own layout.
- `add_ground` returns an initialized `RigidTerrain` over one mesh: pavement, grass, banks and
  the fields around. They are the same triangles the scenery draws, so what you see is what you
  drive on. Pass `offroad_friction=0.6` to get the pavement and the rest as two patches with
  their own friction. The surface is the same, since the two meshes share their boundary, but a
  step of the demo then takes 0.45 ms instead of 0.31 ms: Chrono casts one ray per patch for
  every tire query.
- The site keeps its real elevation. The track is near **z = 295 m**, not z = 0. Use
  `ground_height` to place things: `RigidTerrain.GetHeight` returns 0 until the first
  `DoStepDynamics`.
- Trees, buildings and cones are drawn and nothing else. They have no collision shapes.
- `speedway.manifest(scene)` is everything else the scene records, as a dict. See below.
- For Chrono::Sensor, every material carries a class id. `speedway.labels(scene)` says what
  each id means.

The scene is plain files, so C++ Chrono or any other tool can read it too. `speedway_ground.obj`
works directly with `RigidTerrain::AddPatch`, and so do `speedway_road.obj` and
`speedway_terrain.obj` as a pair.

## What is in the scene

```
speedway_scene.json    manifest: 113 assets, 1,003 placements in 5 groups
speedway_ground.obj    the ground as one welded collision mesh, 643,826 triangles
speedway_road.obj      the pavement's 78,424 of those triangles
speedway_terrain.obj   the other 565,402
ground/                the same triangles split by texture tile, with texture coordinates
textures/standard/     ground photo at 5 cm per pixel, one JPEG per tile
textures/low/          the same at 10 cm
textures/surround.jpg  aerial imagery for the land beyond the scan
trees/                 54 generated tree models, wood and leaves apart
buildings/             4 buildings, walls and textured roof apart
cones/                 one traffic cone, body and base apart
```

Lengths are metres. x is east, y is north and z is up, which is Chrono's own frame. The origin
is at UTM zone 16N easting 329450, northing 4794560 (latitude 43.28455, longitude -89.10212),
and z is elevation above sea level. The scene is 1280 m by 1024 m. The drone scan covers
17.8 hectares in the middle of it, and the fields and woods around come from aerial imagery so
the view has a horizon.

The manifest looks like this:

```json
{
  "version": 1,
  "frame": { "utm_zone": 16, "origin_easting": 329450.0, "origin_northing": 4794560.0,
             "extent": [-640.0, -512.0, 640.0, 512.0] },
  "labels": ["Road", "Terrain", "Buildings", "Trees", "Cones"],
  "collision": { "ground": "speedway_ground.obj", "road": "speedway_road.obj",
                 "terrain": "speedway_terrain.obj" },
  "start": { "x": -49.95, "y": -79.23, "yaw": 0.72 },
  "assets": [
    { "name": "broadleaf_0_1_light",
      "parts": [
        { "name": "wood", "mesh": "trees/broadleaf_0_1_light_wood.obj", "colour": [0.3, 0.25, 0.2] },
        { "name": "leaves", "mesh": "trees/broadleaf_0_1_light_leaves.obj",
          "colour": [0.2, 0.33, 0.1], "double_sided": true } ] } ],
  "instances": [
    { "asset": 61, "group": "Trees", "name": "tree_0003", "kind": "broadleaf",
      "pos": [245.12, 212.12, 292.32], "rot": [0.80238, 0.0, 0.0, 0.59681],
      "scale": [0.97, 0.97, 0.966], "height": 12.6, "crown_radius": 5.2,
      "colours": { "leaves": [0.397, 0.418, 0.086], "wood": [0.127, 0.098, 0.073] } },
    { "asset": 112, "group": "Cones", "name": "cone_020", "height": 0.71,
      "pos": [180.29, 149.7, 294.472], "rot": [1.0, 0.0, 0.0, 0.0],
      "scale": [0.71, 0.71, 0.71],
      "colours": { "body": [0.144, 0.989, 0.073], "base": [0.144, 0.989, 0.073] } } ]
}
```

- An asset is a list of single-material meshes. A part has a `colour` and may have a `texture`.
  A ground texture path contains `<level>`, to be replaced by `standard` or `low`.
- An instance places asset number `asset` at `pos` with `rot` as a (w, x, y, z) quaternion and a
  per-axis `scale`. `colours`, if present, overrides the colour of the named parts for that
  placement. A tree records the `height` and `crown_radius` measured for it, a cone its `height`.
- Colours and textures are stored the way the renderer needs them, not as they look. See
  [the renderer](#the-renderer) below.
- Paths are relative to the manifest.

To fetch the scene without the script:

```sh
curl -LO https://github.com/ksha23/chrono-columbus-speedway/releases/download/v1/speedway_scene_base.tar.gz
echo "97a930c609ed5cb671bc2040746f01346d9939af6e01de9c5088dc9448bba1c0  speedway_scene_base.tar.gz" | shasum -a 256 -c
mkdir scene && tar -xzf speedway_scene_base.tar.gz -C scene
```

## How the scene was built

The source is a drone photogrammetry scan of the site: a single textured mesh of 456,280
triangles with 1.3 GB of photo texture at about 2 cm per pixel. A scan like that is a good
picture and a poor model. Its trees are blobs, its cones are flattened into the pavement, the
shadows of the day are printed on the ground, and this one was the wrong shape. So the scan
supplies the picture and the positions of things, and the shape of the ground comes from
somewhere better.

**The ground is the USGS lidar survey.** The scan was georeferenced from the drone's own GPS
with no ground control. Compared with the USGS 3DEP lidar elevation model it is 3.4% too large
and domed: its ground curves away from the truth by more than 14 m between the middle of the
site and the ends. The ground mesh is therefore built from the lidar, on a 1 m grid where the
scan has coverage and 8 m beyond. On the skid pad the lidar surface is flat to 4 mm.

**The scan is fitted onto it.** Horizontally with an affine map, in effect a scale of 0.967, a
small rotation and a shift, found by sliding 80 m windows of the scan's ground over the lidar
surface until ditches and banks line up. Vertically with a polynomial that removes the dome. After the fit the scan's open ground is
within 4 cm of the lidar at the median and within 26 cm for 95% of points, measured on blocks
of ground held out of the fit. Horizontal position was checked against a second source the fit
never saw, USGS aerial imagery: the two agree within 1.1 m at the median and 2.1 m at worst.
That figure is the limit of the check as much as of the fit, since the imagery was read at
0.5 m per pixel.

**The photo is redrawn from above** in the corrected frame at 5 cm per pixel, and then cleaned:

- *Shadows.* The photographed shadows are relit and then levelled to the tone of the sunlit
  ground around them, so the renderer's own shadows are the only ones. The mosaic was flown at
  two times of day, around midday and mid-afternoon by the shadow directions, and each half is
  treated with its own sun.
- *Cones.* 123 traffic cones are found by colour on the pavement, painted out together with
  their shadows, and stood back up as models at the same spots. A cone's height comes from the
  length of the shadow it cast.
- *Trees and buildings.* Whatever stands more than 1.5 m above the lidar ground is a tree or a
  building. The picture of its top is painted over with the ground around it.

**Road and land are separate meshes.** The pavement's outline is taken from the photo and the
ground mesh is cut along it, so every triangle is either road or not. Where a tree's crown hid
the road's edge from the drone, the edge is carried straight across the gap.

**Trees are generated.** 407 crowns are measured in the scan. A crown too wide to be one tree
is replanted as several, which gives 822 trees. Each is a generated model chosen by its shape
(broadleaf, upright, willow, shrub), stretched to the measured height and crown width, and
coloured with the leaf colour the drone saw. A trunk that would land on pavement is stepped
back to the verge. The 387 trees whose crowns come within 12 m of pavement get the full model,
up to 7,000 triangles. The other 435 get one with 40% of the triangles, because stock
Chrono::VSG draws every triangle of every tree again for each shadow map. An earlier build
with every tree at full detail ran at 16 frames a second on an M4 Pro. This one holds 50.

**Buildings are refitted** as a rectangle with straight walls and a gable roof, from the scan's
roof heights. The roof keeps its own photograph. The walls get one flat colour.

### The renderer

Stock Chrono::VSG takes a texture's values as linear light and encodes the result for the
display, so a photograph used as a texture comes out pale and flat. Measured with grey test
cards on PyChrono build 1187: flat ground in the script's light shows 0.70 of its texture value
in linear terms, and a cast shadow is 0.24 as bright as the sunlit ground beside it. Every
texture and colour in the scene is stored with the inverse of that applied, so that sunlit
ground appears on screen as it does in the drone's photo. `tools/renderer.py` has the numbers.

It also draws one side of a triangle only. Leaves are single triangles meant to be seen from
both sides, and `add_scenery` sets `SetDoubleFaced(True)` on them.

### Rebuilding it

`tools/` holds the whole pipeline, and `tools/build_all.sh` runs it from the scan to the
archive in 13 minutes on an M4 Pro. It needs numpy, scipy, Pillow and PyChrono, and about 15 GB
of memory. Every step is seeded: a second run from scratch gave the same archive, byte for byte.
`tools/fetch_reference.sh` downloads the USGS data. The scan itself is not in this repository.

## What is not right

- **Buildings are boxes.** No doors, windows or wall detail, and the wall colour is a guess
  from the little the drone saw of them. Fences, guard rails, light poles and parked vehicles
  are not modelled at all. The vehicles are painted out of the photo. Fence and pole shadows
  are still in it as thin lines.
- **Trees are stylised.** Up close a leaf is a plain triangle about half a metre long. Species
  are guessed from crown shape. In the woods south of the track the scan could not separate
  crowns, so those trees are spread evenly over the canopy it saw, and thinner than the real
  woods. Trees away from the road are visibly sparser if you drive up to them.
- **The road's edge wanders.** It follows the grass line in the photo, which is ragged by a
  few tenths of a metre, and under tree shadows it is less certain than that.
- **Shadow removal leaves traces.** Where a tree's shadow crossed the road the concrete is
  cleaner and smoother than the pavement around it, because the dapple was levelled out. On
  grass a relit shadow is a little more even and more olive than its surroundings.
- **Nothing has collision but the ground.** A car drives through trees, cones and buildings.
- **The road is as smooth as the lidar.** Joints, cracks and patches are in the picture and not
  in the surface. The lidar has 1 m posts.
- **The surround is a different photograph.** Beyond the scan the ground is USGS aerial imagery
  from another season and far coarser, colour matched. The seam shows from above.
- **The pond is a picture of a pond** on solid ground.

## Tested with

PyChrono 10.0.0 from the `projectchrono` channel, conda build `py313_1187`, on macOS (Apple
silicon). On an M4 Pro the demo holds real time at 50 frames a second, which is the script's
cap, with shadows on. It uses 6.8 GB of memory, or 4.8 GB with `--textures low`. Linux and
Windows have not been tried.

## Licence and credit

`speedway.py` and `tools/` are under the BSD 3-Clause licence in [`LICENSE`](LICENSE), the same
terms as Chrono.

The scene is built from:

- A drone scan of the site flown in September 2024 and processed with
  [WebODM](https://www.opendronemap.org/webodm/).
  <!-- Who flew it and under what terms it may be shared goes here before this is published. -->
- The [USGS 3D Elevation Program](https://www.usgs.gov/3d-elevation-program) lidar elevation
  model and [NAIP](https://naip-usdaonline.hub.arcgis.com/) aerial imagery, both public domain.
- The traffic cone model from [Project Chrono](https://projectchrono.org)'s data directory,
  BSD 3-Clause.

This repository is not affiliated with Columbus 151 Speedway or its owners.
