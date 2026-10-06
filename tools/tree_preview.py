"""Look at treegen output in stock Chrono::VSG: a row of every kind at three sizes, a car-sized
box beside each group for scale, shadows on. Pictures go to inspect/trees/.

    python tools/tree_preview.py                      # every kind, the standard views
    python tools/tree_preview.py --kinds willow shrub --tag try2
    python tools/tree_preview.py --views far near under
    python tools/tree_preview.py --culled --tag culled   # without SetDoubleFaced on the leaves
    python tools/tree_preview.py --bare --tag bare       # limbs only

Views: "all" (the whole row from the air), "far" (30 m from each group at eye height),
"near" (6 m from the middle tree of each group), "under" (standing at the edge of the crown,
looking up), "back" (30 m, looking into the light).
"""
import argparse
import os
import sys
import time

import pychrono as chrono
import pychrono.vsg3d as vsg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import treegen  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "inspect", "trees")

# kind -> three (height, crown radius, triangle budget)
SIZES = {
    "broadleaf": [(8.0, 3.0, 5000), (12.0, 4.5, 6500), (17.0, 6.5, 8000)],
    "upright": [(9.0, 2.0, 5000), (14.0, 3.0, 6500), (18.0, 4.0, 8000)],
    "willow": [(7.0, 3.5, 5000), (10.0, 5.0, 6500), (14.0, 7.0, 8000)],
    "shrub": [(2.0, 1.2, 1500), (3.5, 1.8, 2200), (5.0, 2.5, 3000)],
}
LEAF, WOOD, GROUND, CAR = (0.20, 0.33, 0.10), (0.30, 0.25, 0.20), (0.40, 0.42, 0.28), (0.72, 0.72, 0.76)


def add_mesh(system, path, color, x, y, yaw=0.0, double_faced=False):
    mesh = chrono.ChTriangleMeshConnected.CreateFromWavefrontFile(path, True, True)
    shape = chrono.ChVisualShapeTriangleMesh()
    shape.SetMesh(mesh, False)
    shape.SetMutable(False)
    if double_faced:
        shape.SetDoubleFaced(True)
    material = chrono.ChVisualMaterial()
    material.SetDiffuseColor(chrono.ChColor(*color))
    material.SetRoughness(0.9)
    shape.AddMaterial(material)
    body = chrono.ChBody()
    body.SetFixed(True)
    body.EnableCollision(False)
    system.Add(body)
    body.AddVisualShape(shape, chrono.ChFramed(chrono.ChVector3d(x, y, 0.0), chrono.QuatFromAngleZ(yaw)))


def add_box(system, size, color, pos):
    shape = chrono.ChVisualShapeBox(*size)
    material = chrono.ChVisualMaterial()
    material.SetDiffuseColor(chrono.ChColor(*color))
    material.SetRoughness(0.9)
    shape.AddMaterial(material)
    body = chrono.ChBody()
    body.SetFixed(True)
    body.EnableCollision(False)
    system.Add(body)
    body.AddVisualShape(shape, chrono.ChFramed(chrono.ChVector3d(*pos), chrono.QuatFromAngleZ(0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=list(SIZES))
    ap.add_argument("--views", nargs="+", default=["all", "far", "near"])
    ap.add_argument("--tag", default="preview")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--double-sided", action="store_true", help="generator emits both leaf sides")
    ap.add_argument("--culled", action="store_true", help="leave the leaf shapes single faced (stock culling)")
    ap.add_argument("--bare", action="store_true", help="wood only, to look at the limb structure")
    ap.add_argument("--keep-obj", action="store_true", help="leave the OBJ files in inspect/trees/obj")
    args = ap.parse_args()

    obj_dir = os.path.join(OUT, "obj")
    os.makedirs(obj_dir, exist_ok=True)
    written = []
    system = chrono.ChSystemNSC()
    add_box(system, (600.0, 600.0, 1.0), GROUND, (0.0, 0.0, -0.5))

    shots, x, total = [], 0.0, 0
    for kind in args.kinds:
        x0, mid = x, None
        for i, (h, r, tris) in enumerate(SIZES[kind]):
            x += r + 1.5
            t0 = time.perf_counter()
            tree = treegen.make_tree(kind, h, r, args.seed + i, tris, double_sided=args.double_sided)
            ms = 1000 * (time.perf_counter() - t0)
            n_w, n_l = len(tree["wood"][2]), len(tree["leaves"][2])
            total += n_w + n_l
            print("%-9s h=%4.1f r=%3.1f budget=%5d  wood=%5d leaves=%5d total=%5d  %.0f ms  longest leaf %.2f m (%.1f%% of height)"
                  % (kind, h, r, tris, n_w, n_l, n_w + n_l, ms, tree["info"]["longest_leaf"], 100 * tree["info"]["longest_leaf"] / h))
            for part, color in (("wood", WOOD),) if args.bare else (("wood", WOOD), ("leaves", LEAF)):
                path = os.path.join(obj_dir, "%s_%d_%s.obj" % (kind, i, part))
                treegen.write_obj(path, *tree[part])
                written.append(path)
                add_mesh(system, path, color, x, 0.0, yaw=0.7 * i, double_faced=part == "leaves" and not args.culled)
            if i == 1:
                mid = (x, h, r)
            x += r + 1.5
        add_box(system, (4.5, 1.8, 1.4), CAR, (mid[0] + 3.5, 0.6 * mid[2] + 2.5, 0.7))
        cx, (mx, mh, mr), top = 0.5 * (x0 + x), mid, SIZES[kind][2][0]
        shots += [
            ("far", kind, (cx, 30.0, 1.6), (cx, 0.0, 0.42 * top)),
            ("back", kind, (cx, -30.0, 1.6), (cx, 0.0, 0.42 * top)),
            ("near", kind, (mx - 3.0, 5.2, 1.6), (mx, 0.0, 0.55 * mh)),
            ("under", kind, (mx + 0.3, 0.9 * mr, 1.4), (mx, 0.0, 0.8 * mh)),
        ]
        x += 4.0
    shots.append(("all", "row", (0.5 * x, 0.72 * x, 0.2 * x), (0.5 * x, 0.0, 4.0)))
    shots = [s for s in shots if s[0] in args.views]
    print("triangles in scene: %d" % total)

    vis = vsg.ChVisualSystemVSG()
    vis.AttachSystem(system)
    vis.SetWindowSize(1600, 900)
    vis.SetWindowTitle("tree preview")
    vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
    vis.AddCamera(chrono.ChVector3d(*shots[0][2]), chrono.ChVector3d(*shots[0][3]))
    vis.SetBackgroundColor(chrono.ChColor(0.62, 0.74, 0.88))
    vis.SetLightIntensity(1.0)
    vis.SetLightDirection(1.5 * chrono.CH_PI_2, chrono.CH_PI_4)
    vis.EnableShadows()
    vis.Initialize()
    vis.SetGuiVisibility(False)
    vis.HideLogo()

    def frames(count, save=None):
        for frame in range(count):
            if not vis.Run():
                return
            vis.BeginScene()
            vis.Render()
            vis.EndScene()
            if save and frame == count - 3:
                vis.WriteImageToFile(save)

    # The views all look more or less straight across the row. The camera carries its up vector
    # from one view to the next, so swinging it sideways between views leaves the next one rolled.
    for view, kind, eye, target in shots:
        vis.UpdateCamera(chrono.ChVector3d(*eye), chrono.ChVector3d(*target))
        path = os.path.join(OUT, "%s_%s_%s.png" % (args.tag, view, kind))
        frames(8, save=path)
        print("saved" if os.path.exists(path) else "MISSING", path)
    if not args.keep_obj:  # the meshes were read when the shapes were made
        for path in written:
            os.remove(path)
        if not os.listdir(obj_dir):
            os.rmdir(obj_dir)
    sys.stdout.flush()
    os._exit(0)  # tearing the visual system down from Python can crash


if __name__ == "__main__":
    main()
