"""The cone mesh the scene ships: Chrono's own 750 mm traffic cone, rescaled to be 1 m tall.

Chrono's data directory has the model (models/traffic_cone/trafficCone750mm.obj, BSD-3-Clause
like the rest of Chrono). A copy goes into the scene so the scene stands on its own, as two
triangle meshes, the body and the base plate, so the two can be coloured apart. The base sits
on z = 0. A placement then scales the pair by the cone's height in metres.
"""
import os

import numpy as np

SOURCE = "models/traffic_cone/trafficCone750mm.obj"
PARTS = {"Cone": "body", "Cube": "base"}


def convert(source_path, out_dir):
    """Write cone_body.obj and cone_base.obj into out_dir. Returns the total triangle count."""
    vertices, faces, current = [], {}, None
    with open(source_path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("o "):
                current = faces.setdefault(PARTS[line.split()[1]], [])
            elif line.startswith("f "):
                c = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                current += [(c[0], c[k], c[k + 1]) for k in range(1, len(c) - 1)]
    v = np.array(vertices)
    v[:, 2] -= v[:, 2].min()
    v /= v[:, 2].max()
    v[:, :2] -= (v[:, :2].min(0) + v[:, :2].max(0)) / 2
    total = 0
    for part, tris in faces.items():
        # Flat shading, as the source has it: every triangle gets its own three vertices and normal.
        lines, count = [], 0
        for a, b, c in tris:
            n = np.cross(v[b] - v[a], v[c] - v[a])
            length = np.linalg.norm(n)
            if length < 1e-12:
                continue
            n /= length
            for p in (a, b, c):
                lines.append(f"v {v[p][0]:.5f} {v[p][1]:.5f} {v[p][2]:.5f}\nvn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            count += 1
        with open(os.path.join(out_dir, f"cone_{part}.obj"), "w") as out:
            out.write(f"# Traffic cone {part}, for a cone 1 m tall with its base on z = 0.\n")
            out.write("# From Project Chrono's data/" + SOURCE + " (BSD-3-Clause).\n")
            out.write("".join(lines))
            out.write("".join(f"f {3 * t + 1}//{3 * t + 1} {3 * t + 2}//{3 * t + 2} {3 * t + 3}//{3 * t + 3}\n" for t in range(count)))
        total += count
    return total
