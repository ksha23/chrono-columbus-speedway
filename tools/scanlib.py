"""Shared helpers for the WebODM scan: load the OBJ once, sample it, write images."""
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

OBJ = "odm_textured_model_geo.obj"
TEXTURE = "odm_textured_model_geo_{}_map_Kd.png"


def load_scan(scan_dir, cache_dir):
    """Return (v, vt, tri_v, tri_t, tri_mat, names) for the scan, cached as an .npz.

    v is (N, 3) positions, vt is (M, 2) texture coordinates. tri_v and tri_t are (T, 3) indices
    into them, tri_mat is the material index of each triangle and names lists the materials.
    """
    cache = os.path.join(cache_dir, "scan.npz")
    if os.path.isfile(cache):
        d = np.load(cache, allow_pickle=False)
        return d["v"], d["vt"], d["tri_v"], d["tri_t"], d["tri_mat"], [str(n) for n in d["names"]]

    v, vt, faces, mats, names, current = [], [], [], [], [], -1
    with open(os.path.join(scan_dir, OBJ)) as f:
        for line in f:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("vt "):
                vt.append(line.split()[1:3])
            elif line.startswith("usemtl"):
                name = line.split()[1]
                if name not in names:
                    names.append(name)
                current = names.index(name)
            elif line.startswith("f "):
                faces.append([p.split("/")[:2] for p in line.split()[1:4]])
                mats.append(current)
    v = np.array(v, dtype=np.float64)
    vt = np.array(vt, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64) - 1
    tri_v, tri_t = faces[:, :, 0].copy(), faces[:, :, 1].copy()
    tri_mat = np.array(mats, dtype=np.int32)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache, v=v, vt=vt, tri_v=tri_v, tri_t=tri_t, tri_mat=tri_mat, names=np.array(names))
    return v, vt, tri_v, tri_t, tri_mat, names


def texture_path(scan_dir, name):
    return os.path.join(scan_dir, TEXTURE.format(name))


def triangle_normals(v, tri_v):
    p = v[tri_v]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2 * area, 1e-12)[:, None], area


def barycentric_samples(n, rng):
    """n uniformly distributed barycentric weights, shape (n, 3)."""
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    return np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)


def save_rgb(path, array):
    Image.fromarray(array).save(path)
