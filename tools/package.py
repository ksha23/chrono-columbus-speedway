#!/usr/bin/env python3
"""Pack a built scene into the archives the script downloads.

    python -I package.py SCENE_DIR OUT_DIR [--stamp "text for the archive's README.txt"]

Only files the manifest names go in, so leftovers from earlier builds in SCENE_DIR are not
published. The archives are written the same way every time (sorted, fixed dates and owner), so
the same scene always gives the same hash.

Writes OUT_DIR/speedway_scene_base.tar.gz and SHA256SUMS, and one archive per texture level
that is not part of the base scene.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile

MANIFEST = "speedway_scene.json"
BASE_LEVELS = ("low", "standard")   # texture levels that travel with the scene itself
EPOCH = 1728172800                  # 2024-10-06, the date stamped on every file in the archive


def referenced(scene, doc):
    """(files every scene needs, {level: that level's texture files}), relative to scene."""
    always, by_level = {MANIFEST}, {level: set() for level in doc["texture_levels"]}
    always.update(doc["collision"].values())
    for asset in doc["assets"]:
        for part in asset["parts"]:
            always.add(part["mesh"])
            texture = part.get("texture")
            if not texture:
                continue
            if "<level>" in texture:
                for level in by_level:
                    by_level[level].add(texture.replace("<level>", level))
            else:
                always.add(texture)
    for name in ("LICENSE.txt", "README.txt"):
        if os.path.isfile(os.path.join(scene, name)):
            always.add(name)
    return always, by_level


def write_archive(path, scene, names, extra=None):
    """A gzip tar of names (relative to scene) plus extra {name: bytes}. Returns its SHA-256."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        entries = {name: None for name in names}
        entries.update(extra or {})
        for name in sorted(entries):
            data = entries[name]
            if data is None:
                with open(os.path.join(scene, name), "rb") as f:
                    data = f.read()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = EPOCH
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    with open(path, "wb") as out:
        # mtime=0 keeps the gzip header from carrying the time of packing.
        with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=6, mtime=0, filename="") as gz:
            gz.write(buffer.getvalue())
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("out")
    parser.add_argument("--stamp", default="", help="text recorded in the archive's README.txt: what this copy was built from")
    args = parser.parse_args()

    with open(os.path.join(args.scene, MANIFEST)) as f:
        doc = json.load(f)
    always, by_level = referenced(args.scene, doc)
    missing = [n for n in sorted(always | set().union(*by_level.values())) if not os.path.isfile(os.path.join(args.scene, n))]
    if missing:
        raise SystemExit(f"{len(missing)} files the manifest names are missing, for example {missing[:3]}")

    os.makedirs(args.out, exist_ok=True)
    readme = ("Columbus 151 Speedway as a PyChrono scene.\n"
              "Metres, Z up, x east, y north. z is elevation above sea level.\n"
              "See https://github.com/ksha23/chrono-columbus-speedway for what is in here and how it was made.\n")
    if args.stamp:
        readme += args.stamp.strip() + "\n"

    sums = []
    base = set(always)
    for level in BASE_LEVELS:
        base |= by_level.get(level, set())
    name = "speedway_scene_base.tar.gz"
    sums.append((write_archive(os.path.join(args.out, name), args.scene, base, {"README.txt": readme.encode()}), name))
    for level, files in by_level.items():
        if level in BASE_LEVELS:
            continue
        name = f"speedway_textures_{level}.tar.gz"
        sums.append((write_archive(os.path.join(args.out, name), args.scene, files), name))

    with open(os.path.join(args.out, "SHA256SUMS"), "w") as f:
        for digest, name in sums:
            f.write(f"{digest}  {name}\n")
            print(f"{digest}  {name}  {os.path.getsize(os.path.join(args.out, name)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
