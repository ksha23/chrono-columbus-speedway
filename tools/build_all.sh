#!/bin/sh
# Build the published scene from the drone scan, start to finish.
#
#   sh build_all.sh SCAN_DIR REFERENCE_DIR WORK_DIR SCENE_DIR OUT_DIR
#
# SCAN_DIR       the unpacked WebODM export: odm_textured_model_geo.obj and its textures
# REFERENCE_DIR  what fetch_reference.sh downloaded
# WORK_DIR       scratch space for intermediate rasters, about 2.5 GB
# SCENE_DIR      where the scene is assembled
# OUT_DIR        where the archives and SHA256SUMS are written
#
# PYTHON picks the interpreter. It needs numpy, scipy, Pillow and, for the cone model, PyChrono.
# Peak memory is about 15 GB, in the shadow step. The whole run takes 26 minutes on an M4 Pro.
# Every step is seeded, so the same scan and reference data give the same archive, byte for byte.
set -e
scan="${1:?scan directory}"; ref="${2:?reference directory}"; work="${3:?work directory}"
scene="${4:?scene directory}"; out="${5:?output directory}"
here="$(cd "$(dirname "$0")" && pwd)"
py="${PYTHON:-python3}"
run() { echo "== $1"; "$py" -I "$here/$@"; }

run ortho.py "$scan" "$work" 0.1                  # the scan from above, in its own coordinates
run georef.py "$work" "$ref"                      # where it sits: whole-site match to aerial imagery
run georef_local.py "$work" "$ref"                # its scale and heading, window by window
run register.py "$work" "$ref"                    # its dome and tilt, against the lidar
run register_local.py "$work" "$ref"              # horizontal fit on the shape of the ground
run fit_height.py "$work" "$ref"                  # height correction, scored on held-out blocks
run rasterize.py "$scan" "$work" 0.05             # photo and top surface in the scene frame
run shadows.py "$work" "$work/shadows" --full     # photographed shadows relit
run survey.py "$work" "$ref"                      # trees, buildings, vehicles
run build_scene.py "$work" "$ref" "$scene" --scan "$scan" --levels low,standard --save-photo
run check_scene.py "$scene"                       # trees clear of the road, cones on it, ground watertight
run audit_ground.py "$work" "$work/audit"         # what is still dark on the pavement, for a look
run audit_lines.py "$work" "$scene" "$work/lines" # every painted line from above, kinks circled
run package.py "$scene" "$out"
