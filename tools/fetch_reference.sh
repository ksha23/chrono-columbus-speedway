#!/bin/sh
# Download the public USGS rasters the scene is built on.
#
#   sh fetch_reference.sh REFERENCE_DIR
#
# Two rectangles in UTM zone 16N, each with 3DEP bare-earth elevation (metres above sea level,
# NAVD88, 32-bit float) and NAIP aerial imagery (natural colour):
#
#   dem_utm.tif, naip_utm.png       1300 x 1000 m, what the scan is registered against.
#                                   Elevation at 1 m per pixel, imagery at 0.5 m.
#   dem_scene.tif, naip_scene.png   1280 x 1024 m, the scene rectangle in layout.py, both at
#                                   0.5 m per pixel. The ground mesh and the surround come from these.
#
# Both services mosaic their newest data, so a later download can differ from the one the
# published scene was built with (fetched 2026-10-06).
set -e
out="${1:?usage: fetch_reference.sh REFERENCE_DIR}"
mkdir -p "$out"
dem="https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
naip="https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
float="format=tiff&pixelType=F32&interpolation=RSP_BilinearInterpolation&f=image"

box="bbox=328800,4793900,330100,4794900&bboxSR=26916&imageSR=26916"
curl -sS -m 300 -o "$out/dem_utm.tif" "$dem?$box&size=1300,1000&$float"
curl -sS -m 300 -o "$out/naip_utm.png" "$naip?$box&size=2600,2000&format=png&f=image"

box="bbox=328810,4794048,330090,4795072&bboxSR=26916&imageSR=26916&size=2560,2048"
curl -sS -m 300 -o "$out/dem_scene.tif" "$dem?$box&$float"
curl -sS -m 300 -o "$out/naip_scene.png" "$naip?$box&format=png&f=image"
ls -l "$out"
