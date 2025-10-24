# panogeo

Detect and geolocate people from equirectangular 360° panoramas. Includes calibration from pixel↔geo pairs.

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install .[raster]   # DEM support via rasterio
pip install .[geo]      # map rendering via geopandas/contextily
```

## CLI

```bash
panogeo shift-pano --in-dir <images> --out-dir <shifted> --degrees 180
panogeo detect --images-dir <images> --output-dir <out> --model yolov8s.pt
panogeo calibrate --calib-csv <calib_points.csv> --cam-lat <lat> --cam-lon <lon> --output-dir <out>
panogeo geolocate --detections-csv <detections_all.csv> --calib <calibration_cam2enu.npz> --output-dir <out> [--dem <dem.tif>]
panogeo map --geo-csv <all_people_geo_calibrated.csv> --out-html <map.html>
```

See `360detection.ipynb` for the original workflow this library is based on.
