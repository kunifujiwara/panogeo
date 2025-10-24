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

## Jupyter Calibration UI (make calib CSV by clicks)

Optional UI extras:

```bash
pip install .[ui]
```

In a notebook:

```python
from panogeo import launch_calibration_ui

# Path to a panoramic equirect image and a map center (lat, lon)
ui = launch_calibration_ui(
    pano_path="data/images/IMG_20250906_181558_00_263.jpg",
    map_center=(35.681236, 139.767125),  # e.g., Tokyo Station
    map_zoom=18,
    display_width_px=900,
    default_alt_m=0.0,
)
ui.display()  # or simply `ui` as last expression in a cell

# Instructions:
# - Click the pano image to set pixel (u,v)
# - Click the map to set (lon, lat)
# - When both are picked, a pair is added automatically
# - Use the "Save CSV" button to write calibration_points.csv
```

The saved CSV columns are compatible with `panogeo calibrate` and `solve_calibration`:

```
u_px, v_px, lon, lat, W, H[, alt_m]
```