from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .geometry import pixel_to_cam_ray
from .geodesy import build_context, enu_to_llh

try:
    import rasterio  # optional
    from pyproj import Transformer
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore
    Transformer = None  # type: ignore


@dataclass
class Calib:
    R_cam2enu: np.ndarray
    cam_lat: float
    cam_lon: float
    camera_alt_m: float
    ground_alt_m: float


def load_calibration(npz_path: str) -> Calib:
    nz = np.load(npz_path)
    return Calib(
        R_cam2enu=nz["R_cam2enu"],
        cam_lat=float(nz["CAM_LAT"]),
        cam_lon=float(nz["CAM_LON"]),
        camera_alt_m=float(nz["CAMERA_ALT_M"]),
        ground_alt_m=float(nz["GROUND_ALT_M"]),
    )


def _dem_sampler(dem_path: Optional[str], ground_default: float):
    if dem_path and rasterio is not None and os.path.exists(dem_path):
        dem = rasterio.open(dem_path)
        dem_crs = dem.crs
        from pyproj import Transformer as _T
        to_dem = _T.from_crs("EPSG:4326", dem_crs, always_xy=True)

        def terrain_h(lon: float, lat: float, default: float = ground_default) -> float:
            x, y = to_dem.transform(lon, lat)
            try:
                for val in dem.sample([(x, y)]):
                    h = float(val[0])
                    if np.isnan(h):
                        return default
                    return h
            except Exception:
                return default
        return terrain_h

    def terrain_h(lon: float, lat: float, default: float = ground_default) -> float:
        return default
    return terrain_h


def geolocate_detections(
    detections_csv: str,
    calibration_npz: str,
    output_dir: str,
    dem_path: Optional[str] = None,
    max_range_m: float = 80.0,
    step_m: float = 1.0,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    det = pd.read_csv(detections_csv)
    if det.empty:
        raise ValueError("Detections CSV has no rows")

    calib = load_calibration(calibration_npz)
    ctx = build_context(calib.cam_lat, calib.cam_lon, calib.camera_alt_m, calib.ground_alt_m)
    terrain_h = _dem_sampler(dem_path, calib.ground_alt_m)

    def intersect(u_px: float, v_px: float, W: int, H: int):
        v_cam = pixel_to_cam_ray(u_px, v_px, W, H)
        v_enu = calib.R_cam2enu @ v_cam
        n = np.linalg.norm(v_enu)
        if n == 0:
            return None
        v_enu = v_enu / n

        if dem_path and rasterio is not None and os.path.exists(dem_path):
            prev = (0.0, 0.0, 0.0)
            prev_lon, prev_lat, prev_h = enu_to_llh(ctx, 0.0, 0.0, 0.0)
            prev_terr = terrain_h(prev_lon, prev_lat, calib.ground_alt_m)
            dist = 0.0
            while dist <= max_range_m:
                dist += step_m
                E = v_enu[0] * dist
                N = v_enu[1] * dist
                U = v_enu[2] * dist
                lon, lat, h = enu_to_llh(ctx, E, N, U)
                th = terrain_h(lon, lat, calib.ground_alt_m)
                if h <= th:
                    denom = ((prev_h - prev_terr) - (h - th))
                    alpha = 0.0 if abs(denom) < 1e-6 else np.clip((prev_h - prev_terr) / denom, 0.0, 1.0)
                    E2 = (1 - alpha) * (v_enu[0] * (dist - step_m)) + alpha * (v_enu[0] * dist)
                    N2 = (1 - alpha) * (v_enu[1] * (dist - step_m)) + alpha * (v_enu[1] * dist)
                    U2 = (1 - alpha) * (v_enu[2] * (dist - step_m)) + alpha * (v_enu[2] * dist)
                    Lon, Lat, Hh = enu_to_llh(ctx, E2, N2, U2)
                    rng = math.sqrt(E2 * E2 + N2 * N2 + U2 * U2)
                    return E2, N2, U2, Lon, Lat, th, rng
                prev = (E, N, U)
                prev_lon, prev_lat, prev_h, prev_terr = lon, lat, h, th
            return None
        else:
            if abs(v_enu[2]) < 1e-8:
                return None
            t = -calib.camera_alt_m / v_enu[2]
            if t <= 0 or t > max_range_m:
                return None
            E = v_enu[0] * t
            N = v_enu[1] * t
            U = v_enu[2] * t
            lon, lat, h = enu_to_llh(ctx, E, N, U)
            rng = math.sqrt(E * E + N * N + U * U)
            return E, N, U, lon, lat, calib.ground_alt_m, rng

    east, north, up, lons, lats, rngs = [], [], [], [], [], []
    for r in det.itertuples():
        hit = intersect(float(r.u_px), float(r.v_px), int(r.W), int(r.H))
        if hit is None:
            east.append(np.nan); north.append(np.nan); up.append(np.nan)
            lons.append(np.nan); lats.append(np.nan); rngs.append(np.nan)
        else:
            E, N, U, Lon, Lat, Th, R = hit
            east.append(E); north.append(N); up.append(U)
            lons.append(Lon); lats.append(Lat); rngs.append(R)

    det["east_m"] = east
    det["north_m"] = north
    det["up_m"] = up
    det["lon"] = lons
    det["lat"] = lats
    det["range_m"] = rngs
    det = det.dropna(subset=["east_m", "north_m"]).reset_index(drop=True)

    xy_csv = os.path.join(output_dir, "all_people_xy_calibrated.csv")
    geo_csv = os.path.join(output_dir, "all_people_geo_calibrated.csv")
    det.to_csv(xy_csv, index=False)
    det.to_csv(geo_csv, index=False)
    return xy_csv, geo_csv
