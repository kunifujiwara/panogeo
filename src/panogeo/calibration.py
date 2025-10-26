from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
import pandas as pd

from .geometry import pixel_to_cam_ray
from .geodesy import build_context, llh_to_ecef, enu_basis
from pyproj import Transformer, CRS


@dataclass
class CalibrationResult:
    R_cam2enu: np.ndarray
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    cam_lat: float
    cam_lon: float
    camera_alt_m: float


def _matrix_to_ypr(R: np.ndarray) -> Tuple[float, float, float]:
    f = R @ np.array([1.0, 0.0, 0.0])
    u = R @ np.array([0.0, 1.0, 0.0])
    yaw = math.degrees(math.atan2(f[0], f[1]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
    roll = math.degrees(math.atan2(u[0] * f[1] - u[1] * f[0], u[2]))
    return yaw, pitch, roll


def _ecef_context(cam_lat: float, cam_lon: float, camera_alt_m: float, ground_alt_m: float):
    crs_geod = CRS.from_epsg(4979)
    crs_ecef = CRS.from_epsg(4978)
    ecef_from_llh = Transformer.from_crs(crs_geod, crs_ecef, always_xy=True)
    x, y, z = ecef_from_llh.transform(cam_lon, cam_lat, ground_alt_m + camera_alt_m)
    R_ecef2enu = enu_basis(cam_lat, cam_lon)
    return np.array([x, y, z], dtype=float), R_ecef2enu, ecef_from_llh


def _world_llh_to_enu_vec(lon: float, lat: float, h: float, ecef_ref: np.ndarray, R_ecef2enu: np.ndarray, ecef_from_llh: Transformer) -> np.ndarray:
    X, Y, Z = ecef_from_llh.transform(lon, lat, h)
    v_ecef = np.array([X, Y, Z], dtype=float) - ecef_ref
    v_enu = R_ecef2enu @ v_ecef
    return v_enu


def solve_calibration(
    calib_csv: str,
    cam_lat: float,
    cam_lon: float,
    camera_alt_m: float = 2.0,
    ground_alt_m: float = 0.0,
    default_width: int = 2048,
    default_height: int = 1024,
    optimize_cam_position: bool = True,
) -> CalibrationResult:
    df = pd.read_csv(calib_csv)
    cols = {c.lower(): c for c in df.columns}
    def _col(name: str) -> str:
        for k, v in cols.items():
            if k == name:
                return v
        raise KeyError(name)

    u_col = _col("u_px")
    v_col = _col("v_px")
    lon_col = _col("lon")
    lat_col = _col("lat")
    alt_col = cols.get("alt_m", None)
    W_col = cols.get("w", None)
    H_col = cols.get("h", None)

    # Precompute camera rays and store world LLH targets
    cam_vecs: List[np.ndarray] = []
    world_llh: List[Tuple[float, float, float]] = []

    for r in df.itertuples(index=False):
        W = int(getattr(r, W_col, default_width)) if W_col else default_width
        H = int(getattr(r, H_col, default_height)) if H_col else default_height
        u = float(getattr(r, u_col))
        v = float(getattr(r, v_col))
        lon = float(getattr(r, lon_col))
        lat = float(getattr(r, lat_col))
        alt = float(getattr(r, alt_col)) if alt_col and not pd.isna(getattr(r, alt_col)) else ground_alt_m

        cam_v = pixel_to_cam_ray(u, v, W, H)
        cam_vecs.append(cam_v)
        world_llh.append((lon, lat, alt))

    if len(cam_vecs) < 2:
        raise ValueError("Need at least 2 calibration pairs")

    cam_mat = np.stack(cam_vecs, axis=1)  # 3 x N

    def _compute_rotation_and_residuals(lat: float, lon: float, alt_cam: float) -> Tuple[np.ndarray, np.ndarray]:
        ecef_ref, R_ecef2enu, ecef_from_llh = _ecef_context(lat, lon, alt_cam, ground_alt_m)
        enu_vecs: List[np.ndarray] = []
        for (lon_i, lat_i, alt_i) in world_llh:
            enu_v = _world_llh_to_enu_vec(lon_i, lat_i, alt_i, ecef_ref, R_ecef2enu, ecef_from_llh)
            n = float(np.linalg.norm(enu_v))
            if n < 1e-9:
                enu_vecs.append(np.array([0.0, 0.0, 0.0], dtype=float))
            else:
                enu_vecs.append(enu_v / n)
        enu_mat = np.stack(enu_vecs, axis=1)  # 3 x N
        Hmat = cam_mat @ enu_mat.T
        U, S, Vt = np.linalg.svd(Hmat)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        aligned = R @ cam_mat
        dots = np.sum(aligned * enu_mat, axis=0)
        dots = np.clip(dots, -1.0, 1.0)
        residuals = 1.0 - dots  # prefer alignment (dot->1)
        return R, residuals

    lat0, lon0, alt0 = float(cam_lat), float(cam_lon), float(camera_alt_m)

    if optimize_cam_position:
        def fun(x: np.ndarray) -> np.ndarray:
            R_tmp, res = _compute_rotation_and_residuals(float(x[0]), float(x[1]), float(x[2]))
            return res

        # Reasonable bounds: small box around initial guess for lat/lon, altitude within +/- 20 m
        # If needed, these can be expanded by the caller later.
        lat_eps = 0.001  # ~111 m * 0.001 ≈ 0.111 km in latitude
        lon_eps = 0.001  # ~111 m * cos(lat)
        alt_eps = 20.0
        lower = np.array([lat0 - lat_eps, lon0 - lon_eps, alt0 - alt_eps], dtype=float)
        upper = np.array([lat0 + lat_eps, lon0 + lon_eps, alt0 + alt_eps], dtype=float)
        res_ls = least_squares(fun, x0=np.array([lat0, lon0, alt0], dtype=float), bounds=(lower, upper), method="trf")
        opt_lat, opt_lon, opt_alt = [float(v) for v in res_ls.x]
    else:
        opt_lat, opt_lon, opt_alt = lat0, lon0, alt0

    R_cam2enu, _ = _compute_rotation_and_residuals(opt_lat, opt_lon, opt_alt)
    yaw, pitch, roll = _matrix_to_ypr(R_cam2enu)
    return CalibrationResult(
        R_cam2enu=R_cam2enu,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        cam_lat=opt_lat,
        cam_lon=opt_lon,
        camera_alt_m=opt_alt,
    )


def save_calibration(npz_path: str, calib: CalibrationResult, cam_lat: float, cam_lon: float, camera_alt_m: float, ground_alt_m: float) -> None:
    np.savez(
        npz_path,
        R_cam2enu=calib.R_cam2enu,
        CAM_LAT=cam_lat,
        CAM_LON=cam_lon,
        CAMERA_ALT_M=camera_alt_m,
        GROUND_ALT_M=ground_alt_m,
        YAW_deg=calib.yaw_deg,
        PITCH_deg=calib.pitch_deg,
        ROLL_deg=calib.roll_deg,
    )
