from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from pyproj import CRS, Transformer


@dataclass
class GeoContext:
    cam_lat: float
    cam_lon: float
    camera_alt_m: float
    ground_alt_m: float
    ecef_ref: np.ndarray
    R_ecef2enu: np.ndarray
    R_enu2ecef: np.ndarray
    ecef_from_llh: Transformer
    llh_from_ecef: Transformer


def _make_transformers() -> Tuple[Transformer, Transformer]:
    crs_geod = CRS.from_epsg(4979)
    crs_ecef = CRS.from_epsg(4978)
    ecef_from_llh = Transformer.from_crs(crs_geod, crs_ecef, always_xy=True)
    llh_from_ecef = Transformer.from_crs(crs_ecef, crs_geod, always_xy=True)
    return ecef_from_llh, llh_from_ecef


def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array([
        [-so,          co,         0.0],
        [-sl * co,    -sl * so,    cl ],
        [ cl * co,     cl * so,    sl ],
    ], dtype=float)


def llh_to_ecef(lon: float, lat: float, h: float, ecef_from_llh: Transformer) -> np.ndarray:
    x, y, z = ecef_from_llh.transform(lon, lat, h)
    return np.array([x, y, z], dtype=float)


def build_context(cam_lat: float, cam_lon: float, camera_alt_m: float, ground_alt_m: float) -> GeoContext:
    ecef_from_llh, llh_from_ecef = _make_transformers()
    ecef_ref = llh_to_ecef(cam_lon, cam_lat, ground_alt_m + camera_alt_m, ecef_from_llh)
    R_ecef2enu = enu_basis(cam_lat, cam_lon)
    R_enu2ecef = R_ecef2enu.T
    return GeoContext(
        cam_lat=cam_lat,
        cam_lon=cam_lon,
        camera_alt_m=camera_alt_m,
        ground_alt_m=ground_alt_m,
        ecef_ref=ecef_ref,
        R_ecef2ecef=None,  # kept for compatibility; not used
        R_ecef2enu=R_ecef2enu,
        R_enu2ecef=R_enu2ecef,
        ecef_from_llh=ecef_from_llh,
        llh_from_ecef=llh_from_ecef,
    )


def enu_to_llh(ctx: GeoContext, e: float, n: float, u: float) -> Tuple[float, float, float]:
    v_ecef = ctx.R_enu2ecef @ np.array([e, n, u], dtype=float)
    P = ctx.ecef_ref + v_ecef
    lon, lat, h = ctx.llh_from_ecef.transform(P[0], P[1], P[2])
    return float(lon), float(lat), float(h)
