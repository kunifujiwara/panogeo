from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .geodesy import build_context, enu_to_llh, enu_basis, llh_to_ecef  # type: ignore
from pyproj import CRS, Transformer  # type: ignore

def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV not available. Install with: pip install opencv-python")


def _ecef_transformer() -> Transformer:
    crs_geod = CRS.from_epsg(4979)
    crs_ecef = CRS.from_epsg(4978)
    return Transformer.from_crs(crs_geod, crs_ecef, always_xy=True)


def solve_homography_from_csv(calib_csv: str, ransac_thresh_m: float = 10.0) -> Tuple[np.ndarray, float, float]:
    """
    Estimate a pixel->ENU homography H (maps [u, v, 1]^T -> [E, N, 1]^T in meters) from a CSV of pairs.
    The ENU frame is centered at the mean of provided lon/lat to minimize distortion.
    Required columns (case-insensitive): u_px, v_px, lon, lat
    """
    _require_cv2()
    df = pd.read_csv(calib_csv)
    if df.empty:
        raise ValueError("Calibration CSV has no rows")
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

    src = df[[u_col, v_col]].to_numpy(dtype=np.float32)
    if src.shape[0] < 4:
        raise ValueError("Need at least 4 pairs to compute homography")
    # Compute local ENU for geo points
    lons = df[lon_col].astype(float).to_numpy()
    lats = df[lat_col].astype(float).to_numpy()
    ref_lon = float(np.mean(lons))
    ref_lat = float(np.mean(lats))
    ecef_from_llh = _ecef_transformer()
    ecef_ref = llh_to_ecef(ref_lon, ref_lat, 0.0, ecef_from_llh)
    R_ecef2enu = enu_basis(ref_lat, ref_lon)
    en_points = []
    for lo, la in zip(lons, lats):
        X = llh_to_ecef(float(lo), float(la), 0.0, ecef_from_llh)
        v_ecef = X - ecef_ref
        v_enu = R_ecef2enu @ v_ecef
        en_points.append([float(v_enu[0]), float(v_enu[1])])  # E, N
    dst = np.asarray(en_points, dtype=np.float32)
    # Robust fit with RANSAC in destination units (meters)
    H, status = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=float(max(0.1, ransac_thresh_m)))
    if H is None:
        raise RuntimeError("cv2.findHomography failed")
    return H.astype(np.float64), ref_lat, ref_lon


def save_homography(npz_path: str, H_px2enu: np.ndarray, ref_lat: float, ref_lon: float, ground_alt_m: float = 0.0) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)) or ".", exist_ok=True)
    np.savez(npz_path, H_PX2ENU=H_px2enu, PROJECTION="perspective", REF_LAT=float(ref_lat), REF_LON=float(ref_lon), GROUND_ALT_M=float(ground_alt_m))


def load_homography(npz_path: str) -> Tuple[np.ndarray, Optional[float], Optional[float], Optional[float]]:
    nz = np.load(npz_path)
    if "H_PX2ENU" in nz:
        H = nz["H_PX2ENU"]
        ref_lat = float(nz.get("REF_LAT", np.nan))
        ref_lon = float(nz.get("REF_LON", np.nan))
        g_alt = float(nz.get("GROUND_ALT_M", 0.0))
        return H, ref_lat, ref_lon, g_alt
    # Backward-compat: older files stored lon/lat homography (less accurate)
    if "H_PX2GEO" in nz:
        return nz["H_PX2GEO"], None, None, None
    raise KeyError("No homography matrix found in npz (expected H_PX2ENU or H_PX2GEO)")


def pixel_to_geo(points_uv: np.ndarray, H_px2enu_or_geo: np.ndarray) -> np.ndarray:
    """
    Map Nx2 pixel points (u,v) -> Nx2 coordinates using homography.
    If H is ENU homography, returns EN in meters; if H is lon/lat homography (legacy), returns lon/lat.
    """
    _require_cv2()
    if points_uv.ndim != 2 or points_uv.shape[1] != 2:
        raise ValueError("points_uv must be Nx2 array")
    pts = points_uv.astype(np.float32).reshape((-1, 1, 2))
    mapped = cv2.perspectiveTransform(pts, H_px2enu_or_geo.astype(np.float32))
    return mapped.reshape((-1, 2)).astype(np.float64)


def geolocate_detections_perspective(
    detections_csv: str,
    homography_npz: str,
    output_dir: str,
    debug: bool = False,
    calib_csv: Optional[str] = None,
    gate_margin_m: float = 150.0,
    drop_outside: bool = True,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Geolocate bottom-center detection points from a detections CSV using a pixel->geo homography.
    Adds lon/lat columns and saves two CSVs for consistency with pano pipeline.
    Returns (xy_csv, geo_csv) paths (identical content; retained for downstream compatibility).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(detections_csv)
    if df.empty:
        raise ValueError("Detections CSV has no rows")
    cols = {c.lower(): c for c in df.columns}

    def _col(name: str) -> str:
        for k, v in cols.items():
            if k == name:
                return v
        raise KeyError(name)

    u_col = _col("u_px")
    v_col = _col("v_px")
    # Optional crop offset columns (recorded by tracker when center_crop is used)
    xoff_col = cols.get("x_offset_px")
    yoff_col = cols.get("y_offset_px")

    H, ref_lat, ref_lon, g_alt = load_homography(homography_npz)
    if debug:
        try:
            import os as _os
            sz = _os.path.getsize(homography_npz)
        except Exception:
            sz = -1
        typ = "ENU" if (ref_lat is not None and ref_lon is not None) else "GEO-legacy"
        print(f"[persp] homography file={homography_npz} ({sz} bytes), type={typ}")
        if ref_lat is not None and ref_lon is not None:
            print(f"[persp] ref_lat={ref_lat:.8f}, ref_lon={ref_lon:.8f}, ground_alt_m={float(g_alt) if g_alt is not None else 0.0:.2f}")
        else:
            print("[persp] WARNING: legacy lon/lat homography in use; re-run calibrate-persp to regenerate.")
    uv = df[[u_col, v_col]].to_numpy(dtype=np.float64)
    # If offsets exist, convert crop-relative pixels (u,v) -> full-frame pixels expected by homography
    if (xoff_col is not None) and (yoff_col is not None):
        try:
            xoff = df[xoff_col].to_numpy(dtype=np.float64)
            yoff = df[yoff_col].to_numpy(dtype=np.float64)
            if xoff.size == uv.shape[0] and yoff.size == uv.shape[0]:
                uv[:, 0] += xoff
                uv[:, 1] += yoff
                if debug:
                    try:
                        ux, uy = float(np.nanmean(xoff)), float(np.nanmean(yoff))
                        print(f"[persp] applied crop offsets: mean x_offset={ux:.1f}, y_offset={uy:.1f}")
                    except Exception:
                        pass
        except Exception as _e:
            if debug:
                print(f"[persp] WARNING: failed applying crop offsets: {_e}")
    coords = pixel_to_geo(uv, H)
    if (ref_lat is not None) and (ref_lon is not None):
        # ENU -> lon/lat with local context
        ctx = build_context(float(ref_lat), float(ref_lon), 0.0, float(g_alt if g_alt is not None else 0.0))
        lons = []
        lats = []
        east = []
        north = []
        total = int(coords.shape[0])
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore
                pbar = tqdm(total=total, desc=(progress_desc or "geolocate-persp"), unit="pt")
            except Exception:
                pbar = None
        for idx, (e, n) in enumerate(coords, start=1):
            lon, lat, _h = enu_to_llh(ctx, float(e), float(n), 0.0)
            lons.append(lon)
            lats.append(lat)
            east.append(float(e))
            north.append(float(n))
            if pbar is not None:
                pbar.update(1)
            elif show_progress and (idx % 10000 == 0 or idx == total):
                try:
                    print(f"[geolocate-persp] {idx}/{total}", flush=True)
                except Exception:
                    pass
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass
        df["lon"] = lons
        df["lat"] = lats
        df["east_m"] = east
        df["north_m"] = north
        # Optional gating by calibration extent
        if calib_csv is not None:
            try:
                cdf = pd.read_csv(calib_csv)
                cols = {c.lower(): c for c in cdf.columns}
                u_col_c = cols.get("u_px")
                v_col_c = cols.get("v_px")
                lon_col_c = cols.get("lon")
                lat_col_c = cols.get("lat")
                if (lon_col_c is not None) and (lat_col_c is not None):
                    lons_c = cdf[lon_col_c].astype(float).to_numpy()
                    lats_c = cdf[lat_col_c].astype(float).to_numpy()
                    # Convert calib points into same ENU frame
                    e_list = []
                    n_list = []
                    for lo, la in zip(lons_c, lats_c):
                        Xc = llh_to_ecef(float(lo), float(la), 0.0, _ecef_transformer())
                        v_ecef_c = Xc - llh_to_ecef(ref_lon, ref_lat, 0.0, _ecef_transformer())
                        v_enu_c = enu_basis(ref_lat, ref_lon) @ v_ecef_c
                        e_list.append(float(v_enu_c[0]))
                        n_list.append(float(v_enu_c[1]))
                    e_min = float(np.min(e_list)) - float(gate_margin_m)
                    e_max = float(np.max(e_list)) + float(gate_margin_m)
                    n_min = float(np.min(n_list)) - float(gate_margin_m)
                    n_max = float(np.max(n_list)) + float(gate_margin_m)
                    mask_inside = (df["east_m"].astype(float) >= e_min) & (df["east_m"].astype(float) <= e_max) & (df["north_m"].astype(float) >= n_min) & (df["north_m"].astype(float) <= n_max)
                    if debug:
                        try:
                            total = int(len(df))
                            inside = int(mask_inside.sum())
                            print(f"[persp] gating by calib bbox E[{e_min:.1f},{e_max:.1f}] N[{n_min:.1f},{n_max:.1f}] margin={gate_margin_m:.1f} -> keep {inside}/{total}")
                        except Exception:
                            pass
                    if drop_outside:
                        df = df[mask_inside].reset_index(drop=True)
                    else:
                        df.loc[~mask_inside, ["lon", "lat"]] = np.nan
            except Exception as ge:
                if debug:
                    print("[persp] gating failed:", ge)
        if debug:
            try:
                print(f"[persp] ENU stats: E:[{np.nanmin(east):.2f},{np.nanmax(east):.2f}] N:[{np.nanmin(north):.2f},{np.nanmax(north):.2f}]")
                print(f"[persp] lon:[{np.nanmin(lons):.8f},{np.nanmax(lons):.8f}] lat:[{np.nanmin(lats):.8f},{np.nanmax(lats):.8f}]")
            except Exception:
                pass
    else:
        # Legacy: coords is lon/lat already
        df["lon"] = coords[:, 0]
        df["lat"] = coords[:, 1]
    xy_csv = os.path.join(output_dir, "all_people_xy_calibrated.csv")
    geo_csv = os.path.join(output_dir, "all_people_geo_calibrated.csv")
    df.to_csv(xy_csv, index=False)
    df.to_csv(geo_csv, index=False)
    return xy_csv, geo_csv


