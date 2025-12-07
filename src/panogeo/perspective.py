from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from math import isfinite

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .geodesy import build_context, enu_to_llh, enu_basis, llh_to_ecef  # type: ignore
from pyproj import CRS, Transformer  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


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


def _google_elevation(locations: np.ndarray, api_key: Optional[str]) -> Optional[np.ndarray]:
    """
    locations: Nx2 array [lat, lon]
    Returns elevations in meters as (N,) or None on failure/unavailable.
    """
    if api_key is None or requests is None:
        return None
    try:
        url = "https://maps.googleapis.com/maps/api/elevation/json"
        elevations: list[float] = []
        # API allows up to 512 locations per request. Batch rows to be safe.
        BATCH = 256
        for i in range(0, locations.shape[0], BATCH):
            chunk = locations[i : i + BATCH]
            loc_str = "|".join(f"{float(lat)},{float(lon)}" for lat, lon in chunk)
            r = requests.get(url, params={"locations": loc_str, "key": api_key}, timeout=15.0)
            r.raise_for_status()
            data = r.json()
            if str(data.get("status", "")).upper() != "OK":
                return None
            for res in data.get("results", []):
                elevations.append(float(res.get("elevation", float("nan"))))
        elev = np.asarray(elevations, dtype=float)
        if elev.shape[0] != locations.shape[0]:
            return None
        return elev
    except Exception:
        return None


def _fit_plane_least_squares(E: np.ndarray, N: np.ndarray, U: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Fit plane U = a*E + b*N + c. Returns (a, b, c) or None if ill-conditioned.
    """
    A = np.stack([E, N, np.ones_like(E)], axis=1)
    try:
        # Solve min ||A x - U||_2
        x, *_ = np.linalg.lstsq(A, U, rcond=None)
        a, b, c = float(x[0]), float(x[1]), float(x[2])
        if not (isfinite(a) and isfinite(b) and isfinite(c)):
            return None
        return a, b, c
    except Exception:
        return None


def _build_plane_basis(a: float, b: float, c: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given plane U = a*E + b*N + c, build an orthonormal basis {u_hat, v_hat} on the plane and origin P0.
    - P0 chosen at (E=0, N=0, U=c)
    - u_hat along projected +E axis
    - v_hat = n x u_hat
    Returns (P0, u_hat, v_hat) in ENU coordinates.
    """
    # Plane normal in ENU: n ∝ [-a, -b, 1]
    n = np.array([-a, -b, 1.0], dtype=float)
    n /= np.linalg.norm(n)
    # Origin on plane
    P0 = np.array([0.0, 0.0, float(c)], dtype=float)
    # Project E-axis onto plane
    e_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    u = e_axis - (e_axis @ n) * n
    un = np.linalg.norm(u)
    if un < 1e-9:
        # Fallback: project N-axis instead
        n_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        u = n_axis - (n_axis @ n) * n
        un = np.linalg.norm(u)
        if un < 1e-9:
            # Degenerate; default to horizontal plane basis
            u_hat = np.array([1.0, 0.0, 0.0], dtype=float)
            v_hat = np.array([0.0, 1.0, 0.0], dtype=float)
            return P0, u_hat, v_hat
    u_hat = u / un
    v_hat = np.cross(n, u_hat)
    v_hat /= np.linalg.norm(v_hat)
    return P0, u_hat, v_hat


def _project_points_to_plane_xy(P: np.ndarray, P0: np.ndarray, u_hat: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
    """
    P: Nx3 ENU points on/near plane. Returns Nx2 plane coordinates [x, y] with respect to (P0, u_hat, v_hat).
    """
    d = P - P0[None, :]
    x = d @ u_hat
    y = d @ v_hat
    return np.stack([x, y], axis=1)


def _reconstruct_from_plane_xy(XY: np.ndarray, P0: np.ndarray, u_hat: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
    """
    XY: Nx2 plane coordinates -> Nx3 ENU points.
    """
    return P0[None, :] + XY[:, 0:1] * u_hat[None, :] + XY[:, 1:2] * v_hat[None, :]


def _recompute_h_with_plane(
    df: pd.DataFrame,
    ref_lat: float,
    ref_lon: float,
    ransac_thresh_m: float,
    google_api_key: Optional[str],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Recompute homography mapping pixels -> plane XY using elevations at calibration points.
    Returns (H, P0, u_hat, v_hat) or None if unavailable.
    """
    cols = {c.lower(): c for c in df.columns}
    try:
        u_col = cols["u_px"]
        v_col = cols["v_px"]
        lon_col = cols["lon"]
        lat_col = cols["lat"]
    except Exception:
        return None
    src = df[[u_col, v_col]].to_numpy(dtype=np.float32)
    if src.shape[0] < 4:
        return None
    lons = df[lon_col].astype(float).to_numpy()
    lats = df[lat_col].astype(float).to_numpy()
    locs = np.stack([lats, lons], axis=1)  # lat, lon
    elev = _google_elevation(locs, google_api_key)
    if elev is None:
        return None
    # Build ENU 3D for points
    ecef_from_llh = _ecef_transformer()
    ecef_ref = llh_to_ecef(ref_lon, ref_lat, 0.0, ecef_from_llh)
    R_ecef2enu = enu_basis(ref_lat, ref_lon)
    P_enu: list[list[float]] = []
    for lo, la, h in zip(lons, lats, elev):
        X = llh_to_ecef(float(lo), float(la), float(h), ecef_from_llh)
        v_ecef = X - ecef_ref
        v_enu = R_ecef2enu @ v_ecef
        P_enu.append([float(v_enu[0]), float(v_enu[1]), float(v_enu[2])])
    P = np.asarray(P_enu, dtype=float)
    # Fit plane U = a*E + b*N + c
    fit = _fit_plane_least_squares(P[:, 0], P[:, 1], P[:, 2])
    if fit is None:
        return None
    a, b, c = fit
    P0, u_hat, v_hat = _build_plane_basis(a, b, c)
    # Project to plane local XY and solve H
    XY = _project_points_to_plane_xy(P, P0, u_hat, v_hat).astype(np.float32)
    _require_cv2()
    H, _status = cv2.findHomography(src, XY, method=cv2.RANSAC, ransacReprojThreshold=float(max(0.1, ransac_thresh_m)))
    if H is None:
        return None
    return H.astype(np.float64), P0.astype(np.float64), u_hat.astype(np.float64), v_hat.astype(np.float64)


def _sample_dem_grid_around_calib(
    df: pd.DataFrame,
    ref_lat: float,
    ref_lon: float,
    margin_m: float,
    n_rows: Optional[int],
    n_cols: Optional[int],
    google_api_key: Optional[str],
    dem_spacing_m: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Sample a regular DEM grid around calibration points.
    Returns (N_axis, E_axis, ELEV) where:
      - N_axis: (n_rows,) northing coordinates in meters
      - E_axis: (n_cols,) easting coordinates in meters
      - ELEV: (n_rows, n_cols) elevations in meters
    """
    if google_api_key is None or requests is None:
        return None
    cols = {c.lower(): c for c in df.columns}
    lon_col = cols.get("lon")
    lat_col = cols.get("lat")
    if lon_col is None or lat_col is None:
        return None
    lons = df[lon_col].astype(float).to_numpy()
    lats = df[lat_col].astype(float).to_numpy()
    # Convert calib to ENU to get bounds
    ecef_from_llh = _ecef_transformer()
    ecef_ref = llh_to_ecef(ref_lon, ref_lat, 0.0, ecef_from_llh)
    R_ecef2enu = enu_basis(ref_lat, ref_lon)
    e_vals: list[float] = []
    n_vals: list[float] = []
    for lo, la in zip(lons, lats):
        X = llh_to_ecef(float(lo), float(la), 0.0, ecef_from_llh)
        v_ecef = X - ecef_ref
        v_enu = R_ecef2enu @ v_ecef
        e_vals.append(float(v_enu[0]))
        n_vals.append(float(v_enu[1]))
    e_min = float(np.min(e_vals)) - float(margin_m)
    e_max = float(np.max(e_vals)) + float(margin_m)
    n_min = float(np.min(n_vals)) - float(margin_m)
    n_max = float(np.max(n_vals)) + float(margin_m)
    # Determine grid size
    if (dem_spacing_m is not None) and (float(dem_spacing_m) > 0):
        e_len = float(e_max - e_min)
        n_len = float(n_max - n_min)
        # +1 to include both edges
        n_rows_calc = int(max(2, int(np.ceil(n_len / float(dem_spacing_m))) + 1))
        n_cols_calc = int(max(2, int(np.ceil(e_len / float(dem_spacing_m))) + 1))
    else:
        n_rows_calc = int(n_rows) if n_rows is not None else 0
        n_cols_calc = int(n_cols) if n_cols is not None else 0
    if not (n_rows_calc >= 2 and n_cols_calc >= 2):
        return None
    # Build regular grid in ENU then convert to lat/lon to query elevation
    N_axis = np.linspace(n_max, n_min, int(n_rows_calc), dtype=float)  # north -> south
    E_axis = np.linspace(e_min, e_max, int(n_cols_calc), dtype=float)  # west -> east
    # Convert grid ENU -> LLH (use U=0 for lat/lon)
    ctx = build_context(float(ref_lat), float(ref_lon), 0.0, 0.0)
    lat_lon_points: list[tuple[float, float]] = []
    for n in N_axis:
        for e in E_axis:
            lon, lat, _h = enu_to_llh(ctx, float(e), float(n), 0.0)
            lat_lon_points.append((float(lat), float(lon)))
    # Query elevation in batches
    elev = _google_elevation(np.asarray(lat_lon_points, dtype=float), google_api_key)
    if elev is None:
        return None
    ELEV = np.asarray(elev, dtype=float).reshape((int(n_rows_calc), int(n_cols_calc)))
    return N_axis, E_axis, ELEV


def _interp_bilinear(E: np.ndarray, N: np.ndarray, N_axis: np.ndarray, E_axis: np.ndarray, ELEV: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation of elevation at arbitrary EN points.
    Clamps to grid edges.
    """
    # Ensure ascending axes for searching
    # N_axis provided north->south (descending). Create ascending for indexing.
    N_axis_asc = np.array(N_axis[::-1], dtype=float)
    E_axis_asc = np.array(E_axis, dtype=float)
    elev_flipN = np.array(ELEV[::-1, :], dtype=float)  # match ascending N
    # Compute fractional indices
    def _interp_point(e: float, n: float) -> float:
        # Clamp within bounds
        e = float(np.clip(e, E_axis_asc[0], E_axis_asc[-1]))
        n = float(np.clip(n, N_axis_asc[0], N_axis_asc[-1]))
        # Find indices
        ie = int(np.searchsorted(E_axis_asc, e) - 1)
        in_ = int(np.searchsorted(N_axis_asc, n) - 1)
        ie = max(0, min(ie, len(E_axis_asc) - 2))
        in_ = max(0, min(in_, len(N_axis_asc) - 2))
        e0, e1 = E_axis_asc[ie], E_axis_asc[ie + 1]
        n0, n1 = N_axis_asc[in_], N_axis_asc[in_ + 1]
        t = 0.0 if e1 == e0 else (e - e0) / (e1 - e0)
        u = 0.0 if n1 == n0 else (n - n0) / (n1 - n0)
        z00 = float(elev_flipN[in_, ie])
        z01 = float(elev_flipN[in_, ie + 1])
        z10 = float(elev_flipN[in_ + 1, ie])
        z11 = float(elev_flipN[in_ + 1, ie + 1])
        z0 = (1.0 - t) * z00 + t * z01
        z1 = (1.0 - t) * z10 + t * z11
        return (1.0 - u) * z0 + u * z1
    out = np.empty_like(E, dtype=float)
    for i in range(E.shape[0]):
        out[i] = _interp_point(float(E[i]), float(N[i]))
    return out


def save_homography(
    npz_path: str,
    H_px2enu: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ground_alt_m: float = 0.0,
    calib_csv: Optional[str] = None,
    google_api_key: Optional[str] = None,
    dem_rows: Optional[int] = None,
    dem_cols: Optional[int] = None,
    dem_margin_m: float = 120.0,
    dem_spacing_m: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)) or ".", exist_ok=True)
    # Optionally recompute H using sloped ground plane from elevations and persist plane params
    P0 = None
    u_hat = None
    v_hat = None
    H_to_save = H_px2enu
    N_axis = None
    E_axis = None
    ELEV = None
    if calib_csv is not None:
        try:
            df = pd.read_csv(calib_csv)
        except Exception:
            df = None  # type: ignore
        if df is not None and len(df) >= 4:
            plane_res = _recompute_h_with_plane(df, float(ref_lat), float(ref_lon), ransac_thresh_m=10.0, google_api_key=google_api_key)
            if plane_res is not None:
                H_to_save, P0, u_hat, v_hat = plane_res
            # Optionally sample a DEM grid for non-planar terrain usage
            # Sample DEM either by explicit rows/cols or by target spacing in meters
            if (
                (dem_spacing_m is not None and float(dem_spacing_m) > 0)
                or (dem_rows is not None and dem_cols is not None and dem_rows >= 2 and dem_cols >= 2)
            ):
                dem = _sample_dem_grid_around_calib(
                    df,
                    float(ref_lat),
                    float(ref_lon),
                    float(dem_margin_m),
                    int(dem_rows) if dem_rows is not None else None,
                    int(dem_cols) if dem_cols is not None else None,
                    google_api_key,
                    float(dem_spacing_m) if dem_spacing_m is not None else None,
                )
                if dem is not None:
                    N_axis, E_axis, ELEV = dem
    if P0 is not None and u_hat is not None and v_hat is not None:
        np.savez(
            npz_path,
            H_PX2ENU=H_to_save,
            PROJECTION="perspective",
            REF_LAT=float(ref_lat),
            REF_LON=float(ref_lon),
            GROUND_ALT_M=float(ground_alt_m),
            PLANE_P0_ENU=P0.astype(np.float64),
            PLANE_U_HAT=u_hat.astype(np.float64),
            PLANE_V_HAT=v_hat.astype(np.float64),
            **(
                {
                    "DEM_N_AXIS": N_axis.astype(np.float64),
                    "DEM_E_AXIS": E_axis.astype(np.float64),
                    "DEM_ELEV_M": ELEV.astype(np.float64),
                }
                if (N_axis is not None and E_axis is not None and ELEV is not None)
                else {}
            ),
        )
    else:
        # Flat (horizontal) plane legacy behavior
        np.savez(
            npz_path,
            H_PX2ENU=H_to_save,
            PROJECTION="perspective",
            REF_LAT=float(ref_lat),
            REF_LON=float(ref_lon),
            GROUND_ALT_M=float(ground_alt_m),
            **(
                {
                    "DEM_N_AXIS": N_axis.astype(np.float64),
                    "DEM_E_AXIS": E_axis.astype(np.float64),
                    "DEM_ELEV_M": ELEV.astype(np.float64),
                }
                if (N_axis is not None and E_axis is not None and ELEV is not None)
                else {}
            ),
        )


def load_homography(npz_path: str) -> Tuple[
    np.ndarray,
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    nz = np.load(npz_path)
    if "H_PX2ENU" in nz:
        H = nz["H_PX2ENU"]
        ref_lat = float(nz.get("REF_LAT", np.nan))
        ref_lon = float(nz.get("REF_LON", np.nan))
        g_alt = float(nz.get("GROUND_ALT_M", 0.0))
        P0 = nz.get("PLANE_P0_ENU")
        Uhat = nz.get("PLANE_U_HAT")
        Vhat = nz.get("PLANE_V_HAT")
        N_axis = nz.get("DEM_N_AXIS")
        E_axis = nz.get("DEM_E_AXIS")
        ELEV = nz.get("DEM_ELEV_M")
        # Normalize if present
        if P0 is not None and Uhat is not None and Vhat is not None:
            P0 = P0.astype(np.float64)
            Uhat = Uhat.astype(np.float64)
            Vhat = Vhat.astype(np.float64)
            # ensure orthonormal (best-effort)
            try:
                Uhat = Uhat / np.linalg.norm(Uhat)
                Vhat = Vhat - (Uhat @ Vhat) * Uhat
                Vhat = Vhat / np.linalg.norm(Vhat)
            except Exception:
                pass
        else:
            P0 = None
            Uhat = None
            Vhat = None
        if N_axis is not None and E_axis is not None and ELEV is not None:
            N_axis = N_axis.astype(np.float64)
            E_axis = E_axis.astype(np.float64)
            ELEV = ELEV.astype(np.float64)
        else:
            N_axis = None
            E_axis = None
            ELEV = None
        return H, ref_lat, ref_lon, g_alt, P0, Uhat, Vhat, N_axis, E_axis, ELEV
    # Backward-compat: older files stored lon/lat homography (less accurate)
    if "H_PX2GEO" in nz:
        return nz["H_PX2GEO"], None, None, None, None, None, None, None, None, None
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

    H, ref_lat, ref_lon, g_alt, P0, Uhat, Vhat, N_axis, E_axis, ELEV = load_homography(homography_npz)
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
            if (P0 is not None) and (Uhat is not None) and (Vhat is not None):
                try:
                    nh = np.cross(Uhat, Vhat)
                    print(f"[persp] plane present: |Uhat|={np.linalg.norm(Uhat):.3f}, |Vhat|={np.linalg.norm(Vhat):.3f}, |n|={np.linalg.norm(nh):.3f}")
                except Exception:
                    print("[persp] plane present")
            if (N_axis is not None) and (E_axis is not None) and (ELEV is not None):
                print(f"[persp] DEM grid present: rows={int(N_axis.shape[0])}, cols={int(E_axis.shape[0])}")
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
        # ENU -> lon/lat with local context, optionally reconstruct 3D on sloped plane
        ctx = build_context(float(ref_lat), float(ref_lon), 0.0, float(g_alt if g_alt is not None else 0.0))
        lons: list[float] = []
        lats: list[float] = []
        east: list[float] = []
        north: list[float] = []
        elevs: list[float] = []
        total = int(coords.shape[0])
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore
                pbar = tqdm(total=total, desc=(progress_desc or "geolocate-persp"), unit="pt")
            except Exception:
                pbar = None
        if (N_axis is not None) and (E_axis is not None) and (ELEV is not None):
            # DEM grid available: use bilinear interpolation for per-point elevation
            # coords currently represent EN parameters from H (plane/tps not applied here)
            e_vec = coords[:, 0].astype(float)
            n_vec = coords[:, 1].astype(float)
            u_vec = _interp_bilinear(e_vec, n_vec, N_axis, E_axis, ELEV)
            for idx in range(coords.shape[0]):
                e = float(e_vec[idx])
                n = float(n_vec[idx])
                u = float(u_vec[idx])
                lon, lat, _h = enu_to_llh(ctx, e, n, u)
                lons.append(lon)
                lats.append(lat)
                east.append(e)
                north.append(n)
                elevs.append(u)
                if pbar is not None:
                    pbar.update(1)
                elif show_progress and (idx % 10000 == 0 or idx + 1 == total):
                    try:
                        print(f"[geolocate-persp] {idx+1}/{total}", flush=True)
                    except Exception:
                        pass
        elif P0 is not None and Uhat is not None and Vhat is not None:
            # coords are plane XY -> reconstruct ENU 3D
            P_rec = _reconstruct_from_plane_xy(coords.astype(float), P0.astype(float), Uhat.astype(float), Vhat.astype(float))
            for idx in range(P_rec.shape[0]):
                e = float(P_rec[idx, 0])
                n = float(P_rec[idx, 1])
                u = float(P_rec[idx, 2])
                lon, lat, _h = enu_to_llh(ctx, e, n, u)
                lons.append(lon)
                lats.append(lat)
                east.append(e)
                north.append(n)
                elevs.append(u)
                if pbar is not None:
                    pbar.update(1)
                elif show_progress and (idx % 10000 == 0 or idx + 1 == total):
                    try:
                        print(f"[geolocate-persp] {idx+1}/{total}", flush=True)
                    except Exception:
                        pass
        else:
            # Legacy flat plane: coords already EN; use U=0
            for idx, (e, n) in enumerate(coords, start=1):
                lon, lat, _h = enu_to_llh(ctx, float(e), float(n), 0.0)
                lons.append(lon)
                lats.append(lat)
                east.append(float(e))
                north.append(float(n))
                elevs.append(0.0)
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
        df["elev_m"] = elevs
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


