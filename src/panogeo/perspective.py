"""
Monoplotting module for perspective camera georeferencing.

Implements true monoplotting via ray-DEM intersection:
1. Camera calibration using solvePnP (intrinsics + extrinsics)
2. Ray generation from camera through image pixels
3. Ray-DEM intersection for 3D ground point estimation
4. ENU to geographic coordinate conversion

Optimized for processing speed using vectorized numpy operations.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from pyproj import CRS, Transformer  # type: ignore

from .geodesy import build_context, enu_to_llh, enu_to_llh_batch, enu_basis, llh_to_ecef  # type: ignore


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV not available. Install with: pip install opencv-python")


def _ecef_transformer() -> Transformer:
    crs_geod = CRS.from_epsg(4979)
    crs_ecef = CRS.from_epsg(4978)
    return Transformer.from_crs(crs_geod, crs_ecef, always_xy=True)


# -----------------------------------------------------------------------------
# Monoplotting Calibration Data Structure
# -----------------------------------------------------------------------------

@dataclass
class MonoplottingCalibration:
    """Camera calibration for monoplotting.
    
    Attributes:
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix (camera to world/ENU)
        t: 3x1 translation vector (camera position in ENU)
        cam_pos_enu: 3D camera position in ENU frame
        ref_lat: Reference latitude for ENU origin
        ref_lon: Reference longitude for ENU origin
        ref_alt: Reference altitude for ENU origin
        img_width: Image width in pixels
        img_height: Image height in pixels
        distortion: Optional distortion coefficients (k1, k2, p1, p2, k3)
    """
    K: np.ndarray  # 3x3 intrinsic matrix
    R: np.ndarray  # 3x3 rotation matrix (cam to world)
    t: np.ndarray  # 3x1 translation
    cam_pos_enu: np.ndarray  # 3D camera position in ENU
    ref_lat: float
    ref_lon: float
    ref_alt: float
    img_width: int
    img_height: int
    distortion: Optional[np.ndarray] = None


# -----------------------------------------------------------------------------
# Camera Calibration via solvePnP
# -----------------------------------------------------------------------------

def _llh_to_enu_batch(
    lons: np.ndarray,
    lats: np.ndarray,
    alts: np.ndarray,
    ref_lon: float,
    ref_lat: float,
    ref_alt: float,
) -> np.ndarray:
    """Convert lon/lat/alt arrays to ENU coordinates (vectorized)."""
    ecef_from_llh = _ecef_transformer()
    ecef_ref = llh_to_ecef(ref_lon, ref_lat, ref_alt, ecef_from_llh)
    R_ecef2enu = enu_basis(ref_lat, ref_lon)
    
    n = len(lons)
    enu = np.empty((n, 3), dtype=np.float64)
    
    for i in range(n):
        ecef_pt = llh_to_ecef(float(lons[i]), float(lats[i]), float(alts[i]), ecef_from_llh)
        enu[i] = R_ecef2enu @ (ecef_pt - ecef_ref)
    
    return enu


def solve_monoplotting_from_csv(
    calib_csv: str,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
    focal_length_px: Optional[float] = None,
    use_ransac: bool = True,
    ransac_reproj_thresh: float = 8.0,
) -> MonoplottingCalibration:
    """
    Solve camera pose from control point correspondences using solvePnP.
    
    Uses cv2.calibrateCamera to jointly optimize intrinsics and extrinsics
    for best accuracy.
    
    Args:
        calib_csv: CSV with columns: u_px, v_px, lon, lat, and optionally W, H, alt_m
        img_width: Image width in pixels. If None, read from CSV 'W' column.
        img_height: Image height in pixels. If None, read from CSV 'H' column.
        focal_length_px: Focal length in pixels. If None, optimized via calibration.
        use_ransac: Use RANSAC for robust estimation (not used with calibrateCamera)
        ransac_reproj_thresh: Reprojection threshold in pixels
        
    Returns:
        MonoplottingCalibration object with camera intrinsics and extrinsics
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
        raise KeyError(f"Column '{name}' not found")
    
    # Auto-detect image dimensions from CSV if not provided
    if img_width is None:
        w_col = cols.get("w")
        if w_col is not None:
            img_width = int(df[w_col].iloc[0])
        else:
            raise ValueError("img_width not provided and 'W' column not found in CSV")
    
    if img_height is None:
        h_col = cols.get("h")
        if h_col is not None:
            img_height = int(df[h_col].iloc[0])
        else:
            raise ValueError("img_height not provided and 'H' column not found in CSV")
    
    u_col = _col("u_px")
    v_col = _col("v_px")
    lon_col = _col("lon")
    lat_col = _col("lat")
    alt_col = cols.get("alt_m") or cols.get("elev_m") or cols.get("alt")
    
    # Extract coordinates
    u = df[u_col].to_numpy(dtype=np.float64)
    v = df[v_col].to_numpy(dtype=np.float64)
    lons = df[lon_col].to_numpy(dtype=np.float64)
    lats = df[lat_col].to_numpy(dtype=np.float64)
    
    if alt_col:
        alts = df[alt_col].to_numpy(dtype=np.float64)
    else:
        alts = np.zeros_like(lons)
    
    n_points = len(u)
    if n_points < 4:
        raise ValueError(f"Need at least 4 points for calibration, got {n_points}")
    
    # Reference point: centroid of control points
    ref_lon = float(np.mean(lons))
    ref_lat = float(np.mean(lats))
    ref_alt = float(np.mean(alts))
    
    # Convert to ENU
    enu_points = _llh_to_enu_batch(lons, lats, alts, ref_lon, ref_lat, ref_alt)
    
    # Image points (2D)
    img_points = np.stack([u, v], axis=1).astype(np.float32)
    
    # 3D world points in ENU
    world_points = enu_points.astype(np.float32)
    
    # Principal point at image center
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    # Use calibrateCamera to jointly optimize focal length and pose
    # This gives much better results than fixed focal length + solvePnP
    
    # Prepare data in calibrateCamera format
    object_points_list = [world_points]
    image_points_list = [img_points]
    
    # Initial focal length guess (will be optimized)
    if focal_length_px is None:
        diag = np.sqrt(img_width**2 + img_height**2)
        focal_length_px = diag * 0.8  # Conservative initial estimate
    
    K_init = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Calibrate with fixed principal point, optimize focal length
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS |
        cv2.CALIB_FIX_PRINCIPAL_POINT |
        cv2.CALIB_FIX_ASPECT_RATIO |
        cv2.CALIB_ZERO_TANGENT_DIST |
        cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    )
    
    try:
        rms_error, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points_list,
            image_points_list,
            (img_width, img_height),
            K_init.copy(),
            None,
            flags=flags
        )
        
        rvec = rvecs[0]
        tvec = tvecs[0]
        
        # Check reprojection error
        projected, _ = cv2.projectPoints(world_points.reshape(-1, 1, 3), rvec, tvec, K, None)
        projected = projected.reshape(-1, 2)
        reproj_errors = np.sqrt(np.sum((projected - img_points)**2, axis=1))
        mean_reproj = float(np.mean(reproj_errors))
        
        if mean_reproj > 50:
            # Try with different initial focal lengths
            best_K = None
            best_rvec = None
            best_tvec = None
            best_error = float('inf')
            
            for f_mult in [0.3, 0.5, 0.8, 1.0, 1.5]:
                diag = np.sqrt(img_width**2 + img_height**2)
                f_try = diag * f_mult
                K_try = np.array([
                    [f_try, 0, cx],
                    [0, f_try, cy],
                    [0, 0, 1]
                ], dtype=np.float64)
                
                try:
                    rms, K_opt, _, rvecs_opt, tvecs_opt = cv2.calibrateCamera(
                        object_points_list,
                        image_points_list,
                        (img_width, img_height),
                        K_try.copy(),
                        None,
                        flags=flags
                    )
                    
                    if rms < best_error:
                        best_error = rms
                        best_K = K_opt
                        best_rvec = rvecs_opt[0]
                        best_tvec = tvecs_opt[0]
                except Exception:
                    continue
            
            if best_K is not None:
                K = best_K
                rvec = best_rvec
                tvec = best_tvec
        
    except Exception as e:
        raise RuntimeError(f"Camera calibration failed: {e}")
    
    # Convert rotation vector to matrix
    R_world2cam, _ = cv2.Rodrigues(rvec)
    R_cam2world = R_world2cam.T
    
    # Camera position in world coordinates: C = -R^T * t
    cam_pos_enu = -R_cam2world @ tvec.flatten()
    
    return MonoplottingCalibration(
        K=K.astype(np.float64),
        R=R_cam2world,
        t=tvec.flatten(),
        cam_pos_enu=cam_pos_enu,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        ref_alt=ref_alt,
        img_width=img_width,
        img_height=img_height,
        distortion=None,
    )


def calibrate_monoplotting_with_dem(
    calib_csv: str,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
    dem_folder: Optional[str] = None,
    focal_length_px: Optional[float] = None,
    use_ransac: bool = True,
    ransac_reproj_thresh: float = 8.0,
    optimize_ground_error: bool = True,
) -> Tuple[MonoplottingCalibration, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Calibrate camera with DEM for true monoplotting via ray-DEM intersection.
    
    Uses cv2.calibrateCamera for initial calibration, then optionally refines
    by directly minimizing ground error (recommended for best accuracy).
    
    Args:
        calib_csv: CSV with columns u_px, v_px, lon, lat, and optionally W, H, alt_m
        img_width: Image width in pixels. If None, read from CSV 'W' column.
        img_height: Image height in pixels. If None, read from CSV 'H' column.
        dem_folder: Path to folder with DEM GML/XML files
        focal_length_px: Focal length in pixels. If None, optimized via calibration.
        use_ransac: Use RANSAC for robust estimation
        ransac_reproj_thresh: RANSAC inlier threshold in pixels
        optimize_ground_error: If True, refine calibration by minimizing ground error
        
    Returns:
        (calibration, dem_grid) where dem_grid is (N_axis, E_axis, ELEV) or None
    """
    _require_cv2()
    
    df = pd.read_csv(calib_csv)
    if df.empty:
        raise ValueError("Calibration CSV has no rows")
    
    cols = {c.lower(): c for c in df.columns}
    lon_col = cols.get("lon")
    lat_col = cols.get("lat")
    
    if lon_col is None or lat_col is None:
        raise KeyError("Missing lon/lat columns")
    
    # Auto-detect image dimensions from CSV if not provided
    if img_width is None:
        w_col = cols.get("w")
        if w_col is not None:
            img_width = int(df[w_col].iloc[0])
        else:
            raise ValueError("img_width not provided and 'W' column not found in CSV")
    
    if img_height is None:
        h_col = cols.get("h")
        if h_col is not None:
            img_height = int(df[h_col].iloc[0])
        else:
            raise ValueError("img_height not provided and 'H' column not found in CSV")
    
    lons = df[lon_col].to_numpy(dtype=np.float64)
    lats = df[lat_col].to_numpy(dtype=np.float64)
    
    ref_lon = float(np.mean(lons))
    ref_lat = float(np.mean(lats))
    
    # Load DEM and get elevations for control points
    dem_grid = None
    if dem_folder is not None:
        try:
            from pathlib import Path
            from .dem_gml import build_enu_dem_grid_from_folder_around_calib, _meters_per_degree
            
            dem_grid = build_enu_dem_grid_from_folder_around_calib(
                df, ref_lat, ref_lon, Path(dem_folder),
                margin_m=200.0,
                dem_spacing_m=1.0,
            )
            
            if dem_grid is not None:
                N_axis, E_axis, ELEV = dem_grid
                m_lon, m_lat = _meters_per_degree(ref_lat)
                e_pts = (lons - ref_lon) * m_lon
                n_pts = (lats - ref_lat) * m_lat
                elev_interp = _interp_bilinear_vectorized(e_pts, n_pts, N_axis, E_axis, ELEV)
                
                ref_alt = float(np.mean(elev_interp))
                print(f"[monoplot] DEM elevations at control points: {elev_interp.min():.1f}m to {elev_interp.max():.1f}m")
                
                # Update alt_m column with DEM elevations
                df = df.copy()
                df["alt_m"] = elev_interp
                
                # Save modified CSV temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    df.to_csv(f, index=False)
                    temp_csv = f.name
                
                try:
                    calib = solve_monoplotting_from_csv(
                        temp_csv,
                        img_width=img_width,
                        img_height=img_height,
                        focal_length_px=focal_length_px,
                        use_ransac=use_ransac,
                        ransac_reproj_thresh=ransac_reproj_thresh,
                    )
                    
                    # Optionally refine by minimizing ground error
                    if optimize_ground_error and dem_grid is not None:
                        calib = _refine_calibration_ground_error(
                            calib, df, dem_grid, ref_lon, ref_lat, ref_alt
                        )
                    
                    print(f"[monoplot] Camera position (ENU): E={calib.cam_pos_enu[0]:.1f}, N={calib.cam_pos_enu[1]:.1f}, U={calib.cam_pos_enu[2]:.1f}")
                    print(f"[monoplot] Focal length: {calib.K[0,0]:.1f}px")
                    
                    return calib, dem_grid
                finally:
                    try:
                        os.unlink(temp_csv)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[monoplot] Warning: DEM calibration failed ({e}), falling back to flat ground")
    
    # Fallback: calibrate without DEM elevations
    calib = solve_monoplotting_from_csv(
        calib_csv,
        img_width=img_width,
        img_height=img_height,
        focal_length_px=focal_length_px,
        use_ransac=use_ransac,
        ransac_reproj_thresh=ransac_reproj_thresh,
    )
    
    return calib, dem_grid


# -----------------------------------------------------------------------------
# Ground Error Optimization
# -----------------------------------------------------------------------------

def _refine_calibration_ground_error(
    calib: MonoplottingCalibration,
    df: pd.DataFrame,
    dem_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ref_lon: float,
    ref_lat: float,
    ref_alt: float,
) -> MonoplottingCalibration:
    """
    Refine camera calibration by directly minimizing ground error.
    
    This optimization produces significantly better accuracy than minimizing
    reprojection error because it directly optimizes for the end goal.
    
    Args:
        calib: Initial calibration from cv2.calibrateCamera
        df: DataFrame with control points (u_px, v_px, lon, lat)
        dem_grid: (N_axis, E_axis, ELEV) DEM grid
        ref_lon, ref_lat, ref_alt: Reference point
        
    Returns:
        Refined MonoplottingCalibration
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("[monoplot] scipy not available, skipping ground error optimization")
        return calib
    
    _require_cv2()
    
    N_axis, E_axis, ELEV = dem_grid
    ELEV_ENU = ELEV - ref_alt  # Convert to ENU frame
    
    from .dem_gml import _meters_per_degree
    m_lon, m_lat = _meters_per_degree(ref_lat)
    
    cols = {c.lower(): c for c in df.columns}
    u = df[cols["u_px"]].to_numpy(dtype=np.float64)
    v = df[cols["v_px"]].to_numpy(dtype=np.float64)
    lons = df[cols["lon"]].to_numpy(dtype=np.float64)
    lats = df[cols["lat"]].to_numpy(dtype=np.float64)
    
    target_e = (lons - ref_lon) * m_lon
    target_n = (lats - ref_lat) * m_lat
    uv_pts = np.column_stack([u, v])
    
    img_width = calib.img_width
    img_height = calib.img_height
    
    # Get initial parameters
    R_c2w = calib.R
    R_w2c = R_c2w.T
    rvec_init, _ = cv2.Rodrigues(R_w2c)
    rvec_init = rvec_init.flatten()
    tvec_init = -R_w2c @ calib.cam_pos_enu
    
    def compute_ground_error(params):
        """Compute mean ground error given camera parameters."""
        focal, rvec_x, rvec_y, rvec_z, tx, ty, tz = params
        
        K = np.array([
            [focal, 0, img_width/2],
            [0, focal, img_height/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        rvec = np.array([rvec_x, rvec_y, rvec_z])
        R_w2c_opt, _ = cv2.Rodrigues(rvec)
        R_c2w_opt = R_w2c_opt.T
        
        tvec = np.array([tx, ty, tz])
        cam_pos = -R_c2w_opt @ tvec
        
        # Sanity checks
        if cam_pos[2] < 3:  # Camera should be above ground
            return 1e6
        
        K_inv = np.linalg.inv(K)
        origins, directions = _generate_rays_vectorized(uv_pts, K_inv, R_c2w_opt, cam_pos)
        
        if np.any(directions[:, 2] > -0.01):  # Rays should point down
            return 1e6
        
        intersections, valid = _ray_dem_intersection_fast(
            origins, directions, N_axis, E_axis, ELEV_ENU
        )
        
        if not np.all(valid):
            return 1e6
        
        errors = np.sqrt((intersections[:, 0] - target_e)**2 + 
                         (intersections[:, 1] - target_n)**2)
        
        return np.mean(errors)
    
    # Initial parameters
    params_init = [
        calib.K[0, 0],  # focal
        rvec_init[0], rvec_init[1], rvec_init[2],
        tvec_init[0], tvec_init[1], tvec_init[2],
    ]
    
    err_init = compute_ground_error(params_init)
    
    # Optimize using Nelder-Mead (robust for this problem)
    result = minimize(
        compute_ground_error,
        params_init,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    
    if result.fun < err_init:
        print(f"[monoplot] Ground error optimization: {err_init:.2f}m -> {result.fun:.2f}m")
        
        focal, rx, ry, rz, tx, ty, tz = result.x
        
        K_opt = np.array([
            [focal, 0, img_width/2],
            [0, focal, img_height/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        rvec_opt = np.array([rx, ry, rz])
        R_w2c_opt, _ = cv2.Rodrigues(rvec_opt)
        R_c2w_opt = R_w2c_opt.T
        tvec_opt = np.array([tx, ty, tz])
        cam_pos_opt = -R_c2w_opt @ tvec_opt
        
        return MonoplottingCalibration(
            K=K_opt,
            R=R_c2w_opt,
            t=tvec_opt,
            cam_pos_enu=cam_pos_opt,
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            ref_alt=ref_alt,
            img_width=img_width,
            img_height=img_height,
            distortion=None,
        )
    else:
        print(f"[monoplot] Ground error optimization did not improve: {err_init:.2f}m")
        return calib


# -----------------------------------------------------------------------------
# Ray Generation (Vectorized)
# -----------------------------------------------------------------------------

def _generate_rays_vectorized(
    uv: np.ndarray,
    K_inv: np.ndarray,
    R: np.ndarray,
    cam_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate rays from camera through pixel coordinates (fully vectorized).
    
    Args:
        uv: Nx2 array of pixel coordinates
        K_inv: 3x3 inverse intrinsic matrix
        R: 3x3 rotation matrix (camera to world)
        cam_pos: 3D camera position in world coordinates
        
    Returns:
        (origins, directions): Both Nx3 arrays
    """
    n = uv.shape[0]
    
    # Homogeneous pixel coordinates
    ones = np.ones((n, 1), dtype=np.float64)
    uv_h = np.hstack([uv, ones])  # Nx3
    
    # Ray directions in camera frame (normalized coordinates)
    dirs_cam = (K_inv @ uv_h.T).T  # Nx3
    
    # Transform to world frame
    dirs_world = (R @ dirs_cam.T).T  # Nx3
    
    # Normalize directions
    norms = np.linalg.norm(dirs_world, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    dirs_world = dirs_world / norms
    
    # All rays originate from camera
    origins = np.broadcast_to(cam_pos, (n, 3)).copy()
    
    return origins, dirs_world


# -----------------------------------------------------------------------------
# Ray-DEM Intersection (Optimized)
# -----------------------------------------------------------------------------

def _interp_bilinear_vectorized(
    E: np.ndarray,
    N: np.ndarray,
    N_axis: np.ndarray,
    E_axis: np.ndarray,
    ELEV: np.ndarray,
) -> np.ndarray:
    """
    Vectorized bilinear interpolation on DEM grid.
    
    Args:
        E, N: Arrays of easting/northing coordinates
        N_axis: 1D array of northing values (descending)
        E_axis: 1D array of easting values (ascending)
        ELEV: 2D elevation grid
        
    Returns:
        Interpolated elevations
    """
    # Convert to ascending N for indexing
    N_asc = N_axis[::-1]
    ELEV_flip = ELEV[::-1, :]
    
    n_rows, n_cols = ELEV_flip.shape
    
    # Clamp to grid bounds
    E_clamped = np.clip(E, E_axis[0], E_axis[-1])
    N_clamped = np.clip(N, N_asc[0], N_asc[-1])
    
    # Compute fractional indices
    e_range = E_axis[-1] - E_axis[0]
    n_range = N_asc[-1] - N_asc[0]
    
    if e_range < 1e-10 or n_range < 1e-10:
        return np.full_like(E, np.nanmean(ELEV), dtype=np.float64)
    
    e_idx_f = (E_clamped - E_axis[0]) / e_range * (n_cols - 1)
    n_idx_f = (N_clamped - N_asc[0]) / n_range * (n_rows - 1)
    
    # Integer indices
    e_idx_0 = np.clip(np.floor(e_idx_f).astype(int), 0, n_cols - 2)
    n_idx_0 = np.clip(np.floor(n_idx_f).astype(int), 0, n_rows - 2)
    e_idx_1 = e_idx_0 + 1
    n_idx_1 = n_idx_0 + 1
    
    # Fractional parts
    te = e_idx_f - e_idx_0
    tn = n_idx_f - n_idx_0
    
    # Sample corners
    z00 = ELEV_flip[n_idx_0, e_idx_0]
    z01 = ELEV_flip[n_idx_0, e_idx_1]
    z10 = ELEV_flip[n_idx_1, e_idx_0]
    z11 = ELEV_flip[n_idx_1, e_idx_1]
    
    # Bilinear interpolation
    z0 = z00 * (1 - te) + z01 * te
    z1 = z10 * (1 - te) + z11 * te
    z = z0 * (1 - tn) + z1 * tn
    
    return z


def _ray_dem_intersection_fast(
    origins: np.ndarray,
    directions: np.ndarray,
    N_axis: np.ndarray,
    E_axis: np.ndarray,
    ELEV: np.ndarray,
    max_range: float = 2000.0,
    initial_step: float = 10.0,
    min_step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast adaptive ray-DEM intersection.
    
    Uses adaptive step size based on distance to estimated ground.
    """
    n_rays = origins.shape[0]
    
    intersections = np.full((n_rays, 3), np.nan, dtype=np.float64)
    valid_mask = np.zeros(n_rays, dtype=bool)
    
    # DEM bounds
    e_min, e_max = E_axis[0], E_axis[-1]
    n_min, n_max = N_axis[-1], N_axis[0]
    
    # Process in batches for memory efficiency
    batch_size = 10000
    
    for batch_start in range(0, n_rays, batch_size):
        batch_end = min(batch_start + batch_size, n_rays)
        batch_slice = slice(batch_start, batch_end)
        batch_n = batch_end - batch_start
        
        batch_origins = origins[batch_slice]
        batch_dirs = directions[batch_slice]
        
        # Initialize
        t = np.zeros(batch_n, dtype=np.float64)
        active = np.ones(batch_n, dtype=bool)
        found = np.zeros(batch_n, dtype=bool)
        
        # Estimate initial step based on height above mean DEM
        mean_elev = np.nanmean(ELEV)
        height_above = np.maximum(batch_origins[:, 2] - mean_elev, 1.0)
        step = np.minimum(height_above * 0.5, initial_step)
        
        prev_above = np.ones(batch_n, dtype=bool)
        
        max_iters = int(max_range / min_step) + 100
        
        for _ in range(max_iters):
            if not np.any(active):
                break
            
            pos = batch_origins + t[:, None] * batch_dirs
            
            # Check bounds
            in_bounds = (
                (pos[:, 0] >= e_min) & (pos[:, 0] <= e_max) &
                (pos[:, 1] >= n_min) & (pos[:, 1] <= n_max)
            )
            
            # Sample DEM
            dem_z = np.full(batch_n, np.nan, dtype=np.float64)
            check_mask = active & in_bounds
            if np.any(check_mask):
                dem_z[check_mask] = _interp_bilinear_vectorized(
                    pos[check_mask, 0],
                    pos[check_mask, 1],
                    N_axis, E_axis, ELEV
                )
            
            # Check if ray is above or below DEM
            curr_above = pos[:, 2] > dem_z
            
            # Crossing detected
            crossed = active & prev_above & ~curr_above & np.isfinite(dem_z)
            
            if np.any(crossed):
                # Binary search refinement
                t_lo = np.maximum(t - step, 0.0)
                t_hi = t.copy()
                
                for _ in range(8):
                    t_mid = (t_lo + t_hi) / 2.0
                    pos_mid = batch_origins + t_mid[:, None] * batch_dirs
                    
                    dem_z_mid = np.full(batch_n, np.nan, dtype=np.float64)
                    ref_mask = crossed & (
                        (pos_mid[:, 0] >= e_min) & (pos_mid[:, 0] <= e_max) &
                        (pos_mid[:, 1] >= n_min) & (pos_mid[:, 1] <= n_max)
                    )
                    if np.any(ref_mask):
                        dem_z_mid[ref_mask] = _interp_bilinear_vectorized(
                            pos_mid[ref_mask, 0],
                            pos_mid[ref_mask, 1],
                            N_axis, E_axis, ELEV
                        )
                    
                    below = pos_mid[:, 2] <= dem_z_mid
                    t_hi = np.where(crossed & below, t_mid, t_hi)
                    t_lo = np.where(crossed & ~below, t_mid, t_lo)
                
                t_final = (t_lo + t_hi) / 2.0
                pos_final = batch_origins + t_final[:, None] * batch_dirs
                
                intersections[batch_start:batch_end][crossed] = pos_final[crossed]
                found[crossed] = True
                active[crossed] = False
            
            # Update state
            prev_above = curr_above.copy()
            
            # Adaptive step: smaller when close to DEM
            dist_to_dem = np.abs(pos[:, 2] - dem_z)
            step = np.clip(dist_to_dem * 0.3, min_step, initial_step)
            step = np.where(np.isfinite(step), step, initial_step)
            
            # Advance
            t[active] += step[active]
            
            # Deactivate
            active &= (t < max_range) & (in_bounds | (t < initial_step * 3))
        
        valid_mask[batch_slice] = found
    
    return intersections, valid_mask


# -----------------------------------------------------------------------------
# Main Monoplotting Pipeline
# -----------------------------------------------------------------------------

def monoplot_pixels(
    uv: np.ndarray,
    calib: MonoplottingCalibration,
    N_axis: np.ndarray,
    E_axis: np.ndarray,
    ELEV: np.ndarray,
    max_range: float = 2000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project pixel coordinates to 3D ground points via ray-DEM intersection.
    
    Args:
        uv: Nx2 pixel coordinates
        calib: Camera calibration
        N_axis, E_axis, ELEV: DEM grid
        max_range: Maximum projection distance
        
    Returns:
        (enu_points, valid_mask): Nx3 ENU coordinates and validity mask
    """
    _require_cv2()
    
    # Compute inverse intrinsic matrix
    K_inv = np.linalg.inv(calib.K)
    
    # Generate rays
    origins, directions = _generate_rays_vectorized(
        uv, K_inv, calib.R, calib.cam_pos_enu
    )
    
    # Ray-DEM intersection
    intersections, valid = _ray_dem_intersection_fast(
        origins, directions,
        N_axis, E_axis, ELEV,
        max_range=max_range,
    )
    
    return intersections, valid


def monoplot_to_geo(
    uv: np.ndarray,
    calib: MonoplottingCalibration,
    N_axis: np.ndarray,
    E_axis: np.ndarray,
    ELEV: np.ndarray,
    max_range: float = 2000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project pixels to geographic coordinates using ray-DEM intersection.
    
    Casts rays from camera through each pixel and intersects with DEM
    to find the 3D ground position.
    
    Returns:
        (lon_lat_elev, enu_points, valid_mask)
    """
    n = uv.shape[0]
    
    # Full 3D ray-DEM intersection
    enu_points, valid = monoplot_pixels(
        uv, calib, N_axis, E_axis, ELEV, max_range
    )
    
    lon_lat_elev = np.full((n, 3), np.nan, dtype=np.float64)
    
    if np.any(valid):
        ctx = build_context(calib.ref_lat, calib.ref_lon, 0.0, calib.ref_alt)
        valid_idx = np.where(valid)[0]
        valid_enu = enu_points[valid_idx]
        llh = enu_to_llh_batch(ctx, valid_enu)
        lon_lat_elev[valid_idx] = llh
    
    return lon_lat_elev, enu_points, valid


# -----------------------------------------------------------------------------
# Save/Load Calibration
# -----------------------------------------------------------------------------

def save_monoplotting_calibration(
    npz_path: str,
    calib: MonoplottingCalibration,
    N_axis: Optional[np.ndarray] = None,
    E_axis: Optional[np.ndarray] = None,
    ELEV: Optional[np.ndarray] = None,
) -> None:
    """Save monoplotting calibration to NPZ file.
    
    IMPORTANT: DEM elevations are converted to ENU frame (relative to ref_alt)
    so that ray-DEM intersection works correctly with camera coordinates.
    """
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)) or ".", exist_ok=True)
    
    data = {
        "PROJECTION": "monoplotting",
        "K": calib.K,
        "R": calib.R,
        "T": calib.t,
        "CAM_POS_ENU": calib.cam_pos_enu,
        "REF_LAT": calib.ref_lat,
        "REF_LON": calib.ref_lon,
        "REF_ALT": calib.ref_alt,
        "IMG_WIDTH": calib.img_width,
        "IMG_HEIGHT": calib.img_height,
    }
    
    if calib.distortion is not None:
        data["DISTORTION"] = calib.distortion
    
    if N_axis is not None and E_axis is not None and ELEV is not None:
        data["DEM_N_AXIS"] = N_axis.astype(np.float64)
        data["DEM_E_AXIS"] = E_axis.astype(np.float64)
        # Convert DEM from absolute elevation to ENU frame (relative to ref_alt)
        # This ensures ray-DEM intersection works correctly with camera coordinates
        ELEV_ENU = ELEV.astype(np.float64) - calib.ref_alt
        data["DEM_ELEV_M"] = ELEV_ENU
    
    np.savez(npz_path, **data)


def load_monoplotting_calibration(
    npz_path: str,
) -> Tuple[
    MonoplottingCalibration,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Load monoplotting calibration from NPZ file."""
    nz = np.load(npz_path)
    
    proj = str(nz.get("PROJECTION", ""))
    if proj not in ("monoplotting", "perspective"):
        raise ValueError(f"Expected monoplotting projection, got: {proj}")
    
    # Load calibration
    if "K" in nz:
        # New monoplotting format
        calib = MonoplottingCalibration(
            K=nz["K"].astype(np.float64),
            R=nz["R"].astype(np.float64),
            t=nz["T"].astype(np.float64),
            cam_pos_enu=nz["CAM_POS_ENU"].astype(np.float64),
            ref_lat=float(nz["REF_LAT"]),
            ref_lon=float(nz["REF_LON"]),
            ref_alt=float(nz.get("REF_ALT", 0.0)),
            img_width=int(nz["IMG_WIDTH"]),
            img_height=int(nz["IMG_HEIGHT"]),
            distortion=nz["DISTORTION"] if "DISTORTION" in nz else None,
        )
        
    else:
        raise ValueError("Missing camera calibration data in NPZ file")
    
    # Load DEM if present
    N_axis = nz.get("DEM_N_AXIS")
    E_axis = nz.get("DEM_E_AXIS")
    ELEV = nz.get("DEM_ELEV_M")
    
    if N_axis is not None:
        N_axis = N_axis.astype(np.float64)
    if E_axis is not None:
        E_axis = E_axis.astype(np.float64)
    if ELEV is not None:
        ELEV = ELEV.astype(np.float64)
    
    return calib, N_axis, E_axis, ELEV


# -----------------------------------------------------------------------------
# Legacy Homography Support (for backwards compatibility)
# -----------------------------------------------------------------------------

def solve_homography_from_csv(
    calib_csv: str,
    ransac_thresh_m: float = 10.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Legacy: Estimate pixel->ENU homography from CSV.
    Kept for backwards compatibility.
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
        en_points.append([float(v_enu[0]), float(v_enu[1])])
    
    dst = np.asarray(en_points, dtype=np.float32)
    
    H, status = cv2.findHomography(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(max(0.1, ransac_thresh_m))
    )
    if H is None:
        raise RuntimeError("cv2.findHomography failed")
    
    return H.astype(np.float64), ref_lat, ref_lon


def save_homography(
    npz_path: str,
    H_px2enu: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ground_alt_m: float = 0.0,
    calib_csv: Optional[str] = None,
    google_api_key: Optional[str] = None,
    dem_xml_folder: Optional[str] = None,
    dem_rows: Optional[int] = None,
    dem_cols: Optional[int] = None,
    dem_margin_m: float = 120.0,
    dem_spacing_m: Optional[float] = None,
) -> None:
    """Legacy: Save homography calibration."""
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)) or ".", exist_ok=True)
    
    N_axis = None
    E_axis = None
    ELEV = None
    
    # Load DEM if available
    if calib_csv is not None and dem_xml_folder is not None:
        try:
            from pathlib import Path
            from .dem_gml import build_enu_dem_grid_from_folder_around_calib
            
            df = pd.read_csv(calib_csv)
            if not df.empty and len(df) >= 4:
                built = build_enu_dem_grid_from_folder_around_calib(
                    df,
                    float(ref_lat),
                    float(ref_lon),
                    Path(str(dem_xml_folder)),
                    margin_m=float(dem_margin_m),
                    dem_spacing_m=float(dem_spacing_m) if dem_spacing_m is not None else None,
                )
                if built is not None:
                    N_axis, E_axis, ELEV = built
        except Exception:
            pass
    
    data = {
        "H_PX2ENU": H_px2enu,
        "PROJECTION": "perspective",
        "REF_LAT": float(ref_lat),
        "REF_LON": float(ref_lon),
        "GROUND_ALT_M": float(ground_alt_m),
    }
    
    if N_axis is not None and E_axis is not None and ELEV is not None:
        data["DEM_N_AXIS"] = N_axis.astype(np.float64)
        data["DEM_E_AXIS"] = E_axis.astype(np.float64)
        data["DEM_ELEV_M"] = ELEV.astype(np.float64)
    
    np.savez(npz_path, **data)


def load_homography(
    npz_path: str,
) -> Tuple[
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
    """Legacy: Load homography calibration."""
    nz = np.load(npz_path)
    
    if "H_PX2ENU" in nz:
        H = nz["H_PX2ENU"]
        ref_lat = float(nz.get("REF_LAT", np.nan))
        ref_lon = float(nz.get("REF_LON", np.nan))
        g_alt = float(nz.get("GROUND_ALT_M", 0.0))
        
        # Legacy plane params (not used in new implementation)
        P0 = nz.get("PLANE_P0_ENU")
        Uhat = nz.get("PLANE_U_HAT")
        Vhat = nz.get("PLANE_V_HAT")
        
        N_axis = nz.get("DEM_N_AXIS")
        E_axis = nz.get("DEM_E_AXIS")
        ELEV = nz.get("DEM_ELEV_M")
        
        if P0 is not None:
            P0 = P0.astype(np.float64)
        if Uhat is not None:
            Uhat = Uhat.astype(np.float64)
        if Vhat is not None:
            Vhat = Vhat.astype(np.float64)
        if N_axis is not None:
            N_axis = N_axis.astype(np.float64)
        if E_axis is not None:
            E_axis = E_axis.astype(np.float64)
        if ELEV is not None:
            ELEV = ELEV.astype(np.float64)
        
        return H, ref_lat, ref_lon, g_alt, P0, Uhat, Vhat, N_axis, E_axis, ELEV
    
    if "H_PX2GEO" in nz:
        return nz["H_PX2GEO"], None, None, None, None, None, None, None, None, None
    
    raise KeyError("No homography matrix found in npz")


def pixel_to_geo(points_uv: np.ndarray, H_px2enu_or_geo: np.ndarray) -> np.ndarray:
    """Legacy: Map pixels using homography."""
    _require_cv2()
    if points_uv.ndim != 2 or points_uv.shape[1] != 2:
        raise ValueError("points_uv must be Nx2 array")
    pts = points_uv.astype(np.float32).reshape((-1, 1, 2))
    mapped = cv2.perspectiveTransform(pts, H_px2enu_or_geo.astype(np.float32))
    return mapped.reshape((-1, 2)).astype(np.float64)


# -----------------------------------------------------------------------------
# Unified Geolocate Function
# -----------------------------------------------------------------------------

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
    dem_xml_folder: Optional[str] = None,
    dem_margin_m: float = 120.0,
    dem_spacing_m: Optional[float] = None,
    smoothing_window: int = 0,
    max_velocity_m: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Geolocate detection points using either monoplotting or homography.
    
    Automatically detects calibration type and uses appropriate method:
    - Monoplotting: Ray-DEM intersection for accurate 3D projection
    - Homography: Legacy 2D projective transform
    
    Args:
        ...
        smoothing_window: If > 0, applies post-processing smoothing to the geolocated tracks.
        max_velocity_m: If set, limits the maximum velocity (m/frame) during smoothing.
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
    xoff_col = cols.get("x_offset_px")
    yoff_col = cols.get("y_offset_px")
    
    # Load calibration
    nz = np.load(homography_npz)
    proj_type = str(nz.get("PROJECTION", "perspective"))
    
    if debug:
        print(f"[persp] calibration type: {proj_type}")
    
    # Extract pixel coordinates
    uv = df[[u_col, v_col]].to_numpy(dtype=np.float64)
    
    # Apply crop offsets if present
    if xoff_col is not None and yoff_col is not None:
        try:
            xoff = df[xoff_col].to_numpy(dtype=np.float64)
            yoff = df[yoff_col].to_numpy(dtype=np.float64)
            if xoff.size == uv.shape[0] and yoff.size == uv.shape[0]:
                uv[:, 0] += xoff
                uv[:, 1] += yoff
                if debug:
                    print(f"[persp] applied crop offsets: mean x={np.nanmean(xoff):.1f}, y={np.nanmean(yoff):.1f}")
        except Exception as e:
            if debug:
                print(f"[persp] warning: failed applying offsets: {e}")
    
    n_points = uv.shape[0]
    
    # Initialize output columns
    lons = np.full(n_points, np.nan, dtype=np.float64)
    lats = np.full(n_points, np.nan, dtype=np.float64)
    east = np.full(n_points, np.nan, dtype=np.float64)
    north = np.full(n_points, np.nan, dtype=np.float64)
    elevs = np.full(n_points, np.nan, dtype=np.float64)
    
    # Try monoplotting first, fall back to homography
    if proj_type == "monoplotting" and "K" in nz:
        # Monoplotting mode
        calib, N_axis, E_axis, ELEV = load_monoplotting_calibration(homography_npz)
        
        if debug:
            print(f"[persp] monoplotting: cam_pos={calib.cam_pos_enu}")
            print(f"[persp] ref: lat={calib.ref_lat:.8f}, lon={calib.ref_lon:.8f}")
        
        # Load DEM if not in calibration file
        if N_axis is None or E_axis is None or ELEV is None:
            if dem_xml_folder is not None:
                try:
                    from pathlib import Path
                    from .dem_gml import build_dem_grid_from_folder_for_bbox, _meters_per_degree
                    
                    # Build DEM for detection extent
                    margin = 500.0  # meters
                    m_lon, m_lat = _meters_per_degree(calib.ref_lat)
                    
                    dlon = margin / m_lon
                    dlat = margin / m_lat
                    
                    built = build_dem_grid_from_folder_for_bbox(
                        Path(dem_xml_folder),
                        calib.ref_lon - dlon,
                        calib.ref_lat - dlat,
                        calib.ref_lon + dlon,
                        calib.ref_lat + dlat,
                    )
                    
                    if built is not None:
                        lat_axis, lon_axis, elev_ll = built
                        # Convert to ENU
                        e_list = [(lo - calib.ref_lon) * m_lon for lo in lon_axis]
                        n_list = [(la - calib.ref_lat) * m_lat for la in lat_axis]
                        E_axis = np.array(e_list, dtype=np.float64)
                        N_axis = np.array(n_list, dtype=np.float64)
                        ELEV = elev_ll.astype(np.float64)
                        
                        if debug:
                            print(f"[persp] loaded DEM: {ELEV.shape[0]}x{ELEV.shape[1]}")
                except Exception as e:
                    if debug:
                        print(f"[persp] DEM load failed: {e}")
        
        if N_axis is None or E_axis is None or ELEV is None:
            raise ValueError("Monoplotting requires DEM data")
        
        # Process in batches with progress
        batch_size = 50000
        n_batches = (n_points + batch_size - 1) // batch_size
        
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=n_points, desc=progress_desc or "monoplot", unit="pt")
            except ImportError:
                pass
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_points)
            batch_uv = uv[start:end]
            
            # Monoplot batch
            geo_coords, enu_coords, valid = monoplot_to_geo(
                batch_uv, calib, N_axis, E_axis, ELEV
            )
            
            # Store results
            lons[start:end] = geo_coords[:, 0]
            lats[start:end] = geo_coords[:, 1]
            elevs[start:end] = geo_coords[:, 2]
            east[start:end] = enu_coords[:, 0]
            north[start:end] = enu_coords[:, 1]
            
            if pbar is not None:
                pbar.update(end - start)
            elif show_progress and (batch_idx % 5 == 0 or batch_idx == n_batches - 1):
                print(f"[monoplot] {end}/{n_points}", flush=True)
        
        if pbar is not None:
            pbar.close()
        
        ref_lat = calib.ref_lat
        ref_lon = calib.ref_lon
        
    else:
        # Legacy homography mode
        H, ref_lat, ref_lon, g_alt, P0, Uhat, Vhat, N_axis, E_axis, ELEV = load_homography(homography_npz)
        
        if debug:
            print(f"[persp] homography mode: ref=({ref_lat:.8f}, {ref_lon:.8f})")
            if N_axis is not None:
                print(f"[persp] DEM: {N_axis.shape[0]}x{E_axis.shape[0]}")
        
        # Load DEM if not present
        if (N_axis is None or E_axis is None or ELEV is None) and dem_xml_folder is not None:
            try:
                from pathlib import Path
                from .dem_gml import build_dem_grid_from_folder_for_bbox, _meters_per_degree
                
                # First project to get EN bounds
                coords_en = pixel_to_geo(uv, H)
                e_min = float(np.nanmin(coords_en[:, 0])) - dem_margin_m
                e_max = float(np.nanmax(coords_en[:, 0])) + dem_margin_m
                n_min = float(np.nanmin(coords_en[:, 1])) - dem_margin_m
                n_max = float(np.nanmax(coords_en[:, 1])) + dem_margin_m
                
                m_lon, m_lat = _meters_per_degree(ref_lat)
                
                built = build_dem_grid_from_folder_for_bbox(
                    Path(dem_xml_folder),
                    ref_lon + e_min / m_lon,
                    ref_lat + n_min / m_lat,
                    ref_lon + e_max / m_lon,
                    ref_lat + n_max / m_lat,
                )
                
                if built is not None:
                    lat_axis, lon_axis, elev_ll = built
                    E_axis = np.array([(lo - ref_lon) * m_lon for lo in lon_axis], dtype=np.float64)
                    N_axis = np.array([(la - ref_lat) * m_lat for la in lat_axis], dtype=np.float64)
                    ELEV = elev_ll.astype(np.float64)
                    
                    if debug:
                        print(f"[persp] loaded DEM: {ELEV.shape[0]}x{ELEV.shape[1]}")
            except Exception as e:
                if debug:
                    print(f"[persp] DEM load failed: {e}")
        
        # Project using homography
        coords_en = pixel_to_geo(uv, H)
        east[:] = coords_en[:, 0]
        north[:] = coords_en[:, 1]
        
        # Interpolate elevations from DEM
        if N_axis is not None and E_axis is not None and ELEV is not None:
            elevs[:] = _interp_bilinear_vectorized(east, north, N_axis, E_axis, ELEV)
        else:
            elevs[:] = float(g_alt) if g_alt is not None else 0.0
        
        # Convert to lon/lat (vectorized)
        ctx = build_context(float(ref_lat), float(ref_lon), 0.0, float(g_alt or 0.0))
        
        # Build ENU array for vectorized conversion
        enu_array = np.stack([east, north, elevs], axis=1)
        
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=n_points, desc=progress_desc or "geolocate", unit="pt")
            except ImportError:
                pass
        
        # Process in batches for vectorized conversion
        batch_size = 50000
        for batch_start in range(0, n_points, batch_size):
            batch_end = min(batch_start + batch_size, n_points)
            batch_enu = enu_array[batch_start:batch_end]
            
            # Vectorized conversion
            llh = enu_to_llh_batch(ctx, batch_enu)
            lons[batch_start:batch_end] = llh[:, 0]
            lats[batch_start:batch_end] = llh[:, 1]
            
            if pbar is not None:
                pbar.update(batch_end - batch_start)
            elif show_progress:
                print(f"[geolocate] {batch_end}/{n_points}", flush=True)
        
        if pbar is not None:
            pbar.close()
    
    # Add results to dataframe
    df["lon"] = lons
    df["lat"] = lats
    df["east_m"] = east
    df["north_m"] = north
    df["elev_m"] = elevs
    
    # Gating by calibration extent
    if calib_csv is not None and ref_lat is not None and ref_lon is not None:
        try:
            cdf = pd.read_csv(calib_csv)
            ccols = {c.lower(): c for c in cdf.columns}
            lon_col_c = ccols.get("lon")
            lat_col_c = ccols.get("lat")
            
            if lon_col_c and lat_col_c:
                from .dem_gml import _meters_per_degree
                m_lon, m_lat = _meters_per_degree(ref_lat)
                
                lons_c = cdf[lon_col_c].to_numpy(dtype=np.float64)
                lats_c = cdf[lat_col_c].to_numpy(dtype=np.float64)
                
                e_c = (lons_c - ref_lon) * m_lon
                n_c = (lats_c - ref_lat) * m_lat
                
                e_min = float(np.min(e_c)) - gate_margin_m
                e_max = float(np.max(e_c)) + gate_margin_m
                n_min = float(np.min(n_c)) - gate_margin_m
                n_max = float(np.max(n_c)) + gate_margin_m
                
                mask_inside = (
                    (df["east_m"] >= e_min) & (df["east_m"] <= e_max) &
                    (df["north_m"] >= n_min) & (df["north_m"] <= n_max)
                )
                
                if debug:
                    total = len(df)
                    inside = int(mask_inside.sum())
                    print(f"[persp] gating: E[{e_min:.1f},{e_max:.1f}] N[{n_min:.1f},{n_max:.1f}] -> {inside}/{total}")
                
                if drop_outside:
                    df = df[mask_inside].reset_index(drop=True)
                else:
                    df.loc[~mask_inside, ["lon", "lat"]] = np.nan
        except Exception as e:
            if debug:
                print(f"[persp] gating failed: {e}")
    
    if debug:
        valid_mask = np.isfinite(df["lon"]) & np.isfinite(df["lat"])
        if valid_mask.any():
            print(f"[persp] result: lon=[{df.loc[valid_mask, 'lon'].min():.8f}, {df.loc[valid_mask, 'lon'].max():.8f}]")
            print(f"[persp] result: lat=[{df.loc[valid_mask, 'lat'].min():.8f}, {df.loc[valid_mask, 'lat'].max():.8f}]")
    
    # Save outputs
    xy_csv = os.path.join(output_dir, "all_people_xy_calibrated.csv")
    geo_csv = os.path.join(output_dir, "all_people_geo_calibrated.csv")
    df.to_csv(xy_csv, index=False)
    df.to_csv(geo_csv, index=False)
    
    # Optional: Apply smoothing
    if smoothing_window > 0:
        try:
            from .people_tracking import smooth_geotracks_csv
            if debug:
                print(f"[persp] applying smoothing: window={smoothing_window}, max_vel={max_velocity_m}")
            
            # Overwrite the geo_csv with the smoothed version
            smooth_geotracks_csv(
                geo_csv_path=geo_csv,
                calibration_npz=homography_npz,
                output_path=geo_csv,
                window_size=smoothing_window,
                max_velocity_m=max_velocity_m,
            )
        except Exception as e:
            if debug:
                print(f"[persp] smoothing failed: {e}")
    
    return xy_csv, geo_csv
