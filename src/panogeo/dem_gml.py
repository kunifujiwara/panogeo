from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from .geodesy import build_context, enu_to_llh  # type: ignore

# Namespaces used in GSI DEM GML
NS = {
    "gml": "http://www.opengis.net/gml/3.2",
    "fgd": "http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema",
}


def _meters_per_degree(ref_lat_deg: float) -> Tuple[float, float]:
    """
    Approximate meters per degree in E (lon) and N (lat) directions near ref latitude.
    Returns (m_per_deg_lon, m_per_deg_lat).
    """
    lat_rad = math.radians(float(ref_lat_deg))
    m_per_deg_lat = 111_132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad) - 0.0023 * math.cos(6 * lat_rad)
    m_per_deg_lon = 111_412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad) + 0.118 * math.cos(5 * lat_rad)
    return float(m_per_deg_lon), float(m_per_deg_lat)


def parse_dem_tile(xml_path: Path) -> Tuple[np.ndarray, Dict[str, float], int, int]:
    """
    Parse a single GSI DEM GML tile and return:
      - dem: 2D numpy array (rows, cols)
      - bbox: dict {min_lon, min_lat, max_lon, max_lat}
      - grid info: rows, cols (for spacing)
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    env = root.find(".//gml:Envelope", NS)
    if env is None:
        raise ValueError("No gml:Envelope found")
    lower = list(map(float, env.find("gml:lowerCorner", NS).text.split()))
    upper = list(map(float, env.find("gml:upperCorner", NS).text.split()))
    # GSI uses 'lat lon' order in lower/upperCorner
    min_lat, min_lon = lower
    max_lat, max_lon = upper
    bbox = {"min_lon": float(min_lon), "min_lat": float(min_lat), "max_lon": float(max_lon), "max_lat": float(max_lat)}

    grid_env = root.find(".//gml:GridEnvelope", NS)
    if grid_env is None:
        raise ValueError("No gml:GridEnvelope found")
    low = grid_env.find("gml:low", NS).text.split()
    high = grid_env.find("gml:high", NS).text.split()
    cols = int(high[0]) - int(low[0]) + 1  # x
    rows = int(high[1]) - int(low[1]) + 1  # y

    tup = root.find(".//gml:tupleList", NS)
    if tup is None or tup.text is None:
        raise ValueError("No gml:tupleList found or empty")
    lines = [ln.strip() for ln in tup.text.splitlines() if ln.strip()]
    vals = []
    for ln in lines:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        try:
            vals.append(float(parts[1]))
        except Exception:
            vals.append(float("nan"))
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size != rows * cols:
        raise ValueError(f"Value count mismatch in {xml_path.name}: {arr.size} != {rows}x{cols}")
    dem = arr.reshape((rows, cols))
    dem[dem <= -9990] = np.nan  # nodata
    return dem, bbox, rows, cols


def _tile_lat_lon_centers(bbox: Dict[str, float], rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D vectors of lat centers (rows) and lon centers (cols) for a tile.
    """
    dlat = (bbox["max_lat"] - bbox["min_lat"]) / float(rows)
    dlon = (bbox["max_lon"] - bbox["min_lon"]) / float(cols)
    row_indices = np.arange(rows, dtype=float)
    col_indices = np.arange(cols, dtype=float)
    lat_centers = bbox["max_lat"] - (row_indices + 0.5) * dlat
    lon_centers = bbox["min_lon"] + (col_indices + 0.5) * dlon
    return lat_centers, lon_centers


def _intersect_rect(a: Dict[str, float], b: Dict[str, float]) -> Optional[Dict[str, float]]:
    min_lon = max(a["min_lon"], b["min_lon"])
    max_lon = min(a["max_lon"], b["max_lon"])
    min_lat = max(a["min_lat"], b["min_lat"])
    max_lat = min(a["max_lat"], b["max_lat"])
    if (min_lon >= max_lon) or (min_lat >= max_lat):
        return None
    return {"min_lon": min_lon, "min_lat": min_lat, "max_lon": max_lon, "max_lat": max_lat}


def _crop_tile_to_bbox(dem: np.ndarray, bbox: Dict[str, float], rows: int, cols: int, target_bbox: Dict[str, float]) -> Optional[Tuple[np.ndarray, Dict[str, float], np.ndarray, np.ndarray]]:
    """
    Crop a DEM tile to target bbox. Returns (cropped_dem, cropped_bbox, lat_centers, lon_centers) or None.
    """
    inter = _intersect_rect(bbox, target_bbox)
    if inter is None:
        return None
    lat_centers, lon_centers = _tile_lat_lon_centers(bbox, rows, cols)
    inside = (
        (lat_centers[:, None] >= inter["min_lat"])
        & (lat_centers[:, None] <= inter["max_lat"])
        & (lon_centers[None, :] >= inter["min_lon"])
        & (lon_centers[None, :] <= inter["max_lon"])
    )
    if not inside.any():
        return None
    rows_inside = np.where(inside.any(axis=1))[0]
    cols_inside = np.where(inside.any(axis=0))[0]
    r0, r1 = int(rows_inside[0]), int(rows_inside[-1])
    c0, c1 = int(cols_inside[0]), int(cols_inside[-1])
    cropped_dem = dem[r0 : r1 + 1, c0 : c1 + 1]
    # compute bbox aligned to cell edges
    dlat = (bbox["max_lat"] - bbox["min_lat"]) / float(rows)
    dlon = (bbox["max_lon"] - bbox["min_lon"]) / float(cols)
    cropped_bbox = {
        "max_lat": bbox["max_lat"] - r0 * dlat,
        "min_lat": bbox["max_lat"] - (r1 + 1) * dlat,
        "min_lon": bbox["min_lon"] + c0 * dlon,
        "max_lon": bbox["min_lon"] + (c1 + 1) * dlon,
    }
    lat_c = lat_centers[r0 : r1 + 1]
    lon_c = lon_centers[c0 : c1 + 1]
    return cropped_dem, cropped_bbox, lat_c, lon_c


def _round_arr(x: np.ndarray, ndigits: int = 12) -> np.ndarray:
    return np.asarray([float(f"{v:.{ndigits}f}") for v in x], dtype=float)


def build_dem_grid_from_folder_for_bbox(
    folder: Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build a mosaicked DEM grid from multiple GML/XML tiles within a lon/lat bbox.
    Returns (lat_axis_desc, lon_axis_asc, elev), where elev shape is (n_rows, n_cols).
    """
    xml_files = sorted(folder.glob("*.xml"))
    if not xml_files:
        return None
    target_bbox = {"min_lon": float(min_lon), "min_lat": float(min_lat), "max_lon": float(max_lon), "max_lat": float(max_lat)}
    # Collect cropped pieces
    pieces: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    lat_values: List[float] = []
    lon_values: List[float] = []
    for xml_path in xml_files:
        try:
            dem, bbox, rows, cols = parse_dem_tile(xml_path)
            cropped = _crop_tile_to_bbox(dem, bbox, rows, cols, target_bbox)
            if cropped is None:
                continue
            dem_c, _bbox_c, lat_c, lon_c = cropped
            # Round to stabilize keys
            lat_c_ = _round_arr(lat_c)
            lon_c_ = _round_arr(lon_c)
            pieces.append((dem_c.astype(float), lat_c_, lon_c_))
            lat_values.extend(lat_c_.tolist())
            lon_values.extend(lon_c_.tolist())
        except Exception:
            continue
    if not pieces:
        return None
    # Unique global axes: lat descending, lon ascending
    lat_axis = np.unique(np.asarray(lat_values, dtype=float))[::-1]
    lon_axis = np.unique(np.asarray(lon_values, dtype=float))
    n_rows = int(lat_axis.shape[0])
    n_cols = int(lon_axis.shape[0])
    elev = np.full((n_rows, n_cols), np.nan, dtype=float)
    # index maps
    lat_to_idx: Dict[float, int] = {float(v): i for i, v in enumerate(lat_axis)}
    lon_to_idx: Dict[float, int] = {float(v): i for i, v in enumerate(lon_axis)}
    # Paste pieces
    for dem_c, lat_c, lon_c in pieces:
        for i_local, lat in enumerate(lat_c):
            gi = lat_to_idx.get(float(lat))
            if gi is None:
                continue
            for j_local, lon in enumerate(lon_c):
                gj = lon_to_idx.get(float(lon))
                if gj is None:
                    continue
                val = float(dem_c[i_local, j_local])
                if math.isfinite(val):
                    elev[gi, gj] = val
    # Drop any rows/cols that are entirely NaN (edges)
    valid_row = ~np.all(~np.isfinite(elev), axis=1)
    valid_col = ~np.all(~np.isfinite(elev), axis=0)
    lat_axis = lat_axis[valid_row]
    lon_axis = lon_axis[valid_col]
    elev = elev[np.ix_(valid_row, valid_col)]
    if elev.size == 0:
        return None
    return lat_axis.astype(float), lon_axis.astype(float), elev.astype(float)


def build_enu_dem_grid_from_folder_around_calib(
    calib_df,
    ref_lat: float,
    ref_lon: float,
    folder: Path,
    margin_m: float = 120.0,
    dem_spacing_m: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build a DEM grid in ENU meters given calibration lon/lat points and a DEM folder.
    Returns (N_axis, E_axis, ELEV) compatible with perspective.geolocate pipeline.
    """
    cols = {c.lower(): c for c in calib_df.columns}
    lon_col = cols.get("lon")
    lat_col = cols.get("lat")
    if lon_col is None or lat_col is None:
        return None
    lons = calib_df[lon_col].astype(float).to_numpy()
    lats = calib_df[lat_col].astype(float).to_numpy()
    # Compute lon/lat bbox expanded by margin (in meters -> degrees)
    m_per_deg_lon, m_per_deg_lat = _meters_per_degree(float(ref_lat))
    dlon = float(margin_m) / float(max(1e-6, m_per_deg_lon))
    dlat = float(margin_m) / float(max(1e-6, m_per_deg_lat))
    min_lon = float(np.min(lons)) - dlon
    max_lon = float(np.max(lons)) + dlon
    min_lat = float(np.min(lats)) - dlat
    max_lat = float(np.max(lats)) + dlat
    built = build_dem_grid_from_folder_for_bbox(folder, min_lon, min_lat, max_lon, max_lat)
    if built is None:
        return None
    lat_axis_desc, lon_axis_asc, elev = built
    # Convert lon/lat axes to ENU axes using ref lat/lon
    ctx = build_context(float(ref_lat), float(ref_lon), 0.0, 0.0)
    # For E axis, fix lat = ref_lat and vary lon along lon_axis
    e_list: List[float] = []
    for lo in lon_axis_asc:
        e, _n = _lonlat_to_en_east_north(ctx, float(lo), float(ref_lat))
        e_list.append(float(e))
    E_axis = np.asarray(e_list, dtype=float)
    # For N axis, fix lon = ref_lon and vary lat along lat_axis
    n_list: List[float] = []
    for la in lat_axis_desc:
        _e, n = _lonlat_to_en_east_north(ctx, float(ref_lon), float(la))
        n_list.append(float(n))
    N_axis = np.asarray(n_list, dtype=float)
    # Optional downsampling to approximate target spacing in meters
    if (dem_spacing_m is not None) and (float(dem_spacing_m) > 0):
        if E_axis.shape[0] >= 2:
            de = float(abs(E_axis[1] - E_axis[0]))
            step_e = max(1, int(round(float(dem_spacing_m) / max(1e-6, de))))
        else:
            step_e = 1
        if N_axis.shape[0] >= 2:
            dn = float(abs(N_axis[0] - N_axis[1]))  # N axis is descending in distance
            step_n = max(1, int(round(float(dem_spacing_m) / max(1e-6, dn))))
        else:
            step_n = 1
        N_axis = N_axis[::step_n]
        E_axis = E_axis[::step_e]
        elev = elev[::step_n, ::step_e]
    return N_axis.astype(float), E_axis.astype(float), elev.astype(float)


def _lonlat_to_en_east_north(ctx, lon: float, lat: float) -> Tuple[float, float]:
    # Convert by going from ENU (E,N,0) -> lon/lat inverse using a small search.
    # We do not have a direct llh->enu here; instead approximate by finite differences around ref.
    # Use numerical approach: find E,N so that enu_to_llh(E,N,0) ≈ (lon,lat).
    # For local small extents, linearization works: E ≈ (lon - ref_lon) * m_per_deg_lon, N ≈ (lat - ref_lat) * m_per_deg_lat
    ref_lon = ctx["ref_lon"]
    ref_lat = ctx["ref_lat"]
    m_per_deg_lon, m_per_deg_lat = _meters_per_degree(float(ref_lat))
    e = float((lon - float(ref_lon)) * m_per_deg_lon)
    n = float((lat - float(ref_lat)) * m_per_deg_lat)
    return e, n


