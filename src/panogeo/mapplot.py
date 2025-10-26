from __future__ import annotations

import os
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# Matplotlib in headless mode for PNG export
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from .utils import extract_timestamp_text

try:
    import geopandas as gpd  # type: ignore
    from shapely.geometry import Point  # type: ignore
except Exception:  # pragma: no cover
    gpd = None  # type: ignore
    Point = None  # type: ignore

try:
    import contextily as cx  # type: ignore
    from xyzservices import TileProvider  # type: ignore
except Exception:  # pragma: no cover
    cx = None  # type: ignore
    TileProvider = None  # type: ignore

try:
    from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore

try:
    import h3  # type: ignore
except Exception:  # pragma: no cover
    h3 = None  # type: ignore


def _get_provider(provider: str):
    """
    Resolve a basemap provider for contextily.add_basemap.

    Supports:
      - "carto" (CartoDB Positron)
      - "osm" (OpenStreetMap Mapnik)
      - "google" / "google-satellite" (Google Satellite via XYZ tiles)
      - Any custom XYZ URL starting with http(s)://
    """
    if cx is None:
        raise RuntimeError("contextily not installed; install with `pip install panogeo[geo]`")

    p = provider.lower().strip()
    if p in ("carto", "cartodb", "positron", "cartodb-positron"):
        return cx.providers.CartoDB.Positron
    if p in ("osm", "openstreetmap", "mapnik"):
        return cx.providers.OpenStreetMap.Mapnik
    # Use Esri World Imagery as a reliable satellite source (Google tiles often 404)
    if p in ("google", "google-satellite", "google_satellite", "g_sat", "esri", "esri-world", "worldimagery", "satellite"):
        return cx.providers.Esri.WorldImagery
    # Japan GSI seamless photo tiles (primary key: japan_gsi_seamless). Keep legacy aliases.
    if p in ("japan_gsi_seamless", "japan_gsi", "gsi", "gsi-seamlessphoto", "gsi_seamlessphoto", "seamlessphoto", "gsi-seamless"):
        url = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
        if TileProvider is not None:
            return TileProvider(
                name="GSI Seamless Photo",
                url=url,
                attribution="© 国土地理院",
                min_zoom=2,
                max_zoom=18,
            )
        return url
    # Japan GSI airphoto tiles
    if p in ("japan_gsi_air", "gsi_air", "gsi-air", "airphoto"):
        url = "https://cyberjapandata.gsi.go.jp/xyz/airphoto/{z}/{x}/{y}.png"
        if TileProvider is not None:
            return TileProvider(
                name="GSI Airphoto",
                url=url,
                attribution="© 国土地理院",
                min_zoom=2,
                max_zoom=18,
            )
        return url
    if p.startswith("http://") or p.startswith("https://"):
        # Custom XYZ template
        if TileProvider is not None:
            return TileProvider(name="Custom", url=provider, attribution="", min_zoom=0, max_zoom=22)
        return provider
    raise ValueError(f"Unknown provider '{provider}'. Use 'carto', 'osm', 'google' or an XYZ URL.")


def save_points_basemap(
    geo_csv: str,
    out_png: str,
    provider: str = "carto",
    zoom: Optional[int] = None,
    point_size: float = 12.0,
    alpha: float = 0.9,
    point_color: str = "#9A0EEA",
    dpi: int = 150,
    margin_frac: float = 0.10,
    image_name: Optional[str] = None,
    fixed_extent_merc: Optional[Tuple[float, float, float, float]] = None,
    stamp_timestamp: bool = False,
) -> str:
    """
    Save a PNG image of geolocated points plotted over a basemap.

    Parameters
    ----------
    geo_csv : str
        Path to CSV with at least columns: 'lat', 'lon'. Optional 'conf' for confidence.
    out_png : str
        Output PNG path.
    provider : str
        Basemap provider: 'carto', 'osm', 'google', or an XYZ URL.
    zoom : Optional[int]
        Explicit tile zoom. If None, contextily chooses based on extent.
    point_size : float
        Base point size in pixels (approx.).
    alpha : float
        Point alpha.
    dpi : int
        Output DPI for the PNG.
    margin_frac : float
        Fractional padding around data extent in Web Mercator.
    point_color : str
        Hex or Matplotlib color for point markers.
    """
    if gpd is None or cx is None:
        raise RuntimeError("Geo plotting requires geopandas and contextily; install with `pip install panogeo[geo]`")

    df = pd.read_csv(geo_csv)
    if df.empty:
        raise ValueError("No rows in geo CSV")
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("CSV must contain 'lat' and 'lon' columns")
    if image_name is not None:
        if "image" not in df.columns:
            raise ValueError("--image filter provided but CSV has no 'image' column")
        df = df[df["image"].astype(str) == str(image_name)]
        if df.empty:
            raise ValueError(f"No rows for image '{image_name}' in CSV")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after filtering NaN/inf lat/lon")

    # Build GeoDataFrame (WGS84 → Web Mercator)
    geometry = [Point(float(lon), float(lat)) for lat, lon in zip(df["lat"].astype(float), df["lon"].astype(float))]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf_merc = gdf.to_crs(epsg=3857)

    if fixed_extent_merc is not None:
        extent = (float(fixed_extent_merc[0]), float(fixed_extent_merc[1]), float(fixed_extent_merc[2]), float(fixed_extent_merc[3]))
    else:
        xmin, ymin, xmax, ymax = gdf_merc.total_bounds
        xpad = (xmax - xmin) * margin_frac
        ypad = (ymax - ymin) * margin_frac
        # Fallback padding for a single point (zero-area bbox)
        if xpad == 0:
            xpad = 100.0
        if ypad == 0:
            ypad = 100.0
        extent = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)

    # Match figure aspect ratio to the metric extent (so squares stay square)
    span_x = float(extent[1] - extent[0])
    span_y = float(extent[3] - extent[2])
    base_w_in = 8.0
    fig_w = base_w_in
    fig_h = max(1e-6, base_w_in * (span_y / span_x))
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])  # full-figure axis, no margins
    ax.set_axis_off()

    # Set extent and enforce equal axis scaling in Web Mercator meters
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")

    src = _get_provider(provider)
    cx.add_basemap(ax, source=src, crs="EPSG:3857", zoom=zoom, attribution=True, reset_extent=False)
    # Re-apply exact extent and aspect in case add_basemap adjusted limits internally
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")

    if "conf" in gdf_merc.columns:
        conf = gdf_merc["conf"].astype(float).clip(0.0, 1.0)
        sizes = (conf * (point_size * 3.0) + point_size).values
        gdf_merc.plot(ax=ax, markersize=sizes, alpha=alpha, color=point_color, legend=False)
    else:
        gdf_merc.plot(ax=ax, markersize=point_size, alpha=alpha, color=point_color)

    out_dir = os.path.dirname(os.path.abspath(out_png))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Optional timestamp from image_name or out_png base if available
    if stamp_timestamp:
        source_name = image_name if image_name is not None else os.path.splitext(os.path.basename(out_png))[0]
        ts = extract_timestamp_text(str(source_name))
        if ts:
            # Place text at bottom-left in figure coordinates
            fig.text(
                0.01,
                0.02,
                ts,
                fontsize=10,
                color="#ffffff",
                zorder=1000,
                ha="left",
                va="bottom",
                bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor="none", pad=3.0),
            )
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def save_all_images_basemap(
    geo_csv: str,
    out_dir: str,
    provider: str = "carto",
    zoom: Optional[int] = None,
    point_size: float = 12.0,
    alpha: float = 0.9,
    point_color: str = "#9A0EEA",
    dpi: int = 150,
    margin_frac: float = 0.10,
    fixed_extent_merc: Optional[Tuple[float, float, float, float]] = None,
    stamp_timestamp: bool = False,
) -> List[str]:
    """
    Export a PNG basemap for each unique image, all sharing the same basemap extent.

    Returns the list of saved file paths.
    """
    if gpd is None or cx is None:
        raise RuntimeError("Geo plotting requires geopandas and contextily; install with `pip install panogeo[geo]`")

    df = pd.read_csv(geo_csv)
    if df.empty:
        raise ValueError("No rows in geo CSV")
    if not {"lat", "lon", "image"}.issubset(df.columns):
        raise ValueError("CSV must contain 'lat', 'lon', and 'image' columns")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after filtering NaN/inf lat/lon")

    if fixed_extent_merc is None:
        geometry = [Point(float(lon), float(lat)) for lat, lon in zip(df["lat"].astype(float), df["lon"].astype(float))]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        gdf_merc = gdf.to_crs(epsg=3857)

        xmin, ymin, xmax, ymax = gdf_merc.total_bounds
        xpad = (xmax - xmin) * margin_frac
        ypad = (ymax - ymin) * margin_frac
        if xpad == 0:
            xpad = 100.0
        if ypad == 0:
            ypad = 100.0
        fixed_extent = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)
    else:
        fixed_extent = (
            float(fixed_extent_merc[0]),
            float(fixed_extent_merc[1]),
            float(fixed_extent_merc[2]),
            float(fixed_extent_merc[3]),
        )

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []
    for image_name in sorted(df["image"].astype(str).unique()):
        out_png = os.path.join(out_dir, f"{image_name}_{provider}.png")
        out = save_points_basemap(
            geo_csv=geo_csv,
            out_png=out_png,
            provider=provider,
            zoom=zoom,
            point_size=point_size,
            alpha=alpha,
            point_color=point_color,
            dpi=dpi,
            margin_frac=margin_frac,
            image_name=image_name,
            fixed_extent_merc=fixed_extent,
            stamp_timestamp=stamp_timestamp,
        )
        saved.append(out)
    return saved


def merc_extent_from_center(center_lat: float, center_lon: float, width_m: float, height_m: float) -> Tuple[float, float, float, float]:
    """
    Compute a Web Mercator (EPSG:3857) extent of given width/height in meters
    centered at the provided WGS84 lat/lon.
    """
    if Transformer is None:
        raise RuntimeError("pyproj not installed; install with `pip install pyproj` or `pip install panogeo[geo]`")
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = tf.transform(float(center_lon), float(center_lat))
    half_w = float(width_m) * 0.5
    half_h = float(height_m) * 0.5
    return (x - half_w, x + half_w, y - half_h, y + half_h)


def _h3_geo_to_cell(lat: float, lon: float, res: int):
    """Compatibility wrapper for h3 v3/v4."""
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lon, res)  # type: ignore[attr-defined]
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, res)  # type: ignore[attr-defined]
    raise RuntimeError("Unsupported h3 python API; install 'h3' >= 3.7")


def _h3_cell_to_boundary(cell):
    """Return boundary as list of (lon, lat) tuples, compatible with h3 v3/v4."""
    if hasattr(h3, "h3_to_geo_boundary"):
        # v3 API: returns list of (lat, lng) when geo_json=False (default)
        coords = h3.h3_to_geo_boundary(cell)  # type: ignore[attr-defined]
    elif hasattr(h3, "cell_to_boundary"):
        # v4 API: returns list of (lat, lng). Some builds do not accept keyword args.
        coords = h3.cell_to_boundary(cell)  # type: ignore[attr-defined]
    else:
        raise RuntimeError("Unsupported h3 python API; install 'h3' >= 3.7")
    # Normalize to list of (lon, lat)
    return [(float(lng), float(lat)) for (lat, lng) in coords]


def save_h3_basemap(
    geo_csv: str,
    out_png: str,
    provider: str = "carto",
    zoom: Optional[int] = None,
    h3_res: int = 10,
    weight_col: Optional[str] = None,
    alpha: float = 0.85,
    dpi: int = 150,
    margin_frac: float = 0.10,
    fixed_extent_merc: Optional[Tuple[float, float, float, float]] = None,
    cmap: str = "viridis",
    edgecolor: str = "#ffffff",
    linewidth: float = 0.4,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> str:
    """
    Render and save an H3-aggregated people-location map as a PNG basemap overlay.

    Aggregates points into hex cells at the given H3 resolution. If weight_col is
    provided and exists, values are summed per cell; otherwise counts are used.
    """
    if gpd is None or cx is None:
        raise RuntimeError("Geo plotting requires geopandas and contextily; install with `pip install panogeo[geo]`")
    if h3 is None:
        raise RuntimeError("H3 not installed; install with `pip install h3`")

    df = pd.read_csv(geo_csv)
    if df.empty:
        raise ValueError("No rows in geo CSV")
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("CSV must contain 'lat' and 'lon' columns")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after filtering NaN/inf lat/lon")

    # Compute H3 cell per row
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    cells = [_h3_geo_to_cell(float(la), float(lo), int(h3_res)) for la, lo in zip(lat, lon)]
    df["h3"] = cells

    # Aggregate per cell
    if weight_col is not None and weight_col in df.columns:
        agg = df.groupby("h3")[weight_col].sum().rename("value").reset_index()
    else:
        agg = df.groupby("h3").size().rename("value").reset_index()

    # Build polygons from H3 cells in WGS84
    polys = []
    from shapely.geometry import Polygon  # local import to avoid top-level if missing
    for cell in agg["h3"]:
        boundary = _h3_cell_to_boundary(cell)  # list of (lon, lat)
        # Ensure closed ring
        if boundary[0] != boundary[-1]:
            boundary = boundary + [boundary[0]]
        polys.append(Polygon(boundary))

    gdf_hex = gpd.GeoDataFrame(agg, geometry=polys, crs="EPSG:4326")
    gdf_hex_merc = gdf_hex.to_crs(epsg=3857)

    # Determine extent
    if fixed_extent_merc is not None:
        extent = (
            float(fixed_extent_merc[0]),
            float(fixed_extent_merc[1]),
            float(fixed_extent_merc[2]),
            float(fixed_extent_merc[3]),
        )
    else:
        xmin, ymin, xmax, ymax = gdf_hex_merc.total_bounds
        xpad = (xmax - xmin) * margin_frac
        ypad = (ymax - ymin) * margin_frac
        if xpad == 0:
            xpad = 100.0
        if ypad == 0:
            ypad = 100.0
        extent = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)

    # Figure sizing from extent ratio, full-figure axis
    span_x = float(extent[1] - extent[0])
    span_y = float(extent[3] - extent[2])
    base_w_in = 8.0
    fig_w = base_w_in
    fig_h = max(1e-6, base_w_in * (span_y / span_x))
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()

    # Extent and basemap
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    src = _get_provider(provider)
    cx.add_basemap(ax, source=src, crs="EPSG:3857", zoom=zoom, attribution=True, reset_extent=False)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")

    # Draw hexagons
    gdf_hex_merc.plot(
        ax=ax,
        column="value",
        cmap=cmap,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        vmin=vmin,
        vmax=vmax,
    )

    out_dir = os.path.dirname(os.path.abspath(out_png))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


