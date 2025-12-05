from __future__ import annotations

import os
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# Matplotlib in headless mode for PNG export
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from .utils import extract_timestamp_text

try:
    import geopandas as gpd  # type: ignore
    from shapely.geometry import Point  # type: ignore
except Exception:  # pragma: no cover
    gpd = None  # type: ignore
    Point = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import contextily as cx  # type: ignore
    from xyzservices import TileProvider  # type: ignore
    # For direct tile fetching when we want to pre-cache the basemap
    from contextily import tile as cx_tile  # type: ignore
except Exception:  # pragma: no cover
    cx = None  # type: ignore
    TileProvider = None  # type: ignore
    cx_tile = None  # type: ignore

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

    # Basemap with robust zoom and fallback
    def _resolve_zoom(src_obj, requested):
        if requested is not None:
            try:
                return int(requested)
            except Exception:
                pass
        try:
            z = int(getattr(src_obj, "max_zoom", 18)) - 1
            return max(2, z)
        except Exception:
            return 17

    def _add_basemap_with_fallback(ax, provider_name: str, requested_zoom):
        last_err = None
        # Try primary provider, decreasing zoom a few times
        src_obj = _get_provider(provider_name)
        z = _resolve_zoom(src_obj, requested_zoom)
        for _ in range(5):
            try:
                cx.add_basemap(ax, source=src_obj, crs="EPSG:3857", zoom=z, attribution=True, reset_extent=False)
                return
            except Exception as e:
                last_err = e
                z = max(2, z - 1)
        # Fallback providers
        for fb in ("esri-world", "carto", "osm"):
            try:
                src_fb = _get_provider(fb)
                z = _resolve_zoom(src_fb, requested_zoom)
                cx.add_basemap(ax, source=src_fb, crs="EPSG:3857", zoom=z, attribution=True, reset_extent=False)
                return
            except Exception as e2:
                last_err = e2
        if last_err is not None:
            raise last_err

    _add_basemap_with_fallback(ax, provider, zoom)
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
    add_colorbar: bool = False,
    colorbar_label: Optional[str] = None,
    rasterized: Optional[bool] = None,
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
    # Leave room at the right for colorbar ticks when requested
    ax_rect = [0.0, 0.0, 0.93, 1.0] if add_colorbar else [0.0, 0.0, 1.0, 1.0]
    ax = fig.add_axes(ax_rect)
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
    plot_kwargs = dict(
        ax=ax,
        column="value",
        cmap=cmap,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        vmin=vmin,
        vmax=vmax,
    )
    if rasterized is not None:
        plot_kwargs["rasterized"] = bool(rasterized)
    col_plot = gdf_hex_merc.plot(**plot_kwargs)

    # Optional colorbar
    if add_colorbar:
        # Determine normalization consistent with the plot
        data_vals = gdf_hex_merc["value"].astype(float)
        vmin_val = float(vmin) if vmin is not None else float(data_vals.min())
        vmax_val = float(vmax) if vmax is not None else float(data_vals.max())
        if vmin_val == vmax_val:
            # Avoid degenerate norm
            vmin_val -= 0.5
            vmax_val += 0.5
        cmap_obj = cm.get_cmap(cmap)
        norm = Normalize(vmin=vmin_val, vmax=vmax_val)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
        # Improve readability on busy imagery backgrounds
        try:
            cbar.ax.set_facecolor((1.0, 1.0, 1.0, 0.75))
        except Exception:
            pass
        cbar.outline.set_edgecolor("#000000")
        cbar.ax.tick_params(labelsize=9, colors="#000000")
        # Use integer tick locator when showing counts
        is_count = (weight_col is None) or (weight_col not in df.columns)
        if is_count:
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune=None))
        if colorbar_label is None:
            colorbar_label = "count" if is_count else str(weight_col)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=10, color="#000000")

    out_dir = os.path.dirname(os.path.abspath(out_png))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Tight bbox ensures colorbar tick labels are not clipped
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def save_tracking_map_video(
    geo_csv: str,
    out_mp4: str,
    provider: str = "carto",
    zoom: Optional[int] = None,
    point_size: float = 16.0,
    alpha: float = 0.95,
    traj_max_frames: int = 60,
    dpi: int = 150,
    margin_frac: float = 0.10,
    fps: Optional[float] = None,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> str:
    """
    Render a map video with tracked people positions and short trajectories.

    Expects geo_csv to contain at least: 'frame', 'track_id', 'lat', 'lon' (and optionally 'image').
    """
    if gpd is None or cx is None:
        raise RuntimeError("Geo plotting requires geopandas and contextily; install with `pip install panogeo[geo]`")
    if cv2 is None:
        raise RuntimeError("OpenCV not available; install with `pip install opencv-python`")

    df = pd.read_csv(geo_csv)
    if df.empty:
        raise ValueError("No rows in geo CSV")
    required = {"frame", "track_id", "lat", "lon"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    # Sanitize
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    # Build fixed extent from all data to avoid jitter
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
    extent = (float(xmin - xpad), float(xmax + xpad), float(ymin - ypad), float(ymax + ypad))

    # Figure dims and corresponding video size
    span_x = float(extent[1] - extent[0])
    span_y = float(extent[3] - extent[2])
    base_w_in = 8.0
    fig_w = base_w_in
    fig_h = max(1e-6, base_w_in * (span_y / span_x))
    pixel_w = int(round(fig_w * dpi))
    pixel_h = int(round(fig_h * dpi))

    # Determine fps from data if not provided
    if fps is None:
        # Heuristic: if images (frames) are sparse, 10 fps; else 20
        fps = 20.0

    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, float(fps), (pixel_w, pixel_h))

    try:
        # Pre-split data by frame for speed
        by_frame = {int(k): v.copy() for k, v in gdf_merc.groupby("frame")}
        all_frames = sorted(by_frame.keys())

        def _resolve_zoom(src_obj, requested):
            if requested is not None:
                try:
                    return int(requested)
                except Exception:
                    pass
            try:
                z = int(getattr(src_obj, "max_zoom", 18)) - 1
                return max(2, z)
            except Exception:
                return 17

        def _add_basemap_with_fallback(ax, provider_name: str, requested_zoom):
            last_err = None
            src_obj = _get_provider(provider_name)
            z = _resolve_zoom(src_obj, requested_zoom)
            for _ in range(5):
                try:
                    cx.add_basemap(ax, source=src_obj, crs="EPSG:3857", zoom=z, attribution=False, reset_extent=False)
                    return
                except Exception as e:
                    last_err = e
                    z = max(2, z - 1)
            for fb in ("esri-world", "carto", "osm"):
                try:
                    src_fb = _get_provider(fb)
                    z = _resolve_zoom(src_fb, requested_zoom)
                    cx.add_basemap(ax, source=src_fb, crs="EPSG:3857", zoom=z, attribution=False, reset_extent=False)
                    return
                except Exception as e2:
                    last_err = e2
            if last_err is not None:
                raise last_err

        # Try to prefetch basemap once for the fixed extent to speed up rendering
        prefetched_basemap = None
        try:
            src_obj = _get_provider(provider)
            z0 = _resolve_zoom(src_obj, zoom)
            if cx_tile is not None:
                # cx_tile.bounds2img expects (west, south, east, north)
                img_tile, ext_bounds = cx_tile.bounds2img(
                    extent[0], extent[2], extent[1], extent[3],
                    zoom=z0, source=src_obj, ll=False
                )
                # Ensure we have RGB array (not masked) for imshow
                img_np = np.asarray(img_tile)
                if img_np.ndim == 2:
                    # grayscale -> RGB
                    img_np = np.stack([img_np, img_np, img_np], axis=2)
                elif img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                prefetched_basemap = (img_np, ext_bounds)
        except Exception:
            prefetched_basemap = None

        # Optional progress bar
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore
                pbar = tqdm(total=len(all_frames), desc=(progress_desc or "map-video"), unit="frame")
            except Exception:
                pbar = None

        # For color-coding different tracks deterministically
        def _id_color(oid: int) -> Tuple[float, float, float]:
            r = (37 * oid) % 255
            g = (17 * oid) % 255
            b = (13 * oid) % 255
            return (r / 255.0, g / 255.0, b / 255.0)

        for idx, f in enumerate(all_frames):
            # Collect recent history up to traj_max_frames
            f0 = max(all_frames[0], f - int(traj_max_frames) + 1)
            frames_range = [ff for ff in all_frames if f0 <= ff <= f]
            # Create a fresh figure per frame (Agg)
            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box")
            if prefetched_basemap is not None:
                img_np, ext_bounds = prefetched_basemap
                ax.imshow(img_np, extent=ext_bounds, origin="upper")
            else:
                _add_basemap_with_fallback(ax, provider, zoom)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            # Draw trajectories per track id
            # Build a dict: track_id -> list of (x,y) for frames in range (ordered)
            traj_points = {}
            for ff in frames_range:
                gff = by_frame.get(ff)
                if gff is None or gff.empty:
                    continue
                for r in gff.itertuples():
                    oid = int(getattr(r, "track_id"))
                    x = float(r.geometry.x)
                    y = float(r.geometry.y)
                    traj_points.setdefault(oid, []).append((x, y))
            for oid, pts in traj_points.items():
                if len(pts) >= 2:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    ax.plot(xs, ys, color=_id_color(oid), linewidth=2.0, alpha=0.9, solid_capstyle="round")

            # Draw current-frame points on top
            gcur = by_frame[f]
            if not gcur.empty:
                for r in gcur.itertuples():
                    oid = int(getattr(r, "track_id"))
                    x = float(r.geometry.x)
                    y = float(r.geometry.y)
                    ax.scatter([x], [y], s=point_size, color=_id_color(oid), alpha=alpha, zorder=100)

            # Render to RGB array
            fig.canvas.draw()
            # Prefer buffer_rgba (works across recent Matplotlib versions)
            h_px = int(fig.canvas.get_width_height()[1])
            w_px = int(fig.canvas.get_width_height()[0])
            try:
                buf = fig.canvas.buffer_rgba()
                img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape((h_px, w_px, 4))
                img = img_rgba[:, :, :3]  # drop alpha
            except Exception:
                # Fallback to tostring_rgb if available
                img = np.frombuffer(getattr(fig.canvas, "tostring_rgb")(), dtype=np.uint8).reshape((h_px, w_px, 3))
            plt.close(fig)

            # Convert RGB->BGR for OpenCV and ensure exact video size
            if img.shape[1] != pixel_w or img.shape[0] != pixel_h:
                img = cv2.resize(img, (pixel_w, pixel_h), interpolation=cv2.INTER_LINEAR)
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

            if pbar is not None:
                pbar.update(1)
            elif show_progress and (idx % 25 == 0):
                try:
                    print(f"[map-video] frame {idx+1}/{len(all_frames)}", flush=True)
                except Exception:
                    pass
    finally:
        writer.release()
        try:
            if pbar is not None:
                pbar.close()
        except Exception:
            pass

    return out_mp4

