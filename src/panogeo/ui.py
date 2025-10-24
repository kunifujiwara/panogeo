from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image


try:  # UI stack is optional and used mainly within Jupyter
    import ipywidgets as widgets
    from ipyleaflet import Map, Marker, basemaps, basemap_to_tiles, LayersControl, TileLayer
    from ipyevents import Event
except Exception:  # pragma: no cover
    widgets = None  # type: ignore
    Map = None  # type: ignore
    Marker = None  # type: ignore
    basemaps = None  # type: ignore
    basemap_to_tiles = None  # type: ignore
    LayersControl = None  # type: ignore
    TileLayer = None  # type: ignore
    Event = None  # type: ignore


@dataclass
class CalibPair:
    u_px: float
    v_px: float
    lon: float
    lat: float
    W: int
    H: int
    alt_m: Optional[float] = None


class CalibrationUI:
    def __init__(
        self,
        pano_path: str,
        map_center: Tuple[float, float],
        map_zoom: int = 18,
        display_width_px: int = 900,
        default_alt_m: Optional[float] = None,
    ) -> None:
        if widgets is None or Map is None or Event is None:
            raise RuntimeError(
                "UI dependencies not installed. Install with: pip install ipywidgets ipyleaflet ipyevents"
            )

        if not os.path.exists(pano_path):
            raise FileNotFoundError(pano_path)

        self.pano_path = pano_path
        self.image = Image.open(pano_path).convert("RGB")
        self.W, self.H = self.image.size
        self.display_width_px = int(display_width_px)
        self.display_height_px = int(round(self.H * (self.display_width_px / self.W)))
        self.default_alt_m = default_alt_m

        self.pending_pixel: Optional[Tuple[float, float]] = None
        self.pending_geo: Optional[Tuple[float, float]] = None
        self.pairs: List[CalibPair] = []
        self.markers: List[Marker] = []

        # Widgets
        self.out = widgets.Output(layout=widgets.Layout(max_height="200px", overflow_y="auto"))
        self.status = widgets.HTML(value="")
        self.alt_input = widgets.FloatText(value=default_alt_m if default_alt_m is not None else 0.0, description="alt_m")
        self.fn_input = widgets.Text(value="calibration_points.csv", description="CSV")
        self.btn_save = widgets.Button(description="Save CSV", button_style="success")
        self.btn_undo = widgets.Button(description="Undo last", button_style="warning")
        self.btn_clear = widgets.Button(description="Clear all", button_style="danger")

        # Build image widget
        bio = io.BytesIO()
        self.image.save(bio, format="JPEG", quality=92)
        bio.seek(0)
        self.img_widget = widgets.Image(value=bio.getvalue(), format="jpg")
        self.img_widget.layout.width = f"{self.display_width_px}px"
        self.img_widget.layout.height = f"{self.display_height_px}px"

        # Capture click on image
        self.img_events = Event(source=self.img_widget, watched_events=["click"])
        self.img_events.on_dom_event(self._on_image_click)

        # Build map widget
        center_lat, center_lon = map_center
        self.map = Map(center=(center_lat, center_lon), zoom=map_zoom, scroll_wheel_zoom=True, max_zoom=22)
        # Prefer Google Satellite via direct TileLayer; fallback to Esri WorldImagery
        try:
            google_sat = TileLayer(
                url='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                name='Google Satellite',
                attribution='Google Satellite',
                max_zoom=22,
            )
            self.map.layers = tuple([google_sat])
        except Exception:
            try:
                esri = basemap_to_tiles(basemaps.Esri.WorldImagery)
                try:
                    esri.max_zoom = 22
                except Exception:
                    pass
                self.map.layers = tuple([esri])
            except Exception:
                # Fallback silently if basemap cannot be added
                pass
        self.map.add_control(LayersControl(position="topright"))
        self.map.on_interaction(self._on_map_interaction)

        # Wire buttons
        self.btn_save.on_click(self._on_save)
        self.btn_clear.on_click(self._on_clear)
        self.btn_undo.on_click(self._on_undo)

        # Layout
        controls = widgets.HBox([self.alt_input, self.fn_input, self.btn_save, self.btn_undo, self.btn_clear])
        # Vertical layout: image on top, map below
        self.ui = widgets.VBox([
            widgets.HTML(value=f"<b>Image:</b> click to pick pixel; <b>Map:</b> click to pick geo. A pair is added when both are set."),
            controls,
            self.status,
            self.img_widget,
            self.map,
            widgets.HTML(value="<b>Log</b>"),
            self.out,
        ])
        self._set_status()

    # Event handlers
    def _on_image_click(self, event):
        ox = float(event.get("offsetX", event.get("relativeX", 0.0)))
        oy = float(event.get("offsetY", event.get("relativeY", 0.0)))
        u = ox * (self.W / self.display_width_px)
        v = oy * (self.H / self.display_height_px)
        self.pending_pixel = (u, v)
        self._log(f"Pixel picked: u={u:.1f}, v={v:.1f}")
        self._try_commit_pair()

    def _on_map_interaction(self, **kwargs):
        if kwargs.get("type") != "click":
            return
        latlng = kwargs.get("coordinates", None)
        if not latlng:
            return
        lat, lon = float(latlng[0]), float(latlng[1])
        self.pending_geo = (lon, lat)
        m = Marker(location=(lat, lon))
        self.map.add_layer(m)
        self.markers.append(m)
        self._log(f"Geo picked: lon={lon:.6f}, lat={lat:.6f}")
        self._try_commit_pair()

    def _try_commit_pair(self) -> None:
        if self.pending_pixel is None or self.pending_geo is None:
            self._set_status()
            return
        u, v = self.pending_pixel
        lon, lat = self.pending_geo
        alt_m = float(self.alt_input.value) if self.alt_input.value is not None else None
        self.pairs.append(CalibPair(u_px=float(u), v_px=float(v), lon=float(lon), lat=float(lat), W=self.W, H=self.H, alt_m=alt_m))
        self._log(f"Added pair #{len(self.pairs)}  (u={u:.1f}, v={v:.1f}) ↔ (lon={lon:.6f}, lat={lat:.6f}, alt={alt_m})")
        self.pending_pixel, self.pending_geo = None, None
        self._set_status()

    # Actions
    def _on_save(self, _):
        if not self.pairs:
            self._log("Nothing to save.")
            return
        rows = []
        for p in self.pairs:
            row = {
                "u_px": p.u_px,
                "v_px": p.v_px,
                "lon": p.lon,
                "lat": p.lat,
                "W": p.W,
                "H": p.H,
            }
            if p.alt_m is not None:
                row["alt_m"] = p.alt_m
            rows.append(row)
        df = pd.DataFrame(rows)
        out_path = os.path.abspath(self.fn_input.value)
        df.to_csv(out_path, index=False)
        self._log(f"Saved {len(rows)} pairs to {out_path}")

    def _on_clear(self, _):
        self.pairs.clear()
        self.pending_pixel, self.pending_geo = None, None
        for m in self.markers:
            try:
                self.map.remove_layer(m)
            except Exception:
                pass
        self.markers.clear()
        self._log("Cleared all pairs and markers.")
        self._set_status()

    def _on_undo(self, _):
        if not self.pairs:
            self._log("Nothing to undo.")
            return
        self.pairs.pop()
        if self.markers:
            m = self.markers.pop()
            try:
                self.map.remove_layer(m)
            except Exception:
                pass
        self._log("Removed last pair.")
        self._set_status()

    # Utilities
    def _log(self, msg: str) -> None:
        with self.out:
            print(msg)

    def _set_status(self) -> None:
        n = len(self.pairs)
        missing = []
        if self.pending_pixel is None:
            missing.append("pixel")
        if self.pending_geo is None:
            missing.append("geo")
        cue = " and ".join(missing) if missing else "none"
        self.status.value = f"<span>Pairs: <b>{n}</b>. Waiting for: <i>{cue}</i></span>"

    def display(self):  # convenience
        if widgets is None:
            raise RuntimeError("widgets not available")
        return self.ui

    # Make the object itself render in Jupyter if used as the last expression
    def _ipython_display_(self):  # pragma: no cover
        try:
            from IPython.display import display as _display
            _display(self.ui)
        except Exception:
            pass


def launch_calibration_ui(
    pano_path: str,
    map_center: Tuple[float, float],
    map_zoom: int = 18,
    display_width_px: int = 900,
    default_alt_m: Optional[float] = None,
) -> CalibrationUI:
    """
    Launch an interactive calibration UI inside Jupyter.

    - Click on the panoramic image to select pixel coordinates (u_px, v_px)
    - Click on the map to select geographic coordinates (lon, lat)
    - When both are set, a pair is recorded automatically
    - Save pairs to a CSV compatible with solve_calibration

    Returns a CalibrationUI instance; display() it or rely on returned widget in notebooks.
    """
    ui = CalibrationUI(
        pano_path=pano_path,
        map_center=map_center,
        map_zoom=map_zoom,
        display_width_px=display_width_px,
        default_alt_m=default_alt_m,
    )
    return ui



