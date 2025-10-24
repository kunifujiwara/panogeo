from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw


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
        display_width_px: Optional[int] = None,
        default_alt_m: Optional[float] = None,
        enable_zoom: bool = True,
        image_viewport_height_px: int = 480,
    ) -> None:
        if widgets is None or Map is None or Event is None:
            raise RuntimeError(
                "UI dependencies not installed. Install with: pip install ipywidgets ipyleaflet ipyevents"
            )

        if not os.path.exists(pano_path):
            raise FileNotFoundError(pano_path)

        self.pano_path = pano_path
        self.image_base = Image.open(pano_path).convert("RGB")
        self.W, self.H = self.image_base.size
        # If display width not provided, default to full image width for maximum resolution
        self.base_display_width_px = int(display_width_px) if display_width_px else int(self.W)
        # Zoom state
        self.enable_zoom = bool(enable_zoom)
        self.zoom_value = 1.0
        self.display_width_px = int(round(self.base_display_width_px * self.zoom_value))
        self.display_height_px = int(round(self.H * (self.display_width_px / self.W)))
        self.default_alt_m = default_alt_m
        self.image_viewport_height_px = int(image_viewport_height_px)

        self.pending_pixel: Optional[Tuple[float, float]] = None
        self.pending_geo: Optional[Tuple[float, float]] = None
        self.pairs: List[CalibPair] = []
        self.markers: List[Marker] = []
        # Pan state for drag-to-pan
        self.pan_x_px: float = 0.0
        self.pan_y_px: float = 0.0
        self._is_dragging: bool = False
        self._drag_start_client: Tuple[float, float] = (0.0, 0.0)
        self._drag_start_pan: Tuple[float, float] = (0.0, 0.0)
        self._drag_moved: bool = False
        self._viewport_w_px: Optional[float] = None
        self._viewport_h_px: Optional[float] = None

        # Widgets
        self.out = widgets.Output(layout=widgets.Layout(max_height="200px", overflow_y="auto"))
        self.status = widgets.HTML(value="")
        self.alt_input = widgets.FloatText(value=default_alt_m if default_alt_m is not None else 0.0, description="alt_m")
        self.fn_input = widgets.Text(value="output/calib_points.csv", description="CSV")
        self.btn_save = widgets.Button(description="Save CSV", button_style="success")
        self.btn_undo = widgets.Button(description="Undo last", button_style="warning")
        self.btn_clear = widgets.Button(description="Clear all", button_style="danger")

        # Optional zoom control
        if self.enable_zoom:
            self.zoom_slider = widgets.FloatSlider(
                value=1.0, min=0.25, max=6.0, step=0.01, description="Zoom", readout_format=".2f"
            )
            self.zoom_slider.observe(self._on_zoom_change, names="value")
        else:
            self.zoom_slider = None

        # Build image widget
        self.img_widget = widgets.Image(format="jpg")
        self.img_widget.layout.width = f"{self.display_width_px}px"
        self.img_widget.layout.height = f"{self.display_height_px}px"
        # Prevent flexbox from shrinking the image; keep both dimensions fixed
        try:
            self.img_widget.layout.flex = "0 0 auto"
            self.img_widget.layout.min_width = f"{self.display_width_px}px"
            self.img_widget.layout.min_height = f"{self.display_height_px}px"
            self.img_widget.layout.max_width = f"{self.display_width_px}px"
            self.img_widget.layout.max_height = f"{self.display_height_px}px"
        except Exception:
            pass
        try:
            self.img_widget.layout.cursor = "crosshair"
        except Exception:
            pass

        # Scrollable container to allow viewing at high zoom
        self.img_container = widgets.Box([self.img_widget])
        self.img_container.layout = widgets.Layout(
            overflow="auto",
            width="100%",
            height=f"{self.image_viewport_height_px}px",
            border="1px solid #ddd",
            align_items="flex-start",
            justify_content="flex-start",
        )

        # Capture click and wheel on image
        self.img_events = Event(source=self.img_widget, watched_events=["click", "wheel"])
        self.img_events.on_dom_event(self._on_image_event)

        # Capture drag events on container for panning
        self.container_events = Event(
            source=self.img_container,
            watched_events=["mousedown", "mousemove", "mouseup", "mouseleave"],
        )
        self.container_events.on_dom_event(self._on_container_event)

        # Build map widget
        center_lat, center_lon = map_center
        self.map = Map(center=(center_lat, center_lon), zoom=map_zoom, scroll_wheel_zoom=True, max_zoom=22)
        try:
            self.map.default_style = {"cursor": "crosshair"}
            self.map.dragging_style = {"cursor": "crosshair"}
        except Exception:
            pass
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
        controls_children = [self.alt_input, self.fn_input, self.btn_save, self.btn_undo, self.btn_clear]
        if self.enable_zoom and self.zoom_slider is not None:
            controls_children.append(self.zoom_slider)
        controls = widgets.HBox(controls_children)
        # Vertical layout: image on top, map below
        self.ui = widgets.VBox([
            widgets.HTML(value=f"<b>Image:</b> click to pick pixel; <b>Map:</b> click to pick geo. A pair is added when both are set."),
            controls,
            self.status,
            self.img_container,
            self.map,
            widgets.HTML(value="<b>Log</b>"),
            self.out,
        ])
        self._set_status()
        self._update_image_widget()
        self._apply_image_layout()

    # Event handlers
    def _on_image_event(self, event):
        etype = event.get("type", "")
        if etype == "click":
            # Suppress click if it followed a drag operation
            if self._drag_moved:
                self._drag_moved = False
                return
            self._on_image_click(event)
            return
        if etype == "wheel" and self.enable_zoom and self.zoom_slider is not None:
            try:
                dy = float(event.get("deltaY", 0.0))
            except Exception:
                dy = 0.0
            # Wheel up (negative deltaY) -> zoom in; wheel down -> zoom out
            factor = 1.1 if dy < 0 else 0.9
            new_val = self.zoom_slider.value * factor
            # Clamp
            new_val = max(self.zoom_slider.min, min(self.zoom_slider.max, new_val))
            if abs(new_val - self.zoom_slider.value) > 1e-6:
                self.zoom_slider.value = new_val
            return

    def _on_container_event(self, event):
        etype = event.get("type", "")
        # Record viewport size if available
        vw = event.get("boundingRectWidth", None) or event.get("offsetWidth", None) or event.get("clientWidth", None)
        vh = event.get("boundingRectHeight", None) or event.get("offsetHeight", None) or event.get("clientHeight", None)
        if vw is not None:
            try:
                self._viewport_w_px = float(vw)
            except Exception:
                pass
        if vh is not None:
            try:
                self._viewport_h_px = float(vh)
            except Exception:
                pass

        if etype == "mousedown":
            try:
                cx = float(event.get("clientX", 0.0))
                cy = float(event.get("clientY", 0.0))
            except Exception:
                cx, cy = 0.0, 0.0
            self._is_dragging = True
            self._drag_moved = False
            self._drag_start_client = (cx, cy)
            self._drag_start_pan = (self.pan_x_px, self.pan_y_px)
            return
        if etype == "mousemove" and self._is_dragging:
            try:
                cx = float(event.get("clientX", 0.0))
                cy = float(event.get("clientY", 0.0))
            except Exception:
                cx, cy = 0.0, 0.0
            dx = cx - self._drag_start_client[0]
            dy = cy - self._drag_start_client[1]
            if abs(dx) > 2 or abs(dy) > 2:
                self._drag_moved = True
            # Dragging right should move image right (reduce pan_x)
            new_pan_x = self._drag_start_pan[0] - dx
            new_pan_y = self._drag_start_pan[1] - dy
            self.pan_x_px, self.pan_y_px = self._clamp_pan(new_pan_x, new_pan_y)
            self._apply_image_layout()
            return
        if etype in ("mouseup", "mouseleave"):
            self._is_dragging = False
            return

    def _on_image_click(self, event):
        # Prefer relativeX/Y which are reliably relative to element's content box
        rx = event.get("relativeX", None)
        ry = event.get("relativeY", None)
        ox = float(event.get("offsetX", 0.0))
        oy = float(event.get("offsetY", 0.0))
        x = float(rx) if rx is not None else float(ox)
        y = float(ry) if ry is not None else float(oy)

        # Try to read actual displayed element width/height from the event
        dw = event.get("boundingRectWidth", None) or event.get("offsetWidth", None) or event.get("clientWidth", None)
        dh = event.get("boundingRectHeight", None) or event.get("offsetHeight", None) or event.get("clientHeight", None)
        disp_w = float(dw) if dw is not None else float(self.display_width_px)
        disp_h = float(dh) if dh is not None else float(self.display_height_px)

        # Clamp to [0, disp_*]
        if disp_w <= 0 or disp_h <= 0:
            disp_w, disp_h = float(self.display_width_px), float(self.display_height_px)
        x = max(0.0, min(x, disp_w))
        y = max(0.0, min(y, disp_h))

        u = x * (self.W / disp_w)
        v = y * (self.H / disp_h)
        self.pending_pixel = (u, v)
        self._log(f"Pixel picked: u={u:.1f}, v={v:.1f}")
        self._update_image_widget()
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
        self._update_image_widget()

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
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
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
        self._update_image_widget()

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
        self._update_image_widget()

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

    def _on_zoom_change(self, change) -> None:
        try:
            self.zoom_value = float(change.get("new", 1.0))
        except Exception:
            self.zoom_value = 1.0
        # Recompute display size based on zoom
        self.display_width_px = int(round(self.base_display_width_px * self.zoom_value))
        self.display_height_px = int(round(self.H * (self.display_width_px / self.W)))
        self.img_widget.layout.width = f"{self.display_width_px}px"
        self.img_widget.layout.height = f"{self.display_height_px}px"
        try:
            self.img_widget.layout.flex = "0 0 auto"
            self.img_widget.layout.min_width = f"{self.display_width_px}px"
            self.img_widget.layout.min_height = f"{self.display_height_px}px"
            self.img_widget.layout.max_width = f"{self.display_width_px}px"
            self.img_widget.layout.max_height = f"{self.display_height_px}px"
        except Exception:
            pass
        # Clamp pan to new bounds and apply
        self.pan_x_px, self.pan_y_px = self._clamp_pan(self.pan_x_px, self.pan_y_px)
        self._apply_image_layout()
        self._update_image_widget()

    def _update_image_widget(self) -> None:
        # Render markers on a resized copy for display
        img = self.image_base.resize((self.display_width_px, self.display_height_px), Image.BILINEAR)
        draw = ImageDraw.Draw(img)

        def _draw_dot(x: float, y: float, color_outline: str, color_fill: str, r: int = 6):
            bbox = [x - r, y - r, x + r, y + r]
            # outline for contrast
            draw.ellipse(bbox, outline=color_outline, width=3)
            draw.ellipse([x - (r - 2), y - (r - 2), x + (r - 2), y + (r - 2)], fill=color_fill)

        # Confirmed pairs
        for p in self.pairs:
            xd = (p.u_px / self.W) * self.display_width_px
            yd = (p.v_px / self.H) * self.display_height_px
            _draw_dot(xd, yd, color_outline="#ffffff", color_fill="#e53935", r=7)

        # Pending pixel (if any)
        if self.pending_pixel is not None:
            u, v = self.pending_pixel
            xd = (u / self.W) * self.display_width_px
            yd = (v / self.H) * self.display_height_px
            _draw_dot(xd, yd, color_outline="#000000", color_fill="#ffd54f", r=7)

        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=92)
        bio.seek(0)
        self.img_widget.value = bio.getvalue()

    def _apply_image_layout(self) -> None:
        # Position image within container according to pan offsets
        try:
            self.img_widget.layout.position = "relative"
            self.img_widget.layout.left = f"{-int(round(self.pan_x_px))}px"
            self.img_widget.layout.top = f"{-int(round(self.pan_y_px))}px"
        except Exception:
            pass

    def _clamp_pan(self, px: float, py: float) -> Tuple[float, float]:
        # Clamp pan so image stays within container viewport if viewport size known
        vmax_x = max(0.0, float(self.display_width_px) - float(self._viewport_w_px) if self._viewport_w_px is not None else 0.0)
        vmax_y = max(0.0, float(self.display_height_px) - float(self._viewport_h_px) if self._viewport_h_px is not None else 0.0)
        if self._viewport_w_px is None:
            cx = max(0.0, px)
        else:
            cx = min(max(0.0, px), vmax_x)
        if self._viewport_h_px is None:
            cy = max(0.0, py)
        else:
            cy = min(max(0.0, py), vmax_y)
        return cx, cy

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
    display_width_px: Optional[int] = None,
    default_alt_m: Optional[float] = None,
    enable_zoom: bool = True,
    image_viewport_height_px: int = 480,
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
        enable_zoom=enable_zoom,
        image_viewport_height_px=image_viewport_height_px,
    )
    return ui



