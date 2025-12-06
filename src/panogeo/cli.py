from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from .geometry import shift_equirect
from .calibration import solve_calibration, save_calibration
from .geolocate import geolocate_detections
from .perspective import solve_homography_from_csv, save_homography, geolocate_detections_perspective
from .mapplot import save_points_basemap, save_all_images_basemap
from .video import compose_side_by_side_video  # noqa: F401

try:
    import folium
    from folium.plugins import HeatMap
except Exception:  # pragma: no cover
    folium = None  # type: ignore


def cmd_shift(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG")
    files = [f for f in sorted(os.listdir(args.in_dir)) if f.endswith(exts)]
    for f in files:
        p = os.path.join(args.in_dir, f)
        with Image.open(p) as im:
            exif = im.info.get("exif")
            out = shift_equirect(im, degrees=float(args.degrees))
            out_path = os.path.join(args.out_dir, f)
            save_kwargs = {"exif": exif} if exif and out_path.lower().endswith(('.jpg', '.jpeg')) else {}
            if getattr(args, "stamp", False):
                # Try to overlay timestamp extracted from filename
                try:
                    from .utils import extract_timestamp_text
                    ts = extract_timestamp_text(f)
                except Exception:
                    ts = None
                if ts and out.mode != "RGBA":
                    out = out.convert("RGBA")
                if ts:
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(out)
                    W, H = out.size
                    font_size = max(14, int(H / 40))
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()
                    text_w, text_h = draw.textbbox((0, 0), ts, font=font)[2:]
                    x = int(W * 0.02)
                    y = int(H * 0.96 - text_h)
                    # shadow box
                    box_pad = 6
                    draw.rectangle([x - box_pad, y - box_pad, x + text_w + box_pad, y + text_h + box_pad], fill=(0, 0, 0, 160))
                    draw.text((x, y), ts, fill=(255, 255, 255, 255), font=font)
                if out.mode == "RGBA" and not out_path.lower().endswith(".png"):
                    out = out.convert("RGB")
            out.save(out_path, **save_kwargs)
    print(f"Saved {len(files)} image(s) to {args.out_dir}")


def cmd_detect(args: argparse.Namespace) -> None:
    # Lazy import to avoid loading heavy dependencies (cv2/ultralytics) unless needed
    from .detection import detect_folder, DetectConfig, TilingConfig

    dcfg = DetectConfig(
        model_name=args.model,
        conf_thres=args.conf,
        iou_thres=args.iou,
        containment_thr=args.containment_thr,
        device=args.device,
        batch_tiles=args.batch_tiles,
        fuse_model=args.fuse_model,
        half=bool(args.half),
    )
    tcfg = TilingConfig(
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        overlap=args.overlap,
        imgsz=args.imgsz,
        min_box_h_px=args.min_box_h,
    )
    # Parse bbox color (accepts hex like #RRGGBB or comma-separated B,G,R)
    def _parse_bbox_color(value: Optional[str]):
        if not value:
            return (40, 220, 40)
        s = str(value).strip()
        try:
            if s.startswith("#"):
                s = s.lstrip('#')
                if len(s) == 6:
                    r = int(s[0:2], 16)
                    g = int(s[2:4], 16)
                    b = int(s[4:6], 16)
                    return (b, g, r)
            # comma-separated, allow spaces
            parts = [p.strip() for p in s.split(',')]
            if len(parts) == 3:
                b, g, r = (int(parts[0]), int(parts[1]), int(parts[2]))
                return (b, g, r)
        except Exception:
            pass
        # Fallback to default if parsing fails
        return (40, 220, 40)

    agg_csv = detect_folder(
        args.images_dir,
        args.output_dir,
        dcfg=dcfg,
        tcfg=tcfg,
        annotate=bool(args.annotate),
        annotate_dir=args.annotate_dir,
        stamp_timestamp=bool(getattr(args, "stamp", False)),
        bbox_color_bgr=_parse_bbox_color(getattr(args, "bbox_color", None)),
    )
    print(f"Detections saved: {agg_csv}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    res = solve_calibration(
        calib_csv=args.calib_csv,
        cam_lat=args.cam_lat,
        cam_lon=args.cam_lon,
        camera_alt_m=args.camera_alt_m,
        ground_alt_m=args.ground_alt_m,
        default_width=args.width,
        default_height=args.height,
        optimize_cam_position=args.optimize_cam_position,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    out_npz = os.path.join(args.output_dir, "calibration_cam2enu.npz")
    # Persist optimized camera position (or initial if optimization disabled)
    save_calibration(out_npz, res, res.cam_lat, res.cam_lon, res.camera_alt_m, args.ground_alt_m)
    print(f"Saved calibration to: {out_npz}")
    print(f"yaw={res.yaw_deg:.2f}°, pitch={res.pitch_deg:.2f}°, roll={res.roll_deg:.2f}°")
    print(f"CAM_LAT={res.cam_lat:.8f}, CAM_LON={res.cam_lon:.8f}, CAMERA_ALT_M={res.camera_alt_m:.2f}")


def cmd_calibrate_persp(args: argparse.Namespace) -> None:
    H, ref_lat, ref_lon = solve_homography_from_csv(args.calib_csv)
    os.makedirs(args.output_dir, exist_ok=True)
    out_npz = os.path.join(args.output_dir, "calibration_perspective.npz")
    save_homography(out_npz, H, ref_lat=ref_lat, ref_lon=ref_lon, ground_alt_m=0.0)
    print(f"Saved perspective homography to: {out_npz}")


def cmd_geolocate(args: argparse.Namespace) -> None:
    xy_csv, geo_csv = geolocate_detections(
        detections_csv=args.detections_csv,
        calibration_npz=args.calibration,
        output_dir=args.output_dir,
        dem_path=args.dem,
        max_range_m=args.max_range,
        step_m=args.step,
    )
    print(f"Saved: {xy_csv}\nSaved: {geo_csv}")


def cmd_geolocate_persp(args: argparse.Namespace) -> None:
    xy_csv, geo_csv = geolocate_detections_perspective(
        detections_csv=args.detections_csv,
        homography_npz=args.homography,
        output_dir=args.output_dir,
        debug=bool(getattr(args, "debug", False)),
        calib_csv=getattr(args, "calib_csv", None),
        gate_margin_m=float(getattr(args, "gate_margin_m", 150.0)),
        drop_outside=bool(getattr(args, "drop_outside", True)),
    )
    print(f"Saved: {xy_csv}\nSaved: {geo_csv}")


def cmd_map(args: argparse.Namespace) -> None:
    if folium is None:
        print("folium not installed; install with `pip install folium`", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(args.geo_csv)
    if df.empty:
        print("No rows in geo CSV", file=sys.stderr)
        sys.exit(1)
    lat0 = float(args.center_lat) if args.center_lat is not None else float(df["lat"].mean())
    lon0 = float(args.center_lon) if args.center_lon is not None else float(df["lon"].mean())
    m = folium.Map(location=[lat0, lon0], zoom_start=args.zoom, tiles=None)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='© OpenStreetMap contributors © CARTO',
        name="CartoDB Positron",
    ).add_to(m)
    heat_data = [[float(r.lat), float(r.lon), float(r.conf) if 'conf' in df.columns else 1.0] for r in df.itertuples()]
    HeatMap(heat_data, radius=12, blur=18, min_opacity=0.25, max_zoom=21).add_to(m)
    m.save(args.out_html)
    print(f"Saved map: {args.out_html}")


def cmd_map_png(args: argparse.Namespace) -> None:
    try:
        out = save_points_basemap(
            geo_csv=args.geo_csv,
            out_png=args.out_png,
            provider=args.provider,
            zoom=args.zoom,
            point_size=args.point_size,
            alpha=args.alpha,
            point_color=args.point_color,
            dpi=args.dpi,
            image_name=args.image,
            stamp_timestamp=bool(getattr(args, "stamp", False)),
        )
        print(f"Saved PNG map: {out}")
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)


def cmd_map_png_all(args: argparse.Namespace) -> None:
    try:
        saved = save_all_images_basemap(
            geo_csv=args.geo_csv,
            out_dir=args.out_dir,
            provider=args.provider,
            zoom=args.zoom,
            point_size=args.point_size,
            alpha=args.alpha,
            point_color=args.point_color,
            dpi=args.dpi,
            margin_frac=args.margin,
            stamp_timestamp=bool(getattr(args, "stamp", False)),
        )
        for p in saved:
            print(f"Saved: {p}")
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

def cmd_compose_video(args: argparse.Namespace) -> None:
    out = compose_side_by_side_video(
        args.left,
        args.right,
        args.out,
        layout=args.layout,
        fps=args.fps,
        gap=int(args.gap),
    )
    print(f"Saved combined video: {out}")
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="panogeo", description="Panoramic detection and geolocation")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("shift-pano", help="Shift equirect images by yaw")
    sp.add_argument("--in-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--degrees", type=float, default=180.0)
    sp.add_argument("--stamp", action="store_true", help="overlay timestamp from input filename on saved image")
    sp.set_defaults(func=cmd_shift)

    sp = sub.add_parser("detect", help="Run YOLO detection on panoramas with tiling")
    sp.add_argument("--images-dir", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--model", default="yolov8s.pt")
    sp.add_argument("--conf", type=float, default=0.20)
    sp.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="IOU threshold for YOLO's per-tile NMS (higher merges more boxes). Note: cross-tile merge uses a fixed 0.55 internally."
    )
    sp.add_argument("--tile-w", type=int, default=1280)
    sp.add_argument("--tile-h", type=int, default=960)
    sp.add_argument("--overlap", type=float, default=0.30)
    sp.add_argument("--imgsz", type=int, default=1280)
    sp.add_argument("--min-box-h", type=int, default=40)
    sp.add_argument(
        "--containment-thr",
        type=float,
        default=None,
        help="If set (e.g., 0.98), merge boxes when IoA (smaller-box overlap) ≥ threshold",
    )
    sp.add_argument("--annotate", action="store_true", help="save annotated panorama jpgs")
    sp.add_argument("--annotate-dir", default=None, help="output dir for annotated images (default: OUTPUT/annotated)")
    sp.add_argument("--stamp", action="store_true", help="overlay timestamp from input filename on annotated images")
    sp.add_argument("--bbox-color", default=None, help="annotation bbox color: hex #RRGGBB or 'B,G,R'")
    sp.add_argument("--device", default="auto", help="device for inference: auto|cpu|cuda:0|mps")
    sp.add_argument("--batch-tiles", type=int, default=8, help="number of tiles per forward pass")
    sp.add_argument("--half", action="store_true", help="use half precision on CUDA (may be slightly different on CPU)")
    fuse_group = sp.add_mutually_exclusive_group()
    fuse_group.add_argument("--fuse-model", dest="fuse_model", action="store_true", help="fuse conv+bn for faster inference")
    fuse_group.add_argument("--no-fuse-model", dest="fuse_model", action="store_false", help="disable model fusion")
    sp.set_defaults(fuse_model=True)
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("calibrate", help="Estimate camera rotation and optionally camera position from pixel↔geo pairs")
    sp.add_argument("--calib-csv", required=True)
    sp.add_argument("--cam-lat", type=float, required=True)
    sp.add_argument("--cam-lon", type=float, required=True)
    sp.add_argument("--camera-alt-m", type=float, default=2.0)
    sp.add_argument("--ground-alt-m", type=float, default=0.0)
    sp.add_argument("--width", type=int, default=2048, help="pano width if W not in CSV")
    sp.add_argument("--height", type=int, default=1024, help="pano height if H not in CSV")
    opt_group = sp.add_mutually_exclusive_group()
    opt_group.add_argument("--optimize-cam-position", dest="optimize_cam_position", action="store_true", help="optimize CAM_LAT/LON/ALT jointly")
    opt_group.add_argument("--no-optimize-cam-position", dest="optimize_cam_position", action="store_false", help="disable optimizing camera position")
    sp.set_defaults(optimize_cam_position=True)
    sp.add_argument("--output-dir", required=True)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("calibrate-persp", help="Estimate pixel->geo homography from pixel↔geo pairs (perspective images)")
    sp.add_argument("--calib-csv", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.set_defaults(func=cmd_calibrate_persp)

    sp = sub.add_parser("geolocate", help="Intersect detection rays with ground/DEM using calibration")
    sp.add_argument("--detections-csv", required=True)
    sp.add_argument("--calibration", required=True, help="calibration_cam2enu.npz")
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--dem", default=None)
    sp.add_argument("--max-range", type=float, default=80.0)
    sp.add_argument("--step", type=float, default=1.0)
    sp.set_defaults(func=cmd_geolocate)

    sp = sub.add_parser("geolocate-persp", help="Map detections to lon/lat using pixel->geo homography (perspective images)")
    sp.add_argument("--detections-csv", required=True)
    sp.add_argument("--homography", required=True, help="calibration_perspective.npz")
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--calib-csv", default=None, help="calibration pairs CSV; when given, gate detections to calib bbox (+margin)")
    sp.add_argument("--gate-margin-m", type=float, default=150.0, help="margin (meters) around calib bbox for gating")
    sp.add_argument("--keep-outside", dest="drop_outside", action="store_false", help="do not drop points outside gate (set lon/lat NaN instead)")
    sp.add_argument("--debug", action="store_true")
    sp.set_defaults(func=cmd_geolocate_persp)

    sp = sub.add_parser("map", help="Build a Folium heatmap from geo CSV")
    sp.add_argument("--geo-csv", required=True)
    sp.add_argument("--out-html", required=True)
    sp.add_argument("--center-lat", type=float, default=None)
    sp.add_argument("--center-lon", type=float, default=None)
    sp.add_argument("--zoom", type=int, default=19)
    sp.set_defaults(func=cmd_map)

    sp = sub.add_parser("map-png", help="Export PNG basemap with points from geo CSV")
    sp.add_argument("--geo-csv", required=True)
    sp.add_argument("--out-png", required=True)
    sp.add_argument("--provider", default="carto", help="carto|osm|esri-world|japan_gsi_seamless|japan_gsi_air or XYZ URL")
    sp.add_argument("--zoom", type=int, default=None)
    sp.add_argument("--point-size", type=float, default=12.0)
    sp.add_argument("--alpha", type=float, default=0.9)
    sp.add_argument("--point-color", default="#9A0EEA", help="marker color for points (matplotlib color, e.g., #RRGGBB)")
    sp.add_argument("--dpi", type=int, default=150)
    sp.add_argument("--image", default=None, help="filter CSV by image name (exact match)")
    sp.add_argument("--stamp", action="store_true", help="overlay timestamp from filename on PNG map")
    sp.set_defaults(func=cmd_map_png)

    sp = sub.add_parser("map-png-all", help="Export PNG basemap for all images with shared extent")
    sp.add_argument("--geo-csv", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--provider", default="carto", help="carto|osm|esri-world|japan_gsi_seamless|japan_gsi_air or XYZ URL")
    sp.add_argument("--zoom", type=int, default=None)
    sp.add_argument("--point-size", type=float, default=12.0)
    sp.add_argument("--alpha", type=float, default=0.9)
    sp.add_argument("--point-color", default="#9A0EEA", help="marker color for points (matplotlib color, e.g., #RRGGBB)")
    sp.add_argument("--dpi", type=int, default=150)
    sp.add_argument("--margin", type=float, default=0.10, help="fractional padding for shared extent")
    sp.add_argument("--stamp", action="store_true", help="overlay timestamp from filename on PNG maps")
    sp.set_defaults(func=cmd_map_png_all)

    sp = sub.add_parser("compose-video", help="Compose two videos side-by-side or stacked")
    sp.add_argument("--left", required=True, help="Left (or top) video path, e.g., camera video with trajectories")
    sp.add_argument("--right", required=True, help="Right (or bottom) video path, e.g., map video with trajectories")
    sp.add_argument("--out", required=True, help="Output mp4 path for combined video")
    sp.add_argument("--layout", choices=["h","v"], default="h", help="Layout: 'h' (exclusive) or 'v' (stack)")
    sp.add_argument("--fps", type=float, default=None, help="Override output FPS (default: min of inputs)")
    sp.add_argument("--gap", type=int, default=8, help="Gap in pixels between panels")
    sp.set_defaults(func=cmd_compose_video)

    return p


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
