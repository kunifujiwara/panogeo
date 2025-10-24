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
            out.save(out_path, **save_kwargs)
    print(f"Saved {len(files)} image(s) to {args.out_dir}")


def cmd_detect(args: argparse.Namespace) -> None:
    # Lazy import to avoid loading heavy dependencies (cv2/ultralytics) unless needed
    from .detection import detect_folder, DetectConfig, TilingConfig

    dcfg = DetectConfig(model_name=args.model, conf_thres=args.conf, iou_thres=args.iou)
    tcfg = TilingConfig(
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        overlap=args.overlap,
        imgsz=args.imgsz,
        min_box_h_px=args.min_box_h,
    )
    agg_csv = detect_folder(args.images_dir, args.output_dir, dcfg=dcfg, tcfg=tcfg)
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
    )
    os.makedirs(args.output_dir, exist_ok=True)
    out_npz = os.path.join(args.output_dir, "calibration_cam2enu.npz")
    save_calibration(out_npz, res, args.cam_lat, args.cam_lon, args.camera_alt_m, args.ground_alt_m)
    print(f"Saved calibration to: {out_npz}")
    print(f"yaw={res.yaw_deg:.2f}°, pitch={res.pitch_deg:.2f}°, roll={res.roll_deg:.2f}°")


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="panogeo", description="Panoramic detection and geolocation")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("shift-pano", help="Shift equirect images by yaw")
    sp.add_argument("--in-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--degrees", type=float, default=180.0)
    sp.set_defaults(func=cmd_shift)

    sp = sub.add_parser("detect", help="Run YOLO detection on panoramas with tiling")
    sp.add_argument("--images-dir", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--model", default="yolov8s.pt")
    sp.add_argument("--conf", type=float, default=0.20)
    sp.add_argument("--iou", type=float, default=0.50)
    sp.add_argument("--tile-w", type=int, default=1280)
    sp.add_argument("--tile-h", type=int, default=960)
    sp.add_argument("--overlap", type=float, default=0.30)
    sp.add_argument("--imgsz", type=int, default=1280)
    sp.add_argument("--min-box-h", type=int, default=40)
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("calibrate", help="Estimate camera rotation from pixel↔geo pairs")
    sp.add_argument("--calib-csv", required=True)
    sp.add_argument("--cam-lat", type=float, required=True)
    sp.add_argument("--cam-lon", type=float, required=True)
    sp.add_argument("--camera-alt-m", type=float, default=2.0)
    sp.add_argument("--ground-alt-m", type=float, default=0.0)
    sp.add_argument("--width", type=int, default=2048, help="pano width if W not in CSV")
    sp.add_argument("--height", type=int, default=1024, help="pano height if H not in CSV")
    sp.add_argument("--output-dir", required=True)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("geolocate", help="Intersect detection rays with ground/DEM using calibration")
    sp.add_argument("--detections-csv", required=True)
    sp.add_argument("--calibration", required=True, help="calibration_cam2enu.npz")
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--dem", default=None)
    sp.add_argument("--max-range", type=float, default=80.0)
    sp.add_argument("--step", type=float, default=1.0)
    sp.set_defaults(func=cmd_geolocate)

    sp = sub.add_parser("map", help="Build a Folium heatmap from geo CSV")
    sp.add_argument("--geo-csv", required=True)
    sp.add_argument("--out-html", required=True)
    sp.add_argument("--center-lat", type=float, default=None)
    sp.add_argument("--center-lon", type=float, default=None)
    sp.add_argument("--zoom", type=int, default=19)
    sp.set_defaults(func=cmd_map)

    return p


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
