import os
from pathlib import Path
from datetime import datetime
import cv2
import pandas as pd
from typing import Optional, Tuple, Union, List

from .people_tracking import run_tracking
from .geolocate import geolocate_detections
from .perspective import geolocate_detections_perspective, solve_homography_from_csv, save_homography
from .calibration import solve_calibration, save_calibration
from .mapplot import save_points_basemap, save_tracking_map_video
from .video import compose_side_by_side_video

def run_pipeline(
    video_path: str,
    output_dir: str,
    # Model & Tracking
    model_name: str = "yolo12m.pt",
    conf_thres: float = 0.08,
    iou_thres: float = 0.45,
    img_size: int = 1920,
    max_det: int = 3000,
    person_class_id: int = 0,
    agnostic_nms: bool = True,
    device: Union[int, str] = 0,
    max_disappeared: int = 30,
    max_distance: float = 110.0,
    line_y_fraction: float = 0.55,
    center_crop: Optional[Tuple[int, int]] = (1920, 1080),
    show_traj: bool = True,
    traj_max_points: int = 200,
    traj_thickness: int = 2,
    enable_counting: bool = False,
    max_speed_px_per_frame: float = 32.0,
    min_iou_for_match: float = 0.10,
    # Calibration
    projection: str = "perspective", # "perspective" or "pano"
    calib_csv: Optional[str] = None,
    calib_npz: Optional[str] = None, # If provided, skips solving if run_calibration_step is False
    cam_lat: Optional[float] = None,
    cam_lon: Optional[float] = None,
    camera_alt_m: float = 20.0,
    ground_alt_m: float = 0.0,
    dem_spacing_m: float = 5.0,
    dem_margin_m: float = 120.0,
    dem_xml_folder: Optional[str] = None,
    # Geolocation
    gate_margin_m: float = 120.0,
    drop_outside: bool = True,
    smoothing_window: int = 15,
    max_velocity_m: float = 0.05,
    # Visualization
    google_maps_api_key: Optional[str] = None,
    map_zoom: Optional[int] = None,
    map_point_size: float = 20.0,
    map_alpha: float = 0.95,
    map_traj_max_frames: int = 200,
    map_dpi: int = 150,
    map_margin_frac: float = 0.10,
    # Steps to run
    run_tracking_step: bool = True,
    run_calibration_step: bool = True,
    run_geolocation_step: bool = True,
    run_visualization_step: bool = True,
    run_composition_step: bool = True,
) -> dict:
    """
    Runs the full Panogeo pipeline: Tracking -> Calibration -> Geolocation -> Visualization -> Composition.
    
    Returns a dictionary with paths to generated files.
    """
    
    video_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "output_dir": str(out_dir),
    }
    
    # --- 1. Tracking ---
    track_export_csv = out_dir / "tracks_points.csv"
    cam_mp4 = out_dir / "_module_full_traj.mp4"
    
    if run_tracking_step:
        print(f"--- Starting Tracking: {video_path} ---")
        run_tracking(
            video_path=str(video_path),
            output_path=cam_mp4,
            model=model_name,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            imgsz=img_size,
            max_det=max_det,
            person_class_id=person_class_id,
            agnostic_nms=agnostic_nms,
            device=device,
            half=True,
            amp=True,
            enable_counting=enable_counting,
            line_y_fraction=line_y_fraction,
            center_crop=center_crop,
            show_trajectories=show_traj,
            traj_max_points=traj_max_points,
            traj_thickness=traj_thickness,
            max_disappeared=max_disappeared,
            max_distance=max_distance,
            max_speed_px_per_frame=max_speed_px_per_frame,
            min_iou_for_match=min_iou_for_match,
            export_csv_path=track_export_csv,
            show_progress=True,
            progress_desc="Tracking",
        )
        results["tracks_csv"] = str(track_export_csv)
        results["cam_video"] = str(cam_mp4)
    else:
        if track_export_csv.exists():
            results["tracks_csv"] = str(track_export_csv)
        if cam_mp4.exists():
            results["cam_video"] = str(cam_mp4)

    # --- 2. Calibration ---
    if calib_npz is None:
        if projection == "perspective":
            calib_npz = out_dir / "calibration_perspective.npz"
        else:
            calib_npz = out_dir / "calibration_pano.npz"
    else:
        calib_npz = Path(calib_npz)

    if run_calibration_step:
        print(f"--- Starting Calibration ({projection}) ---")
        if not calib_csv:
            print("Warning: No calib_csv provided, skipping calibration step.")
        else:
            if projection == "perspective":
                H, ref_lat, ref_lon = solve_homography_from_csv(calib_csv)
                save_homography(
                    str(calib_npz),
                    H,
                    ref_lat=ref_lat,
                    ref_lon=ref_lon,
                    ground_alt_m=ground_alt_m,
                    calib_csv=calib_csv,
                    google_api_key=google_maps_api_key,
                    dem_spacing_m=dem_spacing_m,
                    dem_margin_m=dem_margin_m,
                    dem_xml_folder=dem_xml_folder,
                )
            else:
                # Pano
                # We need image dimensions. If tracking ran, we might know them, but better to get from video or args
                # For now, assume 1920x1080 or use img_size if square? 
                # The notebook uses IMG_W, IMG_H. Let's assume center_crop or img_size.
                w, h = 1920, 1080
                if center_crop:
                    w, h = center_crop
                
                res = solve_calibration(
                    calib_csv=calib_csv,
                    cam_lat=cam_lat,
                    cam_lon=cam_lon,
                    camera_alt_m=camera_alt_m,
                    ground_alt_m=ground_alt_m,
                    default_width=w,
                    default_height=h,
                    optimize_cam_position=False,
                )
                save_calibration(
                    npz_path=str(calib_npz),
                    calib=res,
                    cam_lat=res.cam_lat,
                    cam_lon=res.cam_lon,
                    camera_alt_m=res.camera_alt_m,
                    ground_alt_m=ground_alt_m,
                )
            print(f"Saved calibration to: {calib_npz}")
            results["calibration_npz"] = str(calib_npz)
    else:
        if calib_npz.exists():
            results["calibration_npz"] = str(calib_npz)

    # --- 3. Geolocation ---
    geo_csv = None
    if run_geolocation_step:
        print(f"--- Starting Geolocation ---")
        if "tracks_csv" not in results or "calibration_npz" not in results:
             print("Missing tracks_csv or calibration_npz, skipping geolocation.")
        else:
            try:
                if projection == "perspective":
                    xy_csv, geo_csv = geolocate_detections_perspective(
                        detections_csv=results["tracks_csv"],
                        homography_npz=results["calibration_npz"],
                        output_dir=str(out_dir),
                        debug=True,
                        calib_csv=calib_csv,
                        gate_margin_m=gate_margin_m,
                        drop_outside=drop_outside,
                        show_progress=True,
                        progress_desc="Geolocate (perspective)",
                        dem_xml_folder=dem_xml_folder,
                        dem_margin_m=dem_margin_m,
                        dem_spacing_m=dem_spacing_m,
                        smoothing_window=smoothing_window,
                        max_velocity_m=max_velocity_m,
                    )
                else:
                    xy_csv, geo_csv = geolocate_detections(
                        detections_csv=results["tracks_csv"],
                        calibration_npz=results["calibration_npz"],
                        output_dir=str(out_dir),
                        show_progress=True,
                        progress_desc="Geolocate (pano)",
                    )
                results["geo_csv"] = str(geo_csv)
                print(f"Geolocated CSV: {geo_csv}")
            except Exception as e:
                print(f"Geolocation failed: {e}")
    else:
        # Try to find existing geo csv
        # The naming convention in geolocate functions usually appends _geo.csv
        # But let's just check if we can find it based on tracks_csv name
        if "tracks_csv" in results:
             p = Path(results["tracks_csv"])
             candidate = p.parent / (p.stem + "_geo.csv") # This is a guess, geolocate functions might name differently
             # Actually geolocate_detections returns the path.
             # If we didn't run it, we might not know the exact name easily without duplicating logic.
             # But usually it is tracks_points_geo.csv
             candidate = out_dir / "tracks_points_geo.csv"
             if candidate.exists():
                 geo_csv = str(candidate)
                 results["geo_csv"] = geo_csv

    # --- 4. Visualization (Map Video) ---
    map_mp4 = out_dir / "people_tracking_map.mp4"
    if run_visualization_step:
        print(f"--- Starting Visualization ---")
        if not geo_csv:
            print("No geo_csv available, skipping visualization.")
        else:
            try:
                # Video
                cap0 = cv2.VideoCapture(str(video_path))
                video_fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
                cap0.release()
                
                out_mp4 = save_tracking_map_video(
                    geo_csv=str(geo_csv),
                    out_mp4=str(map_mp4),
                    provider="google",
                    zoom=map_zoom,
                    api_key=google_maps_api_key,
                    point_size=map_point_size,
                    alpha=map_alpha,
                    traj_max_frames=map_traj_max_frames,
                    dpi=map_dpi,
                    margin_frac=map_margin_frac,
                    fps=video_fps,
                    show_progress=True,
                    progress_desc="Map video",
                )
                results["map_video"] = str(out_mp4)
                
                # PNG
                map_png = out_dir / "people_tracking_map_carto.png"
                out_png = save_points_basemap(
                    geo_csv=str(geo_csv),
                    out_png=str(map_png),
                    provider="google",
                    zoom=map_zoom,
                    api_key=google_maps_api_key,
                    point_size=12.0,
                    alpha=0.9,
                    point_color="#FF5722",
                    dpi=map_dpi,
                )
                results["map_png"] = str(out_png)
            except Exception as e:
                print(f"Visualization failed: {e}")
    else:
        if map_mp4.exists():
            results["map_video"] = str(map_mp4)

    # --- 5. Composition ---
    if run_composition_step:
        print(f"--- Starting Composition ---")
        if "cam_video" in results and "map_video" in results:
            out_video = out_dir / "people_tracking_split.mp4"
            try:
                cap = cv2.VideoCapture(results["cam_video"])
                compose_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()

                out_path = compose_side_by_side_video(
                    results["cam_video"], 
                    results["map_video"], 
                    str(out_video), 
                    layout="h", 
                    fps=compose_fps, 
                    gap=8
                )
                results["combined_video"] = str(out_path)
                print(f"Saved combined video: {out_path}")
            except Exception as e:
                print(f"Composition failed: {e}")
        else:
            print("Missing cam_video or map_video, skipping composition.")

    return results
