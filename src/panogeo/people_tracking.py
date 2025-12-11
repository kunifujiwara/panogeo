from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd

from .utils import id_color_bgr

# Public API
__all__ = [
    "Track",
    "CentroidTracker",
    "load_model",
    "run_tracking",
    "smooth_tracks_csv",
    "smooth_geotracks_csv",
]

# History length for occlusion detection (number of frames)
_EDGE_HISTORY_LEN = 15


@dataclass
class Track:
    object_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    disappeared: int = 0
    # Edge history for occlusion detection
    top_history: Deque[float] = field(default_factory=lambda: deque(maxlen=_EDGE_HISTORY_LEN))
    bottom_history: Deque[float] = field(default_factory=lambda: deque(maxlen=_EDGE_HISTORY_LEN))
    height_history: Deque[float] = field(default_factory=lambda: deque(maxlen=_EDGE_HISTORY_LEN))

    def update_bbox(self, bbox: Tuple[int, int, int, int]) -> None:
        """Update bbox and record edge history for occlusion detection."""
        x1, y1, x2, y2 = bbox
        self.bbox = bbox
        self.top_history.append(float(y1))
        self.bottom_history.append(float(y2))
        self.height_history.append(float(y2 - y1))

    @property
    def expected_height(self) -> float:
        """Expected box height based on recent history (median)."""
        if len(self.height_history) >= 5:
            return float(np.median(list(self.height_history)))
        return float(self.bbox[3] - self.bbox[1])

    def is_bottom_occluded(
        self,
        bottom_jump_threshold_px: float = 25.0,
        height_shrink_ratio: float = 0.75,
    ) -> bool:
        """
        Detect if bottom edge is likely occluded.
        
        Occlusion signature: bottom edge jumps up suddenly while top edge 
        stays relatively stable, OR current height is significantly smaller
        than expected height from history.
        
        Args:
            bottom_jump_threshold_px: Min upward jump in bottom edge to trigger
            height_shrink_ratio: If current_height < expected * ratio, flag as occluded
            
        Returns:
            True if bottom appears to be occluded
        """
        if len(self.bottom_history) < 3:
            return False
        
        x1, y1, x2, y2 = self.bbox
        current_height = float(y2 - y1)
        
        # Method 1: Height significantly smaller than expected
        if current_height < self.expected_height * height_shrink_ratio:
            return True
        
        # Method 2: Bottom edge jumped up while top stayed stable
        # (positive delta = jumped up, since y increases downward)
        bottom_list = list(self.bottom_history)
        top_list = list(self.top_history)
        
        prev_bottom = bottom_list[-2]
        curr_bottom = bottom_list[-1]
        bottom_delta = prev_bottom - curr_bottom  # positive = jumped up
        
        prev_top = top_list[-2]
        curr_top = top_list[-1]
        top_delta = abs(curr_top - prev_top)
        
        # Occlusion: bottom jumps up significantly, top relatively stable
        if bottom_delta > bottom_jump_threshold_px and top_delta < bottom_jump_threshold_px * 0.5:
            return True
        
        return False

    def get_corrected_bottom(self) -> float:
        """
        Get corrected bottom edge position when occlusion is detected.
        
        Uses the expected height from history to estimate where the 
        bottom edge should be based on the current top edge.
        
        Returns:
            Corrected y2 (bottom edge) coordinate
        """
        x1, y1, x2, y2 = self.bbox
        expected_h = self.expected_height
        corrected_y2 = y1 + expected_h
        return corrected_y2


class CentroidTracker:
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 110.0,
        *,
        max_speed_px_per_frame: Optional[float] = None,
        min_iou_for_match: float = 0.0,
    ):
        self.next_object_id: int = 0
        self.tracks: Dict[int, Track] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        # Optional, tighter gating to prevent unrealistic jumps and id switches
        # If set, a detection will only be matched to a track if the centroid
        # move does not exceed (max_speed_px_per_frame * frames_since_update).
        self.max_speed_px_per_frame = max_speed_px_per_frame
        # Optional IoU gating; when > 0, require at least this IoU between
        # previous track bbox and candidate detection.
        self.min_iou_for_match = float(min_iou_for_match)

    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> None:
        track = Track(
            object_id=self.next_object_id,
            centroid=centroid,
            bbox=bbox,
            disappeared=0,
        )
        # Initialize edge history with current bbox
        track.update_bbox(bbox)
        self.tracks[self.next_object_id] = track
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        if object_id in self.tracks:
            del self.tracks[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Track]:
        if len(rects) == 0:
            for object_id in list(self.tracks.keys()):
                self.tracks[object_id].disappeared += 1
                if self.tracks[object_id].disappeared > self.max_disappeared:
                    self.deregister(object_id)
            return dict(self.tracks)

        input_centroids = np.zeros((len(rects), 2), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) * 0.5)
            cY = int((y1 + y2) * 0.5)
            input_centroids[i] = (cX, cY)

        if len(self.tracks) == 0:
            for i, box in enumerate(rects):
                self.register((int(input_centroids[i][0]), int(input_centroids[i][1])), box)
            return dict(self.tracks)

        object_ids = list(self.tracks.keys())
        object_centroids = np.array([self.tracks[oid].centroid for oid in object_ids], dtype=np.float32)

        D = np.linalg.norm(object_centroids[:, None, :] - input_centroids[None, :, :], axis=2)

        # Optional IoU matrix for gating
        IoU = None
        if self.min_iou_for_match > 0.0:
            IoU = np.zeros_like(D, dtype=np.float32)
            track_bboxes = [self.tracks[oid].bbox for oid in object_ids]
            for r, tb in enumerate(track_bboxes):
                x1t, y1t, x2t, y2t = tb
                at = float(max(0, x2t - x1t)) * float(max(0, y2t - y1t))
                for c, db in enumerate(rects):
                    x1d, y1d, x2d, y2d = db
                    inter_x1 = max(x1t, x1d)
                    inter_y1 = max(y1t, y1d)
                    inter_x2 = min(x2t, x2d)
                    inter_y2 = min(y2t, y2d)
                    iw = max(0, inter_x2 - inter_x1)
                    ih = max(0, inter_y2 - inter_y1)
                    inter = float(iw * ih)
                    ad = float(max(0, x2d - x1d)) * float(max(0, y2d - y1d))
                    union = at + ad - inter if (at + ad - inter) > 0 else 0.0
                    IoU[r, c] = inter / union if union > 0 else 0.0

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            # Speed gating: forbid unrealistic jumps
            if self.max_speed_px_per_frame is not None:
                frames_since_update = self.tracks[object_ids[row]].disappeared + 1
                max_allow = self.max_speed_px_per_frame * max(1, frames_since_update)
                if D[row, col] > max_allow:
                    continue
            # IoU gating: require overlap if requested
            if IoU is not None and IoU[row, col] < self.min_iou_for_match:
                continue

            object_id = object_ids[row]
            centroid = (int(input_centroids[col][0]), int(input_centroids[col][1]))
            self.tracks[object_id].centroid = centroid
            self.tracks[object_id].update_bbox(rects[col])  # Updates bbox and edge history
            self.tracks[object_id].disappeared = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        for row in unused_rows:
            object_id = object_ids[row]
            self.tracks[object_id].disappeared += 1
            if self.tracks[object_id].disappeared > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            centroid = (int(input_centroids[col][0]), int(input_centroids[col][1]))
            self.register(centroid, rects[col])

        return dict(self.tracks)


def load_model(model: Union[str, Path, Sequence[Union[str, Path]]]):
    from ultralytics import YOLO

    last_error: Optional[Exception] = None
    # Single name/path: try directly (Ultralytics will download if needed)
    if isinstance(model, (str, Path)):
        return YOLO(str(model))
    # Sequence of candidates: try in order
    for candidate in model:
        try:
            return YOLO(str(candidate))
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Could not load any model from {model}. Last error: {last_error}")


def _det_to_boxes(result, frame_w: int, frame_h: int, person_class_id: int, conf_thres: float) -> List[Tuple[int, int, int, int]]:
    boxes_xyxy: List[Tuple[int, int, int, int]] = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return boxes_xyxy
    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        if k == person_class_id and c >= conf_thres:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes_xyxy.append((x1, y1, x2, y2))
    return boxes_xyxy


def run_tracking(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]],
    model: Union[str, Path, Sequence[Union[str, Path]], None] = None,
    model_candidates: Optional[Sequence[Union[str, Path]]] = None,
    *,
    conf_thres: float = 0.08,
    iou_thres: float = 0.45,
    imgsz: int = 1920,
    max_det: int = 3000,
    person_class_id: int = 0,
    agnostic_nms: bool = True,
    device: Union[int, str] = 0,
    half: bool = True,
    amp: bool = True,
    enable_counting: bool = False,
    line_y_fraction: float = 0.55,
    center_crop: Optional[Tuple[int, int]] = None,  # (crop_w, crop_h) from center
    show_trajectories: bool = False,
    traj_max_points: int = 40,
    traj_thickness: int = 2,
    max_disappeared: int = 30,
    max_distance: float = 110.0,
    max_speed_px_per_frame: Optional[float] = None,
    min_iou_for_match: float = 0.0,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
    export_csv_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Process a video to detect and track people, optionally center-cropping and drawing trajectories.
    Returns the output video path.
    - Pass a single model name/path via `model` (e.g., "yolo12m.pt"); Ultralytics will download if needed.
    - For backward compatibility, `model_candidates` can still be provided to try multiple in order.
    """
    if model is None and model_candidates is None:
        raise ValueError("Specify `model` (preferred) or `model_candidates`.")
    model_obj = load_model(model if model is not None else model_candidates)  # type: ignore[arg-type]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if center_crop is not None:
        crop_w = min(center_crop[0], src_w)
        crop_h = min(center_crop[1], src_h)
        x0 = max(0, (src_w - crop_w) // 2)
        y0 = max(0, (src_h - crop_h) // 2)
        x1 = x0 + crop_w
        y1 = y0 + crop_h
        out_w, out_h = crop_w, crop_h
    else:
        x0 = 0
        y0 = 0
        x1 = src_w
        y1 = src_h
        out_w, out_h = src_w, src_h

    # Output path
    if output_path is None:
        out_dir = Path(video_path).parent
        output_path = out_dir / "_tracked.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    tracker = CentroidTracker(
        max_disappeared=max_disappeared,
        max_distance=max_distance,
        max_speed_px_per_frame=max_speed_px_per_frame,
        min_iou_for_match=min_iou_for_match,
    )
    prev_centroids: Dict[int, Tuple[int, int]] = {}

    tracks_history: Dict[int, Deque[Tuple[int, int]]] = {}
    if show_trajectories:
        tracks_history = {}
    # Optional CSV export of per-frame track points for geolocation
    csv_rows: List[Dict[str, Union[str, int, float]]] = []
    video_base = Path(video_path).stem

    predict_kwargs = dict(
        conf=conf_thres,
        iou=iou_thres,
        imgsz=imgsz,
        classes=[person_class_id],
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        augment=True,
        verbose=False,
        device=device,
        half=half,
        amp=amp,
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    # Progress bar (tqdm) setup
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            if progress_desc is None:
                if center_crop is not None and show_trajectories:
                    progress_desc = "Center-crop w/ traj"
                elif center_crop is not None:
                    progress_desc = "Center-crop"
                elif show_trajectories:
                    progress_desc = "Full-frame w/ traj"
                else:
                    progress_desc = "Full-frame"
            pbar = tqdm(total=total_frames, desc=progress_desc, unit="frame")
        except Exception:
            pbar = None

    line_y = int(line_y_fraction * out_h)
    start_time = time.time()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        if progress_callback is not None:
            progress_callback(frame_count, total_frames)
        if pbar is not None:
            pbar.update(1)

        roi = frame[y0:y1, x0:x1]
        results = model_obj.predict(roi, **predict_kwargs)
        result = results[0] if len(results) > 0 else None
        boxes_xyxy = _det_to_boxes(result, out_w, out_h, person_class_id, conf_thres)

        tracks = tracker.update(boxes_xyxy)

        if enable_counting:
            for oid, tr in tracks.items():
                cX, cY = tr.centroid
                if oid in prev_centroids:
                    prevY = prev_centroids[oid][1]
                    if prevY < line_y <= cY:
                        # entering
                        pass
                    elif prevY > line_y >= cY:
                        # exiting
                        pass
                prev_centroids[oid] = (cX, cY)

        if show_trajectories:
            for oid, tr in tracks.items():
                cX, cY = tr.centroid
                if oid not in tracks_history:
                    tracks_history[oid] = deque(maxlen=traj_max_points)
                tracks_history[oid].append((cX, cY))

        # Drawing
        if enable_counting:
            cv2.line(roi, (0, line_y), (out_w, line_y), (0, 255, 255), 2)

        for oid, tr in tracks.items():
            x1b, y1b, x2b, y2b = tr.bbox
            cX, cY = tr.centroid
            color = id_color_bgr(oid)
            cv2.rectangle(roi, (x1b, y1b), (x2b, y2b), color, 2)
            cv2.circle(roi, (cX, cY), 3, (0, 255, 255), -1)
            cv2.putText(roi, f"ID {oid}", (x1b, max(0, y1b - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if show_trajectories:
                pts = tracks_history.get(oid, None)
                if pts and len(pts) > 1:
                    for i in range(1, len(pts)):
                        cv2.line(roi, pts[i - 1], pts[i], color, traj_thickness)

        elapsed = time.time() - start_time
        if elapsed > 0:
            fps_text = f"FPS: {frame_count / elapsed:.1f}"
            cv2.putText(roi, fps_text, (out_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)
            if total_frames:
                prog_text = f"{frame_count}/{total_frames}"
                cv2.putText(roi, prog_text, (out_w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 50), 2)

        writer.write(roi)
        # Collect rows for export: use bottom-center of bbox (feet) as v_px
        # With occlusion detection: if bottom appears occluded, provide corrected v_px
        if export_csv_path is not None:
            for oid, tr in tracks.items():
                x1b, y1b, x2b, y2b = tr.bbox
                u = float((x1b + x2b) * 0.5)
                v_bbox = float(y2b)  # Raw bottom edge
                
                # Occlusion detection
                is_occluded = tr.is_bottom_occluded()
                v_corrected = tr.get_corrected_bottom() if is_occluded else v_bbox
                
                csv_rows.append({
                    "video": str(video_path),
                    "image": f"{video_base}_f{frame_count:06d}",
                    "frame": int(frame_count),
                    "track_id": int(oid),
                    "W": int(out_w),
                    "H": int(out_h),
                    # If center-cropping was used, record the pixel offsets so downstream
                    # geolocation (homography defined in full-frame coordinates) can adjust.
                    # When no crop is used, these evaluate to zero.
                    "x_offset_px": int(x0),
                    "y_offset_px": int(y0),
                    # Optional provenance/debug info
                    "SRC_W": int(src_w),
                    "SRC_H": int(src_h),
                    # Use corrected v_px by default (accounts for occlusion)
                    "u_px": u,
                    "v_px": v_corrected,
                    # Occlusion detection outputs
                    "v_px_raw": v_bbox,
                    "bottom_occluded": int(is_occluded),
                    "box_height_px": int(y2b - y1b),
                })

    cap.release()
    writer.release()
    if pbar is not None:
        pbar.close()
    # Persist CSV rows if requested
    if export_csv_path is not None:
        try:
            out_csv_path = Path(export_csv_path)
            out_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(csv_rows).to_csv(out_csv_path, index=False)
        except Exception:
            pass
    return output_path


def _id_color(oid: int) -> Tuple[int, int, int]:
    # Backwards-compat shim: delegate to shared utils
    return id_color_bgr(oid)


def _slew_rate_limit(values: np.ndarray, max_change: float) -> np.ndarray:
    """
    Limit the rate of change of a signal (slew rate limiter).
    
    Args:
        values: Input array of values.
        max_change: Maximum allowed change between consecutive elements.
        
    Returns:
        Array with limited rate of change.
    """
    if len(values) == 0:
        return values
    
    out = np.empty_like(values)
    out[0] = values[0]
    
    # Numba would be faster, but simple loop is fine for track lengths (~100s)
    for i in range(1, len(values)):
        prev = out[i-1]
        curr = values[i]
        diff = curr - prev
        
        if diff > max_change:
            out[i] = prev + max_change
        elif diff < -max_change:
            out[i] = prev - max_change
        else:
            out[i] = curr
            
    return out


def smooth_tracks_csv(
    csv_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    window_size: int = 7,
    max_velocity_px: Optional[float] = None,
) -> Path:
    """
    Apply post-processing smoothing to track CSV to correct bounding box fluctuations.
    
    This is specifically designed for perspective mode where bounding box fluctuations
    (e.g. alternating between full-body and upper-body) cause zigzag trajectories.
    Since the "correct" trajectory is usually on the camera side (larger v_px/y-coordinate),
    this applies a filter that favors larger v_px values (lower in image).
    
    Args:
        csv_path: Path to the input CSV file generated by run_tracking
        output_path: Path to save the smoothed CSV. If None, appends '_smoothed' to filename.
        window_size: Window size for the rolling filter (odd number recommended).
        max_velocity_px: Optional limit for pixel displacement per frame (slew rate limit).
                         If set, restricts sudden leaps in both u and v directions.
        
    Returns:
        Path to the output CSV file.
    """
    df = pd.read_csv(csv_path)
    
    if output_path is None:
        p = Path(csv_path)
        output_path = p.parent / f"{p.stem}_smoothed{p.suffix}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process each track
    grouped = df.groupby("track_id")
    smoothed_dfs = []
    
    for track_id, group in grouped:
        # Sort by frame to ensure temporal order
        group = group.sort_values("frame").copy()
        
        u_series = group["u_px"]
        v_series = group["v_px"]

        # 0. Optional: Apply slew rate limit (velocity clamp) BEFORE smoothing
        # This removes single-frame spikes that would otherwise be smeared by the rolling window.
        if max_velocity_px is not None and max_velocity_px > 0:
            u_vals = u_series.to_numpy()
            v_vals = v_series.to_numpy()
            u_series = pd.Series(_slew_rate_limit(u_vals, max_velocity_px), index=group.index)
            v_series = pd.Series(_slew_rate_limit(v_vals, max_velocity_px), index=group.index)

        # 1. Smooth u_px (horizontal) - simple median filter to remove jitter
        u_smoothed = u_series.rolling(window=window_size, min_periods=1, center=True).median()
        
        # 2. Smooth v_px (vertical/distance) - favor larger values (camera side)
        # Use quantile(0.9) instead of max() to be robust against occasional outliers towards the camera,
        # while still strongly favoring the "feet" position (larger Y).
        v_max = v_series.rolling(window=window_size, min_periods=1, center=True).quantile(0.9)
        
        # The max/quantile filter can introduce "steps". Apply a mean filter to round edges.
        v_smoothed = v_max.rolling(window=window_size, min_periods=1, center=True).mean()
        
        group["u_px"] = u_smoothed
        group["v_px"] = v_smoothed
        
        smoothed_dfs.append(group)
        
    if smoothed_dfs:
        result_df = pd.concat(smoothed_dfs).sort_values(["frame", "track_id"])
        result_df.to_csv(output_path, index=False)
    else:
        # Handle empty case
        df.to_csv(output_path, index=False)
    
    return output_path


def smooth_geotracks_csv(
    geo_csv_path: Union[str, Path],
    calibration_npz: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    window_size: int = 7,
    max_velocity_m: Optional[float] = None,
) -> Path:
    """
    Apply post-processing smoothing to GEOLOCATED track CSV (lat/lon).
    
    This method works in the geographic domain (ENU coordinates relative to camera).
    It addresses the "zigzag" issue by smoothing the distance from the camera,
    favoring points closer to the camera (which correspond to the true "feet" position
    when the bounding box fluctuates upwards).
    
    Args:
        geo_csv_path: Path to the geolocated CSV file (must contain lat, lon).
        calibration_npz: Path to the calibration file (to get camera position).
        output_path: Path to save the smoothed CSV.
        window_size: Window size for the rolling filter.
        max_velocity_m: Optional limit for motion in meters per frame (slew rate limit).
        
    Returns:
        Path to the output CSV file.
    """
    # Lazy import to avoid circular deps if possible, or just standard imports
    from .geodesy import build_context, enu_to_llh_batch, llh_to_ecef
    
    df = pd.read_csv(geo_csv_path)
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("CSV must contain 'lat' and 'lon' columns.")
        
    # Load calibration to get camera position
    calib_data = np.load(str(calibration_npz))
    
    # Helper to get key case-insensitively
    def _get_key(target: str) -> Optional[str]:
        for k in calib_data.files:
            if k.lower() == target.lower():
                return k
        return None

    k_lat = _get_key("ref_lat") or _get_key("cam_lat")
    k_lon = _get_key("ref_lon") or _get_key("cam_lon")
    
    if k_lat and k_lon:
        ref_lat = float(calib_data[k_lat])
        ref_lon = float(calib_data[k_lon])
        
        k_alt = _get_key("ref_alt") or _get_key("ground_alt_m")
        ref_alt = float(calib_data[k_alt]) if k_alt else 0.0
    else:
        raise ValueError(f"Calibration file missing reference coordinates. Keys found: {calib_data.files}")
        
    # Build context for ENU conversion
    # We treat the camera/ref as the origin for ENU conversion
    ctx = build_context(ref_lat, ref_lon, 0.0, ref_alt)
    
    # Convert all points to ENU
    lons = df["lon"].values
    lats = df["lat"].values
    alts = np.zeros_like(lons) # Assume ground level (relative to ref_alt)
    
    # Vectorized LLH -> ENU
    # We can reuse _llh_to_enu_batch logic or just use the context
    # Let's implement a quick batch converter here using the context
    ecef_from_llh = ctx.ecef_from_llh
    ecef_ref = ctx.ecef_ref
    R_ecef2enu = ctx.R_ecef2enu
    
    n = len(lons)
    enu_points = np.empty((n, 3), dtype=np.float64)
    
    # Batch transform to ECEF
    # pyproj transformer can handle arrays
    x, y, z = ecef_from_llh.transform(lons, lats, alts)
    ecef_pts = np.stack([x, y, z], axis=1)
    
    # ECEF to ENU
    # (N, 3) = ((3, 3) @ (N, 3).T).T
    enu_points = (R_ecef2enu @ (ecef_pts - ecef_ref).T).T
    
    # Add ENU to dataframe for processing
    df["E"] = enu_points[:, 0]
    df["N"] = enu_points[:, 1]
    df["U"] = enu_points[:, 2]
    
    # Calculate Range (distance from origin/camera) and Bearing
    # Note: If ref is not exactly camera, this is distance from ref. 
    # For perspective monoplotting, ref is usually the camera location or close to it.
    # If cam_pos_enu is in calib, we should subtract it to get range from camera.
    k_cam_pos = _get_key("cam_pos_enu")
    if k_cam_pos:
        cam_pos = calib_data[k_cam_pos]
        # Adjust ENU to be relative to camera
        rel_E = df["E"] - cam_pos[0]
        rel_N = df["N"] - cam_pos[1]
        # We ignore Z/U difference for ground distance, or include it for slant range
        # Let's use horizontal distance for map smoothing
        df["dist"] = np.sqrt(rel_E**2 + rel_N**2)
        df["bearing"] = np.arctan2(rel_E, rel_N) # rad
    else:
        # Fallback: assume ENU origin is camera
        df["dist"] = np.sqrt(df["E"]**2 + df["N"]**2)
        df["bearing"] = np.arctan2(df["E"], df["N"])

    if output_path is None:
        p = Path(geo_csv_path)
        output_path = p.parent / f"{p.stem}_smoothed_geo{p.suffix}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process each track
    grouped = df.groupby("track_id")
    smoothed_dfs = []
    
    for track_id, group in grouped:
        group = group.sort_values("frame").copy()
        
        dist_series = group["dist"]
        bear_series = group["bearing"] # radians
        
        # 0. Slew rate limit (optional)
        if max_velocity_m is not None and max_velocity_m > 0:
            d_vals = dist_series.to_numpy()
            # For bearing, it's tricky due to wrap-around, but usually people don't orbit fast.
            # We'll just limit distance jumps for now as that's the main zigzag source.
            dist_series = pd.Series(_slew_rate_limit(d_vals, max_velocity_m), index=group.index)
            
        # 1. Smooth Distance (favor smaller values = closer to camera)
        # Use min() to aggressively remove "far" outliers (spikes away from camera).
        # This acts as a morphological erosion on the distance signal.
        dist_min = dist_series.rolling(window=window_size, min_periods=1, center=True).min()
        
        # Smooth the result to remove the "stepped" look from the min filter
        dist_smoothed = dist_min.rolling(window=window_size, min_periods=1, center=True).mean()
        
        # 2. Smooth Bearing (standard median/mean)
        # Unwrapping is needed for correct angular smoothing if crossing +/- pi
        bear_unwrapped = np.unwrap(bear_series.values)
        bear_series_u = pd.Series(bear_unwrapped, index=group.index)
        bear_smoothed_u = bear_series_u.rolling(window=window_size, min_periods=1, center=True).median()
        bear_smoothed = (bear_smoothed_u + np.pi) % (2 * np.pi) - np.pi
        
        # Reconstruct ENU from smoothed dist/bearing
        # We need to add back camera position if we subtracted it
        if k_cam_pos:
            cam_pos = calib_data[k_cam_pos]
            new_E = cam_pos[0] + dist_smoothed * np.sin(bear_smoothed)
            new_N = cam_pos[1] + dist_smoothed * np.cos(bear_smoothed)
        else:
            new_E = dist_smoothed * np.sin(bear_smoothed)
            new_N = dist_smoothed * np.cos(bear_smoothed)
            
        # Update group
        # We keep U as original (ground height)
        group["E"] = new_E
        group["N"] = new_N
        
        smoothed_dfs.append(group)
        
    if smoothed_dfs:
        result_df = pd.concat(smoothed_dfs).sort_values(["frame", "track_id"])
        
        # Convert back to Lat/Lon
        # Vectorized ENU -> LLH
        enu_final = result_df[["E", "N", "U"]].values
        llh_final = enu_to_llh_batch(ctx, enu_final)
        
        result_df["lon"] = llh_final[:, 0]
        result_df["lat"] = llh_final[:, 1]
        # height is usually ignored or kept as 0
        
        # Cleanup temp columns
        result_df = result_df.drop(columns=["E", "N", "U", "dist", "bearing"], errors="ignore")
        
        result_df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
        
    return output_path


