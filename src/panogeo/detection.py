from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from .utils import extract_timestamp_text


@dataclass
class TilingConfig:
    tile_w: int = 1280
    tile_h: int = 960
    overlap: float = 0.30
    imgsz: int = 1280
    min_box_h_px: int = 40


@dataclass
class DetectConfig:
    model_name: str = "yolov8s.pt"
    conf_thres: float = 0.20
    iou_thres: float = 0.50
    classes: Tuple[int, ...] = (0,)  # person
    # Merge boxes that are largely contained within others after cross-tile merge.
    # If None, disabled. If set (e.g., 0.98), uses IoA (intersection over smaller area).
    containment_thr: Optional[float] = None
    # Performance knobs (do not change results):
    # - device: 'cpu', 'cuda:0', etc. None -> ultralytics auto
    device: Optional[str] = None
    # - batch_tiles: number of tiles to run per forward pass
    batch_tiles: int = 8
    # - fuse model conv+bn for faster inference (numerically equivalent)
    fuse_model: bool = True
    # - use half precision on CUDA (should not change boxes materially, but keep off by default)
    half: bool = False


def _resolve_device(requested: Optional[str]) -> Optional[str]:
    """Normalize requested device into one accepted by Ultralytics; fallback to CPU when unavailable.

    Returns:
        - None to let Ultralytics auto-select
        - 'cpu', 'cuda:0', 'mps', or original string if valid
    """
    if requested is None:
        return None
    dev = requested.strip().lower()
    if dev == "" or dev == "auto":
        return None
    if dev == "cpu":
        return "cpu"
    if dev.startswith("cuda") or dev in {"0", "1", "2", "3"} or dev.startswith("gpu"):
        if not torch.cuda.is_available():
            return "cpu"
        return requested
    if dev == "mps":
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return requested


def nms_merge(boxes: List[List[float]], scores: List[float], iou_thr: float = 0.55) -> List[int]:
    if not boxes:
        return []
    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    idxs = s.argsort()[::-1]
    keep: List[int] = []
    while idxs.size:
        i = idxs[0]
        keep.append(int(i))
        if idxs.size == 1:
            break
        xx1 = np.maximum(b[i, 0], b[idxs[1:], 0])
        yy1 = np.maximum(b[i, 1], b[idxs[1:], 1])
        xx2 = np.minimum(b[i, 2], b[idxs[1:], 2])
        yy2 = np.minimum(b[i, 3], b[idxs[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        area_j = (b[idxs[1:], 2] - b[idxs[1:], 0]) * (b[idxs[1:], 3] - b[idxs[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-9)
        idxs = idxs[1:][iou <= iou_thr]
    return keep


def merge_contained_boxes(
    boxes: List[List[float]],
    scores: List[float],
    ioa_thr: float = 0.98,
) -> Tuple[List[List[float]], List[float]]:
    """Merge boxes that are nearly contained within others using IoA.

    Forms clusters where any pair has IoA (intersection over smaller area) >= ioa_thr
    with the current highest-score seed, and replaces the cluster with a single
    union bounding box. The merged score is the max of member scores.
    """
    if not boxes:
        return [], []
    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    idxs = s.argsort()[::-1]
    merged_boxes: List[List[float]] = []
    merged_scores: List[float] = []
    while idxs.size:
        i = idxs[0]
        if idxs.size == 1:
            merged_boxes.append([float(b[i, 0]), float(b[i, 1]), float(b[i, 2]), float(b[i, 3])])
            merged_scores.append(float(s[i]))
            break
        xx1 = np.maximum(b[i, 0], b[idxs[1:], 0])
        yy1 = np.maximum(b[i, 1], b[idxs[1:], 1])
        xx2 = np.minimum(b[i, 2], b[idxs[1:], 2])
        yy2 = np.minimum(b[i, 3], b[idxs[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        area_j = (b[idxs[1:], 2] - b[idxs[1:], 0]) * (b[idxs[1:], 3] - b[idxs[1:], 1])
        min_area = np.minimum(area_i, area_j)
        ioa = inter / (min_area + 1e-9)

        # Members to merge with i (including i itself)
        to_merge_mask = ioa >= ioa_thr
        members = [int(i)]
        if np.any(to_merge_mask):
            members.extend([int(j) for j in idxs[1:][to_merge_mask]])

        x1 = float(np.min(b[members, 0]))
        y1 = float(np.min(b[members, 1]))
        x2 = float(np.max(b[members, 2]))
        y2 = float(np.max(b[members, 3]))
        merged_boxes.append([x1, y1, x2, y2])
        merged_scores.append(float(np.max(s[members])))

        # Remove merged members from idxs
        if len(members) == 1:
            idxs = idxs[1:]
        else:
            member_set = set(members)
            idxs = np.array([idx for idx in idxs if int(idx) not in member_set], dtype=idxs.dtype)
    return merged_boxes, merged_scores


def detect_folder(
    images_dir: str,
    output_dir: str,
    dcfg: DetectConfig = DetectConfig(),
    tcfg: TilingConfig = TilingConfig(),
    annotate: bool = False,
    annotate_dir: Optional[str] = None,
    stamp_timestamp: bool = False,
    bbox_color_bgr: Tuple[int, int, int] = (40, 220, 40),
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    det_dir = os.path.join(output_dir, "detections")
    os.makedirs(det_dir, exist_ok=True)
    ann_dir = None
    if annotate:
        ann_dir = annotate_dir if annotate_dir else os.path.join(output_dir, "annotated")
        os.makedirs(ann_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG")
    img_paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir)) if f.endswith(exts)]

    model = YOLO(dcfg.model_name)
    # Optionally place model on device and fuse layers
    resolved_device = _resolve_device(dcfg.device)
    try:
        if resolved_device is not None:
            model.to(resolved_device)
        if dcfg.fuse_model and hasattr(model, "fuse"):
            model.fuse()
    except Exception:
        # Safe-guard: continue without device move/fuse if unsupported
        pass
    all_rows: List[dict] = []

    for path in img_paths:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        base = os.path.splitext(os.path.basename(path))[0]

        step_x = max(1, int(tcfg.tile_w * (1 - tcfg.overlap)))
        step_y = max(1, int(tcfg.tile_h * (1 - tcfg.overlap)))
        xs = list(range(0, max(W - tcfg.tile_w, 0) + 1, step_x))
        ys = list(range(0, max(H - tcfg.tile_h, 0) + 1, step_y))
        if xs[-1] != W - tcfg.tile_w:
            xs.append(W - tcfg.tile_w)
        if ys[-1] != H - tcfg.tile_h:
            ys.append(H - tcfg.tile_h)

        boxes_g: List[List[float]] = []
        confs_g: List[float] = []

        # Prepare all tiles (RGB as required by ultralytics) and their offsets
        coords: List[Tuple[int, int]] = []
        tiles_rgb: List[np.ndarray] = []
        for y0 in ys:
            for x0 in xs:
                tile = img_bgr[y0:y0 + tcfg.tile_h, x0:x0 + tcfg.tile_w]
                tiles_rgb.append(tile[:, :, ::-1])
                coords.append((x0, y0))

        # Run inference in batches to reduce Python overhead while preserving results
        bs = max(1, int(dcfg.batch_tiles))
        for i in range(0, len(tiles_rgb), bs):
            batch = tiles_rgb[i:i + bs]
            batch_coords = coords[i:i + bs]
            if not batch:
                continue
            results = model.predict(
                source=batch,
                imgsz=tcfg.imgsz,
                conf=dcfg.conf_thres,
                iou=dcfg.iou_thres,
                classes=list(dcfg.classes),
                verbose=False,
                device=resolved_device,
                half=(bool(dcfg.half) and torch.cuda.is_available()),
            )
            for res, (x0, y0) in zip(results, batch_coords):
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                xyxy[:, [0, 2]] += x0
                xyxy[:, [1, 3]] += y0
                hpx = xyxy[:, 3] - xyxy[:, 1]
                keep = hpx >= tcfg.min_box_h_px
                xyxy = xyxy[keep]
                conf = conf[keep]
                for b, s in zip(xyxy, conf):
                    boxes_g.append(b.tolist())
                    confs_g.append(float(s))

        keep_idx = nms_merge(boxes_g, confs_g, iou_thr=0.55)
        boxes_g = [boxes_g[i] for i in keep_idx]
        confs_g = [confs_g[i] for i in keep_idx]
        if dcfg.containment_thr is not None:
            boxes_g, confs_g = merge_contained_boxes(boxes_g, confs_g, ioa_thr=float(dcfg.containment_thr))

        rows = []
        for (x1, y1, x2, y2), c in zip(boxes_g, confs_g):
            u = float(np.clip((x1 + x2) * 0.5, 0, W - 1))
            v = float(np.clip(y2, 0, H - 1))
            rows.append({
                "image": base, "input_path": path, "W": W, "H": H,
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "u_px": u, "v_px": v, "conf": c
            })
        per_csv = os.path.join(det_dir, f"{base}_detections.csv")
        pd.DataFrame(rows).to_csv(per_csv, index=False)
        all_rows.extend(rows)

        if annotate and ann_dir is not None:
            vis = img_bgr.copy()
            for (x1, y1, x2, y2), c in zip(boxes_g, confs_g):
                p1 = (int(round(x1)), int(round(y1)))
                p2 = (int(round(x2)), int(round(y2)))
                cv2.rectangle(vis, p1, p2, bbox_color_bgr, thickness=15)
                label = f"{c:.2f}"
                tx = int(round(x1))
                ty = max(0, int(round(y1)) - 6)
                cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 255, 240), 1, cv2.LINE_AA)
            # Optional timestamp overlay from input filename
            if stamp_timestamp:
                ts = extract_timestamp_text(base)
                if ts:
                    h, w = vis.shape[:2]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # Larger, more prominent scaling with dynamic thickness
                    font_scale = float(max(1.2, min(3.5, h / 400.0)))
                    thickness = int(max(2, round(font_scale * 2)))

                    # Compute text size and position near bottom-left
                    (text_w, text_h), baseline = cv2.getTextSize(ts, font, font_scale, thickness)
                    pad = int(max(8, round(h * 0.015)))
                    x = int(round(0.02 * w))
                    y = int(round(0.96 * h))
                    # Background box
                    x1 = x - pad
                    y1 = y - text_h - pad
                    x2 = x + text_w + pad
                    y2 = y + baseline + pad
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w - 1, x2)
                    y2 = min(h - 1, y2)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                    # White text centered within box baseline
                    cv2.putText(vis, ts, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            out_img = os.path.join(ann_dir, f"{base}_annotated.jpg")
            cv2.imwrite(out_img, vis)

    agg_csv = os.path.join(det_dir, "detections_all.csv")
    pd.DataFrame(all_rows).to_csv(agg_csv, index=False)
    return agg_csv
