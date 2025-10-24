from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


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


def detect_folder(images_dir: str, output_dir: str, dcfg: DetectConfig = DetectConfig(), tcfg: TilingConfig = TilingConfig()) -> str:
    os.makedirs(output_dir, exist_ok=True)
    det_dir = os.path.join(output_dir, "detections")
    os.makedirs(det_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG")
    img_paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir)) if f.endswith(exts)]

    model = YOLO(dcfg.model_name)
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

        for y0 in ys:
            for x0 in xs:
                tile = img_bgr[y0:y0 + tcfg.tile_h, x0:x0 + tcfg.tile_w]
                res = model.predict(
                    source=tile[:, :, ::-1], imgsz=tcfg.imgsz,
                    conf=dcfg.conf_thres, iou=dcfg.iou_thres, classes=list(dcfg.classes), verbose=False
                )[0]
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

    agg_csv = os.path.join(det_dir, "detections_all.csv")
    pd.DataFrame(all_rows).to_csv(agg_csv, index=False)
    return agg_csv
