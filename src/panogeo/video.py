from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


def _read_frame(cap: cv2.VideoCapture):
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _compute_canvas_size(
    size_a: Tuple[int, int],
    size_b: Tuple[int, int],
    layout: str,
    gap: int,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Return (out_w, out_h), (wA, hA), (wB, hB) after scaling to common dimension."""
    w1, h1 = size_a
    w2, h2 = size_b

    def make_even(x: int) -> int:
        # Many codecs require even dimensions
        return x if (x % 2 == 0) else max(2, x - 1)

    if layout == "h":  # side-by-side
        target_h = min(h1, h2)
        scale1 = target_h / float(h1)
        scale2 = target_h / float(h2)
        new_w1 = make_even(int(round(w1 * scale1)))
        new_w2 = make_even(int(round(w2 * scale2)))
        target_h = make_even(int(target_h))
        out_w = make_even(new_w1 + new_w2 + gap)
        out_h = target_h
        return (out_w, out_h), (new_w1, target_h), (new_w2, target_h)
    else:  # vertical stack
        target_w = min(w1, w2)
        scale1 = target_w / float(w1)
        scale2 = target_w / float(w2)
        new_h1 = make_even(int(round(h1 * scale1)))
        new_h2 = make_even(int(round(h2 * scale2)))
        target_w = make_even(int(target_w))
        out_w = target_w
        out_h = make_even(new_h1 + new_h2 + gap)
        return (out_w, out_h), (target_w, new_h1), (target_w, new_h2)


def compose_side_by_side_video(
    left_video: str,
    right_video: str,
    out_path: str,
    *,
    layout: str = "h",
    gap: int = 8,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    show_progress: bool = True,
    codec: Optional[str] = None,
) -> str:
    """
    Compose two videos side-by-side (layout='h') or stacked vertically (layout='v').

    Args:
        left_video: Path to first (left/top) video, e.g. camera with trajectories.
        right_video: Path to second (right/bottom) video, e.g. map with trajectories.
        out_path: Output mp4 path.
        layout: 'h' for horizontal side-by-side, 'v' for vertical stack.
        gap: Pixels of spacing between panels.
        bg_color: BGR tuple for gap/background color.
        fps: Optional output FPS (defaults to min of inputs or first's FPS).
        max_frames: Optional cap on number of frames to write.
        show_progress: If True, print simple progress every ~30 frames.
    """
    cap_a = cv2.VideoCapture(str(left_video))
    cap_b = cv2.VideoCapture(str(right_video))
    if not cap_a.isOpened():
        raise FileNotFoundError(f"Cannot open left video: {left_video}")
    if not cap_b.isOpened():
        cap_a.release()
        raise FileNotFoundError(f"Cannot open right video: {right_video}")

    w1 = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 30.0
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = float(fps if (fps and fps > 0) else min(float(fps_a), float(fps_b)))
    n1 = int(max(0, cap_a.get(cv2.CAP_PROP_FRAME_COUNT)))
    n2 = int(max(0, cap_b.get(cv2.CAP_PROP_FRAME_COUNT)))
    n_total = n1 if n1 and n2 == 0 else (n2 if n2 and n1 == 0 else min(n1, n2))
    if max_frames is not None:
        n_total = min(n_total, max_frames)

    (out_w, out_h), (wa, ha), (wb, hb) = _compute_canvas_size((w1, h1), (w2, h2), layout, gap)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or "", exist_ok=True)

    # Pick a codec that's widely supported; try multiple if not specified
    ext = os.path.splitext(str(out_path))[1].lower()
    candidate_codecs = []
    if codec:
        candidate_codecs = [codec]
    else:
        if ext == ".mp4":
            # Prefer H.264 ('avc1' or 'H264'); fall back to 'mp4v' if unavailable
            candidate_codecs = ["avc1", "H264", "X264", "mp4v"]
        elif ext == ".avi":
            candidate_codecs = ["MJPG", "XVID", "DIVX"]
        else:
            # Default to mp4v which ships with OpenCV builds commonly
            candidate_codecs = ["mp4v"]

    writer = None
    last_err = None
    for cc in candidate_codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*cc)
            tmp_writer = cv2.VideoWriter(str(out_path), fourcc, float(out_fps), (int(out_w), int(out_h)))
            if tmp_writer is not None and tmp_writer.isOpened():
                writer = tmp_writer
                if show_progress:
                    print(f"[compose] using codec={cc}, size=({out_w}x{out_h}) @ {out_fps:.2f}fps")
                break
            else:
                if tmp_writer is not None:
                    tmp_writer.release()
        except Exception as e:
            last_err = e
            continue
    if writer is None:
        raise RuntimeError(
            f"Failed to open VideoWriter for '{out_path}'. Tried codecs: {candidate_codecs}. "
            f"Consider changing file extension to '.avi' or passing codec='MJPG'. "
            f"Last error: {last_err}"
        )

    try:
        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            frame_a = _read_frame(cap_a)
            frame_b = _read_frame(cap_b)
            if frame_a is None or frame_b is None:
                break
            # Resize to match target panel sizes
            if (frame_a.shape[1], frame_a.shape[0]) != (wa, ha):
                frame_a = cv2.resize(frame_a, (wa, ha))
            if (frame_b.shape[1], frame_b.shape[0]) != (wb, hb):
                frame_b = cv2.resize(frame_b, (wb, hb))
            # Create canvas and place
            canvas = np.full((out_h, out_w, 3), bg_color, dtype=np.uint8)
            if layout == "h":
                canvas[0:ha, 0:wa] = frame_a
                canvas[0:hb, wa + gap : wa + gap + wb] = frame_b
            else:
                # vertical stack
                canvas[0:ha, 0:wa] = frame_a
                canvas[ha + gap : ha + gap + hb, 0:wb] = frame_b
            writer.write(canvas)
            frame_idx += 1
            if show_progress and frame_idx % 30 == 0:
                print(f"[compose] frame {frame_idx}/{n_total if n_total else ''}")
    finally:
        cap_a.release()
        cap_b.release()
        writer.release()
    return out_path


