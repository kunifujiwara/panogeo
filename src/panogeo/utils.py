from __future__ import annotations

import os
import re
from typing import Optional, Tuple


_TS_RE = re.compile(r"IMG_(\d{8})_(\d{6})_")


def extract_timestamp_text(name: str) -> Optional[str]:
    """Extract human-readable timestamp from filenames like IMG_YYYYMMDD_HHMMSS_**_**.jpg

    Returns a string like "YYYY-MM-DD HH:MM:SS" or None if pattern not found.
    """
    base = os.path.basename(name)
    m = _TS_RE.search(base)
    if not m:
        return None
    ymd, hms = m.groups()
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]} {hms[:2]}:{hms[2:4]}:{hms[4:]}"


def id_color_rgb(oid: int) -> Tuple[int, int, int]:
    """
    Deterministic RGB color for a given track/object id.

    This function ensures consistent colors across different visualizations
    (e.g., perspective video via OpenCV and map video via Matplotlib).
    """
    r = (37 * oid) % 255
    g = (17 * oid + 85) % 255
    b = (97 * oid + 170) % 255
    return int(r), int(g), int(b)


def id_color_bgr(oid: int) -> Tuple[int, int, int]:
    """
    Deterministic BGR color for OpenCV, derived from id_color_rgb.
    """
    r, g, b = id_color_rgb(oid)
    return int(b), int(g), int(r)


def id_color_rgb01(oid: int) -> Tuple[float, float, float]:
    """
    Deterministic RGB color in [0,1] for Matplotlib.
    """
    r, g, b = id_color_rgb(oid)
    return (r / 255.0, g / 255.0, b / 255.0)






