from __future__ import annotations

import os
import re
from typing import Optional


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







