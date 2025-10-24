from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def shift_equirect(img: Image.Image, degrees: float = 180.0) -> Image.Image:
    """Circularly shift an equirectangular image horizontally by degrees.

    Positive degrees shifts content to the right (wrap-around).
    """
    arr = np.array(img)
    h, w = arr.shape[0], arr.shape[1]
    px_shift = int(round((degrees / 360.0) * w)) % w
    if px_shift == 0:
        return img.copy()
    shifted = np.roll(arr, shift=px_shift, axis=1)
    return Image.fromarray(shifted)


def pixel_to_cam_ray(u_px: float, v_px: float, width: int, height: int) -> np.ndarray:
    """Equirectangular pixel to unit ray in camera frame.

    Camera frame:
    - +x forward (towards pano center)
    - +y up
    - +z left
    """
    lam = 2.0 * math.pi * (u_px / float(width) - 0.5)
    phi = (math.pi / 2.0 - math.pi * (v_px / float(height)))
    cphi, sphi = math.cos(phi), math.sin(phi)
    x = cphi * math.cos(lam)
    y = sphi
    z = cphi * math.sin(lam)
    v3 = np.array([x, y, z], dtype=float)
    n = np.linalg.norm(v3)
    if n == 0:
        return v3
    return v3 / n
