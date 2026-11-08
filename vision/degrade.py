"""Controlled image degradations that mimic what changes on a real inspection station.

Each degradation has four severity levels (1 = mild, 4 = severe). Level 0 is the clean image.
They are applied to the test images only: the models are trained on clean images, so the experiment answers
"what happens to each method when the lighting, focus, camera or part position changes after commissioning?".
"""
from __future__ import annotations

import cv2
import numpy as np

# name: (what it simulates on the line, unit, values for levels 1..4)
DEGRADATIONS: dict[str, tuple[str, str, list[float]]] = {
    "underexposure": ("lamp ageing, shorter exposure time", "brightness factor", [0.75, 0.55, 0.40, 0.25]),
    "overexposure": ("reflections, longer exposure time", "brightness factor", [1.3, 1.6, 2.0, 2.5]),
    "low_contrast": ("flat, diffuse lighting", "contrast factor", [0.75, 0.55, 0.40, 0.25]),
    "uneven_light": ("side light, vignetting", "brightness drop across image", [0.2, 0.4, 0.6, 0.8]),
    "defocus": ("wrong working distance, focus drift", "blur sigma (px)", [0.75, 1.5, 2.5, 4.0]),
    "motion_blur": ("part moving on the conveyor", "blur length (px)", [3, 7, 11, 15]),
    "noise": ("high camera gain in poor light", "noise sigma (grey levels)", [5, 10, 20, 35]),
    "low_resolution": ("cheaper camera, larger field of view", "scale factor", [0.75, 0.5, 0.35, 0.25]),
    "rotation": ("part orientation on the fixture", "degrees", [5, 10, 20, 45]),
    "camera_tilt": ("camera not perpendicular to the part", "degrees", [5, 10, 15, 25]),
    "jpeg": ("image compression on the network", "JPEG quality", [50, 30, 15, 8]),
}
LEVELS = [1, 2, 3, 4]


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def apply(img: np.ndarray, kind: str, level: int, seed: int = 0) -> np.ndarray:
    """Return a degraded copy of a greyscale uint8 image."""
    if level == 0 or kind == "clean":
        return img.copy()
    v = DEGRADATIONS[kind][2][level - 1]
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    if kind in ("underexposure", "overexposure"):
        return _clip(f * v)
    if kind == "low_contrast":
        m = f.mean()
        return _clip(m + (f - m) * v)
    if kind == "uneven_light":
        ramp = np.linspace(1.0, 1.0 - v, w, dtype=np.float32)[None, :]
        return _clip(f * ramp)
    if kind == "defocus":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=float(v))
    if kind == "motion_blur":
        k = int(v)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    if kind == "noise":
        rng = np.random.default_rng(seed)
        return _clip(f + rng.normal(0, v, img.shape))
    if kind == "low_resolution":
        small = cv2.resize(img, (max(8, int(w * v)), max(8, int(h * v))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "rotation":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(v), 1.0)
        return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if kind == "camera_tilt":
        # tilt about the horizontal axis: the far edge appears shorter
        d = 0.5 * w * (1 - np.cos(np.radians(v))) + w * 0.25 * np.sin(np.radians(v)) * 0.5
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[d, 0], [w - d, 0], [w, h], [0, h]])
        m = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(v)])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    raise ValueError(f"unknown degradation {kind}")


def conditions() -> list[tuple[str, int]]:
    """All test conditions: clean plus every degradation at every level."""
    return [("clean", 0)] + [(k, lv) for k in DEGRADATIONS for lv in LEVELS]
