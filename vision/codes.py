"""Stage 3: reading part markings (QR code, Data Matrix code, plain-text serial number).

Traceability needs the part ID on every image. The test images are generated: a code or text with a known
content is marked onto a real steel-surface crop from the NEU dataset (like a laser marking on a metal part), so
the correct answer is always known. The same degradations as for the inspection tasks are then applied.

Readers compared:
  QR:          OpenCV QRCodeDetector, ZBar (pyzbar), ZXing-C++
  Data Matrix: ZXing-C++ (OpenCV and ZBar cannot read Data Matrix)
  Text:        RapidOCR (PP-OCR models in ONNX format)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 256


@dataclass
class Sample:
    kind: str     # "qr" | "datamatrix" | "text"
    truth: str
    image: np.ndarray


def serial(i: int) -> str:
    return f"KV{26:02d}-{(7919 * i) % 100000:05d}-{'ABCDEFGHJK'[i % 10]}"


def _matrix(kind: str, text: str) -> np.ndarray:
    """Binary module matrix (True = dark) of a QR or Data Matrix code."""
    import zxingcpp
    fmt = zxingcpp.BarcodeFormat.QRCode if kind == "qr" else zxingcpp.BarcodeFormat.DataMatrix
    try:  # zxing-cpp >= 2.3
        bc = zxingcpp.create_barcode(text, fmt, **({"force_square": True} if kind == "datamatrix" else {}))
        img = np.array(bc.to_image(scale=1, add_quiet_zones=False))
    except AttributeError:  # older API
        img = np.array(zxingcpp.write_barcode(fmt, text, width=0, height=0, quiet_zone=0))
    return img < 128


def _background(textures: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = textures[rng.integers(len(textures))]
    t = cv2.resize(t, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    t = (t - t.mean()) * 0.45 + rng.uniform(150, 185)   # brighter, lower-contrast metal surface
    return t


def _mark(bg: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dark marking with slight ink/laser variation and soft edges."""
    depth = rng.uniform(85, 115)
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.6)
    var = 1 + rng.normal(0, 0.08, mask.shape).astype(np.float32)
    return np.clip(bg - depth * m * var, 0, 255).astype(np.uint8)


def make_sample(kind: str, i: int, textures: np.ndarray, seed: int = 0) -> Sample:
    rng = np.random.default_rng(seed * 100003 + i * 7 + {"qr": 1, "datamatrix": 2, "text": 3}[kind])
    text = serial(i)
    bg = _background(textures, rng)
    mask = np.zeros((SIZE, SIZE), np.float32)
    if kind in ("qr", "datamatrix"):
        mod = _matrix(kind, text)
        px = int(min(rng.integers(5, 8), (SIZE - 50) // max(mod.shape)))  # module size in pixels
        big = np.kron(mod, np.ones((px, px))).astype(np.float32)
        h, w = big.shape
        y0 = int(rng.integers(20, SIZE - h - 19)); x0 = int(rng.integers(20, SIZE - w - 19))
        mask[y0:y0 + h, x0:x0 + w] = big
    else:
        canvas = Image.new("L", (SIZE, SIZE), 0)
        draw = ImageDraw.Draw(canvas)
        size = int(rng.integers(22, 27))
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            font = ImageFont.load_default()
        tw = draw.textlength(text, font=font)
        x0 = int((SIZE - tw) / 2 + rng.integers(-10, 11)); y0 = int(rng.integers(95, 130))
        draw.text((x0, y0), text, fill=255, font=font)
        mask = np.array(canvas, np.float32) / 255.0
    return Sample(kind, text, _mark(bg, mask, rng))


# ----------------------------------------------------------------------------------------------- readers
def read_opencv_qr(img: np.ndarray) -> str:
    det = cv2.QRCodeDetector()
    text, _, _ = det.detectAndDecode(img)
    return text or ""


def read_zbar(img: np.ndarray) -> str:
    from pyzbar import pyzbar
    res = pyzbar.decode(img)
    return res[0].data.decode("utf-8", "replace") if res else ""


def read_zxing(img: np.ndarray) -> str:
    import zxingcpp
    res = zxingcpp.read_barcodes(img)
    return res[0].text if res else ""


_OCR = None


def read_ocr(img: np.ndarray) -> str:
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    res, _ = _OCR(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    if not res:
        return ""
    # longest recognised line, spaces removed
    return max((r[1] for r in res), key=len).replace(" ", "").upper()


READERS = {
    "qr": {"OpenCV": read_opencv_qr, "ZBar": read_zbar, "ZXing-C++": read_zxing},
    "datamatrix": {"ZXing-C++": read_zxing},
    "text": {"RapidOCR": read_ocr},
}


def timed(fn, img) -> tuple[str, float]:
    t0 = time.perf_counter()
    out = fn(img)
    return out, (time.perf_counter() - t0) * 1000
