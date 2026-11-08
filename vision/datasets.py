"""Loading the two real inspection datasets.

* steel: NEU surface defect database (hot-rolled steel strip), 6 defect classes x 300 images, 200 x 200 px.
  Task: which defect is it? (multi-class classification). The official train/validation split (240/60 per class)
  is used as train/test.
* casting: casting product images (top view of submersible pump impellers), 519 OK and 781 defective,
  512 x 512 px, the version without augmentation. Task: OK or defective? Our own stratified split
  (70 % train, 30 % test, seed 0), because the augmented version of the dataset contains rotated copies of the
  same parts in train and test.

All images are loaded as greyscale uint8 at a fixed working size.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data"
STEEL_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SIZES = {"steel": 200, "casting": 256}


@dataclass
class Split:
    name: str
    classes: list[str]
    x_train: np.ndarray  # (n, h, w) uint8
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    files_test: list[str]

    @property
    def size(self) -> int:
        return self.x_train.shape[1]


def _read(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def available(name: str) -> bool:
    d = ROOT / ("neu" if name == "steel" else "casting")
    return d.exists() and any(d.rglob("*.jp*g"))


def load_steel() -> Split:
    size = SIZES["steel"]
    out = {}
    for part in ("train", "test"):
        xs, ys, fs = [], [], []
        for ci, c in enumerate(STEEL_CLASSES):
            for p in sorted((ROOT / "neu" / part / c).glob("*.jpg")):
                xs.append(_read(p, size)); ys.append(ci); fs.append(p.name)
        out[part] = (np.stack(xs), np.array(ys), fs)
    return Split("steel", STEEL_CLASSES, out["train"][0], out["train"][1], out["test"][0], out["test"][1], out["test"][2])


def load_casting(test_share: float = 0.3, seed: int = 0) -> Split:
    size = SIZES["casting"]
    rng = np.random.default_rng(seed)
    tr_x, tr_y, te_x, te_y, te_f = [], [], [], [], []
    for ci, c in enumerate(["ok", "defect"]):
        files = sorted((ROOT / "casting" / c).glob("*.jp*g"))
        idx = rng.permutation(len(files))
        n_test = int(round(len(files) * test_share))
        for k, i in enumerate(idx):
            img = _read(files[i], size)
            if k < n_test:
                te_x.append(img); te_y.append(ci); te_f.append(files[i].name)
            else:
                tr_x.append(img); tr_y.append(ci)
    return Split("casting", ["ok", "defect"], np.stack(tr_x), np.array(tr_y), np.stack(te_x), np.array(te_y), te_f)


def load(name: str) -> Split:
    return load_steel() if name == "steel" else load_casting()


def subsample(x: np.ndarray, y: np.ndarray, share: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subset of the training data (for the 'how much data do I need' experiment)."""
    if share >= 1:
        return x, y
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        n = max(2, int(round(len(idx) * share)))
        keep.extend(rng.choice(idx, n, replace=False))
    keep = np.sort(np.array(keep))
    return x[keep], y[keep]
