"""Method 1: classical machine vision.

Hand-crafted texture and intensity features (what a vision engineer would configure in a traditional vision
software: grey-level statistics, texture measures, edge density) and a support vector machine on top.
No neural network, no pre-trained weights, small and fast.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

_GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def features(img: np.ndarray) -> np.ndarray:
    """Feature vector of one greyscale image (about 80 numbers)."""
    f = img.astype(np.float32)
    flat = f.ravel()
    # 1. grey-level statistics
    g = [flat.mean(), flat.std(), stats.skew(flat), stats.kurtosis(flat),
         *np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])]
    # 2. texture: local binary patterns at two scales (rotation invariant, uniform)
    lbp = []
    for p, r in ((8, 1), (16, 2)):
        codes = local_binary_pattern(img, p, r, method="uniform")
        hist, _ = np.histogram(codes, bins=p + 2, range=(0, p + 2), density=True)
        lbp.extend(hist)
    # 3. texture: grey-level co-occurrence matrix (32 grey levels, 4 directions, 2 distances)
    q = (img // 8).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32,
                        symmetric=True, normed=True)
    tex = []
    for prop in _GLCM_PROPS:
        v = graycoprops(glcm, prop)  # (distances, angles)
        tex.extend(v.mean(axis=1))
        tex.extend(v.std(axis=1))
    # 4. edges and gradients
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    edges = cv2.Canny(img, 50, 150)
    ang = np.arctan2(gy, gx)[mag > np.percentile(mag, 75)]
    ah, _ = np.histogram(np.abs(ang), bins=6, range=(0, np.pi), density=True)
    e = [mag.mean(), mag.std(), np.percentile(mag, 95), (edges > 0).mean(), *ah]
    # 5. blob-like dark/bright spots (pits, holes): share of pixels far from the local mean
    local = cv2.blur(f, (15, 15))
    d = f - local
    s = d.std() + 1e-6
    b = [(d < -2.5 * s).mean(), (d > 2.5 * s).mean(), np.abs(d).mean()]
    # constant images (e.g. completely over- or under-exposed) give undefined skew/correlation -> 0
    return np.nan_to_num(np.array(g + lbp + tex + e + b, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def feature_matrix(images: np.ndarray) -> np.ndarray:
    return np.stack([features(im) for im in images])


class ClassicalSVM:
    name = "Classical features + SVM"
    short = "classical"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray, feats: np.ndarray | None = None) -> "ClassicalSVM":
        fx = feature_matrix(x) if feats is None else feats
        pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=self.seed))
        n_splits = int(min(5, np.bincount(y).min()))
        grid = GridSearchCV(pipe, {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01]},
                            cv=StratifiedKFold(n_splits, shuffle=True, random_state=self.seed), n_jobs=1)
        grid.fit(fx, y)
        self.params = grid.best_params_
        # probability outputs (needed for thresholds and AUROC) by calibrating the selected SVM
        self.model = CalibratedClassifierCV(grid.best_estimator_, cv=n_splits).fit(fx, y)
        return self

    def predict_proba(self, x: np.ndarray, feats: np.ndarray | None = None) -> np.ndarray:
        fx = feature_matrix(x) if feats is None else feats
        return self.model.predict_proba(fx)
