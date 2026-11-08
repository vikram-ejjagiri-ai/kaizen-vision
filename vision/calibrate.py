"""Camera calibration and pixel-to-millimetre conversion with a ChArUco board (works with a laptop webcam).

Why: a robot or PLC needs positions in millimetres, the camera delivers pixels. Calibration removes the lens
distortion and measures the camera; the board then defines the work plane, so every pixel can be converted to
millimetres on that plane.

    python -m vision.calibrate board                  # docs/charuco_board_A4.pdf (print at 100 %) and .png
    python -m vision.calibrate capture --camera 0     # webcam: SPACE saves a view (15-20 views), Q quits
    python -m vision.calibrate run --square-mm 30     # calibrate from data/calib/*.png -> reports/calibration.json
    python -m vision.calibrate measure --camera 0     # click two points: their distance in mm on the board plane
    python -m vision.calibrate synthetic              # self-check with rendered views of a known camera

No printer? Show docs/charuco_board.png full-screen on a phone or tablet, measure one black square with a ruler
and pass that size as --square-mm.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = ROOT / "data" / "calib"
REPORTS = ROOT / "reports"
DICT = cv2.aruco.DICT_5X5_100
SQUARES = (7, 5)              # columns, rows
MARKER_RATIO = 0.75           # marker side / square side


def board(square_mm: float = 30.0) -> cv2.aruco.CharucoBoard:
    return cv2.aruco.CharucoBoard(SQUARES, square_mm, square_mm * MARKER_RATIO, cv2.aruco.getPredefinedDictionary(DICT))


def board_image(px_per_mm: float = 300 / 25.4, square_mm: float = 30.0, margin_mm: float = 10.0) -> np.ndarray:
    b = board(square_mm)
    w = int(round((SQUARES[0] * square_mm + 2 * margin_mm) * px_per_mm))
    h = int(round((SQUARES[1] * square_mm + 2 * margin_mm) * px_per_mm))
    return b.generateImage((w, h), marginSize=int(round(margin_mm * px_per_mm)), borderBits=1)


def write_board_files() -> None:
    """A4 PDF at true size (square 30 mm) and a PNG for screens."""
    from PIL import Image
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    img = board_image()
    a4 = Image.new("L", (int(297 / 25.4 * 300), int(210 / 25.4 * 300)), 255)   # landscape A4 at 300 dpi
    pil = Image.fromarray(img)
    a4.paste(pil, ((a4.width - pil.width) // 2, (a4.height - pil.height) // 2))
    a4.save(docs / "charuco_board_A4.pdf", resolution=300)
    cv2.imwrite(str(docs / "charuco_board.png"), board_image(px_per_mm=6))
    print("written docs/charuco_board_A4.pdf (print at 100 %, squares 30 mm) and docs/charuco_board.png")


# ------------------------------------------------------------------------------------------------ detection
def detect(img: np.ndarray, square_mm: float) -> tuple[np.ndarray, np.ndarray] | None:
    """ChArUco corners: (image points n x 2, board points n x 3 in mm), or None if fewer than 6 corners."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = board(square_mm)
    det = cv2.aruco.CharucoDetector(b)
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    obj, imgp = b.matchImagePoints(corners, ids)
    return imgp.reshape(-1, 2).astype(np.float32), obj.reshape(-1, 3).astype(np.float32)


# ------------------------------------------------------------------------------------------------ calibration
def calibrate(images: list[np.ndarray], square_mm: float) -> dict:
    obj_pts, img_pts, used = [], [], []
    for i, im in enumerate(images):
        d = detect(im, square_mm)
        if d is not None:
            img_pts.append(d[0]); obj_pts.append(d[1]); used.append(i)
    if len(used) < 5:
        raise RuntimeError(f"only {len(used)} usable views; need at least 5 (better 15) with the board fully visible")
    h, w = images[0].shape[:2]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
    per_view = []
    for o, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip) ** 2, 1)))))
    plane = plane_error_mm(images, square_mm, K, dist)
    return {"image_size": [w, h], "views_used": len(used), "views_total": len(images), "square_mm": square_mm,
            "rms_reprojection_px": float(rms), "per_view_px": per_view,
            "camera_matrix": K.tolist(), "distortion": dist.ravel().tolist(),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "plane_error_mm": plane}


def homography_mm(img_pts: np.ndarray, obj_pts: np.ndarray, K=None, dist=None) -> np.ndarray:
    """Pixel -> board-plane millimetres (after removing lens distortion if the camera is calibrated)."""
    p = img_pts if K is None else cv2.undistortPoints(img_pts.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    H, _ = cv2.findHomography(p, obj_pts[:, :2], cv2.RANSAC, 3.0)
    return H


def to_mm(H: np.ndarray, pts_px: np.ndarray, K=None, dist=None) -> np.ndarray:
    p = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
    if K is not None:
        p = cv2.undistortPoints(p, K, dist, P=K)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def plane_error_mm(images, square_mm, K, dist) -> dict:
    """Measuring accuracy on the board plane: build the pixel-to-mm mapping from every second corner and predict
    the millimetre position of the other corners (which the mapping has not seen)."""
    errs_cal, errs_raw = [], []
    for im in images:
        d = detect(im, square_mm)
        if d is None or len(d[0]) < 12:
            continue
        ip, op = d
        fit, test = np.arange(len(ip)) % 2 == 0, np.arange(len(ip)) % 2 == 1
        for k, d_, errs in ((K, dist, errs_cal), (None, None, errs_raw)):
            H = homography_mm(ip[fit], op[fit], k, d_)
            if H is None:
                continue
            pred = to_mm(H, ip[test], k, d_)
            errs.extend(np.linalg.norm(pred - op[test][:, :2], axis=1).tolist())
    f = lambda e: {"mean": round(float(np.mean(e)), 3), "p95": round(float(np.percentile(e, 95)), 3),
                   "max": round(float(np.max(e)), 3)} if e else None
    return {"with_calibration": f(errs_cal), "without_calibration": f(errs_raw)}


# ------------------------------------------------------------------------------------------------ synthetic check
def render_view(K, dist, rvec, tvec, size, square_mm=30.0, px_per_mm=8.0, noise=2.0, seed=0) -> np.ndarray:
    """Image of the board as a camera with matrix K and distortion would see it at pose (rvec, tvec)."""
    tex = board_image(px_per_mm=px_per_mm, square_mm=square_mm, margin_mm=10.0)
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)          # normalised ray directions
    R, _ = cv2.Rodrigues(rvec)
    n, t = R[:, 2], tvec.ravel()
    rays = np.c_[xy, np.ones(len(xy))]
    s = (n @ t) / (rays @ n)
    P = rays * s[:, None]
    B = (P - t) @ R                                                  # board coordinates in mm
    mx = ((B[:, 0] + 10.0) * px_per_mm).reshape(h, w).astype(np.float32)
    my = ((B[:, 1] + 10.0) * px_per_mm).reshape(h, w).astype(np.float32)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderValue=140)
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def synthetic_check(n_views: int = 15, seed: int = 0) -> dict:
    """Render views of a known webcam-like camera, calibrate, compare with the truth."""
    rng = np.random.default_rng(seed)
    size = (1280, 720)
    K = np.array([[900.0, 0, 652.0], [0, 905.0, 355.0], [0, 0, 1]])
    dist = np.array([-0.32, 0.12, 0.001, -0.0015, 0.0])
    views = []
    cx_board, cy_board = SQUARES[0] * 30 / 2, SQUARES[1] * 30 / 2
    for i in range(n_views):
        rv = np.array([rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(-0.3, 0.3)])
        R, _ = cv2.Rodrigues(rv)
        z = rng.uniform(380, 560)
        centre = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), z])
        t = centre - R @ np.array([cx_board, cy_board, 0])
        views.append(render_view(K, dist, rv, t, size, seed=i))
    res = calibrate(views, 30.0)
    res["truth"] = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "k1": dist[0], "k2": dist[1]}
    res["source"] = "synthetic check: rendered views of a known camera (1280 x 720, strong barrel distortion)"
    res["error_vs_truth"] = {"fx_%": round(abs(res["fx"] / K[0, 0] - 1) * 100, 3),
                             "fy_%": round(abs(res["fy"] / K[1, 1] - 1) * 100, 3),
                             "cx_px": round(abs(res["cx"] - K[0, 2]), 2), "cy_px": round(abs(res["cy"] - K[1, 2]), 2),
                             "k1_abs": round(abs(res["distortion"][0] - dist[0]), 4)}
    cv2.imwrite(str(REPORTS / "calibration_synthetic_example.png"), views[0])
    return res


# ------------------------------------------------------------------------------------------------ webcam tools
def capture(camera: int, square_mm: float) -> None:
    """Live view with detected corners; SPACE saves the current frame if the board is found, Q quits."""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    saved = len(list(CALIB_DIR.glob("*.png")))
    print("SPACE = save view, Q = quit. Move the board: different angles, distances and image corners.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("no image from camera", camera); break
        show = frame.copy()
        d = detect(frame, square_mm)
        if d is not None:
            for p in d[0]:
                cv2.circle(show, tuple(int(c) for c in p), 4, (0, 200, 0), -1)
        cv2.putText(show, f"corners: {0 if d is None else len(d[0])}   saved: {saved}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        cv2.imshow("Kaizen Vision calibration", show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and d is not None:
            saved += 1
            cv2.imwrite(str(CALIB_DIR / f"view_{saved:02d}.png"), frame)
        elif key in (ord("q"), 27):
            break
    cap.release(); cv2.destroyAllWindows()
    print(f"{saved} views in {CALIB_DIR}")


def measure(camera: int, square_mm: float) -> None:
    """Live: board defines the plane; click two points to see their positions and distance in mm."""
    c = json.loads((REPORTS / "calibration.json").read_text())
    K, dist = np.array(c["camera_matrix"]), np.array(c["distortion"])
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["image_size"][0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["image_size"][1])
    clicks: list[tuple[int, int]] = []
    cv2.namedWindow("measure")
    cv2.setMouseCallback("measure", lambda e, x, y, *_: clicks.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    H = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        d = detect(frame, square_mm)
        if d is not None:
            H = homography_mm(d[0], d[1], K, dist)
        show = frame.copy()
        pts = clicks[-2:]
        if H is not None and pts:
            mm = to_mm(H, np.array(pts, np.float32), K, dist)
            for (x, y), (X, Y) in zip(pts, mm):
                cv2.circle(show, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(show, f"({X:.1f}, {Y:.1f}) mm", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if len(pts) == 2:
                cv2.line(show, pts[0], pts[1], (0, 0, 255), 2)
                cv2.putText(show, f"distance {np.linalg.norm(mm[0] - mm[1]):.1f} mm", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("measure", show)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    cap.release(); cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["board", "capture", "run", "measure", "synthetic"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--square-mm", type=float, default=30.0)
    a = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    if a.command == "board":
        write_board_files()
    elif a.command == "capture":
        capture(a.camera, a.square_mm)
    elif a.command == "run":
        files = sorted(CALIB_DIR.glob("*.png"))
        res = calibrate([cv2.imread(str(f)) for f in files], a.square_mm)
        res["source"] = f"webcam, {len(files)} views, {time.strftime('%Y-%m-%d %H:%M')}"
        (REPORTS / "calibration.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ("views_used", "rms_reprojection_px", "plane_error_mm")}, indent=1))
    elif a.command == "measure":
        measure(a.camera, a.square_mm)
    else:
        res = synthetic_check()
        (REPORTS / "calibration_synthetic.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ("views_used", "rms_reprojection_px", "error_vs_truth", "plane_error_mm")},
                         indent=1))


if __name__ == "__main__":
    main()
