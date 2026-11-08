import numpy as np
import pytest

from vision import augment, classical, codes, datasets, degrade, experiments

rng = np.random.default_rng(0)
IMG = (rng.normal(128, 30, (200, 200))).clip(0, 255).astype(np.uint8)


# ------------------------------------------------------------------ degradations
def test_every_degradation_keeps_shape_and_type():
    for kind in degrade.DEGRADATIONS:
        for lv in degrade.LEVELS:
            out = degrade.apply(IMG, kind, lv)
            assert out.shape == IMG.shape and out.dtype == np.uint8, kind


def test_level_zero_is_identity():
    assert np.array_equal(degrade.apply(IMG, "defocus", 0), IMG)


def test_severity_increases():
    """Higher level = stronger change, for every degradation."""
    for kind in degrade.DEGRADATIONS:
        diffs = [np.abs(degrade.apply(IMG, kind, lv).astype(float) - IMG).mean() for lv in degrade.LEVELS]
        assert diffs[-1] > diffs[0], kind


def test_exposure_and_contrast_physics():
    assert degrade.apply(IMG, "underexposure", 4).mean() < 0.3 * IMG.mean() + 1
    lc = degrade.apply(IMG, "low_contrast", 4)
    assert abs(lc.mean() - IMG.mean()) < 2 and lc.std() < 0.3 * IMG.std()


def test_conditions_count():
    assert len(degrade.conditions()) == 1 + len(degrade.DEGRADATIONS) * 4


# ------------------------------------------------------------------ features and helpers
def test_classical_feature_vector_is_finite_and_fixed_length():
    f1 = classical.features(IMG)
    f2 = classical.features(degrade.apply(IMG, "noise", 3))
    assert f1.shape == f2.shape and np.isfinite(f1).all()


def test_subsample_is_stratified():
    y = np.repeat(np.arange(3), 100)
    x = np.arange(300)[:, None]
    xs, ys = datasets.subsample(x, y, 0.1)
    assert np.bincount(ys).tolist() == [10, 10, 10]


def test_tolerance_rule():
    curve = {"1": 0.99, "2": 0.97, "3": 0.80, "4": 0.96}
    assert experiments.tolerance(curve, base=1.0) == 2   # stops at the first level below the limit


def test_metrics_casting_counts_misses_and_false_rejects():
    y = np.array([0, 0, 1, 1])
    m = experiments.metrics("casting", y, score=np.array([0.1, 0.9, 0.8, 0.2]), threshold=0.5)
    assert m["missed_defects"] == 0.5 and m["false_rejects"] == 0.5 and m["accuracy"] == 0.5


def test_augmentation_adds_copies():
    x = np.stack([IMG] * 4); y = np.arange(4)
    xa, ya = augment.augment_set(x, y, copies=1)
    assert len(xa) == 8 and (ya[:4] == ya[4:]).all() and not np.array_equal(xa[4], x[0])


# ------------------------------------------------------------------ codes
@pytest.mark.parametrize("kind", ["qr", "datamatrix"])
def test_generated_codes_are_readable_when_clean(kind):
    textures = np.full((1, 200, 200), 170, np.uint8)
    ok = sum(codes.read_zxing(codes.make_sample(kind, i, textures).image) == codes.serial(i) for i in range(5))
    assert ok == 5


def test_reader_reports_empty_string_when_nothing_found():
    assert codes.read_zxing(np.full((100, 100), 128, np.uint8)) == ""


# ------------------------------------------------------------------ real data (skipped if not downloaded)
@pytest.mark.skipif(not datasets.available("steel"), reason="NEU data not downloaded")
def test_steel_split_sizes():
    sp = datasets.load_steel()
    assert sp.x_train.shape == (1440, 200, 200) and sp.x_test.shape == (360, 200, 200)
    assert np.bincount(sp.y_test).tolist() == [60] * 6


@pytest.mark.skipif(not datasets.available("casting"), reason="casting data not downloaded")
def test_casting_split_is_stratified_and_disjoint():
    sp = datasets.load_casting()
    assert len(sp.x_train) + len(sp.x_test) == 1300
    share = sp.y_test.mean() / np.concatenate([sp.y_train, sp.y_test]).mean()
    assert 0.97 < share < 1.03
    train_keys = {im.tobytes() for im in sp.x_train}
    assert not any(im.tobytes() in train_keys for im in sp.x_test)


# ------------------------------------------------------------------ report text
@pytest.mark.skipif(not any((experiments.REPORTS / n).exists() for n in ("results.json", "results_quick.json")),
                    reason="no test run yet")
def test_findings_and_reports_are_written_in_both_languages():
    import json
    from vision import findings, report
    name = "results.json" if (experiments.REPORTS / "results.json").exists() else "results_quick.json"
    r = json.loads((experiments.REPORTS / name).read_text())
    en, de = findings.english(r), findings.german(r)
    assert "Recommendation" in en and "Empfehlung" in de
    # every percentage in the English findings comes from the results file
    assert f"{r['summary']['steel'][findings.analyse(r)['steel']['best']]['mean_accuracy_degraded'] * 100:.1f} %" in en
    assert "## 4. Ergebnisse" in report.build(r, True) or not (experiments.REPORTS / "findings_de.md").exists()


# ------------------------------------------------------------------ station integration (OPC UA handshake)
def _fake_inspect(img):
    from vision.station import Inspection
    if img is None:
        return Inspection(False, 1.0, 0, 0, 0, error=1)
    return Inspection(bool(img.mean() > 100), 0.1, 12.5, -3.0, 45.0)


def test_handshake_passes_results_and_is_fail_safe():
    import asyncio
    from vision import station
    good, bad = np.full((32, 32), 200, np.uint8), np.full((32, 32), 50, np.uint8)
    frames = {1: good, 2: bad, 3: None, 4: good}
    run = asyncio.run(station.run_line(lambda p: frames[p], _fake_inspect, 4, port=48490))
    c = run["cycles"]
    assert [r["ok"] for r in c] == [True, False, False, True]      # part 3: no image -> rejected
    assert c[2]["error"] == 1 and c[0]["x_mm"] == 12.5 and c[0]["angle_deg"] == 45.0
    assert not run["alarms"]


def test_watchdog_stops_the_line_when_station_hangs():
    import asyncio
    from vision import station
    run = asyncio.run(station.run_line(lambda p: np.full((32, 32), 200, np.uint8), _fake_inspect, 5, port=48491,
                                       hang_after=2, timeout_ms=400))
    assert len(run["cycles"]) == 3 and run["cycles"][-1]["error"] == -1 and run["cycles"][-1]["ok"] is False
    assert "line stopped" in run["alarms"][0]["alarm"]


def test_locate_finds_centre_and_angle_of_a_part():
    from vision.station import locate
    img = np.full((200, 200), 220, np.uint8)
    cv2 = pytest.importorskip("cv2")
    box = cv2.boxPoints(((130, 80), (90, 30), 30)).astype(np.int32)   # centre 30 px right, 20 px up
    cv2.fillPoly(img, [box], 40)
    x, y, a = locate(img, mm_per_px=0.5)
    assert abs(x - 15) < 1 and abs(y - 10) < 1 and abs(abs(a) - 30) < 3


# ------------------------------------------------------------------ calibration
def test_calibration_recovers_a_known_camera():
    import cv2
    from vision import calibrate
    rng = np.random.default_rng(1)
    K = np.array([[600.0, 0, 322.0], [0, 602.0, 238.0], [0, 0, 1]])
    dist = np.array([-0.25, 0.08, 0, 0, 0])
    views = []
    for i in range(8):
        rv = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.2)])
        R, _ = cv2.Rodrigues(rv)
        t = np.array([0, 0, rng.uniform(320, 420)]) - R @ np.array([105, 75, 0])
        views.append(calibrate.render_view(K, dist, rv, t, (640, 480), seed=i))
    res = calibrate.calibrate(views, 30.0)
    assert res["rms_reprojection_px"] < 1.0
    assert abs(res["fx"] / 600 - 1) < 0.02 and abs(res["cx"] - 322) < 10
    assert res["plane_error_mm"]["with_calibration"]["mean"] < 1.0
