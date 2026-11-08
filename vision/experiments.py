"""The test series: every method on every task, under every test condition.

Run:  python -m vision.experiments            (full run, about 1.5 h on a laptop CPU)
      python -m vision.experiments --quick    (fewer images and conditions, about 10 min, for a quick check)

Writes reports/results.json (all numbers) and reports/results.md (summary tables).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from . import augment, classical, codes, datasets, deep, degrade

warnings.filterwarnings("ignore")
REPORTS = Path(__file__).resolve().parents[1] / "reports"


# ------------------------------------------------------------------------------------------------ helpers
def degraded(x: np.ndarray, kind: str, level: int) -> np.ndarray:
    return np.stack([degrade.apply(im, kind, level, seed=i) for i, im in enumerate(x)])


def metrics(task: str, y: np.ndarray, prob: np.ndarray | None = None, score: np.ndarray | None = None,
            threshold: float = 0.5) -> dict:
    """Accuracy for steel; for casting also AUROC, missed defects and false rejects."""
    if task == "steel":
        return {"accuracy": float(accuracy_score(y, prob.argmax(1)))}
    s = prob[:, 1] if score is None else score
    pred = (s > threshold).astype(int)
    out = {"accuracy": float(accuracy_score(y, pred)),
           "missed_defects": float(1 - pred[y == 1].mean()),
           "false_rejects": float(pred[y == 0].mean())}
    out["auroc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    return out


def timer(fn, *a, repeat: int = 1):
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn(*a)
    return out, (time.perf_counter() - t0) / repeat


def n_params(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))


# ------------------------------------------------------------------------------------------- inspection
def run_inspection(task: str, net, quick: bool, log) -> dict:
    sp = datasets.load(task)
    xtr, ytr, xte, yte = sp.x_train, sp.y_train, sp.x_test, sp.y_test
    if quick:
        keep = np.random.default_rng(0).permutation(len(xte))[:120]
        xte, yte = xte[keep], yte[keep]
    conds = degrade.conditions()
    if quick:
        conds = [("clean", 0)] + [(k, lv) for k in degrade.DEGRADATIONS for lv in (2, 4)]
    epochs = 3 if quick else 12
    res = {"task": task, "classes": sp.classes, "n_train": int(len(xtr)), "n_test": int(len(xte)),
           "image_px": sp.size, "methods": {}, "robustness": {}, "data_needs": {}, "speed": {}}

    # ---------------- training on clean images (plus robust variants trained with photometric variation)
    log(f"[{task}] features of {len(xtr)} training images")
    ftr, t_feat = timer(classical.feature_matrix, xtr)
    etr, t_emb = timer(deep.embed, net, xtr)
    xaug, yaug = augment.augment_set(xtr, ytr, copies=1, seed=1)
    faug = np.concatenate([ftr, classical.feature_matrix(xaug[len(xtr):])])
    eaug = np.concatenate([etr, deep.embed(net, xaug[len(xtr):])])

    models = {}
    t0 = time.perf_counter(); models["classical"] = classical.ClassicalSVM().fit(xtr, ytr, ftr)
    train_s = {"classical": t_feat + time.perf_counter() - t0}
    t0 = time.perf_counter(); models["features"] = deep.DeepFeatures(net).fit(xtr, ytr, etr)
    train_s["features"] = t_emb + time.perf_counter() - t0
    log(f"[{task}] fine-tuning ({epochs} epochs)")
    models["finetuned"] = deep.FineTuned(len(sp.classes), epochs=epochs).fit(xtr, ytr)
    train_s["finetuned"] = models["finetuned"].train_seconds
    models["classical_robust"] = classical.ClassicalSVM().fit(xaug, yaug, faug)
    models["features_robust"] = deep.DeepFeatures(net).fit(xaug, yaug, eaug)
    log(f"[{task}] fine-tuning robust variant")
    models["finetuned_robust"] = deep.FineTuned(len(sp.classes), epochs=epochs).fit(xaug, yaug)
    if task == "casting":
        good = xtr[ytr == 0]
        n_bank = int(len(good) * 0.8)
        log(f"[{task}] anomaly memory bank from {n_bank} good parts")
        t0 = time.perf_counter()
        models["anomaly"] = deep.PatchCore(net).fit(good[:n_bank])
        models["anomaly"].calibrate(good[n_bank:])
        train_s["anomaly"] = time.perf_counter() - t0

    def predict(name: str, x: np.ndarray, f=None, e=None) -> dict:
        m = models[name]
        if name.startswith("classical"):
            return metrics(task, yte_cur, m.predict_proba(None, f))
        if name.startswith("features"):
            return metrics(task, yte_cur, m.predict_proba(None, e))
        if name == "anomaly":
            return metrics(task, yte_cur, score=m.score(x), threshold=m.threshold)
        return metrics(task, yte_cur, m.predict_proba(x))

    # ---------------- robustness: every condition
    yte_cur = yte
    for kind, level in conds:
        x = xte if kind == "clean" else degraded(xte, kind, level)
        f = classical.feature_matrix(x)
        e = deep.embed(net, x)
        row = {name: predict(name, x, f, e) for name in models}
        res["robustness"].setdefault(kind, {})[str(level)] = row
        if kind == "clean":
            log(f"[{task}] clean: " + ", ".join(f"{k} {v['accuracy']:.3f}" for k, v in row.items()))
    log(f"[{task}] robustness done")

    # ---------------- speed on CPU (per image, batch of 32, after warm-up)
    xs = xte[:32]
    speed = {}
    speed["classical"] = timer(lambda: models["classical"].predict_proba(xs), repeat=2)[1] / len(xs) * 1000
    speed["features"] = timer(lambda: models["features"].predict_proba(xs), repeat=2)[1] / len(xs) * 1000
    speed["finetuned"] = timer(lambda: models["finetuned"].predict_proba(xs), repeat=2)[1] / len(xs) * 1000
    if "anomaly" in models:
        speed["anomaly"] = timer(lambda: models["anomaly"].score(xs), repeat=2)[1] / len(xs) * 1000
    res["speed"] = {k: {"ms_per_image": round(v, 1), "train_seconds": round(train_s[k], 1)} for k, v in speed.items()}
    res["speed"]["classical"]["parameters"] = "about 80 features, SVM"
    res["speed"]["features"]["parameters"] = n_params(net)
    res["speed"]["finetuned"]["parameters"] = n_params(models["finetuned"].net)
    if "anomaly" in models:
        res["speed"]["anomaly"]["parameters"] = f"{n_params(net)} + memory bank {tuple(models['anomaly'].bank.shape)}"
        res["anomaly_threshold"] = models["anomaly"].threshold

    # ---------------- data needs: how many training images per class are enough?
    shares = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0] if not quick else [0.05, 0.25, 1.0]
    seeds = [0, 1, 2] if not quick else [0]
    fte, ete = classical.feature_matrix(xte), deep.embed(net, xte)
    for share in shares:
        rows = {}
        for name in ("classical", "features", "finetuned"):
            vals = []
            for sd in (seeds if name != "finetuned" else [0]):
                xs_, ys_ = datasets.subsample(xtr, ytr, share, seed=sd)
                idx = _index_of(xtr, xs_)
                if name == "classical":
                    m = classical.ClassicalSVM().fit(xs_, ys_, ftr[idx]); p = m.predict_proba(None, fte)
                elif name == "features":
                    m = deep.DeepFeatures(net).fit(xs_, ys_, etr[idx]); p = m.predict_proba(None, ete)
                else:
                    if share == 1.0:
                        m = models["finetuned"]
                    else:
                        m = deep.FineTuned(len(sp.classes), epochs=epochs).fit(xs_, ys_)
                    p = m.predict_proba(xte)
                vals.append(metrics(task, yte, p)["accuracy"])
            rows[name] = {"accuracy": float(np.mean(vals)), "min": float(np.min(vals)),
                          "images_per_class": int(min(np.bincount(ys_)))}
        res["data_needs"][str(share)] = rows
        log(f"[{task}] data share {share}: " + ", ".join(f"{k} {v['accuracy']:.3f}" for k, v in rows.items()))
    if task == "casting":
        good = xtr[ytr == 0]
        curve = {}
        for n in ([5, 10, 25, 50, 100, len(good) - 60] if not quick else [10, 50]):
            m = deep.PatchCore(net).fit(good[:n])
            m.calibrate(good[-60:])
            curve[str(n)] = metrics(task, yte, score=m.score(xte), threshold=m.threshold)
            log(f"[{task}] anomaly with {n} good images: AUROC {curve[str(n)]['auroc']:.3f}")
        res["data_needs_anomaly"] = curve

    # ---------------- example heat maps for the dashboard (anomaly)
    if task == "casting":
        m = models["anomaly"]
        ex = np.flatnonzero(yte == 1)[:4].tolist() + np.flatnonzero(yte == 0)[:2].tolist()
        maps = m.score_maps(xte[ex])
        np.savez_compressed(REPORTS / "anomaly_examples.npz", images=xte[ex], maps=maps, labels=yte[ex],
                            threshold=m.threshold)
    return res


def _index_of(full: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """Positions of the subset rows in the full array (subsample keeps order, so match by content hash)."""
    keys = {full[i].tobytes(): i for i in range(len(full))}
    return np.array([keys[s.tobytes()] for s in subset])


# -------------------------------------------------------------------------------------------------- codes
def run_codes(textures: np.ndarray, quick: bool, log) -> dict:
    n = 15 if quick else 60
    conds = degrade.conditions() if not quick else [("clean", 0)] + [(k, lv) for k in degrade.DEGRADATIONS for lv in (2, 4)]
    res = {"n_per_kind": n, "read_rate": {}, "ms_per_image": {}}
    samples = {k: [codes.make_sample(k, i, textures) for i in range(n)] for k in codes.READERS}
    times = {}
    for kind_c, level in conds:
        row = {}
        for kind, readers in codes.READERS.items():
            for rname, fn in readers.items():
                ok = 0
                for i, s in enumerate(samples[kind]):
                    img = s.image if kind_c == "clean" else degrade.apply(s.image, kind_c, level, seed=i)
                    out, ms = codes.timed(fn, img)
                    ok += out == s.truth
                    times.setdefault(f"{kind}|{rname}", []).append(ms)
                row[f"{kind}|{rname}"] = ok / n
        res["read_rate"].setdefault(kind_c, {})[str(level)] = row
        if kind_c == "clean":
            log("[codes] clean: " + ", ".join(f"{k} {v:.2f}" for k, v in row.items()))
    res["ms_per_image"] = {k: round(float(np.median(v)), 1) for k, v in times.items()}
    np.savez_compressed(REPORTS / "code_examples.npz", **{k: samples[k][0].image for k in samples})
    log("[codes] done")
    return res


# ------------------------------------------------------------------------------------------------ summary
METHOD_NAMES = {
    "classical": "Classical features + SVM",
    "features": "Pretrained CNN features + linear classifier",
    "finetuned": "Fine-tuned CNN",
    "anomaly": "Anomaly detection (good parts only)",
    "classical_robust": "Classical + SVM, trained with lighting/blur/noise variation",
    "features_robust": "CNN features, trained with variation",
    "finetuned_robust": "Fine-tuned CNN, trained with variation",
}


def tolerance(curve: dict, base: float, drop: float = 0.05) -> int:
    """Highest severity level at which the accuracy is still within `drop` of the clean accuracy."""
    ok = 0
    for lv in sorted(int(k) for k in curve):
        if curve[str(lv)] >= base - drop:
            ok = lv
        else:
            break
    return ok


def summarise(res: dict) -> dict:
    out = {}
    for task in ("steel", "casting"):
        r = res[task]
        clean = r["robustness"]["clean"]["0"]
        tol = {}
        for name in clean:
            base = clean[name]["accuracy"]
            per = {}
            for kind, levels in r["robustness"].items():
                if kind == "clean":
                    continue
                per[kind] = tolerance({lv: levels[lv][name]["accuracy"] for lv in levels}, base)
            worst = {kind: min(levels[lv][name]["accuracy"] for lv in levels)
                     for kind, levels in r["robustness"].items() if kind != "clean"}
            tol[name] = {"clean": base, "tolerance_levels": per,
                         "mean_accuracy_degraded": float(np.mean([levels[lv][name]["accuracy"]
                                                                 for kind, levels in r["robustness"].items()
                                                                 if kind != "clean" for lv in levels])),
                         "worst_case": worst}
        out[task] = tol
    return out


def report_md(res: dict) -> str:
    s = res["summary"]
    lines = ["# Kaizen Vision: test results", "", f"Run: {res['meta']['date']}, {res['meta']['mode']} mode, "
             f"{res['meta']['minutes']} min on {res['meta']['cpu']}", ""]
    for task, title in (("steel", "Steel surface: which defect? (6 classes)"),
                        ("casting", "Cast impeller: OK or defective?")):
        lines += [f"## {title}", "", "| Method | Clean accuracy | Mean accuracy over all 44 degraded conditions | ms per image |",
                  "|---|---|---|---|"]
        for name, v in s[task].items():
            ms = res[task]["speed"].get(name, {}).get("ms_per_image", "")
            lines.append(f"| {METHOD_NAMES[name]} | {v['clean']:.1%} | {v['mean_accuracy_degraded']:.1%} | {ms} |")
        lines.append("")
    rr = res["codes"]["read_rate"]
    lines += ["## Part markings: read rate", "", "| Reader | Clean | Mean over degraded conditions | ms |", "|---|---|---|---|"]
    for key in rr["clean"]["0"]:
        vals = [rr[k][lv][key] for k in rr if k != "clean" for lv in rr[k]]
        lines.append(f"| {key.replace('|', ': ')} | {rr['clean']['0'][key]:.0%} | {np.mean(vals):.0%} | "
                     f"{res['codes']['ms_per_image'][key]} |")
    return "\n".join(lines) + "\n"


def main(quick: bool = False) -> dict:
    REPORTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    logf = open(REPORTS / ("run_quick.log" if quick else "run.log"), "w")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    net = deep.load_network()
    res = {}
    res["steel"] = run_inspection("steel", net, quick, log)
    res["casting"] = run_inspection("casting", net, quick, log)
    res["codes"] = run_codes(datasets.load_steel().x_train, quick, log)
    res["degradations"] = {k: {"simulates": v[0], "unit": v[1], "levels": v[2]} for k, v in degrade.DEGRADATIONS.items()}
    res["summary"] = summarise(res)
    import platform
    res["meta"] = {"date": time.strftime("%Y-%m-%d %H:%M"), "mode": "quick" if quick else "full",
                   "minutes": round((time.perf_counter() - t0) / 60, 1),
                   "cpu": f"{platform.processor() or platform.machine()}, {deep.torch.get_num_threads()} threads",
                   "network": deep.WEIGHTS}
    name = "results_quick" if quick else "results"
    (REPORTS / f"{name}.json").write_text(json.dumps(res, indent=1))
    (REPORTS / f"{name}.md").write_text(report_md(res))
    log(f"finished in {res['meta']['minutes']} min")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
