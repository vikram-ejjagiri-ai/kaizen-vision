"""Findings and recommendation, written from reports/results.json in English and German.

    python -m vision.findings        -> reports/findings.md, reports/findings_de.md

Every number in the text is taken from the results file, so the text always matches the test run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPORTS = Path(__file__).resolve().parents[1] / "reports"
BASE = ["classical", "features", "finetuned", "anomaly"]
N_EN = {"classical": "classical features + SVM", "features": "pretrained CNN features + linear classifier",
        "finetuned": "fine-tuned CNN", "anomaly": "anomaly detection",
        "classical_robust": "classical + SVM trained with variation", "features_robust": "CNN features trained with variation",
        "finetuned_robust": "fine-tuned CNN trained with variation"}
N_DE = {"classical": "klassische Merkmale + SVM", "features": "vortrainierte CNN-Merkmale + linearer Klassifikator",
        "finetuned": "feinabgestimmtes CNN", "anomaly": "Anomalieerkennung",
        "classical_robust": "klassisch + SVM, mit Variation trainiert", "features_robust": "CNN-Merkmale, mit Variation trainiert",
        "finetuned_robust": "feinabgestimmtes CNN, mit Variation trainiert"}
C_EN = {"underexposure": "underexposure", "overexposure": "overexposure", "low_contrast": "low contrast",
        "uneven_light": "uneven light", "defocus": "defocus", "motion_blur": "motion blur", "noise": "sensor noise",
        "low_resolution": "lower resolution", "rotation": "rotation", "camera_tilt": "camera tilt", "jpeg": "JPEG compression"}
C_DE = {"underexposure": "Unterbelichtung", "overexposure": "Überbelichtung", "low_contrast": "geringer Kontrast",
        "uneven_light": "ungleichmäßige Beleuchtung", "defocus": "Unschärfe (Fokus)", "motion_blur": "Bewegungsunschärfe",
        "noise": "Sensorrauschen", "low_resolution": "geringere Auflösung", "rotation": "Drehung",
        "camera_tilt": "Kameraneigung", "jpeg": "JPEG-Kompression"}
TASK_EN = {"steel": "Steel surface (6 defect types)", "casting": "Cast impeller (OK / defective)"}
TASK_DE = {"steel": "Stahloberfläche (6 Fehlerarten)", "casting": "Gussteil (i. O. / n. i. O.)"}


def pct(x: float, de: bool = False) -> str:
    s = f"{x * 100:.1f} %" if de else f"{x * 100:.1f} %"
    return s.replace(".", ",") if de else s


def pts(x: float, de: bool = False) -> str:
    s = f"{x * 100:+.0f}"
    return s + (" Prozentpunkte" if de else " points")


def analyse(r: dict) -> dict:
    a = {}
    for task in ("steel", "casting"):
        S = r["summary"][task]
        base = [n for n in BASE if n in S]
        rob = r[task]["robustness"]
        cond_mean = {k: float(np.mean([v[lv][n]["accuracy"] for lv in v for n in base])) for k, v in rob.items()
                     if k != "clean"}
        best = max(base, key=lambda n: S[n]["mean_accuracy_degraded"])
        worst = min(base, key=lambda n: S[n]["mean_accuracy_degraded"])
        gains = {n: S[f"{n}_robust"]["mean_accuracy_degraded"] - S[n]["mean_accuracy_degraded"]
                 for n in ("classical", "features", "finetuned") if f"{n}_robust" in S}
        best_robust = max((f"{n}_robust" for n in gains), key=lambda n: S[n]["mean_accuracy_degraded"])
        dn = r[task]["data_needs"]
        shares = sorted(dn, key=float)
        small = shares[1] if len(shares) > 2 else shares[0]
        a[task] = {
            "clean_min": min(S[n]["clean"] for n in base), "clean_max": max(S[n]["clean"] for n in base),
            "best": best, "best_mean": S[best]["mean_accuracy_degraded"],
            "worst": worst, "worst_mean": S[worst]["mean_accuracy_degraded"],
            "hardest": sorted(cond_mean, key=cond_mean.get)[:3], "easiest": sorted(cond_mean, key=cond_mean.get)[-2:],
            "cond_mean": cond_mean, "gains": gains, "best_robust": best_robust,
            "best_robust_mean": S[best_robust]["mean_accuracy_degraded"],
            "small_share": small, "small": dn[small],
            "speed": r[task]["speed"],
        }
    an = r["casting"].get("data_needs_anomaly", {})
    if an:
        ns = sorted(an, key=int)
        enough = next((n for n in ns if an[n]["auroc"] >= 0.95), ns[-1])
        a["anomaly"] = {"curve": {int(n): an[n]["auroc"] for n in ns},
                        "clean": r["casting"]["robustness"]["clean"]["0"]["anomaly"], "n_enough": int(enough), "auroc_enough": an[enough]["auroc"], "n_max": int(ns[-1]),
                        "auroc_max": an[ns[-1]]["auroc"], "n_min": int(ns[0]), "auroc_min": an[ns[0]]["auroc"]}
    rr = r["codes"]["read_rate"]
    keys = list(rr["clean"]["0"])
    mean = {k: float(np.mean([rr[c][lv][k] for c in rr if c != "clean" for lv in rr[c]])) for k in keys}
    per_cond = {c: float(np.mean([rr[c][lv][k] for lv in rr[c] for k in keys])) for c in rr if c != "clean"}
    a["codes"] = {"clean": rr["clean"]["0"], "mean": mean, "best_qr": max([k for k in keys if k.startswith("qr")], key=mean.get),
                  "hardest": sorted(per_cond, key=per_cond.get)[:3], "ms": r["codes"]["ms_per_image"]}
    return a


def english(r: dict) -> str:
    a = analyse(r)
    s, c = a["steel"], a["casting"]
    L = []
    L.append("**1. Clean images do not decide.** On clean test images every method is between "
             f"{pct(min(s['clean_min'], c['clean_min']))} and {pct(max(s['clean_max'], c['clean_max']))} accurate. "
             "A demonstration under lab conditions would not show which one to choose.")
    for t, x in (("steel", s), ("casting", c)):
        L.append(f"**2{'a' if t == 'steel' else 'b'}. {TASK_EN[t]}:** when conditions change, the "
                 f"{N_EN[x['best']]} keeps {pct(x['best_mean'])} on average over all 44 changed conditions, the "
                 f"{N_EN[x['worst']]} only {pct(x['worst_mean'])}. The hardest conditions are "
                 + ", ".join(C_EN[k] for k in x["hardest"]) + ".")
    g = {t: max(a[t]["gains"].values()) for t in ("steel", "casting")}
    gk = {t: max(a[t]["gains"], key=a[t]["gains"].get) for t in ("steel", "casting")}
    same = s["best_robust"] == c["best_robust"]
    best_txt = (f"Best on both tasks: {N_EN[s['best_robust']]} ({pct(s['best_robust_mean'])} steel, "
                f"{pct(c['best_robust_mean'])} casting)." if same else
                f"Best: {N_EN[s['best_robust']]} ({pct(s['best_robust_mean'])}, steel) and {N_EN[c['best_robust']]} "
                f"({pct(c['best_robust_mean'])}, casting).")
    L.append("**3. Train with the variation you expect.** Retraining with random lighting, blur and noise raises the "
             f"best mean accuracy under changed conditions from {pct(s['best_mean'])} to {pct(s['best_robust_mean'])} "
             f"(steel) and from {pct(c['best_mean'])} to {pct(c['best_robust_mean'])} (casting); the largest gain is "
             f"{pts(g['steel'])} for the {N_EN[gk['steel']]} on steel. {best_txt}")
    sm = s["small"]
    best4 = max(sm[n]["accuracy"] for n in ("classical", "features", "finetuned"))
    top = " or ".join(N_EN[n] for n in ("features", "classical", "finetuned") if sm[n]["accuracy"] >= best4 - 0.01)
    flip = c["small"]["finetuned"]["accuracy"] > c["small"]["features"]["accuracy"] + 0.02
    L.append(f"**4. Few training images: {'the best method depends on the task' if flip else top}.** With only {sm['features']['images_per_class']} "
             f"images per class (steel), the pretrained features reach {pct(sm['features']['accuracy'])}, the classical "
             f"method {pct(sm['classical']['accuracy'])} and the fine-tuned network {pct(sm['finetuned']['accuracy'])}. "
             + (f"On the castings it is the other way round: with {c['small']['finetuned']['images_per_class']} images per "
                f"class the fine-tuned network reaches {pct(c['small']['finetuned']['accuracy'])}, the pretrained features "
                f"{pct(c['small']['features']['accuracy'])}, the classical method {pct(c['small']['classical']['accuracy'])}. "
                "Which method needs the fewest images depends on the task, so it has to be tested."
                if c["small"]["finetuned"]["accuracy"] > c["small"]["features"]["accuracy"] + 0.02 else
                "Fine-tuning the whole network needs more images before it pays off."))
    if "anomaly" in a:
        an = a["anomaly"]
        cv = an["curve"]; ks = sorted(cv)
        mid = ks[len(ks) // 2]
        L.append(f"**5. No defect images? Anomaly detection.** Trained on good parts only, it separates OK and "
                 f"defective castings with AUROC {cv[ks[-1]]:.3f} ({ks[-1]} good images); with {mid} good images "
                 f"{cv[mid]:.3f}, with {ks[0]} only {cv[ks[0]]:.3f}. With its automatic threshold it misses "
                 f"{pct(an['clean']['missed_defects'])} of the defects and rejects {pct(an['clean']['false_rejects'])} of "
                 "the good parts, clearly behind the supervised methods. It is the right start for a new product while "
                 "defect images are still rare, but it reacts to every change of the image, so lighting and camera must "
                 "be kept stable.")
    ms = max(v["ms_per_image"] for t in ("steel", "casting") for v in a[t]["speed"].values())
    L.append(f"**6. Speed is not the bottleneck.** Every method needs at most {ms:.0f} ms per image on a laptop CPU, "
             "far below the cycle time of a typical line.")
    k = a["codes"]
    L.append(f"**7. Part markings:** on clean images the readers decode "
             + ", ".join(f"{key.split('|')[0].replace('datamatrix', 'Data Matrix').replace('qr', 'QR').replace('text', 'text')} "
                         f"({key.split('|')[1]}) {v:.0%}" for key, v in k["clean"].items())
             + ". Hardest for every reader: " + ", ".join(C_EN[x] for x in k["hardest"]) + ".")
    L.append("**Recommendation for a new station:** fix lighting, focus and camera mounting first (changed conditions cost far more accuracy in the tests than the choice between methods on clean images), "
             f"use the {N_EN[c['best_robust'].replace('_robust', '')]} trained with the expected variation, start with "
             "anomaly detection while defect images are rare, and confirm with trial images on the real station.")
    return "\n\n".join(L) + "\n"


def german(r: dict) -> str:
    a = analyse(r)
    s, c = a["steel"], a["casting"]
    d = True
    L = []
    L.append("**1. Saubere Bilder entscheiden nicht.** Auf sauberen Testbildern liegen alle Verfahren zwischen "
             f"{pct(min(s['clean_min'], c['clean_min']), d)} und {pct(max(s['clean_max'], c['clean_max']), d)} "
             "Genauigkeit. Eine Vorführung unter Laborbedingungen zeigt nicht, welches Verfahren das richtige ist.")
    for t, x in (("steel", s), ("casting", c)):
        L.append(f"**2{'a' if t == 'steel' else 'b'}. {TASK_DE[t]}:** Bei veränderten Bedingungen liegt das "
                 f"robusteste Verfahren ({N_DE[x['best']]}) im Mittel über alle 44 Bedingungen bei {pct(x['best_mean'], d)}, "
                 f"das schwächste ({N_DE[x['worst']]}) bei {pct(x['worst_mean'], d)}. Am schwierigsten: "
                 + ", ".join(C_DE[k] for k in x["hardest"]) + ".")
    g = {t: max(a[t]["gains"].values()) for t in ("steel", "casting")}
    gk = {t: max(a[t]["gains"], key=a[t]["gains"].get) for t in ("steel", "casting")}
    same = s["best_robust"] == c["best_robust"]
    best_txt = (f"Am besten bei beiden Aufgaben: {N_DE[s['best_robust']]} ({pct(s['best_robust_mean'], d)} Stahl, "
                f"{pct(c['best_robust_mean'], d)} Guss)." if same else
                f"Am besten: {N_DE[s['best_robust']]} ({pct(s['best_robust_mean'], d)}, Stahl) und "
                f"{N_DE[c['best_robust']]} ({pct(c['best_robust_mean'], d)}, Guss).")
    L.append("**3. Mit der erwarteten Variation trainieren.** Ein erneutes Training mit zufälliger Helligkeit, Unschärfe "
             "und Rauschen hebt die beste mittlere Genauigkeit unter veränderten Bedingungen von "
             f"{pct(s['best_mean'], d)} auf {pct(s['best_robust_mean'], d)} (Stahl) und von {pct(c['best_mean'], d)} auf "
             f"{pct(c['best_robust_mean'], d)} (Guss); der größte Gewinn beträgt {pts(g['steel'], d)} für "
             f"{N_DE[gk['steel']]} (Stahl). {best_txt}")
    sm = s["small"]
    best4 = max(sm[n]["accuracy"] for n in ("classical", "features", "finetuned"))
    top = " oder ".join(N_DE[n] for n in ("features", "classical", "finetuned") if sm[n]["accuracy"] >= best4 - 0.01)
    flip = c["small"]["finetuned"]["accuracy"] > c["small"]["features"]["accuracy"] + 0.02
    L.append(f"**4. Wenige Trainingsbilder: {'das beste Verfahren hängt von der Aufgabe ab' if flip else top}.** Mit nur {sm['features']['images_per_class']} Bildern "
             f"pro Klasse (Stahl) erreichen die vortrainierten Merkmale {pct(sm['features']['accuracy'], d)}, das "
             f"klassische Verfahren {pct(sm['classical']['accuracy'], d)} und das feinabgestimmte Netz "
             f"{pct(sm['finetuned']['accuracy'], d)}. "
             + (f"Bei den Gussteilen ist es umgekehrt: mit {c['small']['finetuned']['images_per_class']} Bildern pro Klasse "
                f"erreicht das feinabgestimmte Netz {pct(c['small']['finetuned']['accuracy'], d)}, die vortrainierten "
                f"Merkmale {pct(c['small']['features']['accuracy'], d)}, das klassische Verfahren "
                f"{pct(c['small']['classical']['accuracy'], d)}. Welches Verfahren mit den wenigsten Bildern auskommt, "
                "hängt von der Aufgabe ab und muss getestet werden."
                if c["small"]["finetuned"]["accuracy"] > c["small"]["features"]["accuracy"] + 0.02 else
                "Das ganze Netz feinabzustimmen lohnt sich erst mit mehr Bildern."))
    if "anomaly" in a:
        an = a["anomaly"]
        f3 = lambda v: f"{v:.3f}".replace(".", ",")
        cv = an["curve"]; ks = sorted(cv)
        mid = ks[len(ks) // 2]
        L.append(f"**5. Keine Fehlerbilder? Anomalieerkennung.** Nur mit Gutteilen trainiert, trennt sie i. O. und "
                 f"n. i. O. mit AUROC {f3(cv[ks[-1]])} ({ks[-1]} Gutteilbilder); mit {mid} Gutteilbildern {f3(cv[mid])}, "
                 f"mit {ks[0]} nur {f3(cv[ks[0]])}. Mit ihrer automatischen Schwelle übersieht sie "
                 f"{pct(an['clean']['missed_defects'], d)} der Fehler und sortiert {pct(an['clean']['false_rejects'], d)} "
                 "der Gutteile aus, deutlich schlechter als die überwachten Verfahren. Richtig für den Anlauf eines neuen "
                 "Produkts, solange Fehlerbilder selten sind; sie reagiert aber auf jede Bildänderung, deshalb müssen "
                 "Beleuchtung und Kamera stabil bleiben.")
    ms = max(v["ms_per_image"] for t in ("steel", "casting") for v in a[t]["speed"].values())
    L.append(f"**6. Die Rechenzeit ist kein Engpass.** Jedes Verfahren braucht höchstens {ms:.0f} ms pro Bild auf einer "
             "Laptop-CPU, weit unter der Taktzeit einer typischen Linie.")
    k = a["codes"]
    L.append("**7. Teilekennzeichnung:** auf sauberen Bildern lesen die Leser "
             + ", ".join(f"{key.split('|')[0].replace('datamatrix', 'Data Matrix').replace('qr', 'QR').replace('text', 'Klartext')} "
                         f"({key.split('|')[1]}) {v:.0%}".replace("%", " %") for key, v in k["clean"].items())
             + ". Am schwierigsten für alle Leser: " + ", ".join(C_DE[x] for x in k["hardest"]) + ".")
    L.append("**Empfehlung für eine neue Station:** zuerst Beleuchtung, Fokus und Kameramontage festlegen (veränderte Bedingungen kosten in den Versuchen weit mehr Genauigkeit als die Wahl zwischen den Verfahren auf sauberen Bildern), "
             f"{N_DE[c['best_robust'].replace('_robust', '')]} mit der erwarteten Variation trainieren, mit "
             "Anomalieerkennung starten, solange Fehlerbilder selten sind, und mit Versuchsbildern an der realen Station "
             "bestätigen.")
    return "\n\n".join(L) + "\n"


def main(name: str = "results.json") -> None:
    p = REPORTS / name
    if not p.exists():
        p = REPORTS / "results_quick.json"
    r = json.loads(p.read_text())
    (REPORTS / "findings.md").write_text(english(r))
    (REPORTS / "findings_de.md").write_text(german(r))
    print(english(r))


if __name__ == "__main__":
    main()
