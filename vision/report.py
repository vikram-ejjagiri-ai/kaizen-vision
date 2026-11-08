"""Test report (English) and Versuchsbericht (German), generated from reports/results.json.

    python -m vision.report     -> docs/test_report.md, docs/versuchsbericht.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import findings

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
BASE = ["classical", "features", "finetuned", "anomaly"]
ALL = BASE + ["classical_robust", "features_robust", "finetuned_robust"]


def f(x: float, de: bool) -> str:
    s = f"{x * 100:.1f} %"
    return s.replace(".", ",") if de else s


def tables(r: dict, de: bool) -> list[str]:
    N = findings.N_DE if de else findings.N_EN
    C = findings.C_DE if de else findings.C_EN
    T = findings.TASK_DE if de else findings.TASK_EN
    out = []
    for task in ("steel", "casting"):
        S = r["summary"][task]
        names = [n for n in ALL if n in S]
        out += [f"### {T[task]}", ""]
        if de:
            out += ["| Verfahren | sauber | Mittel unter veränderten Bedingungen | schlechtester Fall |", "|---|---|---|---|"]
        else:
            out += ["| Method | clean | mean under changed conditions | worst case |", "|---|---|---|---|"]
        for n in names:
            worst_k = min(S[n]["worst_case"], key=S[n]["worst_case"].get)
            out.append(f"| {N[n]} | {f(S[n]['clean'], de)} | {f(S[n]['mean_accuracy_degraded'], de)} | "
                       f"{f(S[n]['worst_case'][worst_k], de)} ({C[worst_k]}) |")
        if task == "steel":
            chance = 1 / 6
        else:
            chance = None
        if chance:
            out.append(("Zufallsniveau: 16,7 % (6 Klassen)." if de else "Chance level: 16.7 % (6 classes)."))
        else:
            out.append(("Wer alles als n. i. O. meldet, erreicht hier etwa 60 % (Anteil der Fehlerteile im Testsatz)."
                        if de else "Calling every part defective would score about 60 % here (share of defective parts in the test set)."))
        out.append("")
        kinds = list(next(iter(S.values()))["tolerance_levels"])
        out += [("**Toleranz** (höchste Stufe 0-4 innerhalb von 5 Prozentpunkten der sauberen Genauigkeit)" if de else
                 "**Tolerance** (highest level 0-4 within 5 points of the clean accuracy)"), ""]
        out.append("| | " + " | ".join(C[k] for k in kinds) + " |")
        out.append("|---|" + "---|" * len(kinds))
        for n in names:
            out.append(f"| {N[n]} | " + " | ".join(str(S[n]["tolerance_levels"][k]) for k in kinds) + " |")
        out.append("")
        if task == "casting":
            cl = r[task]["robustness"]["clean"]["0"]
            if de:
                out += ["| Verfahren (sauber) | AUROC | Fehlerschlupf | Pseudoausschuss |", "|---|---|---|---|"]
            else:
                out += ["| Method (clean) | AUROC | missed defects | false rejects |", "|---|---|---|---|"]
            for n in names:
                out.append(f"| {N[n]} | {cl[n]['auroc']:.3f} | {f(cl[n]['missed_defects'], de)} | {f(cl[n]['false_rejects'], de)} |"
                           .replace(f"{cl[n]['auroc']:.3f}", f"{cl[n]['auroc']:.3f}".replace(".", ",") if de else f"{cl[n]['auroc']:.3f}"))
            out.append("")
        dn = r[task]["data_needs"]
        out += [("**Trainingsdaten** (Genauigkeit auf dem festen Testsatz)" if de else
                 "**Training data** (accuracy on the fixed test set)"), ""]
        out.append(("| Bilder pro Klasse | " if de else "| images per class | ") +
                   " | ".join(N[n] for n in ("classical", "features", "finetuned")) + " |")
        out.append("|---|---|---|---|")
        for share in sorted(dn, key=float):
            row = dn[share]
            out.append(f"| {row['features']['images_per_class']} | " +
                       " | ".join(f(row[n]["accuracy"], de) for n in ("classical", "features", "finetuned")) + " |")
        out.append("")
        sp = r[task]["speed"]
        out += [("| Verfahren | ms pro Bild (CPU) | Trainingszeit (s) |" if de else
                 "| Method | ms per image (CPU) | training time (s) |"), "|---|---|---|"]
        for n, v in sp.items():
            ms, ts = str(v['ms_per_image']), str(v['train_seconds'])
            out.append(f"| {N[n]} | {ms.replace('.', ',') if de else ms} | {ts.replace('.', ',') if de else ts} |")
        out.append("")
    rr = r["codes"]["read_rate"]
    keys = list(rr["clean"]["0"])
    out += [("### Teilekennzeichnung: Leserate" if de else "### Part markings: read rate"), ""]
    out.append(("| Leser | sauber | Mittel verändert | ms |" if de else "| Reader | clean | mean changed | ms |"))
    out.append("|---|---|---|---|")
    for k in keys:
        m = np.mean([rr[c][lv][k] for c in rr if c != "clean" for lv in rr[c]])
        ms = str(r['codes']['ms_per_image'][k])
        out.append(f"| {k.replace('|', ': ')} | {f(rr['clean']['0'][k], de)} | {f(m, de)} | {ms.replace('.', ',') if de else ms} |")
    out.append("")
    return out


def station_section(de: bool) -> list[str]:
    sp, cr, cs = REPORTS / "station.json", REPORTS / "calibration.json", REPORTS / "calibration_synthetic.json"
    if not sp.exists():
        return []
    st = json.loads(sp.read_text())
    num = (lambda x: str(x).replace(".", ",")) if de else str
    out = ["### Anbindung an die Steuerung (OPC UA)" if de else "### Station integration (OPC UA)", ""]
    if de:
        out += ["Simulierte SPS (OPC-UA-Server) und Prüfstation (OPC-UA-Client) mit dem Handshake einer realen "
                "Prüfzelle: Trigger → Busy → Ergebnis, Greifposition, Winkel, Fehlercode → Done → Trigger = 0. "
                "SPS-Zykluszeit 2 ms.", "",
                "| Szenario | Teile | Antwortzeit ms (Mittel / p95 / max) | Reaktion der SPS |", "|---|---|---|---|"]
        lab = {"normal": "Normalbetrieb", "camera_faults": "Kamera liefert jedes 10. Bild nicht",
               "crash_during_cycle": "Station fällt während eines Zyklus aus",
               "crash_between_parts": "Station fällt zwischen zwei Teilen aus"}
        react = {"none": "keine nötig"}
    else:
        out += ["Simulated PLC (OPC UA server) and vision station (OPC UA client) with the handshake of a real "
                "inspection cell: Trigger → Busy → result, pick position, angle, error code → Done → Trigger = 0. "
                "PLC scan time 2 ms.", "",
                "| Scenario | parts | response ms (mean / p95 / max) | PLC reaction |", "|---|---|---|---|"]
        lab = {"normal": "normal operation", "camera_faults": "camera misses every 10th image",
               "crash_during_cycle": "station crashes during a cycle", "crash_between_parts": "station crashes between parts"}
        react = {"none": "none needed"}
    for k, l in lab.items():
        v = st.get(k)
        if not v:
            continue
        rm = v["response_ms"]
        de_alarm = {"no answer within timeout: part rejected, line stopped":
                    "keine Antwort innerhalb der Überwachungszeit: Teil ausgeschleust, Linie gestoppt",
                    "station heartbeat lost: line stopped": "Lebenszeichen der Station fehlt: Linie gestoppt"}
        alarms = "; ".join(de_alarm.get(a["alarm"], a["alarm"]) if de else a["alarm"] for a in v["alarms"]) or react["none"]
        if k == "camera_faults":
            n_err = sum(n for e, n in v["errors"].items() if e != "ok")
            alarms = (f"{n_err} Teile mit Fehlercode ausgeschleust (fail-safe)" if de else
                      f"{n_err} parts rejected with an error code (fail-safe)")
        out.append(f"| {l} | {v['parts']} | {num(rm['mean'])} / {num(rm['p95'])} / {num(rm['max'])} | {alarms} |")
    n = st["normal"]
    out += ["", (f"Normalbetrieb: davon Prüfung {num(n['inspect_ms'])} ms, OPC-UA-Schreiben {num(n['write_ms'])} ms; "
                 f"{n['missed_defects']} Fehlschlupf und {n['false_rejects']} Pseudoausschuss bei {n['parts']} Teilen." if de else
                 f"Normal operation: inspection {n['inspect_ms']} ms, OPC UA writes {n['write_ms']} ms; "
                 f"{n['missed_defects']} missed defect and {n['false_rejects']} false rejects in {n['parts']} parts."), ""]
    cp = cr if cr.exists() else cs if cs.exists() else None
    if cp:
        c = json.loads(cp.read_text())
        pe = c["plane_error_mm"]
        out += ["### Kamerakalibrierung" if de else "### Camera calibration", ""]
        src = ("Webcam" if cp == cr else "Selbsttest mit gerenderten Ansichten einer bekannten Kamera") if de else \
              ("webcam" if cp == cr else "self-check with rendered views of a known camera")
        out += [(f"Quelle: {src}. Reprojektionsfehler {num(round(c['rms_reprojection_px'], 2))} px ({c['views_used']} Ansichten). "
                 f"Positionsfehler auf der Arbeitsebene: {num(pe['with_calibration']['mean'])} mm im Mittel "
                 f"(p95 {num(pe['with_calibration']['p95'])} mm) mit Linsenkorrektur, "
                 f"{num(pe['without_calibration']['mean'])} mm ohne." if de else
                 f"Source: {src}. Reprojection error {round(c['rms_reprojection_px'], 2)} px ({c['views_used']} views). "
                 f"Position error on the work plane: {pe['with_calibration']['mean']} mm mean "
                 f"(p95 {pe['with_calibration']['p95']} mm) with lens correction, {pe['without_calibration']['mean']} mm without."),
                ""]
    return out


def build(r: dict, de: bool) -> str:
    meta = r["meta"]
    if de:
        head = ["# Versuchsbericht: Vergleich von Bildverarbeitungsverfahren für die Qualitätsprüfung", "",
                f"Kaizen Vision · Vikram Ejjagiri · Versuchslauf {meta['date']} ({meta['minutes']} min, {meta['cpu']})", "",
                "## 1. Ziel", "",
                "Für eine neue Prüfstation soll ein Bildverarbeitungsverfahren ausgewählt werden. Verglichen werden "
                "klassische Bildverarbeitung, vortrainierte und feinabgestimmte neuronale Netze sowie Anomalieerkennung: "
                "Genauigkeit, Robustheit gegenüber veränderten Aufnahmebedingungen, Bedarf an Trainingsbildern und "
                "Rechenzeit. Zusätzlich werden Leser für QR-Codes, Data-Matrix-Codes und Klartext verglichen.", "",
                "## 2. Versuchsaufbau", "",
                "- **Stahloberfläche (NEU-Datenbank):** 6 Fehlerarten, 1.440 Trainings- und 360 Testbilder, 200 x 200 px.",
                "- **Gussteile (Pumpenlaufräder):** i. O. / n. i. O., 910 Trainings- und 390 Testbilder, 256 x 256 px.",
                "- **Kennzeichnungen:** je 60 erzeugte QR-, Data-Matrix- und Klartext-Markierungen auf echter Stahltextur.",
                "- **Rechner:** Laptop-CPU ohne GPU; vortrainiertes Netz YOLO11s-cls (ImageNet).", "",
                "## 3. Versuchsdurchführung", "",
                "1. Training aller Verfahren auf sauberen Bildern.",
                "2. Prüfung auf sauberen Testbildern und unter 11 veränderten Bedingungen in je 4 Stufen (44 Bedingungen): "
                "Unter- und Überbelichtung, geringer Kontrast, ungleichmäßige Beleuchtung, Fokus- und Bewegungsunschärfe, "
                "Sensorrauschen, geringere Auflösung, Drehung, Kameraneigung, JPEG-Kompression.",
                "3. Training mit 2 % bis 100 % der Trainingsbilder (je drei zufällige Teilmengen).",
                "4. Messung der Rechenzeit pro Bild und der Trainingszeit.",
                "5. Gegenmaßnahme: erneutes Training mit zufälliger Helligkeit, Unschärfe und Rauschen.",
                "6. Anbindung: Handshake mit einer simulierten SPS über OPC UA (Normalbetrieb, Kamerafehler, Ausfall "
                "der Station) und Kamerakalibrierung für die Umrechnung von Pixeln in Millimeter.", "",
                "## 4. Ergebnisse", ""]
        tail = ["## 5. Bewertung und Empfehlung", "", (REPORTS / "findings_de.md").read_text(), "",
                "## 6. Grenzen", "",
                "- Die veränderten Bedingungen sind simuliert; sie ersetzen keine Versuche an der realen Station.",
                "- Öffentliche Datensätze mit kontrollierter Aufnahme; reale Linien haben zusätzliche Störgrößen "
                "(Staub, Öl, Teilevarianten).",
                "- Die Kennzeichnungen sind erzeugt (bekannter Inhalt), die Oberfläche ist echt.",
                "- Ein Lauf mit festem Zufallswert; kleine Unterschiede (unter etwa 1 Prozentpunkt) sind nicht belastbar.",
                "- SPS und Roboter sind simuliert (OPC UA auf demselben Rechner); im Netzwerk kommen Latenzen hinzu."]
    else:
        head = ["# Test report: comparing machine vision methods for quality inspection", "",
                f"Kaizen Vision · Vikram Ejjagiri · test run {meta['date']} ({meta['minutes']} min, {meta['cpu']})", "",
                "## 1. Objective", "",
                "Select a vision method for a new inspection station. Compared: classical image processing, pretrained "
                "and fine-tuned neural networks, and anomaly detection, for accuracy, robustness to changed imaging "
                "conditions, training data needed and computing time. Readers for QR codes, Data Matrix codes and plain "
                "text are compared as well.", "",
                "## 2. Test setup", "",
                "- **Steel surface (NEU database):** 6 defect types, 1,440 training and 360 test images, 200 x 200 px.",
                "- **Castings (pump impellers):** OK / defective, 910 training and 390 test images, 256 x 256 px.",
                "- **Markings:** 60 generated QR, Data Matrix and text markings each, on real steel texture.",
                "- **Hardware:** laptop CPU without GPU; pretrained network YOLO11s-cls (ImageNet).", "",
                "## 3. Procedure", "",
                "1. Train every method on clean images.",
                "2. Test on clean images and under 11 changed conditions at 4 levels each (44 conditions): under- and "
                "overexposure, low contrast, uneven light, defocus, motion blur, sensor noise, lower resolution, rotation, "
                "camera tilt, JPEG compression.",
                "3. Train with 2 % to 100 % of the training images (three random subsets each).",
                "4. Measure computing time per image and training time.",
                "5. Counter-measure: retrain with random lighting, blur and noise variation.",
                "6. Integration: handshake with a simulated PLC over OPC UA (normal operation, camera faults, station "
                "failure) and camera calibration for converting pixels to millimetres.", "",
                "## 4. Results", ""]
        tail = ["## 5. Evaluation and recommendation", "", (REPORTS / "findings.md").read_text(), "",
                "## 6. Limitations", "",
                "- The changed conditions are simulated; they do not replace trials on the real station.",
                "- Public datasets with controlled imaging; real lines add dust, oil and part variants.",
                "- The markings are generated (known content); the surface is real.",
                "- One run with a fixed random seed; differences below about 1 point are not meaningful.",
                "- PLC and robot are simulated (OPC UA on the same computer); a real network adds latency."]
    return "\n".join(head + tables(r, de) + station_section(de) + tail) + "\n"


def main() -> None:
    p = REPORTS / "results.json"
    if not p.exists():
        p = REPORTS / "results_quick.json"
    r = json.loads(p.read_text())
    findings.main(p.name)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "test_report.md").write_text(build(r, False))
    (DOCS / "versuchsbericht.md").write_text(build(r, True))
    print("written docs/test_report.md and docs/versuchsbericht.md")


if __name__ == "__main__":
    main()
