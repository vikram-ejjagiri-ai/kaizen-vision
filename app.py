"""Kaizen Vision dashboard:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vision import degrade

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
st.set_page_config(page_title="Kaizen Vision", page_icon="🔍", layout="wide")

NAMES = {
    "classical": "Classical features + SVM",
    "features": "Pretrained CNN features + linear classifier",
    "finetuned": "Fine-tuned CNN",
    "anomaly": "Anomaly detection (good parts only)",
    "classical_robust": "Classical, trained with variation",
    "features_robust": "CNN features, trained with variation",
    "finetuned_robust": "Fine-tuned CNN, trained with variation",
}
BASE = ["classical", "features", "finetuned", "anomaly"]
COLORS = {"classical": "#8c6d1f", "features": "#2f6b8f", "finetuned": "#2f6b1f", "anomaly": "#8f2f5a",
          "classical_robust": "#c9a54a", "features_robust": "#79a8c9", "finetuned_robust": "#8cc06b"}
TASKS = {"steel": "Steel surface: which of 6 defects?", "casting": "Cast impeller: OK or defective?"}
DEG_LABEL = {k: k.replace("_", " ") for k in degrade.DEGRADATIONS}


@st.cache_data
def load_results() -> dict | None:
    for name in ("results.json", "results_quick.json"):
        p = REPORTS / name
        if p.exists():
            return json.loads(p.read_text())
    return None


R = load_results()
st.title("🔍 Kaizen Vision")
st.caption("Comparing machine vision methods for quality inspection: which method, under which conditions, "
           "with how much data, how fast. Real public datasets, controlled test conditions.")
if R is None:
    st.error("No results yet. Run `python -m vision.experiments` (or `--quick`) first.")
    st.stop()
if R["meta"]["mode"] == "quick":
    st.warning("Showing the quick test run (fewer images and conditions). Run the full test series for final numbers.")

tabs = st.tabs(["Overview", "1 · Robustness", "2 · Training data", "3 · Speed & cost", "4 · Part markings",
                "5 · Camera check", "6 · Station integration", "Method"])


def acc_table(task: str, names: list[str]) -> pd.DataFrame:
    rob = R[task]["robustness"]
    rows = []
    for kind, levels in rob.items():
        for lv, row in levels.items():
            for n in names:
                if n in row:
                    rows.append({"condition": kind, "level": int(lv), "method": n, **row[n]})
    return pd.DataFrame(rows)


def present(task: str, names: list[str]) -> list[str]:
    return [n for n in names if n in R[task]["robustness"]["clean"]["0"]]


# ================================================================================================ overview
with tabs[0]:
    S = R["summary"]
    c1, c2, c3 = st.columns(3)
    for col, task in zip((c1, c2), ("steel", "casting")):
        best_clean = max(v["clean"] for v in S[task].values())
        best = max(present(task, BASE), key=lambda n: S[task][n]["mean_accuracy_degraded"])
        col.metric(TASKS[task], f"{best_clean:.1%} best clean accuracy",
                   f"most robust: {NAMES[best]} ({S[task][best]['mean_accuracy_degraded']:.1%} under changed conditions)",
                   delta_color="off")
    rr = R["codes"]["read_rate"]["clean"]["0"]
    c3.metric("Part markings (clean)", f"{np.mean(list(rr.values())):.0%} mean read rate",
              f"{len(rr)} reader / code combinations", delta_color="off")

    st.subheader("Clean images versus changed conditions")
    st.write("On clean images, as in a well-set-up lab, almost every method is near-perfect. The differences appear "
             "when something on the line changes: lighting, focus, motion, camera position. Each bar shows the mean "
             "accuracy over all 44 degraded test conditions.")
    cols = st.columns(2)
    for col, task in zip(cols, TASKS):
        names = present(task, list(NAMES))
        fig = go.Figure()
        fig.add_bar(y=[NAMES[n] for n in names], x=[S[task][n]["clean"] for n in names], name="clean images",
                    orientation="h", marker_color="#c8c8c8", text=[f"{S[task][n]['clean']:.0%}" for n in names],
                    textposition="inside")
        fig.add_bar(y=[NAMES[n] for n in names], x=[S[task][n]["mean_accuracy_degraded"] for n in names],
                    name="changed conditions (mean)", orientation="h",
                    marker_color=[COLORS[n] for n in names],
                    text=[f"{S[task][n]['mean_accuracy_degraded']:.0%}" for n in names], textposition="inside")
        fig.update_layout(title=TASKS[task], barmode="group", height=130 + 55 * len(names), xaxis_tickformat=".0%",
                          xaxis_range=[0, 1], yaxis_autorange="reversed", margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.12))
        col.plotly_chart(fig, use_container_width=True)
    p = REPORTS / "findings.md"
    if p.exists():
        st.subheader("Findings and recommendation")
        st.markdown(p.read_text())

# ================================================================================================ robustness
with tabs[1]:
    task = st.radio("Task", list(TASKS), format_func=TASKS.get, horizontal=True, key="rob_task")
    show_robust = st.toggle("Also show the variants trained with lighting, blur and noise variation", value=False)
    names = present(task, BASE + (["classical_robust", "features_robust", "finetuned_robust"] if show_robust else []))
    df = acc_table(task, names)
    clean = df[df.condition == "clean"].set_index("method")["accuracy"]

    st.markdown("**Tolerance:** highest severity level (0-4) at which a method stays within 5 points of its clean "
                "accuracy. 4 = robust to everything tested, 0 = already fails at the mildest level.")
    tol = pd.DataFrame({NAMES[n]: R["summary"][task][n]["tolerance_levels"] for n in names}).T
    tol.columns = [DEG_LABEL[c] for c in tol.columns]
    fig = px.imshow(tol, text_auto=True, color_continuous_scale=["#b2182b", "#f4a582", "#f7f7f7", "#a6dba0", "#1b7837"],
                    zmin=0, zmax=4, aspect="auto")
    fig.update_layout(height=120 + 45 * len(names), margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    kind = st.selectbox("Condition", list(degrade.DEGRADATIONS), format_func=lambda k:
                        f"{DEG_LABEL[k]}: {degrade.DEGRADATIONS[k][0]}")
    sim, unit, values = degrade.DEGRADATIONS[kind]
    sub = pd.concat([df[df.condition == kind], df[df.condition == "clean"].assign(condition=kind)])
    sub = sub.sort_values("level")
    sub["setting"] = sub["level"].map(lambda lv: "clean" if lv == 0 else f"L{lv}: {values[lv - 1]}")
    fig = px.line(sub, x="setting", y="accuracy", color="method", markers=True,
                  color_discrete_map=COLORS, labels={"setting": unit, "accuracy": "accuracy"})
    fig.for_each_trace(lambda t: t.update(name=NAMES[t.name]))
    fig.update_layout(height=380, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    if task == "casting":
        st.caption("For OK/defective, accuracy hides which way the errors go. Missed defects reach the customer; false "
                   "rejects cost scrap and rework.")
        piv = sub.pivot_table(index="setting", columns="method", values=["missed_defects", "false_rejects"])
        piv.columns = [f"{NAMES[m]} · {k.replace('_', ' ')}" for k, m in piv.columns]
        st.dataframe(piv.style.format("{:.0%}"), use_container_width=True)

    ex = ROOT / "data" / ("neu" if task == "steel" else "casting")
    if ex.exists():
        import cv2
        from vision import datasets
        sp_files = sorted(ex.rglob("*.jp*g"))
        img = cv2.imread(str(sp_files[len(sp_files) // 3]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (datasets.SIZES[task],) * 2)
        st.image([img] + [degrade.apply(img, kind, lv) for lv in degrade.LEVELS], width=150,
                 caption=["clean"] + [f"L{lv}: {values[lv - 1]}" for lv in degrade.LEVELS])

# ================================================================================================ data needs
with tabs[2]:
    task = st.radio("Task", list(TASKS), format_func=TASKS.get, horizontal=True, key="data_task")
    dn = R[task]["data_needs"]
    rows = [{"images per class": v["images_per_class"], "method": NAMES[m], "accuracy": v["accuracy"],
             "worst seed": v["min"]} for share, row in dn.items() for m, v in row.items()]
    df = pd.DataFrame(rows)
    fig = px.line(df, x="images per class", y="accuracy", color="method", markers=True, log_x=True,
                  color_discrete_map={NAMES[k]: c for k, c in COLORS.items()})
    ticks = sorted(df["images per class"].unique())
    fig.update_layout(height=400, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(tickvals=ticks, ticktext=[str(t) for t in ticks]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mean of three random training subsets (one for the fine-tuned network). Test set always the same.")
    if task == "casting" and "data_needs_anomaly" in R[task]:
        st.markdown("**Anomaly detection needs only good parts.** No defect images are needed for training, only for "
                    "checking the result.")
        an = R[task]["data_needs_anomaly"]
        st.dataframe(pd.DataFrame({"good images for training": [int(k) for k in an],
                                   "AUROC": [v["auroc"] for v in an.values()],
                                   "missed defects": [v["missed_defects"] for v in an.values()],
                                   "false rejects": [v["false_rejects"] for v in an.values()]})
                     .style.format({"AUROC": "{:.3f}", "missed defects": "{:.1%}", "false rejects": "{:.1%}"}),
                     hide_index=True)
    ae = REPORTS / "anomaly_examples.npz"
    if task == "casting" and ae.exists():
        z = np.load(ae)
        st.markdown("**Where the anomaly detector looks** (red = unusual compared with good parts)")
        import cv2
        cols = st.columns(len(z["images"]))
        for c, im, m, lab in zip(cols, z["images"], z["maps"], z["labels"]):
            heat = cv2.resize(m, im.shape[::-1], interpolation=cv2.INTER_CUBIC)
            heat = np.clip(heat / (2 * float(z["threshold"])), 0, 1)
            col = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
            over = (0.55 * np.dstack([im] * 3) + 0.45 * col).astype(np.uint8)
            c.image(over, caption=("defective" if lab else "OK") + f" · score {m.max():.2f}")

# ================================================================================================ speed
with tabs[3]:
    rows = []
    for task in TASKS:
        for m, v in R[task]["speed"].items():
            rows.append({"task": TASKS[task], "method": NAMES[m], "ms per image (CPU)": v["ms_per_image"],
                         "training time (s)": v["train_seconds"], "model": str(v.get("parameters", ""))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"Measured on {R['meta']['cpu']}, batches of 32 images, no GPU. A production line with 1 part per second "
               "has a budget of 1,000 ms per part for image capture, inspection and the result signal.")

# ================================================================================================ codes
with tabs[4]:
    cr = R["codes"]
    rows = []
    for kind, levels in cr["read_rate"].items():
        for lv, row in levels.items():
            for key, v in row.items():
                rows.append({"condition": kind, "level": int(lv), "reader": key.replace("|", ": "), "read rate": v})
    df = pd.DataFrame(rows)
    mean = df[df.condition != "clean"].groupby("reader")["read rate"].mean()
    clean = df[df.condition == "clean"].set_index("reader")["read rate"]
    st.dataframe(pd.DataFrame({"clean": clean, "mean under changed conditions": mean,
                               "ms per image": {k.replace("|", ": "): v for k, v in cr["ms_per_image"].items()}})
                 .style.format({"clean": "{:.0%}", "mean under changed conditions": "{:.0%}", "ms per image": "{:.1f}"}),
                 use_container_width=True)
    kind = st.selectbox("Condition ", list(degrade.DEGRADATIONS), format_func=DEG_LABEL.get, key="code_kind")
    sub = pd.concat([df[df.condition == kind], df[df.condition == "clean"]]).sort_values("level")
    vals = degrade.DEGRADATIONS[kind][2]
    fig = px.line(sub, x="level", y="read rate", color="reader", markers=True)
    fig.update_layout(height=360, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(tickvals=[0, 1, 2, 3, 4], ticktext=["clean"] + [f"L{i + 1}: {v}" for i, v in enumerate(vals)],
                                 title=degrade.DEGRADATIONS[kind][1]))
    st.plotly_chart(fig, use_container_width=True)
    ce = REPORTS / "code_examples.npz"
    if ce.exists():
        z = np.load(ce)
        st.image([z[k] for k in z.files], width=200, caption=list(z.files))
    st.caption(f"{cr['n_per_kind']} generated markings per type with known content on real steel-surface crops.")

# ================================================================================================ camera check
with tabs[5]:
    st.markdown("Translate the measured tolerances into camera and lighting requirements for a station. Enter the "
                "station data; the tolerances come from the test series (best method of the chosen task).")
    task = st.radio("Task", list(TASKS), format_func=TASKS.get, horizontal=True, key="cam_task")
    c1, c2, c3 = st.columns(3)
    fov = c1.number_input("Field of view (mm across the image)", 10.0, 2000.0, 100.0, step=10.0)
    px_img = c2.number_input("Pixels across the image (working size)", 64, 8000, 256 if task == "casting" else 200)
    speed = c3.number_input("Part speed on the conveyor (mm/s)", 0.0, 5000.0, 200.0, step=50.0)
    mm_px = fov / px_img
    S = R["summary"][task]
    best = max(present(task, BASE), key=lambda n: S[n]["mean_accuracy_degraded"])
    tl = S[best]["tolerance_levels"]

    def tol(kind):
        """(tolerated value, True) or (mildest tested value, False) if already the mildest level failed."""
        lv = tl.get(kind, 0)
        vals = degrade.DEGRADATIONS[kind][2]
        return (vals[lv - 1], True) if lv else (vals[0], False)

    (mb, mb_ok), (res_, res_ok), (tilt, tilt_ok), (dark, dark_ok) = (tol(k) for k in (
        "motion_blur", "low_resolution", "camera_tilt", "underexposure"))
    st.markdown(f"Best method for this task: **{NAMES[best]}** · resolution {mm_px:.3f} mm per pixel")
    blur_px = mb if mb_ok else 1.0  # if even 3 px failed, keep motion blur below about 1 px
    t_exp = blur_px * mm_px / speed * 1e6 if speed > 0 else float("inf")
    rows = [
        ("Exposure time", f"≤ {t_exp:,.0f} µs" if speed > 0 else "no limit (part stands still)",
         (f"motion blur tolerated up to {mb:.0f} px" if mb_ok else f"already {mb:.0f} px motion blur failed, "
          "so blur is kept below about 1 px") + "; blur in px = speed × exposure time / (mm per pixel)"),
        ("Camera resolution", f"≥ {int(np.ceil(px_img * res_))} px across the field of view" if res_ok else
         f"≥ {px_img} px across the field of view (no reduction tolerated)",
         f"accuracy kept down to {res_:.0%} of the working resolution" if res_ok else
         f"already {res_:.0%} of the working resolution failed"),
        ("Mounting angle", f"≤ {tilt:.0f}° from perpendicular" if tilt_ok else f"< {tilt:.0f}°, mount perpendicular",
         "camera tilt tolerance"),
        ("Lighting", f"brightness may fall to {dark:.0%}" if dark_ok else
         f"keep brightness constant (a drop to {dark:.0%} already failed): stable light, automatic exposure",
         "underexposure tolerance"),
    ]
    st.table(pd.DataFrame(rows, columns=["Requirement", "Value", "Reason"]))
    st.caption("A first estimate for selecting a camera, lens and light; confirm with trial images on the real station.")

# ================================================================================================ method
with tabs[6]:
    st.markdown("**From result to machine.** The inspection result only matters if it reaches the PLC or robot. A "
                "simulated PLC (OPC UA server) and the vision station (OPC UA client) run the handshake of a real "
                "inspection cell; the station also sends the pick position and angle for a robot.")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f2f2f2", fontname="Helvetica", fontsize=11];
      edge [fontname="Helvetica", fontsize=10];
      plc [label="PLC (simulated)\nOPC UA server", fillcolor="#e3eef7"];
      cam [label="Vision station\nOPC UA client", fillcolor="#e8f3e3"];
      robot [label="Robot\n(pick X, Y, angle)", fillcolor="#f7ece3"];
      plc -> cam [label="1  PartId, Trigger = 1"];
      cam -> plc [label="2  Busy, then ResultOK, DefectScore,\nPickX/Y mm, Angle, ErrorCode, Done = 1"];
      plc -> cam [label="3  Trigger = 0  →  Done = 0"];
      cam -> plc [label="Heartbeat every 200 ms", style=dashed];
      plc -> robot [label="pick / sort"];
    }""")
    sp = REPORTS / "station.json"
    if sp.exists():
        stn = json.loads(sp.read_text())
        labels = {"normal": "Normal operation", "camera_faults": "Camera fails on every 10th part",
                  "crash_during_cycle": "Station crashes during a cycle", "crash_between_parts": "Station crashes between parts"}
        rows = []
        for k, lab in labels.items():
            v = stn.get(k)
            if not v:
                continue
            rows.append({"scenario": lab, "parts handled": v["parts"],
                         "response ms (mean / p95 / max)": f"{v['response_ms']['mean']} / {v['response_ms']['p95']} / {v['response_ms']['max']}",
                         "errors": ", ".join(f"{e}: {n}" for e, n in v["errors"].items()),
                         "PLC reaction": "; ".join(a["alarm"] for a in v["alarms"]) or "none needed"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        n = stn["normal"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trigger → Done (mean)", f"{n['response_ms']['mean']} ms")
        c2.metric("of which inspection", f"{n['inspect_ms']} ms")
        c3.metric("OPC UA writes", f"{n['write_ms']} ms")
        c4.metric("Correct decisions", f"{n.get('accuracy', 0):.1%}", f"{n.get('missed_defects', 0)} missed, "
                  f"{n.get('false_rejects', 0)} false rejects of {n['parts']}", delta_color="off")
        st.caption("Fail-safe: a missing image or failed inspection is answered with ResultOK = 0 and an error code, "
                   "so the part is rejected, never passed. Watchdog: no answer within the timeout stops the line. "
                   "Heartbeat: a silent station is noticed before the next part is triggered. "
                   f"Run {stn['meta']['date']}, {stn['meta']['transport']}, PLC scan {stn['meta']['plc_scan_ms']} ms.")
    else:
        st.info("Run `python -m vision.station` to measure the handshake.")

    st.subheader("Camera calibration: pixels to millimetres")
    real, syn = REPORTS / "calibration.json", REPORTS / "calibration_synthetic.json"
    cal_path = real if real.exists() else syn if syn.exists() else None
    if cal_path:
        cal = json.loads(cal_path.read_text())
        if cal_path == syn:
            st.caption("Self-check with rendered views of a known camera. Calibrate your own webcam with "
                       "`python -m vision.calibrate capture` and `run`; the results then appear here.")
        st.markdown(f"Source: {cal['source']}")
        pe = cal["plane_error_mm"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Reprojection error", f"{cal['rms_reprojection_px']:.2f} px", f"{cal['views_used']} views",
                  delta_color="off")
        c2.metric("Position error on the work plane", f"{pe['with_calibration']['mean']:.2f} mm",
                  f"p95 {pe['with_calibration']['p95']:.2f} mm", delta_color="off")
        c3.metric("Without lens correction", f"{pe['without_calibration']['mean']:.2f} mm",
                  f"p95 {pe['without_calibration']['p95']:.2f} mm", delta_color="off")
        if "error_vs_truth" in cal:
            st.dataframe(pd.DataFrame([cal["error_vs_truth"]]), hide_index=True)
        ex = REPORTS / "calibration_synthetic_example.png"
        if cal_path == syn and ex.exists():
            st.image(str(ex), width=480, caption="one of the rendered calibration views")

with tabs[7]:
    st.markdown((ROOT / "docs" / "method.md").read_text() if (ROOT / "docs" / "method.md").exists() else "")
    st.markdown("**Test conditions**")
    st.dataframe(pd.DataFrame([{"condition": DEG_LABEL[k], "simulates": v[0], "unit": v[1],
                                "levels 1-4": ", ".join(str(x) for x in v[2])} for k, v in degrade.DEGRADATIONS.items()]),
                 hide_index=True, use_container_width=True)
    st.caption(f"Results from {R['meta']['date']} ({R['meta']['mode']} run, {R['meta']['minutes']} min).")
