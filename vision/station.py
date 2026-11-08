"""Stage 4: station integration. The inspection result has to reach the machine.

A simulated PLC (OPC UA server) and the vision station (OPC UA client) run the handshake used on real inspection
cells:

    PLC                                   Vision station
    ---                                   --------------
    waits for Ready = 1, Done = 0
    writes PartId, sets Trigger = 1  -->  sees Trigger rising edge
                                          Busy = 1, takes the image, inspects
                                          writes ResultOK, DefectScore, PickX/Y, Angle, ErrorCode
                                     <--  Busy = 0, Done = 1
    reads the result, sorts the part
    sets Trigger = 0                 -->  sees Trigger = 0, Done = 0 (ready for the next part)

Safety rules (as a PLC programmer would require them):
  * fail-safe: if the image cannot be taken or inspected, the station still answers, with ResultOK = 0 and an
    ErrorCode, so the part is rejected, never passed by mistake;
  * watchdog: the PLC rejects the part and raises an alarm if Done does not come within the timeout;
  * heartbeat: the station increments a counter every 200 ms; if it stops, the PLC knows the station is down
    before the next part arrives.

    python -m vision.station              # 200 parts from the casting test set, writes reports/station.json
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
NS_URI = "urn:kaizen-vision:plc"
TAGS = {  # name: (initial value, written by)
    "Trigger": (False, "PLC"), "PartId": (0, "PLC"),
    "Ready": (False, "Vision"), "Busy": (False, "Vision"), "Done": (False, "Vision"),
    "ResultOK": (False, "Vision"), "DefectScore": (0.0, "Vision"), "PickX_mm": (0.0, "Vision"),
    "PickY_mm": (0.0, "Vision"), "Angle_deg": (0.0, "Vision"), "ErrorCode": (0, "Vision"), "Heartbeat": (0, "Vision"),
}
ERRORS = {0: "ok", 1: "no image from camera", 2: "inspection failed", 3: "part not found"}


# ----------------------------------------------------------------------------------------------- inspection
@dataclass
class Inspection:
    ok: bool
    score: float
    x_mm: float
    y_mm: float
    angle_deg: float
    error: int = 0


def locate(img: np.ndarray, mm_per_px: float) -> tuple[float, float, float] | None:
    """Centre and orientation of the part: largest dark-on-light or light-on-dark blob, via Otsu threshold and
    image moments. Returns (x_mm, y_mm, angle_deg) relative to the image centre, or None."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th[0, :].mean() > 127:  # background bright -> part is the dark region
        th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.01 * img.size:
        return None
    m = cv2.moments(c)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    angle = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    h, w = img.shape
    return (cx - w / 2) * mm_per_px, (h / 2 - cy) * mm_per_px, float(angle)


class Inspector:
    """Casting OK / defective with the pretrained-feature classifier trained with lighting/blur/noise variation
    (fast on a CPU and robust in the test series). Trained once and cached in models/."""

    def __init__(self, mm_per_px: float = 0.5):
        self.mm_per_px = mm_per_px
        self.path = ROOT / "models" / "station_classifier.joblib"
        from . import deep
        self.net = deep.load_network()
        self.clf = self._load_or_train()

    def _load_or_train(self):
        import joblib
        from . import augment, datasets, deep
        if self.path.exists():
            return joblib.load(self.path)
        sp = datasets.load_casting()
        x, y = augment.augment_set(sp.x_train, sp.y_train, copies=1, seed=1)
        clf = deep.DeepFeatures(self.net).fit(x, y).model
        self.path.parent.mkdir(exist_ok=True)
        joblib.dump(clf, self.path)
        return clf

    def __call__(self, img: np.ndarray | None) -> Inspection:
        from . import deep
        if img is None:
            return Inspection(False, 1.0, 0, 0, 0, error=1)
        try:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            p = float(self.clf.predict_proba(deep.embed(self.net, g[None]))[0, 1])
            pos = locate(g, self.mm_per_px * img.shape[1] / 256)
        except Exception:
            return Inspection(False, 1.0, 0, 0, 0, error=2)
        if pos is None:
            return Inspection(False, p, 0, 0, 0, error=3)
        return Inspection(p < 0.5, p, *pos)


# ----------------------------------------------------------------------------------------------- PLC side
@dataclass
class PlcLog:
    cycles: list[dict] = field(default_factory=list)
    alarms: list[dict] = field(default_factory=list)


async def start_plc_server(port: int):
    """OPC UA server with the station's tags (the PLC's data block)."""
    from asyncua import Server, ua
    server = Server()
    await server.init()
    server.set_endpoint(f"opc.tcp://127.0.0.1:{port}/kaizen-vision/")
    server.set_server_name("Kaizen Vision simulated PLC")
    server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
    idx = await server.register_namespace(NS_URI)
    obj = await server.nodes.objects.add_object(idx, "InspectionStation")
    nodes = {}
    for name, (init, _) in TAGS.items():
        vt = (ua.VariantType.Boolean if isinstance(init, bool) else ua.VariantType.Double if isinstance(init, float)
              else ua.VariantType.Int32)
        nodes[name] = await obj.add_variable(idx, name, ua.Variant(init, vt))
        await nodes[name].set_writable()
    await server.start()
    return server, nodes


async def plc_program(nodes: dict, parts: int, log: PlcLog, timeout_ms: float = 2000, scan_ms: float = 2,
                      labels: list[int] | None = None, hb_timeout_ms: float = 1000, part_interval_ms: float = 0) -> None:
    """The PLC's sequence (like a step chain in structured text), with watchdog and heartbeat check."""
    sort = {"ok": 0, "reject": 0}
    last_hb, last_hb_change = None, time.perf_counter()
    async def heartbeat_ok() -> bool:
        nonlocal last_hb, last_hb_change
        hb = await nodes["Heartbeat"].read_value()
        if hb != last_hb:
            last_hb, last_hb_change = hb, time.perf_counter()
        return (time.perf_counter() - last_hb_change) * 1000 <= hb_timeout_ms

    next_part = time.perf_counter()
    for part in range(1, parts + 1):
        # step 0: wait for the next part on the conveyor and until the station is ready;
        # no trigger while the heartbeat is missing
        while True:
            if time.perf_counter() < next_part:
                if not await heartbeat_ok():
                    log.alarms.append({"part": part, "alarm": "station heartbeat lost: line stopped",
                                       "silent_for_ms": round((time.perf_counter() - last_hb_change) * 1000)})
                    return
                await asyncio.sleep(scan_ms / 1000)
                continue
            if not await heartbeat_ok():
                log.alarms.append({"part": part, "alarm": "station heartbeat lost: line stopped",
                                   "silent_for_ms": round((time.perf_counter() - last_hb_change) * 1000)})
                return
            if (await nodes["Ready"].read_value()) and not (await nodes["Done"].read_value()):
                break
            await asyncio.sleep(scan_ms / 1000)
        # step 1: trigger
        await nodes["PartId"].write_value(part, varianttype=_vt("Int32"))
        t0 = time.perf_counter()
        next_part = t0 + part_interval_ms / 1000
        await nodes["Trigger"].write_value(True)
        # step 2: wait for Done with watchdog
        timed_out = False
        while not await nodes["Done"].read_value():
            if (time.perf_counter() - t0) * 1000 > timeout_ms:
                timed_out = True
                break
            await asyncio.sleep(scan_ms / 1000)
        response_ms = (time.perf_counter() - t0) * 1000
        if timed_out:
            # watchdog: reject this part, stop the line and call the operator (no further triggers)
            log.alarms.append({"part": part, "alarm": "no answer within timeout: part rejected, line stopped",
                               "timeout_ms": timeout_ms})
            log.cycles.append({"part": part, "response_ms": round(response_ms, 2), "ok": False, "error": -1,
                               "score": float("nan"), "x_mm": 0.0, "y_mm": 0.0, "angle_deg": 0.0})
            await nodes["Trigger"].write_value(False)
            return
        ok = await nodes["ResultOK"].read_value()
        err = await nodes["ErrorCode"].read_value()
        score = await nodes["DefectScore"].read_value()
        x = await nodes["PickX_mm"].read_value(); y = await nodes["PickY_mm"].read_value()
        a = await nodes["Angle_deg"].read_value()
        ok = bool(ok) and err == 0           # fail-safe: any error -> reject
        sort["ok" if ok else "reject"] += 1
        row = {"part": part, "response_ms": round(response_ms, 2), "ok": ok, "error": int(err),
               "score": round(float(score), 4), "x_mm": round(x, 2), "y_mm": round(y, 2), "angle_deg": round(a, 1)}
        if labels is not None:
            row["truth_defect"] = int(labels[part - 1])
        log.cycles.append(row)
        # step 3: reset trigger, wait until the station has dropped Done
        await nodes["Trigger"].write_value(False)
        t1 = time.perf_counter()
        while await nodes["Done"].read_value():
            if (time.perf_counter() - t1) * 1000 > timeout_ms:
                log.alarms.append({"part": part, "alarm": "Done not reset"})
                return
            await asyncio.sleep(scan_ms / 1000)


def _vt(name: str):
    from asyncua import ua
    return getattr(ua.VariantType, name)


# ----------------------------------------------------------------------------------------------- vision side
async def vision_station(url: str, capture: Callable[[int], np.ndarray | None], inspect: Callable,
                         stop: asyncio.Event, timings: list, hang_after: int | None = None,
                         crash_idle_after: int | None = None) -> None:
    """OPC UA client: reacts to the PLC's trigger, inspects, writes the result.
    `hang_after`: stops answering when trigger number hang_after + 1 arrives (crash during a cycle).
    `crash_idle_after`: dies after answering that many parts, between two parts (crash while idle)."""
    from asyncua import Client, ua
    async with Client(url) as client:
        idx = await client.get_namespace_index(NS_URI)
        obj = await client.nodes.objects.get_child(f"{idx}:InspectionStation")
        n = {name: await obj.get_child(f"{idx}:{name}") for name in TAGS}

        async def heartbeat():
            k = 0
            while not stop.is_set():
                k += 1
                await n["Heartbeat"].write_value(ua.Variant(k, ua.VariantType.Int32))
                await asyncio.sleep(0.2)

        hb = asyncio.create_task(heartbeat())
        await n["Ready"].write_value(True)
        handled = 0
        try:
            while not stop.is_set():
                if not await n["Trigger"].read_value():
                    if await n["Done"].read_value():
                        await n["Done"].write_value(False)
                    await asyncio.sleep(0.001)
                    continue
                if await n["Done"].read_value():          # already answered this trigger
                    await asyncio.sleep(0.001)
                    continue
                if hang_after is not None and handled >= hang_after:
                    hb.cancel()                            # simulated crash: no answer, no heartbeat
                    await stop.wait()
                    break
                await n["Busy"].write_value(True)
                part = await n["PartId"].read_value()
                t0 = time.perf_counter()
                img = await asyncio.to_thread(capture, int(part))
                t1 = time.perf_counter()
                res: Inspection = await asyncio.to_thread(inspect, img)
                t2 = time.perf_counter()
                await n["ResultOK"].write_value(bool(res.ok))
                await n["DefectScore"].write_value(float(res.score))
                await n["PickX_mm"].write_value(float(res.x_mm))
                await n["PickY_mm"].write_value(float(res.y_mm))
                await n["Angle_deg"].write_value(float(res.angle_deg))
                await n["ErrorCode"].write_value(ua.Variant(int(res.error), ua.VariantType.Int32))
                await n["Busy"].write_value(False)
                await n["Done"].write_value(True)
                t3 = time.perf_counter()
                timings.append({"capture_ms": (t1 - t0) * 1000, "inspect_ms": (t2 - t1) * 1000,
                                "write_ms": (t3 - t2) * 1000})
                handled += 1
                if crash_idle_after is not None and handled >= crash_idle_after:
                    while await n["Trigger"].read_value():   # finish the handshake of the last part, then die
                        await asyncio.sleep(0.001)
                    await n["Done"].write_value(False)
                    hb.cancel()
                    await stop.wait()
                    break
        finally:
            hb.cancel()


async def run_line(capture, inspect, parts: int, port: int = 48410, labels=None, timeout_ms: float = 2000,
                   hang_after: int | None = None, crash_idle_after: int | None = None,
                   part_interval_ms: float = 0) -> dict:
    server, nodes = await start_plc_server(port)
    log, timings, stop = PlcLog(), [], asyncio.Event()
    url = f"opc.tcp://127.0.0.1:{port}/kaizen-vision/"
    vis = asyncio.create_task(vision_station(url, capture, inspect, stop, timings, hang_after, crash_idle_after))
    try:
        await asyncio.sleep(0.3)
        await plc_program(nodes, parts, log, timeout_ms=timeout_ms, labels=labels, part_interval_ms=part_interval_ms)
    finally:
        stop.set()
        try:
            await asyncio.wait_for(vis, 3)
        except Exception:
            vis.cancel()
        await server.stop()
    return {"cycles": log.cycles, "alarms": log.alarms, "timings": timings}


# ----------------------------------------------------------------------------------------------- experiment
def summarise(run: dict) -> dict:
    c = run["cycles"]
    rt = np.array([r["response_ms"] for r in c]) if c else np.array([np.nan])
    out = {"parts": len(c), "alarms": run["alarms"],
           "response_ms": {"mean": round(float(np.mean(rt)), 1), "p95": round(float(np.percentile(rt, 95)), 1),
                           "max": round(float(np.max(rt)), 1)}}
    if run["timings"]:
        for k in ("capture_ms", "inspect_ms", "write_ms"):
            out[k] = round(float(np.mean([t[k] for t in run["timings"]])), 1)
    if c and "truth_defect" in c[0]:
        truth = np.array([r["truth_defect"] for r in c]); passed = np.array([r["ok"] for r in c])
        out["accuracy"] = round(float(np.mean(passed != truth.astype(bool))), 4)
        out["missed_defects"] = int(np.sum(passed & (truth == 1)))
        out["false_rejects"] = int(np.sum(~passed & (truth == 0)))
    names = {**ERRORS, -1: "timeout (no answer)"}
    out["errors"] = {names.get(e, str(e)): int(sum(r["error"] == e for r in c)) for e in {r["error"] for r in c}}
    return out


def main(parts: int = 200) -> dict:
    from . import datasets
    sp = datasets.load_casting()
    idx = np.random.default_rng(0).permutation(len(sp.x_test))[:parts]
    imgs, labels = sp.x_test[idx], sp.y_test[idx]
    inspector = Inspector(mm_per_px=0.5)   # assumption for the dataset images: 128 mm field of view over 256 px
    inspector(imgs[0])                      # warm-up

    def capture(part: int):
        return imgs[(part - 1) % len(imgs)]

    def capture_with_faults(part: int):
        return None if part % 10 == 0 else imgs[(part - 1) % len(imgs)]   # every 10th image missing

    res = {}
    res["normal"] = summarise(asyncio.run(run_line(capture, inspector, parts, labels=list(labels))))
    res["camera_faults"] = summarise(asyncio.run(run_line(capture_with_faults, inspector, 50, port=48411,
                                                          labels=list(labels[:50]))))
    res["crash_during_cycle"] = summarise(asyncio.run(run_line(capture, inspector, 20, port=48412, hang_after=10,
                                                               timeout_ms=1000)))
    res["crash_between_parts"] = summarise(asyncio.run(run_line(capture, inspector, 20, port=48413, crash_idle_after=5,
                                                                part_interval_ms=1500)))
    res["meta"] = {"date": time.strftime("%Y-%m-%d %H:%M"), "transport": "OPC UA (asyncua), localhost",
                   "plc_scan_ms": 2, "classifier": "pretrained CNN features + logistic regression, trained with variation",
                   "mm_per_px_assumed": 0.5}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "station.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
