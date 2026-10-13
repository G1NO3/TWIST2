#!/usr/bin/env python3
"""Sensor-only TWIST2/GMR diagnostic. NEVER publishes robot commands.

Uses the same XRobotStreamer coordinate conversion and GMR configuration as
xrobot_teleop_to_robot_w_hand.py, without its Redis, hands or robot transport.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

# XRoboToolkit Body ordering, also used by GMR's XRobotStreamer.
BODY_NAMES = (
    "Pelvis Left_Hip Right_Hip Spine1 Left_Knee Right_Knee Spine2 Left_Ankle "
    "Right_Ankle Spine3 Left_Foot Right_Foot Neck Left_Collar Right_Collar Head "
    "Left_Shoulder Right_Shoulder Left_Elbow Right_Elbow Left_Wrist Right_Wrist "
    "Left_Hand Right_Hand"
).split()


def valid_pose(pose):
    p = np.asarray(pose, dtype=float)
    return p.shape == (7,) and np.isfinite(p).all() and .9 < p[3:] @ p[3:] < 1.1


def json_safe(value):
    """Preserve bad-input evidence as null without emitting invalid JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def read_sample(sdk):
    """Best-effort snapshot; SDK does not expose an atomic multi-field read."""
    stamp = int(sdk.get_time_stamp_ns())
    controllers = {s: list(getattr(sdk, f"get_{s}_controller_pose")())
                   for s in ("left", "right")}
    body = None
    body_clock = None
    if sdk.is_body_data_available():
        body = np.asarray(sdk.get_body_joints_pose(), dtype=float)
        body_clock = (int(sdk.get_body_timestamp_ns()),
                      *map(int, sdk.get_body_joints_timestamp()))
    if stamp <= 0 or stamp != int(sdk.get_time_stamp_ns()):
        return None
    return stamp, controllers, body, body_clock


def human_data(body, transform, required):
    if body is None or body.shape != (24, 7):
        raise ValueError("Body unavailable: enable Body Tracking and finish calibration")
    raw = {}
    for name in required:
        p = body[BODY_NAMES.index(name)]
        if not valid_pose(p):
            raise ValueError(f"invalid Body joint: {name}")
        raw[name] = [p[:3].tolist(), p[[6, 3, 4, 5]].tolist()]
    return transform(raw)


class Displacements:
    def __init__(self):
        self.anchors = {}

    def measure(self, key, xyz):
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("nonfinite XYZ")
        anchor = self.anchors.setdefault(key, xyz.copy())
        delta = 1000 * (xyz - anchor)
        return {"xyz_mm": delta.tolist(), "norm_mm": float(np.linalg.norm(delta))}


def pico_reader_script(argv):
    """Identify known SDK readers, excluding this checkout's import-only checks.

    Do not exempt arbitrary commands just because they contain --preflight:
    legacy teleop (or another checkout) may not implement an SDK-free check.
    """
    args = [os.fsdecode(a) for a in argv if a]
    preflight_scripts = {
        "check_pico_gmr.py", "pico_intervention_source.py",
        "pico_gmr_intervention_source.py",
    }
    for i, arg in enumerate(args):
        name = Path(arg).name
        if name not in preflight_scripts | {"xrobot_teleop_to_robot_w_hand.py"}:
            continue
        # Only exempt an explicit Python script invocation (or a directly
        # executed script), not shell command text or a flag before the script.
        script_invocation = i == 0 or (
            i == 1 and Path(args[0]).name.startswith("python"))
        if (script_invocation and name in preflight_scripts
                and Path(arg).is_absolute()
                and Path(arg).resolve() == Path(__file__).resolve().with_name(name)
                and "--preflight" in args[i + 1:]):
            continue
        return name
    return None


def assert_no_other_pico_readers(proc_root=Path("/proc")):
    # This supplements the output-file lock for legacy teleop/different paths;
    # it is not an atomic SDK-wide lock shared by every possible reader.
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = proc.joinpath("cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        script = pico_reader_script(argv)
        if script:
            raise RuntimeError(
                f"Other PICO reader PID {proc.name} ({script}); "
                "end its session safely first")


def acquire_input_lock(stack):
    # Share the production source's lock, but never write its state file.
    path = Path("/tmp/policy_bridge/pico_state.json.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = stack.enter_context(path.open("a"))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("PICO input is in use. Safely END SESSION in the GUI first; "
                           "this diagnostic will not stop any processes.") from None
    assert_no_other_pico_readers()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--hz", type=float, default=30)
    ap.add_argument("--human-height", type=float, default=1.6,
                    help="TWIST2 teleop.sh default; use the same value as the other PC")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--preflight", action="store_true", help="Load dependencies/model; never initialize SDK")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if not (np.isfinite([args.seconds, args.hz, args.human_height]).all()
            and args.seconds > 0 and 1 <= args.hz <= 100 and .5 <= args.human_height <= 2.5):
        ap.error("invalid duration, rate or human height")

    import mujoco
    import xrobotoolkit_sdk as sdk
    from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer, XRobotStreamer

    retarget = GeneralMotionRetargeting("xrobot", "unitree_g1",
                                      actual_human_height=args.human_height, verbose=False)
    print(f"SDK module: {sdk.__file__}", flush=True)
    print(f"GMR model: {retarget.xml_file}", flush=True)
    if args.preflight:
        print("PREFLIGHT OK: model/dependencies loaded; no SDK connection or robot I/O.")
        return 0
    if not args.headless and not os.environ.get("DISPLAY"):
        ap.error("No DISPLAY; run from a desktop terminal or add --headless")

    output = args.output or Path(__file__).resolve().parents[1] / "logs" / (
        "pico_gmr_compare_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
    reset_requested = [False]
    def key_callback(key):
        if key == ord("R"):
            reset_requested[0] = True

    with ExitStack() as stack:
        acquire_input_lock(stack)
        output.parent.mkdir(parents=True, exist_ok=True)
        log = stack.enter_context(output.open("x", buffering=1))
        def record(data):
            log.write(json.dumps(json_safe(data), allow_nan=False) + "\n")
        record(dict(event="start", sdk=sdk.__file__, model=retarget.xml_file,
                    human_height=args.human_height, hz=args.hz, wall_ns=time.time_ns(),
                    body_names=BODY_NAMES, sensor_only=True))
        viewer = None
        if not args.headless:
            viewer = RobotMotionViewer("unitree_g1", motion_fps=args.hz,
                                       keyboard_callback=key_callback)
            stack.callback(viewer.close)
        streamer = XRobotStreamer()  # only SDK init, no robot or Redis
        stack.callback(sdk.close)
        if list(streamer.body_joint_names) != BODY_NAMES:
            raise RuntimeError("GMR Body joint ordering changed; check diagnostic mapping")
        print(f"SENSOR ONLY — no robot commands. Log: {output}", flush=True)
        print("Do NOT press X/A. Move one hand ~10 cm, hold, return; then the other. "
              "R in viewer resets displacement baselines; Ctrl+C exits.", flush=True)
        measure = Displacements()
        start = time.monotonic()
        last_stamp = None
        last_body_clock = None
        last_body_update = float("-inf")
        last_packet_update = start
        last_print = float("-inf")
        count = 0
        segment = 0
        try:
            while time.monotonic() - start < args.seconds:
                tick = time.monotonic()
                if viewer and not viewer.viewer.is_running():
                    break
                sample = read_sample(sdk)
                if sample is None or sample[0] == last_stamp:
                    if tick - last_print > 1:
                        print(f"WAIT: no advancing SDK packet ({tick-last_packet_update:.1f}s)", flush=True)
                        record(dict(event="wait", reason="no advancing SDK packet",
                                    t_s=tick-start, packet_age_s=tick-last_packet_update))
                        last_print = tick
                    if viewer:
                        viewer.viewer.sync()
                    time.sleep(1 / args.hz)
                    continue
                stamp, controllers, body, body_clock = sample
                if reset_requested[0] or (last_stamp is not None and
                        (stamp < last_stamp or tick - last_packet_update > .5)):
                    measure = Displacements()
                    segment += 1
                    record(dict(event="baseline_reset", segment=segment, wall_ns=time.time_ns()))
                    reset_requested[0] = False
                    last_body_clock = None
                last_stamp, last_packet_update = stamp, tick
                if body_clock is not None and body_clock != last_body_clock:
                    last_body_clock, last_body_update = body_clock, tick
                rec = dict(event="sample", t_s=tick-start, wall_ns=time.time_ns(),
                           source_ns=stamp, segment=segment, controllers_raw=controllers,
                           body_raw=None if body is None else body.tolist(),
                           body_clock=body_clock, deltas={})
                for side, pose in controllers.items():
                    if valid_pose(pose):
                        rec["deltas"][f"controller_{side}"] = measure.measure(f"controller_{side}", pose[:3])
                try:
                    if tick - last_body_update > .5:
                        raise ValueError("Body timestamps are not advancing")
                    human = human_data(body, streamer.coordinate_transform_unity_data,
                                       retarget.human_scale_table)
                    for side, idx in (("left", 20), ("right", 21)):
                        for key, xyz in ((f"body_{side}", body[idx, :3]),
                                         (f"body_pelvis_relative_{side}", body[idx, :3]-body[0, :3])):
                            rec["deltas"][key] = measure.measure(key, xyz)
                    qpos = retarget.retarget(human, offset_to_ground=True)
                    if not np.isfinite(qpos).all():
                        raise ValueError("nonfinite GMR output")
                    mujoco.mj_forward(retarget.model, retarget.configuration.data)
                    for side in ("left", "right"):
                        xyz = retarget.configuration.data.body(f"{side}_wrist_yaw_link").xpos.copy()
                        rec["deltas"][f"gmr_{side}"] = measure.measure(f"gmr_{side}", xyz)
                    rec["qpos"] = qpos.tolist()
                    rec["gmr_ok"] = True
                    count += 1
                    if viewer:
                        viewer.step(qpos[:3], qpos[3:7], qpos[7:],
                                    human_motion_data=retarget.scaled_human_data,
                                    rate_limit=False)
                except ValueError as exc:
                    rec["gmr_ok"] = False
                    rec["reason"] = str(exc)
                    if viewer:
                        viewer.viewer.sync()
                record(rec)
                if tick-last_print >= .5:
                    parts = [f"{k}={v['norm_mm']:.1f}mm" for k, v in rec["deltas"].items()
                             if not k.startswith("body_pelvis")]
                    print(" | ".join(parts) + ("" if rec["gmr_ok"] else " | WAIT: " + rec["reason"]), flush=True)
                    last_print = tick
                time.sleep(max(0., 1 / args.hz - (time.monotonic()-tick)))
        except KeyboardInterrupt:
            pass
        record(dict(event="end", gmr_samples=count, elapsed_s=time.monotonic()-start))
        print(f"Finished: {count} GMR samples. Log: {output}", flush=True)
        return 0 if count else 2


if __name__ == "__main__":
    raise SystemExit(main())
