"""Body -> virtual G1 -> pelvis-relative wrists. Sensor-only, no robot I/O."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from check_pico_gmr import assert_no_other_pico_readers, human_data
from pico_intervention_source import DEFAULT_FILE, write_packet


def pelvis_relative_pose(data, side):
    """Remove the virtual free-root translation AND orientation before clutching."""
    root = data.body("pelvis")
    wrist = data.body(f"{side}_wrist_yaw_link")
    R = np.asarray(root.xmat).reshape(3, 3)
    p = R.T @ (wrist.xpos - root.xpos)
    r = R.T @ np.asarray(wrist.xmat).reshape(3, 3)
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("nonfinite virtual FK")
    return np.r_[p, Rotation.from_matrix(r).as_quat()].tolist()


def read_body_and_buttons(sdk):
    """Body path deliberately does NOT require native controller pose tracking."""
    stamp = int(sdk.get_time_stamp_ns())
    if not sdk.is_body_data_available():
        raise ValueError("Body unavailable: wear headset/trackers and calibrate Body Tracking")
    body = np.asarray(sdk.get_body_joints_pose(), dtype=float)
    clock = (int(sdk.get_body_timestamp_ns()), *map(int, sdk.get_body_joints_timestamp()))
    if len(clock) != 25 or min(clock) < 0 or max(clock) <= 0:
        raise ValueError("invalid Body clocks")
    hands = {}
    for side, one, two in (("left", "X", "Y"), ("right", "A", "B")):
        axis = list(getattr(sdk, f"get_{side}_axis")())
        grip = float(getattr(sdk, f"get_{side}_grip")())
        trigger = float(getattr(sdk, f"get_{side}_trigger")())
        if (len(axis) != 2 or not np.isfinite(axis + [grip, trigger]).all()
                or max(map(abs, axis)) > 1 or not 0 <= grip <= 1 or not 0 <= trigger <= 1):
            raise ValueError("invalid controller buttons/levers")
        hands[side] = dict(axis=axis, grip=grip, index_trig=trigger,
            key_one=bool(getattr(sdk, f"get_{one}_button")()),
            key_two=bool(getattr(sdk, f"get_{two}_button")()),
            axis_click=bool(getattr(sdk, f"get_{side}_axis_click")()))
    if stamp <= 0 or stamp != int(sdk.get_time_stamp_ns()):
        return None
    return stamp, body, clock, hands


class FreshBody:
    """Global button traffic must never keep a frozen Body pose alive."""
    def __init__(self):
        self.stamp = 0
        self.clock = None

    def accept(self, stamp, clock):
        if stamp < self.stamp or (self.clock is not None and
                any(a < b for a, b in zip(clock, self.clock))):
            raise ValueError("SDK/Body clock regressed; restart source after recalibration")
        if stamp == self.stamp or clock == self.clock:
            return False
        self.stamp, self.clock = stamp, clock
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path(DEFAULT_FILE))
    ap.add_argument("--hz", type=float, default=30.)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    height, stale = float(cfg.get("gmr_human_height", 1.6)), float(cfg["stale_s"])
    if (cfg.get("input_frame") != "gmr_pelvis"
            or not np.isfinite([height, stale, args.hz]).all()
            or not .5 <= height <= 2.5 or not 0 < stale <= .5 or not 1 <= args.hz <= 200):
        ap.error("invalid GMR config/rate")
    hz = min(args.hz, 30.)  # Solver is isolated from Bridge's 60 Hz publisher.
    import mujoco
    import xrobotoolkit_sdk as sdk
    from general_motion_retargeting import GeneralMotionRetargeting, XRobotStreamer
    retarget = GeneralMotionRetargeting("xrobot", "unitree_g1",
                                      actual_human_height=height, verbose=False)
    print(f"GMR pelvis source: SDK={sdk.__file__}; model={retarget.xml_file}; height={height}; Hz={hz}", flush=True)
    if args.preflight:
        print("PREFLIGHT OK: model/imports only; no SDK init or robot I/O")
        return 0
    with ExitStack() as stack:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context(Path(str(args.output) + ".lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_no_other_pico_readers()
        streamer = XRobotStreamer()
        stack.callback(sdk.close)
        freshness = FreshBody()
        session, seq = uuid.uuid4().hex, 0
        last_ok, last_print = float('-inf'), float('-inf')
        invalid = True
        print("SENSOR ONLY: virtual wrists in pelvis frame; left X/right A clutches. No robot transport.", flush=True)
        try:
            while True:
                start, acquired_ns = time.monotonic(), time.time_ns()
                try:
                    sample = read_body_and_buttons(sdk)
                    if sample is not None:
                        stamp, body, body_clock, hands = sample
                        if freshness.accept(stamp, body_clock):
                            human = human_data(body, streamer.coordinate_transform_unity_data,
                                               retarget.human_scale_table)
                            qpos = retarget.retarget(human, offset_to_ground=True)
                            if not np.isfinite(qpos).all():
                                raise ValueError("nonfinite GMR result")
                            mujoco.mj_forward(retarget.model, retarget.configuration.data)
                            for side in hands:
                                hands[side]['pose'] = pelvis_relative_pose(retarget.configuration.data, side)
                            if time.monotonic() - start >= stale:
                                raise ValueError("GMR processing exceeded freshness budget")
                            if invalid or start - last_ok > stale:
                                session, seq = uuid.uuid4().hex, 0
                            seq += 1
                            packet = dict(schema="pico_relative.v1", frame="gmr_pelvis",
                                session=session, seq=seq, source_ns=stamp,
                                # Acquisition time, NOT post-solver time: no stale rejuvenation.
                                emitted_ns=acquired_ns, hands=hands, body_clock=body_clock,
                                gmr_qpos=np.asarray(qpos).tolist(),
                                processing_s=time.monotonic()-start,
                                tracking_quality="body_clocks_only_not_hardware_validity")
                            if args.inspect:
                                print(json.dumps(packet, allow_nan=False), flush=True)
                            else:
                                write_packet(args.output, packet)
                            last_ok, invalid = start, False
                except (ValueError, TypeError, KeyError) as exc:
                    invalid = True
                    if start-last_print > 1:
                        print(f"GMR input refused: {exc}", flush=True)
                        last_print = start
                # No new Body frame => no output refresh. Consumer expires/HOLDs.
                time.sleep(max(0., 1/hz-(time.monotonic()-start)))
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
