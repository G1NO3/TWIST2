"""PICO sensor-only source. No GMR, viewer, Redis action keys or robot driver.

Run this INSTEAD of teleop.sh/teleop_hil.sh for relative bimanual HIL.
The SDK's raw poses are metres + xyzw, native controller API values (not GMR
poses). The historical `unity_raw` wire label is not a handedness guarantee;
the consumer must explicitly calibrate the source basis.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid

DEFAULT_FILE = "/tmp/policy_bridge/pico_state.json"


def read_sdk(sdk):
    """Reject a read spanning SDK packet timestamps; retry on the next tick.

    This is a best-effort consistency check, NOT an atomic SDK snapshot.
    The upstream SDK has no controller-specific tracking-valid API.
    """
    stamp = int(sdk.get_time_stamp_ns())
    out = {}
    for side, one, two in (("left", "X", "Y"), ("right", "A", "B")):
        pose = list(getattr(sdk, f"get_{side}_controller_pose")())
        axis = list(getattr(sdk, f"get_{side}_axis")())
        grip = float(getattr(sdk, f"get_{side}_grip")())
        trigger = float(getattr(sdk, f"get_{side}_trigger")())
        if (len(pose) != 7 or len(axis) != 2
                or not all(math.isfinite(x) for x in pose + axis + [grip, trigger])
                or not .9 < sum(x*x for x in pose[3:]) < 1.1
                or not 0 <= grip <= 1 or not 0 <= trigger <= 1
                or any(abs(x) > 1 for x in axis)):
            raise ValueError(f"invalid {side} pose/controller input")
        out[side] = dict(pose=pose, axis=axis, grip=grip, index_trig=trigger,
                         key_one=bool(getattr(sdk, f"get_{one}_button")()),
                         key_two=bool(getattr(sdk, f"get_{two}_button")()),
                         axis_click=bool(getattr(sdk, f"get_{side}_axis_click")()))
    if stamp <= 0 or int(sdk.get_time_stamp_ns()) != stamp:
        return None
    return stamp, out


def write_packet(path, packet):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(packet, allow_nan=False))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=DEFAULT_FILE)
    ap.add_argument("--hz", type=float, default=100.)
    ap.add_argument("--inspect", action="store_true", help="Print input only; write no state file")
    ap.add_argument("--config", type=Path, help="Select unity_raw or isolated GMR sensor source")
    ap.add_argument("--preflight", action="store_true", help="Import only; never initialize SDK")
    args = ap.parse_args()
    if args.config:
        config = json.loads(args.config.read_text())
        frame = config.get("input_frame", "unity_raw")
        if frame in ("gmr_pelvis", "body_pelvis"):
            import sys
            wrapper = Path(__file__).resolve().parents[1] / "pico_gmr_source.sh"
            os.execv("/bin/bash", ["bash", str(wrapper), *sys.argv[1:]])
        if frame != "unity_raw":
            ap.error("unsupported input_frame")
    if args.preflight:
        import xrobotoolkit_sdk
        print(f"PREFLIGHT OK (no SDK init): {xrobotoolkit_sdk.__file__}")
        return
    if not math.isfinite(args.hz) or not 1 <= args.hz <= 200:
        ap.error("--hz must be 1..200")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A second input writer must fail, not alternate sessions on every tick.
    lock = open(str(path) + ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import xrobotoolkit_sdk as sdk
    sdk.init()
    session = uuid.uuid4().hex
    seq, previous = 0, 0
    print("SENSOR ONLY: X=left clutch, A=right clutch; B/left stick-click=hold request.", flush=True)
    print("Keep Unitree physical stop available. No robot commands are published.", flush=True)
    try:
        while True:
            try:
                sample = read_sdk(sdk)
                if sample is not None:
                    stamp, hands = sample
                    if stamp < previous:
                        raise ValueError("SDK timestamp regressed; restart source after checking tracking")
                    if stamp > previous:
                        previous = stamp
                        seq += 1
                        packet = dict(schema="pico_relative.v1", frame="unity_raw",
                                      session=session, seq=seq, source_ns=stamp,
                                      emitted_ns=time.time_ns(), hands=hands,
                                      tracking_quality="unavailable")
                        if args.inspect:
                            print(json.dumps(packet), flush=True)
                        else:
                            write_packet(path, packet)
                # Unchanged timestamps do NOT refresh the file freshness.
            except (ValueError, TypeError) as exc:
                # Leave the last file to expire. Restart/release is required
                # by the consumer; malformed values are never made fresh.
                print(f"PICO input refused: {exc}", flush=True)
            time.sleep(1 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        sdk.close()
        lock.close()
        # Do not rewrite a final 'released' packet: exit must cause HOLD.


if __name__ == "__main__":
    main()
