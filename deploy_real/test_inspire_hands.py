#!/usr/bin/env python3
"""
Standalone Inspire hand test — no robot, no Unitree SDK needed.

Reads PICO controller trigger/grip from Redis (published by xrobot_teleop_to_robot_w_hand.py)
and sends commands directly to Inspire hands via Modbus TCP.

Usage:
  Terminal 1 (gmr env): python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1 --hand_type inspire
  Terminal 2 (twist2 env): python test_inspire_hands.py

Or run with --no-hardware to just monitor Redis values without connecting to hands.
"""
import argparse
import json
import time
import sys

import numpy as np
import redis

sys.path.insert(0, '.')


def main():
    parser = argparse.ArgumentParser(description="Test Inspire hands from Redis")
    parser.add_argument("--left_ip", default="192.168.123.210", help="Left hand IP")
    parser.add_argument("--right_ip", default="192.168.123.211", help="Right hand IP")
    parser.add_argument("--no-hardware", action="store_true",
                        help="Monitor Redis only, don't connect to hands")
    parser.add_argument("--rate", type=float, default=50, help="Control loop Hz")
    args = parser.parse_args()

    # Connect to Redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("Redis connected")

    # Connect to Inspire hands (optional)
    hand_ctrl = None
    if not args.no_hardware:
        try:
            from robot_control.inspire_hand_wrapper import InspireHandController
            hand_ctrl = InspireHandController(
                left_ip=args.left_ip,
                right_ip=args.right_ip,
                re_init=False
            )
            print(f"Inspire hands connected: L={args.left_ip}, R={args.right_ip}")
        except Exception as e:
            print(f"Failed to connect to hands: {e}")
            print("Running in monitor-only mode")
    else:
        print("Monitor-only mode (no hardware)")

    dt = 1.0 / args.rate
    count = 0

    print("\nWaiting for hand data on Redis...")
    print("  Keys: action_hand_left_unitree_g1_with_hands")
    print("        action_hand_right_unitree_g1_with_hands")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            t0 = time.time()

            # Read hand actions from Redis
            left_raw = r.get("action_hand_left_unitree_g1_with_hands")
            right_raw = r.get("action_hand_right_unitree_g1_with_hands")

            if left_raw is None and right_raw is None:
                if count % 50 == 0:
                    print("No hand data in Redis yet...", end="\r")
                count += 1
                time.sleep(dt)
                continue

            left_cmd = np.array(json.loads(left_raw), dtype=np.float32) if left_raw else np.zeros(6)
            right_cmd = np.array(json.loads(right_raw), dtype=np.float32) if right_raw else np.zeros(6)

            # Send to hardware
            if hand_ctrl is not None:
                hand_ctrl.ctrl_dual_hand(left_cmd, right_cmd)

            # Print diagnostics
            count += 1
            if count % 25 == 0:
                l_fingers = left_cmd[:4]
                l_thumb = left_cmd[4:]
                r_fingers = right_cmd[:4]
                r_thumb = right_cmd[4:]
                print(f"[{count:6d}] "
                      f"L fingers={l_fingers[0]:4.0f} thumb={l_thumb[0]:4.0f} | "
                      f"R fingers={r_fingers[0]:4.0f} thumb={r_thumb[0]:4.0f}")

                # Also read state feedback if hardware connected
                if hand_ctrl is not None:
                    try:
                        l_state, r_state = hand_ctrl.get_hand_state()
                        print(f"         L state={[f'{v:.0f}' for v in l_state]} | "
                              f"R state={[f'{v:.0f}' for v in r_state]}")
                    except Exception:
                        pass

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if hand_ctrl is not None:
            # Open hands on exit
            print("Opening hands...")
            hand_ctrl.ctrl_dual_hand(np.zeros(6), np.zeros(6))
            time.sleep(0.5)
        print("Done.")


if __name__ == "__main__":
    main()
