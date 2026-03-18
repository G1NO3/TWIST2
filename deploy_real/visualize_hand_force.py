#!/usr/bin/env python3
"""
Live visualization of Inspire hand force/current data.

Reads force_act and motor current from both Inspire hands via Modbus TCP
and displays a real-time bar chart showing per-finger force levels.

This provides contact/grasp feedback — the closest the Inspire RH56DFTP
has to tactile sensing (it does not have fingertip pressure arrays).

Usage (from deploy_real/):
  python visualize_hand_force.py
  python visualize_hand_force.py --left_ip 192.168.123.210 --right_ip 192.168.123.211
  python visualize_hand_force.py --mode current   # use motor current instead of force_act
  python visualize_hand_force.py --text_only       # terminal-only, no GUI needed
"""
import argparse
import time
import sys

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend which works without libgtk
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from robot_control.inspire_hand_wrapper import (
    InspireHandController, REG_FORCE_ACT, REG_CURRENT,
    Inspire_Num_Motors,
)

FINGER_NAMES = ["Pinky", "Ring", "Middle", "Index", "ThBend", "ThRot"]


def read_force_act(controller):
    """Read REG_FORCE_ACT (actual force) from both hands."""
    left = controller._read_registers_signed(controller.left_client, REG_FORCE_ACT, 6)
    right = controller._read_registers_signed(controller.right_client, REG_FORCE_ACT, 6)
    return np.array(left, dtype=np.float32), np.array(right, dtype=np.float32)


def read_current(controller):
    """Read REG_CURRENT (motor current) from both hands."""
    controller.get_hand_state()
    return controller.Ltau.copy(), controller.Rtau.copy()


def main():
    parser = argparse.ArgumentParser(description="Visualize Inspire hand force/current")
    parser.add_argument("--left_ip", default="192.168.123.210")
    parser.add_argument("--right_ip", default="192.168.123.211")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--rate", type=float, default=10, help="Update rate Hz")
    parser.add_argument("--mode", choices=["force", "current"], default="force",
                        help="force = REG_FORCE_ACT, current = REG_CURRENT (motor current)")
    parser.add_argument("--max_val", type=float, default=0,
                        help="Max bar value (0 = auto-scale)")
    parser.add_argument("--text_only", action="store_true",
                        help="Print to terminal instead of GUI window")
    args = parser.parse_args()

    # Connect to hands (read-only, no re-init to avoid moving the fingers)
    hand_ctrl = InspireHandController(
        left_ip=args.left_ip,
        right_ip=args.right_ip,
        port=args.port,
        re_init=False,
    )

    read_fn = read_force_act if args.mode == "force" else read_current
    mode_label = "Force (force_act)" if args.mode == "force" else "Current (motor mA)"
    max_val = args.max_val if args.max_val > 0 else 500
    auto_scale = args.max_val == 0

    dt = 1.0 / args.rate

    print(f"Mode: {mode_label}")
    print(f"Update rate: {args.rate} Hz")

    if args.text_only:
        print("Text-only mode. Press Ctrl+C to quit.\n")
        try:
            while True:
                t0 = time.time()
                left_vals, right_vals = read_fn(hand_ctrl)
                l_str = " ".join(f"{v:6.0f}" for v in left_vals)
                r_str = " ".join(f"{v:6.0f}" for v in right_vals)
                print(f"L: [{l_str}]  R: [{r_str}]", end="\r")
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        except KeyboardInterrupt:
            print("\nDone.")
        return

    # --- Matplotlib live bar chart ---
    plt.ion()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Inspire Hand — {mode_label}", fontsize=14)

    x = np.arange(6)
    bars_l = ax_l.bar(x, np.zeros(6), color='#3498db', edgecolor='white')
    bars_r = ax_r.bar(x, np.zeros(6), color='#e67e22', edgecolor='white')

    ax_l.set_title("LEFT Hand")
    ax_r.set_title("RIGHT Hand")
    for ax in (ax_l, ax_r):
        ax.set_xticks(x)
        ax.set_xticklabels(FINGER_NAMES, fontsize=9)
        ax.set_ylim(0, max_val)
        ax.set_ylabel(mode_label)
        ax.grid(axis='y', alpha=0.3)

    # Value text labels on bars
    texts_l = [ax_l.text(i, 0, "", ha='center', va='bottom', fontsize=8) for i in x]
    texts_r = [ax_r.text(i, 0, "", ha='center', va='bottom', fontsize=8) for i in x]

    fig.tight_layout()
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    print("Close the plot window or press Ctrl+C to quit.\n")

    try:
        while plt.fignum_exists(fig.number):
            t0 = time.time()

            left_vals, right_vals = read_fn(hand_ctrl)

            if auto_scale:
                observed_max = max(np.max(np.abs(left_vals)), np.max(np.abs(right_vals)), 1)
                if observed_max > max_val:
                    max_val = observed_max * 1.2  # add 20% headroom
                    ax_l.set_ylim(0, max_val)
                    ax_r.set_ylim(0, max_val)

            for i in range(6):
                bars_l[i].set_height(abs(left_vals[i]))
                bars_r[i].set_height(abs(right_vals[i]))
                texts_l[i].set_position((i, abs(left_vals[i])))
                texts_l[i].set_text(f"{left_vals[i]:.0f}")
                texts_r[i].set_position((i, abs(right_vals[i])))
                texts_r[i].set_text(f"{right_vals[i]:.0f}")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        plt.close('all')
        print("Done.")


if __name__ == "__main__":
    main()
