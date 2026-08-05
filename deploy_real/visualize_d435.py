#!/usr/bin/env python3
"""
Live viewer for the D435i RGBD stream from the robot.

Connects to the RealSense ZMQ streamer running on the robot's Orin
and displays RGB and depth images on your host computer.

The Orin must be running realsense_streamer.py (onboard/start_realsense.sh).

Usage (from deploy_real/):
  python visualize_d435.py
  python visualize_d435.py --robot_ip 192.168.123.164 --port 5555
  python visualize_d435.py --no-depth          # RGB only
  python visualize_d435.py --save_dir frames   # also save frames to disk
"""
import argparse
import os
import signal
import struct
import time

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import zmq


def main():
    parser = argparse.ArgumentParser(description="View D435i RGBD stream from robot")
    parser.add_argument("--robot_ip", default="192.168.123.164", help="Robot / Orin IP")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth display")
    parser.add_argument("--save_dir", default="", help="Save frames to this directory (empty = don't save)")
    parser.add_argument("--max_depth_mm", type=int, default=3000, help="Max depth for colormap (mm)")
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{args.robot_ip}:{args.port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.setsockopt(zmq.CONFLATE, 1)

    print(f"Connecting to tcp://{args.robot_ip}:{args.port} ...")

    if args.save_dir:
        os.makedirs(os.path.join(args.save_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "depth"), exist_ok=True)
        print(f"Saving frames to {args.save_dir}/")

    running = True
    def _sigint(_s, _f):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _sigint)

    # Setup matplotlib figure
    show_depth = not args.no_depth
    plt.ion()
    if show_depth:
        fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax_rgb = plt.subplots(1, 1, figsize=(6, 5))
        ax_depth = None

    # Initialize with blank images (424x240 default)
    blank_rgb = np.zeros((480, 848, 3), dtype=np.uint8)
    im_rgb = ax_rgb.imshow(blank_rgb)
    ax_rgb.set_title("D435i RGB")
    ax_rgb.axis('off')

    im_depth = None
    if show_depth and ax_depth is not None:
        blank_depth = np.zeros((480, 848), dtype=np.float32)
        im_depth = ax_depth.imshow(blank_depth, cmap='jet', vmin=0, vmax=args.max_depth_mm)
        ax_depth.set_title("D435i Depth")
        ax_depth.axis('off')
        fig.colorbar(im_depth, ax=ax_depth, fraction=0.046, pad=0.04, label='mm')

    fps_text = ax_rgb.text(0.02, 0.95, "FPS: --", transform=ax_rgb.transAxes,
                           fontsize=10, color='lime', verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    fig.tight_layout()
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    frame_count = 0
    t_start = time.time()
    fps_display = 0.0
    save_count = 0

    print("Waiting for frames... (close window or Ctrl+C to quit)")

    try:
        while running and plt.fignum_exists(fig.number):
            try:
                message = sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                # No message yet — still pump the GUI event loop
                fig.canvas.flush_events()
                time.sleep(0.005)
                continue

            if len(message) < 16:
                continue

            width, height, rgb_len, depth_len = struct.unpack("iiii", message[:16])
            if len(message) - 16 != rgb_len + depth_len:
                continue

            # Decode RGB (cv2.imdecode works even with headless OpenCV)
            jpeg_data = message[16:16 + rgb_len]
            bgr_img = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr_img is None:
                continue
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # Decode depth
            depth_img = None
            if depth_len > 0 and show_depth:
                try:
                    depth_img = np.frombuffer(message[16 + rgb_len:], dtype=np.uint16).reshape(height, width)
                except ValueError:
                    depth_img = None

            # FPS
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # Update matplotlib images
            im_rgb.set_data(rgb_img)
            fps_text.set_text(f"FPS: {fps_display:.1f}")

            if im_depth is not None and depth_img is not None:
                im_depth.set_data(depth_img.astype(np.float32))

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            # Save frames
            if args.save_dir:
                cv2.imwrite(os.path.join(args.save_dir, "rgb", f"{save_count:06d}.jpg"), bgr_img)
                if depth_img is not None:
                    np.save(os.path.join(args.save_dir, "depth", f"{save_count:06d}.npy"), depth_img)
                save_count += 1

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sock.close()
        ctx.term()
        plt.close('all')
        print(f"Done. Total saved: {save_count} frames" if args.save_dir else "Done.")


if __name__ == "__main__":
    main()
