#!/usr/bin/env python3

"""
Data collection script for the internal RealSense D435i camera.

Collects RGB + aligned depth from the D435i (via ZMQ), plus body/hand
state and action from Redis, and writes episodes to disk. Optionally also
merges in RGB from the two Arducam wrist cameras (--wrist_left_port /
--wrist_right_port), so a single episode/data.json stays frame-aligned
across all camera views and the robot state/action.

Differences from server_data_record.py (ZED):
  - Single 424x240 RGB image (not stereo 1280x360)
  - Aligned uint16 depth at 424x240
  - Default 60 Hz recording frequency (configurable via --frequency)
  - Task description via CLI args (--goal, --desc, --steps)
"""

import argparse
import json
import os
import signal
import time

import cv2
import numpy as np
import redis
from datetime import datetime
from multiprocessing import shared_memory
import threading

from data_utils.episode_writer import EpisodeWriter
from data_utils.rl_episode_follow import RLEpisodeFollower
from data_utils.vision_client import VisionClient
from rich import print


def write_rl_sidecar(episode_dir, rl_id, label, reason):
    """<episode>/rl_episode.json: the RL replay episode this recording is, and
    its latest outcome (a later discard rewrites only this small file)."""
    path = os.path.join(episode_dir, "rl_episode.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"rl_episode": rl_id, "label": label, "outcome": reason,
                   "time": time.time()}, f)
    os.replace(tmp, path)


def main(args):
    # ---- Redis connection ----
    try:
        redis_pool = redis.ConnectionPool(
            host="localhost",
            port=6379,
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.1,
            socket_connect_timeout=0.1,
        )
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_pipeline = redis_client.pipeline()
        redis_client.ping()
        print(f"Connected to Redis at localhost:6379, DB=0 with connection pool")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return

    # ---- Shared memory for RGB (424x240x3) ----
    image_shape = (args.height, args.width, 3)
    image_shared_memory = shared_memory.SharedMemory(
        create=True, size=int(np.prod(image_shape) * np.uint8().itemsize)
    )
    image_array = np.ndarray(image_shape, dtype=np.uint8, buffer=image_shared_memory.buf)

    # ---- Shared memory for depth (424x240, float32) ----
    depth_shape = (args.height, args.width)
    depth_shared_memory = shared_memory.SharedMemory(
        create=True, size=int(np.prod(depth_shape) * np.float32().itemsize)
    )
    depth_array = np.ndarray(depth_shape, dtype=np.float32, buffer=depth_shared_memory.buf)

    # ---- Shared memory for wrist cameras (RGB only, no depth) ----
    wrist_image_shape = (args.wrist_height, args.wrist_width, 3)

    wrist_left_shared_memory = None
    wrist_left_array = None
    if args.wrist_left_port is not None:
        wrist_left_shared_memory = shared_memory.SharedMemory(
            create=True, size=int(np.prod(wrist_image_shape) * np.uint8().itemsize)
        )
        wrist_left_array = np.ndarray(wrist_image_shape, dtype=np.uint8, buffer=wrist_left_shared_memory.buf)

    wrist_right_shared_memory = None
    wrist_right_array = None
    if args.wrist_right_port is not None:
        wrist_right_shared_memory = shared_memory.SharedMemory(
            create=True, size=int(np.prod(wrist_image_shape) * np.uint8().itemsize)
        )
        wrist_right_array = np.ndarray(wrist_image_shape, dtype=np.uint8, buffer=wrist_right_shared_memory.buf)

    # ---- Vision client ----
    try:
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        image_show = True
    except cv2.error:
        image_show = False
        print("[Warning] OpenCV GUI not available (headless build). Disabling image preview.")
    vision_client = VisionClient(
        server_address=args.robot_ip,
        port=args.camera_port,
        img_shape=image_shape,
        img_shm_name=image_shared_memory.name,
        depth_shape=depth_shape,
        depth_shm_name=depth_shared_memory.name,
        image_show=False,
        depth_show=False,
        unit_test=True,
    )
    vision_thread = threading.Thread(target=vision_client.receive_process, daemon=True)
    vision_thread.start()

    if args.wrist_left_port is not None:
        wrist_left_client = VisionClient(
            server_address=args.robot_ip,
            port=args.wrist_left_port,
            img_shape=wrist_image_shape,
            img_shm_name=wrist_left_shared_memory.name,
            unit_test=True,
        )
        threading.Thread(target=wrist_left_client.receive_process, daemon=True).start()

    if args.wrist_right_port is not None:
        wrist_right_client = VisionClient(
            server_address=args.robot_ip,
            port=args.wrist_right_port,
            img_shape=wrist_image_shape,
            img_shm_name=wrist_right_shared_memory.name,
            unit_test=True,
        )
        threading.Thread(target=wrist_right_client.receive_process, daemon=True).start()

    # ---- Episode writer ----
    recording = False
    save_data_keys = ["rgb", "depth"]
    if args.wrist_left_port is not None:
        save_data_keys.append("rgb_wrist_left")
    if args.wrist_right_port is not None:
        save_data_keys.append("rgb_wrist_right")
    task_dir = os.path.join(args.data_folder, args.task_name)
    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=args.frequency,
        image_shape=image_shape,
        data_keys=save_data_keys,
    )
    recorder.text_desc(goal=args.goal, desc=args.desc, steps=args.steps)

    control_dt = 1.0 / args.frequency
    rl_dirs = {}  # RL episode id -> episode dir (for a post-terminal discard)
    step_count = 0
    frame_counter = 0  # for sub-sampling
    running = True

    # Sub-sampling: how many camera frames to skip between recorded frames.
    # e.g. camera at 60fps, --frequency 30 → record every 2nd frame.
    subsample_interval = max(1, round(60.0 / args.frequency))

    print(f"Recorded control frequency: {args.frequency} Hz (subsample every {subsample_interval} frames)")

    if args.episode_status is None:
        from robot_control.speaker import Speaker  # pyttsx3; speak() blocks the loop
        speaker = Speaker()
    else:
        # RL episodes open/close on the bridge's clock: a blocking voice line
        # would drop the first frames of the next episode. Log only.
        class speaker:
            speak = staticmethod(lambda text: print(f"[recorder] {text}"))
    prev_button_pressed = False
    prev_right_axis_click_pressed = False

    # --episode_status: episodes follow the RL bridge's status file instead of
    # controller_data (which is then never read). Default off: unchanged.
    follower = None
    rl_status = None
    pending_start = None
    missing_warned = set()
    if args.episode_status is not None:
        follower = RLEpisodeFollower(
            args.episode_status, stale_s=args.episode_stale_s,
            instruction_path=os.path.join(os.path.dirname(args.episode_status), "phase_instruction.txt"))
        print(f"Episodes follow {args.episode_status} (controller_data ignored)")

        def _terminate(signum, frame):
            raise KeyboardInterrupt  # GUI stop: save the open episode as interrupted
        signal.signal(signal.SIGTERM, _terminate)

    def save_rl_episode(rl_id, label, reason):
        recorder.data["rl"] = {"episode": rl_id, "outcome": reason, "label": label,
                               "root": (rl_status or {}).get("root"),
                               "contract_sha256": (rl_status or {}).get("contract_sha256"),
                               "episode_scope": (rl_status or {}).get("episode_scope")}
        write_rl_sidecar(recorder.episode_dir, rl_id, label, reason)
        recorder.save_episode(label=label)
        speaker.speak(f"episode saved as {label}.")

    try:
        while running:
            start_time = time.time()

            if follower is not None:
                rl_status = follower.read()
                for event in follower.step(rl_status, start_time):
                    if event[0] == "save" and recording:
                        save_rl_episode(*event[1:])
                        recording = False
                    elif event[0] == "start":
                        pending_start = event[1]
                    elif event[0] == "relabel" and event[1] in rl_dirs:
                        write_rl_sidecar(rl_dirs[event[1]], event[1], event[2], event[3])
                        print(f"RL episode {event[1]} relabelled {event[2]} -> {rl_dirs[event[1]]}/rl_episode.json")

                # The writer holds one episode: while the previous save is still
                # writing, retry every tick instead of blocking the status loop.
                if pending_start is not None and pending_start != follower.current:
                    pending_start = None           # closed before it could start
                if pending_start is not None and recorder.is_available:
                    recording = recorder.create_episode()
                    if recording:
                        rl_dirs[pending_start] = recorder.episode_dir
                        speaker.speak("episode recording started.")
                        print(f"episode recording started for RL episode {pending_start}...")
                    pending_start = None
                    step_count = 0
                    frame_counter = 0

            # ---- Controller input ----
            controller_data = (json.loads(redis_client.get("controller_data"))
                               if follower is None else None)
            button_pressed = follower is None and controller_data["LeftController"]["key_two"]

            quit_key = follower is None and controller_data["LeftController"]["axis_click"]
            if quit_key:
                running = False
                speaker.speak("Recording stopped.")
                print("\nQuitting...")
                break

            right_axis_click = follower is None and controller_data["RightController"]["axis_click"]

            # Rising-edge toggle
            if button_pressed and not prev_button_pressed:
                print("button pressed")
                recording = not recording
                if recording:
                    speaker.speak("episode recording started.")
                    if not recorder.create_episode():
                        recording = False
                    step_count = 0
                    frame_counter = 0
                    print("episode recording started...")
                else:
                    recorder.save_episode(label="successful")
                    speaker.speak("episode saved as successful.")

            # Right axis_click: save as unsuccessful
            if right_axis_click and not prev_right_axis_click_pressed and recording:
                recorder.save_episode(label="unsuccessful")
                recording = False
                speaker.speak("episode saved as unsuccessful.")

            prev_button_pressed = button_pressed
            prev_right_axis_click_pressed = right_axis_click

            if recording:
                frame_counter += 1

                # Sub-sample: only record on the correct cadence
                if frame_counter % subsample_interval != 0:
                    elapsed = time.time() - start_time
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
                    continue

                data_dict = {"idx": step_count}
                if follower is not None:
                    data_dict.update(follower.frame_fields(rl_status))

                # ---- Vision data ----
                data_dict["rgb"] = image_array.copy()
                data_dict["t_img"] = int(time.time() * 1000)

                # Depth: read shared memory and convert float32 → uint16 (mm)
                raw_depth = depth_array.copy()
                if np.any(raw_depth > 0):
                    data_dict["depth"] = raw_depth.astype(np.uint16)
                else:
                    data_dict["depth"] = None

                # Wrist cameras: same-tick copy keeps frames aligned with D435 + state/action
                if wrist_left_array is not None:
                    data_dict["rgb_wrist_left"] = wrist_left_array.copy()
                    data_dict["t_img_wrist_left"] = int(time.time() * 1000)
                if wrist_right_array is not None:
                    data_dict["rgb_wrist_right"] = wrist_right_array.copy()
                    data_dict["t_img_wrist_right"] = int(time.time() * 1000)

                # ---- Redis state & action ----
                redis_keys = [
                    "state_body_unitree_g1_with_hands",
                    "state_hand_left_unitree_g1_with_hands",
                    "state_hand_right_unitree_g1_with_hands",
                    "state_neck_unitree_g1_with_hands",
                    "t_state",
                    "action_body_unitree_g1_with_hands",
                    "action_hand_left_unitree_g1_with_hands",
                    "action_hand_right_unitree_g1_with_hands",
                    "action_neck_unitree_g1_with_hands",
                    "t_action",
                    "action_low_level_unitree_g1_with_hands",
                    "force_hand_left_unitree_g1_with_hands",
                    "force_hand_right_unitree_g1_with_hands",
                    "tactile_hand_left_unitree_g1_with_hands",
                    "tactile_hand_right_unitree_g1_with_hands",
                    "effort_body_unitree_g1_with_hands",
                ]

                data_dict_keys = [
                    "state_body",
                    "state_hand_left",
                    "state_hand_right",
                    "state_neck",
                    "t_state",
                    "action_body",
                    "action_hand_left",
                    "action_hand_right",
                    "action_neck",
                    "t_action",
                    "action_low_level",
                    "force_hand_left",
                    "force_hand_right",
                    "tactile_hand_left",
                    "tactile_hand_right",
                    "effort_body",
                ]

                try:
                    for key in redis_keys:
                        redis_pipeline.get(key)
                    redis_results = redis_pipeline.execute()

                    for i, (result, dict_key) in enumerate(zip(redis_results, data_dict_keys)):
                        if result is not None:
                            try:
                                data_dict[dict_key] = json.loads(result)
                            except json.JSONDecodeError:
                                print(f"Warning: Failed to decode JSON for key {redis_keys[i]}")
                                data_dict[dict_key] = None
                        else:
                            if follower is None or redis_keys[i] not in missing_warned:
                                missing_warned.add(redis_keys[i])
                                print(f"Warning: No data found for key {redis_keys[i]}")
                            data_dict[dict_key] = None
                except Exception as e:
                    print(f"Error in Redis pipeline operation: {e}")
                    continue

                recorder.add_item(data_dict)

                if image_show:
                    if image_array is not None and image_array.size > 0:
                        window_name = "D435i - Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, image_array.shape[1], image_array.shape[0])
                        cv2.moveWindow(window_name, 50, 50)
                        cv2.imshow(window_name, image_array)
                        cv2.waitKey(1)

                step_count += 1
                elapsed = time.time() - start_time
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                if image_show:
                    if image_array is not None and image_array.size > 0:
                        window_name = "D435i - Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, image_array.shape[1], image_array.shape[0])
                        cv2.moveWindow(window_name, 50, 50)
                        cv2.imshow(window_name, image_array)
                        cv2.waitKey(1)
                else:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, exiting...")
        running = False
    finally:
        if follower is not None and recording:
            # close() would otherwise save it as "successful"
            save_rl_episode(follower.current, "interrupted", "recorder_stopped")
        if follower is not None:
            # close() re-saves a still-pending episode as "successful": let the
            # queued save (with its RL label) finish first.
            deadline = time.monotonic() + 30.0
            while not recorder.is_available and time.monotonic() < deadline:
                time.sleep(0.05)
        print(f"\nDone! Recorded {recorder.episode_id + 1} episodes to {task_dir}")

        image_shared_memory.unlink()
        image_shared_memory.close()
        depth_shared_memory.unlink()
        depth_shared_memory.close()
        if wrist_left_shared_memory is not None:
            wrist_left_shared_memory.unlink()
            wrist_left_shared_memory.close()
        if wrist_right_shared_memory is not None:
            wrist_right_shared_memory.unlink()
            wrist_right_shared_memory.close()
        recorder.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        print("Exiting the recording...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D435i data recording with RGB + depth.")
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")

    parser.add_argument("--data_folder", default="twist2_demonstration", help="Data folder")
    parser.add_argument("--task_name", default=f"{cur_time}", help="Task name")
    parser.add_argument("--frequency", default=60, type=int, help="Recording frequency in Hz")
    parser.add_argument("--robot", default="unitree_g1", choices=["unitree_g1"], help="Robot name")
    parser.add_argument("--robot_ip", default="192.168.123.164", help="Robot / Orin IP")
    parser.add_argument("--camera_port", default=5555, type=int, help="ZMQ port for D435i streamer")
    parser.add_argument("--width", default=1280, type=int, help="Image width")
    parser.add_argument("--height", default=720, type=int, help="Image height")

    # Wrist camera options. Omit both ports to keep D435-only behavior.
    parser.add_argument("--wrist_left_port", default=None, type=int, help="ZMQ port for left wrist streamer (omit to disable)")
    parser.add_argument("--wrist_right_port", default=None, type=int, help="ZMQ port for right wrist streamer (omit to disable)")
    parser.add_argument("--wrist_width", default=640, type=int, help="Wrist camera image width")
    parser.add_argument("--wrist_height", default=480, type=int, help="Wrist camera image height")

    # Task description metadata
    parser.add_argument("--goal", default="pick up the red cup", help="Task goal description")
    parser.add_argument("--desc", default="A humanoid robot picks up a red cup from the table.", help="Task description")
    parser.add_argument("--steps", default="step1: approach table. step2: grasp cup. step3: lift cup.", help="Task steps")

    # Episodes from the RL bridge instead of controller_data (default off).
    parser.add_argument("--episode_status", default=None,
                        help="RL bridge status json (e.g. /tmp/policy_bridge/rlt_status.json): record every "
                             "frame of every RL episode, labelled with its outcome; controller_data is not read")
    parser.add_argument("--episode_stale_s", default=5.0, type=float,
                        help="with --episode_status: a status older than this ends the episode as aborted")

    args = parser.parse_args()
    main(args)
