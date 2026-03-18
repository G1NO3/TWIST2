"""
Convert TWIST2 demonstration data to LeRobot v2.0 dataset format.

Supports two action modes:
  - high_level: teleop target poses (action_body + optional hand/neck)
  - low_level:  motor commands from RL policy (action_low_level + optional hand)

Usage:
  python convert_twist2_to_lerobot.py \
      --data_dir twist2_demonstration/20260210_1017 \
      --output_dir /path/to/output \
      --repo_id "user/dataset_name" \
      --action_mode high_level
"""

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------- Dimension constants ----------
DIM_STATE_BODY = 34       # ang_vel(3) + roll_pitch(2) + dof_pos(29)
DIM_STATE_NECK = 2

DIM_ACTION_BODY = 35      # high-level teleop target
DIM_ACTION_NECK = 2
DIM_ACTION_LOW_LEVEL = 29  # low-level motor commands

HAND_DIM = {'dex3': 7, 'inspire': 6}
DIM_FORCE_HAND = {'dex3': 7, 'inspire': 6}  # motor current per finger


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TWIST2 data to LeRobot format")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to a single session dir containing episode_XXXX/ folders")
    parser.add_argument("--task_name", type=str, default="g1 task",
                        help="Task description string for all episodes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output LeRobot dataset root directory")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g. user/dataset_name)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Recording frequency (default: 60)")
    parser.add_argument("--action_mode", type=str, required=True, choices=["high_level", "low_level"],
                        help="Action mode: high_level (teleop targets) or low_level (motor commands)")

    # Hand inclusion defaults differ by mode — handled after parsing
    parser.add_argument("--include_hand", action="store_true", dest="include_hand", default=None,
                        help="Include hand/neck in action vector")
    parser.add_argument("--no_include_hand", action="store_false", dest="include_hand",
                        help="Exclude hand/neck from action vector")

    parser.add_argument("--use_videos", action="store_true", dest="use_videos", default=True,
                        help="Use video storage (default)")
    parser.add_argument("--no_videos", action="store_false", dest="use_videos",
                        help="Use image storage instead of video")

    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="Push dataset to HuggingFace Hub")
    parser.add_argument("--image_writer_processes", type=int, default=0)
    parser.add_argument("--image_writer_threads", type=int, default=4)
    parser.add_argument("--hand_type", type=str, default="dex3", choices=["dex3", "inspire"],
                        help="Hand type: dex3 (7-DOF) or inspire (6-DOF). Must match what was used during recording.")
    parser.add_argument("--include_force", action="store_true", default=False,
                        help="Include hand force/current data as observation.force (requires force-enabled recordings)")

    args = parser.parse_args()

    # Set include_hand default based on action_mode
    if args.include_hand is None:
        args.include_hand = (args.action_mode == "high_level")

    return args


def get_action_dim(action_mode: str, include_hand: bool, hand_dof: int) -> int:
    if action_mode == "high_level":
        dim = DIM_ACTION_BODY
        if include_hand:
            dim += hand_dof * 2 + DIM_ACTION_NECK
        return dim
    else:  # low_level
        dim = DIM_ACTION_LOW_LEVEL
        if include_hand:
            dim += hand_dof * 2
        return dim


def safe_array(value, expected_dim: int, field_name: str, frame_idx: int) -> np.ndarray:
    """Convert value to float32 array, zero-fill if None."""
    if value is None:
        warnings.warn(f"Frame {frame_idx}: '{field_name}' is None, zero-filling ({expected_dim}d)")
        return np.zeros(expected_dim, dtype=np.float32)
    arr = np.array(value, dtype=np.float32)
    if arr.shape[0] != expected_dim:
        warnings.warn(
            f"Frame {frame_idx}: '{field_name}' has dim {arr.shape[0]}, expected {expected_dim}. Zero-filling."
        )
        return np.zeros(expected_dim, dtype=np.float32)
    return arr


def build_state(frame: dict, idx: int, hand_dof: int) -> np.ndarray:
    """Build observation state vector."""
    state_body = safe_array(frame.get("state_body"), DIM_STATE_BODY, "state_body", idx)
    hand_left = safe_array(frame.get("state_hand_left"), hand_dof, "state_hand_left", idx)
    hand_right = safe_array(frame.get("state_hand_right"), hand_dof, "state_hand_right", idx)
    neck = safe_array(frame.get("state_neck"), DIM_STATE_NECK, "state_neck", idx)
    return np.concatenate([state_body, hand_left, hand_right, neck])


def build_action(frame: dict, idx: int, action_mode: str, include_hand: bool, hand_dof: int) -> np.ndarray:
    """Build action vector based on mode and hand flag."""
    if action_mode == "high_level":
        action = safe_array(frame.get("action_body"), DIM_ACTION_BODY, "action_body", idx)
        if include_hand:
            hand_left = safe_array(frame.get("action_hand_left"), hand_dof, "action_hand_left", idx)
            hand_right = safe_array(frame.get("action_hand_right"), hand_dof, "action_hand_right", idx)
            neck = safe_array(frame.get("action_neck"), DIM_ACTION_NECK, "action_neck", idx)
            action = np.concatenate([action, hand_left, hand_right, neck])
    else:  # low_level
        action = safe_array(frame.get("action_low_level"), DIM_ACTION_LOW_LEVEL, "action_low_level", idx)
        if include_hand:
            hand_left = safe_array(frame.get("action_hand_left"), hand_dof, "action_hand_left", idx)
            hand_right = safe_array(frame.get("action_hand_right"), hand_dof, "action_hand_right", idx)
            action = np.concatenate([action, hand_left, hand_right])
    return action


def build_force(frame: dict, idx: int, hand_dof: int) -> np.ndarray:
    """Build force observation vector (motor current from both hands)."""
    force_left = safe_array(frame.get("force_hand_left"), hand_dof, "force_hand_left", idx)
    force_right = safe_array(frame.get("force_hand_right"), hand_dof, "force_hand_right", idx)
    return np.concatenate([force_left, force_right])


def main():
    args = parse_args()

    hand_dof = HAND_DIM[args.hand_type]
    dim_state = DIM_STATE_BODY + hand_dof * 2 + DIM_STATE_NECK
    print(f"Hand type: {args.hand_type} ({hand_dof}-DOF per hand)")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    output_dir = Path(args.output_dir)

    # Discover episodes
    episode_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    ])
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_XXXX directories found in {data_dir}")
    print(f"Found {len(episode_dirs)} episodes in {data_dir}")

    # Read first image to get actual dimensions
    first_episode_json = episode_dirs[0] / "data.json"
    with open(first_episode_json) as f:
        first_data = json.load(f)
    first_rgb_path = episode_dirs[0] / first_data["data"][0]["rgb"]
    first_img = cv2.imread(str(first_rgb_path))
    if first_img is None:
        raise FileNotFoundError(f"Cannot read image: {first_rgb_path}")
    height, width = first_img.shape[:2]
    print(f"Image dimensions: {height}x{width}")

    # Compute action dim
    action_dim = get_action_dim(args.action_mode, args.include_hand, hand_dof)
    dim_force = hand_dof * 2  # left + right
    print(f"Action mode: {args.action_mode}, include_hand: {args.include_hand}, action_dim: {action_dim}")
    print(f"State dim: {dim_state}")
    if args.include_force:
        print(f"Force dim: {dim_force} (motor current, {hand_dof} per hand)")

    # Define features
    vision_dtype = "video" if args.use_videos else "image"
    features = {
        "observation.images.head_rgb": {
            "dtype": vision_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (dim_state,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }

    if args.include_force:
        features["observation.force"] = {
            "dtype": "float32",
            "shape": (dim_force,),
            "names": ["force"],
        }

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=str(output_dir),
        robot_type="unitree_g1",
        fps=args.fps,
        features=features,
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    total_frames = 0

    for ep_dir in tqdm(episode_dirs, desc="Converting episodes"):
        json_path = ep_dir / "data.json"
        with open(json_path) as f:
            ep_data = json.load(f)

        frames = ep_data["data"]
        num_frames = len(frames)

        for frame in frames:
            idx = frame["idx"]

            # Load RGB image (BGR -> RGB)
            rgb_path = ep_dir / frame["rgb"]
            img = cv2.imread(str(rgb_path))
            if img is None:
                warnings.warn(f"Cannot read image {rgb_path}, skipping frame {idx}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Build state and action
            state = build_state(frame, idx, hand_dof)
            action = build_action(frame, idx, args.action_mode, args.include_hand, hand_dof)

            frame_data = {
                "observation.images.head_rgb": img_rgb,
                "observation.state": state,
                "action": action,
                "task": args.task_name,
            }

            if args.include_force:
                frame_data["observation.force"] = build_force(frame, idx, hand_dof)
            dataset.add_frame(frame_data)

        dataset.save_episode()
        total_frames += num_frames
        tqdm.write(f"  Saved {ep_dir.name}: {num_frames} frames")

    print("Finalizing dataset...")
    dataset.finalize()

    if args.push_to_hub:
        print("Pushing to HuggingFace Hub...")
        dataset.push_to_hub(private=True)

    # Summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Episodes:   {len(episode_dirs)}")
    print(f"  Frames:     {total_frames}")
    print(f"  State dim:  {dim_state}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hand type:  {args.hand_type} ({hand_dof}-DOF)")
    print(f"  Image size: {height}x{width}")
    print(f"  Action mode: {args.action_mode}")
    print(f"  Include hand: {args.include_hand}")
    print(f"  Include force: {args.include_force}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
