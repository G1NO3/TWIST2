"""
Convert TWIST2 demonstration data to LeRobot v2.0 dataset format
with per-frame subtask labels based on hand state transitions.

Subtask workflow (bottle packing):
  1. "grasp the bottle"           — until fingers close on bottle
  2. "place the bottle into the box" — until fingers release bottle
  3. "close the box"              — until end of episode

Hand state convention (Inspire 6-DOF: [pinky, ring, middle, index, thumb_bend, thumb_rot]):
  ~1000 = OPEN (fingers extended)
  ~0    = CLOSED (fingers curled, gripping)

Usage:
  python convert_twist2_to_lerobot_subtask.py \
      --data_dir twist2_demonstration/20260210_1017 \
      --output_dir /path/to/output \
      --repo_id "user/dataset_name" \
      --action_mode high_level \
      --hand_type inspire

Note: Neck state and neck action commands are excluded from the converted dataset.
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

DIM_ACTION_BODY = 35      # high-level teleop target
DIM_ACTION_LOW_LEVEL = 29  # low-level motor commands

HAND_DIM = {'dex3': 7, 'inspire': 6}
DIM_FORCE_HAND = {'dex3': 7, 'inspire': 6}  # motor current per finger

# ---------- Subtask definitions ----------
SUBTASKS = [
    "grasp the bottle",
    "place the bottle into the box",
    "close the box",
]


def detect_subtasks(frames, close_thresh=500, open_thresh=700):
    """Return list of subtask label strings, one per frame.

    Detection logic (Inspire hand, indices 0-3 = pinky/ring/middle/index):
      - State values: ~1000 = open, ~0 = closed
      - Transition 1 (grasp): >=3 of 4 fingers of either hand drop below close_thresh
      - Transition 2 (release): >=3 of 4 fingers of the grasping hand rise above open_thresh

    Returns None if not all transitions are found (episode should be skipped).
    """
    current_subtask = 0  # index into SUBTASKS
    grasp_hand = None    # "left" or "right" once grasp detected
    labels = []

    for frame in frames:
        left = frame.get("state_hand_left")
        right = frame.get("state_hand_right")

        if current_subtask == 0:
            # Looking for grasp: >=3 of 4 fingers drop below close_thresh
            for hand_vals, hand_name in [(left, "left"), (right, "right")]:
                if hand_vals is not None:
                    fingers = hand_vals[:4]  # pinky, ring, middle, index
                    closed_count = sum(1 for v in fingers if v < close_thresh)
                    if closed_count >= 3:
                        current_subtask = 1
                        grasp_hand = hand_name
                        break

        elif current_subtask == 1:
            # Looking for release: >=3 of 4 fingers of grasping hand rise above open_thresh
            hand_vals = left if grasp_hand == "left" else right
            if hand_vals is not None:
                fingers = hand_vals[:4]
                open_count = sum(1 for v in fingers if v > open_thresh)
                if open_count >= 3:
                    current_subtask = 2

        labels.append(SUBTASKS[current_subtask])

    # All transitions must have been found
    if current_subtask < 2:
        return None

    return labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TWIST2 data to LeRobot format with subtask labels"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to a single session dir (with episode_XXXX/ folders) "
                             "or a parent dir containing multiple session dirs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output LeRobot dataset root directory "
                             "(default: ~/.cache/huggingface/lerobot/<repo_id>)")
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
    parser.add_argument("--hand_type", type=str, default="inspire", choices=["dex3", "inspire"],
                        help="Hand type: dex3 (7-DOF) or inspire (6-DOF). Default: inspire")
    parser.add_argument("--include_force", action="store_true", default=False,
                        help="Include hand force/current data as observation.force")

    # Subtask detection thresholds
    parser.add_argument("--close_thresh", type=int, default=500,
                        help="Finger value below this = closed/gripping (default: 500)")
    parser.add_argument("--open_thresh", type=int, default=700,
                        help="Finger value above this = open/released (default: 700)")

    args = parser.parse_args()

    # Set include_hand default based on action_mode
    if args.include_hand is None:
        args.include_hand = (args.action_mode == "high_level")

    # Default output_dir to HuggingFace cache
    if args.output_dir is None:
        args.output_dir = str(
            Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id
        )

    return args


def get_action_dim(action_mode: str, include_hand: bool, hand_dof: int) -> int:
    if action_mode == "high_level":
        dim = DIM_ACTION_BODY
        if include_hand:
            dim += hand_dof * 2
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
    return np.concatenate([state_body, hand_left, hand_right])


def build_action(frame: dict, idx: int, action_mode: str, include_hand: bool, hand_dof: int) -> np.ndarray:
    """Build action vector based on mode and hand flag."""
    if action_mode == "high_level":
        action = safe_array(frame.get("action_body"), DIM_ACTION_BODY, "action_body", idx)
        if include_hand:
            hand_left = safe_array(frame.get("action_hand_left"), hand_dof, "action_hand_left", idx)
            hand_right = safe_array(frame.get("action_hand_right"), hand_dof, "action_hand_right", idx)
            action = np.concatenate([action, hand_left, hand_right])
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
    dim_state = DIM_STATE_BODY + hand_dof * 2
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
    if episode_dirs:
        print(f"Found {len(episode_dirs)} episodes in {data_dir}")
    else:
        session_dirs = sorted([
            d for d in data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        for sdir in session_dirs:
            eps = sorted([
                d for d in sdir.iterdir()
                if d.is_dir() and d.name.startswith("episode_")
            ])
            episode_dirs.extend(eps)
        if not episode_dirs:
            raise FileNotFoundError(
                f"No episode_XXXX directories found in {data_dir} "
                f"(checked both as session dir and as parent of session dirs)"
            )
        print(f"Found {len(episode_dirs)} episodes across {len(session_dirs)} sessions in {data_dir}")

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
    dim_force = hand_dof * 2
    print(f"Action mode: {args.action_mode}, include_hand: {args.include_hand}, action_dim: {action_dim}")
    print(f"State dim: {dim_state}")
    if args.include_force:
        print(f"Force dim: {dim_force} (motor current, {hand_dof} per hand)")
    print(f"Subtask thresholds: close={args.close_thresh}, open={args.open_thresh}")

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
    skipped_episodes = 0
    skipped_subtask = 0
    subtask_summary = []  # per-episode subtask frame counts

    for ep_dir in tqdm(episode_dirs, desc="Converting episodes"):
        json_path = ep_dir / "data.json"
        with open(json_path) as f:
            ep_data = json.load(f)

        # Skip unsuccessful episodes
        label = ep_data.get("label", "")
        if label == "unsuccessful":
            skipped_episodes += 1
            tqdm.write(f"  Skipped {ep_dir.name}: label={label}")
            continue

        frames = ep_data["data"]

        # Detect subtask transitions
        subtask_labels = detect_subtasks(frames, args.close_thresh, args.open_thresh)
        if subtask_labels is None:
            skipped_subtask += 1
            tqdm.write(f"  Skipped {ep_dir.name}: subtask transitions not all found")
            continue

        # Count frames per subtask for this episode
        ep_subtask_counts = {}
        for st in SUBTASKS:
            ep_subtask_counts[st] = subtask_labels.count(st)
        subtask_summary.append((ep_dir.name, ep_subtask_counts))

        num_frames = len(frames)

        for i, frame in enumerate(frames):
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
                "task": subtask_labels[i],
            }

            if args.include_force:
                frame_data["observation.force"] = build_force(frame, idx, hand_dof)
            dataset.add_frame(frame_data)

        dataset.save_episode()
        total_frames += num_frames
        tqdm.write(f"  Saved {ep_dir.name}: {num_frames} frames "
                   f"[{ep_subtask_counts[SUBTASKS[0]]}/{ep_subtask_counts[SUBTASKS[1]]}/{ep_subtask_counts[SUBTASKS[2]]}]")

    print("Finalizing dataset...")
    dataset.finalize()

    if args.push_to_hub:
        print("Pushing to HuggingFace Hub...")
        dataset.push_to_hub(private=True)

    # Summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Episodes:   {len(episode_dirs) - skipped_episodes - skipped_subtask} converted")
    print(f"    Skipped (unsuccessful): {skipped_episodes}")
    print(f"    Skipped (no subtask transitions): {skipped_subtask}")
    print(f"  Frames:     {total_frames}")
    print(f"  State dim:  {dim_state}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hand type:  {args.hand_type} ({hand_dof}-DOF)")
    print(f"  Image size: {height}x{width}")
    print(f"  Action mode: {args.action_mode}")
    print(f"  Include hand: {args.include_hand}")
    print(f"  Include force: {args.include_force}")
    print(f"  Subtask thresholds: close={args.close_thresh}, open={args.open_thresh}")
    print(f"  Output:     {output_dir}")

    # Per-episode subtask breakdown
    if subtask_summary:
        print(f"\n{'Episode':<25} {'grasp':>8} {'place':>8} {'close':>8}")
        print("-" * 51)
        for ep_name, counts in subtask_summary:
            print(f"{ep_name:<25} {counts[SUBTASKS[0]]:>8} {counts[SUBTASKS[1]]:>8} {counts[SUBTASKS[2]]:>8}")

    print("=" * 60)


if __name__ == "__main__":
    main()
