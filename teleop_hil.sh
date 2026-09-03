#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniconda3/bin/activate" gmr
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${script_dir}/deploy_real"
python xrobot_teleop_to_robot_w_hand.py \
    --robot unitree_g1 \
    --actual_human_height 1.6 \
    --redis_ip localhost \
    --target_fps 100 \
    --measure_fps 1 \
    --hand_type inspire \
    --hil_controller_only
