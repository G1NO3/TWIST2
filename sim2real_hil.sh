#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniconda3/bin/activate" twist2_deploy
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkpoint="${script_dir}/assets/ckpts/twist2_1017_20k.onnx"
urdf="${script_dir}/assets/g1/g1_29dof_rev_1_0.urdf"

cd "${script_dir}/deploy_real"
python server_low_level_g1_real.py \
    --policy "${checkpoint}" \
    --net enp128s31f6 \
    --device cuda \
    --use_hand \
    --hand_type inspire \
    --check_stale \
    --hil_safety \
    --safety_urdf "${urdf}"
