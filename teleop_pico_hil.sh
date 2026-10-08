#!/usr/bin/env bash
set -euo pipefail
# Activate the lab's gmr/XR SDK environment before running. Never start the
# whole-body teleop writer alongside this sensor-only source.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PICO_PYTHON:-python}" "${script_dir}/deploy_real/pico_intervention_source.py" "$@"
