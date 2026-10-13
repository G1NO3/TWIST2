#!/usr/bin/env bash
# Sensor-only: no Redis, DDS, low-level host or robot command publisher.
set -euo pipefail
twist_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname -- "$twist_dir")"
gmr_python="${GMR_TEST_PY:-$workspace_dir/.venv-gmr/bin/python}"
gmr_root="${GMR_ROOT:-$workspace_dir/GMR}"
# Reuse the EXACT Python 3.10 SDK binary used by the GUI source. Do not rebuild
# or upgrade it while comparing the two input paths. No SDK init in preflight.
pico_sdk_path="${PICO_SDK_SITE:-$workspace_dir/.venv-pico/lib/python3.10/site-packages}"
export PYTHONPATH="$gmr_root:$pico_sdk_path${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$gmr_python" "$twist_dir/deploy_real/check_pico_gmr.py" "$@"
