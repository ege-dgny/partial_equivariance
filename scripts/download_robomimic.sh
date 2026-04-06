#!/usr/bin/env bash
# Download robomimic proficient-human (PH) demos for PEFM tasks.
#
# Usage:
#   bash scripts/download_robomimic.sh              # all tasks
#   bash scripts/download_robomimic.sh can           # single task
#   bash scripts/download_robomimic.sh can square    # multiple tasks

set -euo pipefail

BASE_URL="http://downloads.cs.stanford.edu/downloads/rt_benchmark/v0.1"
OUT_DIR="data/robomimic"

TASKS=("can" "square" "tool_hang")

if [ $# -gt 0 ]; then
    TASKS=("$@")
fi

for task in "${TASKS[@]}"; do
    dest_dir="${OUT_DIR}/${task}/ph"
    dest="${dest_dir}/low_dim.hdf5"
    url="${BASE_URL}/${task}/ph/low_dim.hdf5"

    mkdir -p "${dest_dir}"

    if [ -f "${dest}" ]; then
        echo "[skip] ${dest} already exists"
        continue
    fi

    echo "[download] ${task} -> ${dest}"
    wget -q --show-progress -O "${dest}" "${url}" || \
        curl -L -o "${dest}" "${url}"
done

echo ""
echo "=== Downloads complete ==="
ls -lh ${OUT_DIR}/*/ph/low_dim.hdf5 2>/dev/null || echo "No files found"
echo ""
echo "Convert with:"
echo "  MUJOCO_GL=egl python -m pefm_envs.sim_robosuite.convert_robomimic --task can --out_dir ../data/pick_place_fixed"
echo "  MUJOCO_GL=egl python -m pefm_envs.sim_robosuite.convert_robomimic --task square --out_dir ../data/nut_assembly_fixed"
echo "  MUJOCO_GL=egl python -m pefm_envs.sim_robosuite.convert_robomimic --task tool_hang --out_dir ../data/tool_hang"
