#!/bin/bash
# Sim_mobile sanity anchor for EquiBot install (CoRL fairness packet).
#
# WHY:
# The robomimic results comparison rests on the assumption that our local
# EquiBot install reproduces the published numbers. EquiBot paper Fig. 4
# (sim_mobile, Original setup) reports ~0.6-0.9 final reward on fold/cover/close
# at 50 synthetic demos / 2000 epochs / 3 seeds. If our build reproduces that,
# any robomimic underperformance is domain/normalization, not code regression.
#
# This script runs ONE EquiBot fold task end-to-end with no robomimic-side
# changes touched. The DOF=2 path goes through the legacy action-derived
# normalizer (Fix 1 only altered DOF=7 branch).
#
# Run:    bash /home/ege/pefm/scripts/sanity_sim_mobile.sh
# Watch:  tail -f /home/ege/pefm/logs_runs/fold_equibot_sanity.summary.log
# Expected final reward (Original setup): ~0.6+ (paper Fig. 4 fold).

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pefm

export PYTHONPATH=/home/ege/pefm/partial_equivariance:/home/ege/pefm/equibot
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export WANDB_ENTITY=ege-doganay-bilkent-niversitesi

EQUIBOT_DIR=/home/ege/pefm/equibot/equibot
DATA_DIR=/home/ege/pefm/data/fold_sim_mobile
LOG_DIR=/home/ege/pefm/logs_runs
mkdir -p "$LOG_DIR"

# ---------------- Step 1: generate demos (only if missing) ----------------
if [ ! -d "$DATA_DIR/pcs" ]; then
  echo "[sanity] generating 50 fold demos at $DATA_DIR ..."
  cd "$EQUIBOT_DIR/.."
  python -m equibot.envs.sim_mobile.generate_demos \
    --data_out_dir "$DATA_DIR" \
    --num_demos 50 \
    --cam_dist 2 \
    --cam_pitches -75 \
    --task_name fold
else
  echo "[sanity] demos already exist at $DATA_DIR, skipping generation"
fi

# ---------------- Step 2: train EquiBot fold (paper-matched) ----------------
prefix=fold_equibot_sanity
raw="$LOG_DIR/${prefix}.raw.log"
sum="$LOG_DIR/${prefix}.summary.log"
: > "$raw"; : > "$sum"

cd "$EQUIBOT_DIR"
echo "[sanity] starting EquiBot fold training (50 demos, 2000 epochs, seed=0) ..." | tee -a "$sum"

stdbuf -oL -eL python -u -m equibot.policies.train \
  --config-name fold_mobile_equibot \
  prefix="$prefix" \
  seed=0 \
  device=cuda:0 \
  data.dataset.path="$DATA_DIR/pcs" \
  +data.dataset.num_demos=50 \
  training.batch_size=32 \
  training.lr=3e-5 \
  training.num_epochs=2000 \
  training.eval_interval=500 \
  training.save_interval=200 \
  use_wandb=true \
  wandb.entity="$WANDB_ENTITY" \
  wandb.project=equibot \
  > >(tee -a "$raw" | grep -E -i 'epoch|loss|eval|reward|saved|ckpt|wandb|error|traceback|warning' | tee -a "$sum") \
  2> >(tee -a "$raw" | tee -a "$sum" >&2)

rc=$?
if [ $rc -eq 0 ]; then
  echo "[sanity] DONE — inspect final reward in $sum (target ~0.6+)" | tee -a "$sum"
else
  echo "[sanity] FAILED (exit $rc) — see $raw" | tee -a "$sum" >&2
fi
exit $rc
