#!/bin/bash
# v8: correct robomimic hyperparams.
# Previous v6/v7 used sim_mobile defaults (batch=32, lr=3e-5) instead of robomimic config
# defaults (batch=256/512, lr=2.4e-4/4.8e-4) — explains near-zero robomimic rewards.
#
# Changes vs v7:
#   EquiBot can:  batch=256, lr=2.4e-4 (from pick_place_fixed_dp.yaml defaults)
#   EquiBot nut:  batch=512, lr=4.8e-4 (from nut_assembly_fixed_dp.yaml defaults)
#   PEFM can:     batch=256, lr=1e-4   (scaled to match equibot batch size)
#   PEFM nut:     batch=512, lr=1e-4   (scaled similarly)
#   PEFM fold:    batch=32,  lr=1e-4   (same as v7; now fixed vec_eval PC shape crash)
#
# GPU layout:
#   GPU 0: PEFM fold + PEFM can + PEFM nut (PointNet, lighter)
#   GPU 1: EquiBot can + EquiBot nut       (VecPointNet, heavier)

set -u

PEFM_DIR=/home/ege/pefm/partial_equivariance
EQUIBOT_DIR=/home/ege/pefm/equibot/equibot
LOG_DIR=/home/ege/pefm/logs_runs
mkdir -p "$LOG_DIR"

SESSION=pefm_v8
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launch] session $SESSION already exists; aborting"
  exit 1
fi

CONDA="source ~/miniconda3/etc/profile.d/conda.sh && conda activate pefm"
COMMON_ENV="export MUJOCO_GL=egl PYTHONUNBUFFERED=1"

# ---------- PEFM fold (cuda:0) ----------
PEFM_FOLD="$CONDA && $COMMON_ENV PYTHONPATH=$PEFM_DIR \
  && cd $PEFM_DIR \
  && python3 -u -m pefm.train \
    --config-name fold_pefm \
    prefix=fold_pefm_v8 seed=0 device=cuda:0 \
    data.dataset.path=/home/ege/pefm/data/fold_sim_mobile/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=32 training.lr=1e-4 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    model.use_torch_compile=true \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=pefm \
  2>&1 | tee $LOG_DIR/fold_pefm_v8.log"

# ---------- PEFM can (cuda:0) ----------
PEFM_CAN="$CONDA && $COMMON_ENV PYTHONPATH=$PEFM_DIR \
  && cd $PEFM_DIR \
  && python3 -u -m pefm.train \
    --config-name pick_place_fixed_pefm \
    prefix=can_pefm_v8 seed=0 device=cuda:0 \
    data.dataset.path=/home/ege/pefm/data_rm/pick_place_fixed/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=256 training.lr=1e-4 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    model.use_torch_compile=true \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=pefm \
  2>&1 | tee $LOG_DIR/can_pefm_v8.log"

# ---------- PEFM nut (cuda:0) ----------
PEFM_NUT="$CONDA && $COMMON_ENV PYTHONPATH=$PEFM_DIR \
  && cd $PEFM_DIR \
  && python3 -u -m pefm.train \
    --config-name nut_assembly_fixed_pefm \
    prefix=nut_pefm_v8 seed=0 device=cuda:0 \
    data.dataset.path=/home/ege/pefm/data_rm/nut_assembly_fixed/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=512 training.lr=1e-4 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    model.use_torch_compile=true \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=pefm \
  2>&1 | tee $LOG_DIR/nut_pefm_v8.log"

# ---------- EquiBot can (cuda:1) — use config defaults (batch=256, lr=2.4e-4) ----------
EQB_CAN="$CONDA && $COMMON_ENV PYTHONPATH=$PEFM_DIR \
  && cd $EQUIBOT_DIR \
  && python3 -u -m equibot.policies.train \
    --config-name pick_place_fixed_equibot \
    prefix=can_equibot_v8 seed=0 device=cuda:1 \
    data.dataset.path=/home/ege/pefm/data_rm/pick_place_fixed/pcs \
    +data.dataset.num_demos=50 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=equibot \
  2>&1 | tee $LOG_DIR/can_equibot_v8.log"

# ---------- EquiBot nut (cuda:1) — use config defaults (batch=512, lr=4.8e-4) ----------
EQB_NUT="$CONDA && $COMMON_ENV PYTHONPATH=$PEFM_DIR \
  && cd $EQUIBOT_DIR \
  && python3 -u -m equibot.policies.train \
    --config-name nut_assembly_fixed_equibot \
    prefix=nut_equibot_v8 seed=0 device=cuda:1 \
    data.dataset.path=/home/ege/pefm/data_rm/nut_assembly_fixed/pcs \
    +data.dataset.num_demos=50 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=equibot \
  2>&1 | tee $LOG_DIR/nut_equibot_v8.log"

# Layout:
#   Window 0 "fold":     pane0=PEFM fold
#   Window 1 "can":      pane0=EquiBot can | pane1=PEFM can
#   Window 2 "nut":      pane0=EquiBot nut | pane1=PEFM nut

tmux new-session    -d -s "$SESSION" -n fold    "$PEFM_FOLD"
tmux new-window        -t "$SESSION"  -n can     "$EQB_CAN"
tmux split-window   -h -t "$SESSION:can"         "$PEFM_CAN"
tmux new-window        -t "$SESSION"  -n nut     "$EQB_NUT"
tmux split-window   -h -t "$SESSION:nut"         "$PEFM_NUT"
tmux select-window     -t "$SESSION:fold"

echo "[launch] session $SESSION started"
echo "  attach: tmux attach -t $SESSION"
echo "  GPU 0 → fold_pefm_v8, can_pefm_v8, nut_pefm_v8"
echo "  GPU 1 → can_equibot_v8, nut_equibot_v8"
echo "  logs:   $LOG_DIR/{fold_pefm,can_pefm,nut_pefm,can_equibot,nut_equibot}_v8.log"
