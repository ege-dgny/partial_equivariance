#!/bin/bash
# v7 launch: sim_mobile sanity (fold for both methods) + EquiBot can.
#   PEFM fold     = object-centric anchor (PEFM matches EquiBot here per paper claim)
#   EquiBot fold  = reproducibility anchor (verify install vs paper Fig 4)
#   EquiBot can   = completes the can comparison alongside v6 PEFM can
#
# Hyperparams: bare_equibot paper defaults (batch=32, lr=3e-5 EquiBot; lr=1e-4 PEFM).
# Preserves v3/v4/v5/v6.

set -u

PEFM_DIR=/home/ege/pefm/partial_equivariance
EQUIBOT_DIR=/home/ege/pefm/equibot/equibot
LOG_DIR=/home/ege/pefm/logs_runs
mkdir -p "$LOG_DIR"

SESSION=pefm_v7
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launch] session $SESSION already exists; aborting"
  exit 1
fi

# ---------- PEFM fold (cuda:0) ----------
PEFM_FOLD="source ~/miniconda3/etc/profile.d/conda.sh && conda activate pefm \
  && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH=$PEFM_DIR \
  && cd $PEFM_DIR \
  && python3 -u -m pefm.train \
    --config-name fold_pefm \
    prefix=fold_pefm_v7 seed=0 device=cuda:0 \
    data.dataset.path=/home/ege/pefm/data/fold_sim_mobile/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=32 training.lr=1e-4 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    model.use_torch_compile=true \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=pefm \
  2>&1 | tee $LOG_DIR/fold_pefm_v7.log"

# ---------- EquiBot fold (cuda:1) ----------
EQB_FOLD="source ~/miniconda3/etc/profile.d/conda.sh && conda activate pefm \
  && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH=$PEFM_DIR \
  && cd $EQUIBOT_DIR \
  && python3 -u -m equibot.policies.train \
    --config-name fold_mobile_equibot \
    prefix=fold_equibot_v7 seed=0 device=cuda:1 \
    data.dataset.path=/home/ege/pefm/data/fold_sim_mobile/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=32 training.lr=3e-5 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=equibot \
  2>&1 | tee $LOG_DIR/fold_equibot_v7.log"

# ---------- EquiBot can (cuda:1, shares GPU with EquiBot fold) ----------
EQB_CAN="source ~/miniconda3/etc/profile.d/conda.sh && conda activate pefm \
  && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH=$PEFM_DIR \
  && cd $EQUIBOT_DIR \
  && python3 -u -m equibot.policies.train \
    --config-name pick_place_fixed_equibot \
    prefix=can_equibot_v7 seed=0 device=cuda:1 \
    data.dataset.path=/home/ege/pefm/data_rm/pick_place_fixed/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=32 training.lr=3e-5 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=equibot \
  2>&1 | tee $LOG_DIR/can_equibot_v7.log"

tmux new-session -d -s "$SESSION" -n fold "$PEFM_FOLD"
tmux split-window -h -t "$SESSION:fold" "$EQB_FOLD"
tmux new-window -t "$SESSION" -n can "$EQB_CAN"
tmux select-window -t "$SESSION:fold"

echo "[launch] session $SESSION started"
echo "  attach: tmux attach -t $SESSION"
echo "  panes:"
echo "    fold.0 = PEFM fold    (cuda:0)  log: $LOG_DIR/fold_pefm_v7.log"
echo "    fold.1 = EquiBot fold (cuda:1)  log: $LOG_DIR/fold_equibot_v7.log"
echo "    can.0  = EquiBot can  (cuda:1)  log: $LOG_DIR/can_equibot_v7.log"
echo "  wandb: pefm/fold_pefm_v7 + equibot/{fold_equibot_v7,can_equibot_v7}"
