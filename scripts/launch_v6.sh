#!/bin/bash
# v6 launch: nut PEFM + nut EquiBot + can PEFM in tmux session "pefm_v6".
# Codepath: gripper hysteresis removed from eval/vec_eval on both sides
# (paper-faithful sim_mobile single-threshold-at-0.5 inside env only).
# Hyperparams: bare_equibot defaults (batch=32, lr=3e-5 EquiBot; lr=1e-4 PEFM).
# Preserves v3/v4/v5 results (separate prefix).

set -u

PEFM_DIR=/home/ege/pefm/partial_equivariance
EQUIBOT_DIR=/home/ege/pefm/equibot/equibot
DATA_DIR=/home/ege/pefm/data_rm
LOG_DIR=/home/ege/pefm/logs_runs
mkdir -p "$LOG_DIR"

SESSION=pefm_v6
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launch] session $SESSION already exists; aborting"
  exit 1
fi

# ---------- PEFM nut (cuda:0) ----------
PEFM_NUT="source ~/miniconda3/etc/profile.d/conda.sh && conda activate pefm \
  && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH=$PEFM_DIR \
  && cd $PEFM_DIR \
  && python3 -u -m pefm.train \
    --config-name nut_assembly_fixed_pefm \
    prefix=nut_pefm_v6 seed=0 device=cuda:0 \
    data.dataset.path=$DATA_DIR/nut_assembly_fixed/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=32 training.lr=1e-4 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    model.use_torch_compile=true \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=pefm \
  2>&1 | tee $LOG_DIR/nut_pefm_v6.log"

# ---------- EquiBot nut (cuda:1) ----------
EQB_NUT="source ~/miniconda3/etc/profile.d/conda.sh && conda activate pefm \
  && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH=$PEFM_DIR \
  && cd $EQUIBOT_DIR \
  && python3 -u -m equibot.policies.train \
    --config-name nut_assembly_fixed_equibot \
    prefix=nut_equibot_v6 seed=0 device=cuda:1 \
    data.dataset.path=$DATA_DIR/nut_assembly_fixed/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=32 training.lr=3e-5 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=equibot \
  2>&1 | tee $LOG_DIR/nut_equibot_v6.log"

# ---------- PEFM can (cuda:0, shares GPU with PEFM nut) ----------
PEFM_CAN="source ~/miniconda3/etc/profile.d/conda.sh && conda activate pefm \
  && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH=$PEFM_DIR \
  && cd $PEFM_DIR \
  && python3 -u -m pefm.train \
    --config-name pick_place_fixed_pefm \
    prefix=can_pefm_v6 seed=0 device=cuda:0 \
    data.dataset.path=$DATA_DIR/pick_place_fixed/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=32 training.lr=1e-4 \
    training.num_epochs=2000 training.eval_interval=500 training.save_interval=200 \
    model.use_torch_compile=true \
    use_wandb=true wandb.entity=ege-doganay-bilkent-niversitesi wandb.project=pefm \
  2>&1 | tee $LOG_DIR/can_pefm_v6.log"

tmux new-session -d -s "$SESSION" -n nut "$PEFM_NUT"
tmux split-window -h -t "$SESSION:nut" "$EQB_NUT"
tmux new-window -t "$SESSION" -n can "$PEFM_CAN"
tmux select-window -t "$SESSION:nut"

echo "[launch] session $SESSION started"
echo "  attach: tmux attach -t $SESSION"
echo "  panes:"
echo "    nut.0 = PEFM nut    (cuda:0)  log: $LOG_DIR/nut_pefm_v6.log"
echo "    nut.1 = EquiBot nut (cuda:1)  log: $LOG_DIR/nut_equibot_v6.log"
echo "    can.0 = PEFM can    (cuda:0)  log: $LOG_DIR/can_pefm_v6.log"
echo "  wandb: pefm/{nut_pefm_v6,can_pefm_v6} + equibot/nut_equibot_v6"
