#!/bin/bash
# Launch nut_v4 PEFM + EquiBot side-by-side in tmux session "pefm_v4".
# Uses the post-fairness-fix codepath (object-segmented PCs, data-driven
# normalizer, gripper hysteresis parity, continuous-yaw eval).
# Preserves v3 results (different prefix).
#
# Run on the LIRA server.

set -u

PEFM_DIR=/home/ege/pefm/partial_equivariance
EQUIBOT_DIR=/home/ege/pefm/equibot/equibot
DATA_DIR=/home/ege/pefm/data_rm
LOG_DIR=/home/ege/pefm/logs_runs
mkdir -p "$LOG_DIR"

# Activate conda env in each pane and start training
SESSION=pefm_v4

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launch] session $SESSION already exists; aborting"
  exit 1
fi

PEFM_CMD="source ~/miniconda3/etc/profile.d/conda.sh && conda activate pefm \
  && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH=$PEFM_DIR \
  && cd $PEFM_DIR \
  && python3 -u -m pefm.train \
    --config-name nut_assembly_fixed_pefm \
    prefix=nut_pefm_v4 \
    seed=0 \
    device=cuda:0 \
    data.dataset.path=$DATA_DIR/nut_assembly_fixed/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=32 \
    training.lr=1e-4 \
    training.num_epochs=2000 \
    training.eval_interval=500 \
    training.save_interval=200 \
    model.use_torch_compile=true \
    use_wandb=true \
    wandb.entity=ege-doganay-bilkent-niversitesi \
    wandb.project=pefm \
  2>&1 | tee $LOG_DIR/nut_pefm_v4.log"

EQUIBOT_CMD="source ~/miniconda3/etc/profile.d/conda.sh && conda activate pefm \
  && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH=$PEFM_DIR \
  && cd $EQUIBOT_DIR \
  && python3 -u -m equibot.policies.train \
    --config-name nut_assembly_fixed_equibot \
    prefix=nut_equibot_v4 \
    seed=0 \
    device=cuda:1 \
    data.dataset.path=$DATA_DIR/nut_assembly_fixed/pcs \
    +data.dataset.num_demos=50 \
    training.batch_size=32 \
    training.lr=3e-5 \
    training.num_epochs=2000 \
    training.eval_interval=500 \
    training.save_interval=200 \
    use_wandb=true \
    wandb.entity=ege-doganay-bilkent-niversitesi \
    wandb.project=equibot \
  2>&1 | tee $LOG_DIR/nut_equibot_v4.log"

# Create tmux session: 1 window, 2 panes (left: PEFM, right: EquiBot)
tmux new-session -d -s "$SESSION" -n nut "$PEFM_CMD"
tmux split-window -h -t "$SESSION:nut" "$EQUIBOT_CMD"
tmux select-layout -t "$SESSION:nut" even-horizontal

echo "[launch] tmux session '$SESSION' started"
echo "  attach:  tmux attach -t $SESSION"
echo "  detach:  Ctrl-b d"
echo "  pefm log:    $LOG_DIR/nut_pefm_v4.log"
echo "  equibot log: $LOG_DIR/nut_equibot_v4.log"
echo "  wandb:   pefm/nut_pefm_v4  +  equibot/nut_equibot_v4"
