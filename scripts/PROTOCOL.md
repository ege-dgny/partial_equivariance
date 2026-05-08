# Robomimic Comparison Protocol Lockfile

This file fixes the experimental protocol for the EquiBot vs PEFM comparison
on robomimic tasks for the CoRL submission. Any deviation requires a new
entry below with rationale and a fresh commit-hash row.

## Tasks (paper alignment)

| PEFM task name | Robomimic env | EquiBot paper label |
|---|---|---|
| `pick_place_fixed` | `PickPlaceCan` | "Can" (Fig. 5) |
| `nut_assembly_fixed` | `NutAssemblySquare` | "Square" (Fig. 5) |
| `tool_hang` | `ToolHang` | not in paper — exploratory |

## Dataset

- Source: robomimic v0.1 PH `low_dim.hdf5`, downloaded via `convert_robomimic.py --download`.
  - Can:       `data/robomimic/can/ph/low_dim.hdf5`
  - Square:    `data/robomimic/square/ph/low_dim.hdf5`
  - Tool_hang: `data/robomimic/tool_hang/ph/low_dim.hdf5`
- Conversion command (per task):
  ```
  python -m pefm_envs.sim_robosuite.convert_robomimic \
      --task <can|square|tool_hang> \
      --hdf5 data/robomimic/<task>/ph/low_dim.hdf5 \
      --out_dir data_rm/<pefm_task_name> \
      --num_demos 50
  ```
- **Segmentation**: object-only via `TASK_OBJECT_KEYWORDS`
  (`convert_robomimic.py`). Drops table/walls/bins; keeps relevant objects.
  Paper §3.1 ("point clouds of relevant objects").
  - can: `[can]`
  - square: `[squarenut, peg1]`
  - tool_hang: `[tool, frame]`

## Action representation

- Stored layout: `[grip, vx, vy, vz, drx, dry, drz]`, length 7.
- Units: OSC-input (no `*freq`), clipped to ±1 at conversion time.
- Gripper convention: 0 = open, 1 = closed (mapped from robosuite −1/+1).

## Horizons

| Field | Value | Source |
|---|---|---|
| `obs_horizon` | 2 | both base configs |
| `pred_horizon` | 16 | both base configs |
| `ac_horizon` | 8 | both base configs |
| `num_points` (PC subsample) | 1024 (training) / 4096 (data file) | both |

## Training hyperparameters

| Field | PEFM | EquiBot | Source |
|---|---|---|---|
| Optimizer | AdamW | AdamW | base configs |
| Batch size | 32 | 32 | EquiBot paper |
| Learning rate | 1e-4 | 3e-5 | each method's base config |
| Epochs (nut, can) | 2000 | 2000 | EquiBot paper §4.1.2 |
| Epochs (tool_hang) | 3000 | 3000 | scaled for difficulty |
| Eval interval | 500 | 500 | matches |
| Save interval | 200 (nut/can), 300 (tool_hang) | same | matches |
| `num_demos` | 50 | 50 | EquiBot Fig. 5 middle bar |

## Eval protocol

- Vectorized eval via `vec_eval.py` in each repo.
- **Gripper logic**: hysteresis latch (`>0.9` close, `<0.1` open, hold otherwise).
  Identical in PEFM and EquiBot eval loops post-Fix 2.
- **Yaw at reset**: `randomize_rotation: false` (continuous-yaw, PH demo
  distribution) for the primary table. C4 yaw eval is a separate
  appendix-only protocol; do not enable it for main results.
- Reported metric: **final episode reward**, mean over `last 5 ckpts × 10
  episodes × 3 seeds = 150 trials per bar`.
- Seeds: `{0, 1, 2}`.

## Code commit hashes (lock for reproduction)

| Repo | Hash | Title |
|---|---|---|
| partial_equivariance | `a252f29` | chore(scripts): add sim_mobile sanity anchor |
| partial_equivariance | `85608cf` | chore(configs): lock continuous-yaw as primary robomimic protocol |
| partial_equivariance | `30d5e6c` | fix(robomimic): object-segmented PC in convert_robomimic |
| partial_equivariance | `203464a` | fix(pefm): correctness batch from prior sessions |
| equibot | `474af93` | chore(configs): lock continuous-yaw as primary robomimic protocol |
| equibot | `05719c1` | fix(equibot): gripper hysteresis parity in vec_eval and eval |
| equibot | `50b5641` | fix(equibot): data-driven pc/state normalizer for robomimic dof=7 |

These hashes correspond to the fairness fix set described in
`Plan: eventual-zooming-wolf.md`. Re-run anything against this exact set.

## Reviewer-trust sanity anchor

`scripts/sanity_sim_mobile.sh` runs one EquiBot fold task end-to-end on
sim_mobile (50 demos, 2000 epochs). The DOF=2 sim_mobile path is unaffected
by the robomimic fixes (Fix 1 only altered the DOF=7 branch). Final reward
should reach ~0.6+ (paper Fig. 4 fold, Original setup), confirming the
EquiBot install reproduces published numbers.

## Known caveats (declare in supplementary)

- **Action `A^d`**: paper §3.1 specifies normalized direction form for
  rotations; we store axis-angle. Both methods receive the same
  representation, so the comparison is apples-to-apples; the encoder
  treats the rotation as a generic 3-vector.
- **`hidden_dim` / `lr`**: PEFM 64 vs EquiBot 32; PEFM 1e-4 vs EquiBot 3e-5.
  These are each method's published defaults. Not unified to keep each
  baseline at its tuned setting.
- **Tool_hang has no published EquiBot baseline.** Report as exploratory.

## Stage A causality matrix (deferred)

Run on `nut_assembly_fixed`, single seed, 2000 epochs, 50 demos, EquiBot only:

| Run | PC seg | Norm | Gripper | Yaw | Expected |
|---|---|---|---|---|---|
| A0 (pre-fix baseline) | full-scene | action-derived | none | C4 | ~0 |
| A1 (+PC seg only) | object-only | action-derived | none | C4 | small lift |
| A2 (+norm) | object-only | data-driven | none | C4 | larger lift |
| A3 (+gripper) | object-only | data-driven | latch | C4 | small further lift |
| A4 (+yaw protocol; current main) | object-only | data-driven | latch | continuous | converge to paper |

If A4 ∈ [0.65, 0.85], the fix set is paper-equivalent for Square. Stage B (3 tasks × 3 seeds) only after this confirms.
