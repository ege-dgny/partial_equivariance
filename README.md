# Partially Equivariant Flow Matching (PEFM)

PEFM replaces rigid architectural equivariance with a learnable symmetry selector. A distribution `p_phi(g|o)` over group elements is conditioned on the observation and used to perform Monte Carlo group averaging over a non-equivariant base vector field:

```
v_PE(t, x, o) = E_{g ~ p_phi(.|o)} [ rho_out(g^-1) . v_base(t, rho_in(g).x, rho_in(g).o) ]
```

Training uses flow matching (OT conditional paths) with entropy regularization:

```
L = L_flow - lambda * H(p_phi)
```

When the task is symmetric (e.g., grasping), `p_phi` stays uniform and full equivariance is preserved. When the task breaks symmetry (e.g., placing on a fixed-position tray), `p_phi` collapses toward identity.

## Setup

Requires Python 3.10 and PyTorch >= 2.1.

```bash
conda env create -f environment.yml
conda activate pefm
pip install -e pefm_envs/
pip install -e pefm/
```

Or with pip only:

```bash
pip install -r requirements.txt
pip install -e pefm_envs/
pip install -e pefm/
```

## Structure

```
pefm/                   Model, training, evaluation
  agents/               PEFMAgent (training loop), PEFMPolicy (network composition)
  symmetry/             Group ops (SO2, C4), distributions, selector network
  flow_matching/        OT sampler, ODE solver
  networks/             ConditionalUnet1D (base vector field)
  vision/               PointNet encoder
  datasets/             NPZ episode loader
  configs/              Hydra YAML configs per task
pefm_envs/              PyBullet simulation environments
  sim_franka/           Single Franka Panda tabletop tasks (primary)
  sim_mobile/           Dual Kinova mobile-base tasks (legacy)
```

## Usage

All commands run from this directory (`Partial_Equivariance/`).

### Generate demonstrations

```bash
# Pick-and-Place (SO2 conflict: symmetric grasp, fixed-position tray)
python -m pefm_envs.sim_franka.generate_demos \
    --task_name pick_place --num_demos 50 --data_out_dir ../data/pick_place \
    --randomize_rotation

# Peg Insertion (C4 conflict: C4 peg, fixed-orientation socket)
python -m pefm_envs.sim_franka.generate_demos \
    --task_name peg_insert --num_demos 50 --data_out_dir ../data/peg_insert \
    --randomize_rotation

# Centering (SO2 fully symmetric control — no conflict, entropy stays maximal)
python -m pefm_envs.sim_franka.generate_demos \
    --task_name centering --num_demos 50 --data_out_dir ../data/centering \
    --randomize_rotation
```

Videos are recorded with 2 views (front + side) stitched side-by-side.

### Train

```bash
python -m pefm.train --config-name pick_place_pefm \
    prefix=pick_place_v1 \
    data.dataset.path=../data/pick_place/pcs \
    use_wandb=false
```

Available configs: `pick_place_pefm`, `peg_insert_pefm`, `centering_pefm`.

To log to W&B: replace `use_wandb=false` with `wandb.entity=<entity> wandb.project=<project>`.

### Evaluate

```bash
python -m pefm.eval --config-name pick_place_pefm \
    prefix=eval_pick_place_v1 mode=eval \
    training.ckpt="logs/train/pick_place_v1/ckpt01999.pth" \
    env.vectorize=true
```

### Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `model.symmetry.group_type` | `so2` | Group: `so2` (continuous) or `c4` (discrete) |
| `model.symmetry.entropy_weight` | `0.01` | Lambda for entropy regularization |
| `model.symmetry.num_samples` | `8` | MC group samples during training |
| `model.symmetry.num_samples_eval` | `16` | MC group samples during inference |
| `model.ode_steps` | `50` | ODE integration steps at inference |
| `model.symmetry.gumbel_tau` | `1.0` | Gumbel-Softmax temperature (C4 only) |

## Tasks

| Task | Group | Distribution | Symmetry conflict |
|---|---|---|---|
| Pick-and-Place | SO(2) | ProjectedNormal | SO(2) cylinder grasp vs fixed-position tray |
| Peg Insertion | C4 | GumbelSoftmax | C4 peg grasp vs fixed-orientation keyed socket |
| Centering | SO(2) | ProjectedNormal | Fully symmetric control (no conflict) |

## Robot

Single **Franka Panda** 7-DOF arm, fixed to table at origin. Uses PyBullet's built-in URDF from `pybullet_data`. Action space: `[gripper, vx, vy, vz, drx, dry, drz]`. Observation: `[eef_xyz, x_dir, z_dir, gravity, grip]` (13-dim).

This matches the lab's real Franka Panda setup for sim-to-real transfer.

## Task Status

| Task | Status | Notes |
|------|--------|-------|
| pick_place | Working | Reward threshold 0.9, reliable demos |
| centering | Partial | Works with threshold 0.5, some drift |
| peg_insert | Failing | Needs wrist rotation, collision issues |

## Real-Time Visualization

Watch demo generation live with the `--vis` flag:

```bash
python -m pefm_envs.sim_franka.generate_demos --task pick_place --num_demos 1 --vis
```

## Known Issues

- **IK offset**: `panda_hand` EE link has ~4cm Z offset from fingertip
- **Constraint drift**: Grasped objects may drift during fast movements
- **peg_insert**: Needs wrist rotation control and collision-free approach

See [docs/FRANKA_ENV.md](docs/FRANKA_ENV.md) for detailed troubleshooting.
