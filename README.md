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

For **Franka tabletop tasks**, the primary backend is **Genesis** (contact-based grasping, collisions ON). Install with: `pip install genesis-world` or `pip install -e "pefm_envs[genesis]/"`. PyBullet envs (`sim_franka`) remain available; set `env.env_class=peg_insert` (etc.) to use them.

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
pefm_envs/              Simulation environments
  sim_genesis/          Genesis Franka tabletop (primary; contact-based grasping)
  sim_franka/           PyBullet Franka tabletop (fallback)
  sim_mobile/           Dual Kinova mobile-base tasks (legacy)
```

## Usage

All commands run from this directory (`Partial_Equivariance/`).

### Generate demonstrations

Franka demos use **Genesis** (contact-based grasping). From `Partial_Equivariance/`:

```bash
python -m pefm_envs.sim_genesis.generate_demos \
    --task_name peg_insert --num_demos 50 --data_out_dir ../data/peg_insert --randomize_rotation
python -m pefm_envs.sim_genesis.generate_demos \
    --task_name cup_pour --num_demos 50 --data_out_dir ../data/cup_pour --randomize_rotation
python -m pefm_envs.sim_genesis.generate_demos \
    --task_name book_insert --num_demos 50 --data_out_dir ../data/book_insert --randomize_rotation
```

Videos: 2 views (front + side). Configs default to `*_genesis` envs.

### Train

```bash
python -m pefm.train --config-name peg_insert_pefm \
    prefix=peg_insert_v1 \
    data.dataset.path=../data/peg_insert/pcs \
    use_wandb=false
```

Available configs: `peg_insert_pefm`, `cup_pour_pefm`, `book_insert_pefm`, `push_t_gym_pefm`.

To log to W&B: replace `use_wandb=false` with `wandb.entity=<entity> wandb.project=<project>`.

### Evaluate

```bash
python -m pefm.eval --config-name peg_insert_pefm \
    prefix=eval_peg_insert_v1 mode=eval \
    training.ckpt="logs/train/peg_insert_v1/ckpt01999.pth" \
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
| Peg Insertion | C4 | GumbelSoftmax | C4 peg grasp vs fixed-orientation keyed socket |
| Cup Pour | SO(2) | ProjectedNormal | Grasp vs gravity-fixed pour |
| Book Insert | — | — | Side-grasp book to shelf |

## Robot

Single **Franka Panda** 7-DOF arm. **Primary backend: Genesis** (contact-based grasping, collisions ON). Action: `[gripper, vx, vy, vz, drx, dry, drz]`. Obs: `[eef_xyz, x_dir, z_dir, gravity, grip]` (13-dim). PyBullet (`sim_franka`) available as fallback.

## Task Status

| Task | Status | Notes |
|------|--------|-------|
| peg_insert | Genesis | Contact-based; config uses `peg_insert_genesis` |
| cup_pour | Genesis | Contact-based |
| book_insert | Genesis | Contact-based |

## Real-Time Visualization

Watch demo generation live with the `--vis` flag:

```bash
python -m pefm_envs.sim_genesis.generate_demos --task_name peg_insert --num_demos 1 --vis
```

## Known Issues

- **Genesis**: Requires `genesis-world`; GPU backend preferred. Render/point-cloud may need camera sensor integration for full demo quality.
- **PyBullet**: See [docs/FRANKA_ENV.md](docs/FRANKA_ENV.md) (IK offset, constraint drift) when using `env.env_class=peg_insert` etc.
