# Partially Equivariant Flow Matching (PEFM)

PEFM replaces rigid architectural equivariance with a learnable symmetry selector. A distribution `p_phi(g|o)` over group elements is conditioned on the observation and used to perform Monte Carlo group averaging over a non-equivariant base vector field:

```
v_PE(t, x, o) = E_{g ~ p_phi(.|o)} [ rho_out(g^-1) . v_base(t, rho_in(g).x, rho_in(g).o) ]
```

Training uses flow matching (OT conditional paths) with entropy regularization:

```
L = L_flow - lambda * H(p_phi)
```

When the task is symmetric (e.g., grasping), `p_phi` stays uniform and full equivariance is preserved. When the task breaks symmetry (e.g., pouring into a fixed-position bowl), `p_phi` collapses toward identity.

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
  sim_mobile/           Pouring, insertion, compass closing tasks
```

## Usage

All commands run from this directory (`Partial_Equivariance/`).

### Generate demonstrations

```bash
# PEFM tasks
python -m pefm_envs.sim_mobile.generate_demos \
    --task_name pour --num_demos 50 --data_out_dir ../data/pour --randomize_rotation

python -m pefm_envs.sim_mobile.generate_demos \
    --task_name insert --num_demos 50 --data_out_dir ../data/insert --randomize_rotation

python -m pefm_envs.sim_mobile.generate_demos \
    --task_name compass_close --num_demos 50 --data_out_dir ../data/compass_close --randomize_rotation
```

### Train

```bash
python -m pefm.train --config-name pour_pefm \
    prefix=pour_v1 \
    data.dataset.path=../data/pour/pcs \
    use_wandb=false
```

Available configs: `pour_pefm`, `insert_pefm`, `compass_close_pefm`.

To log to W&B: replace `use_wandb=false` with `wandb.entity=<entity> wandb.project=<project>`.

### Evaluate

```bash
python -m pefm.eval --config-name pour_pefm \
    prefix=eval_pour_v1 mode=eval \
    training.ckpt="logs/train/pour_v1/ckpt01999.pth" \
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
| Pouring | SO(2) | ProjectedNormal | C-inf grasp symmetry vs fixed bowl position |
| Insertion | C4 | GumbelSoftmax | C4 peg symmetry vs keyed socket orientation |
| Compass Closing | SO(2) | ProjectedNormal | Full geometric symmetry vs cardinal flap order |
