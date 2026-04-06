#!/usr/bin/env bash
# Setup PEFM conda environment on the HP server.
#
# Target: 2x NVIDIA RTX PRO 6000 Blackwell, CUDA 13.1, Linux
#
# Usage:
#   bash scripts/setup_pefm_server.sh
#
# This creates a fresh 'pefm' conda env — does NOT modify existing envs.

set -euo pipefail

ENV_NAME="pefm"
PYTHON_VERSION="3.10"

echo "=== PEFM Server Setup ==="
echo "Creating conda env: ${ENV_NAME} (Python ${PYTHON_VERSION})"

# ---- 1. Create conda environment ----
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ---- 2. PyTorch (CUDA 13.1) ----
# Check for CUDA 13.1 compatible wheel. If the stable index doesn't have
# cu131, fall back to nightly or cu124 (should still work with 13.1 driver).
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131 2>/dev/null \
  || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
  || pip install torch torchvision torchaudio

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}')"

# ---- 3. MuJoCo + robosuite ----
echo "Installing MuJoCo and robosuite..."
pip install mujoco>=3.0
pip install robosuite==1.5.1

# ---- 4. Core dependencies ----
echo "Installing PEFM dependencies..."
pip install \
  numpy scipy einops diffusers hydra-core omegaconf wandb \
  cloudpickle tqdm opencv-python gym

# ---- 5. Install PEFM packages (editable) ----
echo "Installing pefm_envs and pefm packages..."
cd "$(dirname "$0")/.."  # Partial_Equivariance/
pip install -e "pefm_envs/[robosuite]"
pip install -e pefm/

# ---- 6. Environment variables for headless rendering ----
echo "Setting up MuJoCo EGL rendering..."
# Add to conda activate script so these are set automatically
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${ACTIVATE_DIR}"
cat > "${ACTIVATE_DIR}/pefm_env_vars.sh" << 'ENVEOF'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Use GPU 0 for rendering by default; set EGL_DEVICE_ID=1 for GPU 1
export EGL_DEVICE_ID=0
ENVEOF

# Also install EGL support
pip install PyOpenGL PyOpenGL_accelerate 2>/dev/null || true

# ---- 7. Verification ----
echo ""
echo "=== Verification ==="

python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
"

python -c "import mujoco; print(f'  MuJoCo: {mujoco.__version__}')"
python -c "import robosuite; print(f'  robosuite: {robosuite.__version__}')"

# Test robosuite env creation + rendering
MUJOCO_GL=egl python -c "
import robosuite as suite
env = suite.make(
    'Lift',
    robots='Panda',
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    use_object_obs=True,
)
obs = env.reset()
print(f'  robosuite Lift env: OK (obs keys: {len(obs)} keys)')
img = env.sim.render(camera_name='agentview', width=240, height=240)
print(f'  EGL offscreen render: OK ({img.shape})')
env.close()
"

python -c "import pefm; print(f'  pefm package: OK')"
python -c "import pefm_envs; print(f'  pefm_envs package: OK')"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "EGL rendering is configured automatically on activation."
echo ""
echo "Quick start:"
echo "  # Generate demos"
echo "  python -m pefm_envs.sim_robosuite.generate_demos --task_name pick_place_fixed --num_demos 50"
echo ""
echo "  # Train"
echo "  python -m pefm.train --config-name pick_place_fixed_pefm prefix=pp_v1 use_wandb=false device=cuda:0"
