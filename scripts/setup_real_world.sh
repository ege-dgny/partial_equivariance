#!/usr/bin/env bash
# ===================================================================
# setup_real_world.sh — one-command install for the PEFM real-world
# human-demo parsing pipeline.
#
# Installs into:
#   third_party/   (cloned repos — gitignored)
#   weights/       (model checkpoints — gitignored)
#
# Usage (from Partial_Equivariance/):
#   bash scripts/setup_real_world.sh
#
# Optional env-var overrides:
#   SKIP_DEVA=1        skip DEVA (tracking is a stub for now)
#   DEVICE=cpu         set default device hint in printed env vars
# ===================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY="$ROOT_DIR/third_party"
WEIGHTS="$ROOT_DIR/weights"

# Detect architecture (arm64 = Apple Silicon M1/M2, x86_64 = Intel/AMD)
ARCH="$(uname -m)"
export PEFM_ARCH="${ARCH}"

mkdir -p "$THIRD_PARTY" "$WEIGHTS"

echo ""
echo "============================================="
echo "  PEFM Real-World Pipeline Setup"
echo "  root: $ROOT_DIR"
echo "  arch: $ARCH"
echo "============================================="
echo ""

# ------------------------------------------------------------------
# 1. Segment Anything (SAM)
# ------------------------------------------------------------------
echo "=== [1/4] Segment Anything (SAM) ==="

if [ ! -d "$THIRD_PARTY/segment-anything" ]; then
    echo "  Cloning segment-anything …"
    git clone --depth 1 https://github.com/facebookresearch/segment-anything.git \
        "$THIRD_PARTY/segment-anything"
    pip install -e "$THIRD_PARTY/segment-anything"
else
    echo "  Already cloned."
fi

if [ ! -f "$WEIGHTS/sam_vit_h_4b8939.pth" ]; then
    echo "  Downloading SAM ViT-H checkpoint (~2.4 GB) …"
    wget -q --show-progress -O "$WEIGHTS/sam_vit_h_4b8939.pth" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
    echo "  Checkpoint already present."
fi
echo ""

# ------------------------------------------------------------------
# 2. Grounding DINO
# ------------------------------------------------------------------
echo "=== [2/4] Grounding DINO ==="

if [ ! -d "$THIRD_PARTY/GroundingDINO" ]; then
    echo "  Cloning GroundingDINO …"
    git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git \
        "$THIRD_PARTY/GroundingDINO"
    echo "  Ensuring torch is available (avoids build-isolation failure) …"
    pip install -q torch torchvision
    echo "  Installing Grounding DINO (no build isolation) …"
    pip install --no-build-isolation -e "$THIRD_PARTY/GroundingDINO"
else
    echo "  Already cloned."
    if ! python -c "import groundingdino" 2>/dev/null; then
        echo "  Installing Grounding DINO (no build isolation) …"
        pip install -q torch torchvision
        pip install --no-build-isolation -e "$THIRD_PARTY/GroundingDINO"
    fi
fi

if [ ! -f "$WEIGHTS/groundingdino_swint_ogc.pth" ]; then
    echo "  Downloading Grounding DINO SwinT checkpoint (~662 MB) …"
    wget -q --show-progress -O "$WEIGHTS/groundingdino_swint_ogc.pth" \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "  Checkpoint already present."
fi
echo ""

# ------------------------------------------------------------------
# 3. HaMeR  (+ Detectron2, ViTPose)
# ------------------------------------------------------------------
echo "=== [3/4] HaMeR (+ Detectron2 + ViTPose) ==="

if [ "$ARCH" = "arm64" ]; then
    echo "  Apple Silicon (arm64) detected."
    echo "  Attempting HaMeR install (Detectron2 builds from source with --no-build-isolation)."
fi

if [ ! -d "$THIRD_PARTY/hamer" ]; then
    echo "  Cloning HaMeR (recursive for ViTPose) …"
    git clone --recursive https://github.com/geopavlakos/hamer.git \
        "$THIRD_PARTY/hamer"
fi

# Detectron2 must be built with --no-build-isolation so its setup.py sees torch (isolated env has no torch).
echo "  Ensuring torch/torchvision available …"
pip install -q torch torchvision

echo "  Installing Detectron2 (no build isolation so torch is visible) …"
if pip install --no-build-isolation 'detectron2@ git+https://github.com/facebookresearch/detectron2'; then
    echo "  Detectron2 installed."
else
    echo "  Detectron2 install failed; will try full HaMeR install anyway."
fi

# xtcocotools (HaMeR/mmpose dep) needs Cython to generate _mask.c from _mask.pyx; pre-install so HaMeR's pip sees it.
echo "  Installing Cython (required to build xtcocotools) …"
pip install -q cython
echo "  Pre-installing xtcocotools (no build isolation so Cython generates _mask.c) …"
pip install --no-build-isolation xtcocotools || true

echo "  Installing HaMeR …"
if pip install --no-build-isolation -e "$THIRD_PARTY/hamer[all]"; then
    echo "  HaMeR installed successfully."
else
    echo "  HaMeR install failed. Pipeline will use mock hand detector until HaMeR is installed."
    echo "  To retry manually: pip install -q torch torchvision && pip install --no-build-isolation -e $THIRD_PARTY/hamer[all]"
fi

if [ -d "$THIRD_PARTY/hamer/third-party/ViTPose" ]; then
    echo "  Installing ViTPose (no build isolation for chumpy/mmpose) …"
    pip install --no-build-isolation -v -e "$THIRD_PARTY/hamer/third-party/ViTPose" || true
fi

if [ -d "$THIRD_PARTY/hamer" ]; then
    if [ ! -f "$THIRD_PARTY/hamer/_DATA/hamer_demo_data.tar.gz" ] && \
       [ ! -d "$THIRD_PARTY/hamer/_DATA/data" ]; then
        echo "  Downloading HaMeR demo data (~6 GB; can skip with Ctrl+C and run later) …"
        (cd "$THIRD_PARTY/hamer" && bash fetch_demo_data.sh) || echo "  Demo data download skipped or failed. Run later: cd $THIRD_PARTY/hamer && bash fetch_demo_data.sh"
    else
        echo "  Demo data already present."
    fi

    # MANO model (requires manual registration)
    MANO_PATH="$THIRD_PARTY/hamer/_DATA/data/mano/MANO_RIGHT.pkl"
    if [ ! -f "$MANO_PATH" ]; then
        echo ""
        echo "  ⚠  MANO_RIGHT.pkl not found!"
        echo "     1. Register at https://mano.is.tue.mpg.de"
        echo "     2. Download MANO_RIGHT.pkl"
        echo "     3. Place it at: $MANO_PATH"
        echo ""
    fi
fi
echo ""

# ------------------------------------------------------------------
# 4. DEVA (optional — tracking stub for now)
# ------------------------------------------------------------------
SKIP_DEVA="${SKIP_DEVA:-0}"
echo "=== [4/4] DEVA (optional video tracking) ==="

if [ "$SKIP_DEVA" = "1" ]; then
    echo "  Skipped (SKIP_DEVA=1)."
elif [ ! -d "$THIRD_PARTY/Tracking-Anything-with-DEVA" ]; then
    echo "  Cloning DEVA …"
    git clone --depth 1 https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git \
        "$THIRD_PARTY/Tracking-Anything-with-DEVA"
    pip install -e "$THIRD_PARTY/Tracking-Anything-with-DEVA"
    echo "  Downloading DEVA checkpoints …"
    (cd "$THIRD_PARTY/Tracking-Anything-with-DEVA" && bash scripts/download_models.sh)
else
    echo "  Already cloned."
fi
echo ""

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
if [ "$ARCH" = "arm64" ]; then
    DEVICE="${DEVICE:-cpu}"
    echo "  (Apple Silicon: default device set to 'cpu'; use --device cpu when running.)"
else
    DEVICE="${DEVICE:-cuda}"
fi

echo "============================================="
echo "  Setup complete!"
echo "============================================="
echo ""
echo "Add these to your shell profile or run before processing:"
echo ""
echo "  export SAM_CHECKPOINT=$WEIGHTS/sam_vit_h_4b8939.pth"
echo "  export SAM_MODEL_TYPE=vit_h"
echo "  export GROUNDING_DINO_CHECKPOINT=$WEIGHTS/groundingdino_swint_ogc.pth"
echo "  export GROUNDING_DINO_CONFIG=$THIRD_PARTY/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
echo "  export HAMER_DIR=$THIRD_PARTY/hamer"
echo ""
echo "Then process demos:"
echo ""
echo "  cd $ROOT_DIR"
echo "  python -m real_world.process_demos \\"
echo "    --input_dir data_rw/book_shelf_vert \\"
echo "    --output_dir data_rw/book_shelf_vert_processed \\"
echo "    --object_prompt \"book. shelf.\" \\"
echo "    --use_human_parsing \\"
echo "    --device $DEVICE"
echo ""
