"""
Visualize what the PEFM symmetry selector learned for a real observation.

Usage:
  # C4 run (nut_assembly):
  python visualize_selector.py \
    --run_dir logs/train/nut_assembly_50demos_v1 \
    --ckpt ckpt01999.pth \
    --data_dir /home/ege/pefm/data/nut_assembly_fixed/pcs \
    --out_dir figs/selector_nut

  # SO2 run (pick_place):
  python visualize_selector.py \
    --run_dir logs/train/pick_place_50demos_v1 \
    --ckpt ckpt01999.pth \
    --data_dir /home/ege/pefm/data/pick_place_fixed/pcs \
    --out_dir figs/selector_pick
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# ── resolve imports relative to this file's directory ──────────────────────
sys.path.insert(0, os.path.dirname(__file__))


def load_run(run_dir, ckpt_name, device):
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg.device = device

    # num_training_steps is ??? in saved config; dummy value satisfies the check
    OmegaConf.update(cfg, "data.dataset.num_training_steps", 1, merge=True)
    # checkpoint was saved without compiled submodule keys; skip recompile on load
    OmegaConf.update(cfg, "model.use_torch_compile", False, merge=True)

    from pefm.agents.pefm_agent import PEFMAgent
    agent = PEFMAgent(cfg)
    agent.load_snapshot(os.path.join(run_dir, ckpt_name))
    agent.actor.eval()
    return agent, cfg


def load_obs(data_dir, episode=0, obs_horizon=2, num_points=1024):
    """Load obs_horizon consecutive frames from one episode."""
    pattern = os.path.join(data_dir, f"*ep{episode:06d}_view0_t????.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        # fallback: just grab first obs_horizon files
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    files = files[:obs_horizon]

    pcs, states = [], []
    for f in files:
        d = np.load(f)
        pc = d["pc"]  # (4096, 3)
        rng = np.random.default_rng(42)
        idx = rng.choice(pc.shape[0], num_points, replace=False)
        pcs.append(pc[idx])
        states.append(d["eef_pos"].squeeze(0))  # (13,)

    pc_arr = torch.tensor(np.stack(pcs), dtype=torch.float32)      # (obs_h, P, 3)
    state_arr = torch.tensor(np.stack(states), dtype=torch.float32) # (obs_h, 13)
    return pc_arr.unsqueeze(0), state_arr.unsqueeze(0)              # (1, ...)


@torch.no_grad()
def get_selector_output(agent, pc, state, n_samples=2000):
    """Return raw params, sampled group elements, and entropy for one obs."""
    actor = agent.actor
    device = agent.device

    pc = agent.pc_normalizer.normalize(pc.to(device))
    state = agent.state_normalizer.normalize(state.to(device))

    ema_enc = actor.ema.averaged_model["encoder"]
    ema_sel = actor.ema.averaged_model["selector"]

    obs_cond = actor._encode_obs(pc, state, ema_enc)           # (1, obs_cond_dim)
    params = ema_sel(obs_cond)                                  # (1, param_dim)
    g_samples, entropy = ema_sel.sample_and_entropy(obs_cond, n_samples)
    # g_samples: (1, n_samples)

    return params.cpu(), g_samples.squeeze(0).cpu(), entropy.cpu()


def plot_c4(params, g_samples, entropy, out_path):
    """Bar chart of C4 class probabilities."""
    import torch.nn.functional as F

    # params: (1, 4) — raw logits (pre-softmax)
    probs = F.softmax(params[0], dim=-1).numpy()
    labels = ["0°", "90°", "180°", "270°"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, probs, color="#4C72B0", width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(f"Symmetry Selector — C4 (nut_assembly)\nEntropy = {entropy.item():.3f}")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Uniform (H=max)")
    ax.legend(fontsize=8)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.015,
                f"{p:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_so2(g_samples, entropy, out_path, n_bins=36):
    """Polar rose histogram of SO2 sampled angles."""
    angles = g_samples.numpy()  # radians in (-pi, pi] or [0, 2pi)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(4.5, 4.5))
    counts, bin_edges = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = 2 * np.pi / n_bins
    bars = ax.bar(bin_centers, counts / counts.sum(), width=width,
                  color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_title(f"Symmetry Selector — SO(2) (pick_place)\nEntropy = {entropy.item():.3f}",
                 pad=14)
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    uniform_r = 1 / n_bins
    ax.axhline(uniform_r, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ckpt", default="ckpt01999.pth")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="figs/selector")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    agent, cfg = load_run(args.run_dir, args.ckpt, args.device)
    group_type = cfg.model.symmetry.group_type

    pc, state = load_obs(args.data_dir, episode=args.episode,
                         obs_horizon=cfg.model.obs_horizon)

    params, g_samples, entropy = get_selector_output(agent, pc, state, args.n_samples)
    print(f"Group: {group_type} | Entropy: {entropy.item():.4f} | "
          f"Params: {params[0].tolist()}")

    os.makedirs(args.out_dir, exist_ok=True)
    if group_type == "c4":
        plot_c4(params, g_samples, entropy,
                os.path.join(args.out_dir, "c4_distribution.png"))
    else:
        plot_so2(g_samples, entropy,
                 os.path.join(args.out_dir, "so2_distribution.png"))


if __name__ == "__main__":
    main()
