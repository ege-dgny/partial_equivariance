"""
Polar visualization of the learned selector distribution p_phi(g|o) per task
(Phase 4). Uses the per-step `selector_params` cached in rollout NPZs (saved
by Phase 1.4 of the instrumentation). No env access, no re-encoding — the
params are exactly what the policy produced at eval time.

Inputs:
    --rollout_dir   path to a directory containing eval_*_ep*_rew*.npz files
    --picks         comma-separated timestep indices OR phase labels (auto-mapped)
    --distribution  so2 | c4    (auto-detected from selector_params.shape[-1] if omitted)
    --n_samples     int (default 1000)  number of polar samples for SO(2)
    --out           output figure path (.pdf or .png)

The plot is a side-by-side panel of polar histograms (SO(2)) or bar charts
(C4), one per picked timestep, with H(p_phi) annotated. A reader sees
"uniform ring = equivariant" vs "Dirac spike = symmetry broken" at a glance.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HAAR_BOUND = math.log(2 * math.pi)


# ---- SO(2) projected-normal sampling (numpy-only mirror of distributions.py) ----

def _sample_so2(mu_u: float, mu_v: float, log_sigma_u: float, log_sigma_v: float,
                n: int) -> np.ndarray:
    """Return n angles in (-pi, pi] from ProjectedNormalSO2 with the given params."""
    log_sigma_u = float(np.clip(log_sigma_u, -4, 4))
    log_sigma_v = float(np.clip(log_sigma_v, -4, 4))
    sigma_u = math.exp(log_sigma_u)
    sigma_v = math.exp(log_sigma_v)
    u = mu_u + sigma_u * np.random.randn(n)
    v = mu_v + sigma_v * np.random.randn(n)
    return np.arctan2(v, u)


def _entropy_so2(mu_u, mu_v, log_sigma_u, log_sigma_v) -> float:
    """Capped projected-normal entropy approximation (matches distributions.py)."""
    log_sigma_u = float(np.clip(log_sigma_u, -4, 4))
    log_sigma_v = float(np.clip(log_sigma_v, -4, 4))
    sigma_u = math.exp(log_sigma_u)
    sigma_v = math.exp(log_sigma_v)
    conc_sq = (mu_u / sigma_u) ** 2 + (mu_v / sigma_v) ** 2
    h = math.log(2 * math.pi) + 0.5 * (log_sigma_u + log_sigma_v) - 0.5 * conc_sq
    return min(h, HAAR_BOUND)


def _entropy_c4(logits: np.ndarray) -> float:
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


# ---- IO ----

def _collect_rollouts(rollout_dir: str) -> list[dict]:
    paths = sorted(
        glob.glob(os.path.join(rollout_dir, "eval_*.npz"))
        + glob.glob(os.path.join(rollout_dir, "eval_ep*.npz"))
    )
    out = []
    for p in paths:
        try:
            d = np.load(p, allow_pickle=False)
        except Exception as e:
            print(f"[warn] skip {p}: {e}", file=sys.stderr)
            continue
        if "selector_params" not in d.files or d["selector_params"].size == 0:
            print(f"[warn] {p} has no selector_params; rerun eval with new instrumentation",
                  file=sys.stderr)
            continue
        out.append(
            dict(
                path=p,
                selector_params=d["selector_params"],   # (T, param_dim)
                entropy=d["entropy"] if "entropy" in d.files else None,
                eef_pos=d["eef_pos"] if "eef_pos" in d.files else None,
                reward=_parse_rew(p),
            )
        )
    return out


def _parse_rew(path: str) -> float:
    base = os.path.basename(path)
    try:
        token = base.split("rew")[-1].split(".npz")[0]
        return float(token)
    except Exception:
        return float("nan")


# ---- Pick selection ----

DEFAULT_PHASE_LABELS = ["reach", "grasp", "lift", "place"]


def _resolve_picks(T: int, picks: list[str]) -> list[tuple[str, int]]:
    """Map pick tokens to (label, timestep). Numeric tokens are used directly;
    word tokens map to evenly-spaced quartile timesteps."""
    out = []
    word_idx = 0
    for tok in picks:
        if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
            t = int(tok)
            if t < 0:
                t = T + t
            out.append((f"t={t}", t))
        else:
            # Phase labels: divide T into len(picks) bins, pick midpoints.
            n = len(picks)
            t = int(((word_idx + 0.5) / n) * T)
            t = max(0, min(T - 1, t))
            out.append((tok, t))
            word_idx += 1
    return out


# ---- Plotting ----

def _plot_so2_polar(ax, params: np.ndarray, n_samples: int, label: str):
    mu_u, mu_v, log_sigma_u, log_sigma_v = params.tolist()
    angles = _sample_so2(mu_u, mu_v, log_sigma_u, log_sigma_v, n_samples)
    H = _entropy_so2(mu_u, mu_v, log_sigma_u, log_sigma_v)
    # Wrap to [0, 2pi) for hist binning
    angles_mod = (angles + 2 * np.pi) % (2 * np.pi)
    bins = np.linspace(0, 2 * np.pi, 73)  # 72 bins
    counts, edges = np.histogram(angles_mod, bins=bins, density=True)
    width = edges[1] - edges[0]
    ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=width,
           bottom=0.0, alpha=0.85, edgecolor="none")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_yticklabels([])
    ax.set_title(f"{label}\nH = {H:.3f}  (max {HAAR_BOUND:.2f})", fontsize=9)


def _plot_c4_bar(ax, params: np.ndarray, label: str):
    logits = np.asarray(params, dtype=float)
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    K = p.size
    ax.bar(np.arange(K), p, color="C0", alpha=0.85)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f"g{i}" for i in range(K)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("prob")
    H = _entropy_c4(logits)
    ax.set_title(f"{label}\nH = {H:.3f}  (max {math.log(K):.2f})", fontsize=9)


# ---- Main ----

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", required=True)
    p.add_argument("--picks", default="reach,grasp,lift,place",
                   help="Comma-separated phase labels or integer timesteps")
    p.add_argument("--distribution", choices=["so2", "c4"], default=None,
                   help="If omitted, inferred from param_dim")
    p.add_argument("--episode", type=int, default=0,
                   help="Which eval episode to visualize (default: 0)")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    rollouts = _collect_rollouts(args.rollout_dir)
    if not rollouts:
        raise SystemExit(f"No usable rollout NPZs found in {args.rollout_dir}")
    if args.episode >= len(rollouts):
        raise SystemExit(f"episode {args.episode} out of range; only {len(rollouts)} rollouts")
    roll = rollouts[args.episode]

    params_seq: np.ndarray = roll["selector_params"]  # (T, param_dim)
    T, param_dim = params_seq.shape

    if args.distribution is None:
        # SO(2) ProjectedNormal has 4 params; C4 GumbelSoftmaxCategorical has 4 logits.
        # Heuristic: if log_sigma columns (indices 2-3) look like log-sigmas (small,
        # not heavily peaked) treat as so2; otherwise c4. As a safe default we ask
        # the user to specify when param_dim == 4 (ambiguous).
        # For now: assume so2 if param_dim==4 — matches the project's default config.
        args.distribution = "so2" if param_dim == 4 else "c4"

    pick_tokens = [t.strip() for t in args.picks.split(",") if t.strip()]
    picks = _resolve_picks(T, pick_tokens)

    n = len(picks)
    if args.distribution == "so2":
        fig = plt.figure(figsize=(3.2 * n, 3.4))
        axes = [fig.add_subplot(1, n, i + 1, projection="polar") for i in range(n)]
        for ax, (label, t) in zip(axes, picks):
            _plot_so2_polar(ax, params_seq[t], args.n_samples, label)
    else:
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0))
        if n == 1:
            axes = [axes]
        for ax, (label, t) in zip(axes, picks):
            _plot_c4_bar(ax, params_seq[t], label)

    fig.suptitle(
        os.path.basename(roll["path"]) + f"  (rew={roll['reward']:.2f})",
        fontsize=10,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=200)
    if not args.out.endswith(".png"):
        fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200)
    print(f"wrote {args.out}  (T={T}, dist={args.distribution}, picks={picks})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
