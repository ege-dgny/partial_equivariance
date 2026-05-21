"""
Workspace entropy heatmap (Phase 5): aggregate per-step entropy across many
eval rollouts, bin by EEF (x, y) position, and plot mean H per bin.

Coverage is limited to where the policy actually went — exactly the region of
interest. No env modifications. Reads the NPZ files produced by the Phase 1.4
eval instrumentation.

Usage:
    python scripts/plot_workspace_heatmap.py \
        --rollout_dir logs/eval/<prefix>/ \
        --bin_size 0.02 \
        --out figs/heatmap_<task>.pdf \
        [--filter_success]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_rew(path: str) -> float:
    base = os.path.basename(path)
    try:
        token = base.split("rew")[-1].split(".npz")[0]
        return float(token)
    except Exception:
        return float("nan")


def _load_all(rollout_dir: str, filter_success: bool, success_thresh: float
              ) -> tuple[np.ndarray, np.ndarray, dict]:
    paths = sorted(
        glob.glob(os.path.join(rollout_dir, "eval_*.npz"))
        + glob.glob(os.path.join(rollout_dir, "eval_ep*.npz"))
    )
    xs: list[float] = []
    ys: list[float] = []
    hs: list[float] = []
    n_loaded = 0
    n_skipped = 0
    for p in paths:
        try:
            d = np.load(p, allow_pickle=False)
        except Exception as e:
            print(f"[warn] skip {p}: {e}", file=sys.stderr)
            n_skipped += 1
            continue
        if "entropy" not in d.files or "eef_pos" not in d.files:
            print(f"[warn] {p} missing entropy or eef_pos; rerun eval with new instrumentation",
                  file=sys.stderr)
            n_skipped += 1
            continue
        rew = _parse_rew(p)
        if filter_success and rew < success_thresh:
            n_skipped += 1
            continue
        eef = np.asarray(d["eef_pos"])
        # eef may be (T, obs_horizon, state_dim) — collapse obs_horizon if present.
        if eef.ndim == 3:
            eef = eef[:, -1, :]
        H = np.asarray(d["entropy"]).squeeze()
        T = min(len(H), len(eef))
        # First three state dims are EEF xyz by project convention.
        xs.extend(eef[:T, 0].tolist())
        ys.extend(eef[:T, 1].tolist())
        hs.extend(H[:T].tolist())
        n_loaded += 1

    info = dict(n_loaded=n_loaded, n_skipped=n_skipped)
    return np.asarray(xs), np.asarray(ys), np.asarray(hs), info


def _bin(xs: np.ndarray, ys: np.ndarray, hs: np.ndarray, bin_size: float
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(hs)
    xs, ys, hs = xs[mask], ys[mask], hs[mask]
    if xs.size == 0:
        raise SystemExit("No valid (x, y, H) samples.")
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    pad = bin_size
    x_min -= pad; x_max += pad
    y_min -= pad; y_max += pad
    nx = max(1, int(np.ceil((x_max - x_min) / bin_size)))
    ny = max(1, int(np.ceil((y_max - y_min) / bin_size)))
    sum_h = np.zeros((ny, nx))
    cnt = np.zeros((ny, nx), dtype=int)
    ix = np.clip(((xs - x_min) / bin_size).astype(int), 0, nx - 1)
    iy = np.clip(((ys - y_min) / bin_size).astype(int), 0, ny - 1)
    for x_i, y_i, h_i in zip(ix, iy, hs):
        sum_h[y_i, x_i] += h_i
        cnt[y_i, x_i] += 1
    mean_h = np.full_like(sum_h, np.nan)
    nonzero = cnt > 0
    mean_h[nonzero] = sum_h[nonzero] / cnt[nonzero]
    extent = (x_min, x_max, y_min, y_max)
    return mean_h, cnt, np.array([x_min, y_min, bin_size, nx, ny]), extent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", required=True)
    p.add_argument("--bin_size", type=float, default=0.02, help="Bin size in meters")
    p.add_argument("--out", required=True)
    p.add_argument("--filter_success", action="store_true",
                   help="Use only successful rollouts (rew >= --success_thresh)")
    p.add_argument("--success_thresh", type=float, default=0.5)
    p.add_argument("--target_xy", type=float, nargs=2, default=None,
                   help="Mark a target/goal position on the plot, e.g. --target_xy 0.0 -0.2")
    p.add_argument("--min_count", type=int, default=3,
                   help="Bins with fewer samples are shown muted")
    args = p.parse_args()

    xs, ys, hs, info = _load_all(
        args.rollout_dir, args.filter_success, args.success_thresh
    )
    print(f"loaded {info['n_loaded']} rollouts, skipped {info['n_skipped']}")

    mean_h, cnt, meta, extent = _bin(xs, ys, hs, args.bin_size)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax0, ax1 = axes

    # Panel 1: mean entropy heatmap
    cmap = plt.get_cmap("viridis")
    im = ax0.imshow(
        mean_h, origin="lower", extent=extent, aspect="equal", cmap=cmap,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax0, label=r"mean $H(p_\phi)$")
    ax0.set_title(f"workspace H heatmap  ({info['n_loaded']} rollouts)")
    ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    if args.target_xy is not None:
        ax0.scatter(args.target_xy[0], args.target_xy[1],
                    marker="x", color="red", s=120, linewidths=2, label="target")
        ax0.legend(loc="best", fontsize=8)

    # Panel 2: sample count (so reviewers can see where coverage is thin)
    im2 = ax1.imshow(
        cnt, origin="lower", extent=extent, aspect="equal", cmap="magma",
        interpolation="nearest",
    )
    fig.colorbar(im2, ax=ax1, label="bin sample count")
    ax1.set_title("coverage (sample count per bin)")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    if not args.out.endswith(".png"):
        fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200)
    print(f"wrote {args.out}  bins={meta[3]}x{meta[4]}  N={len(xs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
