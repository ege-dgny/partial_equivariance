"""
Cross-task entropy overlay (Phase 3): export `train/entropy` curves from W&B
runs across different tasks and plot them on a single axes with reference
lines at log(2*pi) (Haar — full equivariance) and 0 (Dirac — no equivariance).

This is the paper figure showing PEFM auto-discovers the right symmetry level
per task without any task-specific tuning.

Usage:
    python scripts/plot_cross_task_entropy.py \
        --runs entity/project/run_id_fold:Fold \
               entity/project/run_id_pickplace:PickPlace \
               entity/project/run_id_pusht:PushT \
        --out figs/cross_task_entropy.pdf

If wandb is unavailable, --csv lets you load pre-exported CSV files instead.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HAAR_BOUND = math.log(2 * math.pi)  # ~1.838


def _ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average; preserves NaNs at the start."""
    out = np.full_like(values, np.nan, dtype=float)
    running: Optional[float] = None
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        running = float(v) if running is None else alpha * float(v) + (1 - alpha) * running
        out[i] = running
    return out


def _fetch_wandb_run(run_path: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch (steps, values) for a metric from a W&B run path entity/project/run_id."""
    try:
        import wandb
    except ImportError as e:
        raise SystemExit(
            "wandb is required for --runs. Install with `pip install wandb`, "
            "or use --csv to load pre-exported curves."
        ) from e

    api = wandb.Api()
    run = api.run(run_path)
    history = run.history(keys=[metric, "_step"], pandas=True)
    if history is None or history.empty:
        return np.array([]), np.array([])
    steps = history["_step"].to_numpy()
    values = history[metric].to_numpy()
    mask = np.isfinite(values)
    return steps[mask], values[mask]


def _fetch_csv(csv_path: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        # Try W&B's typical column name
        candidates = [c for c in df.columns if c.endswith(metric)]
        if not candidates:
            raise SystemExit(f"Metric '{metric}' not in CSV columns: {list(df.columns)}")
        metric = candidates[0]
    step_col = "_step" if "_step" in df.columns else df.columns[0]
    steps = df[step_col].to_numpy()
    values = df[metric].to_numpy()
    mask = np.isfinite(values)
    return steps[mask], values[mask]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        nargs="+",
        default=[],
        help="W&B run specs, format entity/project/run_id:LABEL",
    )
    p.add_argument(
        "--csv",
        nargs="+",
        default=[],
        help="Local CSV files, format path/to/file.csv:LABEL",
    )
    p.add_argument("--metric", default="train/entropy", help="W&B metric key")
    p.add_argument("--ema_alpha", type=float, default=0.05)
    p.add_argument("--out", required=True, help="Output figure path (.pdf or .png)")
    p.add_argument("--title", default="Selector entropy across tasks")
    args = p.parse_args()

    sources: list[tuple[str, np.ndarray, np.ndarray]] = []  # (label, steps, values)
    for spec in args.runs:
        if ":" not in spec:
            raise SystemExit(f"--runs spec must be entity/project/run_id:LABEL, got {spec!r}")
        run_path, label = spec.rsplit(":", 1)
        steps, vals = _fetch_wandb_run(run_path, args.metric)
        sources.append((label, steps, vals))
    for spec in args.csv:
        if ":" not in spec:
            raise SystemExit(f"--csv spec must be path:LABEL, got {spec!r}")
        path, label = spec.rsplit(":", 1)
        steps, vals = _fetch_csv(path, args.metric)
        sources.append((label, steps, vals))

    if not sources:
        raise SystemExit("Provide at least one --runs or --csv source.")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    cmap = plt.get_cmap("tab10")
    for i, (label, steps, vals) in enumerate(sources):
        if vals.size == 0:
            print(f"[warn] no data for {label}", file=sys.stderr)
            continue
        smoothed = _ema(vals, alpha=args.ema_alpha)
        ax.plot(steps, vals, color=cmap(i), alpha=0.25, linewidth=0.8)
        ax.plot(steps, smoothed, color=cmap(i), linewidth=1.8, label=label)

    ax.axhline(
        HAAR_BOUND, linestyle="--", color="gray", linewidth=1.0,
        label=f"Haar (full equivariance) = log(2π) ≈ {HAAR_BOUND:.3f}",
    )
    ax.axhline(
        0.0, linestyle=":", color="gray", linewidth=1.0,
        label="Dirac (no equivariance) = 0",
    )

    ax.set_xlabel("training step")
    ax.set_ylabel(r"$H(p_\phi(\cdot \mid o))$")
    ax.set_title(args.title)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(alpha=0.25)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    png = os.path.splitext(args.out)[0] + ".png"
    if not args.out.endswith(".png"):
        fig.savefig(png, dpi=200)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
