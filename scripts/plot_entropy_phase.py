"""
Phase-aligned selector-entropy figure for the PEFM partial-equivariance claim.

Loads the per-rollout eval NPZs saved by pefm/vec_eval.py (entropy (T,),
eef_pos (T,1,13), reward in filename) for one or more runs and produces:

  (left)  mean +/- std selector entropy H(p_phi) vs normalized rollout phase,
          showing the within-episode high(grasp) -> low(insert) collapse.
  (right) entropy by semantic phase (approach/grasp, transport, insert) as a
          grouped bar chart -- the strict-equivariance baseline stays flat at
          ln|G| while PEFM collapses during the asymmetric insert phase.

Usage:
  python -m scripts.plot_entropy_phase \
      --runs pefm=logs/train/peg_insert_v1 strict=logs/train/peg_insert_strict_v1 \
      --epoch 00199 --group_size 4 --socket 0.35 -0.2 \
      --out /home/ege/pefm/data/_figs/peg_entropy_phase.png
(--epoch optional: defaults to latest saved eval per run.)
"""
import argparse, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(log_dir, epoch=None):
    fs = glob.glob(os.path.join(log_dir, "eval_*_ep*.npz"))
    if not fs:
        return None
    epochs = sorted(set(re.search(r"eval_(\d+)_", f).group(1) for f in fs))
    ep = epoch if (epoch is not None and epoch in epochs) else epochs[-1]
    files = sorted(glob.glob(os.path.join(log_dir, f"eval_{ep}_ep*.npz")))
    eps = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        H = np.array(d["entropy"]).astype(float).squeeze()
        if H.ndim == 0 or H.size < 2:
            continue
        eef = np.array(d["eef_pos"]).astype(float)
        eef = eef.reshape(eef.shape[0], -1)
        m = re.search(r"rew([-0-9.]+)\.npz", f)
        rew = float(m.group(1)) if m else float("nan")
        eps.append(dict(H=H, grip=eef[:, 12], z=eef[:, 2], xy=eef[:, :2], rew=rew))
    return ep, eps


def resample(y, n=100):
    y = np.asarray(y, float)
    xs = np.linspace(0, 1, len(y))
    xt = np.linspace(0, 1, n)
    return np.interp(xt, xs, y)


def phase_means(eps, socket, near=0.12):
    """Mean entropy in approach/grasp (grip open), transport (closed, far),
    insert (closed, near socket xy)."""
    g, tr, ins = [], [], []
    sx, sy = socket
    for e in eps:
        grip = e["grip"] > 0.5
        dist = np.linalg.norm(e["xy"] - np.array([sx, sy]), axis=1)
        g.extend(e["H"][~grip].tolist())
        tr.extend(e["H"][grip & (dist > near)].tolist())
        ins.extend(e["H"][grip & (dist <= near)].tolist())
    f = lambda a: (np.nanmean(a) if len(a) else np.nan, np.nanstd(a) if len(a) else 0.0)
    return f(g), f(tr), f(ins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=log_dir ...")
    ap.add_argument("--epoch", default=None)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--socket", nargs=2, type=float, default=[0.35, -0.2])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Hmax = np.log(args.group_size)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    phase_lbls = ["approach/\ngrasp", "transport", "insert"]
    width = 0.8 / max(1, len(args.runs))
    csv = ["run,epoch,phase,mean_H,std_H,mean_rew,success_rate"]

    for ri, spec in enumerate(args.runs):
        label, path = spec.split("=", 1)
        res = load_run(path, args.epoch)
        if not res:
            print(f"[warn] no eval data in {path}")
            continue
        ep, eps = res
        x = np.linspace(0, 1, 100)
        M = np.stack([resample(e["H"], 100) for e in eps])
        mean, std = np.nanmean(M, 0), np.nanstd(M, 0)
        ln = ax1.plot(x, mean, label=f"{label} (ep{ep})")[0]
        ax1.fill_between(x, mean - std, mean + std, alpha=0.18, color=ln.get_color())

        pm = phase_means(eps, args.socket)
        xs = np.arange(3) + ri * width
        ax2.bar(xs, [p[0] for p in pm], width=width,
                yerr=[p[1] for p in pm], capsize=3, label=label, color=ln.get_color())
        rews = np.array([e["rew"] for e in eps])
        for pl, (mn, sd) in zip(["approach_grasp", "transport", "insert"], pm):
            csv.append(f"{label},{ep},{pl},{mn:.4f},{sd:.4f},{rews.mean():.4f},{(rews>=0.5).mean():.4f}")

    ax1.axhline(Hmax, ls="--", c="gray", lw=1, label=f"uniform = ln{args.group_size}={Hmax:.2f}")
    ax1.set_xlabel("normalized rollout phase (0 start -> 1 end)")
    ax1.set_ylabel(r"selector entropy $H(p_\phi)$")
    ax1.set_title("Within-episode symmetry collapse")
    ax1.legend(fontsize=8); ax1.set_ylim(0, Hmax * 1.05)

    ax2.axhline(Hmax, ls="--", c="gray", lw=1)
    ax2.set_xticks(np.arange(3) + width * (len(args.runs) - 1) / 2)
    ax2.set_xticklabels(phase_lbls)
    ax2.set_ylabel(r"mean $H(p_\phi)$")
    ax2.set_title("Entropy by task phase")
    ax2.legend(fontsize=8); ax2.set_ylim(0, Hmax * 1.05)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=150)
    with open(args.out.rsplit(".", 1)[0] + ".csv", "w") as fcsv:
        fcsv.write("\n".join(csv) + "\n")
    print("saved", args.out)
    print("saved", args.out.rsplit(".", 1)[0] + ".csv")
    print("\n".join(csv))


if __name__ == "__main__":
    main()
