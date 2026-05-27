"""
Point-cloud viewer for PEFM/EquiBot NPZ datasets.

Loads saved per-timestep NPZs (keys: pc, eef_pos[, action]) for one episode and
renders a matplotlib 3D-scatter montage (points colored by height, end-effector
marked) to a PNG. The server is headless, so output is always a saved image; an
optional --ply dump lets you inspect a single frame interactively with Open3D
locally.

Usage:
  python scripts/view_pc.py --data_dir /path/to/dataset/pcs --episode 0 \
      --out /tmp/ep0.png
  python scripts/view_pc.py --data_dir .../pcs --episode 0 --frames 0,20,40,60 \
      --ply /tmp/ep0_grasp.ply
"""

import argparse
import glob
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def _episode_files(data_dir, episode):
    """Return sorted NPZ paths for an episode. `episode` may be an int index
    (matched as ep%06d) or any substring of the filename."""
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not all_files:
        raise FileNotFoundError(f"No .npz files under {data_dir}")
    if isinstance(episode, int) or str(episode).isdigit():
        tag = f"ep{int(episode):06d}"
    else:
        tag = str(episode)
    files = [f for f in all_files if tag in os.path.basename(f)]
    if not files:
        # Fall back to the first episode present.
        stems = sorted({os.path.basename(f).split("_t")[0] for f in all_files})
        raise SystemExit(
            f"No files match '{tag}'. Available episodes: {stems[:8]}"
            + (" ..." if len(stems) > 8 else "")
        )
    return files


def _eef_xyz(eef_pos):
    """Extract per-eef xyz from a stored eef_pos array of shape (num_eef, 13),
    (1, 13) or flat (13,)/(26,). Returns (num_eef, 3)."""
    arr = np.asarray(eef_pos).reshape(-1)
    if arr.size % 13 == 0:
        num_eef = arr.size // 13
        return arr.reshape(num_eef, 13)[:, :3]
    return arr[:3].reshape(1, 3)


def _unique_pc(pc):
    pc = np.asarray(pc).reshape(-1, 3)
    return np.unique(pc, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dataset pcs/ directory")
    ap.add_argument("--episode", default="0", help="episode index or filename substring")
    ap.add_argument("--frames", default="auto", help="comma list of frame indices, or 'auto'")
    ap.add_argument("--n_auto", type=int, default=6, help="#frames when --frames auto")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--ply", default=None, help="optional: dump one frame (the grasp frame) to PLY")
    args = ap.parse_args()

    files = _episode_files(args.data_dir, args.episode)
    T = len(files)

    # eef trajectory + grasp frame (lowest eef z)
    eefs = np.stack([_eef_xyz(np.load(f)["eef_pos"])[0] for f in files])  # (T,3) first eef
    grasp_i = int(eefs[:, 2].argmin())

    if args.frames == "auto":
        idxs = sorted(set(np.linspace(0, T - 1, args.n_auto).astype(int).tolist() + [grasp_i]))
    else:
        idxs = [int(x) for x in args.frames.split(",") if x.strip() != ""]

    print(f"episode files: {T}  grasp_frame(min eef z): {grasp_i}")
    ncol = min(4, len(idxs))
    nrow = int(np.ceil(len(idxs) / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 4.0 * nrow))

    for k, fi in enumerate(idxs):
        d = np.load(files[fi])
        pc = _unique_pc(d["pc"])
        eef = _eef_xyz(d["eef_pos"])  # (num_eef,3)
        nearest = (
            np.linalg.norm(pc[:, :2] - eef[0, :2], axis=1).min() if len(pc) > 1 else float("nan")
        )
        print(
            f"  frame {fi:4d}: n_uniq={len(pc):5d} "
            f"bbox x[{pc[:,0].min():.2f},{pc[:,0].max():.2f}] "
            f"y[{pc[:,1].min():.2f},{pc[:,1].max():.2f}] "
            f"z[{pc[:,2].min():.2f},{pc[:,2].max():.2f}] "
            f"eef0_to_PCxy={nearest:.3f}m"
        )
        ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
        if len(pc) > 1:
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], s=2, cmap="viridis")
        colors = ["red", "magenta"]
        for e in range(eef.shape[0]):
            ax.scatter(
                eef[e, 0], eef[e, 1], eef[e, 2], c=colors[e % 2], s=120,
                marker="*", edgecolors="k", label=f"eef{e}",
            )
        ax.set_title(f"t={fi} (n={len(pc)})", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        # consistent table-region view
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.3, 0.5); ax.set_zlim(0.78, 1.05)
        ax.view_init(elev=25, azim=-60)

    out = args.out or os.path.join("/tmp", f"pcview_ep{args.episode}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved montage -> {out}")

    if args.ply:
        d = np.load(files[grasp_i])
        pc = _unique_pc(d["pc"])
        with open(args.ply, "w") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {len(pc)}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            fh.write("end_header\n")
            for p in pc:
                fh.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        print(f"saved grasp-frame PLY ({len(pc)} pts) -> {args.ply}")


if __name__ == "__main__":
    main()
