"""
Convert robomimic HDF5 demos to PEFM NPZ format.

Downloads proficient-human (PH) demos from robomimic, replays MuJoCo states
to render depth/RGB, extracts point clouds, and saves per-timestep NPZ files
compatible with pefm.datasets.dataset.BaseDataset.

Usage:
    # Step 1: Download robomimic datasets (one-time)
    python -m pefm_envs.sim_robosuite.convert_robomimic --download --task can

    # Step 2: Convert to PEFM format
    python -m pefm_envs.sim_robosuite.convert_robomimic \
        --task can \
        --hdf5 data/robomimic/can/ph/low_dim.hdf5 \
        --out_dir ../data/pick_place_fixed

Tasks:
    can            -> PickPlaceCan   (SO2 symmetry)
    square         -> NutAssemblySquare (C4 symmetry)
    tool_hang      -> ToolHang       (SO2 multi-phase)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import cv2
except ImportError:
    cv2 = None


# ------------------------------------------------------------------ #
#  robomimic task mapping
# ------------------------------------------------------------------ #

TASK_MAP = {
    "can": {
        "env_name": "PickPlaceCan",
        "pefm_name": "pick_place_fixed",
    },
    "square": {
        "env_name": "NutAssemblySquare",
        "pefm_name": "nut_assembly_fixed",
    },
    "tool_hang": {
        "env_name": "ToolHang",
        "pefm_name": "tool_hang",
    },
}

# robomimic v0.1 download URLs (Stanford)
DOWNLOAD_URLS = {
    "can": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/v0.1/can/ph/low_dim.hdf5",
    "square": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/v0.1/square/ph/low_dim.hdf5",
    "tool_hang": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/v0.1/tool_hang/ph/low_dim.hdf5",
}


# ------------------------------------------------------------------ #
#  Download
# ------------------------------------------------------------------ #

def download_dataset(task: str, out_dir: str = "data/robomimic") -> str:
    """Download robomimic PH dataset for a task. Returns path to HDF5."""
    import urllib.request

    url = DOWNLOAD_URLS[task]
    dest_dir = os.path.join(out_dir, task, "ph")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "low_dim.hdf5")

    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return dest

    print(f"Downloading {task} from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")
    return dest


# ------------------------------------------------------------------ #
#  robosuite env creation for replay
# ------------------------------------------------------------------ #

def create_replay_env(
    env_name: str,
    render_res: int = 240,
    control_freq: int = 20,
):
    """Create robosuite env configured for state replay + rendering."""
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(
        controller="BASIC",
    )

    env = suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=control_freq,
        horizon=10000,
        camera_names=["agentview", "sideview"],
        camera_heights=[render_res, render_res],
        camera_widths=[render_res, render_res],
        camera_depths=[True, True],
        camera_segmentations=["element", "element"],
    )
    return env


# ------------------------------------------------------------------ #
#  Point cloud extraction
# ------------------------------------------------------------------ #

def extract_point_cloud(
    sim,
    depth: np.ndarray,
    seg: np.ndarray,
    camera_name: str = "agentview",
    num_points: int = 4096,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Extract world-frame point cloud from depth + segmentation.

    Filters out robot, table, floor — keeps task objects only.
    Returns (num_points, 3) array, padded/subsampled as needed.
    """
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if seg.ndim == 3:
        seg = seg[:, :, 0]

    H, W = depth.shape

    # Camera intrinsics
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2.0)
    cx, cy = W / 2.0, H / 2.0

    # Camera extrinsics
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)

    # Find object body IDs (exclude robot/table/floor)
    skip_keywords = {"robot", "gripper", "base", "world", "table", "floor", "mount"}
    object_ids = []
    for i in range(sim.model.nbody):
        name = sim.model.body_id2name(i)
        if name and not any(s in name.lower() for s in skip_keywords):
            object_ids.append(i)

    if not object_ids:
        # Fallback: use all non-zero depth
        valid_mask = (depth > 0.01) & (depth < 5.0)
    else:
        valid_mask = np.isin(seg, object_ids) & (depth > 0.01) & (depth < 5.0)

    if not valid_mask.any():
        return np.zeros((num_points, 3), dtype=np.float32)

    v_grid, u_grid = np.where(valid_mask)
    z_vals = depth[v_grid, u_grid]

    # Unproject to camera frame
    x_cam = (u_grid - cx) * z_vals / f
    y_cam = (v_grid - cy) * z_vals / f

    # MuJoCo camera: -Z forward, X right, Y down
    pts_cam = np.stack([x_cam, y_cam, -z_vals], axis=-1)

    # World frame
    pts_world = (cam_mat.T @ pts_cam.T).T + cam_pos

    # Subsample / pad to num_points
    n = len(pts_world)
    if rng is None:
        rng = np.random.RandomState(0)

    if n >= num_points:
        idx = rng.choice(n, num_points, replace=False)
    else:
        idx = rng.choice(n, num_points, replace=True)

    return pts_world[idx].astype(np.float32)


# ------------------------------------------------------------------ #
#  Action conversion
# ------------------------------------------------------------------ #

def convert_action(
    robo_action: np.ndarray,
    freq: int = 20,
) -> np.ndarray:
    """Convert robosuite action to PEFM format.

    robosuite: [dx, dy, dz, dax, day, daz, grip]  grip in [-1, +1]
    PEFM:      [grip, vx, vy, vz, drx, dry, drz]  grip in [0, 1]

    Delta -> velocity: vel = delta * freq
    """
    dx, dy, dz = robo_action[0], robo_action[1], robo_action[2]
    dax, day, daz = robo_action[3], robo_action[4], robo_action[5]
    grip_robo = robo_action[6]

    # Grip: -1/+1 -> 0/1
    grip_pefm = np.clip((grip_robo + 1.0) / 2.0, 0.0, 1.0)

    # Delta -> velocity
    vx, vy, vz = dx * freq, dy * freq, dz * freq
    drx, dry, drz = dax * freq, day * freq, daz * freq

    return np.array([grip_pefm, vx, vy, vz, drx, dry, drz], dtype=np.float32)


# ------------------------------------------------------------------ #
#  EEF observation
# ------------------------------------------------------------------ #

def build_eef_obs(
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    """Build PEFM eef observation (1, 13).

    Format: [eef_xyz(3), x_dir(3), z_dir(3), gravity(3), grip(1)]
    """
    rot_mat = Rotation.from_quat(eef_quat).as_matrix()
    x_dir = rot_mat[:, 0]
    z_dir = rot_mat[:, 2]
    gravity = np.array([0.0, 0.0, -1.0])

    # Grip from finger qpos: small opening = closed
    grip = 1.0 if np.mean(gripper_qpos) < 0.02 else 0.0

    state = np.concatenate([eef_pos, x_dir, z_dir, gravity, [grip]])
    return state.reshape(1, 13).astype(np.float32)


# ------------------------------------------------------------------ #
#  Main conversion
# ------------------------------------------------------------------ #

def convert_hdf5(
    hdf5_path: str,
    out_dir: str,
    task: str,
    freq: int = 20,
    num_points: int = 4096,
    render_res: int = 240,
    max_demos: int | None = None,
    subsample: int = 1,
):
    """Convert robomimic HDF5 to PEFM NPZ files."""
    task_info = TASK_MAP[task]
    env_name = task_info["env_name"]
    prefix = task_info["pefm_name"]

    pcs_dir = os.path.join(out_dir, "pcs")
    vid_dir = os.path.join(out_dir, "videos")
    os.makedirs(pcs_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    print(f"Opening {hdf5_path} ...")
    f = h5py.File(hdf5_path, "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

    if max_demos is not None:
        demos = demos[:max_demos]

    print(f"Found {len(demos)} demos, converting with env={env_name}")
    print(f"Creating robosuite env for replay ...")
    env = create_replay_env(env_name, render_res=render_res, control_freq=freq)

    rng = np.random.RandomState(0)

    for ep_idx, demo_key in enumerate(demos):
        demo = f["data"][demo_key]
        states = demo["states"][:]
        actions = demo["actions"][:]
        T = len(actions)

        # Obs keys from HDF5
        obs_grp = demo["obs"]
        has_eef = "robot0_eef_pos" in obs_grp
        has_quat = "robot0_eef_quat" in obs_grp
        has_gripper = "robot0_gripper_qpos" in obs_grp

        # Reset env and set initial state
        env.reset()

        frame_count = 0
        vid_frames = []

        for t in range(0, T, subsample):
            # Set MuJoCo state from HDF5
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()

            # Render
            obs_dict = env._get_observations()

            # Extract depth + segmentation for PC
            depth = obs_dict.get("agentview_depth")
            seg = obs_dict.get("agentview_segmentation_element")

            if depth is not None and seg is not None:
                pc = extract_point_cloud(
                    env.sim, depth, seg,
                    camera_name="agentview",
                    num_points=num_points,
                    rng=rng,
                )
            else:
                pc = np.zeros((num_points, 3), dtype=np.float32)

            # RGB for reference
            rgb = obs_dict.get("agentview_image")
            if rgb is not None:
                rgb = rgb[::-1].copy()  # MuJoCo renders upside down
            else:
                rgb = np.zeros((render_res, render_res, 3), dtype=np.uint8)

            # Action conversion
            action = convert_action(actions[t], freq=freq)

            # EEF observation
            if has_eef and has_quat and has_gripper:
                eef_pos_val = obs_grp["robot0_eef_pos"][t]
                eef_quat_val = obs_grp["robot0_eef_quat"][t]
                gripper_qpos_val = obs_grp["robot0_gripper_qpos"][t]
                eef_obs = build_eef_obs(eef_pos_val, eef_quat_val, gripper_qpos_val)
            else:
                # Build from sim state
                eef_pos_val = obs_dict.get("robot0_eef_pos", np.zeros(3))
                eef_quat_val = obs_dict.get("robot0_eef_quat", np.array([0, 0, 0, 1.0]))
                gripper_qpos_val = obs_dict.get("robot0_gripper_qpos", np.array([0.04, 0.04]))
                eef_obs = build_eef_obs(eef_pos_val, eef_quat_val, gripper_qpos_val)

            # Save NPZ
            fn = f"{prefix}_ep{ep_idx:06d}_view0_t{frame_count:04d}.npz"
            save_path = os.path.join(pcs_dir, fn)
            np.savez(
                save_path,
                pc=pc,
                rgb=rgb,
                action=action,
                eef_pos=eef_obs,
            )

            # Video frame (dual view)
            front = rgb
            side_img = obs_dict.get("sideview_image")
            if side_img is not None:
                side_img = side_img[::-1].copy()
                dual = np.concatenate([front, side_img], axis=1)
            else:
                dual = front
            vid_frames.append(dual)

            frame_count += 1

        # Save video
        if vid_frames and cv2 is not None:
            vid_path = os.path.join(vid_dir, f"ep{ep_idx:06d}.mp4")
            h, w = vid_frames[0].shape[:2]
            writer = cv2.VideoWriter(
                vid_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                freq // max(subsample, 1),
                (w, h),
            )
            for frame in vid_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

        print(
            f"  [{ep_idx + 1}/{len(demos)}] {demo_key}: "
            f"{T} steps -> {frame_count} frames saved"
        )

    env.close()
    f.close()
    print(f"\nDone. Output in {out_dir}/pcs/")
    print(f"  Total demos: {len(demos)}")
    print(f"  NPZ format: pc=(4096,3), eef_pos=(1,13), action=(7,), rgb=(240,240,3)")


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Convert robomimic HDF5 demos to PEFM NPZ format",
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=list(TASK_MAP.keys()),
        help="robomimic task name",
    )
    parser.add_argument(
        "--hdf5", type=str, default=None,
        help="Path to robomimic HDF5 file (if not downloading)",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: ../data/{pefm_name})",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download the dataset first",
    )
    parser.add_argument(
        "--download_dir", type=str, default="data/robomimic",
        help="Where to save downloaded HDF5 files",
    )
    parser.add_argument(
        "--freq", type=int, default=20,
        help="Control frequency (must match env)",
    )
    parser.add_argument(
        "--num_points", type=int, default=4096,
        help="Point cloud size",
    )
    parser.add_argument(
        "--render_res", type=int, default=240,
        help="Camera resolution",
    )
    parser.add_argument(
        "--max_demos", type=int, default=None,
        help="Limit number of demos to convert",
    )
    parser.add_argument(
        "--subsample", type=int, default=1,
        help="Subsample every N timesteps (1 = keep all)",
    )
    args = parser.parse_args()

    task_info = TASK_MAP[args.task]

    # Download if requested
    if args.download:
        args.hdf5 = download_dataset(args.task, args.download_dir)

    if args.hdf5 is None:
        # Try default location
        default_path = os.path.join(
            args.download_dir, args.task, "ph", "low_dim.hdf5",
        )
        if os.path.exists(default_path):
            args.hdf5 = default_path
        else:
            print(f"No HDF5 found. Run with --download to fetch it, or pass --hdf5.")
            sys.exit(1)

    if args.out_dir is None:
        args.out_dir = os.path.join("..", "data", task_info["pefm_name"])

    convert_hdf5(
        hdf5_path=args.hdf5,
        out_dir=args.out_dir,
        task=args.task,
        freq=args.freq,
        num_points=args.num_points,
        render_res=args.render_res,
        max_demos=args.max_demos,
        subsample=args.subsample,
    )


if __name__ == "__main__":
    main()
