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

# Per-task receptacle body names that aren't surfaced as Python attributes
# on the robosuite env. Manipulated objects come from env.objects / env.nuts
# / env.tool etc. (see resolve_object_body_names below). This dict only
# names the receptacles/targets that the policy needs to see for placement
# but that robosuite stores as raw bodies in the model rather than as
# Python-side object instances.
TASK_RECEPTACLE_BODIES = {
    "can":       ["bin1", "bin2"],   # source + target bins
    "square":    ["peg1"],            # square peg only (peg2 = round, excluded)
    "tool_hang": [],                  # frame + stand are env attributes -> handled below
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
        camera_names=["agentview", "frontview"],
        camera_heights=[render_res, render_res],
        camera_widths=[render_res, render_res],
        camera_depths=[True, True],
        camera_segmentations=["element", "element"],
    )
    return env


# ------------------------------------------------------------------ #
#  Point cloud extraction
# ------------------------------------------------------------------ #

def _get_robot_geom_ids(sim) -> set[int]:
    """Get geom IDs belonging to robot/gripper/mount (to exclude from PC)."""
    skip_keywords = {"robot", "gripper", "mount", "base", "fixed_mount"}
    robot_geom_ids = set()
    for body_id in range(sim.model.nbody):
        name = sim.model.body_id2name(body_id)
        if name and any(s in name.lower() for s in skip_keywords):
            geom_start = sim.model.body_geomadr[body_id]
            geom_count = sim.model.body_geomnum[body_id]
            for g in range(geom_start, geom_start + geom_count):
                robot_geom_ids.add(g)
    return robot_geom_ids


def resolve_object_body_names(env, task: str) -> list[str]:
    """Resolve which robosuite bodies count as 'relevant objects' for the
    paper-style segmentation (EquiBot §3.1: "point clouds of relevant objects").

    Mirrors sim_mobile's design: the env declares its objects, the converter
    trusts the env. Where robosuite exposes object instances as Python
    attributes (env.objects / env.nuts / env.tool / env.frame / env.stand),
    we read them directly. Where it doesn't (bins, square peg), we list
    body names explicitly via TASK_RECEPTACLE_BODIES.

    Subtask filtering: PickPlaceCan loads all 4 PickPlace objects and
    NutAssemblySquare loads both nuts. We keep only the manipulated object
    of the subtask; the others are clutter.
    """
    names: list[str] = []
    if task == "can":
        for obj in getattr(env, "objects", []):
            if "can" in obj.root_body.lower():
                names.append(obj.root_body)
    elif task == "square":
        for nut in getattr(env, "nuts", []):
            if "square" in nut.root_body.lower():
                names.append(nut.root_body)
    elif task == "tool_hang":
        for attr in ("tool", "frame", "stand"):
            obj = getattr(env, attr, None)
            if obj is not None and hasattr(obj, "root_body"):
                names.append(obj.root_body)
    else:
        raise KeyError(f"Unknown task: {task}")

    # Per-task receptacles (not surfaced as Python attributes)
    names.extend(TASK_RECEPTACLE_BODIES.get(task, []))

    if not names:
        raise RuntimeError(
            f"No object bodies resolved for task={task}. "
            f"Inspect env.objects / env.nuts / TASK_RECEPTACLE_BODIES."
        )
    return names


def _get_object_geom_ids(sim, body_names, verbose: bool = False) -> set[int]:
    """Expand a list of MuJoCo body names into geom IDs, recursively.

    Strict equality on body name -- no substring matching. Body names are
    authoritative (came from env attributes or the per-task receptacle list).

    Recurses through every descendant body so composite objects (e.g.
    ToolHang's tool_root, whose geoms live on child bodies) are captured.
    """
    name_to_id = {sim.model.body_id2name(i): i for i in range(sim.model.nbody)}
    parent_of = sim.model.body_parentid  # array, parent_of[i] = parent body id

    # Build children adjacency once.
    children: dict[int, list[int]] = {}
    for i in range(sim.model.nbody):
        children.setdefault(parent_of[i], []).append(i)

    def descendants(root_id: int) -> list[int]:
        """All bodies under root (inclusive). Stops at body 0 self-loop."""
        stack = [root_id]
        out = []
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(c for c in children.get(b, []) if c != b)
        return out

    obj_geom_ids: set[int] = set()
    matched: list[str] = []
    missing: list[str] = []
    for name in body_names:
        bid = name_to_id.get(name)
        if bid is None:
            missing.append(name)
            continue
        n_geoms_before = len(obj_geom_ids)
        for db in descendants(bid):
            gstart = sim.model.body_geomadr[db]
            gcount = sim.model.body_geomnum[db]
            for g in range(gstart, gstart + gcount):
                obj_geom_ids.add(g)
        matched.append(f"{name}({len(obj_geom_ids) - n_geoms_before}g)")
    if verbose:
        print(f"[object-seg] matched bodies: {matched}")
        if missing:
            print(f"[object-seg] WARNING missing bodies: {missing}")
        print(f"[object-seg] total object geoms: {len(obj_geom_ids)}")
    if missing:
        raise RuntimeError(
            f"Bodies not found in MuJoCo model: {missing}. "
            f"Available body names: {sorted(name_to_id.keys())}"
        )
    if not obj_geom_ids:
        raise RuntimeError(
            f"No geoms resolved from body names {body_names} (incl. descendants)."
        )
    return obj_geom_ids


def extract_point_cloud(
    sim,
    depth: np.ndarray,
    seg: np.ndarray,
    camera_name: str = "agentview",
    camera_height: int = 240,
    camera_width: int = 240,
    num_points: int = 4096,
    rng: np.random.RandomState | None = None,
    object_geom_ids: set[int] | None = None,
    legacy_unprojection: bool = False,
) -> np.ndarray:
    """Extract world-frame point cloud using robosuite's camera utilities.

    Uses robosuite's get_real_depth_map, get_camera_intrinsic_matrix,
    and get_camera_extrinsic_matrix for correct depth linearization
    and camera-to-world transforms.

    If `object_geom_ids` is given, KEEPS only those geoms (paper-faithful,
    "point clouds of relevant objects"). Otherwise falls back to dropping
    robot geoms only (legacy behavior, full scene minus robot).

    Filters by workspace bounds.
    Returns (num_points, 3) padded/subsampled.
    """
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if seg.ndim == 3:
        seg = seg[:, :, 0]

    H, W = depth.shape

    # Linearize depth (robosuite handles znear/zfar internally)
    # Clamp out-of-range depth values (occasional state-replay glitches push
    # values outside [0,1] which would trip robosuite's assertion).
    depth = np.clip(np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    z_metric = get_real_depth_map(sim, depth)

    # Camera matrices (robosuite handles MuJoCo conventions)
    intrinsic = get_camera_intrinsic_matrix(
        sim, camera_name, camera_height, camera_width,
    )
    extrinsic = get_camera_extrinsic_matrix(sim, camera_name)
    # extrinsic is world-to-camera; invert for camera-to-world
    # Legacy (pre-1c15d86) unprojection for tasks whose data was generated with it
    # (pick_place/can). Must match the eval env's LEGACY_UNPROJECTION setting.
    if legacy_unprojection:
        cam2world = np.linalg.inv(extrinsic)
    else:
        cam2world = extrinsic  # already camera->world; do NOT invert

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Segmentation mask: prefer object-only (paper §3.1) when object_geom_ids
    # is provided; otherwise legacy behavior (drop robot only).
    if object_geom_ids is not None:
        valid_mask = (
            np.isin(seg, list(object_geom_ids))
            & (z_metric > 0.1)
            & (z_metric < 5.0)
        )
    else:
        robot_geom_ids = _get_robot_geom_ids(sim)
        valid_mask = (
            ~np.isin(seg, list(robot_geom_ids))
            & (z_metric > 0.1)
            & (z_metric < 5.0)
        )

    if not valid_mask.any():
        return np.zeros((num_points, 3), dtype=np.float32)

    v_grid, u_grid = np.where(valid_mask)
    z_vals = z_metric[v_grid, u_grid]

    # Unproject to camera frame (OpenCV convention from robosuite extrinsic)
    x_cam = (u_grid - cx) * z_vals / fx
    if legacy_unprojection:
        y_cam = (v_grid - cy) * z_vals / fy   # legacy sign
    else:
        y_cam = -(v_grid - cy) * z_vals / fy  # OpenGL: y up, row v down
    pts_cam = np.stack([x_cam, y_cam, z_vals, np.ones_like(z_vals)], axis=-1)

    # Camera to world
    pts_world = (cam2world @ pts_cam.T).T[:, :3]

    # Workspace bounds (table area)
    ws_mask = (
        (pts_world[:, 2] > 0.78)
        & (pts_world[:, 2] < 1.3)
        & (pts_world[:, 0] > -0.5)
        & (pts_world[:, 0] < 0.8)
        & (pts_world[:, 1] > -0.6)
        & (pts_world[:, 1] < 0.6)
    )
    pts_world = pts_world[ws_mask]

    n = len(pts_world)
    if rng is None:
        rng = np.random.RandomState(0)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
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

    robosuite: [dx, dy, dz, dax, day, daz, grip]  in OSC-input units [-1, +1]
    PEFM:      [grip, vx, vy, vz, drx, dry, drz]  same OSC-input units, grip in [0, 1]

    Note: stored action stays in OSC-input units (no freq multiplication).
    base_robosuite_env.step() passes these directly to robosuite's OSC controller.
    """
    dx, dy, dz = robo_action[0], robo_action[1], robo_action[2]
    dax, day, daz = robo_action[3], robo_action[4], robo_action[5]
    grip_robo = robo_action[6]

    # Grip: -1/+1 -> 0/1
    grip_pefm = np.clip((grip_robo + 1.0) / 2.0, 0.0, 1.0)

    # Keep in OSC-input units (no scaling)
    vx, vy, vz = dx, dy, dz
    drx, dry, drz = dax, day, daz

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

    # Object-only segmentation (paper §3.1). Computed once per env.
    # Mirrors EquiBot sim_mobile's pattern: env declares its objects, we
    # trust the env. Body names come from authoritative env attributes
    # (env.objects / env.nuts / env.tool / env.frame / env.stand) plus
    # explicit per-task receptacle names (bins, peg1).
    object_body_names = resolve_object_body_names(env, task)
    print(f"[object-seg] resolved object bodies for task={task}: {object_body_names}")
    object_geom_ids = _get_object_geom_ids(env.sim, object_body_names, verbose=True)

    # Renderer warmup: two consecutive env.reset() calls. A freshly
    # created robosuite env has a fragile initial render state; the FIRST
    # reset triggers the renderer's lazy init and the SECOND moves it to
    # the stable state. Caught in bisect: with one reset only, demo 0
    # loses ~99% of its frames. With two, all demos are clean.
    env.reset()
    env.reset()

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

            # Render via _get_observations(force_update=True). Multiple
            # consecutive sim.render(...) calls (rgb / depth / segmentation)
            # corrupt MuJoCo's internal scene cache after the first frame.
            # _get_observations produces all three buffers in one pass.
            #
            # IMPORTANT: copy() each buffer before downstream ops. The returned
            # arrays are views into MuJoCo's internal pixel buffers; passing
            # them through numpy ops without copying corrupts the buffer and
            # makes ALL subsequent renders return empty seg masks. Caught in
            # bisect: with copy() seg.overlap stays at 6 across frames; without
            # copy() it drops to 0 from t=1 onward.
            obs_dict = env._get_observations(force_update=True)
            rgb       = obs_dict["agentview_image"].copy()
            depth_map = obs_dict["agentview_depth"][:, :, 0].copy()
            seg_map   = obs_dict["agentview_segmentation_element"][:, :, 0].copy()
            # Pull frontview HERE, BEFORE extract_point_cloud. Reading
            # frontview after extract_point_cloud puts the renderer in a
            # state where ~85% of subsequent agentview seg masks come back
            # empty. Reading both cameras up front avoids that.
            side_img = obs_dict["frontview_image"].copy()

            pc = extract_point_cloud(
                env.sim, depth_map, seg_map,
                camera_name="agentview",
                camera_height=render_res,
                camera_width=render_res,
                legacy_unprojection=(task == "can"),
                num_points=num_points,
                rng=rng,
                object_geom_ids=object_geom_ids,
            )

            # Action conversion
            action = convert_action(actions[t], freq=freq)

            # EEF observation (from HDF5 — ground truth from collection)
            if has_eef and has_quat and has_gripper:
                eef_pos_val = obs_grp["robot0_eef_pos"][t]
                eef_quat_val = obs_grp["robot0_eef_quat"][t]
                gripper_qpos_val = obs_grp["robot0_gripper_qpos"][t]
            else:
                eef_pos_val = np.zeros(3)
                eef_quat_val = np.array([0, 0, 0, 1.0])
                gripper_qpos_val = np.array([0.04, 0.04])
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

            # Video frame (dual view): side_img was already copied from
            # obs_dict above (must happen before extract_point_cloud).
            front_img = rgb
            dual = np.concatenate([front_img, side_img], axis=1)
            vid_frames.append(dual)

            frame_count += 1

            # Reading sim.data.cam_xpos / cam_xmat inside extract_point_cloud
            # leaves MuJoCo's lazily-computed kinematics in a state that the
            # next set_state_from_flattened doesn't fully refresh, which
            # corrupts the segmentation buffer for subsequent frames. A
            # trailing forward() forces a clean recompute. Caught in bisect:
            # without this call only every other frame produces a valid PC.
            env.sim.forward()

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
