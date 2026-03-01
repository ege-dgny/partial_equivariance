"""
Demonstration generation for PEFM experiment environments.

Generates demonstrations with mixed-frame sketch handling:
- Object-relative phases are rotated with the object
- World-frame phases remain fixed regardless of object orientation

This creates the symmetry conflict that PEFM is designed to learn.
"""

import os
import sys
import cv2
import logging
import argparse
import numpy as np
import pybullet as p
from tqdm import tqdm
from glob import glob

from .utils.anchors import create_trajectory
from .utils.multi_camera import MultiCamera, get_camera_info
from .utils.project import unproject_depth
from .utils.init_utils import rotate_around_z


np.set_printoptions(precision=2, linewidth=150, threshold=10000, suppress=True)


def get_env_class(task_name):
    """Get the environment class for a given task name."""
    if task_name == "pour":
        from .pouring_env import PouringEnv
        return PouringEnv
    elif task_name == "insert":
        from .insertion_env import InsertionEnv
        return InsertionEnv
    elif task_name == "compass_close":
        from .compass_closing_env import CompassClosingEnv
        return CompassClosingEnv
    else:
        raise ValueError(f"Unknown task name: {task_name}")


def plan_actions_from_sketch(
    init_anchor_pos,
    sketch,
    initial_gripper_pose,
    sim_frequency,
    num_sec_per_unit=2.0,
):
    """
    Given a sketch for where manipulation anchors should move to, plan a
    sequence of actions.

    Args:
        init_anchor_pos: initial anchor position
        sketch: a list of tuple; each tuple contains 2 items, that describe
                the desired x and y coordinates of the anchor
        sim_frequency: simulation frequency
        num_sec_per_unit: how much time to take for traversing distance with
                          length equal to 1 unit

    Returns:
        actions: a list of actions to be executed in the environment
    """
    # create buffer for current anchor attach positions
    curr_anchor_pos = init_anchor_pos.copy()[:, :3]

    # init buffer for generated actions and desired positions
    num_anchors = len(sketch[0])
    ac_dim = 7 * num_anchors
    actions = np.zeros([0, ac_dim])

    prev_grip_ac = initial_gripper_pose

    # loop through each item in the sketch and generate actions
    for i, anchor_targets in enumerate(sketch):
        anchor_actions = []
        for j, anchor_target in enumerate(anchor_targets):
            grip_ac, target_x, target_y, target_z = anchor_target
            num_steps = int(num_sec_per_unit * sim_frequency)
            waypoints = [
                curr_anchor_pos[j],
                np.array([target_x, target_y, target_z]),
            ]
            waypoint_dists = [
                np.linalg.norm(waypoints[k + 1] - waypoints[k])
                for k in range(len(waypoints) - 1)
            ]
            waypoint_steps = [
                max(int(waypoint_dist * num_steps), 1)
                for waypoint_dist in waypoint_dists
            ] + [0]

            anchor_j_actions = create_trajectory(
                waypoints, waypoint_steps, sim_frequency
            )
            anchor_j_actions = np.concatenate(
                [
                    np.ones_like(anchor_j_actions[:, [0]]) * prev_grip_ac[j],
                    anchor_j_actions,
                ],
                axis=1,
            )
            anchor_actions.append(anchor_j_actions)

        # fill no-op actions for shorter sequences
        max_len = np.max([len(acs) for acs in anchor_actions])
        for j in range(len(anchor_actions)):
            curr_len = len(anchor_actions[j])
            if curr_len < max_len:
                ac_dim = len(anchor_actions[j][0])
                noop = np.concatenate([anchor_targets[j], np.zeros(3)])
                anchor_actions[j] = np.concatenate(
                    [anchor_actions[j], np.array([noop] * (max_len - curr_len))]
                )
            anchor_actions[j][-1][0] = anchor_targets[j][0]

        actions = np.concatenate([actions, np.concatenate(anchor_actions, axis=1)])

        prev_grip_ac = np.array([x[0] for x in anchor_targets])
        curr_anchor_pos = [np.array([x[1], x[2], x[3]]) for x in anchor_targets]

    return actions


def rotate_sketch(sketch, ang):
    """Rotate all sketch positions around Z axis."""
    sketch = np.array(sketch)  # (T, E, 4)
    gripper_ac, eef_pos = sketch[..., [0]], sketch[..., 1:]  # TE1, TE3
    eef_pos = rotate_around_z(eef_pos.reshape(-1, 3), ang).reshape(eef_pos.shape)
    sketch = np.concatenate([gripper_ac, eef_pos], axis=-1)
    sketch = [[tuple(ss) for ss in s] for s in sketch]
    return sketch


def split_and_rotate_sketch(sketch, object_phases, ang):
    """
    Rotate only object-relative phases; leave world-frame phases untouched.

    This is the core PEFM difference from EquiBot demos: some sketch phases
    use object-relative coordinates (rotated with the object) while others
    use world-frame coordinates (never rotated).

    Args:
        sketch: list of sketch steps
        object_phases: set of phase indices that are object-relative
        ang: rotation angle
    """
    result = []
    for i, step in enumerate(sketch):
        if i in object_phases:
            rotated = rotate_sketch([step], ang)
            result.append(rotated[0])
        else:
            result.append(step)
    return result


def save_video(frames, path, fps=10):
    """Save frames as MP4 video."""
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        writer.write(frame[..., ::-1] if frame.shape[-1] == 3 else frame)
    writer.release()


def run_demo(args, counter=0):
    # setup directories
    os.makedirs(os.path.join(args.data_out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.data_out_dir, "pcs"), exist_ok=True)
    prefix = args.data_out_dir.split("/")[-1]
    episode_name = f"{prefix}_ep{counter:06d}"
    saved_files = []

    # seeding
    np.random.seed(args.seed)

    seed_env = args.seed_env if args.seed_env is not None else args.seed
    seed_cam = args.seed_cam if args.seed_cam is not None else args.seed
    rng_env = np.random.RandomState(seed_env)
    rng_cam = np.random.RandomState(seed_cam)

    # create simulation env (demo_mode signals envs to use simplified settings)
    args.demo_mode = True
    env = get_env_class(args.task_name)(args, rng_env)

    # get initial positions of anchors
    obs = env.reset()

    # plan actions based on task
    wait_steps = 0
    init_gripper_pose_val = 0
    ang = env._object_rotation[-1]

    if args.task_name == "pour":
        init_gripper_pose_val = 0
        mug_pos, _ = env.sim.getBasePositionAndOrientation(env._mug_id)
        mug_x, mug_y = mug_pos[0], mug_pos[1]
        mug_h = env.MUG_HEIGHT
        bowl_x, bowl_y = env.BOWL_WORLD_POS[0], env.BOWL_WORLD_POS[1]

        # Object-relative phases (0-2): approach, grasp, lift mug
        # World-frame phases (3-4): move to bowl, pour
        sketch = [
            # Phase 0 (object-relative): approach mug from above
            [(0, 0.0, 0.0, mug_h * 2.0), (0, 0.0, 0.0, mug_h * 2.0)],
            # Phase 1 (object-relative): descend to mug grasp height
            [(1, 0.0, 0.0, mug_h * 0.8), (0, 0.0, 0.0, mug_h * 2.0)],
            # Phase 2 (object-relative): lift mug
            [(1, 0.0, 0.0, mug_h * 3.0), (0, 0.0, 0.0, mug_h * 2.0)],
            # Phase 3 (world-frame): move over bowl
            [(1, bowl_x, bowl_y, mug_h * 3.0), (0, bowl_x + 0.1, bowl_y, mug_h * 2.0)],
            # Phase 4 (world-frame): lower toward bowl (pour)
            [(1, bowl_x, bowl_y, mug_h * 1.5), (0, bowl_x + 0.1, bowl_y, mug_h * 2.0)],
        ]
        object_phases = {0, 1, 2}
        sketch = split_and_rotate_sketch(sketch, object_phases, ang)

    elif args.task_name == "insert":
        init_gripper_pose_val = 0
        peg_h = env.PEG_HEIGHT
        socket_x, socket_y = env.SOCKET_WORLD_POS[0], env.SOCKET_WORLD_POS[1]
        socket_top_z = env.PLATE_THICKNESS + env.WALL_HEIGHT

        # Object-relative phases (0-2): approach, grasp, lift peg
        # World-frame phases (3-4): move over socket, insert
        # Insertion target: lower the EEF to just above the plate so the peg
        # descends into the socket cavity (plate_thickness + small margin)
        insert_z = env.PLATE_THICKNESS + peg_h * 0.1  # EEF target for insertion
        sketch = [
            # Phase 0 (object-relative): approach peg from above
            [(0, 0.0, 0.0, peg_h * 2.5), (0, 0.0, 0.0, peg_h * 2.5)],
            # Phase 1 (object-relative): descend to grasp
            [(1, 0.0, 0.0, peg_h * 0.7), (0, 0.0, 0.0, peg_h * 2.5)],
            # Phase 2 (object-relative): lift peg
            [(1, 0.0, 0.0, peg_h * 3.0), (0, 0.0, 0.0, peg_h * 2.5)],
            # Phase 3 (world-frame): move over socket
            [(1, socket_x, socket_y, peg_h * 3.0), (0, socket_x + 0.1, socket_y, peg_h * 2.5)],
            # Phase 4 (world-frame): insert into socket
            [(1, socket_x, socket_y, insert_z), (0, socket_x + 0.1, socket_y, peg_h * 2.5)],
        ]
        object_phases = {0, 1, 2}
        sketch = split_and_rotate_sketch(sketch, object_phases, ang)

    elif args.task_name == "compass_close":
        init_gripper_pose_val = 0
        L, W, H = env._box_size
        T = env._box_thickness

        # All phases are in box-local coordinates (rotated with the box),
        # but the ORDER of which flap to close is determined by the
        # cardinal mapping (world-frame)
        ordered_flaps = env._ordered_flap_joints
        flap_names = {v: k for k, v in env._FLAP_JOINT_INDICES.items()}

        # Build sketch: 2 phases per flap (approach + push)
        sketch = []
        for flap_joint_idx in ordered_flaps:
            flap_name = flap_names[flap_joint_idx]
            # Determine push direction based on flap
            if flap_name == "flap_left":
                approach = [(-L * 0.8, W * 0.0, H + L / 2), (L * 0.4, W * 0.0, H + L)]
                push = [(-L * 0.4, W * 0.0, H + L / 2), (L * 0.4, W * 0.0, H + L)]
            elif flap_name == "flap_right":
                approach = [(L * 0.8, W * 0.0, H + L / 2), (-L * 0.4, W * 0.0, H + L)]
                push = [(L * 0.4, W * 0.0, H + L / 2), (-L * 0.4, W * 0.0, H + L)]
            elif flap_name == "flap_back":
                approach = [(-L * 0.4, W * 0.7, H + L / 2), (L * 0.4, W * 0.7, H + L / 2)]
                push = [(-L * 0.4, W * 1.8, H * 0.8), (L * 0.4, W * 1.8, H * 0.8)]
            elif flap_name == "flap_front":
                approach = [(-L * 0.4, -W * 0.7, H + L / 2), (L * 0.4, -W * 0.7, H + L / 2)]
                push = [(-L * 0.4, -W * 1.8, H * 0.8), (L * 0.4, -W * 1.8, H * 0.8)]
            else:
                continue

            # Convert to sketch format (grip_ac, x, y, z)
            sketch.append([(0, *approach[0]), (0, *approach[1])])
            sketch.append([(0, *push[0]), (0, *push[1])])

        # All phases are object-relative
        object_phases = set(range(len(sketch)))
        sketch = split_and_rotate_sketch(sketch, object_phases, ang)

    else:
        raise ValueError(f"Task name {args.task_name} not found.")

    # create buffers for episode info
    imgs = []

    # execute the planned actions
    t = 0
    record_t = 0
    cam_dist = args.cam_dist
    sim_unstable = False
    for step_idx, step in enumerate(sketch):
        if sim_unstable:
            break
        if step_idx == 0:
            initial_gripper_pose = (
                np.array([x[0] for x in sketch[0]]) * init_gripper_pose_val
            )
        else:
            initial_gripper_pose = np.array([x[0] for x in sketch[step_idx - 1]])
        sketch_step = [step]
        sim_freq = env.freq
        num_sec_per_unit = 20.0 / args.speed_multiplier
        actions = plan_actions_from_sketch(
            obs,
            sketch_step,
            initial_gripper_pose,
            sim_freq,
            num_sec_per_unit=num_sec_per_unit,
        )
        print(f"Length of actions: {len(actions)}")

        for step_t, action in enumerate(actions):
            if sim_unstable:
                break

            action = action.reshape(-1, 7).copy()
            grip_actions = action[:, [0]]
            expected_eef = action[:, 1:4]
            eef_actions = (action[:, 1:4] - obs[:, :3]) * env.freq
            eef_actions = np.clip(eef_actions, -1.0, 1.0)
            action = np.concatenate([grip_actions, eef_actions], axis=-1).flatten()
            if args.dof == 7:
                action = np.concatenate(
                    [action.reshape(-1, 4), np.zeros((len(action) // 4, 3))], axis=-1
                )

            should_record = (
                t % args.cam_rec_interval == 0
                if args.cam_rec_interval > 0
                else (step_t == len(actions) - 1 or (step_idx == 0 and step_t == 0))
            )
            if should_record:
                yaws, pitches = [], []

                def sample_vals(s):
                    if len(s) == 2:
                        return (
                            rng_cam.rand(args.cam_num_views) * (s[1] - s[0]) + s[0]
                        )
                    elif len(s) == 1:
                        return s
                    else:
                        raise ValueError("Length of {s} should be 1 or 2.")

                for y in sample_vals(args.cam_yaws):
                    for pp in sample_vals(args.cam_pitches):
                        yaws.append(y)
                        pitches.append(pp)
                num_views = len(yaws)

                # Scale cam_dist by rigid object scale (matching original behavior)
                scale_factor = np.max(env._rigid_object_scale) if np.max(env._rigid_object_scale) > 0 else 1.0

                cam_info = {
                    "yaws": yaws,
                    "pitches": pitches,
                    "dist": cam_dist * scale_factor,
                    "views": list(np.arange(num_views)),
                    "fov": 30,
                    "width": 240,
                    "height": 240,
                }
                H, W = cam_info["height"], cam_info["width"]
                render_dict = env.render(
                    cam_info=cam_info,
                    return_depth=True,
                    return_seg=True,
                    hide_eef=True,
                )
                view_images = render_dict["images"]
                view_depths = render_dict["depths"]
                view_segs = render_dict["segs"]

                # get mesh data of the object of interest
                soft_obj_ids = (
                    env.soft_obj_ids
                    if hasattr(env, "soft_obj_ids")
                    else env.soft_ids
                )
                rigid_obj_ids = (
                    env.rigid_obj_ids
                    if hasattr(env, "rigid_obj_ids")
                    else env.rigid_ids
                )
                cam_target = (
                    env._cam_target
                    if hasattr(env, "_cam_target")
                    else env.camera_config["target"]
                )
                obj_ids = soft_obj_ids[:1] + rigid_obj_ids
                mesh_xyzs_list = [
                    env.sim.getMeshData(obj_id)[1] for obj_id in soft_obj_ids[:1]
                ]
                mesh_xyzs_list += [
                    env._get_rigid_body_mesh(obj_id) for obj_id in rigid_obj_ids
                ]
                mesh_idxs = np.concatenate(
                    [np.full((len(x),), i) for i, x in enumerate(mesh_xyzs_list)]
                )
                mesh_xyzs = np.concatenate(mesh_xyzs_list)
                cam_vals = MultiCamera.get_cam_vals(
                    [0] * num_views,
                    yaws,
                    pitches,
                    cam_info["dist"],
                    cam_target,
                    cam_info["fov"],
                    float(W / H),
                )

                for img_ix, img in enumerate(view_images):
                    img_name = (
                        f"{prefix}_ep{counter:06d}_view{img_ix}_t{record_t:02d}"
                    )

                    # process images
                    img = img[..., :3]

                    # compute segmentation
                    segs = view_segs[img_ix]
                    seg = np.isin(segs, obj_ids).astype(np.uint8) * 255
                    object_pixels = np.array(np.where(seg == 255)).T

                    # save projected mesh vertex pixel locations
                    view_mat, p_proj_mat = cam_vals[img_ix][:2]
                    view_mat = np.array(view_mat).reshape((4, 4), order="C")
                    p_proj_mat = np.array(p_proj_mat).reshape((4, 4), order="C")
                    cx = (1 - p_proj_mat[0, 2]) * W / 2
                    cy = (p_proj_mat[1, 2] + 1) * H / 2
                    proj_mat = np.array(
                        [
                            [-p_proj_mat[0, 0] * W / 2, 0, cx],
                            [0, p_proj_mat[1, 1] * H / 2, cy],
                            [0, 0, 1],
                        ]
                    )

                    # unproject segmentation mask to get partial PC
                    view_depth = view_depths[img_ix].T
                    extrinsics = view_mat.copy().T
                    extrinsics[[1, 2]] *= -1
                    extrinsics = np.linalg.inv(extrinsics)
                    intrinsics = proj_mat.copy()
                    intrinsics[0, 0] *= -1
                    partial_pc = unproject_depth(
                        [view_depth],
                        [intrinsics],
                        [extrinsics],
                        filter_pixels=[object_pixels],
                        clip_radius=10.0,
                    )
                    save_dir = os.path.join(args.data_out_dir, "pcs")
                    save_path = os.path.join(save_dir, f"{img_name}.npz")

                    if len(partial_pc) == 0:
                        sim_unstable = True
                        print(
                            f"Warning: simulation unstable; cutting episode short."
                        )
                        break
                    if (
                        np.min(partial_pc[:, 2]) < -0.05
                        or np.max(partial_pc[:, 2] > 1.0)
                    ):
                        sim_unstable = True
                        print(
                            f"Warning: simulation unstable; cutting episode short."
                        )
                        break

                    # subsample pc if too large
                    num_points = 4096
                    if len(partial_pc) >= num_points:
                        sampled_indices = np.random.choice(
                            len(partial_pc), size=4096, replace=False
                        )
                        partial_pc = partial_pc[sampled_indices]

                    np.savez(
                        save_path,
                        pc=partial_pc,
                        rgb=img,
                        action=action,
                        eef_pos=obs,
                    )
                    saved_files.append(save_path)

                cam_info_render = get_camera_info(args)
                cam_info_render["dist"] *= scale_factor
                img = env.render(cam_info=cam_info_render)["images"][0][..., :3]
                imgs.append(img)

                record_t += 1

            obs, _, _, _ = env.step(action, dummy_reward=True)

            t += 1

    final_rew = env.compute_reward()
    # Debug: print final state for diagnosing low rewards
    if args.task_name == "pour":
        mug_pos, _ = env.sim.getBasePositionAndOrientation(env._mug_id)
        print(f"Episode reward: {final_rew:.3f} | mug_pos: [{mug_pos[0]:.3f}, {mug_pos[1]:.3f}, {mug_pos[2]:.3f}] | bowl: [{env.BOWL_WORLD_POS[0]:.2f}, {env.BOWL_WORLD_POS[1]:.2f}]")
    elif args.task_name == "insert":
        peg_pos, peg_quat = env.sim.getBasePositionAndOrientation(env._peg_id)
        peg_euler = np.array(p.getEulerFromQuaternion(peg_quat))
        socket_top = env.PLATE_THICKNESS + env.WALL_HEIGHT
        print(f"Episode reward: {final_rew:.3f} | peg_pos: [{peg_pos[0]:.3f}, {peg_pos[1]:.3f}, {peg_pos[2]:.3f}] | socket: [{env.SOCKET_WORLD_POS[0]:.2f}, {env.SOCKET_WORLD_POS[1]:.2f}] | peg_z_rot: {np.degrees(peg_euler[2]):.1f}deg | socket_top_z: {socket_top:.3f}")
    else:
        print(f"Episode reward: {final_rew:.3f}")

    if final_rew >= args.data_rew_threshold:
        # write video to file
        video_path = os.path.join(args.data_out_dir, "images", episode_name + ".mp4")
        if len(imgs) > 0:
            save_video(np.array(imgs), video_path, fps=10)
            print(f"Saved video to {video_path}.")
        return 1
    else:
        for f in saved_files:
            if os.path.exists(f):
                os.remove(f)
        return 0


def get_args(parent=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="PEFM Demo Generation", add_help=False)

    # Main/demo args.
    parser.add_argument(
        "--task_name", type=str, default="pour",
        choices=["pour", "insert", "compass_close"],
        help="Name of the task"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--seed_env", type=int, default=0, help="Random seed for environment"
    )
    parser.add_argument(
        "--seed_cam", type=int, default=0, help="Random seed for camera"
    )
    # Simulation args.
    parser.add_argument(
        "--sim_frequency",
        type=int,
        default=500,
        help="Number of simulation steps per second",
    )
    parser.add_argument("--dof", type=int, default=7, help="Action dim")
    parser.add_argument("--num_eef", type=int, default=2, help="Number of end-effectors")
    parser.add_argument("--max_episode_length", type=int, default=50, help="Max episode length")
    parser.add_argument("--ac_noise", type=float, default=0.02, help="Action noise")
    # Object randomization args
    parser.add_argument("--randomize_rotation", action="store_true")
    parser.add_argument("--randomize_scale", action="store_true")
    parser.add_argument("--uniform_scaling", action="store_true")
    parser.add_argument(
        "--deform_bending_stiffness", type=float, default=0.01,
    )
    parser.add_argument(
        "--deform_damping_stiffness", type=float, default=1.0,
    )
    parser.add_argument(
        "--deform_elastic_stiffness", type=float, default=300.0,
    )
    parser.add_argument(
        "--deform_friction_coeff", type=float, default=10.0,
    )
    # Camera args.
    parser.add_argument(
        "--cam_resolution", type=int, default=240, help="Point cloud resolution"
    )
    parser.add_argument(
        "--cam_rec_interval", type=int, default=5,
        help="How many steps to skip between each cam shot",
    )
    parser.add_argument(
        "--cam_num_views", type=int, default=1, help="Number of views to sample."
    )
    parser.add_argument("--vis", action="store_true")
    # Data generation.
    parser.add_argument("--num_demos", type=int, default=1)
    parser.add_argument("--data_out_dir", type=str, default=None)
    parser.add_argument("--data_rew_threshold", type=float, default=0.9)
    parser.add_argument("--cam_pitches", type=int, nargs="*", default=[-75])
    parser.add_argument("--cam_yaws", type=int, nargs="*", default=[0])
    parser.add_argument("--cam_fov", type=int, default=45)
    parser.add_argument("--cam_dist", type=float, default=2.0)
    parser.add_argument("--speed_multiplier", type=float, default=1.0)

    args, unknown = parser.parse_known_args()

    # Set task-specific defaults
    if args.task_name == "pour":
        if args.cam_dist == 2.0:
            args.cam_dist = 1.5
        if args.cam_pitches == [-75]:
            args.cam_pitches = [-60]
    elif args.task_name == "insert":
        if args.cam_dist == 2.0:
            args.cam_dist = 1.2
        if args.cam_pitches == [-75]:
            args.cam_pitches = [-50]
    elif args.task_name == "compass_close":
        if args.cam_dist == 2.0:
            args.cam_dist = 1.5
        if args.cam_pitches == [-75]:
            args.cam_pitches = [-45]

    return args, unknown


def main():
    # read args
    args, _ = get_args()
    seed = args.seed
    seed_env = args.seed_env
    seed_cam = args.seed_cam

    pattern_ix = 0
    num_success = 0
    for i in range(args.num_demos * 10):  # retry budget
        if num_success >= args.num_demos:
            break
        args.seed = (seed * 99999 + pattern_ix) % 100001
        args.seed_env = (seed_env * 99999 + pattern_ix) % 100001
        args.seed_cam = (seed_cam * 99999 + pattern_ix) % 100001
        success = run_demo(args, pattern_ix)
        pattern_ix += 1
        if success:
            num_success += 1
            print(f"[{num_success}/{args.num_demos}] demos completed")
    print(f"Done. Generated {num_success}/{args.num_demos} demos.")


if __name__ == "__main__":
    main()
