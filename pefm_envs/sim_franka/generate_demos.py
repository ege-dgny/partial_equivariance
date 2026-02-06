"""
Demo generation for single Franka Panda tabletop environments.

Generates demonstrations with mixed-frame sketch handling:
- Object-relative phases are rotated with the object
- World-frame phases remain fixed regardless of object orientation

Records 2-camera views (front + side) stitched side-by-side.
"""

import os
import sys
import cv2
import logging
import argparse
import numpy as np
import pybullet

from pefm_envs.sim_mobile.utils.anchors import create_trajectory
from pefm_envs.sim_mobile.utils.init_utils import rotate_around_z
from pefm_envs.sim_mobile.utils.multi_camera import MultiCamera

np.set_printoptions(precision=2, linewidth=150, threshold=10000, suppress=True)


# ------------------------------------------------------------------ #
#  Environment factory
# ------------------------------------------------------------------ #

def get_env_class(task_name):
    if task_name == "pick_place":
        from .pick_place_env import PickPlaceEnv
        return PickPlaceEnv
    elif task_name == "peg_insert":
        from .peg_insert_env import PegInsertEnv
        return PegInsertEnv
    elif task_name == "centering":
        from .centering_env import CenteringEnv
        return CenteringEnv
    else:
        raise ValueError(f"Unknown task: {task_name}")


# ------------------------------------------------------------------ #
#  Sketch utilities
# ------------------------------------------------------------------ #

def rotate_sketch(sketch, ang):
    """Rotate all sketch positions around Z axis."""
    sketch = np.array(sketch)  # (T, 4) for single EEF
    gripper_ac = sketch[..., [0]]
    eef_pos = sketch[..., 1:]
    eef_pos = rotate_around_z(eef_pos.reshape(-1, 3), ang).reshape(eef_pos.shape)
    sketch = np.concatenate([gripper_ac, eef_pos], axis=-1)
    return [tuple(s) for s in sketch] if sketch.ndim == 2 else sketch.tolist()


def split_and_rotate_sketch(sketch, object_phases, ang, object_center=None):
    """
    Rotate only object-relative phases; leave world-frame phases untouched.

    Object-relative phases are defined as offsets from the object center
    (e.g. (0, 0, z) means "directly above the object"). After rotation,
    the object's world-frame XY position is added.

    Args:
        sketch: list of (grip, x, y, z) tuples — one per phase
        object_phases: set of phase indices that are object-relative
        ang: rotation angle
        object_center: (2,) XY position of the object in world frame
    """
    if object_center is None:
        object_center = np.array([0.0, 0.0])
    result = []
    for i, step in enumerate(sketch):
        if i in object_phases:
            step_arr = np.array([step])
            rotated = rotate_sketch(step_arr, ang)
            g, x, y, z = rotated[0] if isinstance(rotated[0], (list, np.ndarray, tuple)) else rotated[0]
            result.append((float(g), float(x + object_center[0]),
                           float(y + object_center[1]), float(z)))
        else:
            result.append(step)
    return result


def plan_actions_from_sketch(init_eef_pos, sketch, init_grip, sim_freq,
                              num_sec_per_unit=2.0):
    """
    Convert sketch waypoints into action sequence for a single EEF.

    Args:
        init_eef_pos: (3,) initial EEF position
        sketch: list of (grip, x, y, z) tuples
        init_grip: initial gripper state
        sim_freq: control frequency
        num_sec_per_unit: time per unit distance

    Returns:
        actions: (T, 7) array [grip, vx, vy, vz, drx, dry, drz]
    """
    curr_pos = init_eef_pos.copy()[:3]
    actions = []
    prev_grip = init_grip

    for grip, tx, ty, tz in sketch:
        target = np.array([tx, ty, tz])
        num_steps = int(num_sec_per_unit * sim_freq)
        dist = np.linalg.norm(target - curr_pos)
        waypoint_steps = max(int(dist * num_steps), 1)

        traj = create_trajectory(
            [curr_pos, target], [waypoint_steps, 0], sim_freq
        )

        # Prepend gripper action and zero orientation velocity
        for t_step in range(len(traj)):
            g = prev_grip if t_step < len(traj) - 1 else grip
            pos_target = traj[t_step, :3]
            action = np.array([g, pos_target[0], pos_target[1], pos_target[2],
                               0.0, 0.0, 0.0])
            actions.append(action)

        prev_grip = grip
        curr_pos = target

    return np.array(actions)


# ------------------------------------------------------------------ #
#  Video
# ------------------------------------------------------------------ #

def save_video(frames, path, fps=10):
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        f = np.clip(frame, 0, 255).astype(np.uint8)
        writer.write(f[..., ::-1] if f.shape[-1] == 3 else f)
    writer.release()


# ------------------------------------------------------------------ #
#  Demo runner
# ------------------------------------------------------------------ #

def run_demo(args, counter=0):
    os.makedirs(os.path.join(args.data_out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.data_out_dir, "pcs"), exist_ok=True)
    prefix = args.data_out_dir.split("/")[-1]
    episode_name = f"{prefix}_ep{counter:06d}"
    saved_files = []

    np.random.seed(args.seed)
    seed_env = args.seed_env if args.seed_env is not None else args.seed
    rng_env = np.random.RandomState(seed_env)

    # Create env
    args.demo_mode = True
    env = get_env_class(args.task_name)(args, rng_env)
    obs = env.reset()

    ang = env._object_rotation[-1]

    # Build sketch based on task
    if args.task_name == "pick_place":
        cyl_pos, _ = env.sim.getBasePositionAndOrientation(env._cylinder_id)
        cx, cy = cyl_pos[0], cyl_pos[1]
        ch = env.CYLINDER_HEIGHT
        tray_x, tray_y = env.TRAY_POS[0], env.TRAY_POS[1]
        tray_z = env.TRAY_SIZE[2]

        sketch = [
            # Phase 0 (object-relative): approach above cylinder
            (0, 0.0, 0.0, ch * 2.5),
            # Phase 1 (object-relative): descend to grasp
            (1, 0.0, 0.0, ch * 0.7),
            # Phase 2 (object-relative): lift
            (1, 0.0, 0.0, ch * 3.0),
            # Phase 3 (world-frame): move over tray
            (1, tray_x, tray_y, ch * 3.0),
            # Phase 4 (world-frame): lower onto tray and release
            (0, tray_x, tray_y, tray_z + ch * 0.6),
        ]
        object_phases = {0, 1, 2}
        sketch = split_and_rotate_sketch(
            sketch, object_phases, ang,
            object_center=np.array([cx, cy]),
        )

    elif args.task_name == "peg_insert":
        peg_pos, _ = env.sim.getBasePositionAndOrientation(env._peg_id)
        px, py = peg_pos[0], peg_pos[1]
        peg_h = env.PEG_HEIGHT
        sx, sy = env.SOCKET_POS[0], env.SOCKET_POS[1]
        insert_z = env.PLATE_THICKNESS + peg_h * 0.1

        # Grasp the peg at its center. The peg geometry center is at peg_h/2
        # above the URDF origin (which is at table level + 0.001).
        # We use panda_grasptarget as EE, which has accurate IK.
        grasp_z = peg_h * 0.5  # Peg center height
        safe_z = 0.30  # Safe height to avoid collisions during lateral moves

        sketch = [
            # Phase 0 (world-frame): first go to safe height near home X
            # This avoids collision during the initial approach
            (0, 0.35, 0.0, safe_z),
            # Phase 1 (object-relative): approach above peg at safe height
            (0, 0.0, 0.0, safe_z),
            # Phase 2 (object-relative): descend to just above peg
            (0, 0.0, 0.0, peg_h * 2.5),
            # Phase 3 (object-relative): descend to grasp
            (1, 0.0, 0.0, grasp_z),
            # Phase 4 (object-relative): lift
            (1, 0.0, 0.0, safe_z),
            # Phase 5 (world-frame): move over socket
            (1, sx, sy, safe_z),
            # Phase 6 (world-frame): insert into socket
            (1, sx, sy, insert_z),
        ]
        object_phases = {1, 2, 3, 4}  # Phases that are object-relative
        sketch = split_and_rotate_sketch(
            sketch, object_phases, ang,
            object_center=np.array([px, py]),
        )

    elif args.task_name == "centering":
        cyl_pos, _ = env.sim.getBasePositionAndOrientation(env._cylinder_id)
        cx, cy = cyl_pos[0], cyl_pos[1]
        ch = env.CYLINDER_HEIGHT
        lift_h = env.LIFT_HEIGHT

        sketch = [
            # Phase 0 (object-relative): approach above cylinder
            (0, 0.0, 0.0, ch * 2.5),
            # Phase 1 (object-relative): descend to grasp
            (1, 0.0, 0.0, ch * 0.7),
            # Phase 2 (object-relative): lift
            (1, 0.0, 0.0, lift_h),
            # Phase 3 (object-relative): lower back to spawn
            (1, 0.0, 0.0, ch * 0.8),
            # Phase 4 (object-relative): release
            (0, 0.0, 0.0, ch * 0.8),
        ]
        # All phases object-relative (fully symmetric task)
        object_phases = {0, 1, 2, 3, 4}
        sketch = split_and_rotate_sketch(
            sketch, object_phases, ang,
            object_center=np.array([cx, cy]),
        )

    else:
        raise ValueError(f"Unknown task: {args.task_name}")

    # Convert sketch to actions
    init_eef_pos = obs[0, :3]
    init_grip = 0.0
    sim_freq = env.freq
    num_sec_per_unit = 20.0 / args.speed_multiplier

    # Execute sketch step by step
    t = 0
    record_t = 0
    imgs = []
    sim_unstable = False

    for step_idx, step_target in enumerate(sketch):
        if sim_unstable:
            break

        grip, tx, ty, tz = step_target
        prev_grip = init_grip if step_idx == 0 else sketch[step_idx - 1][0]

        step_actions = plan_actions_from_sketch(
            obs[0], [(grip, tx, ty, tz)], prev_grip, sim_freq,
            num_sec_per_unit=num_sec_per_unit,
        )

        for step_t, action_raw in enumerate(step_actions):
            if sim_unstable:
                break

            # Convert absolute position to velocity
            grip_ac = action_raw[0]
            target_pos = action_raw[1:4]
            eef_vel = (target_pos - obs[0, :3]) * env.freq
            eef_vel = np.clip(eef_vel, -1.0, 1.0)
            action = np.array([grip_ac, *eef_vel, 0.0, 0.0, 0.0])

            # Record
            should_record = (
                t % args.cam_rec_interval == 0
                if args.cam_rec_interval > 0
                else (step_t == len(step_actions) - 1 or
                      (step_idx == 0 and step_t == 0))
            )
            if should_record:
                # Render point cloud from front camera
                front_cam = env.default_front_camera.copy()
                render_dict = env.render(
                    cam_config=front_cam,
                    return_depth=True,
                    return_pc=True,
                    return_seg=True,
                    resolution=240,
                )

                pc = render_dict["pc"]
                img = render_dict["images"][0][..., :3]

                if len(pc) == 0:
                    sim_unstable = True
                    print("Warning: no point cloud; cutting episode short.")
                    break

                if np.min(pc[:, 2]) < -0.05 or np.max(pc[:, 2]) > 1.0:
                    sim_unstable = True
                    print("Warning: simulation unstable; cutting episode.")
                    break

                # Subsample point cloud
                num_points = 4096
                if len(pc) >= num_points:
                    idx = np.random.choice(len(pc), size=num_points, replace=False)
                    pc = pc[idx]

                img_name = f"{prefix}_ep{counter:06d}_view0_t{record_t:02d}"
                save_path = os.path.join(
                    args.data_out_dir, "pcs", f"{img_name}.npz"
                )
                np.savez(
                    save_path, pc=pc, rgb=img, action=action, eef_pos=obs,
                )
                saved_files.append(save_path)

                # Record dual-view frame for video
                dual_frame = env.render_dual(resolution=240)
                imgs.append(dual_frame)

                record_t += 1

            obs, _, _, _ = env.step(action, dummy_reward=True)
            t += 1

    # Evaluate
    final_rew = env.compute_reward()
    task = args.task_name
    if task == "pick_place":
        cyl_pos, _ = env.sim.getBasePositionAndOrientation(env._cylinder_id)
        print(f"Reward: {final_rew:.3f} | cyl: [{cyl_pos[0]:.3f}, {cyl_pos[1]:.3f}, {cyl_pos[2]:.3f}] | tray: [{env.TRAY_POS[0]:.2f}, {env.TRAY_POS[1]:.2f}]")
    elif task == "peg_insert":
        peg_pos, peg_quat = env.sim.getBasePositionAndOrientation(env._peg_id)
        peg_euler = pybullet.getEulerFromQuaternion(peg_quat)
        print(f"Reward: {final_rew:.3f} | peg: [{peg_pos[0]:.3f}, {peg_pos[1]:.3f}, {peg_pos[2]:.3f}] | z_rot: {np.degrees(peg_euler[2]):.1f}deg")
    elif task == "centering":
        cyl_pos, _ = env.sim.getBasePositionAndOrientation(env._cylinder_id)
        print(f"Reward: {final_rew:.3f} | cyl: [{cyl_pos[0]:.3f}, {cyl_pos[1]:.3f}, {cyl_pos[2]:.3f}] | target: [{env._target_xy[0]:.2f}, {env._target_xy[1]:.2f}]")
    else:
        print(f"Reward: {final_rew:.3f}")

    if final_rew >= args.data_rew_threshold:
        video_path = os.path.join(
            args.data_out_dir, "images", episode_name + ".mp4"
        )
        if len(imgs) > 0:
            save_video(np.array(imgs), video_path, fps=10)
            print(f"Saved video to {video_path}")
        return 1
    else:
        for f in saved_files:
            if os.path.exists(f):
                os.remove(f)
        return 0


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def get_args():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Franka PEFM Demo Generation")

    # Task
    parser.add_argument(
        "--task_name", type=str, default="pick_place",
        choices=["pick_place", "peg_insert", "centering"],
    )
    # Seeds
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_env", type=int, default=666)
    parser.add_argument("--seed_cam", type=int, default=66666)
    # Simulation
    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--num_eef", type=int, default=1)
    parser.add_argument("--max_episode_length", type=int, default=80)
    parser.add_argument("--ac_noise", type=float, default=0.02)
    parser.add_argument("--freq", type=int, default=5)
    # Object randomization
    parser.add_argument("--randomize_rotation", action="store_true")
    parser.add_argument("--randomize_scale", action="store_true")
    parser.add_argument("--uniform_scaling", action="store_true")
    # Camera
    parser.add_argument("--cam_resolution", type=int, default=240)
    parser.add_argument("--cam_rec_interval", type=int, default=5)
    parser.add_argument("--vis", action="store_true")
    # Data
    parser.add_argument("--num_demos", type=int, default=1)
    parser.add_argument("--data_out_dir", type=str, default=None)
    parser.add_argument("--data_rew_threshold", type=float, default=0.9)
    parser.add_argument("--speed_multiplier", type=float, default=1.0)

    args, _ = parser.parse_known_args()

    if args.data_out_dir is None:
        args.data_out_dir = os.path.join("data", args.task_name)

    return args


def main():
    args = get_args()
    seed = args.seed
    seed_env = args.seed_env

    num_success = 0
    for i in range(args.num_demos * 10):
        if num_success >= args.num_demos:
            break
        args.seed = (seed * 99999 + i) % 100001
        args.seed_env = (seed_env * 99999 + i) % 100001
        success = run_demo(args, i)
        if success:
            num_success += 1
            print(f"[{num_success}/{args.num_demos}] demos completed")
    print(f"Done. Generated {num_success}/{args.num_demos} demos.")


if __name__ == "__main__":
    main()
