"""
Demo generation for Genesis-backed Franka tabletop environments.

Reuses sketch and action planning from sim_franka; runs on Genesis with
contact-based grasping (collisions ON). Output NPZ format matches PEFM dataset.
"""

import os
import sys
import re
import logging
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

# Reuse sketch/planning from Franka (guarded PyBullet import)
from pefm_envs.sim_franka.generate_demos import (
    quat_error_axis_angle,
    plan_actions_from_sketch,
    plan_actions_with_orientation,
    is_7dim_sketch,
    split_and_rotate_sketch_7d,
    save_video,
)
from pefm_envs.sim_genesis.genesis_robot import _to_numpy

np.set_printoptions(precision=2, linewidth=150, threshold=10000, suppress=True)


def get_env_class(task_name):
    if task_name == "peg_insert":
        from .peg_insert_env import GenesisPegInsertEnv
        return GenesisPegInsertEnv
    elif task_name == "cup_pour":
        from .cup_pour_env import GenesisCupPourEnv
        return GenesisCupPourEnv
    elif task_name == "book_insert":
        from .book_insert_env import GenesisBookInsertEnv
        return GenesisBookInsertEnv
    else:
        raise ValueError(f"Unknown task: {task_name}")


def _euler_to_quat_xyzw(euler):
    """Euler (roll, pitch, yaw) in rad -> quat (x, y, z, w)."""
    r = Rotation.from_euler("xyz", euler)
    return r.as_quat()  # scipy uses xyzw


def run_demo(args, counter=0):
    os.makedirs(os.path.join(args.data_out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.data_out_dir, "pcs"), exist_ok=True)
    prefix = args.data_out_dir.split("/")[-1]
    episode_name = f"{prefix}_ep{counter:06d}"
    saved_files = []

    np.random.seed(args.seed)
    seed_env = args.seed_env if getattr(args, "seed_env", None) is not None else args.seed
    rng_env = np.random.RandomState(seed_env)

    args.demo_mode = True
    env = get_env_class(args.task_name)(args, rng_env)
    obs = env.reset()

    if args.task_name == "peg_insert":
        ang = env._object_rotation[-1] + getattr(env, "_peg_spawn_rotation", 0.0)
    else:
        ang = env._object_rotation[-1]

    # Build sketch — matching PyBullet reference structure
    if args.task_name == "peg_insert":
        peg_pos = _to_numpy(env._peg_entity.get_pos())
        px, py = peg_pos[0], peg_pos[1]
        peg_h = env.PEG_HEIGHT
        sx, sy = env.SOCKET_POS[0], env.SOCKET_POS[1]
        insert_z = env.PLATE_THICKNESS + peg_h * 2
        grasp_z = peg_h * 1.35
        safe_z = 0.30

        # Phase 0 is WORLD-FRAME (safe starting position near home X)
        # Phases 1-4 are object-relative (approach, descend, grasp, lift)
        # Phases 5-8 are WORLD-FRAME (transport, insert, release, retract)
        sketch = [
            (0, 0.35, 0.0, safe_z, np.pi, 0.0, 0.0),       # Phase 0: world-frame
            (0, 0.0, 0.0, safe_z, np.pi, 0.0, 0.0),         # Phase 1: object-relative
            (0, 0.0, 0.0, peg_h * 2.5, np.pi, 0.0, 0.0),    # Phase 2: object-relative
            (1, 0.0, 0.0, grasp_z, np.pi, 0.0, 0.0),        # Phase 3: object-relative
            (1, 0.0, 0.0, safe_z, np.pi, 0.0, 0.0),         # Phase 4: object-relative
            (1, sx, sy, safe_z, np.pi, 0.0, 0.0),            # Phase 5: world-frame
            (1, sx, sy, insert_z, np.pi, 0.0, 0.0),          # Phase 6: world-frame
            (0, sx, sy, insert_z, np.pi, 0.0, 0.0),          # Phase 7: world-frame
            (0, sx, sy, safe_z, np.pi, 0.0, 0.0),            # Phase 8: world-frame
        ]
        object_phases = {1, 2, 3, 4}  # Match PyBullet reference
        sketch = split_and_rotate_sketch_7d(sketch, object_phases, ang, object_center=np.array([px, py]))

    elif args.task_name == "cup_pour":
        cup_pos = _to_numpy(env._cup_entity.get_pos())
        cx, cy = cup_pos[0], cup_pos[1]
        cup_h = env.CUP_HEIGHT
        bowl_x, bowl_y = env.BOWL_POS[0], env.BOWL_POS[1]
        bowl_r = env.BOWL_RADIUS
        bowl_h = env.BOWL_HEIGHT
        safe_z = 0.25
        tilt_angle = getattr(env, "POUR_TILT_ANGLE", 2 * np.pi / 3)
        side_roll = -np.pi / 2
        side_pitch = -np.pi / 2
        approach_offset = 0.08
        grasp_z = cup_h * 0.5
        pour_x = bowl_x - bowl_r - 0.04
        pour_y = bowl_y - 0.05
        pour_z = bowl_h + cup_h
        tilt_yaw = 0.0
        sketch = [
            (0, 0.0, -approach_offset, safe_z, side_roll, side_pitch, 0.0),
            (0, 0.0, -approach_offset, grasp_z, side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z, side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z, side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z, side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z, side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, safe_z, side_roll, side_pitch, 0.0),
            (1, pour_x, pour_y, pour_z, side_roll, side_pitch, tilt_yaw),
            (1, pour_x, pour_y, pour_z, side_roll, side_pitch + tilt_angle, tilt_yaw),
            (1, pour_x, pour_y, pour_z, side_roll, side_pitch + tilt_angle, tilt_yaw),
            (1, pour_x, pour_y, pour_z, side_roll, side_pitch + tilt_angle, tilt_yaw),
            (1, pour_x, pour_y, safe_z, side_roll, side_pitch, tilt_yaw),
        ]
        object_phases = {0, 1, 2, 3, 4, 5, 6}
        sketch = split_and_rotate_sketch_7d(sketch, object_phases, ang, object_center=np.array([cx, cy]))

    elif args.task_name == "book_insert":
        book_pos = _to_numpy(env._book_entity.get_pos())
        bx, by = book_pos[0], book_pos[1]
        bl = env.BOOK_LENGTH
        hx, hy = env.HOLDER_POS[0], env.HOLDER_POS[1]
        shelf2_z = env.SHELF2_TOP_Z
        safe_z = 0.30
        side_roll = -np.pi / 2
        side_pitch = -np.pi / 2
        grasp_z = bl * 0.5
        approach_offset = 0.08
        place_z = shelf2_z + bl / 2 + 0.02
        place_y = hy - 0.12
        sketch = [
            (0, 0.0, -approach_offset, safe_z, side_roll, side_pitch, 0.0),
            (0, 0.0, -approach_offset, grasp_z, side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z, side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z, side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z, side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z, side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, safe_z, side_roll, side_pitch, 0.0),
            (1, hx, place_y, safe_z, side_roll, side_pitch, 0.0),
            (1, hx, place_y, place_z, side_roll, side_pitch, 0.0),
            (0, hx, place_y, place_z, side_roll, side_pitch, 0.0),
        ]
        object_phases = {0, 1, 2, 3, 4, 5, 6}
        sketch = split_and_rotate_sketch_7d(sketch, object_phases, ang, object_center=np.array([bx, by]))

    else:
        raise ValueError(f"Unknown task: {args.task_name}")

    sim_freq = env.freq
    num_sec_per_unit = 20.0 / getattr(args, "speed_multiplier", 0.5)
    t = 0
    record_t = 0
    imgs = []
    sim_unstable = False
    use_orientation = is_7dim_sketch(sketch)
    if use_orientation:
        ee_pos, ee_quat, _, _ = env.robot.get_ee_pos_quat_vel()
        init_eef_euler = Rotation.from_quat(ee_quat).as_euler("xyz")
    else:
        init_eef_euler = np.array([np.pi, 0.0, 0.0])
    curr_euler = init_eef_euler.copy()

    for step_idx, step_target in enumerate(sketch):
        if sim_unstable:
            break
        if use_orientation:
            grip = step_target[0]
            tx, ty, tz = step_target[1:4]
            target_euler = np.array(step_target[4:7])
            prev_grip = 0.0 if step_idx == 0 else sketch[step_idx - 1][0]
            step_actions = plan_actions_with_orientation(
                obs[0, :3], curr_euler, [(grip, tx, ty, tz, *target_euler)],
                sim_freq, num_sec_per_unit=num_sec_per_unit, init_grip=prev_grip,
            )
            curr_euler = target_euler.copy()
        else:
            grip, tx, ty, tz = step_target
            prev_grip = 0.0 if step_idx == 0 else sketch[step_idx - 1][0]
            step_actions = plan_actions_from_sketch(
                obs[0], [(grip, tx, ty, tz)], prev_grip, sim_freq,
                num_sec_per_unit=num_sec_per_unit,
            )

        for step_t, action_raw in enumerate(step_actions):
            if sim_unstable:
                break
            grip_ac = action_raw[0]
            target_pos = action_raw[1:4]
            # Cap position change per step
            delta = target_pos - obs[0, :3]
            dist = np.linalg.norm(delta) + 1e-9
            max_step = 0.04
            if dist > max_step:
                target_pos = obs[0, :3] + delta * (max_step / dist)
            eef_vel = (target_pos - obs[0, :3]) * env.freq
            eef_vel = np.clip(eef_vel, -1.0, 1.0)
            if use_orientation and len(action_raw) >= 7:
                ee_pos, ee_quat, _, _ = env.robot.get_ee_pos_quat_vel()
                target_euler = action_raw[4:7]
                target_quat = _euler_to_quat_xyzw(target_euler)
                axis_angle_err = quat_error_axis_angle(ee_quat, target_quat)
                ori_vel = axis_angle_err * env.freq
                ori_vel = np.clip(ori_vel, -1.0, 1.0)
                action = np.array([grip_ac, *eef_vel, *ori_vel])
            else:
                action = np.array([grip_ac, *eef_vel, 0.0, 0.0, 0.0])

            should_record = (
                t % getattr(args, "cam_rec_interval", 2) == 0
                if getattr(args, "cam_rec_interval", 2) > 0
                else (step_t == len(step_actions) - 1 or (step_idx == 0 and step_t == 0))
            )
            if should_record:
                render_dict = env.render(
                    cam_config=env.default_front_camera,
                    return_depth=True, return_pc=True, return_seg=True, resolution=240,
                )
                pc = render_dict["pc"]
                img = render_dict["images"][0][..., :3] if render_dict["images"] else np.zeros((240, 240, 3))
                if pc is None or len(pc) == 0:
                    pc = np.zeros((4096, 3), dtype=np.float32)
                    pc[0] = obs[0, :3]
                if len(pc) > 0 and (np.min(pc[:, 2]) < -0.05 or np.max(pc[:, 2]) > 1.0):
                    sim_unstable = True
                    break
                num_points = 4096
                if len(pc) >= num_points:
                    idx = np.random.choice(len(pc), size=num_points, replace=False)
                    pc = pc[idx]
                elif len(pc) > 0:
                    # Pad with repeated points if fewer than num_points
                    idx = np.random.choice(len(pc), size=num_points, replace=True)
                    pc = pc[idx]
                img_name = f"{prefix}_ep{counter:06d}_view0_t{record_t:02d}"
                save_path = os.path.join(args.data_out_dir, "pcs", f"{img_name}.npz")
                np.savez(save_path, pc=pc, rgb=img, action=action, eef_pos=obs)
                saved_files.append(save_path)
                dual_frame = env.render_dual(resolution=240)
                imgs.append(dual_frame)
                record_t += 1

            obs, _, _, _ = env.step(action, dummy_reward=True)
            t += 1

    final_rew = env.compute_reward()
    if args.task_name == "peg_insert":
        peg_pos = _to_numpy(env._peg_entity.get_pos())
        print(f"Reward: {final_rew:.3f} | peg: [{peg_pos[0]:.3f}, {peg_pos[1]:.3f}, {peg_pos[2]:.3f}]")
    elif args.task_name == "cup_pour":
        cup_pos = _to_numpy(env._cup_entity.get_pos())
        ball_pos = _to_numpy(env._ball_entity.get_pos())
        ball_xy = np.linalg.norm(ball_pos[:2] - env.BOWL_POS[:2])
        print(f"Reward: {final_rew:.3f} | cup: [{cup_pos[0]:.3f}, {cup_pos[1]:.3f}, {cup_pos[2]:.3f}] | ball-bowl: {ball_xy:.3f}")
    elif args.task_name == "book_insert":
        book_pos = _to_numpy(env._book_entity.get_pos())
        print(f"Reward: {final_rew:.3f} | book: [{book_pos[0]:.3f}, {book_pos[1]:.3f}, {book_pos[2]:.3f}] | target: [{env._target_pos[0]:.2f}, {env._target_pos[1]:.2f}, {env._target_pos[2]:.2f}]")
    else:
        print(f"Reward: {final_rew:.3f}")

    env.close()

    data_rew_threshold = getattr(args, "data_rew_threshold", 0.9)
    if final_rew >= data_rew_threshold:
        video_path = os.path.join(args.data_out_dir, "images", episode_name + ".mp4")
        if len(imgs) > 0:
            save_video(np.array(imgs), video_path, fps=10)
            print(f"Saved video to {video_path}")
        return 1
    else:
        for f in saved_files:
            if os.path.exists(f):
                os.remove(f)
        return 0


def get_next_episode_index(data_out_dir):
    images_dir = os.path.join(data_out_dir, "images")
    if not os.path.isdir(images_dir):
        return 0
    pattern = re.compile(r"_ep(\d{6})\.mp4$")
    max_idx = -1
    for name in os.listdir(images_dir):
        match = pattern.search(name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def get_args():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    parser = argparse.ArgumentParser(description="Genesis Franka PEFM Demo Generation")
    parser.add_argument("--task_name", type=str, default="peg_insert", choices=["peg_insert", "cup_pour", "book_insert"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_env", type=int, default=None)
    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--num_eef", type=int, default=1)
    parser.add_argument("--max_episode_length", type=int, default=80)
    parser.add_argument("--ac_noise", type=float, default=0)
    parser.add_argument("--freq", type=int, default=5)
    parser.add_argument("--randomize_rotation", action="store_true")
    parser.add_argument("--randomize_scale", action="store_true")
    parser.add_argument("--uniform_scaling", action="store_true")
    parser.add_argument("--cam_resolution", type=int, default=240)
    parser.add_argument("--cam_rec_interval", type=int, default=2)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--num_demos", type=int, default=1)
    parser.add_argument("--data_out_dir", type=str, default=None)
    parser.add_argument("--data_rew_threshold", type=float, default=0.9)
    parser.add_argument("--speed_multiplier", type=float, default=0.5)
    args, _ = parser.parse_known_args()
    if args.data_out_dir is None:
        args.data_out_dir = os.path.join("data", args.task_name)
    if args.seed_env is None:
        args.seed_env = args.seed
    return args


def main():
    args = get_args()
    seed = args.seed
    seed_env = args.seed_env
    start_idx = get_next_episode_index(args.data_out_dir)
    num_success = 0
    for i in range(args.num_demos * 10):
        if num_success >= args.num_demos:
            break
        args.seed = (seed * 99999 + i) % 100001
        args.seed_env = (seed_env * 99999 + i) % 100001
        success = run_demo(args, start_idx + num_success)
        if success:
            num_success += 1
            print(f"[{num_success}/{args.num_demos}] demos completed")
    print(f"Done. Generated {num_success}/{args.num_demos} demos.")


if __name__ == "__main__":
    main()
