"""
Sketch-based demo generation for robosuite tasks.

Reuses the sketch/rotate/plan pattern from sim_franka: build 7D waypoints
from object state, split into object-relative and world-frame phases,
rotate object phases by spawn angle, execute through env.step().

Usage:
    python -m pefm_envs.sim_robosuite.generate_demos \
        --task_name pick_place_fixed --num_demos 50 \
        --data_out_dir ../data/pick_place_fixed
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

# Reuse sketch utilities from sim_franka
from pefm_envs.sim_franka.generate_demos import (
    is_7dim_sketch,
    plan_actions_with_orientation,
    split_and_rotate_sketch_7d,
)

try:
    import cv2
except ImportError:
    cv2 = None


# ------------------------------------------------------------------ #
#  Task-specific sketch builders
# ------------------------------------------------------------------ #

def _build_pick_place_sketch(env, obs):
    """Build 7D sketch for PickPlaceFixed.

    Phases:
      0: Move above can (object-relative)
      1: Descend to can (object-relative)
      2: Close gripper (object-relative)
      3: Lift (object-relative)
      4: Move to bin (world-frame)  <-- symmetry breaks here
      5: Lower into bin (world-frame)
      6: Release (world-frame)
      7: Retract (world-frame)
    """
    can_pos = env.get_can_pos()
    bin_pos = env.get_bin_pos()

    cx, cy, cz = can_pos[0], can_pos[1], can_pos[2]
    bx, by, bz = bin_pos[0], bin_pos[1], bin_pos[2]

    safe_z = 0.95  # safe height above table
    grasp_z = cz + 0.02  # just above can center
    place_z = bz + 0.05  # above bin lip

    # Gripper pointing down: roll=pi, pitch=0, yaw=0
    down = (np.pi, 0.0, 0.0)

    sketch = [
        # Object-relative phases (offsets from can center)
        (0, 0.0, 0.0, safe_z, *down),        # 0: above can
        (0, 0.0, 0.0, grasp_z, *down),       # 1: descend to can
        (1, 0.0, 0.0, grasp_z, *down),       # 2: close grip
        (1, 0.0, 0.0, safe_z, *down),        # 3: lift
        # World-frame phases (absolute coords)
        (1, bx, by, safe_z, *down),           # 4: move to bin
        (1, bx, by, place_z, *down),          # 5: lower into bin
        (0, bx, by, place_z, *down),          # 6: release
        (0, bx, by, safe_z, *down),           # 7: retract
    ]
    object_phases = {0, 1, 2, 3}

    return sketch, object_phases, np.array([cx, cy])


def _build_nut_assembly_sketch(env, obs):
    """Build 7D sketch for NutAssemblyFixed (C4).

    Phases:
      0: Move above nut (object-relative)
      1: Descend to nut (object-relative)
      2: Close gripper (object-relative)
      3: Lift nut (object-relative)
      4: Move above peg (world-frame)  <-- symmetry breaks here
      5: Lower onto peg (world-frame)
      6: Release (world-frame)
      7: Retract (world-frame)
    """
    nut_pos = env.get_nut_pos()
    peg_pos = env.get_peg_pos()

    nx, ny, nz = nut_pos[0], nut_pos[1], nut_pos[2]
    px, py, pz = peg_pos[0], peg_pos[1], peg_pos[2]

    safe_z = 0.95
    grasp_z = nz + 0.02
    insert_z = pz + 0.05

    down = (np.pi, 0.0, 0.0)

    sketch = [
        # Object-relative phases
        (0, 0.0, 0.0, safe_z, *down),        # 0: above nut
        (0, 0.0, 0.0, grasp_z, *down),       # 1: descend
        (1, 0.0, 0.0, grasp_z, *down),       # 2: close grip
        (1, 0.0, 0.0, safe_z, *down),        # 3: lift
        # World-frame phases
        (1, px, py, safe_z, *down),           # 4: above peg
        (1, px, py, insert_z, *down),         # 5: lower onto peg
        (0, px, py, insert_z, *down),         # 6: release
        (0, px, py, safe_z, *down),           # 7: retract
    ]
    object_phases = {0, 1, 2, 3}

    return sketch, object_phases, np.array([nx, ny])


def _build_tool_hang_sketch(env, obs):
    """Build 7D sketch for ToolHang (multi-phase).

    This is a simplified 4-phase sketch. ToolHang is extremely hard to
    script — this provides a best-effort trajectory. Success rate may be
    low; the main value is producing training data with the correct
    symmetry structure (alternating object-relative/world-frame phases).

    Phase group 1: Grasp frame (object-relative, SO2)
    Phase group 2: Insert frame (world-frame)
    Phase group 3: Grasp tool (object-relative, SO2)
    Phase group 4: Hang tool (world-frame)
    """
    frame_pos = env.get_frame_pos()
    stand_pos = env.get_stand_pos()
    tool_pos = env.get_tool_pos()

    fx, fy, fz = frame_pos[0], frame_pos[1], frame_pos[2]
    sx, sy, sz = stand_pos[0], stand_pos[1], stand_pos[2]
    tx, ty, tz = tool_pos[0], tool_pos[1], tool_pos[2]

    safe_z = 1.1
    down = (np.pi, 0.0, 0.0)

    sketch = [
        # Phase group 1: grasp frame (object-relative)
        (0, 0.0, 0.0, safe_z, *down),         # 0: above frame
        (0, 0.0, 0.0, fz + 0.02, *down),      # 1: descend to frame
        (1, 0.0, 0.0, fz + 0.02, *down),      # 2: close grip
        (1, 0.0, 0.0, safe_z, *down),          # 3: lift frame
        # Phase group 2: insert frame (world-frame)
        (1, sx, sy, safe_z, *down),            # 4: above stand
        (1, sx, sy, sz + 0.05, *down),         # 5: lower into stand
        (0, sx, sy, sz + 0.05, *down),         # 6: release frame
        (0, sx, sy, safe_z, *down),            # 7: retract
        # Phase group 3: grasp tool (object-relative)
        (0, 0.0, 0.0, safe_z, *down),         # 8: above tool
        (0, 0.0, 0.0, tz + 0.02, *down),      # 9: descend to tool
        (1, 0.0, 0.0, tz + 0.02, *down),      # 10: close grip
        (1, 0.0, 0.0, safe_z, *down),          # 11: lift tool
        # Phase group 4: hang tool (world-frame)
        (1, sx, sy, safe_z, *down),            # 12: above stand
        (1, sx, sy, sz + 0.10, *down),         # 13: lower to hook
        (0, sx, sy, sz + 0.10, *down),         # 14: release tool
        (0, sx, sy, safe_z, *down),            # 15: retract
    ]
    # Phases 0-3 relative to frame, 8-11 relative to tool
    object_phases = {0, 1, 2, 3, 8, 9, 10, 11}

    return sketch, object_phases, np.array([fx, fy])


# ------------------------------------------------------------------ #
#  Main demo generation
# ------------------------------------------------------------------ #

def get_env_class(task_name: str):
    if task_name == "pick_place_fixed":
        from pefm_envs.sim_robosuite.pick_place_fixed import PickPlaceFixedEnv
        return PickPlaceFixedEnv
    elif task_name == "nut_assembly_fixed":
        from pefm_envs.sim_robosuite.nut_assembly_fixed import NutAssemblyFixedEnv
        return NutAssemblyFixedEnv
    elif task_name == "tool_hang":
        from pefm_envs.sim_robosuite.tool_hang import ToolHangEnv
        return ToolHangEnv
    else:
        raise ValueError(f"Unknown task: {task_name}")


def run_demo(args, counter: int) -> bool:
    """Run a single demo episode. Returns True if successful."""
    rng = np.random.RandomState(args.seed)
    EnvClass = get_env_class(args.task_name)
    env = EnvClass(args, rng)
    obs = env.reset()

    # Object rotation angle
    ang = env._object_rotation[-1]

    # Build task-specific sketch
    if args.task_name == "pick_place_fixed":
        sketch, object_phases, obj_center = _build_pick_place_sketch(env, obs)
    elif args.task_name == "nut_assembly_fixed":
        sketch, object_phases, obj_center = _build_nut_assembly_sketch(env, obs)
    elif args.task_name == "tool_hang":
        sketch, object_phases, obj_center = _build_tool_hang_sketch(env, obs)
        # ToolHang has 2 object groups; handle tool group separately
        # For now, rotate all object phases by frame angle
    else:
        env.close()
        return False

    # Rotate object-relative phases by object spawn angle
    sketch = split_and_rotate_sketch_7d(
        sketch, object_phases, ang, object_center=obj_center,
    )

    # Execute sketch
    prefix = os.path.basename(args.data_out_dir)
    saved_files = []
    imgs = []
    t = 0
    record_t = 0
    sim_unstable = False

    sim_freq = env.freq
    speed = getattr(args, "speed_multiplier", 1.0)
    num_sec_per_unit = 20.0 / speed

    # Get initial EEF state
    init_pos = obs[0, :3]
    init_euler = np.array([np.pi, 0.0, 0.0])  # gripper down default
    curr_euler = init_euler.copy()

    for step_idx, step_target in enumerate(sketch):
        if sim_unstable:
            break

        grip = step_target[0]
        tx, ty, tz = step_target[1:4]
        target_euler = np.array(step_target[4:7])
        prev_grip = 0.0 if step_idx == 0 else sketch[step_idx - 1][0]

        step_actions = plan_actions_with_orientation(
            obs[0, :3], curr_euler,
            [(grip, tx, ty, tz, *target_euler)],
            sim_freq, num_sec_per_unit=num_sec_per_unit,
            init_grip=prev_grip,
        )
        curr_euler = target_euler.copy()

        for step_t, action_raw in enumerate(step_actions):
            if sim_unstable:
                break

            # Convert absolute targets to velocities
            grip_ac = action_raw[0]
            target_pos = action_raw[1:4]
            eef_vel = (target_pos - obs[0, :3]) * env.freq
            eef_vel = np.clip(eef_vel, -1.0, 1.0)

            if len(action_raw) >= 7:
                target_ori = action_raw[4:7]
                # Simple proportional orientation control
                ori_err = target_ori - curr_euler
                ori_err = np.mod(ori_err + np.pi, 2 * np.pi) - np.pi
                ori_vel = ori_err * env.freq
                ori_vel = np.clip(ori_vel, -1.0, 1.0)
                action = np.array([grip_ac, *eef_vel, *ori_vel])
            else:
                action = np.array([grip_ac, *eef_vel, 0.0, 0.0, 0.0])

            # Record at subsampled rate
            cam_interval = getattr(args, "cam_rec_interval", 0)
            should_record = (
                (t % cam_interval == 0) if cam_interval > 0
                else (step_t == len(step_actions) - 1 or
                      (step_idx == 0 and step_t == 0))
            )

            if should_record:
                # Get point cloud
                pc = env._get_point_cloud()
                if len(pc) == 0:
                    sim_unstable = True
                    print("Warning: empty point cloud; cutting episode.")
                    break

                pc = env._subsample_pc(pc)
                img = env.render()["images"][0][..., :3]

                fn = f"{prefix}_ep{counter:06d}_view0_t{record_t:02d}.npz"
                save_path = os.path.join(args.data_out_dir, "pcs", fn)
                np.savez(save_path, pc=pc, rgb=img, action=action, eef_pos=obs)
                saved_files.append(save_path)

                # Dual-view frame for video
                dual = env.render_dual(resolution=240)
                imgs.append(dual)
                record_t += 1

            obs, _, done, _ = env.step(action, dummy_reward=True)
            t += 1

            if done:
                break

    # Evaluate
    final_rew = env.compute_reward()
    print(f"  Reward: {final_rew:.3f} | steps: {t} | frames: {record_t}")

    env.close()

    # Only keep successful demos
    success_thresh = getattr(args, "reward_thresh", 0.5)
    if final_rew < success_thresh:
        # Clean up saved files
        for f in saved_files:
            try:
                os.remove(f)
            except OSError:
                pass
        return False

    # Save video
    if imgs and cv2 is not None:
        vid_dir = os.path.join(args.data_out_dir, "videos")
        os.makedirs(vid_dir, exist_ok=True)
        vid_path = os.path.join(vid_dir, f"ep{counter:06d}.mp4")
        h, w = imgs[0].shape[:2]
        writer = cv2.VideoWriter(
            vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h),
        )
        for frame in imgs:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    return True


def get_next_episode_index(data_dir: str) -> int:
    """Find next available episode index."""
    pcs_dir = os.path.join(data_dir, "pcs")
    if not os.path.exists(pcs_dir):
        return 0
    existing = [f for f in os.listdir(pcs_dir) if f.endswith(".npz")]
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            parts = f.split("_ep")[1].split("_")[0]
            indices.append(int(parts))
        except (IndexError, ValueError):
            pass
    return max(indices) + 1 if indices else 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["pick_place_fixed", "nut_assembly_fixed", "tool_hang"])
    parser.add_argument("--num_demos", type=int, default=50)
    parser.add_argument("--data_out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_env", type=int, default=None)
    parser.add_argument("--max_episode_length", type=int, default=200)
    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--freq", type=int, default=20)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--speed_multiplier", type=float, default=1.0)
    parser.add_argument("--cam_rec_interval", type=int, default=0)
    parser.add_argument("--reward_thresh", type=float, default=0.5)
    parser.add_argument("--randomize_rotation", action="store_true", default=True)
    parser.add_argument("--ac_noise", type=float, default=0.0)
    args = parser.parse_args()

    if args.data_out_dir is None:
        args.data_out_dir = os.path.join("data", args.task_name)
    if args.seed_env is None:
        args.seed_env = args.seed
    if args.task_name == "tool_hang":
        args.max_episode_length = max(args.max_episode_length, 700)

    return args


def main():
    args = get_args()
    os.makedirs(os.path.join(args.data_out_dir, "pcs"), exist_ok=True)

    seed = args.seed
    start_idx = get_next_episode_index(args.data_out_dir)
    num_success = 0

    for i in range(args.num_demos * 20):
        if num_success >= args.num_demos:
            break
        args.seed = (seed * 99999 + i) % 100001
        success = run_demo(args, start_idx + num_success)
        if success:
            num_success += 1
            print(f"[{num_success}/{args.num_demos}] demos completed")

    print(f"Done. Generated {num_success}/{args.num_demos} demos.")


if __name__ == "__main__":
    main()
