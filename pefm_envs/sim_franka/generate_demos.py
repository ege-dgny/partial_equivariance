"""
Demo generation for single Franka Panda tabletop environments.

Generates demonstrations with mixed-frame sketch handling:
- Object-relative phases are rotated with the object
- World-frame phases remain fixed regardless of object orientation

Records 2-camera views (front + side) stitched side-by-side.
"""

import os
import sys
import re
import logging
import argparse
import numpy as np

# Guard heavy imports so Genesis demo gen can import pure-math helpers
# without requiring PyBullet or OpenCV
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pybullet
except ImportError:
    pybullet = None

from pefm_envs.sim_mobile.utils.anchors import create_trajectory
from pefm_envs.sim_mobile.utils.init_utils import rotate_around_z

try:
    from pefm_envs.sim_mobile.utils.multi_camera import MultiCamera
except ImportError:
    MultiCamera = None

np.set_printoptions(precision=2, linewidth=150, threshold=10000, suppress=True)


# ------------------------------------------------------------------ #
#  Quaternion orientation utilities
# ------------------------------------------------------------------ #

def quat_error_axis_angle(q_current, q_target):
    """Axis-angle rotation vector from q_current to q_target.

    Quaternions in PyBullet [x, y, z, w] format.  Returns (3,) axis*angle
    that, applied to q_current, yields q_target.  Singularity-free
    (unlike Euler-angle differencing).
    """
    cx, cy, cz, cw = q_current
    tx, ty, tz, tw = q_target
    # q_error = q_target * conjugate(q_current)
    ix, iy, iz, iw = -cx, -cy, -cz, cw
    ew = tw * iw - tx * ix - ty * iy - tz * iz
    ex = tw * ix + tx * iw + ty * iz - tz * iy
    ey = tw * iy - tx * iz + ty * iw + tz * ix
    ez = tw * iz + tx * iy - ty * ix + tz * iw
    if ew < 0:
        ex, ey, ez, ew = -ex, -ey, -ez, -ew
    sin_half = np.sqrt(ex * ex + ey * ey + ez * ez)
    if sin_half < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, ew)
    return np.array([ex, ey, ez]) * (angle / sin_half)


# ------------------------------------------------------------------ #
#  Environment factory
# ------------------------------------------------------------------ #

def get_env_class(task_name):
    if task_name == "peg_insert":
        from .peg_insert_env import PegInsertEnv
        return PegInsertEnv
    elif task_name == "cup_pour":
        from .cup_pour_env import CupPourEnv
        return CupPourEnv
    elif task_name == "book_insert":
        from .book_insert_env import BookInsertEnv
        return BookInsertEnv
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


def plan_actions_with_orientation(init_pos, init_euler, waypoints, sim_freq,
                                   num_sec_per_unit=2.0, init_grip=0.0):
    """
    Convert 7-dim waypoints into action sequence with orientation control.

    This function handles waypoints that include both position and orientation,
    enabling EEF reorientation during manipulation (e.g., for cup upright task).

    Args:
        init_pos: (3,) initial EEF position [x, y, z]
        init_euler: (3,) initial EEF orientation [roll, pitch, yaw]
        waypoints: list of (grip, x, y, z, roll, pitch, yaw) tuples
        sim_freq: control frequency
        num_sec_per_unit: time per unit distance/rotation

    Returns:
        actions: (T, 7) array [grip, vx, vy, vz, drx, dry, drz]
    """
    curr_pos = np.array(init_pos).copy()
    curr_euler = np.array(init_euler).copy()
    actions = []
    prev_grip = init_grip

    for waypoint in waypoints:
        grip = waypoint[0]
        target_pos = np.array(waypoint[1:4])
        target_euler = np.array(waypoint[4:7])

        # Compute distance for both position and orientation
        pos_dist = np.linalg.norm(target_pos - curr_pos)
        # Wrap orientation difference to [-pi, pi]
        ori_diff = target_euler - curr_euler
        ori_diff = np.mod(ori_diff + np.pi, 2 * np.pi) - np.pi
        ori_dist = np.linalg.norm(ori_diff)

        # Scale by a factor to make orientation changes take reasonable time
        effective_dist = max(pos_dist, ori_dist * 0.5)
        num_steps = max(int(effective_dist * num_sec_per_unit * sim_freq), 1)

        for t in range(num_steps):
            alpha = (t + 1) / num_steps

            # Linear interpolation for position
            pos_t = curr_pos + alpha * (target_pos - curr_pos)

            # Linear interpolation for orientation (short path)
            euler_t = curr_euler + alpha * ori_diff

            # Compute position targets (will be converted to velocity in main loop)
            # Also compute orientation targets
            g = prev_grip if t < num_steps - 1 else grip

            # Store as absolute targets (conversion to velocity happens later)
            action = np.concatenate([[g], pos_t, euler_t])
            actions.append(action)

        prev_grip = grip
        curr_pos = target_pos.copy()
        curr_euler = target_euler.copy()

    return np.array(actions)


def is_7dim_sketch(sketch):
    """Check if sketch uses 7-dim waypoints (with orientation)."""
    if len(sketch) == 0:
        return False
    return len(sketch[0]) == 7


def split_and_rotate_sketch_7d(sketch, object_phases, ang, object_center=None):
    """
    Rotate only object-relative phases for 7-dim sketches.

    Similar to split_and_rotate_sketch but handles orientation components.
    Position (x, y) is rotated; orientation (roll, pitch, yaw) has yaw adjusted.

    Args:
        sketch: list of (grip, x, y, z, roll, pitch, yaw) tuples
        object_phases: set of phase indices that are object-relative
        ang: rotation angle around Z
        object_center: (2,) XY position of the object in world frame
    """
    if object_center is None:
        object_center = np.array([0.0, 0.0])

    cos_a, sin_a = np.cos(ang), np.sin(ang)
    result = []

    for i, step in enumerate(sketch):
        if i in object_phases:
            g, x, y, z, roll, pitch, yaw = step
            # Rotate position around Z
            x_rot = cos_a * x - sin_a * y
            y_rot = sin_a * x + cos_a * y
            # Add object center offset
            x_final = x_rot + object_center[0]
            y_final = y_rot + object_center[1]
            # Adjust yaw by rotation angle
            yaw_final = yaw + ang
            result.append((g, x_final, y_final, z, roll, pitch, yaw_final))
        else:
            result.append(step)

    return result


# ------------------------------------------------------------------ #
#  Video
# ------------------------------------------------------------------ #

def save_video(frames, path, fps=10):
    if len(frames) == 0:
        return
    if cv2 is None:
        print(f"Warning: cv2 not available, skipping video save to {path}")
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

    # Get the rotation angle for sketch transformation
    # For C4 tasks, we need both spawn position angle AND object C4 rotation
    if args.task_name == "peg_insert":
        ang = env._object_rotation[-1] + getattr(env, '_peg_spawn_rotation', 0.0)
    else:
        ang = env._object_rotation[-1]

    # Build sketch based on task
    if args.task_name == "peg_insert":
        peg_pos, _ = env.sim.getBasePositionAndOrientation(env._peg_id)
        px, py = peg_pos[0], peg_pos[1]
        peg_h = env.PEG_HEIGHT
        sx, sy = env.SOCKET_POS[0], env.SOCKET_POS[1]
        insert_z = env.PLATE_THICKNESS + peg_h * 2

        # Grasp the peg at its center. The peg geometry center is at peg_h/2
        # above the URDF origin (which is at table level + 0.001).
        # We use panda_grasptarget as EE, which has accurate IK.
        grasp_z = peg_h*1.2  # Higher grasp for panda_hand→fingertip offset
        safe_z = 0.30  # Safe height to avoid collisions during lateral moves

        # Use 7D waypoints so gripper yaw follows peg initialization during grasp.
        # Format: (grip, x, y, z, roll, pitch, yaw), with gripper-down base pose.
        sketch = [
            # Phase 0 (world-frame): first go to safe height near home X
            # This avoids collision during the initial approach
            (0, 0.35, 0.0, safe_z, np.pi, 0.0, 0.0),
            # Phase 1 (object-relative): approach above peg at safe height
            (0, 0.0, 0.0, safe_z, np.pi, 0.0, 0.0),
            # Phase 2 (object-relative): descend to just above peg
            (0, 0.0, 0.0, peg_h * 2.5, np.pi, 0.0, 0.0),
            # Phase 3 (object-relative): descend to grasp
            (1, 0.0, 0.0, grasp_z, np.pi, 0.0, 0.0),
            # Phase 4 (object-relative): lift
            (1, 0.0, 0.0, safe_z, np.pi, 0.0, 0.0),
            # Phase 5 (world-frame): move over socket
            (1, sx, sy, safe_z, np.pi, 0.0, 0.0),
            # Phase 6 (world-frame): insert into socket
            (1, sx, sy, insert_z, np.pi, 0.0, 0.0),
            (0, sx, sy, insert_z, np.pi, 0.0, 0.0),
            (0, sx, sy, safe_z, np.pi, 0.0, 0.0),
        ]
        object_phases = {1, 2, 3, 4}  # Phases that are object-relative
        sketch = split_and_rotate_sketch_7d(
            sketch, object_phases, ang,
            object_center=np.array([px, py]),
        )

    elif args.task_name == "cup_pour":
        # Cup pouring: side-grasp cup, carry to bowl edge, tilt to pour ball
        cup_pos, _ = env.sim.getBasePositionAndOrientation(env._cup_id)
        cx, cy = cup_pos[0], cup_pos[1]
        cup_h = env.CUP_HEIGHT
        bowl_x, bowl_y = env.BOWL_POS[0], env.BOWL_POS[1]
        bowl_r = env.BOWL_RADIUS
        bowl_h = env.BOWL_HEIGHT
        safe_z = 0.25
        tilt_angle = env.POUR_TILT_ANGLE

        # Side-grasp orientation: same as book_insert [-pi/2, -pi/2, 0].
        # EEF Z = [0,1,0] (approach from -Y toward +Y),
        # EEF Y = [1,0,0] (fingers close along X, horizontal),
        # EEF X = [0,0,1] (gripper body vertical, cup "up" axis).
        side_roll = -np.pi / 2
        side_pitch = -np.pi / 2
        approach_offset = 0.08
        grasp_z = cup_h * 0.5

        # Fixed pour position for debugging.
        # Bowl is at BOWL_POS = [0.4, 0.0, 0.0].
        # tilt_yaw=0 → local Z = [0,1,0], so finger midpoint (cup center)
        # is ~5.8cm in +Y from panda_hand.  Shift pour_y by -0.05 to
        # center the cup on the bowl.
        pour_x = bowl_x - bowl_r - 0.04
        pour_y = bowl_y - 0.05
        pour_z = bowl_h + cup_h
        tilt_yaw = 0.0

        # 7-dim waypoints: (grip, x, y, z, roll, pitch, yaw)
        sketch = [
            # Phase 0 (object-relative): safe height, offset from cup (-Y)
            (0, 0.0, -approach_offset, safe_z,
             side_roll, side_pitch, 0.0),
            # Phase 1 (object-relative): lower to grasp height, still offset
            (0, 0.0, -approach_offset, grasp_z,
             side_roll, side_pitch, 0.0),
            # Phase 2 (object-relative): slide to cup center, grip closes
            (1, 0.0, 0.0, grasp_z,
             side_roll, side_pitch, 0.0),
            # Phases 3-5: dwell for grasp constraint to fire
            (1, 0.0, 0.0, grasp_z,
             side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z,
             side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z,
             side_roll, side_pitch, 0.0),
            # Phase 6 (object-relative): lift
            (1, 0.0, 0.0, safe_z,
             side_roll, side_pitch, 0.0),
            # Phase 7 (world-frame): move to pour position (upright)
            (1, pour_x, pour_y, pour_z,
             side_roll, side_pitch, tilt_yaw),
            # Phase 8 (world-frame): tilt to pour
            (1, pour_x, pour_y, pour_z,
             side_roll, side_pitch + tilt_angle, tilt_yaw),
            # Phases 9-10: hold tilt (ball falls out)
            (1, pour_x, pour_y, pour_z,
             side_roll, side_pitch + tilt_angle, tilt_yaw),
            (1, pour_x, pour_y, pour_z,
             side_roll, side_pitch + tilt_angle, tilt_yaw),
            # Phase 11: return upright, keep holding cup
            (1, pour_x, pour_y, safe_z,
             side_roll, side_pitch, tilt_yaw),
        ]
        object_phases = {0, 1, 2, 3, 4, 5, 6}
        sketch = split_and_rotate_sketch_7d(
            sketch, object_phases, ang,
            object_center=np.array([cx, cy]),
        )

    elif args.task_name == "book_insert":
        # Side-grasp book, carry to 2nd shelf, lower into slot (cup_pour pattern)
        book_pos, _ = env.sim.getBasePositionAndOrientation(env._book_id)
        bx, by = book_pos[0], book_pos[1]
        bl = env.BOOK_LENGTH   # 0.15 — height when standing
        hx, hy = env.HOLDER_POS[0], env.HOLDER_POS[1]
        shelf2_z = env.SHELF2_TOP_Z   # top of 2nd shelf
        safe_z = 0.30

        # Side-grip orientation (same as cup_pour):
        # EEF Z = [0,1,0] (approach from -Y toward +Y),
        # EEF Y = [1,0,0] (fingers close along X),
        # EEF X = [0,0,1] (gripper body vertical).
        side_roll = -np.pi / 2
        side_pitch = -np.pi / 2
        grasp_z = bl * 0.5
        approach_offset = 0.08

        # Place on 2nd shelf: book center at shelf2_z + half book height
        place_z = shelf2_z + bl / 2 + 0.02   # slight margin above shelf for release
        # World-frame EEF offset: at yaw=0, book center is ~12cm in +Y from panda_hand
        place_y = hy - 0.12

        # 7-dim waypoints: (grip, x, y, z, roll, pitch, yaw)
        sketch = [
            # Phase 0 (object-relative): safe height, offset from center (-Y)
            (0, 0.0, -approach_offset, safe_z,
             side_roll, side_pitch, 0.0),
            # Phase 1 (object-relative): lower to grasp height
            (0, 0.0, -approach_offset, grasp_z,
             side_roll, side_pitch, 0.0),
            # Phase 2 (object-relative): slide to book center, grip closes
            (1, 0.0, 0.0, grasp_z,
             side_roll, side_pitch, 0.0),
            # Phases 3-5: dwell for grasp constraint
            (1, 0.0, 0.0, grasp_z,
             side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z,
             side_roll, side_pitch, 0.0),
            (1, 0.0, 0.0, grasp_z,
             side_roll, side_pitch, 0.0),
            # Phase 6 (object-relative): lift
            (1, 0.0, 0.0, safe_z,
             side_roll, side_pitch, 0.0),
            # Phase 7 (world-frame): move above 2nd shelf slot (offset for EEF chain)
            (1, hx, place_y, safe_z,
             side_roll, side_pitch, 0.0),
            # Phase 8 (world-frame): lower into slot on 2nd shelf
            (1, hx, place_y, place_z,
             side_roll, side_pitch, 0.0),
            # Phase 9 (world-frame): release
            (0, hx, place_y, place_z,
             side_roll, side_pitch, 0.0),
        ]
        object_phases = {0, 1, 2, 3, 4, 5, 6}
        sketch = split_and_rotate_sketch_7d(
            sketch, object_phases, ang,
            object_center=np.array([bx, by]),
        )

    else:
        raise ValueError(f"Unknown task: {args.task_name}")

    # Convert sketch to actions
    init_eef_pos = obs[0, :3]
    init_grip = 0.0
    sim_freq = env.freq
    num_sec_per_unit = 20.0 / args.speed_multiplier

    t = 0
    record_t = 0
    imgs = []
    sim_unstable = False

    # ---- Sketch-based execution ----
    # Check if this is a 7-dim sketch (with orientation)
    use_orientation = is_7dim_sketch(sketch)

    # Get initial EEF orientation if needed
    if use_orientation:
        ee_pos, ee_quat, _, _ = env.robot.get_ee_pos_quat_vel()
        init_eef_euler = np.array(pybullet.getEulerFromQuaternion(ee_quat))
    else:
        init_eef_euler = np.array([np.pi, 0.0, 0.0])  # Default: gripper pointing down

    curr_euler = init_eef_euler.copy()

    for step_idx, step_target in enumerate(sketch):
        if sim_unstable:
            break

        if use_orientation:
            # 7-dim waypoint: (grip, x, y, z, roll, pitch, yaw)
            grip = step_target[0]
            tx, ty, tz = step_target[1:4]
            target_euler = np.array(step_target[4:7])
            prev_grip = init_grip if step_idx == 0 else sketch[step_idx - 1][0]

            step_actions = plan_actions_with_orientation(
                obs[0, :3], curr_euler, [(grip, tx, ty, tz, *target_euler)],
                sim_freq, num_sec_per_unit=num_sec_per_unit, init_grip=prev_grip,
            )
            # Update current euler for next waypoint
            curr_euler = target_euler.copy()
        else:
            # 4-dim waypoint: (grip, x, y, z)
            grip, tx, ty, tz = step_target
            prev_grip = init_grip if step_idx == 0 else sketch[step_idx - 1][0]

            step_actions = plan_actions_from_sketch(
                obs[0], [(grip, tx, ty, tz)], prev_grip, sim_freq,
                num_sec_per_unit=num_sec_per_unit,
            )

        for step_t, action_raw in enumerate(step_actions):
            if sim_unstable:
                break

            # Convert absolute targets to velocities
            grip_ac = action_raw[0]
            target_pos = action_raw[1:4]
            eef_vel = (target_pos - obs[0, :3]) * env.freq
            eef_vel = np.clip(eef_vel, -1.0, 1.0)

            if use_orientation and len(action_raw) >= 7:
                ee_pos, ee_quat, _, _ = env.robot.get_ee_pos_quat_vel()
                target_euler = action_raw[4:7]
                target_quat = pybullet.getQuaternionFromEuler(
                    target_euler.tolist()
                )
                axis_angle_err = quat_error_axis_angle(ee_quat, target_quat)
                ori_vel = axis_angle_err * env.freq
                ori_vel = np.clip(ori_vel, -1.0, 1.0)
                action = np.array([grip_ac, *eef_vel, *ori_vel])
            else:
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
    if task == "peg_insert":
        peg_pos, peg_quat = env.sim.getBasePositionAndOrientation(env._peg_id)
        peg_euler = pybullet.getEulerFromQuaternion(peg_quat)
        print(f"Reward: {final_rew:.3f} | peg: [{peg_pos[0]:.3f}, {peg_pos[1]:.3f}, {peg_pos[2]:.3f}] | z_rot: {np.degrees(peg_euler[2]):.1f}deg")
    elif task == "cup_pour":
        cup_pos, _ = env.sim.getBasePositionAndOrientation(env._cup_id)
        ball_pos, _ = env.sim.getBasePositionAndOrientation(env._ball_id)
        ball_xy = np.linalg.norm(np.array(ball_pos[:2]) - env.BOWL_POS[:2])
        print(f"Reward: {final_rew:.3f} | cup: [{cup_pos[0]:.3f}, {cup_pos[1]:.3f}, {cup_pos[2]:.3f}] | ball: [{ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f}] | ball-bowl: {ball_xy:.3f}")
    elif task == "book_insert":
        book_pos, book_quat = env.sim.getBasePositionAndOrientation(env._book_id)
        rot_mat = np.array(env.sim.getMatrixFromQuaternion(book_quat)).reshape(3, 3)
        book_z = rot_mat[:, 2]
        vert = abs(np.dot(book_z, [0, 0, 1]))
        print(f"Reward: {final_rew:.3f} | book: [{book_pos[0]:.3f}, {book_pos[1]:.3f}, {book_pos[2]:.3f}] | vert: {vert:.2f} | target: [{env._target_pos[0]:.2f}, {env._target_pos[1]:.2f}, {env._target_pos[2]:.2f}]")
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


def get_next_episode_index(data_out_dir):
    """Return next episode index by scanning existing saved demos."""
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
        "--task_name", type=str, default="peg_insert",
        choices=["peg_insert", "cup_pour", "book_insert"],
    )
    # Seeds
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_env", type=int, default=0)
    parser.add_argument("--seed_cam", type=int, default=0)
    # Simulation
    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--num_eef", type=int, default=1)
    parser.add_argument("--max_episode_length", type=int, default=80)
    parser.add_argument("--ac_noise", type=float, default=0)
    parser.add_argument("--freq", type=int, default=5)
    # Object randomization
    parser.add_argument("--randomize_rotation", action="store_true")
    parser.add_argument("--randomize_scale", action="store_true")
    parser.add_argument("--uniform_scaling", action="store_true")
    # Camera
    parser.add_argument("--cam_resolution", type=int, default=240)
    parser.add_argument("--cam_rec_interval", type=int, default=2)
    parser.add_argument("--vis", action="store_true")
    # Data
    parser.add_argument("--num_demos", type=int, default=1)
    parser.add_argument("--data_out_dir", type=str, default=None)
    parser.add_argument("--data_rew_threshold", type=float, default=0.9)
    parser.add_argument("--speed_multiplier", type=float, default=0.5)

    args, _ = parser.parse_known_args()

    if args.data_out_dir is None:
        args.data_out_dir = os.path.join("data", args.task_name)

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
