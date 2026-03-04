"""
Genesis-backed base environment for single Franka Panda tabletop tasks.

Uses contact-based grasping (collisions ON, no magnetization).
Grasp detected by proximity + finger gap (no velocity alignment needed).
"""

import os
import time
import gym
import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None

from pefm_envs.sim_mobile.utils.transformations import (
    quat_multiply,
    axisangle2quat,
    quat2mat,
)

from .genesis_robot import (
    GenesisRobot,
    GRIPPER_DOWN_QUAT,
    FINGER_OPEN,
    FINGER_CLOSED,
    _ensure_genesis,
    _to_numpy,
    genesis_quat_to_scipy,
)


def _ensure_genesis_module():
    if gs is None:
        raise ImportError(
            "Genesis is required for sim_genesis. Install with: pip install genesis-world"
        )


def _cam_pos_from_config(cfg):
    """Convert camera config dict to Genesis (pos, lookat) tuple."""
    target = np.array(cfg["target"])
    dist = cfg["distance"]
    pitch = np.deg2rad(cfg["pitch"])
    yaw = np.deg2rad(cfg["yaw"])
    # Spherical to Cartesian offset from target
    dx = dist * np.cos(pitch) * np.cos(yaw)
    dy = dist * np.cos(pitch) * np.sin(yaw)
    dz = -dist * np.sin(pitch)
    pos = target + np.array([dx, dy, dz])
    return tuple(pos), tuple(target)


def _depth_to_pointcloud(depth, fov_deg, resolution):
    """Unproject depth image to point cloud in camera frame, then to world frame.

    Returns (N, 3) point cloud of valid (non-inf, non-zero) pixels.
    """
    h, w = depth.shape[:2]
    fov = np.deg2rad(fov_deg)
    focal = 0.5 * w / np.tan(fov / 2.0)

    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    # Camera-frame 3D points (OpenGL convention: -Z forward)
    z = depth
    valid = (z > 0) & (z < 10.0) & np.isfinite(z)
    x = (uu - w / 2.0) * z / focal
    y = (vv - h / 2.0) * z / focal

    pts = np.stack([x[valid], -y[valid], -z[valid]], axis=-1)
    return pts.astype(np.float32)


class GenesisFrankaEnv:
    """
    Base environment for a single Franka Panda on a tabletop in Genesis.
    Collisions are ON; grasping is contact-based (no weld/magnetization).
    """

    SIM_FREQ = 360
    SIM_GRAVITY = -9.8
    CONTROL_FREQ = 5
    TABLE_HEIGHT = 0.0
    ARM_BASE_POS = np.array([0.0, 0.0, 0.0])

    # Contact-based grasp: force applied when gripper command is "close"
    GRASP_FORCE = 0.5  # N per finger — matches Genesis tutorial
    RELEASE_FORCE = 0.0
    # Grasp detection thresholds
    GRASP_DIST_THRESH = 0.06
    GRASP_FINGER_GAP_THRESH = 0.07  # fingers < this → object likely between them

    # Franka home position (9 DOFs: 7 arm + 2 fingers)
    FRANKA_HOME_QPOS = np.array(
        [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
        dtype=np.float64,
    )

    def __init__(self, args, rng=None):
        _ensure_genesis_module()
        self.args = args
        self.num_eef = 1
        self.dof = getattr(args, "dof", 7)
        self.max_episode_length = args.max_episode_length
        self.rng = rng if rng is not None else np.random.RandomState(getattr(args, "seed", 0))
        self.debug = getattr(args, "debug", False)
        self.vis = getattr(args, "vis", False)
        self.freq = getattr(args, "freq", self.CONTROL_FREQ)

        self.randomize_rotation = getattr(args, "randomize_rotation", False)
        self.randomize_scale = getattr(args, "randomize_scale", False)
        self.uniform_scaling = getattr(args, "uniform_scaling", False)
        self.ac_noise = getattr(args, "ac_noise", 0.0)
        self.demo_mode = getattr(args, "demo_mode", False)

        self._object_rotation = np.array([0.0, 0.0, 0.0])
        self._rigid_object_scale = np.array([1.0, 1.0, 1.0])
        self.scene_offset = np.array([0.0, 0.0])

        self._grasped_obj_id = None
        self._gripper_target_closed = False
        self._close_cmd_steps = 0
        self._last_action_time = None

        # Task objects: list of entity indices and which are graspable
        self.rigid_ids = []
        self._rigid_graspable = []
        self._rigid_entities = []

        # Scene state tracking
        self._scene_built = False
        self._front_cam = None
        self._side_cam = None

        self._init_genesis()

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 7), dtype=np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 13), dtype=np.float32)

    @property
    def name(self):
        return "genesis_franka_base"

    @property
    def spawn_angle_range(self):
        return (0.0, 2 * np.pi)

    def _create_task_objects(self):
        """Override in subclasses to add task bodies."""
        pass

    def _reset_task_objects(self):
        """Override in subclasses to reposition task objects on reset.

        Called on subsequent resets (scene already built).
        Default: delegates to _create_task_objects (rebuild scene).
        """
        return False  # signal: must rebuild scene

    def compute_reward(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Camera config (for render)
    # ------------------------------------------------------------------ #

    @property
    def default_front_camera(self):
        return {
            "pitch": -45,
            "yaw": 0,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.4, 0.0, 0.1],
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -30,
            "yaw": 90,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.4, 0.0, 0.1],
        }

    # ------------------------------------------------------------------ #
    #  Genesis init and scene
    # ------------------------------------------------------------------ #

    def _init_genesis(self):
        """Initialize Genesis backend (once per process)."""
        if not hasattr(gs, "_pefm_initialized") or not gs._pefm_initialized:
            try:
                gs.init(backend=gs.gpu)
            except Exception:
                gs.init(backend=gs.cpu)
            gs._pefm_initialized = True

    def _create_scene(self):
        """Create a fresh Genesis scene (call before adding entities)."""
        # Genesis allows only one interactive viewer; show it only for the first scene.
        had_previous = hasattr(self, "scene") and self.scene is not None
        if had_previous:
            try:
                del self.scene
            except Exception:
                pass
            self.scene = None

        viewer_options = gs.options.ViewerOptions(
            camera_pos=(0, -2.0, 1.5),
            camera_lookat=(0.4, 0.0, 0.2),
            camera_fov=40,
            max_FPS=60,
        ) if self.vis else None

        show_viewer = self.vis and not had_previous
        self.scene = gs.Scene(
            viewer_options=viewer_options,
            sim_options=gs.options.SimOptions(dt=1.0 / self.SIM_FREQ),
            show_viewer=show_viewer,
        )

    def _setup_sim(self):
        """Build scene: plane, Franka, cameras, task objects."""
        self._create_scene()
        scene = self.scene

        # Ground plane
        scene.add_entity(gs.morphs.Plane())

        # Franka Panda
        panda_path = "xml/franka_emika_panda/panda.xml"
        if not os.path.isabs(panda_path):
            try:
                import genesis as _gs
                gdir = os.path.dirname(_gs.__file__)
                candidate = os.path.join(gdir, panda_path)
                if os.path.exists(candidate):
                    panda_path = candidate
            except Exception:
                pass
        self._franka_entity = scene.add_entity(gs.morphs.MJCF(file=panda_path))

        # Task objects (subclass adds them)
        self.rigid_ids = []
        self._rigid_graspable = []
        self._rigid_entities = []
        self._create_task_objects()

        # Cameras (must be added before build)
        front_pos, front_lookat = _cam_pos_from_config(self.default_front_camera)
        side_pos, side_lookat = _cam_pos_from_config(self.default_side_camera)
        self._front_cam = scene.add_camera(
            res=(240, 240),
            pos=front_pos,
            lookat=front_lookat,
            fov=self.default_front_camera["fov"],
            GUI=False,
        )
        self._side_cam = scene.add_camera(
            res=(240, 240),
            pos=side_pos,
            lookat=side_lookat,
            fov=self.default_side_camera["fov"],
            GUI=False,
        )

        scene.build()
        self._scene_built = True

        # Robot wrapper (after build)
        self.robot = GenesisRobot(self.scene, self._franka_entity, ee_link_name="hand")
        self.robot.detect_quat_convention()

        # Settle physics
        for _ in range(50):
            self.robot.step_scene()

        # Move to rest pose
        self._move_to_rest_pose()

    def _move_to_rest_pose(self):
        """IK to rest position and settle."""
        rest_pos = np.array([0.4, 0.0, 0.3], dtype=np.float64)
        rest_quat = GRIPPER_DOWN_QUAT.copy()
        init_qpos = self.robot.ee_pos_to_qpos(rest_pos, rest_quat, fing_dist=2 * FINGER_OPEN)
        if init_qpos is not None:
            self.robot.control_dofs_position(init_qpos)
            for _ in range(100):
                self.robot.step_scene()

    def _randomize_object_scales(self):
        if self.randomize_scale:
            scale_low = getattr(self.args, "scale_low", 1.0)
            scale_high = getattr(self.args, "scale_high", 1.0)
            self._rigid_object_scale = (
                self.rng.rand(3) * (scale_high - scale_low) + scale_low
            )
            if self.uniform_scaling:
                self._rigid_object_scale[:] = self._rigid_object_scale[0]
        else:
            self._rigid_object_scale = np.array([1.0, 1.0, 1.0])
        if self.randomize_rotation:
            lo, hi = self.spawn_angle_range
            ang = self.rng.rand() * (hi - lo) + lo
            self._object_rotation = np.array([0.0, 0.0, ang])
        else:
            self._object_rotation = np.array([0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ #
    #  Reset / Step
    # ------------------------------------------------------------------ #

    def reset(self):
        self._t = 0
        self._internal_t = 0
        self._frames = []
        self._episode_reward = 0.0
        self._ac_noise_multiplier = self.rng.rand()
        self._grasped_obj_id = None
        self._gripper_target_closed = False
        self._close_cmd_steps = 0
        self._randomize_object_scales()

        # For demo generation each env is used once, so always rebuild.
        # For training eval with repeated resets, subclasses can override
        # _reset_task_objects() to reposition entities without rebuilding.
        can_reuse = self._scene_built and self._reset_task_objects()
        if can_reuse:
            # Reset arm to home
            self.robot.franka.set_dofs_position(self.FRANKA_HOME_QPOS)
            for _ in range(50):
                self.robot.step_scene()
            self._move_to_rest_pose()
        else:
            self._setup_sim()

        return self._get_obs()

    def step(self, action, dummy_reward=False):
        action = np.array(action).reshape(1, self.dof)
        if self.ac_noise > 0:
            action[0, 1:4] += (
                self.rng.randn(3) * self.ac_noise * self._ac_noise_multiplier
            )

        grip_action = action[0, 0]
        if grip_action >= 0.9:
            self._gripper_target_closed = True
        elif grip_action <= 0.1:
            self._gripper_target_closed = False

        if self._gripper_target_closed:
            self._close_cmd_steps += 1
            finger_force = -self.GRASP_FORCE
        else:
            self._close_cmd_steps = 0
            finger_force = self.RELEASE_FORCE
            self._grasped_obj_id = None

        # Arm: EE velocity -> target pose -> IK -> position control
        pos_vel = action[0, 1:4]
        ori_vel = action[0, 4:7] if self.dof >= 7 else np.zeros(3)
        dt = 1.0 / self.freq
        ee_pos, ee_quat, _, _ = self.robot.get_ee_pos_quat_vel()
        target_pos = ee_pos + pos_vel * dt
        target_pos[2] = max(target_pos[2], 0.005)
        if np.linalg.norm(ori_vel) > 1e-6:
            ori_delta = ori_vel * dt
            delta_quat = axisangle2quat(ori_delta)
            target_quat = quat_multiply(delta_quat, ee_quat)
        else:
            target_quat = ee_quat
        fing_dist = 0.0 if self._gripper_target_closed else self.robot.get_max_fing_dist()
        target_qpos = self.robot.ee_pos_to_qpos(target_pos, target_quat, fing_dist=fing_dist)

        steps_per_action = self.SIM_FREQ // self.freq
        if target_qpos is not None:
            curr_qpos = self.robot.get_qpos()
            for st in range(steps_per_action):
                alpha = (st + 1) / steps_per_action
                interp_qpos = curr_qpos + alpha * (target_qpos - curr_qpos)
                self.robot.control_dofs_position(
                    interp_qpos[self.robot.motors_dof], self.robot.motors_dof
                )
                self.robot.control_dofs_force(
                    [finger_force, finger_force], self.robot.fingers_dof
                )
                self.robot.step_scene()
                self._internal_t += 1
        else:
            # IK failed — hold current arm pose, still apply finger force
            for _ in range(steps_per_action):
                self.robot.control_dofs_force(
                    [finger_force, finger_force], self.robot.fingers_dof
                )
                self.robot.step_scene()
                self._internal_t += 1

        self._update_grasp_state()

        if self.vis and self._last_action_time is not None:
            elapsed = time.time() - self._last_action_time
            if elapsed < 1.0 / self.freq:
                time.sleep(1.0 / self.freq - elapsed)
        self._last_action_time = time.time()

        obs = self._get_obs()
        self._t += 1
        rew = 0.0 if dummy_reward else self.compute_reward()
        done = self._t >= self.max_episode_length
        return obs, rew, done, {}

    def _update_grasp_state(self):
        """Detect grasp: gripper closed + object nearby + fingers closed on object."""
        if self._grasped_obj_id is not None:
            return
        if not self._gripper_target_closed or self._close_cmd_steps < 3:
            return
        ee_pos, _, _, _ = self.robot.get_ee_pos_quat_vel()
        fing_dist = self.robot.get_fing_dist()
        for i, (ent, graspable) in enumerate(zip(self._rigid_entities, self._rigid_graspable)):
            if not graspable:
                continue
            obj_pos = _to_numpy(ent.get_pos())
            dist = np.linalg.norm(obj_pos[:3] - ee_pos[:3])
            if dist > self.GRASP_DIST_THRESH:
                continue
            # Fingers closed enough → object between them
            if fing_dist < self.GRASP_FINGER_GAP_THRESH:
                self._grasped_obj_id = self.rigid_ids[i]
                return

    def _get_obs(self):
        """Return (1, 13): eef_xyz, x_dir, z_dir, gravity, grip."""
        ee_pos, ee_quat, _, _ = self.robot.get_ee_pos_quat_vel()
        rot_mat = quat2mat(ee_quat)
        x_dir = rot_mat[:, 0]
        z_dir = rot_mat[:, 2]
        gravity_dir = np.array([0.0, 0.0, -1.0])
        grip = 1.0 if self._grasped_obj_id is not None else 0.0
        obs = np.concatenate([ee_pos[:3], x_dir, z_dir, gravity_dir, [grip]])
        return obs.reshape(1, 13).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #

    def _render_cam(self, cam, cam_config, return_depth=True, return_pc=True, return_seg=False, resolution=240):
        """Render from a Genesis camera, return dict with images/depths/segs/pc."""
        if cam is None:
            return {
                "images": [np.zeros((resolution, resolution, 4), dtype=np.uint8)],
                "depths": [np.zeros((resolution, resolution), dtype=np.float32)],
                "segs": [np.zeros((resolution, resolution), dtype=np.int32)],
                "pc": np.zeros((0, 3), dtype=np.float32),
            }

        render_kwargs = {"rgb": True}
        if return_depth or return_pc:
            render_kwargs["depth"] = True
        if return_seg:
            render_kwargs["segmentation"] = True

        results = cam.render(**render_kwargs)

        # Unpack render results (Genesis returns tuple based on requested modes)
        if isinstance(results, tuple):
            idx = 0
            rgb = results[idx]; idx += 1
            depth = results[idx] if (return_depth or return_pc) else None; idx += (1 if (return_depth or return_pc) else 0)
            seg = results[idx] if return_seg else None
        else:
            rgb = results
            depth = None
            seg = None

        # Convert to numpy
        rgb_np = _to_numpy(rgb, dtype=np.uint8) if rgb is not None else np.zeros((resolution, resolution, 3), dtype=np.uint8)
        if rgb_np.ndim == 1:
            rgb_np = rgb_np.reshape(resolution, resolution, -1)
        if hasattr(rgb, "cpu"):
            rgb_np = rgb.cpu().numpy().astype(np.uint8)
            if rgb_np.ndim == 4:
                rgb_np = rgb_np[0]  # remove batch dim

        depth_np = None
        if depth is not None:
            if hasattr(depth, "cpu"):
                depth_np = depth.cpu().numpy().astype(np.float32)
                if depth_np.ndim == 3:
                    depth_np = depth_np[0]  # remove batch dim
            else:
                depth_np = np.array(depth, dtype=np.float32)

        seg_np = None
        if seg is not None:
            if hasattr(seg, "cpu"):
                seg_np = seg.cpu().numpy().astype(np.int32)
                if seg_np.ndim == 3:
                    seg_np = seg_np[0]
            else:
                seg_np = np.array(seg, dtype=np.int32)

        # Build RGBA image
        if rgb_np.shape[-1] == 3:
            alpha = np.full((*rgb_np.shape[:2], 1), 255, dtype=np.uint8)
            rgba = np.concatenate([rgb_np, alpha], axis=-1)
        else:
            rgba = rgb_np

        # Point cloud from depth
        pc = np.zeros((0, 3), dtype=np.float32)
        if return_pc and depth_np is not None:
            fov = cam_config.get("fov", 45)
            pc = _depth_to_pointcloud(depth_np, fov, resolution)

        return {
            "images": [rgba],
            "depths": [depth_np if depth_np is not None else np.zeros((resolution, resolution), dtype=np.float32)],
            "segs": [seg_np if seg_np is not None else np.zeros((resolution, resolution), dtype=np.int32)],
            "pc": pc,
        }

    def render(self, cam_config=None, return_depth=True, return_pc=True, return_seg=False, resolution=240):
        """Render from front camera. Returns dict with images/depths/segs/pc."""
        if cam_config is None:
            cam_config = self.default_front_camera
        # Determine which Genesis camera to use
        cam = self._front_cam
        if cam_config == self.default_side_camera:
            cam = self._side_cam
        return self._render_cam(cam, cam_config, return_depth, return_pc, return_seg, resolution)

    def render_dual(self, resolution=240):
        """Two-view image side by side."""
        front = self.render(cam_config=self.default_front_camera, return_depth=False, return_pc=False, resolution=resolution)
        side = self.render(cam_config=self.default_side_camera, return_depth=False, return_pc=False, resolution=resolution)
        fimg = front["images"][0][..., :3]
        simg = side["images"][0][..., :3]
        return np.concatenate([fimg, simg], axis=1)

    def close(self):
        """Clean up Genesis scene."""
        if hasattr(self, "scene") and self.scene is not None:
            try:
                del self.scene
            except Exception:
                pass
            self.scene = None
        self._scene_built = False
        self._front_cam = None
        self._side_cam = None
