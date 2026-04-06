"""
Base wrapper: robosuite env -> PEFM observation/action format.

Converts robosuite's MuJoCo-based environments to match the interface
used by sim_franka and sim_genesis backends.

Action:  (1, 7) = [gripper, vx, vy, vz, drx, dry, drz]
Obs:     (1, 13) = [eef_xyz(3), x_dir(3), z_dir(3), gravity(3), grip(1)]

Key convention mappings handled here:
  - Grip: PEFM 0/1 (threshold 0.9/0.1) <-> robosuite -1/+1
  - Motion: PEFM velocities (m/s) -> robosuite delta pose (vel * dt)
  - Gripper index: PEFM first [0] <-> robosuite last [6]
"""

from __future__ import annotations

import numpy as np

try:
    import gym
except ImportError:
    import gymnasium as gym

from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _spherical_to_cartesian(
    pitch_deg: float,
    yaw_deg: float,
    distance: float,
    target: np.ndarray,
) -> np.ndarray:
    """Convert pitch/yaw/distance to camera world position (matching sim_franka)."""
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    x = distance * np.cos(pitch) * np.sin(yaw)
    y = -distance * np.cos(pitch) * np.cos(yaw)
    z = -distance * np.sin(pitch)
    return np.array(target) + np.array([x, y, z])


# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------

class RobosuiteBaseEnv:
    """Wraps a robosuite environment for PEFM compatibility.

    Subclasses must set:
      - robosuite_env_name (property): str
      - _modify_env_kwargs(): dict of extra kwargs for suite.make()
      - _get_object_body_ids(): list[int] for PC segmentation
    """

    # Matching sim_franka/sim_genesis camera configs
    FRONT_CAMERA = {
        "pitch": -45, "yaw": 0, "roll": 0,
        "distance": 1.2, "fov": 45,
        "target": [0.4, 0.0, 0.1],
    }
    SIDE_CAMERA = {
        "pitch": -30, "yaw": 90, "roll": 0,
        "distance": 1.2, "fov": 45,
        "target": [0.4, 0.0, 0.1],
    }

    RENDER_RES = 240
    PC_NUM_POINTS = 4096

    # Grip thresholds matching sim_franka
    GRIP_CLOSE_THRESH = 0.9
    GRIP_OPEN_THRESH = 0.1

    def __init__(self, args, rng=None):
        self.args = args
        self.num_eef = 1
        self.dof = getattr(args, "dof", 7)
        self.max_episode_length = args.max_episode_length
        self.rng = rng or np.random.RandomState(getattr(args, "seed", 0))
        self.vis = getattr(args, "vis", False)
        self.freq = getattr(args, "freq", 20)
        self.ac_noise = getattr(args, "ac_noise", 0)

        self.randomize_rotation = getattr(args, "randomize_rotation", True)
        self._object_rotation = np.array([0.0, 0.0, 0.0])

        self._gripper_target_closed = False
        self._grasped = False
        self._t = 0
        self._frames = []
        self._ac_noise_multiplier = 1.0

        # Build robosuite env
        self._build_env()

        # Cached observation from last step (for render reuse)
        self._cached_obs_dict = None

    def _build_env(self):
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        # OSC_POSE: 7D action [dx, dy, dz, dax, day, daz, grip]
        controller_config = load_composite_controller_config(
            controller="BASIC",
        )

        env_kwargs = dict(
            env_name=self.robosuite_env_name,
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=self.vis,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            reward_shaping=True,
            control_freq=self.freq,
            horizon=self.max_episode_length * 10,  # oversized; our step() handles termination
            camera_names=["agentview"],
            camera_heights=[self.RENDER_RES],
            camera_widths=[self.RENDER_RES],
            camera_depths=[True],
            camera_segmentations=["element"],
        )
        env_kwargs.update(self._modify_env_kwargs())
        self.env = suite.make(**env_kwargs)

    # ---- Override in subclasses ----

    @property
    def robosuite_env_name(self) -> str:
        raise NotImplementedError

    def _modify_env_kwargs(self) -> dict:
        """Extra kwargs for suite.make()."""
        return {}

    def _get_object_body_ids(self) -> list[int]:
        """MuJoCo body IDs for task objects (for PC segmentation)."""
        model = self.env.sim.model
        ids = []
        skip = {"robot", "gripper", "base", "world", "table", "floor", "mount"}
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name and not any(s in name.lower() for s in skip):
                ids.append(i)
        return ids

    def compute_reward(self) -> float:
        return float(self.env.reward())

    # ---- Spaces (for vec env compat) ----

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1, 7), dtype=np.float32,
        )

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, 13), dtype=np.float32,
        )

    # ---- Reset / Step ----

    def reset(self):
        self._t = 0
        self._gripper_target_closed = False
        self._grasped = False
        self._frames = []
        self._ac_noise_multiplier = self.rng.rand()

        self._cached_obs_dict = self.env.reset()
        return self._make_pefm_obs(self._cached_obs_dict)

    def step(self, action, dummy_reward=False):
        action = np.array(action, dtype=np.float64).reshape(self.dof)

        # Add action noise (on position velocities only)
        if self.ac_noise > 0:
            action[1:4] += (
                self.rng.randn(3) * self.ac_noise * self._ac_noise_multiplier
            )

        # --- Grip: PEFM 0/1 -> robosuite -1/+1 ---
        grip_pefm = action[0]
        if grip_pefm >= self.GRIP_CLOSE_THRESH:
            self._gripper_target_closed = True
        elif grip_pefm <= self.GRIP_OPEN_THRESH:
            self._gripper_target_closed = False

        grip_robo = 1.0 if self._gripper_target_closed else -1.0

        # --- Motion: velocity -> delta pose ---
        dt = 1.0 / self.freq
        pos_vel = action[1:4]
        ori_vel = action[4:7] if self.dof >= 7 else np.zeros(3)

        delta_pos = pos_vel * dt
        delta_ori = ori_vel * dt

        # Clip deltas to robosuite action range [-1, 1]
        delta_pos = np.clip(delta_pos, -1.0, 1.0)
        delta_ori = np.clip(delta_ori, -1.0, 1.0)

        # robosuite action: [dx, dy, dz, dax, day, daz, grip]
        robosuite_action = np.concatenate([delta_pos, delta_ori, [grip_robo]])

        self._cached_obs_dict, reward, done, info = self.env.step(robosuite_action)
        self._t += 1

        obs = self._make_pefm_obs(self._cached_obs_dict)

        if dummy_reward:
            reward = 0.0

        done = done or (self._t >= self.max_episode_length)

        return obs, reward, done, info

    # ---- Observation conversion ----

    def _make_pefm_obs(self, obs_dict: dict) -> np.ndarray:
        """Convert robosuite obs dict to PEFM (1, 13) state.

        PEFM expects: [eef_xyz(3), x_dir(3), z_dir(3), gravity(3), grip(1)]
        """
        eef_pos = obs_dict["robot0_eef_pos"]
        eef_quat = obs_dict["robot0_eef_quat"]  # xyzw (robosuite convention)

        rot_mat = Rotation.from_quat(eef_quat).as_matrix()
        x_dir = rot_mat[:, 0]
        z_dir = rot_mat[:, 2]

        gravity_dir = np.array([0.0, 0.0, -1.0])

        # Grip state: finger qpos < threshold means fingers are closed
        gripper_qpos = obs_dict.get("robot0_gripper_qpos", np.array([0.04, 0.04]))
        grip = 1.0 if np.mean(gripper_qpos) < 0.02 else 0.0

        state = np.concatenate([eef_pos, x_dir, z_dir, gravity_dir, [grip]])
        return state.reshape(1, 13).astype(np.float32)

    # ---- Point cloud extraction ----

    def _get_point_cloud(self, obs_dict: dict | None = None) -> np.ndarray:
        """Extract segmented point cloud from depth camera.

        Returns (N, 3) world-frame points for task objects only.
        """
        if obs_dict is None:
            obs_dict = self._cached_obs_dict
        if obs_dict is None:
            return np.zeros((0, 3), dtype=np.float32)

        # Try standard key names (robosuite 1.5.x)
        depth_key = "agentview_depth"
        seg_key = "agentview_segmentation_element"

        depth = obs_dict.get(depth_key)
        seg = obs_dict.get(seg_key)

        if depth is None or seg is None:
            # Fallback: try rendering directly
            return np.zeros((0, 3), dtype=np.float32)

        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if seg.ndim == 3:
            seg = seg[:, :, 0]

        H, W = depth.shape

        # Camera intrinsics from MuJoCo
        cam_id = self.env.sim.model.camera_name2id("agentview")
        fovy = self.env.sim.model.cam_fovy[cam_id]
        f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2.0)
        cx, cy = W / 2.0, H / 2.0

        # Camera extrinsics
        cam_pos = self.env.sim.data.cam_xpos[cam_id]
        cam_mat = self.env.sim.data.cam_xmat[cam_id].reshape(3, 3)

        # Filter by object body IDs
        object_body_ids = self._get_object_body_ids()
        valid_mask = (
            np.isin(seg, object_body_ids)
            & (depth > 0.01)
            & (depth < 5.0)
        )

        if not valid_mask.any():
            return np.zeros((0, 3), dtype=np.float32)

        v_grid, u_grid = np.where(valid_mask)
        z_vals = depth[v_grid, u_grid]

        # Unproject to camera frame
        x_cam = (u_grid - cx) * z_vals / f
        y_cam = (v_grid - cy) * z_vals / f

        # MuJoCo camera convention: -Z forward, X right, Y down
        pts_cam = np.stack([x_cam, y_cam, -z_vals], axis=-1)

        # Transform to world frame
        pts_world = (cam_mat.T @ pts_cam.T).T + cam_pos

        return pts_world.astype(np.float32)

    def _subsample_pc(self, pc: np.ndarray) -> np.ndarray:
        """Subsample or pad point cloud to PC_NUM_POINTS."""
        n = len(pc)
        if n == 0:
            return np.zeros((self.PC_NUM_POINTS, 3), dtype=np.float32)
        if n >= self.PC_NUM_POINTS:
            idx = self.rng.choice(n, self.PC_NUM_POINTS, replace=False)
        else:
            idx = self.rng.choice(n, self.PC_NUM_POINTS, replace=True)
        return pc[idx]

    # ---- Rendering ----

    def render(self, cam_config=None, return_depth=True,
               return_pc=True, **kwargs):
        """Render for PEFM eval compatibility."""
        img = self.env.sim.render(
            camera_name="agentview",
            width=self.RENDER_RES, height=self.RENDER_RES,
        )
        img = img[::-1]  # MuJoCo renders upside down

        result = {"images": [img]}
        if return_pc:
            result["pc"] = self._get_point_cloud()
        return result

    def render_dual(self, resolution: int = 240) -> np.ndarray:
        """Two-view image for video recording (front + side concatenated)."""
        front = self.env.sim.render(
            camera_name="agentview",
            width=resolution, height=resolution,
        )[::-1]

        # robosuite may or may not have sideview; fallback to frontview
        try:
            side = self.env.sim.render(
                camera_name="sideview",
                width=resolution, height=resolution,
            )[::-1]
        except Exception:
            side = self.env.sim.render(
                camera_name="agentview",
                width=resolution, height=resolution,
            )[::-1]

        return np.concatenate([front, side], axis=1)

    # ---- Misc compat ----

    def close(self):
        self.env.close()

    def get(self, attr):
        if attr == "args":
            return self.args
        return getattr(self, attr)

    @property
    def name(self) -> str:
        return self.robosuite_env_name.lower()
