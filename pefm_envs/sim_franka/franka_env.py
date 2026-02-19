"""
Single Franka Panda tabletop environment for PEFM.

Clean standalone base environment for a single 7-DOF Franka Panda arm
bolted to a table. Replaces the dual-Kinova mobile-base system with a
simpler fixed-arm architecture matching the lab setup.

Action:  (1, 7) = [gripper, vx, vy, vz, drx, dry, drz]
Obs:     (1, 13) = [eef_xyz(3), x_dir(3), z_dir(3), gravity(3), grip(1)]
"""

import os
import time
import gym
import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bclient
from scipy.spatial.transform import Rotation

from pefm_envs.sim_mobile.utils.bullet_robot import BulletRobot
from pefm_envs.sim_mobile.utils.info import SIM_ROBOT_INFO
from pefm_envs.sim_mobile.utils.multi_camera import MultiCamera
from pefm_envs.sim_mobile.utils.project import unproject_depth
from pefm_envs.sim_mobile.utils.anchors import get_closest
from pefm_envs.sim_mobile.utils.transformations import (
    quat_multiply,
    axisangle2quat,
    quat2mat,
)


class FrankaEnv:
    """
    Base environment for a single Franka Panda arm on a tabletop.

    Subclasses override:
      - _create_task_objects(): spawn task-specific objects
      - compute_reward(): task-specific reward
      - name: task identifier string
    """

    SIM_FREQ = 360
    SIM_GRAVITY = -9.8
    CONTROL_FREQ = 5  # actions per second
    TABLE_HEIGHT = 0.0  # objects sit on ground plane

    # Franka base position (arm mounted at table edge, facing +X)
    ARM_BASE_POS = np.array([0.0, 0.0, 0.0])

    def __init__(self, args, rng=None):
        self.args = args
        self.num_eef = 1
        self.dof = args.dof
        self.max_episode_length = args.max_episode_length
        self.rng = rng if rng is not None else np.random.RandomState(args.seed)
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

        # Constraint tracking for grasping
        self.constraint_id = None
        self._done = None
        self._last_action_time = None
        self._grasped_obj_id = None

        # Initialize simulation
        self._init_sim()

    # ------------------------------------------------------------------ #
    #  Spaces (for vectorized env compatibility)
    # ------------------------------------------------------------------ #

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1, 7), dtype=np.float32
        )

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1, 13), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    #  Properties to override in subclasses
    # ------------------------------------------------------------------ #

    @property
    def name(self):
        return "franka_base"

    @property
    def spawn_angle_range(self):
        """(lo, hi) range for random object rotation around Z.
        Override in subclass to limit to reachable workspace."""
        return (0.0, 2 * np.pi)

    def _create_task_objects(self):
        """Override to spawn task-specific objects. Must populate
        self.rigid_ids and self._rigid_graspable."""
        pass

    def compute_reward(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Camera config
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
    #  Initialization
    # ------------------------------------------------------------------ #

    def _init_sim(self):
        """Create the PyBullet client (once). Actual scene setup in reset."""
        mode = pybullet.GUI if self.vis else pybullet.DIRECT
        self.sim = bclient.BulletClient(connection_mode=mode)
        if self.vis:
            # Configure GUI for better visualization
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 1)
            self.sim.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.4, 0, 0.1],
            )

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

    def _setup_sim(self):
        """Full simulation reset: gravity, floor, robot, objects."""
        sim = self.sim
        sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
        sim.setGravity(0, 0, self.SIM_GRAVITY)
        sim.setTimeStep(1.0 / self.SIM_FREQ)
        
        # [CRITICAL FIX] Increase solver iterations for stable tight-clearance insertion
        sim.setPhysicsEngineParameter(numSolverIterations=100)
        
        sim.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Floor
        self._floor_id = sim.loadURDF("plane.urdf")
        # Apply wood texture if available
        asset_path = os.path.join(
            os.path.dirname(__file__), "..", "sim_mobile", "assets"
        )
        tex_path = os.path.join(asset_path, "textures", "wood.jpg")
        if os.path.exists(tex_path):
            tex_id = sim.loadTexture(tex_path)
            sim.changeVisualShape(
                self._floor_id, -1, rgbaColor=[1, 1, 1, 1],
                textureUniqueId=tex_id,
            )

        # Load Franka Panda
        self._load_robot()

        # Task objects
        self.rigid_ids = []
        self._rigid_graspable = []
        self.soft_ids = []
        self._create_task_objects()

        # Let physics settle
        for _ in range(20):
            sim.stepSimulation()

        # Disable collisions between robot and graspable objects permanently.
        # The Panda's collision meshes (convex hulls) are larger than the visual
        # geometry, creating an invisible "force field" that pushes objects away.
        # Since we use constraint-based grasping, physical contact isn't needed.
        self._disable_graspable_collisions()

    def _load_robot(self):
        """Load Franka Panda from pybullet_data with fixed base."""
        from pefm_envs.sim_mobile.utils.info import FRANKA_HOME_QPOS

        robot_info = SIM_ROBOT_INFO["franka_panda"]
        robot_path = os.path.join(
            pybullet_data.getDataPath(), robot_info["file_name"]
        )

        # Use a reasonable default EE pose for initial IK
        rest_ee_pos = np.array([0.4, 0.0, 0.3])
        rest_ee_quat = pybullet.getQuaternionFromEuler([np.pi, 0, 0])

        self.robot = BulletRobot(
            self.sim,
            robot_path,
            control_mode="position",
            ee_joint_name=robot_info["ee_joint_name"],
            ee_link_name=robot_info["ee_link_name"],
            base_pos=self.ARM_BASE_POS.tolist(),
            base_quat=[0, 0, 0, 1],
            global_scaling=1.0,
            use_fixed_base=True,
            use_fixed_arm=False,
            rest_arm_pos=rest_ee_pos,
            rest_arm_quat=rest_ee_quat,
            kp=0.5,
            kd=1.0,
            debug=False,
        )

        # Override IK rest poses with the standard Franka home configuration.
        home_qpos = np.zeros(self.robot.info.dof)
        home_qpos[:7] = FRANKA_HOME_QPOS
        home_qpos[7:] = 0.04  # fingers open
        self.robot.rest_qpos = home_qpos
        self.robot.default_ik_args["restPoses"] = home_qpos.tolist()

        # Reset robot to the actual home joint configuration directly
        for jid in range(self.robot.info.dof):
            self.sim.resetJointState(
                bodyUniqueId=self.robot.info.robot_id,
                jointIndex=self.robot.info.joint_ids[jid],
                targetValue=home_qpos[jid],
                targetVelocity=0,
            )
        self.robot.clear_motor_control()

    # ------------------------------------------------------------------ #
    #  Reset / Step
    # ------------------------------------------------------------------ #

    def reset(self):
        self._t = 0
        self._internal_t = 0
        self._frames = []
        self._episode_reward = 0.0
        self._ac_noise_multiplier = self.rng.rand()
        self.constraint_id = None
        self._grasped_obj_id = None
        self._randomize_object_scales()
        self._setup_sim()
        self._done = False
        obs = self._get_obs()
        return obs

    def step(self, action, dummy_reward=False):
        action = np.array(action).reshape(1, self.dof)

        # Add action noise
        if self.ac_noise > 0:
            action[0, 1:4] += (
                self.rng.randn(3) * self.ac_noise * self._ac_noise_multiplier
            )

        # Gripper: determine finger distance for IK
        grip_action = action[0, 0]
        gripper_closing = grip_action >= 0.5

        if gripper_closing:
            fing_dist = 0.0  # fingers closed
        else:
            fing_dist = self.robot.get_max_fing_dist()  # fingers open
            self._detach_grasp()

        # EEF velocity → target EEF pose
        pos_vel = action[0, 1:4]  # vx, vy, vz in world frame
        ori_vel = action[0, 4:7] if self.dof >= 7 else np.zeros(3)

        dt = 1.0 / self.freq
        ee_pos, ee_quat, _, _ = self.robot.get_ee_pos_quat_vel()

        # Target position
        target_pos = ee_pos + pos_vel * dt
        # Clamp Z to avoid going below table
        target_pos[2] = max(target_pos[2], 0.005)

        # Target orientation
        if np.linalg.norm(ori_vel) > 1e-6:
            ori_delta = ori_vel * dt  # axis-angle in radians
            delta_quat = axisangle2quat(ori_delta)
            target_quat = quat_multiply(delta_quat, ee_quat)
        else:
            target_quat = ee_quat

        # Single IK solve → joint-space interpolation.
        # Solving IK once and interpolating qpos gives stable, consistent
        # motion without the orientation drift caused by per-sub-step IK
        # (the 7-DOF Franka is redundant, so repeated IK calls find
        # subtly different solutions).
        target_qpos = self.robot.ee_pos_to_qpos(
            target_pos, target_quat, fing_dist=fing_dist
        )

        steps_per_action = self.SIM_FREQ // self.freq
        if self.demo_mode:
            kp_val = 5.0
            kd_val = 2.0
        else:
            kp_val = None  # use robot defaults
            kd_val = None

        if target_qpos is not None:
            curr_qpos = self.robot.get_qpos()
            for st in range(steps_per_action):
                alpha = (st + 1) / steps_per_action
                interp_qpos = curr_qpos + alpha * (target_qpos - curr_qpos)
                self.robot.move_to_qpos(
                    interp_qpos, mode=pybullet.POSITION_CONTROL,
                    kp=kp_val, kd=kd_val,
                )
                self.sim.stepSimulation()
                self._internal_t += 1
        else:
            for _ in range(steps_per_action):
                self.sim.stepSimulation()
                self._internal_t += 1

        # After the full motion (72 sub-steps), fingers have had time to
        # close and make contact. Now try to create the grasp constraint.
        if gripper_closing and self.constraint_id is None:
            self._attach_grasp()

        # Realtime sleep for visualization
        if self.vis and self._last_action_time is not None:
            elapsed = time.time() - self._last_action_time
            target_time = 1.0 / self.freq
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
        self._last_action_time = time.time()

        # Observation
        obs = self._get_obs()

        # Reward and done
        self._t += 1
        rew = 0.0 if dummy_reward else self.compute_reward()
        done = self._t >= self.max_episode_length

        return obs, rew, done, {}

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #

    def _get_obs(self):
        """
        Returns (1, 13) observation:
          [eef_xyz(3), x_dir(3), z_dir(3), gravity(3), grip(1)]
        """
        ee_pos, ee_quat, _, _ = self.robot.get_ee_pos_quat_vel()
        rot_mat = quat2mat(ee_quat)

        x_dir = rot_mat[:, 0]  # first column = local X
        z_dir = rot_mat[:, 2]  # third column = local Z
        gravity_dir = np.array([0.0, 0.0, -1.0])
        grip = np.array([1.0 if self.constraint_id is not None else 0.0])

        obs = np.concatenate([ee_pos, x_dir, z_dir, gravity_dir, grip])
        return obs.reshape(1, 13)

    # ------------------------------------------------------------------ #
    #  Grasping (constraint-based, collisions permanently disabled)
    # ------------------------------------------------------------------ #

    def _disable_graspable_collisions(self):
        """Disable collisions between robot and all graspable objects.

        Called once at scene setup. The Panda URDF's collision meshes are
        convex hulls much larger than the visual geometry, creating an
        invisible force field that knocks objects over during approach.
        Since grasping is constraint-based, physical contact is not needed.
        """
        robot_id = self.robot.info.robot_id
        num_robot_links = self.sim.getNumJoints(robot_id)
        for oid, graspable in zip(self.rigid_ids, self._rigid_graspable):
            if graspable:
                for link_idx in range(-1, num_robot_links):
                    self.sim.setCollisionFilterPair(
                        robot_id, oid, link_idx, -1, 0
                    )

    def _attach_grasp(self):
        """Find the closest graspable object and lock it to the EE.

        Uses geometric (mesh vertex) distance — works regardless of
        collision filter state.
        """
        if self.constraint_id is not None:
            return

        graspable_ids = [
            oid for oid, g in zip(self.rigid_ids, self._rigid_graspable) if g
        ]
        if not graspable_ids:
            return

        ee_pos = self.robot.get_ee_pos()
        target_obj_id, _ = self._find_closest_graspable(
            ee_pos, graspable_ids, max_dist=0.08
        )

        if target_obj_id is not None:
            self._lock_object_in_place(target_obj_id)

    def _lock_object_in_place(self, obj_id):
        """Lock object at its exact current pose relative to EE."""
        robot_id = self.robot.info.robot_id
        ee_link_id = self.robot.info.ee_link_id

        # Compute object pose relative to the EE's URDF link frame.
        # getLinkState[4:6] = world URDF link frame (matches createConstraint's
        # parentFramePosition reference frame). [0:2] is the CoM frame, which
        # is offset from the URDF frame for panda_hand, causing the object to
        # snap to the wrong position.
        link_state = self.sim.getLinkState(
            robot_id, ee_link_id, computeForwardKinematics=1
        )
        ee_pos = link_state[4]  # URDF link frame world position
        ee_ori = link_state[5]  # URDF link frame world orientation
        obj_pos, obj_ori = self.sim.getBasePositionAndOrientation(obj_id)

        inv_ee_pos, inv_ee_ori = self.sim.invertTransform(ee_pos, ee_ori)
        rel_pos, rel_ori = self.sim.multiplyTransforms(
            inv_ee_pos, inv_ee_ori, obj_pos, obj_ori
        )

        self.constraint_id = self.sim.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=ee_link_id,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=list(rel_pos),
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=list(rel_ori),
            childFrameOrientation=[0, 0, 0, 1],
        )
        self.sim.changeConstraint(self.constraint_id, maxForce=2000)
        self._grasped_obj_id = obj_id

    def _detach_grasp(self):
        """Remove grasp constraint."""
        if self.constraint_id is not None:
            self.sim.removeConstraint(self.constraint_id)
            self.constraint_id = None
            self._grasped_obj_id = None

    def _find_closest_graspable(self, ee_pos, obj_ids, max_dist=0.08):
        """Find closest graspable object using geometric mesh distance."""
        best_dist = np.inf
        best_obj = None
        best_pos = None

        for obj_id in obj_ids:
            num_joints = self.sim.getNumJoints(obj_id)
            if num_joints == 0:
                mesh = self._get_rigid_body_mesh(obj_id)
            else:
                parts = []
                base_mesh = self._get_rigid_body_mesh(obj_id, link_index=None)
                if base_mesh.size > 0:
                    parts.append(base_mesh)
                for i in range(num_joints):
                    link_mesh = self._get_rigid_body_mesh(obj_id, link_index=i)
                    if link_mesh.size > 0:
                        parts.append(link_mesh)
                mesh = np.concatenate(parts) if parts else np.zeros((0, 3))

            if mesh.size == 0:
                continue

            closest_ids = get_closest(ee_pos, mesh)
            vertex_pos = mesh[closest_ids[0]]
            dist = np.linalg.norm(ee_pos - vertex_pos)
            if dist < best_dist:
                best_dist = dist
                best_obj = obj_id
                best_pos = vertex_pos

        if best_dist > max_dist:
            return None, None
        return best_obj, best_pos

    # ------------------------------------------------------------------ #
    #  Rigid body mesh (for grasping and point cloud)
    # ------------------------------------------------------------------ #

    def _get_rigid_body_mesh(self, obj_id, link_index=None):
        """Get world-frame mesh points for a rigid body or link."""
        if link_index is not None and link_index >= 0:
            # Specific child link
            try:
                mesh_data = self.sim.getMeshData(obj_id, linkIndex=link_index)
                mesh = (
                    np.array(mesh_data[1])
                    if len(mesh_data[1]) > 0
                    else np.zeros((0, 3))
                )
            except Exception:
                mesh = np.zeros((0, 3))

            if mesh.size == 0:
                mesh = self._sample_link_surface(obj_id, link_index)
                if mesh.size == 0:
                    return np.zeros((0, 3))

            link_state = self.sim.getLinkState(obj_id, link_index)
            pos, ori = np.array(link_state[0]), link_state[1]
            rot = Rotation.from_quat(ori)
            mesh = np.dot(mesh, rot.as_matrix().T) + pos[None]
            return mesh
        else:
            # Base link or free object
            mesh_data = self.sim.getMeshData(obj_id)
            mesh = (
                np.array(mesh_data[1])
                if len(mesh_data[1]) > 0
                else np.zeros((0, 3))
            )
            if mesh.size == 0:
                mesh = self._sample_primitive_surface(obj_id)
                if mesh.size == 0:
                    return np.zeros((0, 3))
            obj_pos, obj_ori = self.sim.getBasePositionAndOrientation(obj_id)
            rot = Rotation.from_quat(obj_ori)
            mesh = np.dot(mesh, rot.as_matrix().T) + np.array(obj_pos)[None]
            return mesh

    def _sample_primitive_surface(self, obj_id, num_samples=200):
        """Generate surface points for PyBullet primitive shapes."""
        shape_data = self.sim.getCollisionShapeData(obj_id, -1)
        if len(shape_data) == 0:
            return np.zeros((0, 3))
        shape_type = shape_data[0][2]
        dims = shape_data[0][3]

        if shape_type == pybullet.GEOM_CYLINDER:
            radius, height = dims[1], dims[0]
            n = num_samples
            angles = np.random.uniform(0, 2 * np.pi, n)
            n_side = int(n * 0.7)
            n_cap = (n - n_side) // 2
            z_side = np.random.uniform(-height / 2, height / 2, n_side)
            pts_side = np.stack([
                radius * np.cos(angles[:n_side]),
                radius * np.sin(angles[:n_side]),
                z_side,
            ], axis=-1)
            r_top = np.sqrt(np.random.uniform(0, 1, n_cap)) * radius
            pts_top = np.stack([
                r_top * np.cos(angles[n_side:n_side + n_cap]),
                r_top * np.sin(angles[n_side:n_side + n_cap]),
                np.full(n_cap, height / 2),
            ], axis=-1)
            n_bot = n - n_side - n_cap
            r_bot = np.sqrt(np.random.uniform(0, 1, n_bot)) * radius
            pts_bot = np.stack([
                r_bot * np.cos(angles[n_side + n_cap:]),
                r_bot * np.sin(angles[n_side + n_cap:]),
                np.full(n_bot, -height / 2),
            ], axis=-1)
            return np.concatenate([pts_side, pts_top, pts_bot], axis=0)

        elif shape_type == pybullet.GEOM_SPHERE:
            radius = dims[0]
            n = num_samples
            phi = np.random.uniform(0, 2 * np.pi, n)
            cos_theta = np.random.uniform(-1, 1, n)
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            return np.stack([
                radius * sin_theta * np.cos(phi),
                radius * sin_theta * np.sin(phi),
                radius * cos_theta,
            ], axis=-1)

        elif shape_type == pybullet.GEOM_BOX:
            half = np.array(dims)
            pts = []
            for axis in range(3):
                for sign in [-1, 1]:
                    n_face = num_samples // 6
                    face = np.random.uniform(-1, 1, (n_face, 3)) * half
                    face[:, axis] = sign * half[axis]
                    pts.append(face)
            return np.concatenate(pts, axis=0)

        return np.zeros((0, 3))

    def _sample_link_surface(self, obj_id, link_index, num_samples=60):
        """Generate surface points for a collision shape on a link."""
        shape_data = self.sim.getCollisionShapeData(obj_id, link_index)
        if len(shape_data) == 0:
            return np.zeros((0, 3))
        dims = shape_data[0][3]
        local_pos = np.array(shape_data[0][5])
        half = np.array(dims)
        pts = []
        n_per_face = num_samples // 6
        for axis in range(3):
            for sign in [-1, 1]:
                face = np.random.uniform(-1, 1, (n_per_face, 3)) * half
                face[:, axis] = sign * half[axis]
                pts.append(face + local_pos)
        if not pts:
            return np.zeros((0, 3))
        return np.concatenate(pts, axis=0)

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #

    def render(self, cam_config=None, return_depth=True, return_pc=True,
               return_seg=False, resolution=240):
        """
        Render from a single camera configuration.

        Args:
            cam_config: dict with pitch, yaw, roll, distance, fov, target
            return_depth: include depth images
            return_pc: include segmented point cloud
            return_seg: include segmentation masks
            resolution: image resolution
        """
        if cam_config is None:
            cam_config = self.default_front_camera

        cam_target = cam_config.get("target", [0.4, 0.0, 0.1])
        cam_info = {
            "yaws": [cam_config["yaw"]],
            "rolls": [cam_config.get("roll", 0)],
            "pitches": [cam_config["pitch"]],
            "dist": cam_config["distance"],
            "views": [0],
            "fov": cam_config["fov"],
            "width": resolution,
            "height": resolution,
            "target": cam_target,
        }

        # MultiCamera.render already linearizes depth and transposes when
        # return_depth=True. Request depth whenever PC is needed.
        need_depth = return_depth or return_pc
        rendered = MultiCamera.render(
            self.sim,
            self.rigid_ids,
            cam_rolls=cam_info["rolls"],
            cam_yaws=cam_info["yaws"],
            cam_pitches=cam_info["pitches"],
            cam_dist=cam_info["dist"],
            views=cam_info["views"],
            fov=cam_info["fov"],
            cam_target=cam_target,
            width=resolution,
            height=resolution,
            return_seg=(return_seg or return_pc),
            return_depth=need_depth,
        )

        rendered["images"] = [
            np.array(x).reshape(resolution, resolution, 4)
            for x in rendered["images"]
        ]

        if (return_seg or return_pc) and len(self.rigid_ids) > 0:
            rendered["segs"] = np.mod(rendered["segs"], (1 << 24))

        if return_pc:
            pc = self._compute_pc(cam_info, rendered["depths"],
                                  rendered["segs"], rendered["images"])
            rendered["pc"] = pc
        else:
            rendered["pc"] = None

        return rendered

    def render_dual(self, resolution=240):
        """Render from both front and side cameras, return side-by-side."""
        front = self.render(
            cam_config=self.default_front_camera,
            return_depth=False, return_pc=False, resolution=resolution,
        )
        side = self.render(
            cam_config=self.default_side_camera,
            return_depth=False, return_pc=False, resolution=resolution,
        )
        front_img = front["images"][0][..., :3]
        side_img = side["images"][0][..., :3]
        return np.concatenate([front_img, side_img], axis=1)

    def _compute_pc(self, cam_info, view_depths, view_segs, view_images):
        """Extract segmented point cloud from depth + segmentation."""
        H, W = cam_info["height"], cam_info["width"]
        cam_vals = MultiCamera.get_cam_vals(
            [0], cam_info["yaws"], cam_info["pitches"],
            cam_info["dist"], cam_info["target"],
            cam_info["fov"], float(W / H),
        )

        view_mat, p_proj_mat = cam_vals[0][:2]
        view_mat = np.array(view_mat).reshape((4, 4), order="C")
        p_proj_mat = np.array(p_proj_mat).reshape((4, 4), order="C")

        cx = (1 - p_proj_mat[0, 2]) * W / 2
        cy = (p_proj_mat[1, 2] + 1) * H / 2
        proj_mat = np.array([
            [-p_proj_mat[0, 0] * W / 2, 0, cx],
            [0, p_proj_mat[1, 1] * H / 2, cy],
            [0, 0, 1],
        ])

        # Segment by object IDs
        obj_ids = self.rigid_ids
        segs = view_segs[0]
        seg = np.isin(segs, obj_ids).astype(np.uint8) * 255
        object_pixels = np.array(np.where(seg == 255)).T

        # Unproject
        view_depth = view_depths[0].T
        extrinsics = view_mat.copy().T
        extrinsics[[1, 2]] *= -1
        extrinsics = np.linalg.inv(extrinsics)
        intrinsics = proj_mat.copy()
        intrinsics[0, 0] *= -1

        pc = unproject_depth(
            [view_depth], [intrinsics], [extrinsics],
            filter_pixels=[object_pixels], clip_radius=10.0,
        )
        return pc

    def close(self):
        """Disconnect from the PyBullet physics server."""
        if hasattr(self, "sim") and self.sim is not None:
            self.sim.disconnect()