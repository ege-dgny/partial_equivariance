"""
Push-T environment: 2.5D planar pushing task.

A T-shaped block on the table must be pushed to a target pose
(position + orientation) using a dowel attached to the Franka gripper.
Adapted from the Diffusion Policy Push-T benchmark (Chi et al., RSS 2023).

The T-block has C2 symmetry (180 deg looks the same from above).
The pushing direction is unconstrained, but the target pose is fixed
in world frame — creating a symmetry conflict.
"""

import numpy as np
import pybullet

from .franka_env import FrankaEnv


class PushTEnv(FrankaEnv):

    # T-block dimensions
    T_BAR_LENGTH = 0.10    # 10cm horizontal bar
    T_BAR_WIDTH = 0.03     # 3cm
    T_STEM_LENGTH = 0.07   # 7cm vertical stem
    T_STEM_WIDTH = 0.03    # 3cm
    T_HEIGHT = 0.02        # 2cm tall

    # Dowel (pusher) dimensions
    DOWEL_RADIUS = 0.01    # 1cm radius
    DOWEL_HEIGHT = 0.04    # 4cm tall

    # Workspace
    PUSH_HEIGHT = 0.005    # EEF height during pushing (just above table)
    WORKSPACE_RADIUS = 0.35  # Reachable workspace radius
    WORKSPACE_CENTER = np.array([0.45, 0.0])

    # Target pose (fixed in world frame)
    TARGET_POS = np.array([0.50, 0.10])
    TARGET_YAW = 0.0  # Target yaw angle

    # Spawn range
    SPAWN_RADIUS_MIN = 0.05
    SPAWN_RADIUS_MAX = 0.15
    SPAWN_ANGLE_RANGE = (-np.pi, np.pi)

    @property
    def name(self):
        return "push_t"

    @property
    def spawn_angle_range(self):
        return self.SPAWN_ANGLE_RANGE

    @property
    def default_front_camera(self):
        return {
            "pitch": -70,
            "yaw": 0,
            "roll": 0,
            "distance": 1.0,
            "fov": 45,
            "target": [0.45, 0.0, 0.0],
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -40,
            "yaw": 90,
            "roll": 0,
            "distance": 1.0,
            "fov": 45,
            "target": [0.45, 0.0, 0.0],
        }

    def _create_task_objects(self):
        # Close gripper permanently (dowel pusher mode)
        self.robot.close_gripper()
        for _ in range(20):
            self.sim.stepSimulation()

        # Create dowel attached to EEF
        self._dowel_id = self._create_dowel()

        # Create T-block at random pose
        self._t_block_id = self._create_t_block()
        self.rigid_ids.append(self._t_block_id)
        self._rigid_graspable.append(False)  # Not graspable — pushed only

        # Create target visualization (transparent)
        self._target_vis_ids = self._create_target_visualization()

        # Let objects settle
        for _ in range(50):
            self.sim.stepSimulation()

    def _create_dowel(self):
        """Create cylindrical dowel pusher attached to EEF via constraint."""
        col = self.sim.createCollisionShape(
            pybullet.GEOM_CYLINDER,
            radius=self.DOWEL_RADIUS,
            height=self.DOWEL_HEIGHT,
        )
        vis = self.sim.createVisualShape(
            pybullet.GEOM_CYLINDER,
            radius=self.DOWEL_RADIUS,
            length=self.DOWEL_HEIGHT,
            rgbaColor=[0.8, 0.2, 0.2, 1.0],  # Red dowel
        )

        ee_pos, ee_quat, _, _ = self.robot.get_ee_pos_quat_vel()
        # Place dowel below EEF (panda_hand link)
        dowel_pos = [ee_pos[0], ee_pos[1], ee_pos[2] - self.DOWEL_HEIGHT / 2 - 0.02]

        dowel_id = self.sim.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=dowel_pos,
        )

        # Attach to EEF with fixed constraint
        robot_id = self.robot.info.robot_id
        ee_link_id = self.robot.info.ee_link_id
        self._dowel_constraint = self.sim.createConstraint(
            robot_id, ee_link_id,
            dowel_id, -1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, -self.DOWEL_HEIGHT / 2 - 0.02],
            childFramePosition=[0, 0, 0],
        )
        self.sim.changeConstraint(self._dowel_constraint, maxForce=10000)
        self.sim.changeDynamics(dowel_id, -1, lateralFriction=1.0)

        return dowel_id

    def _create_t_block(self):
        """Create T-shaped block from two box primitives using compound body."""
        h = self.T_HEIGHT

        # Bar (top of T)
        bar_half = [self.T_BAR_LENGTH / 2, self.T_BAR_WIDTH / 2, h / 2]
        col_bar = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=bar_half)

        # Stem (bottom of T)
        stem_half = [self.T_STEM_WIDTH / 2, self.T_STEM_LENGTH / 2, h / 2]
        col_stem = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=stem_half)

        # Visual shapes
        vis_bar = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=bar_half,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
        )
        vis_stem = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=stem_half,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
        )

        # Random initial pose
        offset_angle = self.rng.uniform(-np.pi, np.pi)
        offset_dist = self.rng.uniform(self.SPAWN_RADIUS_MIN, self.SPAWN_RADIUS_MAX)
        spawn_x = self.WORKSPACE_CENTER[0] + offset_dist * np.cos(offset_angle)
        spawn_y = self.WORKSPACE_CENTER[1] + offset_dist * np.sin(offset_angle)
        spawn_yaw = self.rng.uniform(-np.pi, np.pi)
        spawn_quat = pybullet.getQuaternionFromEuler([0, 0, spawn_yaw])

        # The T center is at the junction of bar and stem.
        # Bar center offset: (0, +bar_width/2, 0) from T junction
        # Stem center offset: (0, -stem_length/2, 0) from T junction
        bar_offset_y = self.T_BAR_WIDTH / 2
        stem_offset_y = -self.T_STEM_LENGTH / 2

        # Create compound body: bar is base, stem is child link
        t_block_id = self.sim.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=col_bar,
            baseVisualShapeIndex=vis_bar,
            basePosition=[spawn_x, spawn_y, h / 2 + 0.001],
            baseOrientation=spawn_quat,
            linkMasses=[0.01],  # Small mass for stem link (inertia from base)
            linkCollisionShapeIndices=[col_stem],
            linkVisualShapeIndices=[vis_stem],
            linkPositions=[[0, -(self.T_BAR_WIDTH / 2 + self.T_STEM_LENGTH / 2), 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[pybullet.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]],
        )

        self.sim.changeDynamics(t_block_id, -1, lateralFriction=1.0)
        self.sim.changeDynamics(t_block_id, 0, lateralFriction=1.0)

        self._initial_block_yaw = spawn_yaw
        return t_block_id

    def _create_target_visualization(self):
        """Create transparent T-shape at target pose for visualization."""
        h = self.T_HEIGHT
        ids = []

        # Bar
        bar_half = [self.T_BAR_LENGTH / 2, self.T_BAR_WIDTH / 2, h / 2]
        vis_bar = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=bar_half,
            rgbaColor=[0.8, 0.2, 0.2, 0.3],  # Transparent red
        )
        target_quat = pybullet.getQuaternionFromEuler([0, 0, self.TARGET_YAW])
        bar_id = self.sim.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis_bar,
            basePosition=[self.TARGET_POS[0], self.TARGET_POS[1], h / 2 + 0.001],
            baseOrientation=target_quat,
        )
        ids.append(bar_id)

        # Stem (offset in T-block local frame, then rotated to world)
        stem_half = [self.T_STEM_WIDTH / 2, self.T_STEM_LENGTH / 2, h / 2]
        vis_stem = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=stem_half,
            rgbaColor=[0.8, 0.2, 0.2, 0.3],
        )
        # Compute stem center in world frame
        stem_local_offset = np.array([0, -(self.T_BAR_WIDTH / 2 + self.T_STEM_LENGTH / 2), 0])
        cos_a, sin_a = np.cos(self.TARGET_YAW), np.sin(self.TARGET_YAW)
        stem_world_offset = np.array([
            cos_a * stem_local_offset[0] - sin_a * stem_local_offset[1],
            sin_a * stem_local_offset[0] + cos_a * stem_local_offset[1],
            0,
        ])
        stem_pos = [
            self.TARGET_POS[0] + stem_world_offset[0],
            self.TARGET_POS[1] + stem_world_offset[1],
            h / 2 + 0.001,
        ]
        stem_id = self.sim.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis_stem,
            basePosition=stem_pos,
            baseOrientation=target_quat,
        )
        ids.append(stem_id)

        return ids

    def get_block_pose(self):
        """Get T-block center position and yaw angle."""
        pos, quat = self.sim.getBasePositionAndOrientation(self._t_block_id)
        euler = pybullet.getEulerFromQuaternion(quat)
        return np.array(pos), euler[2]

    def compute_reward(self):
        """
        Reward: block proximity to target position + orientation alignment.

        Full success (1.0): block at target pose (xy < 2cm, angle < 10deg).
        Shaping: position (0.5) + orientation (0.5).
        """
        block_pos, block_yaw = self.get_block_pose()

        # Position error
        xy_dist = np.linalg.norm(block_pos[:2] - self.TARGET_POS)

        # Orientation error (handle C2 symmetry: 180deg is equivalent)
        yaw_diff = block_yaw - self.TARGET_YAW
        yaw_diff = np.mod(yaw_diff + np.pi, 2 * np.pi) - np.pi
        # C2: also check if 180deg rotated is closer
        yaw_diff_c2 = np.mod(yaw_diff + np.pi, 2 * np.pi) - np.pi
        angle_error = min(abs(yaw_diff), abs(yaw_diff_c2))

        # Full success
        if xy_dist < 0.02 and angle_error < np.deg2rad(10):
            return 1.0

        reward = 0.0
        # Position (0-0.5)
        reward += 0.5 * max(0, 1.0 - xy_dist / 0.3)
        # Orientation (0-0.5)
        reward += 0.5 * max(0, 1.0 - angle_error / np.pi)

        return np.clip(reward, 0.0, 1.0)

    def step(self, action, dummy_reward=False):
        """Override step to constrain EEF to pushing height."""
        action = np.array(action).reshape(1, self.dof)

        # For Push-T, keep gripper closed and Z velocity near zero
        action[0, 0] = 1.0  # Gripper always closed
        # Allow only XY movement; clamp Z velocity toward push height
        ee_pos, _, _, _ = self.robot.get_ee_pos_quat_vel()
        # Compute corrective Z velocity to maintain push height
        z_error = (self.PUSH_HEIGHT + self.DOWEL_HEIGHT + 0.03) - ee_pos[2]
        action[0, 3] = np.clip(z_error * self.freq, -0.5, 0.5)

        return super().step(action, dummy_reward=dummy_reward)
