"""
Orient-and-Place environment: SO(2) symmetry conflict with orientation.

A cylinder with an orientation marker (colored stripe) must be grasped,
reoriented to a target angle, and placed at a FIXED world-frame position.
The grasp has SO(2) symmetry, but placement requires specific world-frame
position AND orientation.

This tests BOTH position AND orientation asymmetry - a stronger signal
for PEFM than pick_place which only tests position.
"""

import numpy as np
import pybullet

from .franka_env import FrankaEnv


class OrientPlaceEnv(FrankaEnv):

    CYLINDER_RADIUS = 0.03   # 3cm
    CYLINDER_HEIGHT = 0.08   # 8cm

    # Fixed world-frame target (within Franka reach)
    TARGET_POS = np.array([0.35, 0.25, 0.0])
    TARGET_ORIENTATION = 0.0  # Target Z-rotation: marker should face +X

    # Object spawn: arc ~0.5m from arm base
    SPAWN_RADIUS = 0.5
    SPAWN_ANGLE_RANGE = (-np.pi / 3, np.pi / 3)  # position angle +-60 deg

    @property
    def name(self):
        return "orient_place"

    @property
    def spawn_angle_range(self):
        return self.SPAWN_ANGLE_RANGE

    @property
    def default_front_camera(self):
        return {
            "pitch": -45,
            "yaw": 0,
            "roll": 0,
            "distance": 1.3,
            "fov": 45,
            "target": [0.35, 0.1, 0.05],
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -30,
            "yaw": 90,
            "roll": 0,
            "distance": 1.3,
            "fov": 45,
            "target": [0.35, 0.1, 0.05],
        }

    def _create_task_objects(self):
        # Random initial orientation for the cylinder marker
        self._marker_initial_angle = self.rng.uniform(0, 2 * np.pi)

        self._cylinder_id = self._create_cylinder_with_marker()

        self.rigid_ids.append(self._cylinder_id)
        self._rigid_graspable.append(True)

        # Record spawn position for reference
        pos, _ = self.sim.getBasePositionAndOrientation(self._cylinder_id)
        self._spawn_pos = np.array(pos)

    def _create_cylinder_with_marker(self):
        """Create cylinder with colored stripe marker on one side."""
        r = self.CYLINDER_RADIUS
        h = self.CYLINDER_HEIGHT

        # Main cylinder (base color)
        col_cyl = self.sim.createCollisionShape(
            pybullet.GEOM_CYLINDER, radius=r, height=h
        )
        vis_cyl = self.sim.createVisualShape(
            pybullet.GEOM_CYLINDER, radius=r, length=h,
            rgbaColor=[0.4, 0.6, 0.85, 1.0],  # Light blue
        )

        # Marker stripe (small box on the side)
        marker_w = 0.005  # 5mm wide
        marker_h = h * 0.8  # 80% of cylinder height
        marker_d = 0.002  # 2mm depth (protrusion)

        col_marker = self.sim.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=[marker_d / 2, marker_w / 2, marker_h / 2]
        )
        vis_marker = self.sim.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=[marker_d / 2, marker_w / 2, marker_h / 2],
            rgbaColor=[1.0, 0.2, 0.2, 1.0],  # Red marker
        )

        # Position: marker on +X side of cylinder (in local frame)
        marker_offset = [r + marker_d / 2, 0, 0]

        # Spawn position on arc around arm base
        ang = self._object_rotation[-1]
        spawn_x = self.SPAWN_RADIUS * np.cos(ang)
        spawn_y = self.SPAWN_RADIUS * np.sin(ang)
        pos = [spawn_x, spawn_y, h / 2 + 0.001]

        # Initial orientation includes random marker angle
        total_rot = self._marker_initial_angle
        quat = pybullet.getQuaternionFromEuler([0, 0, total_rot])

        # Create compound body
        cyl_id = self.sim.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=col_cyl,
            baseVisualShapeIndex=vis_cyl,
            basePosition=pos,
            baseOrientation=quat,
            linkMasses=[0.01],
            linkCollisionShapeIndices=[col_marker],
            linkVisualShapeIndices=[vis_marker],
            linkPositions=[marker_offset],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[pybullet.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]],
        )

        self.sim.changeDynamics(cyl_id, -1, lateralFriction=1.0)
        return cyl_id

    def compute_reward(self):
        """
        Reward: position proximity + orientation alignment + correct height.

        Full success (1.0): cylinder at target pos with correct orientation.
        Shaping: XY proximity (0.3) + height (0.2) + orientation (0.3) + release (0.2).
        """
        cyl_pos, cyl_quat = self.sim.getBasePositionAndOrientation(self._cylinder_id)
        cyl_pos = np.array(cyl_pos)
        cyl_euler = np.array(pybullet.getEulerFromQuaternion(cyl_quat))

        target_xy = self.TARGET_POS[:2]
        xy_dist = np.linalg.norm(cyl_pos[:2] - target_xy)

        # Orientation error (normalize to [-pi, pi])
        z_rot = cyl_euler[2]
        target_rot = self.TARGET_ORIENTATION
        rot_error = np.abs(((z_rot - target_rot + np.pi) % (2 * np.pi)) - np.pi)

        # Height check
        target_z = self.CYLINDER_HEIGHT / 2 + 0.001
        z_err = abs(cyl_pos[2] - target_z)

        on_target = xy_dist < 0.05
        at_height = z_err < 0.02
        oriented = rot_error < np.deg2rad(20)  # Within 20 degrees
        released = self.constraint_id is None

        if on_target and at_height and oriented and released:
            return 1.0

        reward = 0.0
        # XY proximity (0-0.3)
        reward += 0.3 * max(0, 1.0 - xy_dist / 0.4)
        # Height proximity (0-0.2) - only if XY close
        if xy_dist < 0.15:
            reward += 0.2 * max(0, 1.0 - z_err / 0.15)
        # Orientation (0-0.3)
        reward += 0.3 * max(0, 1.0 - rot_error / np.pi)
        # Release bonus (0.2) - only if on target and oriented
        if on_target and oriented and released:
            reward += 0.2

        return np.clip(reward, 0.0, 1.0)
