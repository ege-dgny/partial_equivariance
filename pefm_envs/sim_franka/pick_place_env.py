"""
Pick-and-Place environment: SO(2) symmetry conflict.

A cylinder (SO(2) symmetric) must be grasped from a random position
and placed on a FIXED world-frame target tray. The grasp phase has
full rotational symmetry, but the placement breaks it.
"""

import numpy as np
import pybullet

from .franka_env import FrankaEnv


class PickPlaceEnv(FrankaEnv):

    CYLINDER_RADIUS = 0.03   # 3cm
    CYLINDER_HEIGHT = 0.08   # 8cm

    # Fixed world-frame target (within Franka reach)
    TRAY_POS = np.array([0.35, 0.25, 0.0])
    TRAY_SIZE = np.array([0.08, 0.08, 0.005])

    # Object spawn: arc ~0.5m from arm base
    SPAWN_RADIUS = 0.5
    SPAWN_ANGLE_RANGE = (-np.pi / 3, np.pi / 3)  # ±60 degrees

    @property
    def name(self):
        return "pick_place"

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
        self._cylinder_id = self._create_cylinder()
        self._tray_id = self._create_tray()

        self.rigid_ids.append(self._cylinder_id)
        self._rigid_graspable.append(True)

        self.rigid_ids.append(self._tray_id)
        self._rigid_graspable.append(False)

        # Record spawn position for reference
        pos, _ = self.sim.getBasePositionAndOrientation(self._cylinder_id)
        self._spawn_pos = np.array(pos)

    def _create_cylinder(self):
        r = self.CYLINDER_RADIUS
        h = self.CYLINDER_HEIGHT

        col = self.sim.createCollisionShape(pybullet.GEOM_CYLINDER, radius=r, height=h)
        vis = self.sim.createVisualShape(
            pybullet.GEOM_CYLINDER, radius=r, length=h,
            rgbaColor=[0.4, 0.6, 0.85, 1.0],
        )

        # Spawn on arc around arm base, rotated by _object_rotation
        ang = self._object_rotation[-1]
        spawn_x = self.SPAWN_RADIUS * np.cos(ang)
        spawn_y = self.SPAWN_RADIUS * np.sin(ang)
        pos = [spawn_x, spawn_y, h / 2 + 0.001]

        cyl_id = self.sim.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        self.sim.changeDynamics(cyl_id, -1, lateralFriction=1.0)
        return cyl_id

    def _create_tray(self):
        half = self.TRAY_SIZE / 2
        col = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half.tolist())
        vis = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=half.tolist(),
            rgbaColor=[0.6, 0.3, 0.1, 1.0],
        )
        pos = [self.TRAY_POS[0], self.TRAY_POS[1], self.TRAY_SIZE[2] / 2]
        tray_id = self.sim.createMultiBody(
            baseMass=0,  # fixed
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        return tray_id

    def compute_reward(self):
        """
        Reward: cylinder proximity to tray center + correct height + release.

        Full success (1.0): cylinder on tray, released.
        Shaping: XY proximity (0.4) + height (0.3) + release bonus (0.3).
        """
        cyl_pos, _ = self.sim.getBasePositionAndOrientation(self._cylinder_id)
        cyl_pos = np.array(cyl_pos)

        tray_center = self.TRAY_POS[:2]
        xy_dist = np.linalg.norm(cyl_pos[:2] - tray_center)

        on_tray = xy_dist < self.TRAY_SIZE[0]
        at_height = abs(cyl_pos[2] - (self.TRAY_SIZE[2] + self.CYLINDER_HEIGHT / 2)) < 0.02
        released = self.constraint_id is None

        if on_tray and at_height and released:
            return 1.0

        reward = 0.0
        # XY proximity (0-0.4)
        reward += 0.4 * max(0, 1.0 - xy_dist / 0.4)
        # Height proximity (0-0.3) - only if XY close
        if xy_dist < 0.15:
            target_z = self.TRAY_SIZE[2] + self.CYLINDER_HEIGHT / 2
            z_err = abs(cyl_pos[2] - target_z)
            reward += 0.3 * max(0, 1.0 - z_err / 0.15)
        # Release bonus (0.3) - only if on tray
        if on_tray and released:
            reward += 0.3

        return np.clip(reward, 0.0, 1.0)
