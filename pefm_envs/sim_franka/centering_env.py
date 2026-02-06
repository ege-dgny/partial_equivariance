"""
Centering environment: SO(2) fully symmetric control case.

A cylinder at a random position on the table must be grasped, lifted,
and placed back at its original position. Both grasp and target are
SO(2) symmetric, so the symmetry selector p_phi should remain uniform.
This serves as a control experiment.
"""

import numpy as np
import pybullet

from .franka_env import FrankaEnv


class CenteringEnv(FrankaEnv):

    CYLINDER_RADIUS = 0.03   # 3cm
    CYLINDER_HEIGHT = 0.08   # 8cm
    LIFT_HEIGHT = 0.15       # 15cm lift target

    # Spawn: arc around arm base
    SPAWN_RADIUS = 0.5
    SPAWN_ANGLE_RANGE = (-np.pi / 3, np.pi / 3)

    @property
    def name(self):
        return "centering"

    @property
    def spawn_angle_range(self):
        return self.SPAWN_ANGLE_RANGE

    @property
    def default_front_camera(self):
        return {
            "pitch": -45,
            "yaw": 0,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.4, 0.0, 0.05],
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -30,
            "yaw": 90,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.4, 0.0, 0.05],
        }

    def _create_task_objects(self):
        self._cylinder_id = self._create_cylinder()

        self.rigid_ids.append(self._cylinder_id)
        self._rigid_graspable.append(True)

        # Record spawn position as the target
        pos, _ = self.sim.getBasePositionAndOrientation(self._cylinder_id)
        self._target_xy = np.array(pos[:2])

    def _create_cylinder(self):
        r = self.CYLINDER_RADIUS
        h = self.CYLINDER_HEIGHT

        col = self.sim.createCollisionShape(pybullet.GEOM_CYLINDER, radius=r, height=h)
        vis = self.sim.createVisualShape(
            pybullet.GEOM_CYLINDER, radius=r, length=h,
            rgbaColor=[0.3, 0.7, 0.4, 1.0],
        )

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

    def compute_reward(self):
        """
        Reward: object returned to starting XY within tolerance + height.

        Full success (1.0): cylinder back at spawn, on table, released.
        Shaping: XY proximity (0.4) + height (0.3) + release bonus (0.3).
        """
        cyl_pos, _ = self.sim.getBasePositionAndOrientation(self._cylinder_id)
        cyl_pos = np.array(cyl_pos)

        xy_dist = np.linalg.norm(cyl_pos[:2] - self._target_xy)
        target_z = self.CYLINDER_HEIGHT / 2 + 0.001
        z_err = abs(cyl_pos[2] - target_z)

        on_target = xy_dist < 0.03
        at_height = z_err < 0.02
        released = self.constraint_id is None

        if on_target and at_height and released:
            return 1.0

        reward = 0.0
        reward += 0.4 * max(0, 1.0 - xy_dist / 0.3)

        if xy_dist < 0.1:
            reward += 0.3 * max(0, 1.0 - z_err / 0.15)

        if on_target and released:
            reward += 0.3

        return np.clip(reward, 0.0, 1.0)
