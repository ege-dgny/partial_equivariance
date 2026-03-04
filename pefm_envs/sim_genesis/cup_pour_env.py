"""
Genesis cup pouring task: SO(2) symmetry conflict.
Cup, bowl, ball; contact-based grasping.
"""

import numpy as np

from .genesis_franka_env import GenesisFrankaEnv
from .genesis_robot import _to_numpy

try:
    import genesis as gs
except ImportError:
    gs = None


class GenesisCupPourEnv(GenesisFrankaEnv):

    CUP_RADIUS = 0.03
    CUP_HEIGHT = 0.08
    BOWL_RADIUS = 0.07
    BOWL_HEIGHT = 0.04
    BOWL_POS = np.array([0.4, 0.0, 0.0])
    BALL_RADIUS = 0.012
    SPAWN_RADIUS = 0.45
    SPAWN_ANGLE_RANGE = (-np.pi / 3, np.pi / 3)
    CUP_BOWL_MIN_SEP = 0.12
    POUR_TILT_ANGLE = 2 * np.pi / 3

    @property
    def name(self):
        return "cup_pour_genesis"

    @property
    def spawn_angle_range(self):
        return self.SPAWN_ANGLE_RANGE

    def _randomize_object_scales(self):
        super()._randomize_object_scales()
        R, bowl_xy = self.SPAWN_RADIUS, self.BOWL_POS[:2]
        B, d = np.linalg.norm(bowl_xy), self.CUP_BOWL_MIN_SEP
        if B < 1e-6 or R + B <= d:
            return
        cos_max = np.clip((R*R + B*B - d*d) / (2*R*B), -1.0, 1.0)
        theta_min = np.arccos(cos_max)
        lo, hi = self.SPAWN_ANGLE_RANGE
        if self.randomize_rotation:
            if self.rng.rand() < 0.5 and lo < -theta_min:
                self._object_rotation[-1] = self.rng.uniform(lo, -theta_min)
            elif theta_min < hi:
                self._object_rotation[-1] = self.rng.uniform(theta_min, hi)
            else:
                self._object_rotation[-1] = self.rng.uniform(lo, hi)
        else:
            if abs(self._object_rotation[-1]) < theta_min:
                self._object_rotation[-1] = theta_min

    def _create_task_objects(self):
        if gs is None:
            return
        ang = self._object_rotation[-1]
        cx = self.SPAWN_RADIUS * np.cos(ang)
        cy = self.SPAWN_RADIUS * np.sin(ang)
        cup = self.scene.add_entity(
            gs.morphs.Cylinder(
                radius=self.CUP_RADIUS,
                size=(self.CUP_HEIGHT,),
                pos=(cx, cy, self.CUP_HEIGHT / 2 + 0.001),
                euler=np.rad2deg([0, 0, ang]),
                fixed=False,
            )
        )
        self.rigid_ids.append(cup.idx if hasattr(cup, "idx") else len(self.rigid_ids))
        self._rigid_graspable.append(True)
        self._rigid_entities.append(cup)
        self._cup_entity = cup

        bowl = self.scene.add_entity(
            gs.morphs.Cylinder(
                radius=self.BOWL_RADIUS,
                size=(self.BOWL_HEIGHT,),
                pos=(self.BOWL_POS[0], self.BOWL_POS[1], self.BOWL_HEIGHT / 2),
                fixed=True,
            )
        )
        self.rigid_ids.append(bowl.idx if hasattr(bowl, "idx") else len(self.rigid_ids))
        self._rigid_graspable.append(False)
        self._rigid_entities.append(bowl)
        self._bowl_entity = bowl

        cup_pos = _to_numpy(cup.get_pos())
        ball_z = cup_pos[2] + self.BALL_RADIUS + 0.01
        ball = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=self.BALL_RADIUS,
                pos=(cup_pos[0], cup_pos[1], ball_z),
                fixed=False,
            )
        )
        self.rigid_ids.append(ball.idx if hasattr(ball, "idx") else len(self.rigid_ids))
        self._rigid_graspable.append(False)
        self._rigid_entities.append(ball)
        self._ball_entity = ball

    def compute_reward(self):
        ball_pos = _to_numpy(self._ball_entity.get_pos())
        cup_pos = _to_numpy(self._cup_entity.get_pos())
        bowl_center = self.BOWL_POS[:2]
        ball_xy_dist = np.linalg.norm(ball_pos[:2] - bowl_center)
        ball_in_bowl = ball_xy_dist < self.BOWL_RADIUS and 0 < ball_pos[2] < self.BOWL_HEIGHT + 0.02
        if ball_in_bowl:
            return 1.0
        reward = 0.55 * max(0, 1.0 - ball_xy_dist / 0.4)
        reward += 0.25 * max(0, 1.0 - np.linalg.norm(cup_pos[:2] - bowl_center) / 0.4)
        return np.clip(reward, 0.0, 1.0)
