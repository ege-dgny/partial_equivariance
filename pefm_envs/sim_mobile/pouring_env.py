import os
import numpy as np
import pybullet as p

from .base_env import BaseEnv
from .utils.init_utils import rotate_around_z


class PouringEnv(BaseEnv):
    """
    Constrained pouring environment demonstrating symmetry conflict.

    A mug containing a bead must be grasped (object-centric, rotationally
    symmetric) and then poured into a bowl at a FIXED world-frame position
    (breaking symmetry). The grasp phase has continuous rotational symmetry
    (C_inf around Z), but the pour phase requires moving to a world-fixed
    target, creating the asymmetric constraint that PEFM is designed to handle.

    Objects are created from PyBullet primitives (no mesh files needed).
    """

    MUG_RADIUS = 0.035        # 3.5cm
    MUG_HEIGHT = 0.08          # 8cm
    MUG_WALL_THICKNESS = 0.004 # 4mm
    BOWL_RADIUS = 0.06         # 6cm
    BOWL_HEIGHT = 0.04          # 4cm
    BEAD_RADIUS = 0.008         # 8mm
    BOWL_WORLD_POS = np.array([0.25, 0.0, 0.0])  # FIXED in world frame

    @property
    def robot_config(self):
        # Robot 0: behind the mug (rotated with object)
        # Robot 1: near the bowl (world-frame, secondary/idle)
        mug_pos = rotate_around_z(
            np.array([0.0, 0.0, 0.0]), self._object_rotation[-1]
        )
        r0_pos = np.array([-0.75, 0.0, 0.005])
        r0_pos = rotate_around_z(r0_pos, self._object_rotation[-1])
        r0_pos[:2] += self.scene_offset

        r1_pos = np.array([self.BOWL_WORLD_POS[0] + 0.75, self.BOWL_WORLD_POS[1], 0.005])

        rest_arm_pos = np.array([0.75, 0.0, self.MUG_HEIGHT * 1.5 - self.ARM_MOUNTING_HEIGHT])
        rest_arm_rot = np.array([np.pi * 0.5, np.pi, np.pi / 2])

        robots = [
            {
                "sim_robot_name": "kinova",
                "rest_base_pose": np.array(
                    [r0_pos[0], r0_pos[1], self._object_rotation[-1] + np.pi]
                ),
                "rest_arm_pos": rest_arm_pos,
                "rest_arm_rot": rest_arm_rot,
            },
            {
                "sim_robot_name": "kinova",
                "rest_base_pose": np.array(
                    [r1_pos[0], r1_pos[1], 0.0]
                ),
                "rest_arm_pos": rest_arm_pos,
                "rest_arm_rot": rest_arm_rot,
            },
        ]
        return robots

    @property
    def default_camera_config(self):
        cfg = super().default_camera_config
        cfg["pitch"] = -45
        cfg["yaw"] = 30
        cfg["distance"] = 1.5
        cfg["target"] = [0.1, 0.0, 0.1]
        return cfg

    @property
    def rigid_objects(self):
        return []

    @property
    def anchor_config(self):
        return []

    @property
    def soft_objects(self):
        return []

    @property
    def name(self):
        return "pour"

    def _reset_sim(self):
        super()._reset_sim()

        # Create mug, bowl, and bead
        self._mug_id = self._create_mug()
        self._bowl_id = self._create_bowl()
        self._bead_id = self._create_bead()

        # Register mug as graspable rigid object
        self.rigid_ids.append(self._mug_id)
        self._rigid_graspable.append(True)

        # Register bowl (not graspable)
        self.rigid_ids.append(self._bowl_id)
        self._rigid_graspable.append(False)

        # Register bead (not graspable)
        self.rigid_ids.append(self._bead_id)
        self._rigid_graspable.append(False)

        # Let objects settle
        for _ in range(50):
            self.sim.stepSimulation()

    def _create_mug(self):
        """Create a cylindrical mug from PyBullet primitives."""
        r = self.MUG_RADIUS
        h = self.MUG_HEIGHT
        wt = self.MUG_WALL_THICKNESS

        # Mug body: hollow cylinder approximated by a cylinder collision shape
        col_shape = self.sim.createCollisionShape(
            p.GEOM_CYLINDER, radius=r, height=h
        )
        vis_shape = self.sim.createVisualShape(
            p.GEOM_CYLINDER, radius=r, length=h,
            rgbaColor=[0.85, 0.85, 0.9, 1.0]
        )

        # Position: scene center + rotation offset
        mug_pos = np.array([0.0, 0.0, h / 2 + 0.001])
        mug_pos[:2] += self.scene_offset

        mug_quat = p.getQuaternionFromEuler([0, 0, self._object_rotation[-1]])

        mug_id = self.sim.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=mug_pos.tolist(),
            baseOrientation=mug_quat,
        )

        # Set friction
        self.sim.changeDynamics(mug_id, -1, lateralFriction=1.0)

        return mug_id

    def _create_bowl(self):
        """Create a bowl at a FIXED world-frame position (never rotated)."""
        r = self.BOWL_RADIUS
        h = self.BOWL_HEIGHT

        col_shape = self.sim.createCollisionShape(
            p.GEOM_CYLINDER, radius=r, height=h
        )
        vis_shape = self.sim.createVisualShape(
            p.GEOM_CYLINDER, radius=r, length=h,
            rgbaColor=[0.6, 0.3, 0.1, 1.0]
        )

        # Bowl is ALWAYS at world-frame position (not affected by scene_offset)
        bowl_pos = [
            self.BOWL_WORLD_POS[0],
            self.BOWL_WORLD_POS[1],
            h / 2 + 0.001,
        ]

        bowl_id = self.sim.createMultiBody(
            baseMass=0,  # fixed
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=bowl_pos,
        )

        return bowl_id

    def _create_bead(self):
        """Create a small bead inside the mug."""
        r = self.BEAD_RADIUS

        col_shape = self.sim.createCollisionShape(
            p.GEOM_SPHERE, radius=r
        )
        vis_shape = self.sim.createVisualShape(
            p.GEOM_SPHERE, radius=r,
            rgbaColor=[1.0, 0.2, 0.2, 1.0]
        )

        # Spawn bead inside the mug
        mug_pos, _ = self.sim.getBasePositionAndOrientation(self._mug_id)
        bead_pos = [
            mug_pos[0],
            mug_pos[1],
            mug_pos[2] + self.MUG_HEIGHT * 0.3,
        ]

        bead_id = self.sim.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=bead_pos,
        )

        # High friction so bead stays in mug
        self.sim.changeDynamics(bead_id, -1, lateralFriction=1.0, restitution=0.1)

        return bead_id

    def compute_reward(self):
        """
        Reward based on bead proximity to bowl center.

        Full success (1.0): bead XY within bowl and at correct Z height.
        Shaping (0-0.5): proportional to XY proximity.
        """
        bead_pos, _ = self.sim.getBasePositionAndOrientation(self._bead_id)
        bead_pos = np.array(bead_pos)

        bowl_center = np.array([
            self.BOWL_WORLD_POS[0],
            self.BOWL_WORLD_POS[1],
            self.BOWL_HEIGHT / 2 + 0.001,
        ])

        xy_dist = np.linalg.norm(bead_pos[:2] - bowl_center[:2])

        # Binary success check
        in_bowl_xy = xy_dist < self.BOWL_RADIUS * 0.9
        in_bowl_z = bead_pos[2] >= 0 and bead_pos[2] < self.BOWL_HEIGHT * 1.5

        if in_bowl_xy and in_bowl_z:
            return 1.0

        # Shaping reward: proximity
        shaping = 0.5 * (1.0 - min(xy_dist / 0.5, 1.0))
        return shaping
