"""
Stack-on-Base environment: C4 symmetry conflict.

A square block spawns at a random C4 rotation (0, 90, 180, 270 deg) and
must be stacked on a fixed base platform with edges aligned. The grasp
has C4 symmetry but stacking requires world-frame alignment.

This is simpler than peg_insert (no keyway/socket complexity) while still
demonstrating the C4 symmetry conflict pattern.
"""

import numpy as np
import pybullet

from .franka_env import FrankaEnv


class StackEnv(FrankaEnv):

    BLOCK_SIDE = 0.04   # 4cm cube
    BLOCK_HEIGHT = 0.04  # Same as side (cube)

    # Fixed base platform (larger than block, with alignment markers)
    BASE_SIZE = np.array([0.06, 0.06, 0.01])  # 6cm x 6cm x 1cm
    BASE_POS = np.array([0.35, -0.15, 0.0])

    # Block spawn: arc around arm base
    SPAWN_RADIUS = 0.45
    SPAWN_ANGLE_RANGE = (-np.pi / 4, np.pi / 4)  # +-45 deg

    # C4 rotation set
    C4_ROTATIONS = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

    @property
    def name(self):
        return "stack"

    @property
    def spawn_angle_range(self):
        return self.SPAWN_ANGLE_RANGE

    @property
    def default_front_camera(self):
        return {
            "pitch": -50,
            "yaw": 10,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.35, -0.05, 0.05],
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -30,
            "yaw": 90,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.35, -0.05, 0.05],
        }

    def _create_task_objects(self):
        # Sample C4 rotation for block
        demo_mode = getattr(self.args, "demo_mode", False)
        if demo_mode:
            self._block_spawn_rotation = 0.0
        else:
            self._block_spawn_rotation = self.rng.choice(self.C4_ROTATIONS)

        self._block_id = self._create_block()
        self._base_id = self._create_base()

        self.rigid_ids.append(self._block_id)
        self._rigid_graspable.append(True)

        self.rigid_ids.append(self._base_id)
        self._rigid_graspable.append(False)

        # Let objects settle
        for _ in range(30):
            self.sim.stepSimulation()

    def _create_block(self):
        """Create square block with visual markers on edges."""
        s = self.BLOCK_SIDE
        h = self.BLOCK_HEIGHT
        half = [s / 2, s / 2, h / 2]

        col = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half)
        vis = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.3, 0.5, 0.8, 1.0],  # Blue block
        )

        # Spawn position based on spawn angle
        ang = self._object_rotation[-1]
        spawn_x = self.SPAWN_RADIUS * np.cos(ang)
        spawn_y = self.SPAWN_RADIUS * np.sin(ang)
        pos = [spawn_x, spawn_y, h / 2 + 0.001]

        # Include both spawn position angle and C4 rotation
        total_rot = self._object_rotation[-1] + self._block_spawn_rotation
        quat = pybullet.getQuaternionFromEuler([0, 0, total_rot])

        block_id = self.sim.createMultiBody(
            baseMass=0.15,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=quat,
        )
        self.sim.changeDynamics(block_id, -1, lateralFriction=1.0)
        return block_id

    def _create_base(self):
        """Create base platform with alignment guide lines."""
        half = self.BASE_SIZE / 2

        # Main platform
        col_base = self.sim.createCollisionShape(
            pybullet.GEOM_BOX, halfExtents=half.tolist()
        )
        vis_base = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=half.tolist(),
            rgbaColor=[0.5, 0.5, 0.5, 1.0],  # Gray base
        )

        # Alignment marker: thin line on +X edge
        marker_w = 0.003
        marker_l = self.BASE_SIZE[1]
        marker_h = 0.002
        marker_offset = [self.BASE_SIZE[0] / 2 - marker_w, 0, self.BASE_SIZE[2] / 2 + marker_h / 2]

        col_marker = self.sim.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=[marker_w / 2, marker_l / 2, marker_h / 2]
        )
        vis_marker = self.sim.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=[marker_w / 2, marker_l / 2, marker_h / 2],
            rgbaColor=[1.0, 0.3, 0.3, 1.0],  # Red line
        )

        pos = [self.BASE_POS[0], self.BASE_POS[1], self.BASE_SIZE[2] / 2]

        base_id = self.sim.createMultiBody(
            baseMass=0,  # Fixed
            baseCollisionShapeIndex=col_base,
            baseVisualShapeIndex=vis_base,
            basePosition=pos,
            linkMasses=[0.0],
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
        return base_id

    def compute_reward(self):
        """
        Reward: XY proximity + rotation alignment (C4) + height on base.

        Full success (1.0): block on base, edges aligned, released.
        Shaping: position (0.3) + height (0.3) + rotation (0.2) + release (0.2).
        """
        block_pos, block_quat = self.sim.getBasePositionAndOrientation(self._block_id)
        block_pos = np.array(block_pos)
        block_euler = np.array(pybullet.getEulerFromQuaternion(block_quat))

        base_center = self.BASE_POS[:2]
        xy_dist = np.linalg.norm(block_pos[:2] - base_center)

        # C4 rotation alignment: any multiple of 90 deg is correct
        z_rot = block_euler[2]
        # Distance to nearest C4 angle
        rot_error = min(abs((z_rot - r + np.pi) % (2 * np.pi) - np.pi)
                        for r in self.C4_ROTATIONS)

        # Height check: block should sit on top of base
        target_z = self.BASE_SIZE[2] + self.BLOCK_HEIGHT / 2
        z_err = abs(block_pos[2] - target_z)

        on_base = xy_dist < self.BASE_SIZE[0] / 2
        at_height = z_err < 0.015
        rot_aligned = rot_error < np.deg2rad(15)  # Within 15 degrees of C4 angle
        released = self.constraint_id is None

        if on_base and at_height and rot_aligned and released:
            return 1.0

        reward = 0.0
        # XY proximity (0-0.3)
        reward += 0.3 * max(0, 1.0 - xy_dist / 0.25)
        # Height (0-0.3) - only if XY close
        if xy_dist < 0.1:
            reward += 0.3 * max(0, 1.0 - z_err / 0.1)
        # Rotation alignment (0-0.2)
        reward += 0.2 * max(0, 1.0 - rot_error / (np.pi / 4))
        # Release bonus (0.2) - only if on base and aligned
        if on_base and rot_aligned and released:
            reward += 0.2

        return np.clip(reward, 0.0, 1.0)
