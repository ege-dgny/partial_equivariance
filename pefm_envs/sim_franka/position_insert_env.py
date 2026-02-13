"""
Position-Variable Insertion: Fixed-goal insertion with position asymmetry.

A cylindrical peg (SO(2) symmetric) spawns at variable XY positions and must
be inserted into a hole at a FIXED world-frame location. This emphasizes that
the absolute target position cannot be canonicalized through equivariance.

Key PEFM Signal:
- Grasp phase: SO(2) equivariant (cylindrical peg symmetry)
- Insertion phase: Position breaks symmetry (fixed world-frame target)
"""

import numpy as np
import pybullet

from .franka_env import FrankaEnv


class PositionInsertEnv(FrankaEnv):

    PEG_RADIUS = 0.02       # 2cm radius
    PEG_HEIGHT = 0.08       # 8cm height
    SOCKET_RADIUS = 0.025   # Slightly larger than peg
    SOCKET_DEPTH = 0.03

    # Fixed world-frame target (within Franka reach)
    SOCKET_POS = np.array([0.4, 0.2, 0.0])

    # Peg spawn area: variable XY position
    SPAWN_X_RANGE = (0.35, 0.55)
    SPAWN_Y_RANGE = (-0.20, 0.20)

    # Z-rotation range (full circle since peg is cylindrical)
    SPAWN_ANGLE_RANGE = (0.0, 2 * np.pi)

    @property
    def name(self):
        return "position_insert"

    @property
    def spawn_angle_range(self):
        return self.SPAWN_ANGLE_RANGE

    @property
    def default_front_camera(self):
        return {
            "pitch": -45,
            "yaw": 15,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.4, 0.05, 0.05],
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -30,
            "yaw": 90,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.4, 0.05, 0.05],
        }

    def _randomize_object_scales(self):
        super()._randomize_object_scales()

        # Random XY spawn position (independent of rotation)
        self._spawn_xy = np.array([
            self.rng.uniform(*self.SPAWN_X_RANGE),
            self.rng.uniform(*self.SPAWN_Y_RANGE),
        ])

    def _create_task_objects(self):
        self._peg_id = self._create_peg()
        self._socket_id = self._create_socket()

        self.rigid_ids.append(self._peg_id)
        self._rigid_graspable.append(True)

        self.rigid_ids.append(self._socket_id)
        self._rigid_graspable.append(False)

        # Let objects settle
        for _ in range(30):
            self.sim.stepSimulation()

    def _create_peg(self):
        """Create cylindrical peg (SO(2) symmetric, no keyway)."""
        col = self.sim.createCollisionShape(
            pybullet.GEOM_CYLINDER,
            radius=self.PEG_RADIUS,
            height=self.PEG_HEIGHT
        )
        vis = self.sim.createVisualShape(
            pybullet.GEOM_CYLINDER,
            radius=self.PEG_RADIUS,
            length=self.PEG_HEIGHT,
            rgbaColor=[0.4, 0.6, 0.8, 1.0],  # Blue peg
        )

        # Spawn at random XY position
        pos = [self._spawn_xy[0], self._spawn_xy[1], self.PEG_HEIGHT / 2 + 0.001]

        peg_id = self.sim.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        self.sim.changeDynamics(peg_id, -1, lateralFriction=1.0)
        return peg_id

    def _create_socket(self):
        """Create visual socket/hole indicator at fixed world position."""
        # Outer ring as visual indicator
        ring_outer = self.SOCKET_RADIUS + 0.01
        ring_height = 0.005

        col_ring = self.sim.createCollisionShape(
            pybullet.GEOM_CYLINDER,
            radius=ring_outer,
            height=ring_height
        )
        vis_ring = self.sim.createVisualShape(
            pybullet.GEOM_CYLINDER,
            radius=ring_outer,
            length=ring_height,
            rgbaColor=[0.3, 0.3, 0.3, 1.0],  # Gray ring
        )

        socket_id = self.sim.createMultiBody(
            baseMass=0,  # Fixed
            baseCollisionShapeIndex=col_ring,
            baseVisualShapeIndex=vis_ring,
            basePosition=[self.SOCKET_POS[0], self.SOCKET_POS[1], ring_height / 2],
        )
        return socket_id

    def compute_reward(self):
        """
        Reward: XY proximity to fixed socket + descent + release.

        Full success (1.0): peg over socket, descended, released.
        Shaping: position (0.5) + descent (0.3) + release (0.2).
        """
        peg_pos, _ = self.sim.getBasePositionAndOrientation(self._peg_id)
        peg_pos = np.array(peg_pos)

        # XY distance to socket center
        xy_dist = np.linalg.norm(peg_pos[:2] - self.SOCKET_POS[:2])

        # Success criteria
        over_hole = xy_dist < self.SOCKET_RADIUS
        descended = peg_pos[2] < 0.05
        released = self.constraint_id is None

        if over_hole and descended and released:
            return 1.0

        reward = 0.0
        # Position shaping (0-0.5): primary signal
        reward += 0.5 * max(0, 1.0 - xy_dist / 0.3)

        # Descent bonus (0-0.3): only when positioned
        if xy_dist < 0.1:
            descent_progress = max(0, 1.0 - peg_pos[2] / 0.2)
            reward += 0.3 * descent_progress

        # Release bonus (0.2): only if over hole
        if over_hole and released:
            reward += 0.2

        return np.clip(reward, 0.0, 1.0)
