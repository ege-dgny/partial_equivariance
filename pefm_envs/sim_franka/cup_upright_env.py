"""
Gravity-Sensitive Placement: Cup Upright Task.

A cup/container spawns with random tilt (roll/pitch) and Z-rotation.
The policy must grasp, correct the tilt during transport, and place
upright at a fixed target position.

This demonstrates gravity as a world-frame constraint that breaks
the object-centric symmetry of the grasp phase. PEFM's symmetry selector
should collapse during the reorientation/placement phases.

Key PEFM Signal:
- Grasp phase: Object-centric (approach works from any angle) → HIGH entropy
- Lift+reorient phase: Gravity constraint activates → entropy COLLAPSES
- Placement phase: Fixed world position + upright → LOW entropy

EquiBot Failure Mode:
- Strict equivariance: if observation is rotated, output MUST rotate
- Cannot break symmetry for gravity constraint
- Would place cup at tilted orientation matching input tilt
"""

import numpy as np
import pybullet
from scipy.spatial.transform import Rotation

from .franka_env import FrankaEnv


class CupUprightEnv(FrankaEnv):

    CUP_RADIUS = 0.035      # 3.5cm radius
    CUP_HEIGHT = 0.08       # 8cm height

    # Tilt range: up to 45 degrees
    MAX_TILT = np.pi / 4

    # Fixed world-frame target position
    TARGET_POS = np.array([0.35, 0.2, 0.0])

    # Spawn position: arc around arm base
    SPAWN_RADIUS = 0.5
    SPAWN_ANGLE_RANGE = (-np.pi / 3, np.pi / 3)

    @property
    def name(self):
        return "cup_upright"

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

    def _randomize_object_scales(self):
        super()._randomize_object_scales()

        # Random tilt (roll and pitch)
        # Sample tilt magnitude and direction in polar coordinates
        tilt_magnitude = self.rng.uniform(0, self.MAX_TILT)
        tilt_direction = self.rng.uniform(0, 2 * np.pi)

        # Convert to roll (X-axis tilt) and pitch (Y-axis tilt)
        self._initial_roll = tilt_magnitude * np.cos(tilt_direction)
        self._initial_pitch = tilt_magnitude * np.sin(tilt_direction)

    def _create_task_objects(self):
        self._cup_id = self._create_cup()
        self._target_marker_id = self._create_target_marker()

        self.rigid_ids.append(self._cup_id)
        self._rigid_graspable.append(True)

        # Target marker is not graspable
        self.rigid_ids.append(self._target_marker_id)
        self._rigid_graspable.append(False)

        # Let objects settle (but tilted cup may fall over if tilt is too extreme)
        for _ in range(50):
            self.sim.stepSimulation()

    def _create_cup(self):
        """Create cup with initial tilt orientation."""
        r = self.CUP_RADIUS
        h = self.CUP_HEIGHT

        # Main cylinder body
        col = self.sim.createCollisionShape(
            pybullet.GEOM_CYLINDER,
            radius=r,
            height=h
        )
        vis = self.sim.createVisualShape(
            pybullet.GEOM_CYLINDER,
            radius=r,
            length=h,
            rgbaColor=[0.8, 0.6, 0.4, 1.0],  # Beige cup color
        )

        # Spawn position on arc around arm base
        ang = self._object_rotation[-1]
        spawn_x = self.SPAWN_RADIUS * np.cos(ang)
        spawn_y = self.SPAWN_RADIUS * np.sin(ang)

        # Apply tilt: Euler angles [roll, pitch, yaw] in XYZ convention
        euler = [self._initial_roll, self._initial_pitch, self._object_rotation[-1]]
        quat = pybullet.getQuaternionFromEuler(euler)

        # When tilted, the cup's center of mass shifts
        # Compute position accounting for tilt
        rot = Rotation.from_euler('xyz', euler)
        # Cup center is at h/2 above its base
        base_offset = rot.apply([0, 0, h / 2])

        pos = [spawn_x + base_offset[0], spawn_y + base_offset[1],
               base_offset[2] + 0.002]

        cup_id = self.sim.createMultiBody(
            baseMass=0.15,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=quat,
        )
        self.sim.changeDynamics(cup_id, -1, lateralFriction=1.0)
        return cup_id

    def _create_target_marker(self):
        """Create visual marker for target placement location."""
        # Translucent green circle on ground
        marker_radius = self.CUP_RADIUS + 0.01
        marker_height = 0.003

        col = self.sim.createCollisionShape(
            pybullet.GEOM_CYLINDER,
            radius=marker_radius,
            height=marker_height
        )
        vis = self.sim.createVisualShape(
            pybullet.GEOM_CYLINDER,
            radius=marker_radius,
            length=marker_height,
            rgbaColor=[0.2, 0.8, 0.2, 0.5],  # Translucent green
        )

        marker_id = self.sim.createMultiBody(
            baseMass=0,  # Fixed
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[self.TARGET_POS[0], self.TARGET_POS[1], marker_height / 2],
        )
        return marker_id

    def compute_reward(self):
        """
        Reward: uprightness + position proximity + correct height.

        Full success (1.0): cup at target, upright, on ground, released.
        Shaping: position (0.25) + uprightness (0.45) + height (0.15) + release (0.15).

        Uprightness is the PRIMARY signal - this is what makes the task
        gravity-sensitive and breaks equivariance.
        """
        cup_pos, cup_quat = self.sim.getBasePositionAndOrientation(self._cup_id)
        cup_pos = np.array(cup_pos)

        # Check uprightness: cup's local Z-axis should align with world Z
        # cup_quat is [x, y, z, w] in PyBullet but scipy uses [x, y, z, w] too
        rot = Rotation.from_quat(cup_quat)
        cup_z_axis = rot.apply([0, 0, 1])  # Cup's up direction in world frame
        uprightness = np.dot(cup_z_axis, [0, 0, 1])  # cos(tilt_angle), 1 = perfectly upright

        # Position error
        xy_dist = np.linalg.norm(cup_pos[:2] - self.TARGET_POS[:2])

        # Height check: cup should be resting on ground (center at h/2)
        target_z = self.CUP_HEIGHT / 2 + 0.002
        z_err = abs(cup_pos[2] - target_z)

        # Success criteria
        on_target = xy_dist < 0.05
        is_upright = uprightness > 0.95  # Within ~18 degrees of vertical
        at_height = z_err < 0.02
        released = self.constraint_id is None

        if on_target and is_upright and at_height and released:
            return 1.0

        reward = 0.0

        # Position proximity (0-0.25)
        reward += 0.25 * max(0, 1.0 - xy_dist / 0.3)

        # Uprightness (0-0.45): PRIMARY SIGNAL
        # uprightness ranges from -1 (upside down) to 1 (perfectly upright)
        # Map to [0, 1] reward range
        uprightness_normalized = (uprightness + 1) / 2  # Now 0 to 1
        reward += 0.45 * uprightness_normalized

        # Height (0-0.15): only if reasonably upright
        if uprightness > 0.7:
            reward += 0.15 * max(0, 1.0 - z_err / 0.1)

        # Release bonus (0.15): only if on target and upright
        if on_target and is_upright and released:
            reward += 0.15

        return np.clip(reward, 0.0, 1.0)
