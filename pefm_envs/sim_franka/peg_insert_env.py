"""
Peg Insertion environment: C4 symmetry conflict.

A square peg with keyway spawns at a random C4 rotation and must be
grasped, reoriented, and inserted into a fixed-orientation socket.
The grasp has C4 symmetry but insertion requires world-frame alignment.
"""

import os
import re
import numpy as np
import pybullet
from tempfile import NamedTemporaryFile

from .franka_env import FrankaEnv


def evaluate_and_replace_expressions(input_file, output_file, local_vars):
    with open(input_file, "r") as f:
        content = f.read()

    def eval_expression(match):
        expression = match.group(1)
        try:
            locals().update(local_vars)
            return str(eval(expression))
        except Exception as e:
            return f"ERROR: {e}"

    processed = re.sub(r"\${(.*?)}", eval_expression, content)
    with open(output_file, "w") as f:
        f.write(processed)


class PegInsertEnv(FrankaEnv):

    PEG_SIDE = 0.04         # 4cm side (Primary)
    PEG_HEIGHT = 0.08       # 8cm height
    NOTCH_SIZE = 0.008      # 8mm keyway
    NS_SOCKET = 0.012       # 12mm notch size
    CLEARANCE = 0.002       # 2mm clearance
    WALL_THICKNESS = 0.01   # 1cm guide walls
    WALL_HEIGHT = 0.04      # 4cm wall height
    PLATE_THICKNESS = 0.005 # 5mm base plate

    # Fixed socket position (within Franka reach)
    # Note: This is the 'base' position, scene_offset will be added to this
    SOCKET_POS = np.array([0.35, -0.2, 0.0])

    # Peg spawn position (closer to robot to reduce collision risk)
    PEG_SPAWN_RADIUS = 0.4

    # C4 rotation set
    C4_ROTATIONS = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

    PEG_SPAWN_ANGLE_RANGE = (-np.pi / 3, np.pi / 3)

    @property
    def name(self):
        return "peg_insert"

    @property
    def spawn_angle_range(self):
        return self.PEG_SPAWN_ANGLE_RANGE

    @property
    def default_front_camera(self):
        return {
            "pitch": -50,
            "yaw": 10,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.35, -0.1, 0.05],
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -30,
            "yaw": 90,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.35, -0.1, 0.05],
        }

    def _create_task_objects(self):
        # Sample C4 rotation for peg
        demo_mode = getattr(self.args, "demo_mode", False)
        if demo_mode:
            self._peg_spawn_rotation = 0.0
        else:
            self._peg_spawn_rotation = self.rng.choice(self.C4_ROTATIONS)

        self._peg_id = self._create_peg()
        self._socket_id = self._create_socket()

        self.rigid_ids.append(self._peg_id)
        self._rigid_graspable.append(True)

        self.rigid_ids.append(self._socket_id)
        self._rigid_graspable.append(False)

        # Let objects settle
        for _ in range(50):
            self.sim.stepSimulation()

    def _create_peg(self):
        """Create square peg with keyway from URDF template."""
        template_dir = os.path.join(
            os.path.dirname(__file__), "..", "sim_mobile", "assets", "insertion"
        )
        template_path = os.path.join(template_dir, "peg_template.urdf")

        if not os.path.exists(template_path):
            # Fallback: simple box peg
            return self._create_simple_peg()

        local_vars = dict(
            S=self.PEG_SIDE, PH=self.PEG_HEIGHT, NS=self.NOTCH_SIZE,
        )
        with NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            tmp_path = f.name
            evaluate_and_replace_expressions(template_path, tmp_path, local_vars)

        # Spawn position based on object rotation
        # [Task Definition Fix]: Apply scene_offset to spawn center
        ang = self._object_rotation[-1]
        center_x = self.scene_offset[0]
        center_y = self.scene_offset[1]
        
        peg_x = center_x + self.PEG_SPAWN_RADIUS * np.cos(ang)
        peg_y = center_y + self.PEG_SPAWN_RADIUS * np.sin(ang)
        peg_pos = [peg_x, peg_y, 0.001]

        total_rot = self._object_rotation[-1] + self._peg_spawn_rotation
        peg_quat = pybullet.getQuaternionFromEuler([0, 0, total_rot])

        peg_id = self.sim.loadURDF(
            tmp_path, basePosition=peg_pos, baseOrientation=peg_quat,
            useFixedBase=False, flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
        )
        os.unlink(tmp_path)

        pybullet.changeVisualShape(peg_id, -1, rgbaColor=[0.4, 0.6, 0.8, 1.0],
                                   physicsClientId=self.sim._client)
        self.sim.changeDynamics(peg_id, -1, lateralFriction=1.0, mass=0.1)
        return peg_id

    def _create_simple_peg(self):
        """Fallback: Rectangular peg for visual orientation if assets missing."""
        # [Task Definition Fix]: Make fallback non-square (0.04 x 0.03) so orientation 
        # is visually observable. This prevents the 'invisible goal' problem where
        # a square peg at 90deg looks correct but is penalized by reward.
        
        side_x = self.PEG_SIDE
        side_y = self.PEG_SIDE * 0.75 # 3cm
        
        half = [side_x / 2, side_y / 2, self.PEG_HEIGHT / 2]
        col = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half)
        vis = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.4, 0.6, 0.8, 1.0],
        )
        
        # Apply scene offset
        ang = self._object_rotation[-1]
        center_x = self.scene_offset[0]
        center_y = self.scene_offset[1]
        peg_x = center_x + self.PEG_SPAWN_RADIUS * np.cos(ang)
        peg_y = center_y + self.PEG_SPAWN_RADIUS * np.sin(ang)
        
        total_rot = self._object_rotation[-1] + self._peg_spawn_rotation
        peg_quat = pybullet.getQuaternionFromEuler([0, 0, total_rot])
        
        peg_id = self.sim.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[peg_x, peg_y, self.PEG_HEIGHT / 2 + 0.001],
            baseOrientation=peg_quat,
        )
        self.sim.changeDynamics(peg_id, -1, lateralFriction=1.0)
        return peg_id

    def _create_socket(self):
        """Create socket at FIXED world position + Offset."""
        template_dir = os.path.join(
            os.path.dirname(__file__), "..", "sim_mobile", "assets", "insertion"
        )
        template_path = os.path.join(template_dir, "socket_template.urdf")

        # Apply scene offset to socket position
        socket_pos = [
            self.SOCKET_POS[0] + self.scene_offset[0], 
            self.SOCKET_POS[1] + self.scene_offset[1], 
            0.0
        ]
        socket_quat = pybullet.getQuaternionFromEuler([0, 0, 0])

        if not os.path.exists(template_path):
            return self._create_simple_socket(socket_pos)

        local_vars = dict(
            S=self.PEG_SIDE, CL=self.CLEARANCE, WT=self.WALL_THICKNESS,
            WH=self.WALL_HEIGHT, PT=self.PLATE_THICKNESS, NS=self.NS_SOCKET,
        )
        with NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            tmp_path = f.name
            evaluate_and_replace_expressions(template_path, tmp_path, local_vars)

        socket_id = self.sim.loadURDF(
            tmp_path, basePosition=socket_pos, baseOrientation=socket_quat,
            useFixedBase=True, flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
        )
        os.unlink(tmp_path)

        for link_idx in range(-1, self.sim.getNumJoints(socket_id)):
            pybullet.changeVisualShape(
                socket_id, link_idx, rgbaColor=[0.5, 0.5, 0.5, 1.0],
                physicsClientId=self.sim._client,
            )
        return socket_id

    def _create_simple_socket(self, pos):
        """Fallback: simple box socket."""
        inner = self.PEG_SIDE + self.CLEARANCE * 2
        outer = inner + self.WALL_THICKNESS * 2
        half = [outer / 2, outer / 2, self.PLATE_THICKNESS / 2]
        col = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half)
        vis = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
        )
        socket_id = self.sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[pos[0], pos[1], self.PLATE_THICKNESS / 2],
        )
        return socket_id

    def compute_reward(self):
        """
        Reward: XY proximity + rotation alignment + descent depth.
        """
        peg_pos, peg_quat = self.sim.getBasePositionAndOrientation(self._peg_id)
        peg_pos = np.array(peg_pos)
        peg_euler = np.array(pybullet.getEulerFromQuaternion(peg_quat))

        # [Task Definition Fix]: Use actual socket position from Sim (handles randomization)
        socket_pos, _ = self.sim.getBasePositionAndOrientation(self._socket_id)
        socket_center = np.array(socket_pos)[:2]
        
        xy_dist = np.linalg.norm(peg_pos[:2] - socket_center)
        pos_threshold = self.PEG_SIDE * 0.5

        socket_top_z = self.PLATE_THICKNESS + self.WALL_HEIGHT
        descended = peg_pos[2] < socket_top_z

        # Orientation: key should face +X (z_rot ≈ 0)
        # Note: We check absolute error from 0. The fallback peg is rectangular
        # so it physically requires 0 or 180, which aligns with this reward logic.
        z_rot = np.mod(peg_euler[2] + np.pi, 2 * np.pi) - np.pi
        rot_error = abs(z_rot)
        rot_aligned = rot_error < np.deg2rad(15)

        if xy_dist < pos_threshold and descended and rot_aligned:
            return 1.0

        reward = 0.0
        reward += 0.3 * max(0, 1.0 - xy_dist / 0.3)

        if xy_dist < pos_threshold * 2:
            descent_progress = max(0, 1.0 - peg_pos[2] / (socket_top_z * 2.0))
            reward += 0.3 * descent_progress

        rot_progress = max(0, 1.0 - rot_error / np.pi)
        reward += 0.2 * rot_progress

        return np.clip(reward, 0.0, 1.0)