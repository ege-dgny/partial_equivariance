import os
import re
import numpy as np
import pybullet as p
from tempfile import NamedTemporaryFile

from .base_env import BaseEnv
from .utils.init_utils import rotate_around_z


def evaluate_and_replace_expressions(input_file, output_file, local_vars):
    with open(input_file, "r") as file:
        content = file.read()

    def eval_expression(match):
        expression = match.group(1)
        try:
            locals().update(local_vars)
            return str(eval(expression))
        except Exception as e:
            return f"ERROR: {e}"

    processed_content = re.sub(r"\${(.*?)}", eval_expression, content)
    with open(output_file, "w") as file:
        file.write(processed_content)


class InsertionEnv(BaseEnv):
    """
    Polarized insertion environment demonstrating discrete symmetry breaking.

    A square peg with a keyway protrusion (breaking C4 to C1) must be grasped,
    reoriented, and inserted into a fixed-orientation socket. The peg spawns
    at a random rotation from {0, pi/2, pi, 3pi/2} (C4 symmetry set), while
    the socket keyway always faces world +X.

    The grasp phase has C4 discrete symmetry, but the insertion phase requires
    aligning the key to a world-fixed orientation (C1), creating the symmetry
    conflict that PEFM is designed to handle.
    """

    PEG_SIDE = 0.04       # 4cm side length
    PEG_HEIGHT = 0.06      # 6cm height
    NOTCH_SIZE = 0.008     # 8mm keyway protrusion
    CLEARANCE = 0.002      # 2mm clearance
    WALL_THICKNESS = 0.01  # 1cm guide walls
    WALL_HEIGHT = 0.04     # 4cm wall height
    PLATE_THICKNESS = 0.005  # 5mm base plate

    SOCKET_WORLD_POS = np.array([0.15, 0.0, 0.0])  # FIXED in world frame

    # C4 rotation set
    C4_ROTATIONS = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

    @property
    def robot_config(self):
        # Robot 0: near peg (object-relative)
        r0_pos = np.array([-0.75, 0.0, 0.005])
        r0_pos = rotate_around_z(r0_pos, self._object_rotation[-1])
        r0_pos[:2] += self.scene_offset

        # Robot 1: near socket (world-frame, idle)
        r1_pos = np.array([self.SOCKET_WORLD_POS[0] + 0.75, self.SOCKET_WORLD_POS[1], 0.005])

        rest_arm_pos = np.array([0.75, 0.0, self.PEG_HEIGHT * 2.0 - self.ARM_MOUNTING_HEIGHT])
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
        cfg["pitch"] = -50
        cfg["yaw"] = 30
        cfg["distance"] = 1.2
        cfg["target"] = [0.08, 0.0, 0.05]
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
        return "insert"

    def _reset_sim(self):
        # Sample a C4 rotation for the peg
        self._peg_spawn_rotation = self.rng.choice(self.C4_ROTATIONS)

        super()._reset_sim()

        # Create peg and socket
        self._peg_id = self._create_peg()
        self._socket_id = self._create_socket()

        # Register peg as graspable
        self.rigid_ids.append(self._peg_id)
        self._rigid_graspable.append(True)

        # Register socket (not graspable)
        self.rigid_ids.append(self._socket_id)
        self._rigid_graspable.append(False)

        # Let objects settle
        for _ in range(50):
            self.sim.stepSimulation()

    def _create_peg(self):
        """Create a square peg with keyway protrusion from URDF template."""
        dir_path = os.path.dirname(__file__)
        template_path = os.path.join(dir_path, "assets/insertion/peg_template.urdf")
        local_vars = dict(
            S=self.PEG_SIDE,
            PH=self.PEG_HEIGHT,
            NS=self.NOTCH_SIZE,
        )
        with NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            tmp_path = f.name
            evaluate_and_replace_expressions(template_path, tmp_path, local_vars)

        # Spawn position: scene center + offset
        peg_pos = np.array([0.0, 0.0, 0.001])
        peg_pos[:2] += self.scene_offset

        # Apply C4 rotation
        total_rotation = self._object_rotation[-1] + self._peg_spawn_rotation
        peg_quat = p.getQuaternionFromEuler([0, 0, total_rotation])

        peg_id = p.loadURDF(
            tmp_path,
            basePosition=peg_pos.tolist(),
            baseOrientation=peg_quat,
            useFixedBase=False,
            flags=p.URDF_MAINTAIN_LINK_ORDER,
        )

        os.unlink(tmp_path)

        # Set visual color
        p.changeVisualShape(peg_id, -1, rgbaColor=[0.4, 0.6, 0.8, 1.0])
        p.changeVisualShape(peg_id, 0, rgbaColor=[0.8, 0.4, 0.2, 1.0])  # key colored differently

        # Set friction
        p.changeDynamics(peg_id, -1, lateralFriction=1.0, mass=0.1)

        return peg_id

    def _create_socket(self):
        """Create socket at FIXED world-frame position with keyway facing +X."""
        dir_path = os.path.dirname(__file__)
        template_path = os.path.join(dir_path, "assets/insertion/socket_template.urdf")
        local_vars = dict(
            S=self.PEG_SIDE,
            CL=self.CLEARANCE,
            WT=self.WALL_THICKNESS,
            WH=self.WALL_HEIGHT,
            PT=self.PLATE_THICKNESS,
            NS=self.NOTCH_SIZE,
        )
        with NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            tmp_path = f.name
            evaluate_and_replace_expressions(template_path, tmp_path, local_vars)

        # Socket is ALWAYS at world-frame position (not affected by scene_offset)
        socket_pos = [
            self.SOCKET_WORLD_POS[0],
            self.SOCKET_WORLD_POS[1],
            0.0,
        ]

        # Socket keyway always faces +X (angle 0)
        socket_quat = p.getQuaternionFromEuler([0, 0, 0])

        socket_id = p.loadURDF(
            tmp_path,
            basePosition=socket_pos,
            baseOrientation=socket_quat,
            useFixedBase=True,
            flags=p.URDF_MAINTAIN_LINK_ORDER,
        )

        os.unlink(tmp_path)

        # Set visual color
        for link_idx in range(-1, p.getNumJoints(socket_id)):
            p.changeVisualShape(socket_id, link_idx, rgbaColor=[0.5, 0.5, 0.5, 1.0])

        return socket_id

    def compute_reward(self):
        """
        Reward based on position, descent, and orientation alignment.

        Position (0.3): peg XY near socket center
        Descent (0.3): peg Z below socket wall height
        Orientation (0.2): peg Z-rotation aligned to 0 (key facing +X)
        Full success (1.0): all three pass strict checks
        """
        peg_pos, peg_quat = self.sim.getBasePositionAndOrientation(self._peg_id)
        peg_pos = np.array(peg_pos)
        peg_euler = np.array(p.getEulerFromQuaternion(peg_quat))

        socket_center = np.array([
            self.SOCKET_WORLD_POS[0],
            self.SOCKET_WORLD_POS[1],
        ])

        # Position check
        xy_dist = np.linalg.norm(peg_pos[:2] - socket_center)
        pos_threshold = self.PEG_SIDE * 0.5

        # Descent check
        socket_top_z = self.PLATE_THICKNESS + self.WALL_HEIGHT
        descended = peg_pos[2] < socket_top_z * 0.3

        # Orientation check: peg Z-rotation should be near 0 (mod pi/2)
        # The key should face +X, so total Z rotation should be ~0
        z_rot = peg_euler[2]
        # Normalize to [-pi, pi]
        z_rot = np.mod(z_rot + np.pi, 2 * np.pi) - np.pi
        rot_error = abs(z_rot)  # distance from 0
        rot_aligned = rot_error < np.deg2rad(10)

        # Full success
        if xy_dist < pos_threshold and descended and rot_aligned:
            return 1.0

        # Shaping reward
        reward = 0.0

        # Position proximity (0-0.3)
        reward += 0.3 * max(0, 1.0 - xy_dist / 0.3)

        # Descent progress (0-0.3)
        if xy_dist < pos_threshold * 2:
            descent_progress = max(0, 1.0 - peg_pos[2] / (socket_top_z * 1.5))
            reward += 0.3 * descent_progress

        # Rotation alignment (0-0.2)
        rot_progress = max(0, 1.0 - rot_error / np.pi)
        reward += 0.2 * rot_progress

        return np.clip(reward, 0.0, 1.0)
