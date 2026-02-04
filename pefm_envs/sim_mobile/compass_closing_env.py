import os
import re
import pybullet as p
import numpy as np
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

    # find all "${...}" expressions and replace with evaluated results
    processed_content = re.sub(r"\${(.*?)}", eval_expression, content)
    with open(output_file, "w") as file:
        file.write(processed_content)


class CompassClosingEnv(BaseEnv):
    """
    4-flap box closing environment with world-frame cardinal ordering constraint.

    The box has 4 hinged flaps (left, right, back, front). The task requires
    closing them in a fixed cardinal order: North, South, East, West.
    When the box is rotated, different physical flaps map to different cardinal
    directions, creating a conflict between object-centric symmetry and
    world-frame constraints.
    """

    BASE_INIT_ROT = 0
    OTHER_BASE_INIT_ROT = np.pi

    # Fixed world-frame closure order
    CARDINAL_ORDER = ["north", "south", "east", "west"]

    # Local-frame flap normals (direction each flap faces when box is at angle 0)
    # These are the outward normals of the wall each flap is attached to
    _FLAP_LOCAL_NORMALS = {
        "flap_left": np.array([-1.0, 0.0]),   # left wall faces -X
        "flap_right": np.array([1.0, 0.0]),    # right wall faces +X
        "flap_back": np.array([0.0, 1.0]),     # back wall faces +Y
        "flap_front": np.array([0.0, -1.0]),   # front wall faces -Y
    }

    # Joint indices in the URDF (determined by link/joint ordering)
    # 0: bottom, 1: left, 2: right, 3: front, 4: back, 5: flap_left,
    # 6: flap_right, 7: flap_back, 8: flap_front
    _FLAP_JOINT_INDICES = {
        "flap_left": 5,
        "flap_right": 6,
        "flap_back": 7,
        "flap_front": 8,
    }

    @property
    def robot_config(self):
        L, W, H = self._box_size
        left_pos = np.array([-L / 2 - L * 0.35, W * 0.1, 0.005])
        right_pos = np.array([L / 2 + L * 0.35, W * 0.1, 0.005])
        init_base_pos = np.stack([left_pos, right_pos])
        init_base_pos[0, 0] -= 0.75
        init_base_pos[1, 0] += 0.75
        init_base_pos[0, 0] -= L / 3
        init_base_pos[1, 0] += L / 3
        init_base_pos = rotate_around_z(init_base_pos, self._object_rotation[-1])
        init_base_pos[:, :2] += self.scene_offset[None]
        init_base_rot = [self._object_rotation[-1] + np.pi, self._object_rotation[-1]]
        rest_arm_pos = np.array([0.75, 0.0, H * 1.0 - self.ARM_MOUNTING_HEIGHT])
        rest_arm_rot = np.array([np.pi * 0.5, np.pi, np.pi / 2])
        robots = [
            {
                "sim_robot_name": "kinova",
                "rest_base_pose": np.array(
                    [init_base_pos[0, 0], init_base_pos[0, 1], init_base_rot[0]]
                ),
                "rest_arm_pos": rest_arm_pos,
                "rest_arm_rot": rest_arm_rot,
            },
            {
                "sim_robot_name": "kinova",
                "rest_base_pose": np.array(
                    [init_base_pos[1, 0], init_base_pos[1, 1], init_base_rot[1]]
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
        cfg["yaw"] = 45
        cfg["distance"] = 1.5 * np.max(self._box_size / np.array([0.2, 0.2, 0.1]))
        cfg["target"] = [self.scene_offset[0], self.scene_offset[1], 0.1]
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
        return "compass_close"

    def visualize_anchor(self, pos):
        return super().visualize_anchor(pos + np.array([[0, 0, 0.09]]))

    def visualize_pc(self, pos):
        return super().visualize_pc(pos + np.array([[0, 0, 0.09]]))

    def _init_robots(self):
        self._box_size = np.array([0.0, 0.0, 0.0])
        super()._init_robots()

    def _reset_sim(self):
        # constants
        box_size = np.array([0.145, 0.12, 0.115]) * self._rigid_object_scale
        self._box_size = box_size
        self._box_thickness = 0.005

        super()._reset_sim()

        # add box to sim
        self.rigid_ids.append(self._init_box())
        self._rigid_graspable.append(True)

        # compute cardinal-to-flap mapping based on current rotation
        self._compute_cardinal_flap_mapping()

    def _init_box(self):
        dir_path = os.path.dirname(__file__)
        template_path = os.path.join(dir_path, "assets/compass_closing/template.urdf")
        local_vars = dict(
            L=self._box_size[0],
            W=self._box_size[1],
            H=self._box_size[2],
            T=self._box_thickness,
        )
        with NamedTemporaryFile(mode="w", suffix=".urdf") as f:
            evaluate_and_replace_expressions(template_path, f.name, local_vars)
            box_id = p.loadURDF(
                f.name,
                np.zeros([3]),
                useFixedBase=True,
                flags=p.URDF_MAINTAIN_LINK_ORDER
                | p.URDF_USE_SELF_COLLISION
                | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            )

        p.changeVisualShape(box_id, -1, rgbaColor=[0.678, 0.573, 0.439, 1.0])

        init_pos = np.array([0, 0, self._box_size[2] / 2])
        init_pos[:2] += self.scene_offset
        rotation_quaternion = p.getQuaternionFromEuler(
            [0, 0, self._object_rotation[-1]]
        )
        p.resetBasePositionAndOrientation(box_id, init_pos, rotation_quaternion)

        # 9 joints: bottom(0), left(1), right(2), front(3), back(4),
        #           flap_left(5), flap_right(6), flap_back(7), flap_front(8)
        for joint_index in range(p.getNumJoints(box_id)):
            p.changeDynamics(
                box_id,
                joint_index,
                lateralFriction=0.0,
                spinningFriction=0.0,
                rollingFriction=0.0,
                linearDamping=0.0,
                angularDamping=0.0,
            )
            p.setJointMotorControl2(
                box_id, joint_index, controlMode=p.VELOCITY_CONTROL, force=0
            )

        return box_id

    def _compute_cardinal_flap_mapping(self):
        """
        Determine which physical flap faces each cardinal direction after rotation.

        Cardinal directions in world frame:
            North = +Y, South = -Y, East = +X, West = -X
        """
        theta = self._object_rotation[-1]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        cardinal_vectors = {
            "north": np.array([0.0, 1.0]),
            "south": np.array([0.0, -1.0]),
            "east": np.array([1.0, 0.0]),
            "west": np.array([-1.0, 0.0]),
        }

        # Rotate each flap's local normal into world frame
        rotated_normals = {}
        for flap_name, local_normal in self._FLAP_LOCAL_NORMALS.items():
            rotated_normals[flap_name] = rot_mat @ local_normal

        # For each cardinal direction, find the flap whose rotated normal
        # is closest (highest dot product)
        self._cardinal_to_flap = {}
        used_flaps = set()
        for cardinal in self.CARDINAL_ORDER:
            best_flap = None
            best_dot = -np.inf
            for flap_name, world_normal in rotated_normals.items():
                if flap_name in used_flaps:
                    continue
                dot = np.dot(world_normal, cardinal_vectors[cardinal])
                if dot > best_dot:
                    best_dot = dot
                    best_flap = flap_name
            self._cardinal_to_flap[cardinal] = best_flap
            used_flaps.add(best_flap)

        # Build ordered list of joint indices for the required closure sequence
        self._ordered_flap_joints = [
            self._FLAP_JOINT_INDICES[self._cardinal_to_flap[cardinal]]
            for cardinal in self.CARDINAL_ORDER
        ]
        self._current_phase = 0

    def _get_obs(self, dummy_obs=False):
        obs = super()._get_obs(dummy_obs=dummy_obs)
        return obs

    def compute_reward(self):
        """
        Sequential reward: flaps must be closed in cardinal order (N, S, E, W).

        Each correctly-closed flap in order earns 0.25.
        The current-phase flap gets partial credit proportional to angle.
        Premature closure of future-phase flaps incurs a small penalty.
        """
        box_id = self.rigid_ids[0]
        closed_threshold = 2.8  # radians (out of pi ~= 3.14)

        total_reward = 0.0
        phase_completed = 0

        for phase_idx, joint_idx in enumerate(self._ordered_flap_joints):
            angle = p.getJointState(box_id, joint_idx)[0]

            if phase_idx < phase_completed:
                # Already scored
                continue

            if phase_idx == phase_completed:
                # Current phase: give partial credit
                if angle >= closed_threshold:
                    total_reward += 0.25
                    phase_completed += 1
                else:
                    # Partial credit proportional to angle progress
                    total_reward += 0.25 * min(angle / 3.14, 1.0)
                    break  # Can't score future phases
            else:
                # Future phase: penalize premature closure
                if angle > 1.0:
                    total_reward -= 0.1
                break

        return np.clip(total_reward, 0.0, 1.0)

    def _get_rigid_body_mesh(self, obj_id, link_index=None):
        if obj_id in self.rigid_ids:
            mesh_vertices = []
            num_links = p.getNumJoints(obj_id)
            link_idxs = list(range(num_links)) if link_index is None else [link_index]
            for link_idx in link_idxs:
                col_data = p.getCollisionShapeData(obj_id, link_idx)
                size = col_data[0][3]
                local_pos = col_data[0][5]
                link_state = p.getLinkState(obj_id, link_idx)
                pos, ori = link_state[0], link_state[1]

                get_num_pts = lambda s: (
                    20 if s > self._box_thickness * 4 else 2
                )
                for dx in np.linspace(
                    -0.5 * size[0], 0.5 * size[0], get_num_pts(size[0])
                ):
                    for dy in np.linspace(
                        -0.5 * size[1], 0.5 * size[1], get_num_pts(size[1])
                    ):
                        for dz in np.linspace(
                            -0.5 * size[2], 0.5 * size[2], get_num_pts(size[2])
                        ):
                            dxyz = tuple((np.array([dx, dy, dz]) + local_pos).tolist())
                            vertex = p.multiplyTransforms(pos, ori, dxyz, [0, 0, 0, 1])[
                                0
                            ]
                            mesh_vertices.append(vertex)
            verts = np.array(mesh_vertices)
            verts[:, :2] += self.scene_offset
            return verts
        else:
            return super()._get_rigid_body_mesh(obj_id)
