"""
Genesis book-shelf insertion task.
Book and shelf; contact-based grasping.
"""

import numpy as np

from .genesis_franka_env import GenesisFrankaEnv
from .genesis_robot import _to_numpy

try:
    import genesis as gs
except ImportError:
    gs = None


class GenesisBookInsertEnv(GenesisFrankaEnv):

    BOOK_LENGTH = 0.15
    BOOK_WIDTH = 0.10
    BOOK_THICKNESS = 0.04
    HOLDER_POS = np.array([0.35, 0.35, 0.0])
    HOLDER_INNER_W = 0.14
    SHELF2_TOP_Z = 0.05 + 0.08  # second shelf top
    SPAWN_RADIUS = 0.45
    SPAWN_ANGLE_RANGE = (-np.pi / 4, np.pi / 4)

    @property
    def name(self):
        return "book_insert_genesis"

    @property
    def spawn_angle_range(self):
        return self.SPAWN_ANGLE_RANGE

    def _create_task_objects(self):
        if gs is None:
            return
        ang = self._object_rotation[-1]
        bx = self.SPAWN_RADIUS * np.cos(ang)
        by = self.SPAWN_RADIUS * np.sin(ang)
        # Book standing: local Z is height
        book = self.scene.add_entity(
            gs.morphs.Box(
                size=(self.BOOK_LENGTH, self.BOOK_WIDTH, self.BOOK_THICKNESS),
                pos=(bx, by, self.BOOK_LENGTH / 2),
                euler=np.rad2deg([0, 90, ang]),
                fixed=False,
            )
        )
        self.rigid_ids.append(book.idx if hasattr(book, "idx") else len(self.rigid_ids))
        self._rigid_graspable.append(True)
        self._rigid_entities.append(book)
        self._book_entity = book

        self._target_pos = np.array([
            self.HOLDER_POS[0], self.HOLDER_POS[1],
            self.SHELF2_TOP_Z + self.BOOK_LENGTH / 2,
        ])
        # Simple shelf: one box as placeholder
        shelf = self.scene.add_entity(
            gs.morphs.Box(
                size=(self.HOLDER_INNER_W + 0.02, 0.12, 0.25),
                pos=(self.HOLDER_POS[0], self.HOLDER_POS[1], 0.125),
                fixed=True,
            )
        )
        self.rigid_ids.append(shelf.idx if hasattr(shelf, "idx") else len(self.rigid_ids))
        self._rigid_graspable.append(False)
        self._rigid_entities.append(shelf)
        self._holder_entity = shelf

    def compute_reward(self):
        book_pos = _to_numpy(self._book_entity.get_pos())
        dx = abs(book_pos[0] - self._target_pos[0])
        dy = abs(book_pos[1] - self._target_pos[1])
        z_err = abs(book_pos[2] - self._target_pos[2])
        xy_dist = np.sqrt(dx*dx + dy*dy)
        released = self._grasped_obj_id is None
        in_holder = dx < self.HOLDER_INNER_W / 2 and dy < 0.06 and z_err < self.BOOK_LENGTH / 2
        if hasattr(self._book_entity, "get_quat"):
            from scipy.spatial.transform import Rotation
            quat = self._book_entity.get_quat()
            quat = _to_numpy(quat) if quat is not None else np.array([0,0,0,1])
            rot_mat = Rotation.from_quat(quat).as_matrix()
            book_z_axis = rot_mat[:, 2]
            verticality = abs(np.dot(book_z_axis, [0, 0, 1]))
        else:
            verticality = 0.0
        is_vertical = verticality < 0.2
        if in_holder and is_vertical and released:
            return 1.0
        reward = 0.3 * max(0, 1.0 - xy_dist / 0.3)
        reward += 0.2 * max(0, 1.0 - z_err / 0.2)
        reward += 0.2 * max(0, 1.0 - verticality)
        if in_holder and released:
            reward += 0.3
        return np.clip(reward, 0.0, 1.0)
