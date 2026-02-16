"""
Book-Shelf Insertion environment: SO(2) symmetry conflict.

A book (flat box) spawns on the table at a random orientation.
The robot must grasp it, reorient it to vertical, and insert it
into a fixed-position bookcase shelf. The grasp phase has SO(2)
symmetry; the shelf placement breaks it (fixed world-frame position
and orientation).
"""

import numpy as np
import pybullet

from .franka_env import FrankaEnv


class BookInsertEnv(FrankaEnv):

    # Book dimensions (laid flat on table)
    BOOK_LENGTH = 0.15     # 15cm (height when vertical)
    BOOK_WIDTH = 0.10      # 10cm
    BOOK_THICKNESS = 0.02  # 2cm

    # Bookcase dimensions
    CASE_WIDTH = 0.25       # 25cm wide
    CASE_DEPTH = 0.18       # 18cm deep
    CASE_HEIGHT = 0.30      # 30cm tall
    SHELF_THICKNESS = 0.01  # 1cm thick shelves
    SIDE_THICKNESS = 0.01   # 1cm thick sides
    BACK_THICKNESS = 0.005  # 5mm back panel
    NUM_SHELVES = 3         # Bottom, middle, top

    # Fixed bookcase position (within Franka reach)
    CASE_POS = np.array([0.45, -0.20, 0.0])

    # Target: middle shelf slot
    TARGET_SHELF_IDX = 1  # 0-indexed (bottom=0, middle=1, top=2)

    # Book spawn
    SPAWN_RADIUS = 0.45
    SPAWN_ANGLE_RANGE = (-np.pi / 4, np.pi / 4)

    @property
    def name(self):
        return "book_insert"

    @property
    def spawn_angle_range(self):
        return self.SPAWN_ANGLE_RANGE

    @property
    def default_front_camera(self):
        return {
            "pitch": -45,
            "yaw": 15,
            "roll": 0,
            "distance": 1.3,
            "fov": 45,
            "target": [0.35, -0.05, 0.1],
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -30,
            "yaw": 90,
            "roll": 0,
            "distance": 1.3,
            "fov": 45,
            "target": [0.35, -0.05, 0.1],
        }

    def _create_task_objects(self):
        self._book_id = self._create_book()
        self._case_ids = self._create_bookcase()

        self.rigid_ids.append(self._book_id)
        self._rigid_graspable.append(True)

        # Bookcase parts are not graspable
        for cid in self._case_ids:
            self.rigid_ids.append(cid)
            self._rigid_graspable.append(False)

        # Record spawn position
        pos, _ = self.sim.getBasePositionAndOrientation(self._book_id)
        self._spawn_pos = np.array(pos)

        # Compute target shelf position (center of the target slot)
        shelf_spacing = (self.CASE_HEIGHT - self.SHELF_THICKNESS * (self.NUM_SHELVES + 1)) / self.NUM_SHELVES
        slot_bottom = self.SHELF_THICKNESS + self.TARGET_SHELF_IDX * (shelf_spacing + self.SHELF_THICKNESS)
        slot_center_z = slot_bottom + shelf_spacing / 2
        self._target_pos = np.array([
            self.CASE_POS[0],
            self.CASE_POS[1],
            slot_center_z,
        ])
        self._shelf_slot_height = shelf_spacing

        # Let objects settle
        for _ in range(50):
            self.sim.stepSimulation()

    def _create_book(self):
        """Create flat book on table with random Z-rotation."""
        # Book lies flat: half-extents are (length/2, width/2, thickness/2)
        half = [self.BOOK_LENGTH / 2, self.BOOK_WIDTH / 2, self.BOOK_THICKNESS / 2]

        col = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half)
        vis = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.2, 0.3, 0.7, 1.0],  # Dark blue book
        )

        # Spawn on arc
        ang = self._object_rotation[-1]
        spawn_x = self.SPAWN_RADIUS * np.cos(ang)
        spawn_y = self.SPAWN_RADIUS * np.sin(ang)
        pos = [spawn_x, spawn_y, self.BOOK_THICKNESS / 2 + 0.001]
        quat = pybullet.getQuaternionFromEuler([0, 0, ang])

        book_id = self.sim.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=quat,
        )
        self.sim.changeDynamics(book_id, -1, lateralFriction=0.8)
        return book_id

    def _create_bookcase(self):
        """Create a simple bookcase with shelves at fixed world position.

        Structure: 2 sides + (NUM_SHELVES+1) shelves + back panel.
        Opening faces the robot (negative Y direction from case center).
        """
        ids = []
        cx, cy = self.CASE_POS[0], self.CASE_POS[1]
        w = self.CASE_WIDTH
        d = self.CASE_DEPTH
        h = self.CASE_HEIGHT
        st = self.SIDE_THICKNESS
        sht = self.SHELF_THICKNESS
        bt = self.BACK_THICKNESS

        # Left side panel
        side_half = [st / 2, d / 2, h / 2]
        col = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=side_half)
        vis = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=side_half,
            rgbaColor=[0.55, 0.35, 0.2, 1.0],  # Wood brown
        )
        left_x = cx - w / 2 + st / 2
        left_id = self.sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[left_x, cy, h / 2],
        )
        ids.append(left_id)

        # Right side panel
        right_x = cx + w / 2 - st / 2
        col2 = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=side_half)
        vis2 = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=side_half,
            rgbaColor=[0.55, 0.35, 0.2, 1.0],
        )
        right_id = self.sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col2,
            baseVisualShapeIndex=vis2,
            basePosition=[right_x, cy, h / 2],
        )
        ids.append(right_id)

        # Back panel
        inner_w = w - 2 * st
        back_half = [inner_w / 2, bt / 2, h / 2]
        col_b = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=back_half)
        vis_b = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=back_half,
            rgbaColor=[0.55, 0.35, 0.2, 1.0],
        )
        back_y = cy + d / 2 - bt / 2
        back_id = self.sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_b,
            baseVisualShapeIndex=vis_b,
            basePosition=[cx, back_y, h / 2],
        )
        ids.append(back_id)

        # Shelves (bottom + internal + top)
        shelf_half = [inner_w / 2, d / 2, sht / 2]
        shelf_spacing = (h - sht * (self.NUM_SHELVES + 1)) / self.NUM_SHELVES
        for s in range(self.NUM_SHELVES + 1):
            shelf_z = sht / 2 + s * (shelf_spacing + sht)
            col_s = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=shelf_half)
            vis_s = self.sim.createVisualShape(
                pybullet.GEOM_BOX, halfExtents=shelf_half,
                rgbaColor=[0.6, 0.4, 0.25, 1.0],
            )
            shelf_id = self.sim.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_s,
                baseVisualShapeIndex=vis_s,
                basePosition=[cx, cy, shelf_z],
            )
            ids.append(shelf_id)

        return ids

    def compute_reward(self):
        """
        Reward: book proximity to shelf slot + correct orientation + released.

        Full success (1.0): book inserted into shelf slot, vertical, released.
        Shaping: XY proximity (0.3) + Z height (0.2) + vertical orientation (0.2) + release (0.3).
        """
        book_pos, book_quat = self.sim.getBasePositionAndOrientation(self._book_id)
        book_pos = np.array(book_pos)

        # XY distance to shelf center
        target_xy = self._target_pos[:2]
        xy_dist = np.linalg.norm(book_pos[:2] - target_xy)

        # Z height at shelf level
        target_z = self._target_pos[2]
        z_err = abs(book_pos[2] - target_z)

        # Book orientation: should be vertical (book's local Z axis ≈ world Y or X)
        # When the book is flat, its local Z axis points up (world Z).
        # When inserted vertically, the local Z should be horizontal.
        rot_mat = np.array(self.sim.getMatrixFromQuaternion(book_quat)).reshape(3, 3)
        book_z_axis = rot_mat[:, 2]  # Local Z in world frame
        # Vertical means book_z_axis dot world_z ≈ 0
        verticality = abs(np.dot(book_z_axis, [0, 0, 1]))
        is_vertical = verticality < 0.2  # Nearly horizontal local Z

        released = self.constraint_id is None
        in_slot = xy_dist < 0.05 and z_err < self._shelf_slot_height / 2

        if in_slot and is_vertical and released:
            return 1.0

        reward = 0.0
        # XY proximity (0-0.3)
        reward += 0.3 * max(0, 1.0 - xy_dist / 0.3)
        # Z height (0-0.2)
        reward += 0.2 * max(0, 1.0 - z_err / 0.2)
        # Verticality (0-0.2): lower verticality = more vertical book
        reward += 0.2 * max(0, 1.0 - verticality)
        # Release bonus (0.3): only if in slot
        if in_slot and released:
            reward += 0.3

        return np.clip(reward, 0.0, 1.0)
