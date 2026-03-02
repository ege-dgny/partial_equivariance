"""
Book-Shelf Insertion environment: SO(2) symmetry conflict.

A book (flat box) spawns vertically on the table at a random yaw.
The robot must grasp it, carry it to a fixed-position bookcase, and
insert it into the upper shelf slot.  The grasp phase has SO(2)
symmetry; the shelf placement breaks it (fixed world-frame position
and orientation).
"""

import numpy as np
import pybullet
from scipy.spatial.transform import Rotation

from .franka_env import FrankaEnv


def _transform_point_to_local(sim, body_id, point_world):
    """Transform a world-frame point into the body's local frame (base)."""
    pos, ori = sim.getBasePositionAndOrientation(body_id)
    inv_pos, inv_ori = sim.invertTransform(pos, ori)
    local_pos, _ = sim.multiplyTransforms(inv_pos, inv_ori, point_world, (0, 0, 0, 1))
    return np.array(local_pos)


class BookInsertEnv(FrankaEnv):

    # Book dimensions (local frame: X=length, Y=width, Z=thickness)
    BOOK_LENGTH = 0.15     # 15cm — becomes height when standing
    BOOK_WIDTH = 0.10      # 10cm
    BOOK_THICKNESS = 0.04  # 4cm

    # Bookcase dimensions
    CASE_WIDTH = 0.15       # 15cm wide (Y direction); inner ≈ 13cm fits 10cm book
    CASE_DEPTH = 0.18       # 18cm deep (X direction, insertion axis)
    CASE_HEIGHT = 0.40      # 40cm tall
    SHELF_THICKNESS = 0.01  # 1cm thick shelves
    SIDE_THICKNESS = 0.01   # 1cm thick sides
    BACK_THICKNESS = 0.005  # 5mm back panel
    NUM_SHELVES = 2         # 2 slots (bottom + top)

    # Fixed bookcase position — to the left of robot (+Y), opening faces -Y
    CASE_POS = np.array([0.45, 0.45, 0.0])

    # Target: upper shelf slot
    TARGET_SHELF_IDX = 1  # 0=lower, 1=upper

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
            "pitch": -40,
            "yaw": -15,
            "roll": 0,
            "distance": 1.3,
            "fov": 45,
            "target": [0.40, 0.20, 0.15],
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -25,
            "yaw": -70,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.40, 0.25, 0.15],
        }

    def _create_task_objects(self):
        # Disable snap-to-center: spine-edge grasp needs the offset preserved
        self.grasp_snap_to_center = False
        # Side-grip places fingers inside the book body (collisions disabled),
        # so the closest mesh *surface* vertex is ~4cm away.  Default 3cm
        # threshold would delay constraint creation until the lift phase.
        self.grasp_attach_max_dist = 0.06

        # Initialize book spawn rotation like peg_insert: set/resample in task
        # so --randomize_rotation is applied here regardless of base reset order.
        if self.randomize_rotation:
            lo, hi = self.spawn_angle_range
            self._object_rotation[-1] = self.rng.rand() * (hi - lo) + lo

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

    def _is_grasp_valid(self, obj_id, closest_point_world):
        """Only allow grasp when the closest point is on the spine (side-grip).

        Spine is at local Y = -BOOK_WIDTH/2. Rejects attachments triggered by
        top/side faces so the book does not attach to a random part of the hand.
        """
        if obj_id != self._book_id:
            return True
        if closest_point_world is None:
            return False
        local_pt = _transform_point_to_local(self.sim, obj_id, closest_point_world)
        spine_y = -self.BOOK_WIDTH / 2
        tolerance = 0.025  # ~2.5 cm: allow spine edge and nearby binding
        return abs(local_pt[1] - spine_y) <= tolerance

    def _snap_book_to_grasp(self, obj_id):
        """Snap book to canonical side-grip: spine at finger midpoint, standing."""
        finger_mid = self._get_grasp_reference_pos()
        _, ee_quat, _, _ = self.robot.get_ee_pos_quat_vel()
        rot_mat = np.array(pybullet.getMatrixFromQuaternion(ee_quat)).reshape(3, 3)
        hand_y = rot_mat[:, 1]  # hand Y = approach direction (wrist → fingers)

        # Spine at finger midpoint; book center = spine + (BOOK_WIDTH/2) * book_y_axis
        book_y_axis = hand_y.copy()
        book_y_axis[2] = 0
        n = np.linalg.norm(book_y_axis)
        if n < 1e-6:
            book_y_axis = np.array([1.0, 0.0, 0.0])
        else:
            book_y_axis /= n
        book_center = finger_mid + (self.BOOK_WIDTH / 2) * book_y_axis

        # Standing: book X = world Z, book Y = book_y_axis, book Z = X × Y
        book_x_axis = np.array([0.0, 0.0, 1.0])
        book_z_axis = np.cross(book_x_axis, book_y_axis)
        book_z_axis /= np.linalg.norm(book_z_axis)
        book_y_axis = np.cross(book_z_axis, book_x_axis)
        rot = np.column_stack([book_x_axis, book_y_axis, book_z_axis])
        r = Rotation.from_matrix(rot)
        quat_xyzw = r.as_quat()
        book_quat = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])

        self.sim.resetBasePositionAndOrientation(
            obj_id, book_center.tolist(), book_quat
        )

    def _lock_object_in_place(self, obj_id):
        """For the book, snap to side-grip pose first so it stays between the fingers."""
        if obj_id == self._book_id:
            self._snap_book_to_grasp(obj_id)
        super()._lock_object_in_place(obj_id)

    def _create_book(self):
        """Create book standing vertically with random Z-rotation.

        Compound body: base = blue book box, link 0 = dark-red spine
        visual overlay at local Y = -WIDTH/2 (the bound edge).
        """
        half = [self.BOOK_LENGTH / 2, self.BOOK_WIDTH / 2, self.BOOK_THICKNESS / 2]

        col = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half)
        vis = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.2, 0.3, 0.7, 1.0],  # Dark blue book
        )

        # Spine visual: thin colored strip on the bound edge
        spine_half = [self.BOOK_LENGTH / 2, 0.003,
                      self.BOOK_THICKNESS / 2 + 0.001]
        vis_spine = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=spine_half,
            rgbaColor=[0.6, 0.1, 0.1, 1.0],  # Dark red spine
        )

        # Spawn on arc
        ang = self._object_rotation[-1]
        spawn_x = self.SPAWN_RADIUS * np.cos(ang)
        spawn_y = self.SPAWN_RADIUS * np.sin(ang)
        # Standing: pitch=π/2 rotates local X (length) to world Z
        pos = [spawn_x, spawn_y, self.BOOK_LENGTH / 2]
        quat = pybullet.getQuaternionFromEuler([0, np.pi / 2, ang])

        book_id = self.sim.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=quat,
            # Spine visual as fixed child link (no collision)
            linkMasses=[0.001],
            linkCollisionShapeIndices=[-1],
            linkVisualShapeIndices=[vis_spine],
            linkPositions=[[0, -self.BOOK_WIDTH / 2, 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[pybullet.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]],
        )
        self.sim.changeDynamics(book_id, -1, lateralFriction=1.5)
        return book_id

    def _create_bookcase(self):
        """Create bookcase with opening facing -Y (toward robot).

        Structure: 2 side panels (along X) + back panel (+Y side)
                 + (NUM_SHELVES+1) shelves.
        Robot inserts the book along +Y into the opening.
        """
        ids = []
        cx, cy = self.CASE_POS[0], self.CASE_POS[1]
        w = self.CASE_WIDTH    # X span (visible width)
        d = self.CASE_DEPTH    # Y span (depth, insertion axis)
        h = self.CASE_HEIGHT
        st = self.SIDE_THICKNESS
        sht = self.SHELF_THICKNESS
        bt = self.BACK_THICKNESS

        inner_w = w - 2 * st  # internal X span

        # Left side panel (negative X)
        side_half = [st / 2, d / 2, h / 2]
        col = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=side_half)
        vis = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=side_half,
            rgbaColor=[0.55, 0.35, 0.2, 1.0],
        )
        left_x = cx - w / 2 + st / 2
        left_id = self.sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[left_x, cy, h / 2],
        )
        ids.append(left_id)

        # Right side panel (positive X)
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

        # Back panel (+Y side, farthest from robot)
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

        # Distance components
        dx = abs(book_pos[0] - self._target_pos[0])
        dy = abs(book_pos[1] - self._target_pos[1])
        z_err = abs(book_pos[2] - self._target_pos[2])
        xy_dist = np.sqrt(dx**2 + dy**2)

        # Book orientation: local Z (thickness axis) should be horizontal
        rot_mat = np.array(self.sim.getMatrixFromQuaternion(book_quat)).reshape(3, 3)
        book_z_axis = rot_mat[:, 2]
        verticality = abs(np.dot(book_z_axis, [0, 0, 1]))
        is_vertical = verticality < 0.2

        released = self.constraint_id is None
        in_slot = (
            dx < (self.CASE_WIDTH - 2 * self.SIDE_THICKNESS) / 2
            and dy < self.CASE_DEPTH / 2
            and z_err < self._shelf_slot_height / 2
        )

        if in_slot and is_vertical and released:
            return 1.0

        reward = 0.0
        # XY proximity (0-0.3)
        reward += 0.3 * max(0, 1.0 - xy_dist / 0.3)
        # Z height (0-0.2)
        reward += 0.2 * max(0, 1.0 - z_err / 0.2)
        # Verticality (0-0.2)
        reward += 0.2 * max(0, 1.0 - verticality)
        # Release bonus (0.3): only if in slot
        if in_slot and released:
            reward += 0.3

        return np.clip(reward, 0.0, 1.0)
