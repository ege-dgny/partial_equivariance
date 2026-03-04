"""
Book-Shelf Insertion environment: SO(2) symmetry conflict.

A book (flat box) spawns vertically on the table at a random yaw.
The robot must side-grasp it, carry it to a fixed-position shelf unit,
and place it in the upper compartment (on the 2nd of 3 horizontal shelves).
The grasp phase has SO(2) symmetry; the shelf placement breaks it (fixed world frame).

Shelf: 2 side walls + 3 horizontal shelves (2 compartments) + 1 back wall.
Modelled after cup_pour's side-grip pattern for reliable grasping.
"""

import numpy as np
import pybullet

from .franka_env import FrankaEnv


class BookInsertEnv(FrankaEnv):

    # Book dimensions (local frame: X=length, Y=width, Z=thickness)
    BOOK_LENGTH = 0.15     # 15cm — becomes height when standing
    BOOK_WIDTH = 0.10      # 10cm
    BOOK_THICKNESS = 0.04  # 4cm

    # Shelf unit: 2 side walls, 3 horizontal shelves (2 compartments), 1 back wall
    HOLDER_POS = np.array([0.35, 0.35, 0.0])  # XY of shelf unit
    HOLDER_INNER_W = 0.14   # inner width (book 10cm + clearance)
    HOLDER_DEPTH = 0.10     # depth
    WALL_T = 0.01           # wall/shelf thickness

    # Shelf levels: 3 horizontal shelves → 2 compartments
    # Compartment 1 (lower): shelf1 top to shelf2 top
    # Compartment 2 (upper): shelf2 top to shelf3 top — book goes here
    SHELF_THICKNESS = 0.01
    SHELF1_TOP_Z = 0.05                              # first shelf top
    COMPARTMENT1_HEIGHT = 0.08                        # gap for lower compartment
    SHELF2_TOP_Z = SHELF1_TOP_Z + COMPARTMENT1_HEIGHT # second shelf = book placement surface
    BOOK_CLEARANCE = 0.03                             # clearance above book in upper compartment
    SHELF3_TOP_Z = SHELF2_TOP_Z + BOOK_LENGTH + BOOK_CLEARANCE  # third shelf top (clears book)

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
            "target": [0.35, 0.15, 0.18],  # between lower and upper compartment
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -25,
            "yaw": -70,
            "roll": 0,
            "distance": 1.2,
            "fov": 45,
            "target": [0.35, 0.20, 0.18],
        }

    def _create_task_objects(self):
        # Cup_pour-style grasp parameters: fire early, permissive
        self.grasp_snap_to_center = True
        self.grasp_min_closed_dist = 0.07
        self.grasp_attach_max_dist = 0.05

        # Initialize book spawn rotation
        if self.randomize_rotation:
            lo, hi = self.spawn_angle_range
            self._object_rotation[-1] = self.rng.rand() * (hi - lo) + lo

        self._book_id = self._create_book()
        self._holder_ids = self._create_holder()

        self.rigid_ids.append(self._book_id)
        self._rigid_graspable.append(True)

        for hid in self._holder_ids:
            self.rigid_ids.append(hid)
            self._rigid_graspable.append(False)

        # Record spawn position
        pos, _ = self.sim.getBasePositionAndOrientation(self._book_id)
        self._spawn_pos = np.array(pos)

        # Target: holder center XY, book standing on 2nd shelf
        self._target_pos = np.array([
            self.HOLDER_POS[0],
            self.HOLDER_POS[1],
            self.SHELF2_TOP_Z + self.BOOK_LENGTH / 2,
        ])

        # Let objects settle
        for _ in range(50):
            self.sim.stepSimulation()

    def _snap_object_to_grasp(self, obj_id):
        """Snap book center XY to grasp ref point, preserve standing orientation."""
        if obj_id != self._book_id:
            return super()._snap_object_to_grasp(obj_id)

        grasp_center = self._get_grasp_reference_pos()
        obj_pos, obj_ori = self.sim.getBasePositionAndOrientation(obj_id)

        # Center book XY on grasp reference, keep Z and full orientation
        snapped_pos = [grasp_center[0], grasp_center[1], obj_pos[2]]
        self.sim.resetBasePositionAndOrientation(obj_id, snapped_pos, obj_ori)

    def _disable_graspable_collisions(self):
        """Disable robot <-> book (all links) and robot <-> holder."""
        robot_id = self.robot.info.robot_id
        num_robot_links = self.sim.getNumJoints(robot_id)
        num_book_links = self.sim.getNumJoints(self._book_id)

        # Robot <-> book (base + all child links)
        for robot_link in range(-1, num_robot_links):
            for book_link in range(-1, num_book_links):
                self.sim.setCollisionFilterPair(
                    robot_id, self._book_id, robot_link, book_link, 0
                )

        # Robot <-> holder walls (so hand can enter the holder)
        for hid in self._holder_ids:
            num_h_links = self.sim.getNumJoints(hid)
            for robot_link in range(-1, num_robot_links):
                for h_link in range(-1, num_h_links):
                    self.sim.setCollisionFilterPair(
                        robot_id, hid, robot_link, h_link, 0
                    )

    def _create_book(self):
        """Create book standing vertically with random Z-rotation.

        Compound body: base = blue book box, link 0 = dark-red spine
        visual overlay at local Y = -WIDTH/2 (the bound edge).
        """
        half = [self.BOOK_LENGTH / 2, self.BOOK_WIDTH / 2, self.BOOK_THICKNESS / 2]

        col = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half)
        vis = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.2, 0.3, 0.7, 1.0],
        )

        # Spine visual: thin colored strip on the bound edge
        spine_half = [self.BOOK_LENGTH / 2, 0.003,
                      self.BOOK_THICKNESS / 2 + 0.001]
        vis_spine = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=spine_half,
            rgbaColor=[0.6, 0.1, 0.1, 1.0],
        )

        # Spawn on arc
        ang = self._object_rotation[-1]
        spawn_x = self.SPAWN_RADIUS * np.cos(ang)
        spawn_y = self.SPAWN_RADIUS * np.sin(ang)
        pos = [spawn_x, spawn_y, self.BOOK_LENGTH / 2]
        quat = pybullet.getQuaternionFromEuler([0, np.pi / 2, ang])

        book_id = self.sim.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=quat,
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

    def _create_holder(self):
        """Create minimal shelf: 2 side walls, 3 horizontal shelves (2 compartments), 1 back wall.
        Book is placed in the upper compartment (on shelf 2); clearance ensures book fits.
        """
        ids = []
        cx, cy = self.HOLDER_POS[0], self.HOLDER_POS[1]
        iw = self.HOLDER_INNER_W
        d = self.HOLDER_DEPTH
        t = self.WALL_T
        st = self.SHELF_THICKNESS
        s1 = self.SHELF1_TOP_Z
        s2 = self.SHELF2_TOP_Z
        s3 = self.SHELF3_TOP_Z
        full_height = s3

        wood = [0.55, 0.35, 0.2, 1.0]

        # ---- 2 side walls (left, right): full height ----
        wall_half = [t / 2, d / 2, full_height / 2]
        col_w = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=wall_half)
        vis_w = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=wall_half, rgbaColor=wood,
        )
        left_id = self.sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_w,
            baseVisualShapeIndex=vis_w,
            basePosition=[cx - iw / 2 - t / 2, cy, full_height / 2],
        )
        ids.append(left_id)
        right_id = self.sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_w,
            baseVisualShapeIndex=vis_w,
            basePosition=[cx + iw / 2 + t / 2, cy, full_height / 2],
        )
        ids.append(right_id)

        # ---- 1 back wall: full width, full height ----
        back_half = [iw / 2 + t, t / 2, full_height / 2]
        col_b = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=back_half)
        vis_b = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=back_half, rgbaColor=wood,
        )
        back_id = self.sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_b,
            baseVisualShapeIndex=vis_b,
            basePosition=[cx, cy + d / 2 + t / 2, full_height / 2],
        )
        ids.append(back_id)

        # ---- 3 horizontal shelves (creating 2 compartments) ----
        shelf_half = [iw / 2 + t, d / 2 + t, st / 2]
        col_sh = self.sim.createCollisionShape(pybullet.GEOM_BOX, halfExtents=shelf_half)
        vis_sh = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=shelf_half, rgbaColor=wood,
        )
        for shelf_top_z in (s1, s2, s3):
            center_z = shelf_top_z - st / 2
            sid = self.sim.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_sh,
                baseVisualShapeIndex=vis_sh,
                basePosition=[cx, cy, center_z],
            )
            ids.append(sid)

        return ids

    def compute_reward(self):
        """
        Reward: book proximity to holder + correct orientation + released.

        Full success (1.0): book in holder, vertical, released.
        Shaping: XY proximity (0.3) + Z height (0.2) + vertical (0.2) + release (0.3).
        """
        book_pos, book_quat = self.sim.getBasePositionAndOrientation(self._book_id)
        book_pos = np.array(book_pos)

        dx = abs(book_pos[0] - self._target_pos[0])
        dy = abs(book_pos[1] - self._target_pos[1])
        z_err = abs(book_pos[2] - self._target_pos[2])
        xy_dist = np.sqrt(dx**2 + dy**2)

        # Book thickness axis (local Z) should be horizontal when standing
        rot_mat = np.array(self.sim.getMatrixFromQuaternion(book_quat)).reshape(3, 3)
        book_z_axis = rot_mat[:, 2]
        verticality = abs(np.dot(book_z_axis, [0, 0, 1]))
        is_vertical = verticality < 0.2

        released = self.constraint_id is None
        in_holder = (
            dx < self.HOLDER_INNER_W / 2
            and dy < self.HOLDER_DEPTH / 2
            and z_err < self.BOOK_LENGTH / 2
        )

        if in_holder and is_vertical and released:
            return 1.0

        reward = 0.0
        # XY proximity (0-0.3)
        reward += 0.3 * max(0, 1.0 - xy_dist / 0.3)
        # Z height (0-0.2)
        reward += 0.2 * max(0, 1.0 - z_err / 0.2)
        # Verticality (0-0.2)
        reward += 0.2 * max(0, 1.0 - verticality)
        # Release bonus (0.3): only if in holder
        if in_holder and released:
            reward += 0.3

        return np.clip(reward, 0.0, 1.0)
