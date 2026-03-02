"""
Cup Pouring environment: SO(2) symmetry conflict via tilt direction.

A hollow cup with a ball inside spawns at random Z-rotation on an arc
around the arm base.  A bowl sits at a FIXED world-frame position.
The policy must grasp the cup, carry it to the nearest bowl edge, and
tilt toward the bowl center so the ball falls in.

Key PEFM Signal:
- Grasp phase: Object-centric (approach works from any angle) -> HIGH entropy
- Transport + tilt phase: Tilt direction depends on cup-to-bowl vector
  (world-frame constraint) -> entropy COLLAPSES

EquiBot Failure Mode:
- Strict equivariance: rotating the observation rotates the output
- Cannot break symmetry for the tilt direction
- Would tilt in the wrong direction for rotated inputs
"""

import numpy as np
import pybullet
from scipy.spatial.transform import Rotation

from .franka_env import FrankaEnv


class CupPourEnv(FrankaEnv):

    # Cup dimensions (hollow octagonal tube); 3cm radius = 6cm diameter (graspable)
    CUP_RADIUS = 0.03        # 3cm outer radius
    CUP_HEIGHT = 0.08         # 8cm wall height
    CUP_WALL_THICKNESS = 0.004  # 4mm wall/bottom thickness
    N_CUP_WALLS = 8

    # Bowl dimensions (short octagonal container, fixed)
    BOWL_RADIUS = 0.07        # 7cm radius
    BOWL_HEIGHT = 0.04         # 4cm height
    BOWL_WALL_THICKNESS = 0.004
    N_BOWL_WALLS = 8
    BOWL_POS = np.array([0.4, 0.0, 0.0])  # Fixed world-frame position

    # Ball
    BALL_RADIUS = 0.012       # 12mm
    BALL_MASS = 0.02

    # Spawn geometry
    SPAWN_RADIUS = 0.45
    SPAWN_ANGLE_RANGE = (-np.pi / 3, np.pi / 3)
    CUP_BOWL_MIN_SEP = 0.12   # Minimum cup-bowl center distance

    # Pour parameters — must exceed 90° so the opening faces downward
    POUR_TILT_ANGLE = 2 * np.pi / 3  # 120-degree tilt (30° past horizontal)

    @property
    def name(self):
        return "cup_pour"

    @property
    def spawn_angle_range(self):
        return self.SPAWN_ANGLE_RANGE

    def _randomize_object_scales(self):
        """Ensure cup spawns at least CUP_BOWL_MIN_SEP from bowl.

        Always enforces separation — even with randomize_rotation=False the
        default angle (0) places the cup at (0.45, 0) which is only 0.05
        from the bowl at (0.4, 0), well inside the bowl geometry.
        """
        super()._randomize_object_scales()
        R = self.SPAWN_RADIUS
        bowl_xy = self.BOWL_POS[:2]
        B = np.linalg.norm(bowl_xy)
        d = self.CUP_BOWL_MIN_SEP
        if B < 1e-6 or R + B <= d:
            return
        cos_max = (R * R + B * B - d * d) / (2 * R * B)
        cos_max = np.clip(cos_max, -1.0, 1.0)
        theta_min = np.arccos(cos_max)
        lo, hi = self.SPAWN_ANGLE_RANGE

        if self.randomize_rotation:
            left_lo, left_hi = lo, -theta_min
            right_lo, right_hi = theta_min, hi
            if self.rng.rand() < 0.5 and left_hi > left_lo:
                ang = self.rng.uniform(left_lo, left_hi)
            else:
                if right_hi <= right_lo:
                    ang = self.rng.uniform(left_lo, left_hi) if left_hi > left_lo else lo
                else:
                    ang = self.rng.uniform(right_lo, right_hi)
        else:
            ang = self._object_rotation[2]
            if abs(ang) < theta_min:
                ang = theta_min
        self._object_rotation = np.array([0.0, 0.0, ang])

    @property
    def default_front_camera(self):
        return {
            "pitch": -40,
            "yaw": 0,
            "roll": 0,
            "distance": 1.4,
            "fov": 45,
            "target": [0.3, 0.0, 0.1],
        }

    @property
    def default_side_camera(self):
        return {
            "pitch": -30,
            "yaw": 90,
            "roll": 0,
            "distance": 1.3,
            "fov": 45,
            "target": [0.3, 0.0, 0.1],
        }

    # ------------------------------------------------------------------ #
    #  Task object creation
    # ------------------------------------------------------------------ #

    def _create_task_objects(self):
        self.grasp_snap_to_center = True
        self.grasp_min_closed_dist = getattr(self.args, 'grasp_min_closed_dist', 0.07)
        self.grasp_attach_max_dist = getattr(self.args, 'grasp_attach_max_dist', 0.05)

        self._cup_id = self._create_hollow_cup()
        self._bowl_ids = self._create_hollow_bowl()

        # Short settle so cup is stable before placing ball inside
        for _ in range(25):
            self.sim.stepSimulation()

        self._ball_id = self._create_ball()

        # Cup is graspable
        self.rigid_ids.append(self._cup_id)
        self._rigid_graspable.append(True)

        # Bowl parts are not graspable
        for bid in self._bowl_ids:
            self.rigid_ids.append(bid)
            self._rigid_graspable.append(False)

        # Ball is not graspable
        self.rigid_ids.append(self._ball_id)
        self._rigid_graspable.append(False)

        # Extra settling for ball to rest inside cup
        for _ in range(80):
            self.sim.stepSimulation()

    def _snap_object_to_grasp(self, obj_id):
        """Snap cup to finger midpoint and move the ball by the same displacement."""
        grasp_center = self._get_grasp_reference_pos()
        obj_pos, obj_ori = self.sim.getBasePositionAndOrientation(obj_id)
        obj_euler = list(pybullet.getEulerFromQuaternion(obj_ori))

        dx = grasp_center[0] - obj_pos[0]
        dy = grasp_center[1] - obj_pos[1]

        snapped_pos = [grasp_center[0], grasp_center[1], obj_pos[2]]
        snapped_ori = pybullet.getQuaternionFromEuler([0.0, 0.0, obj_euler[2]])
        self.sim.resetBasePositionAndOrientation(obj_id, snapped_pos, snapped_ori)

        ball_pos, ball_ori = self.sim.getBasePositionAndOrientation(self._ball_id)
        self.sim.resetBasePositionAndOrientation(
            self._ball_id,
            [ball_pos[0] + dx, ball_pos[1] + dy, ball_pos[2]],
            ball_ori,
        )

    def _disable_graspable_collisions(self):
        """Disable robot<->cup (ALL links) and robot<->ball."""
        robot_id = self.robot.info.robot_id
        num_robot_links = self.sim.getNumJoints(robot_id)
        num_cup_links = self.sim.getNumJoints(self._cup_id)

        # Robot <-> cup (base link + all wall child links)
        for robot_link in range(-1, num_robot_links):
            for cup_link in range(-1, num_cup_links):
                self.sim.setCollisionFilterPair(
                    robot_id, self._cup_id, robot_link, cup_link, 0
                )

        # Robot <-> ball
        for robot_link in range(-1, num_robot_links):
            self.sim.setCollisionFilterPair(
                robot_id, self._ball_id, robot_link, -1, 0
            )

    # ------------------------------------------------------------------ #
    #  Hollow cup (compound multi-body)
    # ------------------------------------------------------------------ #

    def _create_hollow_cup(self):
        """Create hollow octagonal cup: bottom disk + 8 box walls."""
        R = self.CUP_RADIUS
        H = self.CUP_HEIGHT
        T = self.CUP_WALL_THICKNESS
        N = self.N_CUP_WALLS

        # Base link: thin bottom disk
        col_base = self.sim.createCollisionShape(
            pybullet.GEOM_CYLINDER, radius=R, height=T
        )
        vis_base = self.sim.createVisualShape(
            pybullet.GEOM_CYLINDER, radius=R, length=T,
            rgbaColor=[0.85, 0.85, 0.9, 1.0],
        )

        # Wall segment dimensions
        wall_width = 2 * R * np.sin(np.pi / N)
        wall_half = [wall_width / 2, T / 2, H / 2]

        link_masses = []
        link_col = []
        link_vis = []
        link_pos = []
        link_ori = []
        link_inertial_pos = []
        link_inertial_ori = []
        link_parent = []
        link_joint_type = []
        link_joint_axis = []

        for i in range(N):
            angle = i * (2 * np.pi / N)

            col_wall = self.sim.createCollisionShape(
                pybullet.GEOM_BOX, halfExtents=wall_half
            )
            vis_wall = self.sim.createVisualShape(
                pybullet.GEOM_BOX, halfExtents=wall_half,
                rgbaColor=[0.85, 0.85, 0.9, 0.9],
            )

            # Wall center on the circle, above the bottom disk
            wx = R * np.cos(angle)
            wy = R * np.sin(angle)
            wz = T / 2 + H / 2  # relative to base center

            # Rotate wall so its wide face (X) is tangent to the circle
            wall_quat = pybullet.getQuaternionFromEuler([0, 0, angle + np.pi / 2])

            link_masses.append(0.005)
            link_col.append(col_wall)
            link_vis.append(vis_wall)
            link_pos.append([wx, wy, wz])
            link_ori.append(list(wall_quat))
            link_inertial_pos.append([0, 0, 0])
            link_inertial_ori.append([0, 0, 0, 1])
            link_parent.append(0)
            link_joint_type.append(pybullet.JOINT_FIXED)
            link_joint_axis.append([0, 0, 0])

        # Spawn position on arc
        ang = self._object_rotation[-1]
        spawn_x = self.SPAWN_RADIUS * np.cos(ang)
        spawn_y = self.SPAWN_RADIUS * np.sin(ang)
        pos = [spawn_x, spawn_y, T / 2 + 0.001]
        quat = pybullet.getQuaternionFromEuler([0, 0, ang])

        cup_id = self.sim.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=col_base,
            baseVisualShapeIndex=vis_base,
            basePosition=pos,
            baseOrientation=quat,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_col,
            linkVisualShapeIndices=link_vis,
            linkPositions=link_pos,
            linkOrientations=link_ori,
            linkInertialFramePositions=link_inertial_pos,
            linkInertialFrameOrientations=link_inertial_ori,
            linkParentIndices=link_parent,
            linkJointTypes=link_joint_type,
            linkJointAxis=link_joint_axis,
        )
        self.sim.changeDynamics(cup_id, -1, lateralFriction=1.0)
        return cup_id

    # ------------------------------------------------------------------ #
    #  Hollow bowl (separate static bodies)
    # ------------------------------------------------------------------ #

    def _create_hollow_bowl(self):
        """Create hollow bowl at fixed world position: bottom disk + 8 walls."""
        R = self.BOWL_RADIUS
        H = self.BOWL_HEIGHT
        T = self.BOWL_WALL_THICKNESS
        N = self.N_BOWL_WALLS
        bx, by = self.BOWL_POS[0], self.BOWL_POS[1]

        ids = []

        # Bottom disk
        col_base = self.sim.createCollisionShape(
            pybullet.GEOM_CYLINDER, radius=R, height=T
        )
        vis_base = self.sim.createVisualShape(
            pybullet.GEOM_CYLINDER, radius=R, length=T,
            rgbaColor=[0.5, 0.3, 0.1, 1.0],
        )
        base_id = self.sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_base,
            baseVisualShapeIndex=vis_base,
            basePosition=[bx, by, T / 2 + 0.001],
        )
        ids.append(base_id)

        # Wall segments
        wall_width = 2 * R * np.sin(np.pi / N)
        wall_half = [wall_width / 2, T / 2, H / 2]

        for i in range(N):
            angle = i * (2 * np.pi / N)

            col_wall = self.sim.createCollisionShape(
                pybullet.GEOM_BOX, halfExtents=wall_half
            )
            vis_wall = self.sim.createVisualShape(
                pybullet.GEOM_BOX, halfExtents=wall_half,
                rgbaColor=[0.5, 0.3, 0.1, 0.9],
            )

            wx = bx + R * np.cos(angle)
            wy = by + R * np.sin(angle)
            wz = T + H / 2 + 0.001  # above bottom disk

            wall_quat = pybullet.getQuaternionFromEuler([0, 0, angle + np.pi / 2])

            wall_id = self.sim.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_wall,
                baseVisualShapeIndex=vis_wall,
                basePosition=[wx, wy, wz],
                baseOrientation=wall_quat,
            )
            ids.append(wall_id)

        return ids

    # ------------------------------------------------------------------ #
    #  Ball
    # ------------------------------------------------------------------ #

    def _create_ball(self):
        """Create ball inside the cup."""
        col = self.sim.createCollisionShape(
            pybullet.GEOM_SPHERE, radius=self.BALL_RADIUS
        )
        vis = self.sim.createVisualShape(
            pybullet.GEOM_SPHERE, radius=self.BALL_RADIUS,
            rgbaColor=[1.0, 0.2, 0.2, 1.0],
        )

        cup_pos, _ = self.sim.getBasePositionAndOrientation(self._cup_id)
        ball_z = cup_pos[2] + self.CUP_WALL_THICKNESS / 2 + self.BALL_RADIUS + 0.005
        ball_pos = [cup_pos[0], cup_pos[1], ball_z]

        ball_id = self.sim.createMultiBody(
            baseMass=self.BALL_MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=ball_pos,
        )
        self.sim.changeDynamics(
            ball_id, -1, lateralFriction=0.5, restitution=0.3
        )
        return ball_id

    # ------------------------------------------------------------------ #
    #  Reward
    # ------------------------------------------------------------------ #

    def compute_reward(self):
        """
        Reward: ball in bowl + cup proximity + tilt quality.

        Full success (1.0): ball inside bowl (cup stays held).
        Shaping: ball_in_bowl (0.55) + cup_proximity (0.25) +
                 tilt_toward_bowl (0.20).
        """
        ball_pos, _ = self.sim.getBasePositionAndOrientation(self._ball_id)
        ball_pos = np.array(ball_pos)
        cup_pos, cup_quat = self.sim.getBasePositionAndOrientation(self._cup_id)
        cup_pos = np.array(cup_pos)
        bowl_center = self.BOWL_POS[:2]

        # Ball in bowl check
        ball_xy_dist = np.linalg.norm(ball_pos[:2] - bowl_center)
        ball_in_bowl_xy = ball_xy_dist < self.BOWL_RADIUS
        ball_in_bowl_z = 0 < ball_pos[2] < self.BOWL_HEIGHT + 0.02
        ball_in_bowl = ball_in_bowl_xy and ball_in_bowl_z

        if ball_in_bowl:
            return 1.0

        reward = 0.0

        # Ball in bowl proximity (0-0.55): PRIMARY
        reward += 0.55 * max(0, 1.0 - ball_xy_dist / 0.4)

        # Cup proximity to bowl edge (0-0.25)
        cup_xy_dist = np.linalg.norm(cup_pos[:2] - bowl_center)
        reward += 0.25 * max(0, 1.0 - cup_xy_dist / 0.4)

        # Cup tilt toward bowl center (0-0.20)
        rot = Rotation.from_quat(cup_quat)
        cup_z_axis = rot.apply([0, 0, 1])
        dir_to_bowl = bowl_center - cup_pos[:2]
        dir_norm = np.linalg.norm(dir_to_bowl)
        if dir_norm > 0.01:
            dir_3d = np.array([
                dir_to_bowl[0] / dir_norm,
                dir_to_bowl[1] / dir_norm,
                0.0,
            ])
            tilt_toward = np.dot(cup_z_axis, dir_3d)
            reward += 0.20 * max(0, tilt_toward)

        return np.clip(reward, 0.0, 1.0)
