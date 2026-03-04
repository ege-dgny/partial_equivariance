"""
Genesis peg insertion task: C4 symmetry conflict.
Keyed peg (asymmetric extension) and keyed socket (molded walls + keyway)
to match PyBullet sim; contact-based grasping.
"""

import os
import tempfile
import numpy as np

from .genesis_franka_env import GenesisFrankaEnv
from .genesis_robot import _to_numpy, genesis_quat_to_scipy

try:
    import genesis as gs
except ImportError:
    gs = None


def _make_peg_mjcf(peg_side, peg_height, notch_size):
    """MJCF for keyed peg: main box + key extension on +X face (half-extents in MJCF)."""
    h = peg_height / 2
    sh = peg_side / 2
    nh = notch_size / 2
    key_h = (peg_height * 0.8) / 2
    key_x = sh + nh
    return f"""<?xml version="1.0"?>
<mujoco model="peg_keyed">
  <worldbody>
    <body name="peg">
      <freejoint name="peg_free"/>
      <geom name="main" type="box" size="{sh} {sh} {h}" pos="0 0 {h}" rgba="0.4 0.6 0.8 1"/>
      <geom name="key" type="box" size="{nh} {nh} {key_h}" pos="{key_x} 0 {h}" rgba="0.4 0.6 0.8 1"/>
    </body>
  </worldbody>
</mujoco>"""


def _make_socket_mjcf(peg_side, clearance, wall_thickness, wall_height, plate_thickness, notch_size):
    """MJCF for keyed socket: base plate + 4 walls with keyway gap on +X (half-extents)."""
    pt = plate_thickness
    wh = wall_height
    wt = wall_thickness
    cl = clearance
    s = peg_side
    ns = notch_size
    inner_xy = s / 2 + cl
    outer_xy = inner_xy + wt
    base_hz = pt / 2
    wall_hz = wh / 2
    wx_neg = wt / 2
    wy_neg = inner_xy
    pos_neg_x = -(inner_xy + wx_neg)
    pos_z_wall = pt + wall_hz
    wx_side = outer_xy
    wy_side = wt / 2
    pos_pos_y = inner_xy + wy_side
    pos_neg_y = -(inner_xy + wy_side)
    half_width = inner_xy - ns / 2
    pos_pos_x = inner_xy + wx_neg
    y_upper = (ns / 2 + inner_xy) / 2
    y_lower = -(ns / 2 + inner_xy) / 2
    return f"""<?xml version="1.0"?>
<mujoco model="socket_keyed">
  <worldbody>
    <body name="socket">
      <geom name="base" type="box" size="{outer_xy} {outer_xy} {base_hz}" pos="0 0 {base_hz}" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_neg_x" type="box" size="{wx_neg} {wy_neg} {wall_hz}" pos="{pos_neg_x} 0 {pos_z_wall}" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_pos_y" type="box" size="{wx_side} {wy_side} {wall_hz}" pos="0 {pos_pos_y} {pos_z_wall}" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_neg_y" type="box" size="{wx_side} {wy_side} {wall_hz}" pos="0 {pos_neg_y} {pos_z_wall}" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_pos_x_upper" type="box" size="{wx_neg} {half_width} {wall_hz}" pos="{pos_pos_x} {y_upper} {pos_z_wall}" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_pos_x_lower" type="box" size="{wx_neg} {half_width} {wall_hz}" pos="{pos_pos_x} {y_lower} {pos_z_wall}" rgba="0.5 0.5 0.5 1"/>
    </body>
  </worldbody>
</mujoco>"""


class GenesisPegInsertEnv(GenesisFrankaEnv):

    PEG_SIDE = 0.04
    PEG_HEIGHT = 0.08
    NOTCH_SIZE = 0.008   # key on peg (8mm)
    NS_SOCKET = 0.014    # keyway in socket (14mm)
    CLEARANCE = 0.008
    WALL_THICKNESS = 0.01
    WALL_HEIGHT = 0.04
    PLATE_THICKNESS = 0.005
    SOCKET_POS = np.array([0.35, -0.2, 0.0])
    PEG_SPAWN_RADIUS = 0.4
    C4_ROTATIONS = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    PEG_SPAWN_ANGLE_RANGE = (-np.pi / 3, np.pi / 3)
    PEG_SOCKET_MIN_SEP = 0.12

    @property
    def name(self):
        return "peg_insert_genesis"

    @property
    def spawn_angle_range(self):
        return self.PEG_SPAWN_ANGLE_RANGE

    def _create_task_objects(self):
        if gs is None:
            return
        self._peg_spawn_rotation = self.rng.choice(self.C4_ROTATIONS)
        socket_xy = self.SOCKET_POS[:2] + self.scene_offset
        lo, hi = self.spawn_angle_range
        for _ in range(50):
            ang = self._object_rotation[-1]
            peg_xy = self.scene_offset + self.PEG_SPAWN_RADIUS * np.array([np.cos(ang), np.sin(ang)])
            if np.linalg.norm(peg_xy - socket_xy) >= self.PEG_SOCKET_MIN_SEP:
                break
            self._object_rotation[-1] = self.rng.rand() * (hi - lo) + lo

        ang = self._object_rotation[-1]
        peg_x = self.scene_offset[0] + self.PEG_SPAWN_RADIUS * np.cos(ang)
        peg_y = self.scene_offset[1] + self.PEG_SPAWN_RADIUS * np.sin(ang)
        total_rot = self._object_rotation[-1] + self._peg_spawn_rotation
        euler_deg = tuple(np.rad2deg([0, 0, total_rot]))
        peg_z = self.PEG_HEIGHT / 2 + 0.001

        # Keyed peg via MJCF — do NOT pass is_free since MJCF already has <freejoint>
        peg_mjcf = _make_peg_mjcf(self.PEG_SIDE, self.PEG_HEIGHT, self.NOTCH_SIZE)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(peg_mjcf)
            peg_path = f.name
        try:
            peg = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=peg_path,
                    pos=(peg_x, peg_y, peg_z),
                    euler=euler_deg,
                )
            )
        finally:
            os.unlink(peg_path)

        self.rigid_ids.append(peg.idx if hasattr(peg, "idx") else len(self.rigid_ids))
        self._rigid_graspable.append(True)
        self._rigid_entities.append(peg)

        # Keyed socket via MJCF (fixed body, no freejoint)
        socket_pos_world = [
            self.SOCKET_POS[0] + self.scene_offset[0],
            self.SOCKET_POS[1] + self.scene_offset[1],
            0.0,  # socket base sits on ground
        ]
        socket_mjcf = _make_socket_mjcf(
            self.PEG_SIDE, self.CLEARANCE, self.WALL_THICKNESS,
            self.WALL_HEIGHT, self.PLATE_THICKNESS, self.NS_SOCKET,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(socket_mjcf)
            socket_path = f.name
        try:
            # Socket MJCF has no <freejoint> → fixed in space automatically
            socket = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=socket_path,
                    pos=tuple(socket_pos_world),
                )
            )
        finally:
            os.unlink(socket_path)

        self.rigid_ids.append(socket.idx if hasattr(socket, "idx") else len(self.rigid_ids))
        self._rigid_graspable.append(False)
        self._rigid_entities.append(socket)

        self._peg_entity = peg
        self._socket_entity = socket

    def compute_reward(self):
        peg_pos = _to_numpy(self._peg_entity.get_pos())
        socket_pos = _to_numpy(self._socket_entity.get_pos())
        socket_center = socket_pos[:2]
        xy_dist = np.linalg.norm(peg_pos[:2] - socket_center)
        pos_threshold = self.PEG_SIDE * 0.5
        socket_top_z = self.PLATE_THICKNESS + self.WALL_HEIGHT
        descended = peg_pos[2] < socket_top_z

        # Get peg quaternion and convert to scipy (xyzw) for Rotation
        raw_quat = self._peg_entity.get_quat() if hasattr(self._peg_entity, "get_quat") else None
        if raw_quat is not None:
            peg_quat = genesis_quat_to_scipy(_to_numpy(raw_quat))
        else:
            peg_quat = np.array([0, 0, 0, 1], dtype=np.float64)

        from scipy.spatial.transform import Rotation
        euler = Rotation.from_quat(peg_quat).as_euler("xyz")
        z_rot = np.mod(euler[2] + np.pi, 2 * np.pi) - np.pi
        rot_error = abs(z_rot)
        rot_aligned = rot_error < np.deg2rad(15)

        if xy_dist < pos_threshold and descended and rot_aligned:
            return 1.0
        reward = 0.0
        reward += 0.3 * max(0, 1.0 - xy_dist / 0.3)
        if xy_dist < pos_threshold * 2:
            reward += 0.3 * max(0, 1.0 - peg_pos[2] / (socket_top_z * 2.0))
        reward += 0.2 * max(0, 1.0 - rot_error / np.pi)
        return np.clip(reward, 0.0, 1.0)
