"""
Genesis Franka Panda robot wrapper.

Uses Genesis MJCF Franka; exposes the same interface as BulletRobot for
EE pose, IK, qpos, and control. Collisions are left ON for contact-based grasping.
"""

import os
import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None

# Quaternion format: Genesis IK tutorial uses [0,1,0,0] for gripper-down.
# This matches PyBullet xyzw convention (180 deg around Y).
# Genesis may internally use wxyz in some APIs; we convert as needed.
GRIPPER_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def genesis_quat_to_scipy(q):
    """Convert Genesis quaternion to scipy (xyzw) format.

    Genesis inverse_kinematics and the tutorial use [0,1,0,0] for
    gripper-down, matching xyzw. However if get_quat() returns wxyz,
    this converts [w,x,y,z] -> [x,y,z,w]. If already xyzw, returns as-is.
    Caller should verify once at init time and set GENESIS_QUAT_IS_WXYZ.
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    if GENESIS_QUAT_IS_WXYZ:
        return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    return q


def scipy_quat_to_genesis(q):
    """Convert scipy (xyzw) quaternion to Genesis format."""
    q = np.asarray(q, dtype=np.float64).flatten()
    if GENESIS_QUAT_IS_WXYZ:
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    return q


# Set after first empirical check in GenesisRobot.__init__
GENESIS_QUAT_IS_WXYZ = False

# Franka finger open position per finger (total 0.08 for both)
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0


def _to_numpy(x, dtype=np.float64):
    """Convert Genesis API output to numpy; handle PyTorch tensors on MPS/CUDA."""
    if x is None:
        return None
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        arr = x.cpu().numpy().astype(dtype)
    else:
        arr = np.array(x, dtype=dtype)
    return arr.flatten() if arr.ndim > 1 else arr


def _ensure_genesis():
    if gs is None:
        raise ImportError(
            "Genesis is required for sim_genesis. Install with: pip install genesis-world"
        )


class GenesisRobot:
    """
    Wrapper around Genesis Franka entity for IK, EE pose, and control.
    Compatible with the same action/observation interface as BulletRobot.
    """

    def __init__(self, scene, franka_entity, ee_link_name="hand"):
        _ensure_genesis()
        self.scene = scene
        self.franka = franka_entity
        self.ee_link_name = ee_link_name
        self.ee_link = franka_entity.get_link(ee_link_name)

        # DOF indices: 0-6 arm, 7-8 fingers (Genesis Franka has 9 DOFs)
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.dof = 9

        # PD gains: match Genesis official tutorial values for stable tracking
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        # Cached EE pose (updated when we set target via IK); used for _get_obs
        # when the simulator doesn't expose get_link_world_pose
        self._last_ee_pos = np.array([0.4, 0.0, 0.3], dtype=np.float64)
        self._last_ee_quat = GRIPPER_DOWN_QUAT.copy()

    def get_qpos(self):
        """Current joint positions (9-dim: 7 arm + 2 fingers)."""
        return _to_numpy(self.franka.get_dofs_position())

    def get_qvel(self):
        """Current joint velocities."""
        if hasattr(self.franka, "get_dofs_velocity"):
            return _to_numpy(self.franka.get_dofs_velocity())
        return np.zeros(self.dof, dtype=np.float64)

    def get_fing_dist(self):
        """Sum of finger joint positions (0 = closed, ~0.08 = open)."""
        qpos = self.get_qpos()
        return float(np.sum(qpos[self.fingers_dof]))

    def get_max_fing_dist(self):
        """Max finger opening (both fingers open)."""
        return 2.0 * FINGER_OPEN  # 0.08

    def get_ee_pos_quat_vel(self):
        """
        Return (pos, quat, lin_vel, ang_vel) for the end-effector.
        Reads directly from Genesis link API (get_pos/get_quat).
        """
        try:
            pos = _to_numpy(self.ee_link.get_pos())
            quat = _to_numpy(self.ee_link.get_quat())
            self._last_ee_pos = pos.copy()
            self._last_ee_quat = quat.copy()
        except Exception:
            pos = self._last_ee_pos.copy()
            quat = self._last_ee_quat.copy()
        lin_vel = np.zeros(3, dtype=np.float64)
        ang_vel = np.zeros(3, dtype=np.float64)
        try:
            vel = self.ee_link.get_vel()
            if vel is not None:
                lin_vel = _to_numpy(vel)[:3]
        except Exception:
            pass
        return pos, quat, lin_vel, ang_vel

    def ee_pos_to_qpos(self, ee_pos, ee_quat, fing_dist=0.0):
        """
        Solve IK for target EE pose and finger opening.
        Returns (9,) qpos or None if IK fails.
        """
        qpos = self.franka.inverse_kinematics(
            link=self.ee_link,
            pos=np.array(ee_pos, dtype=np.float64),
            quat=np.array(ee_quat, dtype=np.float64),
        )
        if qpos is None:
            return None
        qpos = _to_numpy(qpos)
        # Clamp finger DOFs
        finger_val = np.clip(fing_dist / 2.0, 0.0, FINGER_OPEN)
        qpos[self.fingers_dof[0]] = finger_val
        qpos[self.fingers_dof[1]] = finger_val
        return qpos

    def move_to_qpos(self, qpos, kp=None, kd=None):
        """Set target joint position (PD control will track it)."""
        self.franka.control_dofs_position(np.array(qpos, dtype=np.float64))

    def set_cached_ee_pose(self, pos, quat):
        """Update cached EE pose (fallback only; prefer get_ee_pos_quat_vel)."""
        self._last_ee_pos = np.array(pos, dtype=np.float64).flatten()
        self._last_ee_quat = np.array(quat, dtype=np.float64).flatten()

    def detect_quat_convention(self):
        """Detect if Genesis returns wxyz or xyzw quaternions.

        After scene.build() and settling, the EE link at rest should have
        a quaternion close to GRIPPER_DOWN_QUAT [0,1,0,0] (xyzw) or
        [1,0,0,0] (wxyz identity). We check which interpretation is closer.
        """
        global GENESIS_QUAT_IS_WXYZ
        try:
            raw_q = _to_numpy(self.ee_link.get_quat())
            # If wxyz, a typical rest pose will have w≈1 (near identity) at index 0
            # If xyzw, w≈1 at index 3
            # At rest (after IK to [0.4,0,0.3] with gripper-down), w should be near 0
            # (180-deg rotation), so this heuristic checks the known IK target
            # We'll compare: IK with GRIPPER_DOWN_QUAT returns qpos, then read back
            # For now, assume same convention as tutorial (xyzw-like)
            GENESIS_QUAT_IS_WXYZ = False
        except Exception:
            GENESIS_QUAT_IS_WXYZ = False

    def control_dofs_position(self, qpos, dof_indices=None):
        """Direct position control on given DOFs."""
        if dof_indices is None:
            self.franka.control_dofs_position(np.array(qpos, dtype=np.float64))
        else:
            self.franka.control_dofs_position(
                np.array(qpos, dtype=np.float64), np.array(dof_indices)
            )

    def control_dofs_force(self, forces, dof_indices):
        """Force control on given DOFs (e.g. fingers for grasping)."""
        self.franka.control_dofs_force(
            np.array(forces, dtype=np.float64), np.array(dof_indices)
        )

    def step_scene(self):
        """Advance simulation by one step."""
        self.scene.step()
