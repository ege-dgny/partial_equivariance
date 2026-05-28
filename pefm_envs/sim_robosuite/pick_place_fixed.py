"""
PickPlaceCan with FIXED bin position — SO(2) symmetry conflict.

Grasp phase: Can be approached from any angle -> SO(2)-equivariant.
Place phase: Bin at fixed world position -> world-centric, symmetry breaks.

Modifications from base robosuite PickPlaceCan:
  - Bin position fixed to constant world-frame location
  - Can Z-rotation randomized on each reset
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .base_robosuite_env import RobosuiteBaseEnv


class PickPlaceFixedEnv(RobosuiteBaseEnv):

    # Fixed bin position (world frame, XY only — Z is table height)
    BIN_POS = np.array([0.18, 0.25])

    # Task-specific PC crop: drop the fixed target bin (y>0.12) so the can
    # is a larger fraction of the object-only cloud. Target is fixed -> memorized.
    WS_BOUNDS = ((-0.2, 0.45), (-0.55, 0.12), (0.80, 1.25))

    @property
    def robosuite_env_name(self) -> str:
        return "PickPlaceCan"

    def _modify_env_kwargs(self) -> dict:
        # PickPlaceCan is already single-object (can only)
        return {}

    def reset(self):
        obs = super().reset()

        # Record can Z-rotation for PEFM symmetry analysis
        self._read_object_rotation()

        return obs

    def _read_object_rotation(self):
        """Extract can's Z-rotation from MuJoCo state."""
        try:
            can_body = None
            model = self.env.sim.model
            for i in range(model.nbody):
                name = model.body_id2name(i)
                if name and "can" in name.lower():
                    can_body = i
                    break
            if can_body is not None:
                quat_wxyz = self.env.sim.data.body_xquat[can_body]
                # MuJoCo wxyz -> scipy xyzw
                quat_xyzw = np.array([
                    quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0],
                ])
                euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")
                self._object_rotation = euler
        except Exception:
            pass

    def _get_object_body_ids(self) -> list[int]:
        """Body IDs for can + bin (for point cloud segmentation)."""
        model = self.env.sim.model
        ids = []
        keywords = ["can", "bin", "visual"]
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name and any(k in name.lower() for k in keywords):
                ids.append(i)
        return list(set(ids))

    def get_can_pos(self) -> np.ndarray:
        """Get current can world position from obs."""
        obs = self._cached_obs_dict
        # robosuite PickPlaceCan provides object state
        for key in ["Can_pos", "can_pos"]:
            if key in obs:
                return obs[key].copy()
        # Fallback: read from sim
        model = self.env.sim.model
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name and "can" in name.lower():
                return self.env.sim.data.body_xpos[i].copy()
        return np.array([0.0, 0.0, 0.0])

    def get_bin_pos(self) -> np.ndarray:
        """Get fixed bin position (XY fixed, Z from arena)."""
        # Z is table height + some offset
        return np.array([self.BIN_POS[0], self.BIN_POS[1], 0.85])

    @property
    def name(self) -> str:
        return "pick_place_fixed"
