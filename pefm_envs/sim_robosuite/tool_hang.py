"""
ToolHang wrapper — multi-phase alternating SO(2) symmetry conflict.

robosuite's ToolHang already contains the symmetry conflict:
  - Stand (with hole) at FIXED world-frame position
  - Frame spawns at random position/rotation (SO(2) grasp symmetry)
  - Tool spawns at random position/rotation (SO(2) grasp symmetry)

Four phases with alternating symmetry:
  1. Grasp frame    -> SO(2)-equivariant  (selector entropy HIGH)
  2. Insert frame   -> world-centric      (selector entropy LOW)
  3. Grasp tool     -> SO(2)-equivariant  (selector entropy HIGH)
  4. Hang tool      -> world-centric      (selector entropy LOW)

Minimal modifications needed — the task is already ideal for PEFM.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .base_robosuite_env import RobosuiteBaseEnv


class ToolHangEnv(RobosuiteBaseEnv):

    @property
    def robosuite_env_name(self) -> str:
        return "ToolHang"

    def _modify_env_kwargs(self) -> dict:
        return {}

    def reset(self):
        obs = super().reset()
        self._read_frame_rotation()
        return obs

    def _read_frame_rotation(self):
        """Record frame Z-rotation for symmetry analysis."""
        try:
            model = self.env.sim.model
            for i in range(model.nbody):
                name = model.body_id2name(i)
                if name and "frame" in name.lower() and "stand" not in name.lower():
                    quat_wxyz = self.env.sim.data.body_xquat[i]
                    quat_xyzw = np.array([
                        quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0],
                    ])
                    euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")
                    self._object_rotation = euler
                    break
        except Exception:
            pass

    def _get_object_body_ids(self) -> list[int]:
        """Body IDs for frame, tool/wrench, and stand."""
        model = self.env.sim.model
        ids = []
        keywords = ["frame", "tool", "wrench", "stand", "hook"]
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name and any(k in name.lower() for k in keywords):
                ids.append(i)
        return ids

    def get_frame_pos(self) -> np.ndarray:
        """Get frame world position."""
        model = self.env.sim.model
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name and "frame" in name.lower() and "stand" not in name.lower():
                return self.env.sim.data.body_xpos[i].copy()
        return np.array([0.0, 0.0, 0.0])

    def get_stand_pos(self) -> np.ndarray:
        """Get stand (fixed) world position."""
        model = self.env.sim.model
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name and "stand" in name.lower():
                return self.env.sim.data.body_xpos[i].copy()
        return np.array([0.0, 0.0, 0.0])

    def get_tool_pos(self) -> np.ndarray:
        """Get tool/wrench world position."""
        model = self.env.sim.model
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name and ("tool" in name.lower() or "wrench" in name.lower()):
                return self.env.sim.data.body_xpos[i].copy()
        return np.array([0.0, 0.0, 0.0])

    @property
    def name(self) -> str:
        return "tool_hang"
