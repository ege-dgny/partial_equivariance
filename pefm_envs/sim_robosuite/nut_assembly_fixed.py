"""
NutAssemblySquare with C4-discretized nut rotation — C4 symmetry conflict.

Grasp phase: Square nut has C4 symmetry (4 equivalent grasp orientations).
Insert phase: Peg is fixed orientation -> C4 breaks to C1.

Modifications from base robosuite NutAssemblySquare:
  - Square nut Z-rotation discretized to {0, pi/2, pi, 3pi/2}
  - Peg orientation is already fixed in robosuite arena
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .base_robosuite_env import RobosuiteBaseEnv


class NutAssemblyFixedEnv(RobosuiteBaseEnv):

    C4_ROTATIONS = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

    @property
    def robosuite_env_name(self) -> str:
        return "NutAssemblySquare"

    def _modify_env_kwargs(self) -> dict:
        return {
            "single_object_mode": 2,  # square nut only
        }

    def reset(self):
        obs = super().reset()

        # Discretize nut rotation to C4
        if self.randomize_rotation:
            self._set_nut_c4_rotation()
            # Re-read obs after modifying nut pose
            self._cached_obs_dict = self.env._get_observations()
            obs = self._make_pefm_obs(self._cached_obs_dict)

        return obs

    def _set_nut_c4_rotation(self):
        """Override nut Z-rotation to a C4 element."""
        c4_angle = self.rng.choice(self.C4_ROTATIONS)
        self._object_rotation = np.array([0.0, 0.0, c4_angle])

        # Find nut joint in MuJoCo state
        model = self.env.sim.model
        nut_joint_id = None
        for i in range(model.njnt):
            name = model.joint_id2name(i)
            if name and "nut" in name.lower():
                nut_joint_id = i
                break

        if nut_joint_id is None:
            return

        qpos_addr = model.jnt_qposadr[nut_joint_id]

        # Read current position, only change quaternion
        nut_pos = self.env.sim.data.qpos[qpos_addr:qpos_addr + 3].copy()

        # C4 rotation as quaternion (MuJoCo wxyz)
        quat_xyzw = Rotation.from_euler("z", c4_angle).as_quat()
        quat_wxyz = np.array([
            quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2],
        ])

        self.env.sim.data.qpos[qpos_addr:qpos_addr + 3] = nut_pos
        self.env.sim.data.qpos[qpos_addr + 3:qpos_addr + 7] = quat_wxyz
        self.env.sim.forward()

    def _get_object_body_ids(self) -> list[int]:
        """Body IDs for nut + peg."""
        model = self.env.sim.model
        ids = []
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name and any(k in name.lower() for k in ["nut", "peg"]):
                ids.append(i)
        return ids

    def get_nut_pos(self) -> np.ndarray:
        """Get current nut world position."""
        obs = self._cached_obs_dict
        for key in ["SquareNut_pos", "object-state"]:
            if key in obs:
                val = obs[key]
                return val[:3].copy() if len(val) > 3 else val.copy()
        return np.array([0.0, 0.0, 0.0])

    def get_peg_pos(self) -> np.ndarray:
        """Get peg (fixed) world position from MuJoCo."""
        model = self.env.sim.model
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name and "peg" in name.lower():
                return self.env.sim.data.body_xpos[i].copy()
        return np.array([0.0, 0.0, 0.85])

    @property
    def name(self) -> str:
        return "nut_assembly_fixed"
