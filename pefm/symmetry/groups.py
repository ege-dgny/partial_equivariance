"""
Group operations for partially equivariant flow matching.

Supports SO(2) (continuous Z-rotation) and C4 (discrete 4-fold rotation).
Each group defines how to transform point clouds, EEF states, and actions.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np


class LieGroup(ABC):
    """Abstract base class for Lie groups used in PEFM."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of group parameterization."""

    @abstractmethod
    def identity(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return identity element(s)."""

    @abstractmethod
    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """Return inverse of group element(s)."""

    @abstractmethod
    def transform_points(self, g: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Apply rho_in(g) to point cloud.
        g: (B,) or (B, N) group elements
        points: (B, ..., 3)
        """

    @abstractmethod
    def transform_action(self, g: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Apply rho_in(g) to action vector.
        g: (B,) or (B, N) group elements
        action: (B, ..., action_dim)
        """

    @abstractmethod
    def inverse_transform_action(
        self, g: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rho_out(g^-1) to velocity/action vector.
        g: (B,) or (B, N) group elements
        action: (B, ..., action_dim)
        """


def _rotate_xy(cos_t, sin_t, xy):
    """
    Rotate 2D coordinates by angle defined by cos/sin.
    cos_t, sin_t: broadcastable to xy[..., 0]
    xy: (..., 2)
    Returns: (..., 2)
    """
    x, y = xy[..., 0], xy[..., 1]
    x_rot = cos_t * x - sin_t * y
    y_rot = sin_t * x + cos_t * y
    return torch.stack([x_rot, y_rot], dim=-1)


class SO2(LieGroup):
    """
    Rotation around Z-axis. Parameterized by angle theta in [0, 2*pi).

    For 7-DOF action [gripper, dx, dy, dz, drx, dry, drz] per EEF:
      - gripper (index 0): scalar, unchanged
      - dx, dy (indices 1, 2): rotated by theta
      - dz (index 3): unchanged
      - drx, dry (indices 4, 5): rotated by theta
      - drz (index 6): unchanged
    """

    @property
    def dim(self) -> int:
        return 1

    def identity(self, batch_size, device):
        return torch.zeros(batch_size, device=device)

    def inverse(self, g):
        return -g

    def transform_points(self, g, points):
        """
        g: (B,) angles
        points: (B, ..., 3) -- rotate x,y, leave z unchanged
        """
        cos_t = torch.cos(g)
        sin_t = torch.sin(g)

        # Broadcast to match points shape
        for _ in range(points.dim() - 1 - cos_t.dim()):
            cos_t = cos_t.unsqueeze(-1)
            sin_t = sin_t.unsqueeze(-1)

        xy_rot = _rotate_xy(cos_t, sin_t, points[..., :2])
        return torch.cat([xy_rot, points[..., 2:]], dim=-1)

    def transform_state(self, g, state, num_eef=2, eef_dim=13):
        """
        Transform EEF state by rotating position and orientation components.
        g: (B,) angles
        state: (B, obs_horizon, num_eef * eef_dim)

        Per EEF, eef_dim=13 layout:
          [0:3] position xyz -> rotate xy
          [3:6] orientation dir1 -> rotate xy
          [6:9] orientation dir2 -> rotate xy
          [9:12] gravity direction -> rotate xy
          [12] gripper -> unchanged

        For even eef_dim != 13 (2D tasks): all dims are consecutive xy pairs.
        """
        orig_shape = state.shape
        # Reshape to (B, ..., num_eef, eef_dim)
        state = state.view(*orig_shape[:-1], num_eef, eef_dim)

        cos_t = torch.cos(g)
        sin_t = torch.sin(g)
        # Broadcast
        for _ in range(state.dim() - 1 - cos_t.dim()):
            cos_t = cos_t.unsqueeze(-1)
            sin_t = sin_t.unsqueeze(-1)

        result = state.clone()

        if eef_dim % 2 == 0 and eef_dim != 13:
            # 2D task: all dims are consecutive xy pairs
            for i in range(eef_dim // 2):
                result[..., 2*i:2*i+2] = _rotate_xy(cos_t, sin_t, state[..., 2*i:2*i+2])
        else:
            # 3D task: eef_dim=13 layout
            result[..., :2] = _rotate_xy(cos_t, sin_t, state[..., :2])
            result[..., 3:5] = _rotate_xy(cos_t, sin_t, state[..., 3:5])
            result[..., 6:8] = _rotate_xy(cos_t, sin_t, state[..., 6:8])
            result[..., 9:11] = _rotate_xy(cos_t, sin_t, state[..., 9:11])

        return result.view(orig_shape)

    def transform_action(self, g, action, dof=7, num_eef=2):
        """
        Transform action by rotating position and rotation deltas.
        g: (B,) angles
        action: (B, pred_horizon, num_eef * dof)

        Per EEF, dof=7 layout: [gripper, dx, dy, dz, drx, dry, drz]
        For dof=2: [x, y] — rotate both.
        """
        orig_shape = action.shape
        B = action.shape[0]
        # Reshape to (B, pred_horizon, num_eef, dof)
        action = action.view(B, -1, num_eef, dof)

        cos_t = torch.cos(g)
        sin_t = torch.sin(g)
        for _ in range(action.dim() - 1 - cos_t.dim()):
            cos_t = cos_t.unsqueeze(-1)
            sin_t = sin_t.unsqueeze(-1)

        result = action.clone()

        if dof == 2:
            # 2D action: [x, y]
            result[..., :2] = _rotate_xy(cos_t, sin_t, action[..., :2])
        else:
            # 3D action: [gripper, dx, dy, dz, drx, dry, drz]
            result[..., 1:3] = _rotate_xy(cos_t, sin_t, action[..., 1:3])
            if dof >= 7:
                result[..., 4:6] = _rotate_xy(cos_t, sin_t, action[..., 4:6])

        return result.view(orig_shape)

    def inverse_transform_action(self, g, action, dof=7, num_eef=2):
        return self.transform_action(-g, action, dof=dof, num_eef=num_eef)


class C4(LieGroup):
    """
    Discrete 4-fold rotation: {0, pi/2, pi, 3*pi/2}.
    Parameterized as integer index 0-3.

    Internally converts to angle and delegates to SO2 rotation logic.
    """

    ANGLES = torch.tensor([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

    @property
    def dim(self) -> int:
        return 4  # categorical logits

    def identity(self, batch_size, device):
        return torch.zeros(batch_size, dtype=torch.long, device=device)

    def inverse(self, g):
        # inverse of index k is (4-k) % 4
        return (4 - g) % 4

    def _idx_to_angle(self, g_idx):
        """Convert integer indices to angles."""
        angles = self.ANGLES.to(g_idx.device)
        return angles[g_idx.long()]

    def transform_points(self, g, points):
        angles = self._idx_to_angle(g)
        cos_t = torch.cos(angles)
        sin_t = torch.sin(angles)

        for _ in range(points.dim() - 1 - cos_t.dim()):
            cos_t = cos_t.unsqueeze(-1)
            sin_t = sin_t.unsqueeze(-1)

        xy_rot = _rotate_xy(cos_t, sin_t, points[..., :2])
        return torch.cat([xy_rot, points[..., 2:]], dim=-1)

    def transform_state(self, g, state, num_eef=2, eef_dim=13):
        angles = self._idx_to_angle(g)
        # Delegate to SO2 logic
        so2 = SO2()
        return so2.transform_state(angles, state, num_eef=num_eef, eef_dim=eef_dim)

    def transform_action(self, g, action, dof=7, num_eef=2):
        angles = self._idx_to_angle(g)
        so2 = SO2()
        return so2.transform_action(angles, action, dof=dof, num_eef=num_eef)

    def inverse_transform_action(self, g, action, dof=7, num_eef=2):
        inv_g = self.inverse(g)
        return self.transform_action(inv_g, action, dof=dof, num_eef=num_eef)


def get_group(group_type: str) -> LieGroup:
    """Factory function for group instances."""
    if group_type == "so2":
        return SO2()
    elif group_type == "c4":
        return C4()
    else:
        raise ValueError(f"Unknown group type: {group_type}")
