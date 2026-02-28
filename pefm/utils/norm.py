import numpy as np
import torch


class Normalizer(object):
    def __init__(self, data, symmetric=False, indices=None):
        if isinstance(data, dict):
            self.stats = data
        elif symmetric:
            if indices is None:
                indices = np.arange(data.shape[-1])[None]

            self.stats = {
                "min": torch.zeros([data.shape[-1]]).to(data.device),
                "max": torch.ones([data.shape[-1]]).to(data.device),
            }
            for group in indices:
                max_abs = torch.abs(data[:, group]).max(0)[0].detach()
                limits = torch.ones_like(max_abs) * torch.max(max_abs)
                self.stats["max"][group] = limits
        else:
            mask = torch.zeros([data.shape[-1]]).to(data.device)
            if indices is not None:
                mask[indices.flatten()] += 1
            else:
                mask += 1
            self.stats = {
                "min": data.min(0)[0].detach() * mask,
                "max": data.max(0)[0].detach() * mask + 1.0 * (1 - mask),
            }

    def normalize(self, data):
        nd = len(data.shape)
        target_shape = (1,) * (nd - 1) + (data.shape[-1],)
        dmin = self.stats["min"].reshape(target_shape)
        dmax = self.stats["max"].reshape(target_shape)
        return (data - dmin) / (dmax - dmin + 1e-12)

    def unnormalize(self, data):
        nd = len(data.shape)
        target_shape = (1,) * (nd - 1) + (data.shape[-1],)
        dmin = self.stats["min"].reshape(target_shape)
        dmax = self.stats["max"].reshape(target_shape)
        return data * (dmax - dmin) + dmin

    def state_dict(self):
        return self.stats

    def load_state_dict(self, state_dict):
        self.stats = state_dict


class RotationAwareNormalizer(object):
    """
    Normalizer preserving rotation geometry for coupled dimensions.

    Coupled dimension groups (e.g., X,Y) share the same scale factor so
    rotations in normalized space correspond to true rotations in world space.
    Each dimension keeps its own center (per-dim mean).
    """

    def __init__(self, data, coupled_groups=None):
        """
        Args:
            data: (N, D) tensor or dict (state_dict for loading).
            coupled_groups: list of lists, e.g. [[0,1], [4,5]].
                Dims in same group share scale. Unlisted dims are independent.
        """
        if isinstance(data, dict):
            self.stats = data
            return

        if coupled_groups is None:
            coupled_groups = []

        center = data.mean(0).detach()

        dmin = data.min(0)[0].detach()
        dmax = data.max(0)[0].detach()
        scale = (dmax - dmin).clamp(min=1e-12)

        for group in coupled_groups:
            max_range = scale[group].max()
            for idx in group:
                scale[idx] = max_range

        self.stats = {"center": center, "scale": scale}

    def normalize(self, data):
        nd = len(data.shape)
        target_shape = (1,) * (nd - 1) + (data.shape[-1],)
        center = self.stats["center"].reshape(target_shape)
        scale = self.stats["scale"].reshape(target_shape)
        return (data - center) / scale.clamp(min=1e-3)

    def unnormalize(self, data):
        nd = len(data.shape)
        target_shape = (1,) * (nd - 1) + (data.shape[-1],)
        center = self.stats["center"].reshape(target_shape)
        scale = self.stats["scale"].reshape(target_shape)
        return data * scale + center

    def state_dict(self):
        return self.stats

    def load_state_dict(self, state_dict):
        self.stats = state_dict
