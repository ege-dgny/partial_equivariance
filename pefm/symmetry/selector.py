"""
Symmetry selector network: maps observation encoding z to distribution
parameters over a Lie group G.

This is the learnable p_phi(g|o) component of PEFM.
"""

import torch
import torch.nn as nn


class SymmetrySelector(nn.Module):
    """
    Lightweight MLP: context vector z -> distribution parameters over group G.

    Input:  z (B, z_dim) -- flattened observation encoding from PointNet
    Output: params (B, param_dim) -- parameters for group distribution

    If force_uniform=True the MLP is bypassed and uniform (zero) params are
    returned, making p_phi the maximum-entropy distribution over G. This yields
    an exact group (Reynolds) average that can never collapse -- i.e. the strict
    equivariance baseline used as the PEFM ablation/failure contrast.
    """

    def __init__(self, z_dim, hidden_dim, group, distribution, force_uniform=False):
        super().__init__()
        self.group = group
        self.distribution = distribution
        self.param_dim = distribution.param_dim
        self.force_uniform = force_uniform

        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.param_dim),
        )

    def forward(self, z):
        """
        Returns distribution parameters.
        z: (B, z_dim)
        Returns: (B, param_dim)
        """
        if self.force_uniform:
            # Zero params = uniform/max-entropy distribution for both C4
            # (uniform logits) and SO2 (mu=0 -> uniform angle). Strict equivariance.
            return torch.zeros(z.shape[0], self.param_dim, device=z.device, dtype=z.dtype)
        return self.mlp(z)

    def sample_and_entropy(self, z, num_samples, return_params=False):
        """
        Sample group elements and compute entropy.

        Args:
            z: (B, z_dim) observation conditioning
            num_samples: int N, number of group element samples
            return_params: if True, also return the raw distribution params

        Returns:
            g_samples: (B, N) group elements (angles for SO2, indices for C4)
            entropy: (B,) entropy of the distribution
            params (optional): (B, param_dim) raw distribution parameters
        """
        params = self.forward(z)
        g_samples = self.distribution.sample(params, num_samples)
        entropy = self.distribution.entropy(params)
        if return_params:
            return g_samples, entropy, params
        return g_samples, entropy
