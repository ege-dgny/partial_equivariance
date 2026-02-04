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
    """

    def __init__(self, z_dim, hidden_dim, group, distribution):
        super().__init__()
        self.group = group
        self.distribution = distribution

        param_dim = distribution.param_dim

        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim),
        )

    def forward(self, z):
        """
        Returns distribution parameters.
        z: (B, z_dim)
        Returns: (B, param_dim)
        """
        return self.mlp(z)

    def sample_and_entropy(self, z, num_samples):
        """
        Sample group elements and compute entropy.

        Args:
            z: (B, z_dim) observation conditioning
            num_samples: int N, number of group element samples

        Returns:
            g_samples: (B, N) group elements (angles for SO2, indices for C4)
            entropy: (B,) entropy of the distribution
        """
        params = self.forward(z)
        g_samples = self.distribution.sample(params, num_samples)
        entropy = self.distribution.entropy(params)
        return g_samples, entropy
