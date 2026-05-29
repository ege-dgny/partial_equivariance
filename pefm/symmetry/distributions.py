"""
Reparameterizable distributions over Lie groups for the symmetry selector.

ProjectedNormalSO2: Continuous distribution on the circle via projected normal.
GumbelSoftmaxCategorical: Discrete distribution with Gumbel-Softmax relaxation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ProjectedNormalSO2(nn.Module):
    """
    Distribution on the circle S^1 via projected normal.

    Sample (u, v) ~ N(mu, diag(sigma^2)) in R^2, then theta = atan2(v, u).
    Reparameterizable through the underlying Gaussian.

    Parameters: (mu_u, mu_v, log_sigma_u, log_sigma_v) -> 4 params total.

    Entropy approximation:
        H approx log(2*pi) - 0.5 * ||mu||^2 / sigma^2 + log(sigma)
        (approaches log(2*pi) when sigma >> ||mu||, i.e. uniform on circle)
    """

    param_dim = 4

    def sample(self, params, num_samples):
        """
        params: (B, 4) -> [mu_u, mu_v, log_sigma_u, log_sigma_v]
        num_samples: int N
        Returns: (B, N) angles in [0, 2*pi)
        """
        mu = params[:, :2]  # (B, 2)
        log_sigma = params[:, 2:].clamp(-4, 4)  # (B, 2)
        sigma = torch.exp(log_sigma)

        B = params.shape[0]
        device = params.device

        # Reparameterization: sample from N(mu, sigma^2)
        eps = torch.randn(B, num_samples, 2, device=device)  # (B, N, 2)
        samples_2d = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps  # (B, N, 2)

        # Project to circle: theta = atan2(v, u)
        angles = torch.atan2(samples_2d[..., 1], samples_2d[..., 0])  # (B, N)

        return angles

    def entropy(self, params):
        """
        Approximate entropy of the projected normal distribution on S^1.

        params: (B, 4)
        Returns: (B,)

        Uses the approximation:
            H ~ log(2*pi) + mean(log(sigma)) - 0.5 * ||mu/sigma||^2
        When sigma is large relative to ||mu||, distribution is near-uniform
        and entropy approaches log(2*pi) ~ 1.838.
        When sigma is small, distribution concentrates and entropy decreases.
        """
        mu = params[:, :2]  # (B, 2)
        log_sigma = params[:, 2:].clamp(-4, 4)  # (B, 2)
        sigma = torch.exp(log_sigma)

        # Angular entropy is BOUNDED by log(2*pi) (uniform on the circle).
        # The previous formula used the ambient 2D-Gaussian entropy (+log_sigma),
        # unbounded in sigma -> the selector inflated sigma to the clamp, giving a
        # uniform distribution AND a vanishing mu-gradient (variance collapse).
        # Fix: use the von Mises entropy of the induced angular distribution with
        # concentration kappa = ||mu|| / sigma. Bounded above by log(2*pi); both
        # mu and sigma receive sensible gradients only through the ratio kappa.
        # May-19 (8e3d31f) clamped form: penalizes collapse but RELEASES
        # (gradient 0) above the log(2*pi) cap, so the flow loss can freely
        # collapse the SO2 selector. The von Mises form regularized toward
        # uniform continuously -> SO2 selector stuck uniform -> Can regressed.
        concentration_sq = (mu / sigma).pow(2).sum(dim=-1)  # (B,)
        H_approx = math.log(2 * math.pi) + log_sigma.mean(dim=-1) - 0.5 * concentration_sq
        H = H_approx.clamp(max=math.log(2 * math.pi))

        return H

    def log_prob(self, params, angles):
        """
        Approximate log probability (for monitoring, not used in training).
        params: (B, 4)
        angles: (B, N)
        Returns: (B, N)
        """
        mu = params[:, :2]  # (B, 2)
        log_sigma = params[:, 2:].clamp(-4, 4)  # (B, 2)
        sigma = torch.exp(log_sigma)

        # Convert angles to unit vectors
        cos_a = torch.cos(angles)  # (B, N)
        sin_a = torch.sin(angles)  # (B, N)
        unit_vec = torch.stack([cos_a, sin_a], dim=-1)  # (B, N, 2)

        # Log prob of projected normal (approximate via von Mises-like)
        mu_norm = mu / (mu.norm(dim=-1, keepdim=True) + 1e-8)
        kappa = mu.norm(dim=-1) / sigma.mean(dim=-1)  # (B,)

        # cos(angle - mu_angle)
        cos_diff = (unit_vec * mu_norm.unsqueeze(1)).sum(dim=-1)  # (B, N)
        log_p = kappa.unsqueeze(1) * cos_diff - math.log(2 * math.pi)

        return log_p


class GumbelSoftmaxCategorical(nn.Module):
    """
    Categorical distribution with Gumbel-Softmax for differentiable sampling.
    Used for discrete groups like C4.

    Uses straight-through estimator: hard one-hot in forward,
    soft gradients in backward.
    """

    def __init__(self, num_classes, tau=1.0, tau_min=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.param_dim = num_classes
        self.tau = tau
        self.tau_min = tau_min

    def sample(self, logits, num_samples):
        """
        logits: (B, num_classes)
        num_samples: int N
        Returns: (B, N) integer indices

        Uses Gumbel-Softmax with straight-through for gradient flow.
        During training, returns differentiable soft indices via
        weighted sum of class indices.
        """
        B = logits.shape[0]
        device = logits.device

        # Expand logits for N samples
        logits_expanded = logits.unsqueeze(1).expand(B, num_samples, self.num_classes)

        # Gumbel-Softmax sampling
        if self.training:
            # Soft samples with straight-through
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(logits_expanded) + 1e-20) + 1e-20
            )
            y_soft = F.softmax((logits_expanded + gumbel_noise) / self.tau, dim=-1)

            # Straight-through: argmax in forward, soft gradients in backward
            idx = y_soft.argmax(dim=-1)  # (B, N) hard indices
            y_hard = F.one_hot(idx, self.num_classes).float()
            y = y_hard - y_soft.detach() + y_soft  # straight-through

            # Return weighted index for differentiability
            class_indices = torch.arange(
                self.num_classes, dtype=torch.float, device=device
            )
            samples = (y * class_indices).sum(dim=-1)  # (B, N)
        else:
            # During eval, use standard categorical sampling
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(
                probs, num_samples, replacement=True
            )  # (B, N)
            samples = idx.float()

        return samples

    def entropy(self, logits):
        """
        Shannon entropy of the categorical distribution.
        logits: (B, num_classes)
        Returns: (B,)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)

    def anneal_tau(self, rate=0.0003):
        """Anneal temperature toward tau_min."""
        self.tau = max(self.tau_min, self.tau * (1 - rate))


def get_distribution(symmetry_cfg):
    """Factory function for distribution instances."""
    dist_type = symmetry_cfg.distribution_type
    if dist_type == "projected_normal":
        return ProjectedNormalSO2()
    elif dist_type == "gumbel_categorical":
        num_classes = 4  # C4
        if hasattr(symmetry_cfg, "num_classes"):
            num_classes = symmetry_cfg.num_classes
        tau = symmetry_cfg.get("gumbel_tau", 1.0)
        tau_min = symmetry_cfg.get("gumbel_tau_min", 0.1)
        return GumbelSoftmaxCategorical(
            num_classes=num_classes, tau=tau, tau_min=tau_min
        )
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
